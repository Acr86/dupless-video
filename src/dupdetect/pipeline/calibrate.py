"""Calibration (step 10): evaluates thresholds against a hand-labeled set.

Key split: `compute_signals` (expensive: analyzes+aligns each pair ONCE) ->
`confusion_by_tier` and `suggest_thresholds` operate on those cached signals, so
the threshold sweep is cheap. Goal: ZERO false positives in T1/T2.
"""
from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dupdetect.config import Thresholds, load_thresholds
from dupdetect.match.tree import decide_tree
from dupdetect.models import AlignResult, Probe, Quality, Record, Verdict

# Labels meaning "same movie" (must fall into a duplicate tier).
SAME_LABELS = {"same", "dup", "duplicate", "dub", "doblaje", "cam", "upgrade", "si", "yes"}
# Strong tiers where we require ZERO false positives.
STRONG_VERDICTS = (Verdict.CERTAIN, Verdict.VERY_HIGH)
DUP_VERDICTS = (Verdict.CERTAIN, Verdict.VERY_HIGH, Verdict.HIGH)


@dataclass
class LabeledSignal:
    label: str
    audio: AlignResult
    video: AlignResult
    scenes: AlignResult
    a_path: str = ""
    b_path: str = ""
    genre: str = ""          # DEV measurement label ONLY — NEVER enters the verdict (§0)

    @property
    def is_same(self) -> bool:
        return self.label.strip().lower() in SAME_LABELS


def load_pairs(path: str | Path) -> list[tuple[str, str, str]]:
    """Loads (path_a, path_b, label) from a JSON [{path_a,path_b,label}] or headered CSV."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        return [(d["path_a"], d["path_b"], d["label"]) for d in json.loads(p.read_text("utf-8"))]
    with open(p, newline="", encoding="utf-8") as f:
        return [(r["path_a"], r["path_b"], r["label"]) for r in csv.DictReader(f)]


def _align_pair(ra: Record, rb: Record, cache, th):
    """Runs the 3 aligners on TWO already-loaded records (no decode). Single point where
    alignment parameters are fixed -> scan and recompute use EXACTLY the same `th` (if
    they diverged, calibration would be invalid)."""
    from dupdetect.align.audio import align_audio
    from dupdetect.align.scenes import align_scenes
    from dupdetect.align.video import align_video

    a = align_audio(ra.audio_fp, rb.audio_fp, min_overlap_s=th.raw["audio"]["min_overlap_s"])
    v = align_video(cache.get(ra.path), cache.get(rb.path), fps=th.fps_sample,
                    band_radius=th.band_radius_frames,
                    superset_extra_ratio=th.superset_min_extra_ratio,
                    min_ad_run_s=th.min_ad_run_s)
    s = align_scenes(ra.scene_cuts, rb.scene_cuts, theta=th.theta_s)
    return a, v, s


def compute_signals(pairs, store, embedder, th, cache=None) -> list[LabeledSignal]:
    """The EXPENSIVE part: analyzes both files of each pair and runs the 3 aligners."""
    from dupdetect.match.cache import EmbeddingCache
    from dupdetect.pipeline.analyze import analyze_file

    cache = cache or EmbeddingCache(store)
    out = []
    for pa, pb, label in pairs:
        ra = analyze_file(pa, store, embedder, th)
        rb = analyze_file(pb, store, embedder, th)
        a, v, s = _align_pair(ra, rb, cache, th)
        out.append(LabeledSignal(label, a, v, s, pa, pb))
    return out


def signal_for_pair(a: str, b: str, label: str, store, cache, th,
                    genre: str = "") -> "LabeledSignal | None":
    """Recompute for an ORPHAN pair (has feedback but NO row in `matches`): aligns from the
    ALREADY-saved fingerprints (audio_fp/scene_cuts from the record + embeddings from .npy via cache) — WITHOUT
    re-decoding from disk and WITHOUT the embedder. Returns None (skip-and-report) if a record or its .npy
    is missing, to avoid crashing recalibration (at scale, the rare case is certain). `genre` is a DEV
    measurement label carried through for per-genre reporting — it NEVER enters decide_tree (§0)."""
    ra = store.load(a, with_embeddings=True)             # loads embeddings (numpy) to validate
    rb = store.load(b, with_embeddings=True)
    if ra is None or rb is None:                         # one endpoint is no longer indexed
        return None
    # LITE / 'exact-only' records have NO fingerprints (embeddings/audio_fp/scene_cuts are empty):
    # aligning them would yield scores of 0 -> a 'same' signal with score 0 POISONS calibration (pushes
    # thresholds down and risks false positives). Don't fabricate: unrecoverable -> re-scan (full).
    if ra.embeddings.shape[0] == 0 or rb.embeddings.shape[0] == 0:
        return None
    # Self-heal audio so the harness measures the REAL pipeline: the app fingerprints on-demand in
    # Pass-2 (matcher._ensure_audio_fp), so a record whose audio_fp is empty (never matched, or cleared
    # by an fp-policy migration) would otherwise score audio=0 here and understate recall. Recompute it
    # the SAME way (duration-gated whole-file), persisting it. No-op when already present.
    from dupdetect.match.matcher import _ensure_audio_fp
    _ensure_audio_fp(ra, store, th)
    _ensure_audio_fp(rb, store, th)
    try:
        au, vi, sc = _align_pair(ra, rb, cache, th)      # cache.get loads the .npy (KeyError if missing)
    except (KeyError, OSError):                           # OSError covers FileNotFoundError (.npy moved)
        return None
    return LabeledSignal(label, au, vi, sc, a, b, genre=genre)


def _parse_align(j: str) -> AlignResult:
    """Reconstructs an AlignResult from the JSON saved in `matches` (tolerant of extra/missing
    fields across versions)."""
    d = json.loads(j) if j else {}
    fields = AlignResult.__dataclass_fields__
    return AlignResult(**{k: v for k, v in d.items() if k in fields}) if d else AlignResult(0.0)


def labeled_signals_from_feedback(store, th=None, cache=None) -> list[LabeledSignal]:
    """Labeled set from user FEEDBACK. Driven from `feedback` (not from `matches`): for each
    labeled pair, reuses scores ALREADY saved in its `matches` row if it exists; if the pair is
    ORPHAN (no row — e.g. `clusters` and `matches` are from different scans), RECOMPUTES from
    the saved fingerprints (without re-decoding). This keeps feedback valid even when the two
    tables drift out of sync. Both tables canonicalize to a<=b."""
    from dupdetect.config import load_thresholds
    from dupdetect.match.cache import EmbeddingCache

    th = th or load_thresholds()
    cache = cache or EmbeddingCache(store)
    rows = {(r["a_path"], r["b_path"]): r for r in store.conn.execute(
        "SELECT a_path, b_path, audio_json, video_json, scenes_json FROM matches")}
    out: list[LabeledSignal] = []
    for a, b, label in store.iter_feedback():            # already canonical (a<=b)
        r = rows.get((a, b))
        if r is not None:                                # fast path: scores already in matches
            out.append(LabeledSignal(
                label, _parse_align(r["audio_json"]), _parse_align(r["video_json"]),
                _parse_align(r["scenes_json"]), a, b))
            continue
        sig = signal_for_pair(a, b, label, store, cache, th)   # orphan -> recompute (or None)
        if sig is not None:
            out.append(sig)
    return out


def _mk(path: str, h: str) -> Record:
    """Minimal Record for decide_tree (only uses path/content_hash/size; distinct hashes
    => never triggers T0, which doesn't depend on thresholds)."""
    return Record(
        path=path, mtime=0.0, size=hash(h) & 0xFFFF, probe=Probe(0, 0, 0, "", None),
        content_hash=h, global_vec=np.zeros(1, np.float32), window_vecs=np.zeros((0, 1), np.float32),
        embeddings=np.zeros((0, 1), np.float16), audio_fp=np.zeros(0, np.uint32),
        scene_cuts=np.zeros(0, np.float32), quality=Quality(),
    )


def verdict_of(sig: LabeledSignal, th: Thresholds) -> Verdict:
    return decide_tree(_mk("a", "ha"), _mk("b", "hb"), sig.audio, sig.video, sig.scenes, th).verdict


def confusion_by_tier(signals: list[LabeledSignal], th: Thresholds) -> dict:
    """Table verdict -> {same, diff}: how many were same vs different movie in each tier."""
    table: dict[str, dict[str, int]] = {}
    for sig in signals:
        cell = table.setdefault(verdict_of(sig, th).value, {"same": 0, "diff": 0})
        cell["same" if sig.is_same else "diff"] += 1
    return table


def confusion_by_genre(signals: list[LabeledSignal], th: Thresholds) -> dict:
    """Per-genre precision / recall / strict-FP, evaluating the REAL `decide_tree` (the ONE global
    logic). Genre is a DEV measurement label only — it never enters the verdict (§0); this just shows
    WHERE the single logic leaks per content type, so each new genre (docs, cinema, AI-shorts) is
    decided with numbers instead of by loosening thresholds. `precision`/`recall` are None when the
    genre has no dup-tier / no 'same' pairs (avoids a misleading 0%)."""
    dup_vals = {v.value for v in DUP_VERDICTS}
    by_g: dict[str, list[LabeledSignal]] = {}
    for s in signals:
        by_g.setdefault(s.genre.strip() or "(unlabeled)", []).append(s)
    out: dict[str, dict] = {}
    for g, sigs in sorted(by_g.items()):
        dup_same = dup_diff = same_total = caught = fp_strict = 0
        for s in sigs:
            v = verdict_of(s, th)
            in_dup = v.value in dup_vals
            if s.is_same:
                same_total += 1
                caught += in_dup
            if in_dup:
                dup_same += s.is_same
                dup_diff += not s.is_same
            fp_strict += (not s.is_same) and (v in STRONG_VERDICTS)
        out[g] = {
            "n": len(sigs),
            "precision": (dup_same / (dup_same + dup_diff)) if (dup_same + dup_diff) else None,
            "recall": (caught / same_total) if same_total else None,
            "fp_strict": int(fp_strict),
            "dup_same": dup_same, "dup_diff": dup_diff, "same_total": same_total,
        }
    return out


def _th_override(base: Thresholds, theta_v: float, theta_a: float) -> Thresholds:
    raw = copy.deepcopy(base.raw)
    raw["video"]["theta_v"] = theta_v
    raw["audio"]["theta_a"] = theta_a
    return Thresholds(raw=raw)


def _false_positives(conf: dict) -> int:
    """'diff' pairs classified in strong tiers (T1/T2)."""
    return sum(conf.get(v.value, {}).get("diff", 0) for v in STRONG_VERDICTS)


def _recall(signals, th) -> float:
    same = [s for s in signals if s.is_same]
    if not same:
        return 0.0
    caught = sum(1 for s in same if verdict_of(s, th) in DUP_VERDICTS)
    return caught / len(same)


def suggest_thresholds(signals: list[LabeledSignal], base: Thresholds | None = None,
                       grid: list[float] | None = None) -> dict:
    """Sweeps (theta_v, theta_a) and picks the combination giving ZERO FP in T1/T2, then MAX recall.
    Returns the suggestion + the resulting confusion matrix."""
    base = base or load_thresholds()
    grid = grid or [round(0.50 + 0.05 * i, 2) for i in range(10)]   # 0.50..0.95
    best = None
    for tv in grid:
        for ta in grid:
            th = _th_override(base, tv, ta)
            fp = _false_positives(confusion_by_tier(signals, th))
            rec = _recall(signals, th)
            cand = (fp, -rec, tv, ta)          # 1st fewest FP, 2nd most recall, 3rd lowest threshold
            if best is None or cand < best:
                best = cand
    fp, neg_rec, tv, ta = best
    chosen = _th_override(base, tv, ta)
    return {
        "theta_v": tv, "theta_a": ta,
        "false_positives_T1T2": fp, "recall_dup": round(-neg_rec, 3),
        "confusion": confusion_by_tier(signals, chosen),
        "n_pairs": len(signals),
    }
