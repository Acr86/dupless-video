"""Orchestration of match(): coarse -> fine -> tree.

C2: canonicalizes pairs (evaluated once per unordered pair).
A1: uses resident EmbeddingCache, no per-candidate disk reload.
A2: beyond top-k coarse, a duration safety net to avoid losing
    recall (a degraded cam rip may not enter the global_vec top-k).
C3: the align offset (pre-roll ads) is propagated here, it is a pair-level property.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from dupdetect.align.audio import align_audio
from dupdetect.features.audio_fp import audio_fingerprint
from dupdetect.align.scenes import align_scenes
from dupdetect.align.video import align_video, resample_to_grid
from dupdetect.config import Thresholds
from dupdetect.match.cache import EmbeddingCache
from dupdetect.match.retrieval import CoarseIndex
from dupdetect.match.tree import decide_tree
from dupdetect.models import AlignResult, Record, Result, Verdict
from dupdetect.store import FingerprintStore
from dupdetect.store.store import canonical_pair


def candidate_paths(rec: Record, store: FingerprintStore, index: CoarseIndex,
                    th: Thresholds) -> set[str]:
    """A2: union of three candidate sources (uncapped recall):
      1. top-k global mean-pool (fast first pass)
      2. multi-vector by temporal window (rescues cam rips / globals with ads)
      3. duration blocking ±tol (always compares similar runtimes)
    """
    cands = {p for p, _ in index.query_global(rec.global_vec, k=th.faiss_k)}
    if th.n_window_vecs > 0 and rec.window_vecs.size:
        cands |= index.query_windows(rec.window_vecs, k=th.raw["retrieval"]["window_faiss_k"])
    # Duration safety net, GATED by global cosine: on a dense library ±tol returns thousands of
    # same-length-but-different videos (O(N^2) Pass-2). Real dups are ~identical-duration AND
    # globally near (>=0.962 measured), so the gate prunes the dragnet without losing them. Only
    # this net is gated; top-k/window retrieval above are untouched.
    dur = duration_blocking(rec, store, th)
    if rec.global_vec is not None and getattr(rec.global_vec, "size", 0):
        dur = index.gate_by_global(rec.global_vec, dur, th.duration_block_cos_gate)
    cands |= dur
    cands.discard(rec.path)
    return cands


def duration_blocking(rec: Record, store: FingerprintStore, th: Thresholds) -> set[str]:
    """A2: safety net. Paths with duration within ±duration_tolerance.
    Cheap thanks to the idx_files_duration index."""
    return set(store.find_by_duration(rec.probe.duration_s, th.duration_tolerance))


def _emb_is_empty(e) -> bool:
    """True when an embedding sequence has no frames. Type-agnostic ON PURPOSE: the
    EmbeddingCache hands back torch tensors, whose `.size` is a METHOD (so the numpy-style
    `e.size == 0` compares a bound method to 0 -> ALWAYS False, letting empty CUDA tensors slip
    through and crash align_video's matmul). torch exposes `.numel()`; numpy exposes `.size` int."""
    if e is None:
        return True
    n = e.numel() if hasattr(e, "numel") else getattr(e, "size", 0)
    return int(n) == 0


def _align_video_pair(rec: Record, other: Record, cache: EmbeddingCache, th: Thresholds):
    """Aligns per-frame embeddings of `rec` and `other`. If BOTH have frame_times,
    resamples both sequences to a uniform temporal grid (step grid_step_s) BEFORE
    align_video -> aligns by TIME, robust to mixed demux/seek sampling. Each sequence uses
    its OWN temporal range (no common end forced) to preserve superset/edition detection
    (a director's cut is still longer). Without frame_times (legacy records),
    falls back to the historical index-of-frame path."""
    try:
        ea, eb = cache.get(rec.path), cache.get(other.path)
    except KeyError:                                    # embeddings ausentes (.npy faltante)
        return AlignResult(score=0.0)                  # no video signal -> audio/scenes decide
    # A missing/orphaned .npy loads as EMPTY embeddings while frame_times still come from the DB
    # row (non-empty) -> guard, or resample_to_grid would index an empty array (crash). §2: at
    # scale the rare is certain; skip-and-report (no video signal), never crash the batch.
    if _emb_is_empty(ea) or _emb_is_empty(eb):     # torch `.size` is a method -> count-agnostic
        return AlignResult(score=0.0)
    ta, tb = rec.frame_times, other.frame_times
    if ta is not None and tb is not None and ta.size and tb.size:
        step = th.grid_step_s
        ra = resample_to_grid(ea, ta, step)
        rb = resample_to_grid(eb, tb, step)
        band = max(1, int(th.max_offset_s / step))
        return align_video(ra, rb, fps=1.0 / step, band_radius=band,
                           superset_extra_ratio=th.superset_min_extra_ratio,
                           min_ad_run_s=th.min_ad_run_s)
    return align_video(ea, eb, fps=th.fps_sample, band_radius=th.band_radius_frames,
                       superset_extra_ratio=th.superset_min_extra_ratio, min_ad_run_s=th.min_ad_run_s)


def _ensure_audio_fp(rec: Record, store: FingerprintStore, th: Thresholds) -> np.ndarray:
    """ON-DEMAND audio fingerprint: computed here (Pass-2) only for files that actually reach a
    candidate pair — most unique movies never pay the full-file read. Cached on the record and
    persisted so re-runs / cluster ranking reuse it. A failure (broken audio) -> empty fp (audio
    just doesn't contribute; video/scenes still decide). Coverage was already measured cheaply
    in Pass-1, so the muted-copy warning does NOT depend on this."""
    if rec.audio_fp is not None and rec.audio_fp.size:
        return rec.audio_fp
    try:
        fp = audio_fingerprint(rec.path, max_length_s=th.audio_fp_max_for(rec.probe.duration_s),
                               timeout=th.audio_fp_timeout_s)
    except Exception:                                   # noqa: BLE001
        fp = np.empty(0, dtype=np.uint32)
    rec.audio_fp = fp
    if fp.size:
        store.set_audio_fp(rec.path, fp)
    return fp


def match(rec: Record, store: FingerprintStore, index: CoarseIndex,
          th: Thresholds, cache: EmbeddingCache | None = None,
          seen: set[tuple[str, str]] | None = None) -> list[Result]:
    """Finds duplicates/upgrades of `rec`. Same engine for full-scan and watcher.

    C2: `seen` can be shared across calls (full-scan) to avoid re-aligning the
    same unordered pair twice. None => local set (watcher: single query).
    """
    cache = cache or EmbeddingCache(store)
    results: list[Result] = []
    seen = seen if seen is not None else set()

    for cand_path in candidate_paths(rec, store, index, th):
        pair = canonical_pair(rec.path, cand_path)    # C2: once per pair
        if pair in seen:
            continue
        seen.add(pair)

        other = store.load(cand_path, with_embeddings=False)
        if other is None:
            continue

        # three independent signals. A1: embeddings from the resident cache.
        # Audio fingerprint computed ON-DEMAND here (only for candidate pairs).
        fa = _ensure_audio_fp(rec, store, th)
        fb = _ensure_audio_fp(other, store, th)
        a = align_audio(fa, fb, min_overlap_s=th.raw["audio"]["min_overlap_s"])
        v = _align_video_pair(rec, other, cache, th)
        s = align_scenes(rec.scene_cuts, other.scene_cuts, theta=th.theta_s)

        res = decide_tree(rec, other, a, v, s, th)
        results.append(res)

    results.sort(key=lambda r: r.confidence, reverse=True)
    return [r for r in results if r.verdict != Verdict.DIFFERENT]


# --------------------------------------------------------------------------- Pass-2 parallel
# Pass-2 is COMPUTE-bound (banded video DP ~89%), not I/O-bound, so it scales across cores
# without disk thrashing (unlike Pass-1). Workers are READ-ONLY store handles (init_schema=False),
# so concurrent openers don't contend. Results are deterministic per pair -> verdict invariant (§0).

_PW: dict = {}                                          # per-worker state (spawn-safe)


def _pass2_init(db_path: str, th: Thresholds) -> None:
    from dupdetect.store import FingerprintStore
    _PW["store"] = FingerprintStore(db_path, init_schema=False)   # read-only worker handle
    _PW["th"] = th


class _DictCache:
    """Minimal EmbeddingCache-compatible shim (.get) backed by an in-worker dict of CPU tensors."""
    def __init__(self, d: dict):
        self._d = d

    def get(self, path: str):
        e = self._d.get(path)
        if e is None:
            raise KeyError(path)
        return e


def _load_safe(store: FingerprintStore, path: str):
    """Record WITH embeddings if the .npy is present, else WITHOUT (empty embeddings). Never
    raises on a missing/moved .npy (mirrors EmbeddingCache resilience): video just doesn't
    contribute and audio/scenes still decide. Real DBs accumulate orphaned .npy refs."""
    try:
        return store.load(path, with_embeddings=True)
    except OSError:                                    # FileNotFoundError is an OSError subclass
        return store.load(path, with_embeddings=False)


def name_pair_content_differs(a_path: str, b_path: str, store: FingerprintStore,
                              th: Thresholds) -> bool:
    """Veto for NAME_COPY grouping: True if a name-sibling pair must NOT be grouped. Returns True
    when the pair CAN'T be verified by content (a copy is LITE/exact-only -> no embeddings) or when
    content says they're DIFFERENT (different videos that happen to share a '(N)' name). NAME_COPY
    fires only when content does NOT contradict -> avoids false positives on reused names (§0).
    DIFFERENT verdicts aren't persisted, so we re-verify the pair here instead of trusting the DB."""
    ra = _load_safe(store, a_path)
    rb = _load_safe(store, b_path)
    if ra is None or rb is None:
        return True
    if (ra.embeddings is None or not ra.embeddings.size
            or rb.embeddings is None or not rb.embeddings.size):
        return True                                # LITE: can't verify content -> don't group
    cdict = {ra.path: np.ascontiguousarray(ra.embeddings, dtype=np.float32),
             rb.path: np.ascontiguousarray(rb.embeddings, dtype=np.float32)}
    a = align_audio(_ensure_audio_fp(ra, store, th), _ensure_audio_fp(rb, store, th),
                    min_overlap_s=th.raw["audio"]["min_overlap_s"])
    v = _align_video_pair(ra, rb, _DictCache(cdict), th)
    s = align_scenes(ra.scene_cuts, rb.scene_cuts, theta=th.theta_s)
    return decide_tree(ra, rb, a, v, s, th).verdict == Verdict.DIFFERENT


def _pass2_pair(pair: tuple[str, str]):
    """Worker: aligns ONE candidate pair (pure CPU compute) and decides. Returns the row to
    persist, or None for DIFFERENT (mirrors match()'s filter). Audio fps were ensured+persisted
    in the main process, so here they load from the row (no fpcalc in workers)."""
    a_path, b_path = pair
    store, th = _PW["store"], _PW["th"]
    ra = _load_safe(store, a_path)
    rb = _load_safe(store, b_path)
    if ra is None or rb is None:
        return None
    # Pure numpy: align_video/resample run on numpy embeddings -> workers never import torch
    # (faster process spawn). The banded DP is numpy/CPU anyway.
    cdict = {}
    for r in (ra, rb):
        e = r.embeddings
        cdict[r.path] = (np.ascontiguousarray(e, dtype=np.float32)
                         if e is not None and e.size else np.empty((0, 0), dtype=np.float32))
    a = align_audio(ra.audio_fp, rb.audio_fp, min_overlap_s=th.raw["audio"]["min_overlap_s"])
    v = _align_video_pair(ra, rb, _DictCache(cdict), th)
    s = align_scenes(ra.scene_cuts, rb.scene_cuts, theta=th.theta_s)
    res = decide_tree(ra, rb, a, v, s, th)
    if res.verdict == Verdict.DIFFERENT:
        return None
    return (a_path, b_path, res.verdict.value, res.confidence, res.reason,
            (v.offset if v else None), asdict(a), asdict(v), asdict(s))


def match_pairs_parallel(paths, store: FingerprintStore, index: CoarseIndex,
                         th: Thresholds, workers: int, progress: bool = False) -> list:
    """Parallel Pass-2: enumerate unique candidate pairs, ensure audio fps once (main, cheap),
    then align+decide each pair across a process pool. Returns rows = the tuple from `_pass2_pair`
    for non-DIFFERENT pairs. Same results as the sequential match() loop, just parallel."""
    from tqdm import tqdm
    pairs: set[tuple[str, str]] = set()
    for p in paths:
        rec = store.load(p, with_embeddings=False)
        if rec is None:
            continue
        for cand in candidate_paths(rec, store, index, th):
            pairs.add(canonical_pair(p, cand))
    pair_list = list(pairs)
    if not pair_list:
        return []
    # Ensure (compute+persist) the on-demand fingerprint once per involved file, in the MAIN
    # process -> workers stay read-only and never run fpcalc concurrently.
    involved = {x for pr in pair_list for x in pr}
    for fp_path in tqdm(involved, desc="Pass 2 (audio fp)", unit="file",
                        disable=not progress, dynamic_ncols=True):
        rec = store.load(fp_path, with_embeddings=False)
        if rec is not None:
            _ensure_audio_fp(rec, store, th)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_pass2_init,
                             initargs=(str(store.db_path), th)) as pool:
        for row in tqdm(pool.map(_pass2_pair, pair_list, chunksize=4), total=len(pair_list),
                        desc="Pass 2 (align pairs)", unit="pair",
                        disable=not progress, dynamic_ncols=True):
            if row is not None:
                rows.append(row)
    return rows
