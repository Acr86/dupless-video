"""Full-scan: indexes the entire library and groups duplicates into clusters.

Two passes:
  Pass 1: analyze_file() over everything -> populates the store (features + emb).
  Pass 2: match() coarse->fine over each file; groups with union-find.

C2: canonicalization in match() avoids double evaluation.
A1: Resident EmbeddingCache for re-ranking.
A5: persists clusters/keep in the DB (source of truth for delete/move).
C3: propagates ad_offset (from align) to the matches table.
Verde: surfaces DIFFERENT_EDITION as "related, not duplicates".
"""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import (
    FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait,
)
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from dupdetect.config import Thresholds
from dupdetect.features.audio_fp import AUDIO_COV_TOL, AUDIO_OK_COVERAGE
from dupdetect.features.embeddings import Embedder
from dupdetect.match.cache import EmbeddingCache
from dupdetect.match.matcher import match, match_pairs_parallel, name_pair_content_differs
from dupdetect.match.retrieval import CoarseIndex
from dupdetect.match.tree import DUPLICATE_VERDICTS, REVIEW_VERDICTS
from dupdetect.models import Verdict
from tqdm import tqdm

from dupdetect.features.frames import decode_frames
from dupdetect.features.hashing import content_hash
from dupdetect.features.probe import ffprobe
from dupdetect.quality.color import CLIP_DOWNGRADE_MARGIN, GRADE_DIVERGENCE
from dupdetect.quality.language import detect_language
from dupdetect.pipeline.analyze import (
    analyze_file, build_record, ensure_audio_coverage, extract_cpu_features, extract_gpu_features,
    feature_version, maybe_emit_viz,
)
from dupdetect.store import FingerprintStore


def _short(path: str, n: int = 34) -> str:
    name = os.path.basename(path)
    return name if len(name) <= n else name[: n - 1] + "…"

VIDEO_EXTS = {
    ".mkv", ".mp4", ".mp4v", ".avi", ".m4v", ".mov", ".wmv", ".ts", ".m2ts", ".mts",
    ".webm", ".mpg", ".mpeg", ".flv", ".ogv", ".vob", ".3gp", ".divx", ".xvid",
}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for node in self.parent:
            out.setdefault(self.find(node), []).append(node)
        return out


def iter_videos(root: str | Path, recursive: bool = True) -> Iterable[str]:
    """Iterates videos under `root`. recursive=False => root level only (no subdirectories).
    Always reads in place: does not copy or move files."""
    paths = Path(root).rglob("*") if recursive else Path(root).glob("*")
    for p in paths:
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield str(p)


def collect_videos(targets: str | Path | list, recursive: bool = True) -> list[str]:
    """Expands one or more paths (files and/or directories) into the video list, IN PLACE
    (no copying). A video file is included as-is; a directory is iterated."""
    if isinstance(targets, (str, Path)):
        targets = [targets]
    out: list[str] = []
    for t in targets:
        p = Path(t)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(str(p))
        elif p.is_dir():
            out.extend(iter_videos(p, recursive=recursive))
    return out


def _media_height(path: str) -> int | None:
    """Height in pixels of the first video stream (fast ffprobe, no decoding)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=height", "-of", "csv=p=0", path]
    try:
        from dupdetect.util import CREATE_NO_WINDOW
        out = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                             timeout=60, creationflags=CREATE_NO_WINDOW).stdout.strip()
        return int(out.splitlines()[0]) if out else None
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def filter_by_height(paths: list[str], max_height: int, workers: int = 8) -> tuple[list, list]:
    """Splits (kept, excluded) by video height. Files that cannot be measured (corrupt)
    are KEPT -> handled by pass 1 (problems table). Parallel probe (fast)."""
    with ThreadPoolExecutor(max_workers=max(1, min(16, workers * 2))) as ex:
        # tqdm: probing resolution for N files can be slow on large libraries -> visible progress.
        heights = list(tqdm(ex.map(_media_height, paths), total=len(paths),
                            desc="Measuring resolution", unit="file", dynamic_ncols=True))
    kept, excluded = [], []
    for p, h in zip(paths, heights):
        (excluded if (h is not None and h > max_height) else kept).append(p)
    return kept, excluded


def _cpu_worker(args):
    """Picklable worker for the ProcessPool: CPU features only (no CUDA)."""
    path, independent_scenes = args
    return extract_cpu_features(path, independent_scenes=independent_scenes)


def _pass1(paths: list[str], store: FingerprintStore, embedder: Embedder,
           th: Thresholds, fv: str, force: bool, workers: int,
           independent_scenes: bool, progress: bool = False,
           decode_workers: int = 1) -> list[tuple[str, str]]:
    """Pass 1 (M3 — split by resource):
      (a) CPU features (probe/hash/fpcalc/[scenes]/language) in ProcessPool — no CUDA.
      (b) NVDEC decode + embed on the main thread. `decode_workers>1` moves DECODE to a
          thread pool (prefetch), leaving only GPU embed on main -> overlaps I/O with compute.
          Only beneficial on SSD/NVMe; on HDD disk concurrency causes thrashing (use 1).
    Incremental: skips already-fresh files. RESILIENT: an unreadable file (corrupt,
    missing moov atom, etc.) is SKIPPED and reported; does not abort the scan. Returns [(path, error)]."""
    todo = [p for p in paths if force or not store.has_fresh(p, os.stat(p), fv)]
    skipped: list[tuple[str, str]] = []
    fresh = len(paths) - len(todo)
    if not todo:
        if progress:
            tqdm.write(f"Pass 1: nothing to analyze ({fresh} already fresh).")
        return skipped

    bar = tqdm(total=len(todo), desc="Pass 1 (analysis)", unit="film",
               disable=not progress, dynamic_ncols=True)
    bar.set_postfix_str(f"fresh={fresh}")

    def _mark(p: str):
        bar.update(1)
        bar.set_postfix_str(f"fresh={fresh} skipped={len(skipped)} | {_short(p)}")

    if workers and workers > 1:
        _pass1_parallel(todo, store, embedder, th, fv, independent_scenes,
                        workers, decode_workers, skipped, _mark)
    else:
        for p in todo:                             # sequential (default, testable)
            try:
                analyze_file(p, store, embedder, th, force=force,
                             independent_scenes=independent_scenes)
            except Exception as e:                                          # noqa: BLE001
                skipped.append((p, str(e)))
                store.save_problem(p, str(e))          # persists the problem in the DB
            _mark(p)
    bar.close()
    return skipped


def _gpu_finish(cpu, frames_times, store, embedder, th, fv, independent_scenes) -> None:
    """Embed (GPU, main) + build_record + save. `frames_times`=(frames,ts) already decoded
    (pipelined path) or None to decode here (serial path)."""
    if frames_times is None:
        emb, times, color = extract_gpu_features(cpu.path, cpu.probe, embedder, th)
    else:
        frames, times, color = frames_times
        maybe_emit_viz(cpu.path, frames)               # live-view (pipelined path; serial path emits inside extract_gpu_features)
        emb = embedder.encode(frames)
    rec = build_record(cpu, emb, times, color, embedder, th, independent_scenes)
    store.save(rec, feature_version=fv)


def _pass1_parallel(todo, store, embedder, th, fv, independent_scenes,
                    workers, decode_workers, skipped, mark) -> None:
    """ProcessPool for CPU features (no CUDA) + GPU on main. If decode_workers>1,
    decode runs in a thread pool with bounded prefetch (SSD)."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cpu_futs = {pool.submit(_cpu_worker, (p, independent_scenes)): p
                    for p in todo}
        if decode_workers and decode_workers > 1:
            _drain_pipelined(cpu_futs, store, embedder, th, fv, independent_scenes,
                             decode_workers, skipped, mark)
            return
        for fut in as_completed(cpu_futs):         # decode+embed serial on main (HDD)
            p = cpu_futs[fut]
            try:
                cpu = fut.result()
                _gpu_finish(cpu, None, store, embedder, th, fv, independent_scenes)
            except Exception as e:                                          # noqa: BLE001
                skipped.append((p, str(e)))
                store.save_problem(p, str(e))
            mark(p)


def _drain_pipelined(cpu_futs, store, embedder, th, fv, independent_scenes,
                     decode_workers, skipped, mark) -> None:
    """SSD: decode (I/O + NVDEC) in a thread pool with BOUNDED prefetch while main
    embeds on GPU -> overlaps I/O with compute (2.19x measured on NVMe). Embed is ALWAYS
    on main (single CUDA context); threads only decode (ffmpeg subprocess + H2D,
    which release the GIL). On HDD this would cause thrashing: hence decode_workers=1 by default."""
    cpu_done = as_completed(cpu_futs)
    with ThreadPoolExecutor(max_workers=decode_workers) as dpool:
        inflight: dict = {}                        # future(decode) -> cpu
        max_inflight = decode_workers + 1          # bounds VRAM (frames reside until embed)

        def submit_next() -> bool:
            for fut in cpu_done:                   # advances the iterator until a decode is queued
                p = cpu_futs[fut]
                try:
                    cpu = fut.result()
                except Exception as e:                                      # noqa: BLE001
                    skipped.append((p, str(e))); store.save_problem(p, str(e)); mark(p)
                    continue
                inflight[dpool.submit(decode_frames, cpu.path)] = cpu
                return True
            return False

        for _ in range(max_inflight):
            if not submit_next():
                break
        while inflight:
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for dfut in done:
                cpu = inflight.pop(dfut)
                try:
                    _gpu_finish(cpu, dfut.result(), store, embedder, th, fv, independent_scenes)
                except Exception as e:                                      # noqa: BLE001
                    skipped.append((cpu.path, str(e))); store.save_problem(cpu.path, str(e))
                mark(cpu.path)
                submit_next()                      # keeps the pipeline full


def full_scan(targets, store: FingerprintStore, embedder: Embedder,
              th: Thresholds, force: bool = False, workers: int = 1,
              recursive: bool = True, independent_scenes: bool = False,
              max_height: int | None = None, progress: bool = False,
              decode_workers: int = 1, match: bool = True,
              eager_coverage: bool = False) -> dict:
    """Full scan. `targets` = a path or list of paths (files and/or directories),
    in place (no copying). Returns and PERSISTS clusters + review queue.
    `workers>1` parallelizes CPU features; `recursive=False` does not descend into subdirectories.
    `independent_scenes`: A (pixel-based scenes, slow) vs B/default (derived from emb).
    `max_height`: excludes videos with height > max_height (e.g. 1080 -> skips 4K).
    `progress`: shows friendly tqdm bars (phase, count, rate, ETA).
    `decode_workers`: >1 parallelizes video DECODE (prefetch) — SSD/NVMe only; on
    HDD disk concurrency causes thrashing (leave at 1).
    `match=False`: Pass-1 ONLY (compute+persist features, no matching). Use to (re)index a large
    library cheaply, then run Pass-2 separately — avoids the O(N^2) candidate blow-up until ready.
    `eager_coverage` (Deep depth): also compute whole-file audio coverage for ALL files (else it is
    deferred and ensured on-demand only for cluster members) — populates the Quality-warnings tab
    for the whole library. Incremental: only files missing coverage are computed."""
    paths = collect_videos(targets, recursive=recursive)
    excluded_by_height: list[str] = []
    if max_height:
        paths, excluded_by_height = filter_by_height(paths, max_height, workers)
    fv = feature_version(embedder, independent_scenes,
                         audio_fp_cap_s=th.audio_fp_cap_s, audio_fp_cap_above_s=th.audio_fp_cap_above_s)
    skipped = _pass1(paths, store, embedder, th, fv, force, workers, independent_scenes,
                     progress, decode_workers=decode_workers)
    if eager_coverage:                             # Deep: whole-file coverage for ALL (incremental)
        _ensure_coverage_all(paths, store, progress)
    if not match:                                  # Pass-1 only: features regenerated, no matching
        return {"clusters": [], "review_queue": [], "editions": [],
                "skipped": skipped, "excluded_by_height": excluded_by_height}

    # --- coarse index (A2: global + window). Small; needed for retrieval in both Pass-2 paths.
    # NOTE: skips LITE/exact-only records (no embeddings) -> a mixed DB doesn't crash here.
    if progress:
        tqdm.write("Building coarse index...")
    all_paths, gvecs = store.all_global_vecs()
    w_owners, wvecs = store.all_window_vecs()
    index = CoarseIndex(dim=gvecs.shape[1] if gvecs.size else th.raw["embeddings"]["dim"])
    index.build(all_paths, gvecs, window_owners=w_owners, window_vecs=wvecs)

    # --- Pass 2: match (persists pairs) + review/editions queue ---
    review, editions = _pass2(paths, store, index, th, progress)

    _apply_name_grouping(store, th)                # name copies (N) -> NAME_COPY (probable)
    clusters_out = _rebuild_clusters(store, th)
    return {"clusters": clusters_out, "review_queue": review, "editions": editions,
            "skipped": skipped, "excluded_by_height": excluded_by_height}


def _ensure_coverage_all(paths, store, progress) -> None:
    """Deep depth: ensure whole-file audio coverage for every scanned file. INCREMENTAL — only
    files whose coverage is still NULL are computed (the rest are reused), so running Deep after a
    Standard scan does NOT re-decode anything; it just fills the missing coverage."""
    bar = tqdm(paths, desc="Audio quality (coverage)", unit="file",
               disable=not progress, dynamic_ncols=True)
    for p in bar:
        rec = store.load(p, with_embeddings=False)
        if rec is not None:
            ensure_audio_coverage(p, store, rec.probe.duration_s, bool(rec.probe.audio_tracks))


def _pass2(paths, store, index, th, progress):
    """Pass-2 dispatcher. Parallel over candidate pairs when multiple cores are available
    (Pass-2 is compute-bound: banded video DP ~89%), else the sequential match() loop.
    Results are deterministic per pair -> verdict invariant (§0). Returns (review, editions)."""
    match_workers = min(16, max(1, (os.cpu_count() or 4) - 2))
    if match_workers > 1 and len(paths) >= 8:      # parallel only when it outweighs pool overhead
        return _pass2_parallel(paths, store, index, th, match_workers, progress)
    # Sequential matcher: lazy + LRU-bounded resident cache -> loads only the films actually
    # compared (never preloads the whole library, which could OOM the GPU on a large DB).
    cache = EmbeddingCache(store, max_items=1500)
    return _pass2_sequential(paths, store, index, th, cache, progress)


def _classify(verdict, a, b, reason, conf, review, editions) -> None:
    if verdict in REVIEW_VERDICTS:
        review.append((a, b, reason, conf))
    elif verdict == Verdict.DIFFERENT_EDITION:
        editions.append((a, b, reason))


def _pass2_parallel(paths, store, index, th, workers, progress):
    review: list = []
    editions: list = []
    if progress:
        tqdm.write(f"Pass 2: matching candidate pairs on {workers} workers…")
    for a, b, vval, conf, reason, ad_off, aj, vj, sj in match_pairs_parallel(
            paths, store, index, th, workers, progress=progress):
        store.save_match(a, b, vval, conf, reason, ad_offset_s=ad_off,
                         audio_json=json.dumps(aj), video_json=json.dumps(vj),
                         scenes_json=json.dumps(sj))
        _classify(Verdict(vval), a, b, reason, conf, review, editions)
    return review, editions


def _pass2_sequential(paths, store, index, th, cache, progress):
    review: list = []
    editions: list = []
    evaluated: set[tuple[str, str]] = set()        # C2: a pair is evaluated/acted on once
    bar2 = tqdm(paths, desc="Pass 2 (duplicates)", unit="film",
                disable=not progress, dynamic_ncols=True)
    for p in bar2:
        rec = store.load(p, with_embeddings=False)
        if rec is None:                            # skipped in pass 1 (unreadable)
            continue
        bar2.set_postfix_str(f"review={len(review)} | {_short(p)}")
        for res in match(rec, store, index, th, cache=cache, seen=evaluated):
            store.save_match(
                p, res.candidate_path, res.verdict.value, res.confidence, res.reason,
                ad_offset_s=(res.video.offset if res.video else None),   # C3
                audio_json=json.dumps(asdict(res.audio)) if res.audio else "",
                video_json=json.dumps(asdict(res.video)) if res.video else "",
                scenes_json=json.dumps(asdict(res.scenes)) if res.scenes else "",
            )
            _classify(res.verdict, p, res.candidate_path, res.reason, res.confidence,
                      review, editions)
    return review, editions


def _apply_name_grouping(store: FingerprintStore, th: Thresholds) -> None:
    """Marks as NAME_COPY pairs that differ only by `(N)` in the same directory, with a CONTENT
    VETO: only when the content does NOT contradict (same video). NAME_COPY ∈ DUPLICATE_VERDICTS,
    so a false one would make a DIFFERENT video deletable -> the veto protects the zero-FP guarantee
    (§0). Different videos that reuse a '(N)' name (real case in the library) are NOT grouped.
    A pair is skipped if it already has a content verdict, if content re-verifies as DIFFERENT, or
    if it can't be verified (a copy is LITE/exact-only with no embeddings). Opt-out: name_copy_grouping."""
    if not th.name_copy_grouping:
        return
    from dupdetect.names import name_sibling_pairs
    for base, other in name_sibling_pairs(store.all_paths()):
        if store.has_match(base, other):           # content already produced a verdict -> respect it
            continue
        if name_pair_content_differs(base, other, store, th):   # veto: DIFFERENT / unverifiable
            continue
        store.save_match(base, other, Verdict.NAME_COPY.value, 0.75,
                         "same name except for (N) in the same folder — content does not contradict")


_DUPLICATE_VALUES = {v.value for v in DUPLICATE_VERDICTS}


def _rebuild_clusters(store: FingerprintStore, th: Thresholds) -> list[dict]:
    """A5: clusters = derived view of the GLOBAL `matches` graph (not the yields of
    this run). Rebuilds the full table -> a re-scan leaves no stale clusters
    dangling (a path in two clusters). Union-find over all duplicate pairs."""
    uf = UnionFind()
    for a, b, verdict in store.all_matches():
        if verdict in _DUPLICATE_VALUES:
            uf.union(a, b)
    store.clear_clusters()
    clusters_out = []
    for cid, members in enumerate(uf.groups().values()):
        if len(members) <= 1:
            continue
        ranked = rank_cluster(members, store, th)
        for m in members:
            store.save_cluster(cid, m, is_keep=(m == ranked["keep"]),
                               rank_reason=ranked["evidence"].get(m, ""))
        clusters_out.append({"cluster_id": cid, **ranked})
    return clusters_out


EXACT_FV = "exact-only-v1"     # feature_version sentinel for LITE records (exact-only)


def _exact_worker(path: str):
    """LITE: only what's needed for EXACT duplicates — hash + probe. NO decode/embed/audio/
    whisper (the expensive part). Picklable for the ProcessPool."""
    st = os.stat(path)
    return (path, st.st_mtime, st.st_size, content_hash(path), ffprobe(path))


def exact_scan(targets, store: FingerprintStore, th: Thresholds, workers: int = 8,
               recursive: bool = True, max_height: int | None = None,
               progress: bool = False) -> dict:
    """'Exact-only' mode: detects BYTE-IDENTICAL duplicates by content_hash, WITHOUT the expensive
    pass (no decode/embed/audio/whisper). ~0.1s/file vs ~12s. Reuses the hash of already-indexed
    files (incremental, doesn't clobber full records) and saves LITE records for the new ones
    (the UI shows them; a FULL scan later re-indexes them). Rebuilds the clusters from the hash
    groups."""
    paths = collect_videos(targets, recursive=recursive)
    excluded: list[str] = []
    if max_height:
        paths, excluded = filter_by_height(paths, max_height, workers)

    by_hash: dict[tuple, list[str]] = {}                  # (hash, size) -> identical paths
    todo: list[str] = []
    for p in paths:                                       # reuse hash if the file didn't change
        try:
            st = os.stat(p)
        except OSError:
            continue
        h = store.content_hash_if_unchanged(p, st)
        if h is not None:
            by_hash.setdefault((h, st.st_size), []).append(p)
        else:
            todo.append(p)

    skipped: list[tuple[str, str]] = []
    bar = tqdm(total=len(todo), desc="Hashing (exact only)", unit="file",
               disable=not progress, dynamic_ncols=True)

    def _consume(res_path, mtime, size, h, probe):
        store.save_meta(res_path, mtime, size, h, probe, EXACT_FV)
        by_hash.setdefault((h, size), []).append(res_path)

    if workers and workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_exact_worker, p): p for p in todo}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    _consume(*fut.result())
                except Exception as e:                                       # noqa: BLE001
                    skipped.append((p, str(e))); store.save_problem(p, str(e))
                bar.update(1)
    else:
        for p in todo:
            try:
                _consume(*_exact_worker(p))
            except Exception as e:                                           # noqa: BLE001
                skipped.append((p, str(e))); store.save_problem(p, str(e))
            bar.update(1)
    bar.close()

    # clusters = byte-identical groups (same hash+size, >1 copy)
    store.clear_clusters()
    groups = [(k, v) for k, v in by_hash.items() if len(v) > 1]
    clusters_out = []
    for cid, (_, members) in enumerate(groups):
        ranked = rank_cluster(members, store, th)        # identical copies: arbitrary keep, ok
        for m in members:
            store.save_cluster(cid, m, is_keep=(m == ranked["keep"]),
                               rank_reason=ranked["evidence"].get(m, ""))
        clusters_out.append({"cluster_id": cid, **ranked})

    return {"clusters": clusters_out, "review_queue": [], "editions": [],
            "skipped": skipped, "excluded_by_height": excluded}


def _cluster_has_ads(store: FingerprintStore, member: str, cluster: set[str], th: Thresholds,
                     min_ad_s: float = 5.0) -> bool:
    """Does `member` carry inserted commercials relative to another cluster member? Two shapes:
      - PREPENDED ads: an alignment offset (C3; offset>0 => b_path starts later, ads at b's head).
      - MID-ROLL ads: foreign blocks spliced INSIDE the content (AlignResult.interleaved_ratio +
        ad_dir, measured 2026-06-12). ad_dir points at the LONGER (ad) copy. Verdict is untouched —
        the ad copy stays a duplicate; this only steers KEEP to the clean copy and flags which is which."""
    rows = store.conn.execute(
        "SELECT a_path, b_path, ad_offset_s, video_json FROM matches WHERE a_path=? OR b_path=?",
        (member, member),
    ).fetchall()
    for r in rows:
        if r["a_path"] not in cluster or r["b_path"] not in cluster:
            continue
        off = r["ad_offset_s"]
        if off is not None and (
                (r["b_path"] == member and off > min_ad_s) or (r["a_path"] == member and off < -min_ad_s)):
            return True
        vj = r["video_json"]
        if vj and th.ad_interleaved_min > 0:
            try:
                d = json.loads(vj)
            except (ValueError, TypeError):
                d = {}
            ratio = d.get("interleaved_ratio") or 0.0
            adir = d.get("ad_dir") or 0
            if ratio >= th.ad_interleaved_min and (
                    (r["b_path"] == member and adir == 1) or (r["a_path"] == member and adir == -1)):
                return True
    return False


def _color_diverges(scored) -> bool:
    """True if any two cluster members' color GRADE (cast/saturation/contrast) differ enough — i.e.
    a copy was re-graded. `scored` items are (score, path, rec, ...)."""
    cs = [t[2].quality.color for t in scored]
    return any(cs[i].grade_distance(cs[j]) > GRADE_DIVERGENCE
               for i in range(len(cs)) for j in range(i + 1, len(cs)))


def _ensure_lang(rec, store: FingerprintStore, th: Thresholds) -> None:
    """DEFERRED language detection: whisper runs here (cluster ranking) instead of in Pass-1,
    since lang_detected is consumed ONLY for KEEP selection within a cluster (it never enters the
    decision tree). Most unique files are never clustered -> never pay for whisper. Computed once
    and persisted -> identical output, far less work at scale. A failure -> None (rank falls back
    to resolution, as before)."""
    if rec.quality.lang_detected:
        return
    try:
        lang = detect_language(rec.path, model=th.raw["quality"]["whisper_model"])
    except Exception:                                   # noqa: BLE001
        lang = None
    rec.quality.lang_detected = lang
    if lang:
        store.set_lang(rec.path, lang)


def rank_cluster(members: list[str], store: FingerprintStore, th: Thresholds) -> dict:
    """Ranks members by QUALITY (not identity) and marks the 'keep'.

    Priority (highest to lowest weight): wanted language >> RESOLUTION >> no ads
    >> lower cam >> higher bitrate. Resolution dominates among the robust signals because
    language and cam proved unreliable on real data (see quality/camrip.py).
    Returns {keep, discard, evidence}.
    """
    wanted = set(th.raw["quality"]["wanted_langs"])
    cluster = set(members)
    scored = []
    for m in members:
        rec = store.load(m, with_embeddings=False)
        _ensure_lang(rec, store, th)               # deferred whisper: only cluster members need it
        # ON-DEMAND audio coverage (deferred out of Pass-1): ensure it for cluster members so the
        # KEEP/audio-warning decision is correct (never keep the muted copy by mistake). Cheap +
        # cached -> reused next run. Mutate the loaded rec so the audio logic below reads the real value.
        rec.quality.audio_coverage = ensure_audio_coverage(
            m, store, rec.probe.duration_s, bool(rec.probe.audio_tracks))
        pixels = (rec.probe.width or 0) * (rec.probe.height or 0)
        br = rec.probe.bitrate_kbps or 0
        lang_ok = rec.quality.lang_detected in wanted
        has_ads = _cluster_has_ads(store, m, cluster, th)
        score = (
            (1_000_000_000 if lang_ok else 0)        # wanted language: dominant
            + pixels                                 # resolution: robust, dominates the rest
            - (500_000 if has_ads else 0)            # ads: penalize (removable)
            - rec.quality.cam_score * 100_000        # cam: weak signal
            - th.color_clip_keep_weight * rec.quality.color.clip   # clipping: destroyed detail
            #                                        # -> prefer the least-clipped copy (the original)
            + br                                     # bitrate: fine tiebreak
        )
        scored.append((score, m, rec, lang_ok, pixels, br, has_ads))

    # Final tiebreak: SHORTEST PATH. Among equivalent copies keep the original
    # ('movie.avi' over 'movie (1).avi') and prefer the shortest-path location.
    scored.sort(key=lambda t: (t[0], -len(t[1])), reverse=True)
    # Audio guard: block auto-KEEP ONLY when a copy has truncated audio AND the copies DIFFER in
    # coverage (one copy really has better audio) -> review, so we never blindly keep the muted one.
    # If every copy shares ~the same coverage (same source, even if truncated), audio is not a
    # differentiator -> auto-pick by the quality score (avoids hiding obvious dups in Review).
    covs = [rec.quality.audio_coverage for _s, _m, rec, *_ in scored]
    audio_bad = any(c < AUDIO_OK_COVERAGE for c in covs) and (max(covs) - min(covs)) > AUDIO_COV_TOL
    keep = None if audio_bad else scored[0][1]
    # Color divergence (a copy was re-graded, e.g. a bad auto color-correct): the score above can
    # pick a higher-res re-grade that CLIPPED detail (crushed blacks). When the grade diverges,
    # prefer the LEAST-CLIPPED copy (the preserved/original look) as the suggestion; the UI also
    # flags '⚠ color differs' so the user can override. Audio warning still takes precedence.
    if keep is not None and _color_diverges(scored):
        # Prefer the least-clipped copy (the preserved/original look) ONLY when the score-winner
        # clips SIGNIFICANTLY more — i.e. a higher-res copy that actually DESTROYED detail (crushed
        # blacks / bad upscale). A trivial clip edge must NOT downgrade a real resolution upgrade
        # (a 1080p @0% clip must never beat a genuine 4K @1% clip). Measured: original ~1% vs bad
        # re-grade ~26% -> CLIP_DOWNGRADE_MARGIN separates noise from destroyed detail.
        least = min(scored, key=lambda t: t[2].quality.color.clip)
        if scored[0][2].quality.color.clip - least[2].quality.color.clip > CLIP_DOWNGRADE_MARGIN:
            keep = least[1]

    def role(m: str) -> str:
        if m == keep:
            return "KEEP"
        return "review" if keep is None else "discard"
    evidence = {
        m: "%s: lang=%s%s, %dx%d, %dkbps, cam=%.2f%s%s" % (
            role(m),
            rec.quality.lang_detected or "?", " (wanted)" if lang_ok else "",
            rec.probe.width or 0, rec.probe.height or 0, br, rec.quality.cam_score,
            ", ads" if has_ads else "",
            _audio_note(rec.quality.audio_coverage, rec.probe.duration_s))
        for _, m, rec, lang_ok, _, br, has_ads in scored
    }
    return {"keep": keep, "discard": [m for _, m, *_ in scored if m != keep],
            "evidence": evidence, "audio_warning": audio_bad}


def _audio_note(cov: float, duration_s: float) -> str:
    """Audio warning text for the cluster evidence ('' if the audio is complete)."""
    if cov >= AUDIO_OK_COVERAGE:
        return ""
    if cov <= 0.001:
        return ", ⚠ NO AUDIO"
    secs = int(cov * duration_s) if duration_s > 0 else 0
    return f", ⚠ audio ends ~{secs // 60}:{secs % 60:02d} of {int(duration_s) // 60}:{int(duration_s) % 60:02d}"
