"""Orchestration of match(): coarse -> fine -> tree.

C2: canonicalizes pairs (evaluated once per unordered pair).
A1: uses resident EmbeddingCache, no per-candidate disk reload.
A2: beyond top-k coarse, a duration safety net to avoid losing
    recall (a degraded cam rip may not enter the global_vec top-k).
C3: the align offset (pre-roll ads) is propagated here, it is a pair-level property.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
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
    # A record with NO global embedding (e.g. a file Pass-1 could not decode any frames from -> empty
    # vec) can't be video-matched, and querying faiss with a dim-0 vector trips `assert d == self.d`
    # and crashes the WHOLE Pass-2 batch. Skip it (§2: skip-and-report, never crash the batch) -> no
    # candidates, so it simply produces no matches.
    if rec.global_vec is None or not getattr(rec.global_vec, "size", 0):
        return set()
    cands = {p for p, _ in index.query_global(rec.global_vec, k=th.faiss_k)}
    if th.n_window_vecs > 0 and rec.window_vecs.size:
        cands |= index.query_windows(rec.window_vecs, k=th.raw["retrieval"]["window_faiss_k"])
    # Duration safety net, GATED by global cosine: on a dense library ±tol returns thousands of
    # same-length-but-different videos (O(N^2) Pass-2). Real dups are ~identical-duration AND
    # globally near (>=0.962 measured), so the gate prunes the dragnet without losing them. Only
    # this net is gated; top-k/window retrieval above are untouched.
    dur = duration_blocking(rec, store, th)
    cands |= index.gate_by_global(rec.global_vec, dur, th.duration_block_cos_gate)
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


def _audio_warranted(v: AlignResult, th: Thresholds) -> bool:
    """LAZY AUDIO gate (§1): does the video alignment reach the zone where the audio score can still
    change the verdict? Only T1 (audio corroborates) and T2 (audio does NOT align => different dub)
    consult audio, and both require `video.score >= theta_v AND coverage >= min_coverage`. Below
    that, no tier reads audio (the audio-only T4b branch was removed, see tree.decide_tree), so the
    fingerprint — the most expensive step (whole-file audio decode) — is skipped with an IDENTICAL
    verdict. Deterministic from the (cached) video align: same input -> same gate (§0)."""
    return v.score >= th.theta_v and v.coverage >= th.min_coverage


def _audio_if_video_warrants(rec: Record, other: Record, v: AlignResult,
                             store: FingerprintStore, th: Thresholds) -> AlignResult:
    """Align the two audio fingerprints ONLY when the video warrants it (see _audio_warranted),
    extracting them on-demand. Otherwise return an empty AlignResult (audio cannot affect the
    verdict here) -> the whole-file audio decode is never paid for video-weak pairs."""
    if not _audio_warranted(v, th):
        return AlignResult(0.0)
    fa = _ensure_audio_fp(rec, store, th)
    fb = _ensure_audio_fp(other, store, th)
    return align_audio(fa, fb, min_overlap_s=th.raw["audio"]["min_overlap_s"])


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

        # LAZY AUDIO (perf, §1): video + scenes run first — both are CHEAP (embeddings are the
        # resident cache / page-cached .npy, scene cuts are in the DB; no disk read). The audio
        # fingerprint is the single most expensive step (decodes the whole audio off disk), so it is
        # extracted ON-DEMAND only when the video is strong enough that audio can change the verdict
        # (the T1/T2 zone: video.score >= theta_v AND coverage >= min_coverage). Outside that zone no
        # tier consults audio (the audio-only T4b branch was removed), so an empty AlignResult yields
        # the IDENTICAL verdict — see decide_tree T4b. Unique films never pay for the fingerprint.
        v = _align_video_pair(rec, other, cache, th)
        s = align_scenes(rec.scene_cuts, other.scene_cuts, theta=th.theta_s)
        a = _audio_if_video_warrants(rec, other, v, store, th)

        res = decide_tree(rec, other, a, v, s, th)
        results.append(res)

    results.sort(key=lambda r: r.confidence, reverse=True)
    return [r for r in results if r.verdict != Verdict.DIFFERENT]


# --------------------------------------------------------------------------- Pass-2 parallel
# Pass-2 is CPU-COMPUTE-bound (per-pair similarity matmul + banded video DP), not I/O-bound: the
# embeddings are small fp16 .npy the OS page cache holds, so they don't thrash the disk. Parallelism
# is across PROCESSES (workers are READ-ONLY store handles, init_schema=False, so concurrent openers
# don't contend). Results are deterministic per pair -> verdict invariant (§0).
#
# THREAD PINNING (perf, §1): each worker's numpy matmul calls OpenBLAS, which by default spawns ~one
# thread per core. With W worker processes on a C-core box that is ~W*C threads fighting over C cores
# -> catastrophic oversubscription (measured the dominant Pass-2 throttle: 30 procs x ~32 BLAS threads
# on 32 cores). Parallelism must come from the PROCESSES, so each worker's BLAS is pinned to 1 thread
# (see single_threaded_blas, applied around the pool). Speed only -> verdict unchanged (§0).

_PW: dict = {}                                          # per-worker state (spawn-safe)

# Env vars every common BLAS/OpenMP backend honors for its internal thread count.
_BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextlib.contextmanager
def single_threaded_blas():
    """Pin BLAS/OpenMP to 1 thread for the duration, then restore the prior values. Wraps the Pass-2
    process pool so its W workers don't each launch a multi-threaded matmul (W*cores threads thrashing
    `cores` CPUs). Set in the PARENT *before* the pool so spawned workers inherit it at interpreter
    start (Windows spawn reads OPENBLAS_NUM_THREADS at the child's numpy import). Restored on exit so
    the parent's own later numpy/faiss work is unaffected. Verdict-invariant (§0: the matmul values are
    unchanged; only BLAS's internal thread count differs)."""
    prev = {k: os.environ.get(k) for k in _BLAS_THREAD_VARS}
    os.environ.update(dict.fromkeys(_BLAS_THREAD_VARS, "1"))
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
    v = _align_video_pair(ra, rb, _DictCache(cdict), th)
    s = align_scenes(ra.scene_cuts, rb.scene_cuts, theta=th.theta_s)
    a = _audio_if_video_warrants(ra, rb, v, store, th)   # lazy: audio only in the T1/T2 zone
    return decide_tree(ra, rb, a, v, s, th).verdict == Verdict.DIFFERENT


def _pass2_pair(pair: tuple[str, str]):
    """Worker: aligns ONE candidate pair and decides. Returns the row to persist, or None for
    DIFFERENT (mirrors match()'s filter).

    LAZY AUDIO (§1): video + scenes run first (CHEAP — embeddings/scene cuts, no disk read). The
    whole-file audio fingerprint is extracted+persisted ON-DEMAND, and only when the video warrants
    it (the T1/T2 zone). Most candidate pairs are weak video matches (a unique film's faiss
    neighbours) decided by video alone -> they never decode audio. The fingerprint is cached in the
    DB (set_audio_fp), so a file shared by several real-dup pairs is fingerprinted once; concurrent
    writes are serialized by WAL (busy_timeout). Verdict identical to the eager path (§0)."""
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
    v = _align_video_pair(ra, rb, _DictCache(cdict), th)
    s = align_scenes(ra.scene_cuts, rb.scene_cuts, theta=th.theta_s)
    a = _audio_if_video_warrants(ra, rb, v, store, th)   # on-demand, only in the T1/T2 zone
    res = decide_tree(ra, rb, a, v, s, th)
    if res.verdict == Verdict.DIFFERENT:
        return None
    return (a_path, b_path, res.verdict.value, res.confidence, res.reason,
            (v.offset if v else None), asdict(a), asdict(v), asdict(s))


def _enumerate_pairs(paths, store: FingerprintStore, index: CoarseIndex, th: Thresholds,
                     evaluated: set, changed: set, progress: bool) -> list:
    """Unique candidate-pair list for Pass-2. Skips pairs already aligned under the current fingerprint
    whose endpoints did NOT change this run (incremental). Shows a bar — enumeration is O(files) faiss/
    duration queries in the MAIN process, otherwise a silent multi-minute gap before the first pair (§2)."""
    from tqdm import tqdm
    pairs: set[tuple[str, str]] = set()
    for p in tqdm(paths, desc="Pass 2 (candidates)", unit="file",
                  disable=not progress, dynamic_ncols=True):
        rec = store.load(p, with_embeddings=False)
        if rec is None:
            continue
        for cand in candidate_paths(rec, store, index, th):
            pr = canonical_pair(p, cand)
            if (evaluated and pr[0] not in changed and pr[1] not in changed
                    and _pair_hash(pr) in evaluated):
                continue                               # already aligned, unchanged -> skip (incremental)
            pairs.add(pr)
    return list(pairs)


def match_pairs_parallel(paths, store: FingerprintStore, index: CoarseIndex,
                         th: Thresholds, workers: int, progress: bool = False,
                         evaluated: set | None = None, changed: set | None = None,
                         fingerprint: str | None = None):
    """Parallel Pass-2: enumerate unique candidate pairs, then align+decide each across a process
    pool. A GENERATOR — YIELDS each non-DIFFERENT row (the `_pass2_pair` tuple) AS its pair finishes,
    so the caller persists incrementally (a cancel/crash keeps what's done) instead of waiting for
    the whole batch. Same results as the sequential match() loop, just parallel.

    PROGRESS (§2 — never look frozen): two visible phases. (1) Candidate enumeration is O(files)
    faiss/duration queries in the MAIN process — a multi-minute SILENT gap before the first pair
    unless it shows a bar. (2) Alignment uses `as_completed`, NOT the ordered `pool.map`: the bar
    advances on EVERY pair that finishes, in ANY order. With the ordered map a single slow giant pair
    at the head stalled the whole bar to 0% while the other W-1 workers had already cleared thousands
    of pairs (CPU pegged, bar frozen — looked hung). Completion order ≠ submission order only changes
    the persist order, which is idempotent per canonical pair (§0 verdict-invariant).

    LAZY AUDIO (§1): there is NO eager whole-library audio-fingerprint pass any more. It used to run
    here, serially in the main process, fpcalc'ing EVERY involved file (= the whole library, since
    faiss top-k makes every file a candidate) before a single pair was compared — the measured
    Pass-2 bottleneck (~3.6s/file off an HDD, hours of serial disk reads). Each worker now extracts
    the fingerprint on-demand and ONLY for pairs whose video warrants it (see _pass2_pair /
    _audio_warranted), so unique films never decode their audio. Fingerprints are persisted, so a
    later run reuses them.

    INCREMENTAL (§1): `evaluated` = pair-hashes already aligned in a PRIOR scan under the SAME
    `fingerprint` (feature_version + θ). Such a pair is SKIPPED — its result is already persisted (a
    row in `matches`, or a DIFFERENT that needs none) — UNLESS an endpoint is in `changed` (re-analyzed
    this run, so its content moved and the pair must re-align). Every pair actually aligned is recorded
    back, so the NEXT run skips it. A re-run with nothing changed aligns ZERO pairs (just enumeration)."""
    pair_list = _enumerate_pairs(paths, store, index, th,
                                 evaluated or set(), changed or set(), progress)
    if not pair_list:
        return
    aligned: list[str] = []
    # single_threaded_blas: pin each worker's BLAS to 1 thread BEFORE the pool spawns, so W processes
    # don't oversubscribe the cores with W*cores matmul threads (perf, §1; verdict unchanged, §0).
    with single_threaded_blas(), ProcessPoolExecutor(
            max_workers=workers, initializer=_pass2_init,
            initargs=(str(store.db_path), th)) as pool:
        for pr, row in _drain_pairs_bounded(pool, pair_list, workers, progress):
            aligned.append(_pair_hash(pr))             # record EVERY aligned pair (match OR different)
            if row is not None:
                yield row
    if fingerprint is not None:
        store.evaluated_pairs_add(aligned, fingerprint)


def _pair_hash(pair: tuple[str, str]) -> str:
    """Stable 128-bit hash of a CANONICAL pair (a<=b) -> the evaluated-pairs ledger key (compact: a
    full library is millions of pairs, so storing paths twice would be far heavier)."""
    return hashlib.blake2b("\x00".join(pair).encode("utf-8", "surrogatepass"),
                           digest_size=16).hexdigest()


def _drain_pairs_bounded(pool, pair_list, workers: int, progress: bool):
    """Align `pair_list` on `pool`, yielding (pair, row) for EVERY pair AS it finishes (row is None for
    DIFFERENT). The caller filters None and records the pair in the ledger.

    BOUNDED submission: a dense library yields MILLIONS of pairs; submitting them all at once builds
    millions of Futures (GBs of RAM). Keep only ~4*workers in flight and refill one slot per completion
    -> memory O(workers), the pool never starves. wait(FIRST_COMPLETED) advances the bar on EVERY finish
    (any order), so one slow giant pair can't stall it (§2). Completion order is idempotent per pair (§0)."""
    from concurrent.futures import FIRST_COMPLETED, wait
    from tqdm import tqdm
    it = iter(pair_list)
    inflight: dict = {}                                # future -> pair
    for _ in range(max(1, workers) * 4):               # prime the window
        pr = next(it, None)
        if pr is None:
            break
        inflight[pool.submit(_pass2_pair, pr)] = pr
    bar = tqdm(total=len(pair_list), desc="Pass 2 (align pairs)", unit="pair",
               disable=not progress, dynamic_ncols=True)
    try:
        while inflight:
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                pr = inflight.pop(fut)
                bar.update(1)
                yield pr, fut.result()
                nxt = next(it, None)                   # refill one slot per completion
                if nxt is not None:
                    inflight[pool.submit(_pass2_pair, nxt)] = nxt
    finally:
        bar.close()
