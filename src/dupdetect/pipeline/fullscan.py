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

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from dupdetect.config import Thresholds
from dupdetect.features.audio_fp import AUDIO_COV_TOL, AUDIO_OK_COVERAGE
from dupdetect.features.embeddings import Embedder
from dupdetect.features.frames import decode_frames
from dupdetect.features.hashing import content_hash
from dupdetect.features.probe import ffprobe
from dupdetect.match.cache import EmbeddingCache
from dupdetect.match.matcher import match, match_pairs_parallel, name_pair_content_differs
from dupdetect.match.retrieval import CoarseIndex
from dupdetect.match.tree import DECISION_VERSION, DUPLICATE_VERDICTS, REVIEW_VERDICTS, T0_REASON
from dupdetect.models import Verdict
from dupdetect.pipeline.analyze import (
    analysis_state,
    analyze_file,
    build_record,
    ensure_audio_coverage,
    extract_cpu_features,
    extract_gpu_features,
    feature_version,
    maybe_emit_viz,
    record_from_donor,
)
from dupdetect.quality.color import CLIP_DOWNGRADE_MARGIN, GRADE_DIVERGENCE
from dupdetect.quality.language import detect_language
from dupdetect.store import FingerprintStore
from dupdetect.store.store import canonical_pair


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


def _needs_analysis(store: FingerprintStore, p: str, fv: str, force: bool) -> bool:
    """A file needs (re)analysis unless the shared freshness contract (analysis_state — same rule the
    watcher applies) says it's 'done' (fresh, or already-failed and unchanged). `force` overrides.
    Unstattable ('gone') -> analyze anyway, so analyze_file surfaces the real error."""
    return force or analysis_state(store, p, fv) in ("gone", "pending")


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
    todo = [p for p in paths if _needs_analysis(store, p, fv, force)]
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
        # M4: before paying the decode, a byte-identical file already indexed (a MOVE/copy) donates
        # its features. Only on this serial-decode path — the pipelined path already decoded.
        donor = record_from_donor(cpu, store, fv)
        if donor is not None:
            store.save(donor, feature_version=fv)
            return
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
    # Incremental Pass-2: the set of files Pass-1 will (re)analyze — their content moved, so their
    # candidate pairs must be re-aligned even if the ledger has them. Captured BEFORE _pass1 (after it
    # runs they're all 'fresh' again). Empty on a no-change re-run -> Pass-2 skips every prior pair.
    # Only needed when we'll actually match (skip the has_fresh sweep on a Pass-1-only re-index).
    changed = _changed_paths(paths, store, fv, force) if match else set()
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
    # Incremental ledger: skip re-aligning pairs already evaluated under this fingerprint (fv + θ),
    # except those touching a file re-analyzed this run (`changed`). A no-change re-run aligns nothing.
    fingerprint = _scan_fingerprint(fv, th)
    evaluated = store.evaluated_pairs_load(fingerprint)
    review, editions = _pass2(paths, store, index, th, progress,
                              evaluated=evaluated, changed=changed, fingerprint=fingerprint)

    _apply_name_grouping(store, th)                # name copies (N) -> NAME_COPY (probable)
    clusters_out = _rebuild_clusters(store, th, progress=progress)
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


def _changed_paths(paths, store: FingerprintStore, fv: str, force: bool) -> set:
    """Files Pass-1 will (re)analyze = those NOT already fresh (or all, when force). Their content may
    have moved, so the incremental Pass-2 must NOT skip their candidate pairs. Cheap: one stat() +
    indexed lookup per file (MFT-cached). Unstattable -> treated as changed (safe: re-evaluate)."""
    if force:
        return set(paths)
    changed = set()
    for p in paths:
        try:
            if not store.has_fresh(p, os.stat(p), fv):
                changed.add(p)
        except OSError:
            changed.add(p)
    return changed


def _scan_fingerprint(fv: str, th: Thresholds) -> str:
    """Identity of 'how Pass-2 would decide a pair': the feature_version (embedding/audio algorithm)
    plus ALL thresholds. Any change here can flip a verdict (e.g. looser θ turns a DIFFERENT into a
    match), so it keys the evaluated-pairs ledger -> a change invalidates every cached evaluation."""
    raw = json.dumps(th.raw, sort_keys=True, default=str)
    # DECISION_VERSION covers code-only changes to the tree that th.raw cannot see (a new guard/tier
    # can flip a verdict for the same signals + thresholds) -> its bump invalidates the ledger too.
    return hashlib.blake2b(("%s\x00%d\x00%s" % (fv, DECISION_VERSION, raw)).encode("utf-8", "surrogatepass"),
                           digest_size=16).hexdigest()


def _pass2(paths, store, index, th, progress, evaluated=None, changed=None, fingerprint=None):
    """Pass-2 dispatcher. Parallel over candidate pairs when multiple cores are available
    (Pass-2 is compute-bound: banded video DP ~89%), else the sequential match() loop.
    Results are deterministic per pair -> verdict invariant (§0). Returns (review, editions).
    `evaluated`/`changed`/`fingerprint` drive the incremental ledger (parallel path only)."""
    # Pass-2 align is CPU-bound (banded DP) with each worker's BLAS pinned to 1 thread, so it scales
    # ~linearly with cores. Use cpu-2 (leave 2 for the main collector + OS), capped at 32 to keep the
    # process count sane on very large boxes. (Was hard-capped at 16, which left half of a 32-core
    # machine idle.) Speed only -> verdict unchanged (§0).
    match_workers = min(32, max(1, (os.cpu_count() or 4) - 2))
    if match_workers > 1 and len(paths) >= 8:      # parallel only when it outweighs pool overhead
        return _pass2_parallel(paths, store, index, th, match_workers, progress,
                               evaluated=evaluated, changed=changed, fingerprint=fingerprint)
    # Sequential matcher: lazy + LRU-bounded resident cache -> loads only the films actually
    # compared (never preloads the whole library, which could OOM the GPU on a large DB). The
    # incremental ledger is parallel-only; small/single-core libraries re-evaluate (cheap at that size).
    cache = EmbeddingCache(store, max_items=1500)
    return _pass2_sequential(paths, store, index, th, cache, progress)


def _classify(verdict, a, b, reason, conf, review, editions) -> None:
    if verdict in REVIEW_VERDICTS:
        review.append((a, b, reason, conf))
    elif verdict in (Verdict.DIFFERENT_EDITION, Verdict.CONTAINS):
        editions.append((a, b, reason))          # "related, not duplicates" (Part 2 splits the types)


def _on_disk(p: str, seen: dict[str, bool]) -> bool:
    """§0 'detect, don't trust' guard against a CONCURRENT deletion. The coarse index is an in-memory
    SNAPSHOT taken at scan start, so a file the user trashes MID-SCAN still produces candidate pairs
    that would RESURRECT the `matches` row the UI just forgot (actions.delete_files -> forget_file).
    Never (re)persist a match for a path that has left the disk -> the user's deletion wins the race.
    Memoized per scan: at most one stat() per path, and stat hits the MFT (cheap, cached)."""
    if p not in seen:
        seen[p] = os.path.exists(p)
    return seen[p]


def _both_on_disk(a: str, b: str, seen: dict[str, bool]) -> bool:
    """Both endpoints of a candidate pair still on disk -> safe to (re)persist the match (see _on_disk)."""
    return _on_disk(a, seen) and _on_disk(b, seen)


def _pass2_parallel(paths, store, index, th, workers, progress,
                    evaluated=None, changed=None, fingerprint=None):
    review: list = []
    editions: list = []
    ondisk: dict[str, bool] = {}                   # §0 concurrent-deletion guard memo
    if progress:
        tqdm.write(f"Pass 2: matching candidate pairs on {workers} workers…")
    for a, b, vval, conf, reason, ad_off, aj, vj, sj in match_pairs_parallel(
            paths, store, index, th, workers, progress=progress,
            evaluated=evaluated, changed=changed, fingerprint=fingerprint):
        if not _both_on_disk(a, b, ondisk):        # a copy was trashed mid-scan -> don't resurrect it
            continue
        store.save_match(a, b, vval, conf, reason, ad_offset_s=ad_off,
                         audio_json=json.dumps(aj), video_json=json.dumps(vj),
                         scenes_json=json.dumps(sj))
        _classify(Verdict(vval), a, b, reason, conf, review, editions)
    return review, editions


def _persist_match(store, src: str, res, review, editions) -> None:
    """Persist ONE match result (C3 ad_offset + audio/video/scenes JSON) and route it into the
    review/editions queues. Extracted to keep `_pass2_sequential`'s loop flat."""
    store.save_match(
        src, res.candidate_path, res.verdict.value, res.confidence, res.reason,
        ad_offset_s=(res.video.offset if res.video else None),   # C3
        audio_json=json.dumps(asdict(res.audio)) if res.audio else "",
        video_json=json.dumps(asdict(res.video)) if res.video else "",
        scenes_json=json.dumps(asdict(res.scenes)) if res.scenes else "",
    )
    _classify(res.verdict, src, res.candidate_path, res.reason, res.confidence, review, editions)


def _pass2_sequential(paths, store, index, th, cache, progress):
    review: list = []
    editions: list = []
    evaluated: set[tuple[str, str]] = set()        # C2: a pair is evaluated/acted on once
    ondisk: dict[str, bool] = {}                   # §0 concurrent-deletion guard memo
    bar2 = tqdm(paths, desc="Pass 2 (duplicates)", unit="film",
                disable=not progress, dynamic_ncols=True)
    for p in bar2:
        if not _on_disk(p, ondisk):                # source trashed mid-scan -> skip (don't resurrect)
            continue
        rec = store.load(p, with_embeddings=False)
        if rec is None:                            # skipped in pass 1 (unreadable)
            continue
        bar2.set_postfix_str(f"review={len(review)} | {_short(p)}")
        for res in match(rec, store, index, th, cache=cache, seen=evaluated):
            if not _both_on_disk(p, res.candidate_path, ondisk):   # candidate trashed mid-scan -> skip
                continue
            _persist_match(store, p, res, review, editions)
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


def _snapshot_clusters(store: FingerprintStore) -> dict:
    """Map the CURRENT clusters by membership signature -> (keep_path, {path: rank_reason}). Lets a
    removal-only reconcile reuse the ranking of clusters whose membership didn't change, skipping the
    expensive rank_cluster (whisper + whole-file audio). Captured BEFORE the forgets so an affected
    cluster's pre-removal signature no longer matches its (smaller) rebuilt group -> it re-ranks."""
    by_cid: dict = {}
    for r in store.conn.execute("SELECT cluster_id, path, is_keep, rank_reason FROM clusters"):
        ent = by_cid.setdefault(r["cluster_id"], {"paths": set(), "keep": None, "reasons": {}})
        ent["paths"].add(r["path"])
        ent["reasons"][r["path"]] = r["rank_reason"] or ""
        if r["is_keep"]:
            ent["keep"] = r["path"]
    return {frozenset(e["paths"]): (e["keep"], e["reasons"]) for e in by_cid.values()}


def _stable_cluster_id(members: list[str]) -> int:
    """A content-derived, STABLE cluster id (56-bit) from the component's lexicographically smallest
    member. Two INDEPENDENT rebuilds therefore agree on the id for a given component, and DISTINCT
    components never collide — so concurrent rebuilds can't fuse unrelated content under a shared
    `enumerate()` index (the bug that merged 88/169 clusters). Stable across rebuilds = the UI/KEEP
    selection survives a refresh too."""
    key = min(members).encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(key, digest_size=7).digest(), "big")


def _rebuild_clusters(store: FingerprintStore, th: Thresholds, reuse: dict | None = None,
                      progress: bool = False) -> list[dict]:
    """A5: clusters = derived view of the GLOBAL `matches` graph (not the yields of this run).
    Rebuilds the full table ATOMICALLY (one transaction via store.replace_clusters) with STABLE,
    content-derived cluster ids -> a concurrent rebuild can neither interleave rows nor reuse an id
    for a different component, so distinct match-components are never fused. Union-find over all
    duplicate pairs; a removed hub correctly SPLITS its cluster.

    `reuse` (a `_snapshot_clusters` map taken BEFORE a removal): a rebuilt group whose membership is
    unchanged keeps its persisted KEEP/rank_reason instead of re-running rank_cluster — so deleting a
    file only re-ranks the clusters it actually touched. NOT passed by full_scan / ingest, where a
    member's data may have changed (re-encode) and ranking must be fresh."""
    reuse = reuse or {}
    uf = UnionFind()
    # A user "not a duplicate" veto OUTRANKS the content verdict: never union a vetoed pair. This is
    # what makes a corrected group STAY corrected — a re-scan re-aligns and would happily re-declare
    # the pair CERTAIN, but the veto lives in `feedback` and is re-applied on every rebuild.
    vetoed = store.vetoed_pairs()
    for a, b, verdict in store.all_matches():
        if verdict in _DUPLICATE_VALUES and canonical_pair(a, b) not in vetoed:
            uf.union(a, b)
    rows, clusters_out = [], []
    # Ranking each group runs the DEFERRED whisper + audio coverage per member -> a long, formerly
    # SILENT phase ("Group" step). Show it (§2: never look frozen); the clusters table is written
    # once at the end (atomic), so without this bar the UI sat blank for the whole ranking.
    groups = [m for m in uf.groups().values() if len(m) > 1]
    for members in tqdm(groups, desc="Pass 2 (grouping + best copy)", unit="group",
                        disable=not progress, dynamic_ncols=True):
        cid = _stable_cluster_id(members)
        cached = reuse.get(frozenset(members))
        if cached is not None:                     # membership unchanged -> reuse ranking (no whisper/audio)
            keep, reasons = cached
            ranked = {"keep": keep, "discard": [m for m in members if m != keep],
                      "evidence": reasons, "audio_warning": False}
        else:
            ranked = rank_cluster(members, store, th)
        for m in members:
            rows.append((cid, m, m == ranked["keep"], ranked["evidence"].get(m, "")))
        clusters_out.append({"cluster_id": cid, **ranked})
    store.replace_clusters(rows)                   # atomic: all-or-nothing, no interleave with a rebuild
    return clusters_out


EXACT_FV = "exact-only-v1"     # feature_version sentinel for LITE records (exact-only)


def _exact_worker(path: str):
    """LITE: only what's needed for EXACT duplicates — hash + probe. NO decode/embed/audio/
    whisper (the expensive part). Picklable for the ProcessPool."""
    st = os.stat(path)
    return (path, st.st_mtime, st.st_size, content_hash(path), ffprobe(path))


def _partition_by_hash(paths, store) -> tuple[dict, list[str]]:
    """Split paths into already-hashed (folded into `by_hash`, reusing the stored hash when the file
    is unchanged -> incremental) and `todo` (new/changed -> still need hashing)."""
    by_hash: dict[tuple, list[str]] = {}                  # (hash, size) -> identical paths
    todo: list[str] = []
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            continue
        h = store.content_hash_if_unchanged(p, st)
        if h is not None:
            by_hash.setdefault((h, st.st_size), []).append(p)
        else:
            todo.append(p)
    return by_hash, todo


def _hash_exact(todo, store, by_hash, workers, progress) -> list[tuple[str, str]]:
    """Hash the new/changed files (ProcessPool when workers>1, else serial), folding each into
    `by_hash` and saving its LITE record. RESILIENT: a corrupt file is skipped-and-reported, never
    aborts the scan (§2). Returns [(path, error)]."""
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
    return skipped


def _build_exact_clusters(by_hash, store, th) -> list[dict]:
    """Rebuild the clusters from byte-identical groups (same hash+size, >1 copy) and stamp each pair
    with the T0 CERTAIN verdict.

    M1: stamp the T0 verdict so each byte-identical pair reads as CERTAIN, not 'Review only'.
    exact_scan builds clusters but historically never wrote `matches`, so the verdict was empty and
    the two tables drifted (ui.data.drift_report). Members share (hash, size) -> the T0 tier holds by
    construction. Star topology (a representative linked to every other copy) is O(N); skip a pair
    that already carries a content verdict so a prior full-scan T1 is not clobbered (mirrors the
    has_match rule)."""
    groups = [v for _, v in by_hash.items() if len(v) > 1]
    rows, clusters_out = [], []
    for members in groups:
        cid = _stable_cluster_id(members)                # stable, content-derived: no cross-rebuild fusion
        ranked = rank_cluster(members, store, th)        # identical copies: arbitrary keep, ok
        for m in members:
            rows.append((cid, m, m == ranked["keep"], ranked["evidence"].get(m, "")))
        hub = ranked["keep"] or members[0]
        for m in members:
            if m != hub and not store.has_match(hub, m):
                store.save_match(hub, m, Verdict.CERTAIN.value, 1.00, T0_REASON)
        clusters_out.append({"cluster_id": cid, **ranked})
    store.replace_clusters(rows)                         # atomic full replace (no interleave with a rebuild)
    return clusters_out


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
    by_hash, todo = _partition_by_hash(paths, store)
    skipped = _hash_exact(todo, store, by_hash, workers, progress)
    clusters_out = _build_exact_clusters(by_hash, store, th)
    return {"clusters": clusters_out, "review_queue": [], "editions": [],
            "skipped": skipped, "excluded_by_height": excluded}


def _prepended_ad(r, member: str, min_ad_s: float) -> bool:
    """PREPENDED ads: an alignment offset (C3; offset>0 => b_path starts later, ads at b's head)
    larger than `min_ad_s` on the side `member` is on."""
    off = r["ad_offset_s"]
    if off is None:
        return False
    return (r["b_path"] == member and off > min_ad_s) or (r["a_path"] == member and off < -min_ad_s)


def _midroll_ad(r, member: str, th: Thresholds) -> bool:
    """MID-ROLL ads: foreign blocks spliced INSIDE the content (AlignResult.interleaved_ratio +
    ad_dir, measured 2026-06-12). ad_dir points at the LONGER (ad) copy."""
    vj = r["video_json"]
    if not (vj and th.ad_interleaved_min > 0):
        return False
    try:
        d = json.loads(vj)
    except (ValueError, TypeError):
        return False
    if (d.get("interleaved_ratio") or 0.0) < th.ad_interleaved_min:
        return False
    adir = d.get("ad_dir") or 0
    return (r["b_path"] == member and adir == 1) or (r["a_path"] == member and adir == -1)


def _row_marks_ads(r, member: str, th: Thresholds, min_ad_s: float) -> bool:
    """Does ONE match row mark `member` as the ad-carrying copy? (prepended OR mid-roll). Extracted
    to keep `_cluster_has_ads` flat; verdict is untouched, this only steers KEEP."""
    return _prepended_ad(r, member, min_ad_s) or _midroll_ad(r, member, th)


def _cluster_has_ads(store: FingerprintStore, member: str, cluster: set[str], th: Thresholds,
                     min_ad_s: float = 5.0) -> bool:
    """Does `member` carry inserted commercials relative to another cluster member? Verdict is
    untouched — the ad copy stays a duplicate; this only steers KEEP to the clean copy and flags
    which is which. Per-row detection (prepended / mid-roll) lives in `_row_marks_ads`."""
    rows = store.conn.execute(
        "SELECT a_path, b_path, ad_offset_s, video_json FROM matches WHERE a_path=? OR b_path=?",
        (member, member),
    ).fetchall()
    for r in rows:
        if r["a_path"] not in cluster or r["b_path"] not in cluster:
            continue
        if _row_marks_ads(r, member, th, min_ad_s):
            return True
    return False


def _color_diverges(scored) -> bool:
    """True if any two cluster members' color GRADE (cast/saturation/contrast) differ enough — i.e.
    a copy was re-graded. `scored` items are (score, path, rec, ...)."""
    cs = [t[2].quality.color for t in scored]
    return any(cs[i].grade_distance(cs[j]) > GRADE_DIVERGENCE
               for i in range(len(cs)) for j in range(i + 1, len(cs)))


def _whisper_device() -> str:
    """GPU for the deferred language detection when one is present, else CPU. Unlike Pass-1's
    language pass (which ran in forked CPU workers that must NOT touch CUDA), this runs in
    rank_cluster on the MAIN process, so CUDA is safe here — and measured ~10x faster on the
    fingerprint inference (CPU 7.0s -> GPU 0.7s per file on an RTX 5090). Speed only: the detected
    language is model-determined, not device-determined, and it only steers KEEP, never the verdict
    (§0). Falls back to CPU on the CPU-only flavor (no CUDA) or if torch is unavailable."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                                   # noqa: BLE001 — no torch -> CPU
        return "cpu"


def _ensure_lang(rec, store: FingerprintStore, th: Thresholds) -> None:
    """DEFERRED language detection: whisper runs here (cluster ranking) instead of in Pass-1,
    since lang_detected is consumed ONLY for KEEP selection within a cluster (it never enters the
    decision tree). Most unique files are never clustered -> never pay for whisper. Computed once
    and persisted -> identical output, far less work at scale. A failure -> None (rank falls back
    to resolution, as before). Runs on the GPU when available (see _whisper_device)."""
    if rec.quality.lang_detected:
        return
    try:
        lang = detect_language(rec.path, model=th.raw["quality"]["whisper_model"],
                               device=_whisper_device())
    except Exception:                                   # noqa: BLE001
        lang = None
    rec.quality.lang_detected = lang
    if lang:
        store.set_lang(rec.path, lang)


# Codec efficiency = quality-per-bit relative to H.264 (=1.0). Modern codecs (AV1/HEVC/VP9) reach the
# same visual quality at far lower bitrate, so RAW bitrate under-rates them; multiplying compares copies
# fairly (an AV1 at 3 Mbps ≈ an H.264 at 6). Steers KEEP only — never the verdict (§0).
_CODEC_EFF = {
    "av1": 2.0, "av01": 2.0, "hevc": 1.7, "h265": 1.7, "hev1": 1.7, "hvc1": 1.7,
    "vp9": 1.6, "vp09": 1.6, "h264": 1.0, "avc": 1.0, "avc1": 1.0, "vp8": 0.9,
    "mpeg4": 0.7, "msmpeg4v3": 0.7, "xvid": 0.7, "divx": 0.7, "vc1": 0.7, "wmv3": 0.7,
    "mpeg2video": 0.5, "mpeg2": 0.5, "mpeg1video": 0.4,
}
# Muted-check granularity for the KEEP candidate: spotting a silent/muted copy needs far fewer seek
# probes than precise coverage (each probe is an HDD seek). Deep's coverage-for-all keeps the default 40.
_KEEP_COVERAGE_POINTS = 12


def _effective_bitrate(codec: str | None, bitrate_kbps: int | None) -> float:
    """Bitrate scaled by codec efficiency -> a codec-fair quality proxy. Unknown codec -> H.264 (1.0)."""
    return (bitrate_kbps or 0) * _CODEC_EFF.get((codec or "").lower().strip(), 1.0)


def _score_member(m: str, store: FingerprintStore, th: Thresholds, cluster: set,
                  wanted: set, detect_lang: bool):
    """Score ONE cluster member by QUALITY from METADATA ONLY (no decode): RESOLUTION >> no ads
    >> lower cam >> codec-aware bitrate, minus clipping. Language is scored ONLY when `detect_lang`
    (opt-in: runs whisper) — OFF by default, since resolution dominates KEEP on real data and whisper is
    expensive; the muted-audio question is handled separately by the KEEP muted-check (_keep_by_audio),
    NOT per member. Returns (score, m, rec, has_ads, lang_ok)."""
    rec = store.load(m, with_embeddings=False)
    lang_ok = False
    if detect_lang:                                # opt-in whisper (on-demand language-preference KEEP)
        _ensure_lang(rec, store, th)
        lang_ok = rec.quality.lang_detected in wanted
    pixels = (rec.probe.width or 0) * (rec.probe.height or 0)
    has_ads = _cluster_has_ads(store, m, cluster, th)
    score = (
        (1_000_000_000 if lang_ok else 0)          # wanted language: dominant — ONLY when opted in
        + pixels                                   # resolution: robust, dominates the rest
        - (500_000 if has_ads else 0)              # ads: penalize (removable)
        - rec.quality.cam_score * 100_000          # cam: weak signal
        - th.color_clip_keep_weight * rec.quality.color.clip   # clipping: destroyed detail
        + _effective_bitrate(rec.probe.vcodec, rec.probe.bitrate_kbps)   # codec-aware fine tiebreak
    )
    return (score, m, rec, has_ads, lang_ok)


def _color_adjusted_keep(keep, scored, covs, store):
    """Color divergence (a copy was re-graded, e.g. a bad auto color-correct): the quality score can
    pick a higher-res re-grade that CLIPPED detail (crushed blacks). When the grade diverges, prefer
    the LEAST-CLIPPED copy (the preserved/original look) — but ONLY when the CURRENT keep clips
    SIGNIFICANTLY more, so a trivial clip edge never downgrades a real resolution upgrade (a 1080p
    @0% clip must not beat a genuine 4K @1% clip; measured: original ~1% vs bad re-grade ~26% ->
    CLIP_DOWNGRADE_MARGIN). No-op when keep is None.

    Won't undo the muted-check: it never downgrades onto a copy that has WORSE audio than the current
    keep (keep has audio, the less-clipped target is muted). The target's coverage is probed lazily
    here (cached) — only when a downgrade would otherwise fire, which is rare. Compares the chosen
    keep's clip (the muted-check may have escalated keep away from scored[0])."""
    if keep is None or not _color_diverges(scored):
        return keep
    least = min(scored, key=lambda t: t[2].quality.color.clip)
    keep_clip = next((t[2].quality.color.clip for t in scored if t[1] == keep), None)
    if keep_clip is None or keep_clip - least[2].quality.color.clip <= CLIP_DOWNGRADE_MARGIN:
        return keep
    keep_cov = covs.get(keep, 0.0)                  # keep was always probed by _keep_by_audio
    least_cov = covs.get(least[1])
    if least_cov is None:                           # not on keep's escalation path -> probe it now
        least_cov = ensure_audio_coverage(least[1], store, least[2].probe.duration_s,
                                          bool(least[2].probe.audio_tracks), n_points=_KEEP_COVERAGE_POINTS)
    if keep_cov >= AUDIO_OK_COVERAGE and least_cov < AUDIO_OK_COVERAGE:
        return keep                                 # downgrade would LOSE audio -> keep the clean copy
    return least[1]


def _rank_evidence(scored, keep, covs: dict) -> dict:
    """Human-readable per-member evidence (KEEP/discard/review + res/codec/bitrate/cam/ads/audio).
    Audio note only for copies actually probed (covs); language only if detected (opt-in)."""
    def role(m: str) -> str:
        if m == keep:
            return "KEEP"
        return "review" if keep is None else "discard"
    return {
        m: "%s: %dx%d, %s %dkbps, cam=%.2f%s%s%s" % (
            role(m), rec.probe.width or 0, rec.probe.height or 0,
            rec.probe.vcodec or "?", rec.probe.bitrate_kbps or 0, rec.quality.cam_score,
            ", ads" if has_ads else "",
            (", lang=%s%s" % (rec.quality.lang_detected, " (wanted)" if lang_ok else ""))
            if rec.quality.lang_detected else "",
            _audio_note(covs[m], rec.probe.duration_s) if m in covs else "")
        for _s, m, rec, has_ads, lang_ok in scored
    }


def _keep_by_audio(scored, store: FingerprintStore):
    """Pick KEEP by audio, probing coverage lazily DOWN the quality ranking (the escalation): KEEP the
    first copy whose audio is NOT muted, so a muted 'best' copy yields to a slightly lower one that
    actually has audio. The kept copy is clean -> no warning.

    If NO copy is clean, keep the top-scored one and warn ONLY when the coverages DIFFER (one copy is
    truly worse — genuine ambiguity -> review). When every copy shares ~the same coverage (same source,
    or all with no audio track) audio is not a differentiator -> auto-keep, NO warning (else obvious
    dups would hide in Review). Never returns None: the cluster always gets a suggested KEEP, and the
    warning routes the muted-ambiguous ones to review (§0: the UI never auto-deletes). Only the copies
    actually inspected get a probe (cheap, cached, _KEEP_COVERAGE_POINTS). Returns (keep, warning, covs)."""
    covs: dict = {}
    for _s, m, rec, *_ in scored:
        cov = ensure_audio_coverage(m, store, rec.probe.duration_s,
                                    bool(rec.probe.audio_tracks), n_points=_KEEP_COVERAGE_POINTS)
        covs[m] = cov
        rec.quality.audio_coverage = cov
        if cov >= AUDIO_OK_COVERAGE:
            return m, False, covs                  # first clean copy down the ranking wins (escalation)
    # no clean copy: keep the best-scored; warn only if coverages genuinely DIFFER (not same-source).
    vals = list(covs.values())
    differ = bool(vals) and (max(vals) - min(vals)) > AUDIO_COV_TOL
    return scored[0][1], differ, covs


def rank_cluster(members: list[str], store: FingerprintStore, th: Thresholds,
                 detect_lang: bool = False) -> dict:
    """Ranks members by QUALITY (not identity) and marks the 'keep'.

    KEEP = highest RESOLUTION, then codec-aware bitrate (an AV1/HEVC copy is NOT penalised for its
    lower raw bitrate), minus ads/cam/clipping — all from METADATA (no decode). Audio is reduced to its
    one decisive question for the KEEP: is it muted? (see _keep_by_audio — escalates to the next-best
    non-muted copy). Language is OFF by default (whisper is expensive and resolution dominates KEEP on
    real data); `detect_lang=True` opts in (on-demand language-preference KEEP). Returns
    {keep, discard, evidence, audio_warning}.
    """
    wanted = set(th.raw["quality"]["wanted_langs"]) if detect_lang else set()
    cluster = set(members)
    scored = [_score_member(m, store, th, cluster, wanted, detect_lang) for m in members]
    # Final tiebreak: SHORTEST PATH (keep 'movie.avi' over 'movie (1).avi').
    scored.sort(key=lambda t: (t[0], -len(t[1])), reverse=True)
    keep, audio_warning, covs = _keep_by_audio(scored, store)
    keep = _color_adjusted_keep(keep, scored, covs, store)   # re-grade clip -> original, but keep audio
    return {"keep": keep, "discard": [m for _, m, *_ in scored if m != keep],
            "evidence": _rank_evidence(scored, keep, covs), "audio_warning": audio_warning}


def _audio_note(cov: float, duration_s: float) -> str:
    """Audio warning text for the cluster evidence ('' if the audio is complete)."""
    if cov >= AUDIO_OK_COVERAGE:
        return ""
    if cov <= 0.001:
        return ", ⚠ NO AUDIO"
    secs = int(cov * duration_s) if duration_s > 0 else 0
    return f", ⚠ audio ends ~{secs // 60}:{secs % 60:02d} of {int(duration_s) // 60}:{int(duration_s) % 60:02d}"
