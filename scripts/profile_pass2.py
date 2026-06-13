"""Profiler for Pass-2: WHERE does the time go per candidate pair?

Wraps the REAL functions that match() uses (production code is NOT modified) and runs the
SEQUENTIAL match loop over a target folder's full-indexed files, then prints a per-stage
breakdown. The sequential path does the same per-pair work as the parallel one, so the stage
PROPORTIONS are representative (only wall-clock differs: parallel spreads it across cores).

Run AFTER any scan finishes — it reads the same disk and would compete with a running scan.
Privacy: prints aggregate timings only, never file names.

Usage:
  python scripts/profile_pass2.py "D:\\Videos\\Some Folder" [--db DB] [--limit N]
Portable: --db defaults to the per-user data dir; override with $DUPDETECT_DB.
"""
import argparse
import time
from collections import defaultdict

from dupdetect.config import load_thresholds
from dupdetect.match import matcher
from dupdetect.match.cache import EmbeddingCache
from dupdetect.match.retrieval import CoarseIndex
from dupdetect.runtime import default_db_path
from dupdetect.store import FingerprintStore

_T: dict = defaultdict(lambda: [0.0, 0])          # stage -> [total_seconds, n_calls]


def _timed(name: str, fn):
    """Wrap fn so its cumulative wall-time and call count land in _T[name]."""
    def wrap(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            d = _T[name]
            d[0] += time.perf_counter() - t0
            d[1] += 1
    return wrap


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-stage Pass-2 profiler.")
    ap.add_argument("target", help="folder prefix; its full-indexed files are the Pass-2 queries")
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--limit", type=int, default=0, help="cap target files (0 = all)")
    args = ap.parse_args()

    th = load_thresholds()
    store = FingerprintStore(args.db, init_schema=False)

    pref = args.target.rstrip("\\/")
    paths = [r[0] for r in store.conn.execute(
        "SELECT path FROM files WHERE path LIKE ? AND feature_version != 'exact-only-v1'",
        (pref + "%",)) if r[0].startswith(pref)]
    if args.limit:
        paths = paths[:args.limit]
    print(f"profiling Pass-2 on {len(paths)} full-indexed files under: {pref}")
    if not paths:
        store.close()
        return

    # Coarse index built EXACTLY like full_scan._pass2 (so candidate sets match production).
    all_paths, gvecs = store.all_global_vecs()
    w_owners, wvecs = store.all_window_vecs()
    index = CoarseIndex(dim=gvecs.shape[1] if gvecs.size else th.raw["embeddings"]["dim"])
    index.build(all_paths, gvecs, window_owners=w_owners, window_vecs=wvecs)
    cache = EmbeddingCache(store, max_items=1500)

    # Wrap the real, non-overlapping top-level stages match() calls (see matcher.match body).
    matcher.candidate_paths = _timed("candidate_paths (retrieval)", matcher.candidate_paths)
    matcher._ensure_audio_fp = _timed("ensure_audio_fp (fpcalc/HDD)", matcher._ensure_audio_fp)
    matcher.align_audio = _timed("align_audio", matcher.align_audio)
    matcher.align_video = _timed("align_video (banded DP)", matcher.align_video)
    matcher.align_scenes = _timed("align_scenes", matcher.align_scenes)
    EmbeddingCache.get = _timed("embedding cache.get (.npy I/O)", EmbeddingCache.get)
    FingerprintStore.load = _timed("store.load (metadata)", FingerprintStore.load)

    seen: set = set()
    n_pairs = 0
    t0 = time.perf_counter()
    for p in paths:
        rec = store.load(p, with_embeddings=False)
        if rec is None:
            continue
        before = len(seen)
        list(matcher.match(rec, store, index, th, cache=cache, seen=seen))
        n_pairs += len(seen) - before
    total = time.perf_counter() - t0

    print(f"\n=== Pass-2 profile: {len(paths)} files | {n_pairs} unique pairs | {total:.1f}s (single-core) ===")
    print(f"{'stage':32} {'total_s':>9} {'%':>6} {'calls':>9} {'ms/call':>9}")
    for name, (secs, calls) in sorted(_T.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:32} {secs:9.2f} {100 * secs / max(total, 1e-9):6.1f} "
              f"{calls:9} {1000 * secs / max(calls, 1):9.2f}")
    print(f"\nper target file: {total / max(len(paths), 1):.2f}s | "
          f"candidates/file: {n_pairs / max(len(paths), 1):.1f}")
    print("note: wall-clock is single-core; the real scan parallelizes the align across cores, "
          "but the per-stage proportions (where the time goes) hold.")
    store.close()


if __name__ == "__main__":
    main()
