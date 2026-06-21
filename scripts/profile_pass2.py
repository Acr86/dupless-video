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

from dupdetect.align import video as _video
from dupdetect.config import load_thresholds
from dupdetect.match import matcher
from dupdetect.match.cache import EmbeddingCache
from dupdetect.match.retrieval import CoarseIndex
from dupdetect.runtime import default_db_path
from dupdetect.store import FingerprintStore

_T: dict = defaultdict(lambda: [0.0, 0])          # stage -> [total_seconds, n_calls]

# `banded_align` is timed too, but it runs INSIDE align_video -> its time is NESTED in
# "align_video (matmul+DP)". Keep it OUT of the top-level table (would double-count to >100%) and
# report it only in the derived matmul/DP split below. This split is the decision gate for the next
# lever: matmul-dominated -> banded matmul (3a); DP-dominated -> JIT the DP loop (3b).
_DP_KEY = "  banded DP (nested)"


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


def _select_targets(store, pref: str, limit: int, biggest: bool):
    """Target files whose Pass-2 we profile. `biggest`: top-N by DURATION (worst-case align-time N,
    resolution-independent) to stress-test 4K/8K/long content; else a library-wide STRIDE sample
    (representative average, not the first N of one folder). Returns (paths, durations_s)."""
    order = "duration_s DESC" if biggest else "path"
    rows = [(r[0], r[1] or 0.0) for r in store.conn.execute(
        f"SELECT path, duration_s FROM files WHERE path LIKE ? AND feature_version != 'exact-only-v1' "
        f"ORDER BY {order}", (pref + "%",)) if r[0].startswith(pref)]
    if biggest:
        rows = rows[:limit] if limit else rows
    elif limit and len(rows) > limit:
        step = len(rows) / limit
        rows = [rows[int(i * step)] for i in range(limit)]
    return [p for p, _ in rows], [d for _, d in rows if d]


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-stage Pass-2 profiler.")
    ap.add_argument("target", help="folder prefix; its full-indexed files are the Pass-2 queries")
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--limit", type=int, default=0, help="cap target files (0 = all)")
    ap.add_argument("--biggest", action="store_true",
                    help="profile the LONGEST-duration files (biggest align-time N = duration/grid) "
                         "instead of a library-wide spread -> stress-test the worst case as the library "
                         "trends to 4K/8K and long runtimes. matmul is O(N^2) vs the DP's O(N*band), so "
                         "on long content the matmul can overtake the DP and the split flips.")
    args = ap.parse_args()

    th = load_thresholds()
    store = FingerprintStore(args.db, init_schema=False)

    pref = args.target.rstrip("\\/")
    paths, durs = _select_targets(store, pref, args.limit, args.biggest)
    span = f" | duration {min(durs) / 60:.0f}-{max(durs) / 60:.0f} min" if durs else ""
    print(f"profiling Pass-2 on {len(paths)} full-indexed files under: {pref}{span}")
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
    matcher.resample_to_grid = _timed("resample_to_grid", matcher.resample_to_grid)
    matcher.align_video = _timed("align_video (matmul+DP)", matcher.align_video)
    matcher.align_scenes = _timed("align_scenes", matcher.align_scenes)
    EmbeddingCache.get = _timed("embedding cache.get (.npy I/O)", EmbeddingCache.get)
    FingerprintStore.load = _timed("store.load (metadata)", FingerprintStore.load)
    # Nested inside align_video: lets us split matmul (a@bᵀ) vs the banded DP (Smith-Waterman).
    _video.banded_align = _timed(_DP_KEY, _video.banded_align)

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
        if name == _DP_KEY:                           # nested in align_video -> shown in the split below
            continue
        print(f"{name:32} {secs:9.2f} {100 * secs / max(total, 1e-9):6.1f} "
              f"{calls:9} {1000 * secs / max(calls, 1):9.2f}")

    # --- decision gate (Lever 2): matmul vs DP WITHIN align_video ---
    av = _T.get("align_video (matmul+DP)", [0.0, 0])[0]
    dp = _T.get(_DP_KEY, [0.0, 0])[0]
    matmul = max(0.0, av - dp)                         # align_video ≈ matmul (a@bᵀ) + banded DP
    if av > 0:
        print(f"\nalign_video internals: matmul(a@b.T) ~= {matmul:.2f}s ({100 * matmul / av:.0f}%) | "
              f"banded DP {dp:.2f}s ({100 * dp / av:.0f}%)")
        lever = ("3a: banded matmul (compute only the [N,W] band)" if matmul >= dp
                 else "3b: JIT the banded DP loop (Numba)")
        print(f"  -> dominant cost suggests Lever {lever}")
    print(f"\nper target file: {total / max(len(paths), 1):.2f}s | "
          f"candidates/file: {n_pairs / max(len(paths), 1):.1f}")
    print("note: wall-clock is single-core; the real scan parallelizes the align across cores, "
          "but the per-stage proportions (where the time goes) hold.")
    store.close()


if __name__ == "__main__":
    main()
