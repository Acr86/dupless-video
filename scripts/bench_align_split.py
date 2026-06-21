"""Micro-benchmark: how does the Pass-2 video-align cost SPLIT between the similarity matmul
(`a @ b.T`, O(N^2*D)) and the banded DP (`banded_align`, O(N*band)) as N grows?

N = align-time frames = duration / grid_step (resolution-INDEPENDENT). So this answers: as the library
trends to long 4K/8K runtimes (big N), does the O(N^2) matmul overtake the O(N*band) DP — i.e. is a
BANDED matmul (compute only the [N, 2*band+1] strip) worth it, on top of JIT-ing the DP?

Synthetic (random L2-normalized embeddings) ON PURPOSE: isolates the ALGORITHMIC scaling from disk /
audio / retrieval confounds (the polluted run was 87 min of HDD wait, ~4% CPU). The DP fill is
O(N*band) and data-independent; the matmul cost is N^2*D regardless of values -> timings are
representative of the real per-pair compute.

Run with BLAS pinned to 1 thread (each parallel worker pays the single-thread matmul after Lever 1):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/Scripts/python.exe scripts/bench_align_split.py
"""
import time

import numpy as np

from dupdetect.align.video import _banded_matmul, banded_align

GRID_S = 2.0          # resample_to_grid step (thresholds.yaml sampling.grid_step_s)
BAND = 600            # band_radius frames = max_offset_s(300) * fps_sample(2)
D = 768               # DINOv2 ViT-B/14 embedding dim
REPEAT = 3            # take the best of N (warm cache, less noise)

# N sweep ~ 16 min .. 4.4 h of runtime at the 2 s align grid.
_NS = (500, 1000, 2000, 3000, 5000, 8000)


def _rand_unit(n: int, seed: int) -> np.ndarray:
    a = np.random.default_rng(seed).standard_normal((n, D)).astype(np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _best(fn):
    best, out = float("inf"), None
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    import dupdetect.align.video as _v
    has_fast = _v._banded_align_fast is not None
    print(f"band_radius={BAND} frames, D={D}, BLAS=1 thread, cython_DP={has_fast}\n")
    print(f"{'N':>6} {'runtime':>8} {'mm_full':>8} {'mm_band':>8} {'DP_pure':>8} {'DP_fast':>8} "
          f"{'before':>8} {'after':>8} {'speedup':>8}")
    for n in _NS:
        a, b = _rand_unit(n, 0), _rand_unit(n, 1)             # distinct seeds -> not a trivial diagonal
        t_mm, sim = _best(lambda a=a, b=b: a @ b.T)            # FULL matmul (today's worker path)
        t_bmm, _ = _best(lambda a=a, b=b: _banded_matmul(a, b, BAND))   # 3a: band only (giants)
        t_dp_fast, _ = _best(lambda sim=sim: banded_align(sim, BAND))   # 3b: Cython DP (if compiled)
        save = _v._banded_align_fast
        _v._banded_align_fast = None                          # force the pure-Python DP for baseline
        t_dp_pure, _ = _best(lambda sim=sim: banded_align(sim, BAND))
        _v._banded_align_fast = save
        before = t_mm + t_dp_pure                             # baseline: full matmul + pure-Python DP
        after = t_bmm + t_dp_fast                             # all levers: banded matmul + Cython DP
        print(f"{n:6d} {n * GRID_S / 60:6.0f}m {1000 * t_mm:7.1f} {1000 * t_bmm:7.1f} "
              f"{1000 * t_dp_pure:7.1f} {1000 * t_dp_fast:7.1f} {1000 * before:7.0f} {1000 * after:7.0f} "
              f"{before / after if after else 1.0:6.1f}x")
    print("\nbefore = full matmul + pure-Python DP (baseline); after = banded matmul (3a) + Cython DP (3b)."
          "\nDP_pure carries the numpy per-row overhead the Cython port removes. scenes DTW (also "
          "Cythonized) is a separate ~34% of real Pass-2, not in this video-only microbench.")


if __name__ == "__main__":
    main()
