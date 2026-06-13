"""Scene-cut signature alignment via DTW over intervals.
Most stubborn signal: survives cam rips, recompression, trims. Fallback in T3/T4."""
from __future__ import annotations

import numpy as np

from dupdetect.models import AlignResult


def align_scenes(cuts_a: np.ndarray, cuts_b: np.ndarray, theta: float = 0.70,
                 band_frac: float = 0.2, gap_penalty: float = 0.2,
                 min_band: int = 25) -> AlignResult:
    """Aligns INTERVALS between cuts (np.diff), not absolute timestamps
    (which shift when there are trims at the start).

    DTW with THREE constraints that make the score DISCRIMINATE (same ~1.0, different
    low) rather than unconstrained warping making everything look similar:
      - Sakoe-Chiba band: |i-j| <= band -> warping is not arbitrary.
      - gap penalty: each insertion/deletion costs `gap_penalty` -> very different
        editing rhythms (many gaps) score low.
      - length cutoff: if cut counts differ > band, they are incompatible
        (cannot reach (n,m) within the band) -> score 0.
    Local cost = relative duration difference, in [0,1]. theta is not used here
    (the tree compares the score against theta_s); kept for compatibility.
    band_frac/gap_penalty are TUNABLE (step 10) depending on cam-rip tolerance.
    """
    ia = np.diff(np.asarray(cuts_a, dtype=np.float64))
    ib = np.diff(np.asarray(cuts_b, dtype=np.float64))
    n, m = len(ia), len(ib)
    if n == 0 or m == 0:
        return AlignResult(0.0)

    coverage = min(n, m) / max(n, m)
    band = max(min_band, int(np.ceil(band_frac * max(n, m))))
    if abs(n - m) > band:                          # incompatible lengths -> no match
        return AlignResult(0.0, coverage=coverage)

    eps = 1e-6
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = max(0.0, ia[i - 1])                                    # non-negative intervals
        for j in range(max(1, i - band), min(m, i + band) + 1):     # within band only
            bj = max(0.0, ib[j - 1])
            cost = min(1.0, abs(ai - bj) / (ai + bj + eps))         # relative diff, bounded [0,1]
            d[i, j] = cost + min(d[i - 1, j - 1],                   # diagonal (no gap)
                                 d[i - 1, j] + gap_penalty,         # deletion (gap)
                                 d[i, j - 1] + gap_penalty)         # insertion (gap)

    final = d[n, m]
    if not np.isfinite(final):                     # (n,m) outside band -> no path
        return AlignResult(0.0, coverage=coverage)
    # normalize by max(n,m): gaps add to cost without inflating the divisor -> they penalize
    score = max(0.0, min(1.0, 1.0 - final / max(n, m)))            # bounded [0,1] (robustness)
    return AlignResult(score=float(score), coverage=coverage)
