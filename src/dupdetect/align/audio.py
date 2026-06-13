"""Audio-fp alignment via cross-correlation (offset search).
Handles trims and bumper ads; fails on different-language audio (covered by
video in T2) and on cam rips (room noise)."""
from __future__ import annotations

import numpy as np

from dupdetect.features.audio_fp import ITEM_RATE_HZ
from dupdetect.models import AlignResult

# 16-bit popcount table -> vectorized 32-bit popcount with no loops.
_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def _popcount32(x: np.ndarray) -> np.ndarray:
    """Set-bit count of each uint32 (vectorized via two 16-bit lookups)."""
    return (_POP16[x & 0xFFFF].astype(np.int32) + _POP16[(x >> 16) & 0xFFFF])


def align_audio(fp_a: np.ndarray, fp_b: np.ndarray, min_overlap_s: float = 60.0,
                item_rate: float = ITEM_RATE_HZ, max_offset_s: float = 300.0) -> AlignResult:
    """Offset that maximizes bitwise agreement between two Chromaprint fingerprints
    (AcoustID-style for partial match).

    For each candidate offset `off` (b relative to a: a[i] ~ b[i+off]) computes the mean
    similarity 1 - hamming/32 over the overlap, requiring `min_overlap_s`. An offset != 0 with
    high coverage => intro/bumper ads. Audio in a different language does not correlate (~0.5,
    random bit agreement) => covered by video in T2.

    Hot path (~92% of Pass-2 was here): the hamming sum per offset is computed for ALL offsets at
    once via FFT cross-correlation, O(N log N) instead of the O(offsets·N) per-offset scan. The
    result is BIT-EXACT to the brute-force scan (integer correlations recovered by rounding the
    FFT output; FP error << 0.5 for these lengths) -> identical (offset, score, coverage) ->
    verdict invariance (§0). See tests/test_audio.py for the equivalence test vs the reference.
    """
    a = np.asarray(fp_a, dtype=np.uint32)
    b = np.asarray(fp_b, dtype=np.uint32)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return AlignResult(0.0)

    min_overlap = max(1, int(min_overlap_s * item_rate))
    max_off = int(max_offset_s * item_rate)
    lo_off, hi_off = -min(max_off, na - 1), min(max_off, nb - 1)
    if hi_off < lo_off:
        return AlignResult(0.0)

    offs = np.arange(lo_off, hi_off + 1)
    lo = np.maximum(0, -offs)                              # overlap start in `a`, per offset
    hi = np.minimum(na, nb - offs)                         # overlap end in `a`, per offset
    length = hi - lo
    if not np.any(length >= min_overlap):
        return AlignResult(0.0)

    # hamming_sum(off) = Σ popcount(a[i] ^ b[i+off]) over the overlap. Using x^y = x+y-2·x·y per
    # bit: = (Σ popcount a) + (Σ popcount b) - 2·Σ_k corr_k(off), where corr_k is the cross-
    # correlation of bit-plane k. The linear terms come from popcount prefix sums; Σ_k corr_k for
    # every offset comes from one batched FFT over the 32 planes.
    pca = np.concatenate(([0], np.cumsum(_popcount32(a), dtype=np.int64)))   # PA[i]=Σ popcount(a[:i])
    pcb = np.concatenate(([0], np.cumsum(_popcount32(b), dtype=np.int64)))
    term_a = pca[hi] - pca[lo]                             # Σ popcount(a[lo:hi])
    term_b = pcb[hi + offs] - pcb[lo + offs]               # Σ popcount(b[lo+off:hi+off])

    bit = np.arange(32, dtype=np.uint32)
    planes_a = ((a[None, :] >> bit[:, None]) & 1).astype(np.float64)         # [32, na]
    planes_b = ((b[None, :] >> bit[:, None]) & 1).astype(np.float64)         # [32, nb]
    L = na + nb - 1
    nfft = 1 << (L - 1).bit_length() if L > 1 else 1       # >= na+nb-1 -> no circular aliasing
    corr = np.fft.irfft(np.conj(np.fft.rfft(planes_a, n=nfft, axis=1))
                        * np.fft.rfft(planes_b, n=nfft, axis=1), n=nfft, axis=1)
    sum_corr = corr.sum(axis=0)                            # Σ_k corr_k, indexed by circular lag
    idx = np.where(offs >= 0, offs, offs + nfft)           # off<0 wraps to the tail
    sum_corr = np.rint(sum_corr[idx]).astype(np.int64)     # exact integer correlations
    bits = term_a + term_b - 2 * sum_corr                  # hamming sum per offset (integer)

    sim = np.where(length >= min_overlap, 1.0 - bits / (32.0 * length), -1.0)
    best = int(np.argmax(sim))                             # first max -> matches brute-force '>' first-wins
    if sim[best] <= 0.0:                                   # brute-force needs sim>0 to record a match
        return AlignResult(0.0)
    coverage = int(length[best]) / min(na, nb)
    return AlignResult(score=float(sim[best]), offset=int(offs[best]) / item_rate,
                       coverage=float(coverage))
