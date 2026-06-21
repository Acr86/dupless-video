# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated hot loops for Pass-2 alignment (perf §1; numerically faithful -> verdict
unchanged, §0). Pure typed memoryviews (no numpy C-API) so the build needs only a C compiler.

  banded_align_fast  -> the Smith-Waterman banded DP (align/video.py banded_align), the dominant
                        per-pair cost once the matmul is banded (measured ~78% of a giant pair).
  scenes_dtw_final   -> the DTW fill of align/scenes.py (the second-biggest Pass-2 cost, ~34% of the
                        real-data profile), returning d[n, m].

Both are SCALAR ports of the exact same recurrences; align/video.py and align/scenes.py keep the
pure-Python versions as the reference + fallback when this module isn't compiled.
"""
from libc.math cimport INFINITY


def banded_align_fast(double[:, ::1] sim, Py_ssize_t r,
                      double gap_penalty=0.3, double match_threshold=0.5):
    """Scalar port of align/video.banded_align. `sim` must be C-contiguous float64. Returns the
    matched path as an int64 [L, 2] numpy array (i_a, i_b), or None. Faithful to the numpy version:
    same float64 recurrence, same first-max-on-tie argmax, same 1e-6 LEFT-gap epsilon."""
    import numpy as np
    cdef Py_ssize_t na = sim.shape[0]
    cdef Py_ssize_t nb = sim.shape[1]
    if na == 0 or nb == 0:
        return None
    cdef Py_ssize_t W = 2 * r + 1
    cdef double gp = gap_penalty
    cdef double mt = match_threshold

    cdef signed char[:, ::1] bp = np.zeros((na, W), dtype=np.int8)   # 0 stop,1 diag,2 up,3 left
    cdef double[::1] h_prev = np.zeros(W, dtype=np.float64)          # row -1 boundary = 0
    cdef double[::1] h0 = np.zeros(W, dtype=np.float64)
    cdef signed char[::1] src = np.zeros(W, dtype=np.int8)
    cdef double[::1] h = np.zeros(W, dtype=np.float64)
    cdef double[::1] rowmax = np.empty(na, dtype=np.float64)
    cdef long long[::1] rowarg = np.zeros(na, dtype=np.int64)

    cdef Py_ssize_t i, k, j
    cdef double diag, up, m, s, hl, h_left_prev, rowmax_i
    cdef int isrc, valid
    cdef Py_ssize_t rowarg_i

    for i in range(na):
        # pass 1: h0[k] = max(0, diag, up) (valid cols), src = argmax over (0, diag, up) first-wins.
        for k in range(W):
            j = i + (k - r)
            valid = (j >= 0) and (j < nb)
            if j < 0:
                s = sim[i, 0] - mt
            elif j >= nb:
                s = sim[i, nb - 1] - mt
            else:
                s = sim[i, j] - mt
            diag = h_prev[k] + s
            up = h_prev[k + 1] - gp if k < W - 1 else -gp
            m = 0.0
            isrc = 0
            if diag > m:
                m = diag
                isrc = 1
            if up > m:
                m = up
                isrc = 2
            if valid:
                h0[k] = m
                src[k] = isrc
            else:
                h0[k] = 0.0
                src[k] = 0
        # pass 2: LEFT-gap propagation h_left[k] = max(h0[k], h_left[k-1]-gp) over ALL k; then bp + h.
        h_left_prev = -INFINITY
        rowmax_i = -INFINITY
        rowarg_i = 0
        for k in range(W):
            j = i + (k - r)
            valid = (j >= 0) and (j < nb)
            hl = h0[k]
            if h_left_prev - gp > hl:
                hl = h_left_prev - gp
            h_left_prev = hl
            if valid:
                bp[i, k] = 3 if hl > h0[k] + 1e-6 else src[k]
                h[k] = hl
            else:
                bp[i, k] = 0
                h[k] = 0.0
            if h[k] > rowmax_i:
                rowmax_i = h[k]
                rowarg_i = k
        rowmax[i] = rowmax_i
        rowarg[i] = rowarg_i
        for k in range(W):
            h_prev[k] = h[k]

    cdef double gi_val = -1.0e18
    cdef Py_ssize_t gi = 0
    for i in range(na):
        if rowmax[i] > gi_val:
            gi_val = rowmax[i]
            gi = i
    if gi_val <= 0.0:
        return None

    # traceback (O(path length))
    cdef list pairs = []
    cdef Py_ssize_t d = rowarg[gi]
    cdef int move
    i = gi
    while i >= 0 and 0 <= d < W:
        move = bp[i, d]
        if move == 0:
            break
        if move == 1:
            pairs.append((i, i + d - r))
            i -= 1
        elif move == 2:
            i -= 1
            d += 1
        else:
            d -= 1
    if len(pairs) == 0:
        return None
    pairs.reverse()
    return np.array(pairs, dtype=np.int64)


def scenes_dtw_final(double[::1] ia, double[::1] ib, Py_ssize_t band, double gap_penalty):
    """Scalar port of the align/scenes.align_scenes DTW fill. Returns d[n, m] (the final cost; may be
    INFINITY if (n,m) is outside the band). `ia`/`ib` are the inter-cut intervals (np.diff), float64.
    Same float64 recurrence and relative cost as the numpy version -> identical score (§0)."""
    import numpy as np
    cdef Py_ssize_t n = ia.shape[0]
    cdef Py_ssize_t m = ib.shape[0]
    cdef double[:, ::1] d = np.full((n + 1, m + 1), INFINITY, dtype=np.float64)
    d[0, 0] = 0.0
    cdef Py_ssize_t i, j, jlo, jhi
    cdef double ai, bj, cost, best
    cdef double eps = 1e-6
    cdef double gp = gap_penalty
    for i in range(1, n + 1):
        ai = ia[i - 1]
        if ai < 0.0:
            ai = 0.0
        jlo = i - band
        if jlo < 1:
            jlo = 1
        jhi = i + band
        if jhi > m:
            jhi = m
        for j in range(jlo, jhi + 1):
            bj = ib[j - 1]
            if bj < 0.0:
                bj = 0.0
            cost = ai - bj
            if cost < 0.0:
                cost = -cost
            cost = cost / (ai + bj + eps)
            if cost > 1.0:
                cost = 1.0
            best = d[i - 1, j - 1]
            if d[i - 1, j] + gp < best:
                best = d[i - 1, j] + gp
            if d[i, j - 1] + gp < best:
                best = d[i, j - 1] + gp
            d[i, j] = cost + best
    return d[n, m]
