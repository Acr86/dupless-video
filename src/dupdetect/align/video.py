"""Video alignment via embeddings — fine re-ranking ViSiL-style, on GPU.

DECISION A (GPU data-path): receives torch CUDA tensors (fp16) from EmbeddingCache.
DECISION A3: DP with Sakoe-Chiba BAND (width = max plausible offset), NOT
full Smith-Waterman over 14k×14k.

The SHAPE of the aligned path distinguishes cases without fragile rules:
  - sharp peak offset 0, full coverage           -> same copy
  - offset jump at start + full coverage         -> BUMPER ADS (offset = garbage)
  - contiguous span but one file longer          -> different EDIT (director's cut)
  - scattered internal gaps                      -> different edit
"""
from __future__ import annotations

import numpy as np

from dupdetect.models import AlignResult


def resample_to_grid(emb, ts, step: float, end: float | None = None):
    """Resamples (emb[N,D], ts[N] in s) to a UNIFORM temporal grid [0, end] with step
    `step`, by nearest-neighbor in TIME. Returns emb'[M,D] (same type as emb: torch
    CUDA or numpy). Key for the mixed-sampling fix: two copies with frames at different
    positions/densities land on the SAME grid -> align_video aligns by time, not
    by frame index. Does not interpolate embeddings (that makes no sense): copies the nearest frame."""
    t = np.asarray(ts, dtype=np.float64)
    n = t.shape[0]
    if n == 0:
        return emb[:0]
    order = np.argsort(t, kind="stable")               # ascending ts (in case they arrive out of order)
    ts_s = t[order]
    hi = float(ts_s[-1]) if end is None else float(end)
    m = int(np.floor(hi / step)) + 1 if hi > 0 else 1       # points 0, step, 2*step, ... <= hi
    grid = np.arange(m, dtype=np.float64) * step
    idx = np.searchsorted(ts_s, grid).clip(0, n - 1)
    left = (idx - 1).clip(0)
    nearest = np.where(np.abs(ts_s[idx] - grid) <= np.abs(ts_s[left] - grid), idx, left)
    sel = order[nearest]                               # indices into the original emb
    if hasattr(emb, "is_cuda"):                        # torch tensor (CUDA or CPU)
        import torch
        return emb[torch.as_tensor(sel, dtype=torch.long, device=emb.device)]
    return emb[sel]


def align_video(emb_a, emb_b, fps: float = 2.0, band_radius: int = 600,
                superset_extra_ratio: float = 0.08, min_ad_run_s: float = 10.0) -> AlignResult:
    """Aligns two per-frame embedding sequences (torch CUDA tensors OR numpy arrays).

    Steps:
      1. Sim = emb_a @ emb_b.T   (cosine; already L2-normalized). Matmul on GPU if given torch
         CUDA tensors, else numpy. Result -> numpy for the DP.
      2. banded_align(Sim, band_radius) -> best contiguous path within the band
      3. derive offset, coverage, score; BIDIRECTIONAL superset (superset_dir)

    The banded DP is a SEQUENTIAL per-row loop, so it runs in numpy on CPU: measured ~4x faster
    than torch-CPU and ~15x vs tiny per-row GPU kernels, with an identical path. This also lets
    the Pass-2 workers stay pure-numpy (no torch import -> faster process spawn).

    band_radius (frames) = max_offset_s * fps (e.g. 300 s * 2 fps = 600). Ad offset is bounded to
    minutes, so capping the band to ±600 frames makes the DP linear instead of quadratic.
    """
    if hasattr(emb_a, "is_cuda"):                  # torch tensor: matmul on its device, then numpy
        if emb_a.numel() == 0 or emb_b.numel() == 0:   # empty tensor: `.size` is a METHOD here, so
            sim = np.empty((0, 0), dtype=np.float32)   # mirror the numpy branch's emptiness guard
        else:
            sim = (emb_a.float() @ emb_b.t().float()).cpu().numpy()
    else:                                          # numpy (Pass-2 worker path: no torch)
        a = np.asarray(emb_a, dtype=np.float32)
        b = np.asarray(emb_b, dtype=np.float32)
        sim = a @ b.T if a.size and b.size else np.empty((0, 0), dtype=np.float32)
    if sim.size == 0:
        return AlignResult(score=0.0)

    path = banded_align(sim, band_radius)          # numpy [L, 2] (i_a, i_b)
    if path is None or len(path) == 0:
        return AlignResult(score=0.0)

    ia, ib = path[:, 0], path[:, 1]
    score = float(sim[ia, ib].mean())
    offset = float((int(ib[0]) - int(ia[0])) / fps)
    na, nb = sim.shape
    coverage = len(path) / min(na, nb)

    contiguous, sdir, extra_ratio = _detect_superset(path, na, nb)
    is_edition = contiguous and extra_ratio >= superset_extra_ratio

    # Inserted-commercials signal (only when NOT a contiguous edition): interleaved extra blocks.
    inter_ratio, ad_dir = (0.0, 0)
    if not is_edition:
        inter_ratio, ad_dir = _interleaved_extra(path, na, nb, min_run=int(min_ad_run_s * fps))

    return AlignResult(
        score=score, offset=offset, coverage=coverage,
        contiguous_superset=is_edition, superset_dir=sdir if is_edition else 0,
        extra_ratio=extra_ratio, interleaved_ratio=inter_ratio, ad_dir=ad_dir,
    )


def banded_align(sim, band_radius: int, gap_penalty: float = 0.3,
                 match_threshold: float = 0.5):
    """Local alignment (Smith-Waterman) with Sakoe-Chiba band over the similarity matrix.
    Returns the best path as a numpy [L, 2] array of indices (i_a, i_b) for MATCHED frames
    (diagonal steps), or None if no alignment found. Accepts numpy or (CPU) torch `sim`.

    Band: only |j - i| <= band_radius (the diagonal and ±band_radius). Ad offset is bounded to
    minutes => small band => O(Na * band) instead of O(Na*Nb).

    SW recurrence (floor 0, linear gaps) in coords (i, d=j-i):
        H0[d] = max(0, H[i-1,d] + s,  H[i-1,d+1] - gap)        # fresh / diag / up
        H[d]  = max(H0[d], H[d-1] - gap)                        # gap in 'a' (left)
    where s = sim[i,j] - match_threshold (reward when similarity exceeds threshold).
    Linear gap lets 'left' be resolved vectorized via cumulative-max; the loop runs only over
    rows (Na), everything else is vectorized over the band width. float64 throughout: the
    +gp*pos / -gp*pos round-trip is exact enough that the strict '>' (with epsilon) doesn't mark
    spurious LEFT on the main diagonal.
    """
    sim = np.asarray(sim, dtype=np.float64)
    na, nb = int(sim.shape[0]), int(sim.shape[1])
    if na == 0 or nb == 0:
        return None
    r = int(band_radius)
    W = 2 * r + 1
    pos = np.arange(W, dtype=np.float64)                        # position within the band
    gp = float(gap_penalty)
    d_off = np.arange(-r, r + 1)

    bp = np.zeros((na, W), dtype=np.int8)                       # 0 stop,1 diag,2 up,3 left
    h_prev = np.zeros(W, dtype=np.float64)                      # row -1: SW boundary = 0
    rowmax = np.full(na, -1e9, dtype=np.float64)
    rowarg = np.zeros(na, dtype=np.int64)

    for i in range(na):
        j = i + d_off
        valid = (j >= 0) & (j < nb)
        s = sim[i, np.clip(j, 0, nb - 1)] - match_threshold    # [W] reward
        diag = h_prev + s                                      # (i-1, d)
        up = np.empty(W, dtype=np.float64)
        up[:-1] = h_prev[1:] - gp                              # (i-1, d+1)
        up[-1] = -gp
        zero = np.zeros(W, dtype=np.float64)
        stacked = np.stack([zero, diag, up], axis=0)           # [3, W]
        src = stacked.argmax(axis=0).astype(np.int8)           # src 0/1/2
        h0 = np.where(valid, stacked.max(axis=0), 0.0)         # out of range -> boundary
        cmax = np.maximum.accumulate(h0 + gp * pos)            # 'left' gap propagation
        h_left = cmax - gp * pos
        h = np.maximum(h0, h_left)
        bp_row = np.where(h_left > h0 + 1e-6, np.int8(3), src)
        bp[i] = np.where(valid, bp_row, np.int8(0))
        h = np.where(valid, h, 0.0)
        am = int(h.argmax())
        rowmax[i] = h[am]
        rowarg[i] = am
        h_prev = h

    gi = int(rowmax.argmax())
    if rowmax[gi] <= 0.0:
        return None                                            # nothing exceeds the threshold

    # --- traceback (O(path length), not O(cells)) ---
    pairs: list[tuple[int, int]] = []
    i, d = gi, int(rowarg[gi])
    while i >= 0 and 0 <= d < W:
        move = int(bp[i, d])
        if move == 0:                                          # stop: local start
            break
        if move == 1:                                          # diag = match
            pairs.append((i, i + d - r))
            i -= 1
        elif move == 2:                                        # up = gap in 'b'
            i -= 1
            d += 1
        else:                                                  # left = gap in 'a'
            d -= 1
    if not pairs:
        return None
    pairs.reverse()
    return np.array(pairs, dtype=np.int64)


def _interleaved_extra(path, na: int, nb: int, min_run: int) -> tuple[float, int]:
    """Inserted-commercials signal: fraction of the LONGER sequence that is UNMATCHED in CONTIGUOUS
    runs (>= `min_run` frames) INSIDE the aligned span. This is the shape of mid-roll ads (blocks of
    foreign content spliced between matching segments). Distinct from:
      - a contiguous EDITION (extra at the ends -> `_detect_superset`, handled before this is called),
      - different-encode JITTER (scattered single misses -> filtered by `min_run`),
      - a clip that is a sub-segment of a longer video (extra is OUTSIDE the matched span, not interior).
    Measured (2026-06-12): real dups/editions/different = 0.0; injected mid-roll ads = the ad fraction.
    Returns (interleaved_ratio, ad_dir): ad_dir=+1 if 'b' is the longer (ad) copy, -1 if 'a', 0 if none.
    """
    import numpy as np

    p = path.detach().cpu().numpy() if hasattr(path, "detach") else np.asarray(path)
    if p.shape[0] == 0 or min_run <= 0:
        return (0.0, 0)
    longer_is_b = nb >= na
    idx = p[:, 1] if longer_is_b else p[:, 0]
    n_long = nb if longer_is_b else na
    matched = np.zeros(n_long, dtype=bool)
    matched[np.unique(idx)] = True
    lo, hi = int(idx.min()), int(idx.max())             # the aligned span in the longer sequence
    gap = ~matched[lo:hi + 1]
    # sum the lengths of contiguous unmatched runs that are long enough to be an ad break
    block_frames = 0
    run = 0
    for g in gap:
        if g:
            run += 1
        else:
            if run >= min_run:
                block_frames += run
            run = 0
    if run >= min_run:
        block_frames += run
    if block_frames == 0:
        return (0.0, 0)
    return (block_frames / n_long, 1 if longer_is_b else -1)


def _detect_superset(path, na: int, nb: int) -> tuple[bool, int, float]:
    """Is one file contained as a CONTIGUOUS span within the other? BIDIRECTIONAL.
    Returns (is_contained_pattern, dir, extra_ratio): dir=+1 if a⊂b, -1 if b⊂a, 0 if not.

    Requires: (1) the path is dense (no large internal gaps) and (2) the shorter file
    is almost entirely covered within the longer one, which has contiguous extra runtime.
    The final threshold (bumper ads vs. real edit) is applied by align_video on
    `extra_ratio` -> 45s of ads won't reach it; 15 min director's cut will.
    """
    import numpy as np

    p = path.detach().cpu().numpy() if hasattr(path, "detach") else np.asarray(path)
    if p.shape[0] == 0:
        return (False, 0, 0.0)
    ia, ib = p[:, 0], p[:, 1]
    span_a = int(ia.max() - ia.min()) + 1
    span_b = int(ib.max() - ib.min()) + 1
    length = p.shape[0]

    # dense = the matched span has almost no internal gaps in either sequence
    if length < 0.9 * max(span_a, span_b):
        return (False, 0, 0.0)

    frac_a, frac_b = span_a / na, span_b / nb
    extra_a, extra_b = (na - span_a) / na, (nb - span_b) / nb
    if frac_a >= 0.9 and extra_b > extra_a:        # 'a' fully contained within longer 'b'
        return (True, +1, float(extra_b))
    if frac_b >= 0.9 and extra_a > extra_b:        # 'b' fully contained within longer 'a'
        return (True, -1, float(extra_a))
    return (False, 0, 0.0)
