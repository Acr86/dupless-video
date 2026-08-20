"""Tests for video alignment (step 4) — the sensitive piece.

Uses synthetic embeddings (vocabulary of distinct unit vectors) to
control the similarity matrix and verify that the PATH SHAPE distinguishes:
  - offset 0, full coverage        -> same copy
  - offset at start (ads)          -> same film, offset != 0, NOT an edit
  - contiguous segment + extra     -> different edit (superset), bidirectional
  - nothing aligns                 -> score 0
Plus direct tests for banded_align and _detect_superset.
"""
from __future__ import annotations

import numpy as np
import pytest

from dupdetect.align.video import _detect_superset, align_video, banded_align, resample_to_grid

torch = pytest.importorskip("torch")


def _vocab(n: int, d: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, d)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m


VOCAB = _vocab(2000)


def _emb(ids: list[int]):
    """Embeddings [len(ids), D] taking vocab vectors by id (same id => cos=1)."""
    return torch.from_numpy(VOCAB[ids])


# --------------------------------------------------------------- align_video

def test_same_copy_zero_offset():
    a = _emb(list(range(100)))
    b = _emb(list(range(100)))
    r = align_video(a, b, fps=2.0, band_radius=30)
    assert r.score > 0.9
    assert r.offset == pytest.approx(0.0, abs=0.6)
    assert r.coverage > 0.95
    assert r.contiguous_superset is False


def test_ads_prepended_offset_not_an_edit():
    # b = 10 distinct ad frames + the same film as 'a' (200 frames)
    a = _emb(list(range(200)))
    b = _emb(list(range(1000, 1010)) + list(range(200)))     # offset +10 frames = +5s
    r = align_video(a, b, fps=2.0, band_radius=30)
    assert r.score > 0.9
    assert r.offset == pytest.approx(5.0, abs=0.6)           # 10 frames / 2 fps
    assert r.coverage > 0.95
    # 10/210 ≈ 4.8% extra < 8% threshold -> NOT flagged as an edit
    assert r.contiguous_superset is False


def test_contiguous_superset_edit_bidirectional():
    # b = same film as 'a' (200) + 40 extra frames at the end (16.7% > 8%)
    a = _emb(list(range(200)))
    b = _emb(list(range(200)) + list(range(1000, 1040)))
    r = align_video(a, b, fps=2.0, band_radius=30)
    assert r.score > 0.9
    assert r.contiguous_superset is True
    assert r.superset_dir == +1                              # a ⊂ b
    assert r.extra_ratio == pytest.approx(40 / 240, abs=0.03)

    # reverse direction -> dir = -1
    r2 = align_video(b, a, fps=2.0, band_radius=30)
    assert r2.contiguous_superset is True
    assert r2.superset_dir == -1                             # b ⊂ a (b is the longer one)


def test_different_films_do_not_align():
    a = _emb(list(range(100)))
    b = _emb(list(range(500, 600)))                          # disjoint ids
    r = align_video(a, b, fps=2.0, band_radius=30)
    # the real discriminator is COVERAGE: at most 1-2 spurious frames align
    assert r.coverage < 0.2


# --------------------------------------------------------------- interleaved ads (commercials)

def test_interleaved_ads_flagged_and_directional():
    """Mid-roll commercials = foreign blocks spliced INSIDE the aligned span. The copy stays a
    strong match (high coverage) but interleaved_ratio > 0 and ad_dir points at the LONGER copy.
    Measured behavior (see _scan_test): real dups/editions = 0.0, only ads register."""
    film = list(range(200))
    ad1, ad2 = list(range(1000, 1030)), list(range(1030, 1060))   # 30-frame foreign blocks (>=10s@2fps)
    a = _emb(film)
    b = _emb(film[:60] + ad1 + film[60:130] + ad2 + film[130:])    # 200 film + 60 ad = 260
    r = align_video(a, b, fps=2.0, band_radius=120, min_ad_run_s=10.0)
    assert r.coverage > 0.95                                       # still a strong match (stays a dup)
    assert r.contiguous_superset is False                         # NOT a director's cut
    assert r.interleaved_ratio > 0.15                            # ~60/260
    assert r.ad_dir == +1                                         # b is the ad copy (longer)
    # reverse the order -> the ad copy is now 'a' -> ad_dir flips sign
    r2 = align_video(b, a, fps=2.0, band_radius=120, min_ad_run_s=10.0)
    assert r2.ad_dir == -1 and r2.interleaved_ratio > 0.15


def test_clean_dup_has_no_interleaved_ads():
    a = _emb(list(range(200)))
    b = _emb(list(range(200)))
    r = align_video(a, b, fps=2.0, band_radius=30, min_ad_run_s=10.0)
    assert r.interleaved_ratio == 0.0 and r.ad_dir == 0


def test_director_cut_not_flagged_as_ads():
    # contiguous +20% tail -> a real EDITION (contiguous_superset), NOT interleaved ads
    a = _emb(list(range(200)))
    b = _emb(list(range(200)) + list(range(1000, 1040)))
    r = align_video(a, b, fps=2.0, band_radius=60, min_ad_run_s=10.0)
    assert r.contiguous_superset is True
    assert r.interleaved_ratio == 0.0                            # extra is at the end, not interleaved


def test_scattered_jitter_not_flagged_as_ads():
    # different-encode noise: scattered SINGLE unmatched frames (runs < min_run) -> not ads
    film = list(range(200))
    noisy = [(1500 + i) if (i % 12 == 0 and 0 < i < 199) else i for i in film]   # ~16 isolated misses
    r = align_video(_emb(film), _emb(noisy), fps=2.0, band_radius=30, min_ad_run_s=10.0)
    assert r.interleaved_ratio == 0.0                            # isolated misses filtered by min_run


# --------------------------------------------------------------- banded_align directo

def test_banded_align_diagonal_path():
    # high similarity on the diagonal -> contiguous path i==j
    sim = torch.full((40, 40), 0.1)
    sim.fill_diagonal_(0.95)
    path = banded_align(sim, band_radius=5)
    assert path is not None
    ia, ib = path[:, 0].tolist(), path[:, 1].tolist()
    assert ia == ib                                          # pure diagonal
    assert len(path) == 40                                   # full diagonal


def test_banded_align_no_alignment_returns_none():
    sim = torch.full((30, 30), 0.1)                          # all below threshold 0.5
    assert banded_align(sim, band_radius=5) is None


# --------------------------------------------------------------- banded matmul (3a: giants) — §0
def test_banded_matmul_band_equals_full():
    """The banded matmul fills exactly the diagonal band the DP reads, bit-faithful to the full a@b.T
    on those cells (off-band cells are never read). Multi-block (na > block=256) so the slab logic runs."""
    from dupdetect.align.video import _banded_matmul
    a, b = _vocab(1000, seed=1), _vocab(1000, seed=2)
    r, nb = 100, b.shape[0]
    full = a @ b.T
    bm = _banded_matmul(a, b, r)
    for i in (0, 1, r, 256, 257, 500, 999):                  # incl. block boundaries (block=256)
        for d in range(-r, r + 1):
            j = i + d
            if 0 <= j < nb:
                assert bm[i, j] == pytest.approx(full[i, j], abs=1e-4)


def test_banded_align_fast_equals_pure(monkeypatch):
    """The compiled banded DP (Cython _fastdp) returns the IDENTICAL path to the pure-Python
    reference across many random shapes/bands -> verdict unchanged (§0). Skipped when not built."""
    import dupdetect.align.video as v
    if v._banded_align_fast is None:
        pytest.skip("Cython _fastdp not compiled")
    fast_fn = v._banded_align_fast
    for seed in range(12):
        rng = np.random.default_rng(seed)
        na, nb = int(rng.integers(20, 400)), int(rng.integers(20, 400))
        sim = rng.standard_normal((na, nb)).astype(np.float64)
        if seed % 2 == 0:                                    # half: inject a partial diagonal -> a path
            for i in range(min(na, nb)):
                if rng.random() < 0.7:
                    sim[i, min(i + int(rng.integers(-2, 3)), nb - 1)] += 2.0
        r = int(rng.integers(5, 80))
        monkeypatch.setattr(v, "_banded_align_fast", fast_fn)
        fast = banded_align(sim, band_radius=r)
        monkeypatch.setattr(v, "_banded_align_fast", None)   # force the pure-Python body
        pure = banded_align(sim, band_radius=r)
        assert (fast is None) == (pure is None)
        if fast is not None:
            assert np.array_equal(fast, pure)


def test_align_video_banded_matmul_matches_full(monkeypatch):
    """align_video's banded-matmul path (giants, numpy) yields the SAME AlignResult as the full
    matmul -> verdict invariant (§0). Force each path by moving the gate; numpy inputs so the banded
    branch runs (torch inputs go to the GPU full-matmul path). Scenario exercises offset + superset."""
    import dupdetect.align.video as v
    film = list(range(600))
    a = VOCAB[np.array(film)]                                # numpy -> banded branch
    b = VOCAB[np.array(list(range(1500, 1510)) + film + list(range(1600, 1660)))]  # +10 offset, +60 tail
    monkeypatch.setattr(v, "_BANDED_MATMUL_MIN_N", 10 ** 9)  # force FULL matmul
    full = align_video(a, b, fps=2.0, band_radius=60)
    monkeypatch.setattr(v, "_BANDED_MATMUL_MIN_N", 0)        # force BANDED matmul
    banded = align_video(a, b, fps=2.0, band_radius=60)
    assert banded.score == pytest.approx(full.score, abs=1e-6)
    assert banded.offset == pytest.approx(full.offset, abs=1e-6)
    assert banded.coverage == pytest.approx(full.coverage, abs=1e-6)
    assert banded.contiguous_superset == full.contiguous_superset
    assert banded.superset_dir == full.superset_dir
    assert banded.extra_ratio == pytest.approx(full.extra_ratio, abs=1e-6)
    assert banded.interleaved_ratio == pytest.approx(full.interleaved_ratio, abs=1e-6)
    assert banded.ad_dir == full.ad_dir


# --------------------------------------------------------------- _detect_superset directo

def test_detect_superset_dup_is_not_superset():
    path = np.array([[i, i] for i in range(100)])
    contained, d, extra = _detect_superset(path, na=100, nb=100)
    assert contained is False and d == 0


def test_detect_superset_a_inside_b():
    # 'a' (100) matched to segment [0,100) of 'b' (130) -> a ⊂ b
    path = np.array([[i, i] for i in range(100)])
    contained, d, extra = _detect_superset(path, na=100, nb=130)
    assert contained is True and d == +1
    assert extra == pytest.approx(30 / 130, abs=1e-3)


def test_detect_superset_internal_gaps_not_counted():
    # path with a large internal gap -> not 'dense' -> not a superset
    left = [[i, i] for i in range(40)]
    right = [[i, i] for i in range(80, 100)]
    path = np.array(left + right)
    contained, d, extra = _detect_superset(path, na=100, nb=200)
    assert contained is False


# --------------------------------------------------------------- resample_to_grid

def test_resample_to_grid_uniform_nearest_neighbor():
    emb = np.arange(5, dtype=np.float32).reshape(5, 1)        # each frame tagged with its index
    ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = resample_to_grid(emb, ts, step=2.0)                # grid 0,2,4 -> frames 0,2,4
    assert out.shape[0] == 3
    assert list(out[:, 0]) == [0.0, 2.0, 4.0]


def test_resample_to_grid_sorts_unordered_ts():
    emb = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    ts = np.array([4.0, 0.0, 2.0], dtype=np.float32)         # unordered
    out = resample_to_grid(emb, ts, step=2.0)               # grid 0,2,4 -> ts 0,2,4 -> 20,30,10
    assert list(out[:, 0]) == [20.0, 30.0, 10.0]


def test_resample_two_sampling_rates_land_on_same_grid():
    """KEY property of the fix: two copies of the SAME content sampled differently (dense vs
    sparse) resample to the SAME result on the shared grid -> time-based align is robust."""
    base = _vocab(11, seed=7)                                # 11 'scenes' at t=0..10
    t_full = np.arange(11, dtype=np.float32)                 # dense: 1 frame/s
    dense = resample_to_grid(base, t_full, step=2.0)
    sparse_idx = [0, 2, 4, 6, 8, 10]                         # sparse: 1 frame/2s (seek)
    sparse = resample_to_grid(base[sparse_idx], t_full[sparse_idx], step=2.0)
    assert dense.shape == sparse.shape
    assert np.allclose(dense, sparse)                        # identical on the shared grid


def test_resample_empty_does_not_crash():
    assert resample_to_grid(np.empty((0, 4), np.float32), np.empty(0, np.float32), 2.0).shape[0] == 0


def test_resample_torch_tensor_preserves_type():
    e = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    ts = np.arange(6, dtype=np.float32)
    out = resample_to_grid(e, ts, step=3.0)                 # grid 0,3 -> frames 0,3
    assert isinstance(out, torch.Tensor)
    assert list(out[:, 0].tolist()) == [0.0, 3.0]
