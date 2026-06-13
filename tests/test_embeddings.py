"""Embedding tests (step 3).

Pure functions (global/window descriptors) are tested without a model. `encode` is
tested with an injected FAKE MODEL to validate batching + L2-norm + fp16 without
downloading DINOv2 (~330 MB). Real DINOv2 loading is validated in a separate demo.
"""
from __future__ import annotations

import numpy as np
import pytest

from dupdetect.features.embeddings import Embedder, global_descriptor, window_descriptors

torch = pytest.importorskip("torch")


# ----------------------------------------------------------- global_descriptor

def test_global_descriptor_is_unit_norm():
    emb = np.random.rand(50, 16).astype(np.float16)
    g = global_descriptor(emb)
    assert g.shape == (16,)
    assert g.dtype == np.float32
    assert np.linalg.norm(g) == pytest.approx(1.0, abs=1e-5)


# ----------------------------------------------------------- window_descriptors

def test_window_descriptors_k_windows_are_unit_norm():
    emb = np.random.rand(100, 8).astype(np.float16)
    w = window_descriptors(emb, k=12)
    assert w.shape == (12, 8)
    assert w.dtype == np.float32
    norms = np.linalg.norm(w, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_window_descriptors_n_less_than_k():
    emb = np.random.rand(3, 8).astype(np.float16)     # N < k
    w = window_descriptors(emb, k=12)
    assert w.shape == (3, 8)                           # one row per frame, not 12


def test_window_descriptors_empty_or_k_zero():
    assert window_descriptors(np.empty((0, 8), np.float16), k=12).shape == (0, 8)
    assert window_descriptors(np.random.rand(10, 8).astype(np.float16), k=0).shape == (0, 8)


# ----------------------------------------------------------- encode (fake model)

class _FakeModel:
    """Returns [B, dim] with distinct rows (non-unit) to verify L2-norm is applied."""
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, x):
        b = x.shape[0]
        scale = torch.arange(1, b + 1, dtype=torch.float32).view(b, 1)
        return torch.ones(b, self.dim) * scale          # row i = i+1 (magnitude != 1)


def _embedder_with_fake(dim=4, batch=2):
    e = Embedder(dim=dim, batch=batch)
    e._model = _FakeModel(dim)        # bypass _ensure (no download)
    e._device = "cpu"
    return e


def test_encode_batched_and_l2norm():
    e = _embedder_with_fake(dim=4, batch=2)
    frames = torch.zeros(5, 3, 8, 8)                    # N=5 > batch=2 -> 3 batches
    out = e.encode(frames)
    assert out.shape == (5, 4)
    assert out.dtype == np.float16
    norms = np.linalg.norm(out.astype(np.float32), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-2)   # each frame unit-norm


def test_encode_empty():
    e = _embedder_with_fake(dim=4)
    out = e.encode(torch.zeros(0, 3, 8, 8))
    assert out.shape == (0, 4) and out.dtype == np.float16


# ----------------------------------------------------------- FP8 (Blackwell, opt-in)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 requires CUDA (loads real DINOv2)")
def test_fp8_embed_maintains_fidelity():
    """FP8 embedding (opt-in) only sticks after the fidelity guard, which requires MEDIAN cosine
    >= 0.99 vs fp16 — the aggregate statistic that upholds the zero-false-positive guarantee for
    T1/T2 (a verdict depends on overall similarity, not a single worst frame). Seeded for
    determinism; we assert the guard's own contract (median, robustly ~0.993), with a sanity floor
    on the per-frame min, which over random-noise inputs legitimately dips to ~0.988."""
    torch.manual_seed(0)
    x = torch.randn(32, 3, 224, 224, device="cuda", dtype=torch.float16)
    e16 = Embedder().encode(x)
    e8m = Embedder(fp8=True)
    e8 = e8m.encode(x)
    assert e8m.fp8 is True                              # fidelity guard passed -> fp8 stuck
    assert e16.shape == e8.shape
    cos = (e16.astype(np.float32) * e8.astype(np.float32)).sum(1)
    assert float(np.median(cos)) >= 0.99               # the guard's contract (upholds zero-FP)
    assert float(cos.min()) >= 0.97                     # sanity floor: catches gross regressions
