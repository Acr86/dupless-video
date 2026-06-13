"""Coarse retrieval tests (step 7): FAISS IndexFlatIP global + multi-vector."""
from __future__ import annotations

import numpy as np
import pytest

from dupdetect.match.retrieval import CoarseIndex

pytest.importorskip("faiss")


def _unit(rows: list[list[float]]) -> np.ndarray:
    m = np.array(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_query_global_finds_nearest_neighbor():
    paths = ["/a.mkv", "/b.mkv", "/c.mkv"]
    vecs = _unit([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    idx = CoarseIndex(dim=4)
    idx.build(paths, vecs)

    q = _unit([[0.9, 0.1, 0, 0]])[0]          # close to 'a'
    res = idx.query_global(q, k=2)
    assert res[0][0] == "/a.mkv"               # nearest is 'a'
    assert res[0][1] > res[1][1]               # 'a' score is higher (cosine)
    assert res[0][1] == pytest.approx(0.9938, abs=1e-2)


def test_query_global_k_larger_than_n():
    idx = CoarseIndex(dim=4)
    idx.build(["/a.mkv"], _unit([[1, 0, 0, 0]]))
    res = idx.query_global(_unit([[1, 0, 0, 0]])[0], k=25)   # k > ntotal
    assert len(res) == 1                       # does not crash, returns what is available


def test_query_global_empty_index():
    idx = CoarseIndex(dim=4)
    idx.build([], np.empty((0, 4), dtype=np.float32))
    assert idx.query_global(_unit([[1, 0, 0, 0]])[0], k=5) == []


def test_query_windows_merges_owners():
    # 4 window-vecs: 2 from 'a', 2 from 'b'
    owners = ["/a.mkv", "/a.mkv", "/b.mkv", "/b.mkv"]
    wv = _unit([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    idx = CoarseIndex(dim=4)
    idx.build(["/a.mkv", "/b.mkv"], _unit([[1, 1, 0, 0], [0, 0, 1, 1]]),
              window_owners=owners, window_vecs=wv)

    # query with 2 window-vecs: one near 'a', one near 'b'
    q = _unit([[1, 0, 0, 0], [0, 0, 1, 0]])
    hits = idx.query_windows(q, k=1)
    assert hits == {"/a.mkv", "/b.mkv"}


def test_query_windows_no_window_index():
    idx = CoarseIndex(dim=4)
    idx.build(["/a.mkv"], _unit([[1, 0, 0, 0]]))   # no window_vecs
    assert idx.query_windows(_unit([[1, 0, 0, 0]]), k=3) == set()


# --------------------------------------------------------------- duration gate (O(N^2) prune)
def test_gate_by_global_prunes_dissimilar_keeps_unindexed():
    """The duration safety net is gated by global cosine: same-length-but-different videos (low
    cosine) are dropped; real dups (>=0.962 measured) pass; LITE paths not in the index are KEPT
    (can't gate -> no recall loss). Top-k/window retrieval are untouched by this."""
    paths = ["/a.mkv", "/b.mkv", "/c.mkv"]
    vecs = _unit([[1, 0, 0, 0], [0.9, 0.4, 0, 0], [0, 1, 0, 0]])   # cosines to q: 1.0, ~0.91, 0.0
    idx = CoarseIndex(dim=4)
    idx.build(paths, vecs)
    q = _unit([[1, 0, 0, 0]])[0]
    assert idx.gate_by_global(q, {"/a.mkv", "/b.mkv", "/c.mkv"}, gate=0.7) == {"/a.mkv", "/b.mkv"}
    kept = idx.gate_by_global(q, {"/a.mkv", "/c.mkv", "/lite.mkv"}, gate=0.7)
    assert "/lite.mkv" in kept and "/a.mkv" in kept and "/c.mkv" not in kept   # unindexed kept
    assert idx.gate_by_global(q, {"/a.mkv", "/c.mkv"}, gate=0.0) == {"/a.mkv", "/c.mkv"}  # disabled


def test_candidate_paths_gates_duration_neighbors_not_topk(monkeypatch):
    """candidate_paths gates the DURATION neighbors by cosine but never the top-k/window retrieval.
    A different-content same-duration neighbor (low cosine) is dropped; a similar one is kept."""
    from dupdetect.config import load_thresholds
    from dupdetect.match import matcher
    from dupdetect.models import Probe, Quality, Record

    th = load_thresholds()
    idx = CoarseIndex(dim=4)
    idx.build(["/dup.mkv", "/diff.mkv"], _unit([[1, 0, 0, 0], [0, 1, 0, 0]]))   # dup~1.0, diff~0.0
    rec = Record(path="/q.mkv", mtime=0.0, size=1, probe=Probe(100.0, 100, 100, "h264", 1000, []),
                 content_hash="q", global_vec=_unit([[1, 0, 0, 0]])[0],
                 window_vecs=np.zeros((0, 4), np.float32), embeddings=np.zeros((0, 4), np.float16),
                 audio_fp=np.zeros(0, np.uint32), scene_cuts=np.zeros(0, np.float32), quality=Quality())
    # both are duration-neighbors; none come from top-k (query_global returns them, but we check the gate)
    monkeypatch.setattr(matcher, "duration_blocking", lambda *a, **k: {"/dup.mkv", "/diff.mkv"})
    monkeypatch.setattr(idx, "query_global", lambda *a, **k: [])     # isolate: only the gated net
    cands = matcher.candidate_paths(rec, store=None, index=idx, th=th)
    assert "/dup.mkv" in cands and "/diff.mkv" not in cands          # low-cosine neighbor pruned
