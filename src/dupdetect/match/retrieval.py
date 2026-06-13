"""Coarse stage: candidate retrieval (A2 multi-vector + duration blocking).

Three candidate sources that are UNIONED (uncapped recall):
  1. global mean-pool (fast first pass)
  2. multi-vector by temporal window (rescues cam rips and globals with ads)
  3. duration blocking (always compares similar runtimes, no matter what)
"""
from __future__ import annotations

import numpy as np


class CoarseIndex:
    """Dual FAISS index: global mean-pool + window-vecs (A2)."""

    def __init__(self, dim: int, metric: str = "ip"):
        self.dim = dim
        self.metric = metric
        self.global_paths: list[str] = []
        self.window_owners: list[str] = []        # owner for each window-vec
        self._global_index = None
        self._window_index = None

    def build(self, global_paths: list[str], global_vecs: np.ndarray,
              window_owners: list[str] | None = None,
              window_vecs: np.ndarray | None = None) -> None:
        """Builds both indexes. L2-normed vecs => IndexFlatIP (IP == cosine).
        Exact: at 1-2k films (and K~12 => ~24k window-vecs) fits comfortably in RAM."""
        import faiss

        self.global_paths = list(global_paths)
        self._global_index = faiss.IndexFlatIP(self.dim)
        # Keep the vecs + a path->row map for the cosine gate (vectorized, no per-call faiss reconstruct).
        self._gvecs = (np.ascontiguousarray(global_vecs, dtype=np.float32)
                       if global_vecs is not None and getattr(global_vecs, "size", 0)
                       else np.empty((0, self.dim), dtype=np.float32))
        self._pos = {p: i for i, p in enumerate(self.global_paths)}
        if self._gvecs.size:
            self._global_index.add(self._gvecs)

        self.window_owners = list(window_owners or [])
        self._window_index = None
        if window_vecs is not None and getattr(window_vecs, "size", 0):
            self._window_index = faiss.IndexFlatIP(self.dim)
            self._window_index.add(np.ascontiguousarray(window_vecs, dtype=np.float32))

    def query_global(self, vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """k nearest neighbors by global mean-pool. Returns [(path, score_coseno)]."""
        if self._global_index is None or self._global_index.ntotal == 0:
            return []
        q = np.ascontiguousarray(np.asarray(vec, dtype=np.float32).reshape(1, -1))
        kk = min(int(k), self._global_index.ntotal)
        scores, idx = self._global_index.search(q, kk)
        return [(self.global_paths[i], float(s))
                for s, i in zip(scores[0], idx[0]) if i >= 0]

    def gate_by_global(self, vec: np.ndarray, paths, gate: float) -> set[str]:
        """Of `paths`, keep those whose GLOBAL cosine to `vec` >= `gate` (L2-normed vecs -> dot ==
        cosine). Prunes the duration safety net so a dense library doesn't explode Pass-2. Paths NOT
        in the global index (LITE/exact-only, no embedding) CAN'T be gated -> kept (no recall loss).
        gate <= 0 disables (returns all)."""
        paths = set(paths)
        if gate <= 0 or self._gvecs.size == 0 or not paths:
            return paths
        q = np.asarray(vec, dtype=np.float32).ravel()
        rows, gated = [], []
        out: set[str] = set()
        for p in paths:
            i = self._pos.get(p)
            if i is None:
                out.add(p)                         # not indexed -> can't gate -> keep (safety net)
            else:
                rows.append(i); gated.append(p)
        if rows:
            sims = self._gvecs[rows] @ q           # vectorized cosine
            out.update(p for p, s in zip(gated, sims) if s >= gate)
        return out

    def query_windows(self, window_vecs: np.ndarray, k: int) -> set[str]:
        """A2: for each query window-vec, its k neighbors; union of OWNERS.
        Rescues cam rips / globals contaminated by ads that mean-pool misses."""
        if self._window_index is None or self._window_index.ntotal == 0:
            return set()
        wv = np.asarray(window_vecs, dtype=np.float32)
        if wv.ndim == 1:
            wv = wv.reshape(1, -1)
        if wv.size == 0:
            return set()
        kk = min(int(k), self._window_index.ntotal)
        _, idx = self._window_index.search(np.ascontiguousarray(wv), kk)
        return {self.window_owners[i] for row in idx for i in row if i >= 0}
