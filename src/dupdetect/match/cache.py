"""A1: embedding residency as CUDA tensors (the README promise).

End-to-end GPU data path in the fine stage (DECISION A):
  - on disk: fp16 (22 MB/film)
  - in cache: torch CUDA tensor, loaded ONCE; 1-2k films all fit
  - LRU only if the library grows beyond available VRAM/RAM

torch is imported lazily so the rest of the package imports without CUDA.
If CUDA is unavailable, falls back to a CPU tensor with a warning (CPU portability).
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any

from dupdetect.store import FingerprintStore


class EmbeddingCache:
    def __init__(self, store: FingerprintStore, max_items: int | None = None,
                 device: str | None = None):
        self.store = store
        self.max_items = max_items          # None => no limit (all fit)
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._device = device               # None => auto (cuda if available, else cpu)

    def _resolve_device(self):
        import torch
        if self._device:
            return self._device
        if torch.cuda.is_available():
            return "cuda"
        warnings.warn("CUDA not available: EmbeddingCache falling back to CPU (slower).")
        return "cpu"

    def get(self, path: str):
        """Resident CUDA tensor [N, D] (fp16). Loaded from disk only on the 1st access.
        align_video operates on these tensors -> matmul and DP on GPU."""
        hit = self._cache.get(path)
        if hit is not None:
            self._cache.move_to_end(path)
            return hit
        import torch
        try:
            rec = self.store.load(path, with_embeddings=True)
        except (FileNotFoundError, OSError) as e:        # .npy moved/deleted -> treat as absent
            raise KeyError(path) from e
        if rec is None:
            raise KeyError(path)
        # emb comes as fp16 from disk; keep fp16 on device (half the VRAM)
        t = torch.from_numpy(rec.embeddings).to(self._resolve_device())
        self._cache[path] = t
        if self.max_items and len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return t

    def warm(self, paths: list[str]) -> None:
        """Full-scan: preload everything to device once before re-ranking."""
        for p in paths:
            try:
                self.get(p)
            except KeyError:
                pass
