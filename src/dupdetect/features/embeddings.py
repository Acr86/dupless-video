"""Per-frame deep embeddings. SOTA cross-language signal.

A1: stored on disk as fp16; in cache, resident CUDA tensors.
A2: in addition to the global mean-pool, K descriptors per temporal window.

Precision: fp16 (autocast). FP8 (torchao, Blackwell sm_120) and NVDEC zero-copy were evaluated;
see docs/HPC_PIPELINE.md — on this stack (Windows, torch 2.11, library on HDD) they do NOT help
(the bottleneck is disk I/O, and FP8 eager is slower without torch.compile, which is broken on Windows).
"""
from __future__ import annotations

import warnings

import numpy as np

DINOV2_REPO = "facebookresearch/dinov2"


class Embedder:
    """DINOv2 backbone (or CLIP / ViSiL weights). Lazy-loads the model to GPU.
    M3: always lives and runs in the main process (NVDEC decode + embed serialized),
    never in a forked worker — to avoid crossing CUDA context."""

    def __init__(self, model: str = "dinov2_vitb14", dim: int = 768, batch: int = 512,
                 fps: float = 2.0, algo_version: int = 3,   # v3: keyframe sampling
                 fp8: bool = False):
        self.model_name = model
        self.dim = dim
        self.batch = batch
        self.fps = fps
        self.algo_version = algo_version
        # FP8 (Blackwell): extreme mixed precision for embedding. OPT-IN, off by default.
        # Only pays off in compute-bound regimes (Linux+NVMe+8K) with fused quant (torch.compile);
        # on this stack it is slower without compile (see docs/HPC_PIPELINE.md). Fidelity guard:
        # if the fp8↔fp16 cosine drops <0.99, reverts to fp16 (protects the zero-FP of T1/T2).
        self.fp8 = fp8
        self._model = None
        self._device: str | None = None

    @property
    def feature_version(self) -> str:
        """C4: changes when model/dim/fps/algorithm change -> invalidates the cache."""
        return f"{self.model_name}|d{self.dim}|fps{self.fps}|algo{self.algo_version}"

    def _ensure(self):
        """Loads DINOv2 from torch.hub to CUDA (or CPU with a warning). Idempotent.
        The first time, torch.hub clones the repo and downloads the checkpoint (~330 MB, cached)."""
        if self._model is not None:
            return
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cpu":
            warnings.warn("CUDA not available: DINOv2 running on CPU (slow).")
        self._model = self._load_model().to(self._device).eval()
        if self.fp8 and self._device == "cuda" and not self._try_enable_fp8():
            self.fp8 = False                            # low fidelity -> reverted to fp16

    def _load_model(self):
        """Load the DINOv2 backbone. OFFLINE (frozen/installed app): from the vendored repo with
        source='local' + TORCH_HOME at the bundle -> zero network. DEV (no bundle): normal online
        torch.hub load (clones + downloads the first time). Validated: source='local' ~1.3s, no GitHub."""
        import torch

        from dupdetect import runtime
        if runtime.configure_offline_model():
            return torch.hub.load(str(runtime.model_repo_dir()), self.model_name, source="local")
        return torch.hub.load(DINOV2_REPO, self.model_name)

    def _try_enable_fp8(self) -> bool:
        """Swaps nn.Linear layers to FP8 (e4m3, `torch._scaled_mm`) + fidelity GUARD: compares
        fp8 vs fp16 embeddings on a sample batch; if the median cosine < 0.99 reverts to fp16
        (cosine precision upholds the zero-false-positives guarantee of T1/T2). Returns whether fp8 stuck."""
        import torch
        import torch.nn.functional as F

        sample = torch.randn(16, 3, 224, 224, device=self._device, dtype=torch.float16)
        with torch.inference_mode(), torch.autocast(self._device, dtype=torch.float16):
            ref = F.normalize(self._model(sample).float(), dim=1)
            _swap_linears_fp8(self._model)
            test = F.normalize(self._model(sample).float(), dim=1)
        cos = float((ref * test).sum(1).median())
        if cos >= 0.99:
            return True
        warnings.warn(f"FP8: low fidelity (cosine {cos:.4f} < 0.99); reverting to fp16 "
                      "to protect the zero-FP guarantee of T1/T2.")
        self._model = self._load_model().to(self._device).eval()
        return False

    def encode(self, frames) -> np.ndarray:
        """[N,C,H,W] (torch, CUDA) -> [N, D] L2-normalized, fp16, on CPU for the store.

        Processes in large batches (a modern GPU handles thousands of frames).
        L2-normalize per frame so that cosine similarity is a dot product.
        fp16 on persist (A1): the loss is irrelevant for cosine.
        """
        import torch
        import torch.nn.functional as F

        self._ensure()
        n = int(frames.shape[0])
        if n == 0:
            return np.empty((0, self.dim), dtype=np.float16)

        outs = []
        with torch.inference_mode():
            for i in range(0, n, self.batch):
                batch = frames[i:i + self.batch].to(self._device)
                with torch.autocast(self._device, dtype=torch.float16,
                                    enabled=(self._device == "cuda")):
                    feat = self._model(batch)              # [B, D] (CLS token)
                feat = F.normalize(feat.float(), dim=1)    # L2 per frame, in fp32
                outs.append(feat.to(torch.float16).cpu())
        return torch.cat(outs).numpy()


def _swap_linears_fp8(model) -> int:
    """Replaces in-place the nn.Linear layers of `model` with an FP8 Linear (e4m3) that quantizes
    activation+weight per-tensor and multiplies with `torch._scaled_mm` (Blackwell FP8
    tensor cores). Only layers with dims that are multiples of 16 (requirement of _scaled_mm).
    Real speedup requires fusing quant+matmul (torch.compile/torchao); in eager it is slower —
    see docs/HPC_PIPELINE.md. Returns the number of layers swapped."""
    import torch
    import torch.nn as nn

    class Fp8Linear(nn.Module):
        def __init__(self, lin: nn.Linear):
            super().__init__()
            w = lin.weight.data.float()
            self.outf, self.inf = w.shape
            ws = (w.abs().amax() / 448.0).clamp(min=1e-8)   # e4m3 max ~448
            self.register_buffer("w8", (w / ws).to(torch.float8_e4m3fn))
            self.register_buffer("wsd", ws.float())
            self.register_buffer("b", lin.bias.data.half().clone() if lin.bias is not None else None)

        def forward(self, x):
            sh = x.shape
            x2 = x.reshape(-1, self.inf).half()
            xs = (x2.abs().amax() / 448.0).clamp(min=1e-8).float()
            x8 = (x2 / xs).to(torch.float8_e4m3fn)
            o = torch._scaled_mm(x8, self.w8.t(), scale_a=xs, scale_b=self.wsd,
                                 out_dtype=torch.float16)
            if self.b is not None:
                o = o + self.b
            return o.reshape(*sh[:-1], self.outf)

    n = 0
    for mod in model.modules():
        for name, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear) and ch.in_features % 16 == 0 and ch.out_features % 16 == 0:
                setattr(mod, name, Fp8Linear(ch))
                n += 1
    return n


def global_descriptor(emb: np.ndarray) -> np.ndarray:
    """Coarse per-video vector: mean-pool of embeddings + L2 norm.
    Fast first-pass retrieval (A2). float32 (single vector, cheap)."""
    v = emb.astype(np.float32).mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def window_descriptors(emb: np.ndarray, k: int) -> np.ndarray:
    """A2: K descriptors per temporal window (mean-pool per segment + L2).
    Rescues cam rips and globals contaminated by ads: a local descriptor
    survives even if the global average shifts.

    Splits [N, D] into min(k, N) contiguous windows, mean-pool + L2 each.
    If N < k returns N rows (one per frame) instead of empty windows.
    """
    if k <= 0 or emb.shape[0] == 0:
        return np.empty((0, emb.shape[1] if emb.ndim == 2 else 0), dtype=np.float32)
    out = []
    for chunk in np.array_split(emb.astype(np.float32), min(k, emb.shape[0]), axis=0):
        v = chunk.mean(axis=0)
        n = np.linalg.norm(v)
        out.append(v / n if n > 0 else v)
    return np.stack(out).astype(np.float32)
