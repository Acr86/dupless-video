# HPC pipeline: CPU/GPU acceleration for duplicate detection

> Production design and configuration for a Blackwell-class NVIDIA GPU (sm_120), with **real
> measurements** on a representative library and honest criteria for when each lever actually pays.
> Executive summary first; the detail and the production recipe (8K / Linux / NVMe) after.

## TL;DR — where the bottleneck is (measured)

In this environment (**Windows 11, torch 2.11+cu128, a Blackwell GPU, library on a spinning HDD — a
multi-disk Storage Space over SATA drives**), the pipeline is **I/O-bound: the bottleneck is the
sequential read/demux off the mechanical disk, not GPU compute.**

Cost per HD film (~5 s/film at `--workers 6`):

| Stage | Cost | Resource | Bottleneck? |
|---|---|---|---|
| probe (ffprobe) | 0.11s | disk (light) | no |
| content_hash (head\|mid\|tail) | 0.12s | disk (24 MB) | no |
| audio_fp `fpcalc -length 0` | 3.3s | **full demux** | partial |
| detect_language (whisper base) | 3.9s | CPU | partial |
| **video keyframe demux** | **~5s** | **disk (serial, main)** | **yes** |
| H2D (CPU→GPU, frames at 224²) | 0.04s | PCIe | no |
| embed DINOv2 (fp16) | 1.4s | GPU | no |

The GPU sits **idle** most of the time, waiting on the spinning platter.

## The three candidate levers, checked against measurement

### 1. Extreme mixed precision (FP8/FP4) for embedding inference

**Measured (Blackwell sm_120, DINOv2 ViT-B/14), three levels:**

| Level | Result | Source |
|---|---|---|
| fp8 vs fp16 GEMM (`torch._scaled_mm`, native, ViT MLP) | **1.83x** (409 vs 224 TFLOP/s) | the ceiling of Blackwell's fp8 tensor cores |
| **WHOLE DINOv2 model in fp8** (swap all 48 Linear to `_scaled_mm`, dynamic e4m3 quant) | **0.84x — SLOWER** (2628 vs 3128 frames/s) | per-layer activation-quant overhead exceeds the GEMM saving |
| fp8 vs fp16 embed fidelity | **cosine 0.9938 mean / 0.9892 min** | FP8 would NOT break the tiers; the issue is throughput, not precision |

The fp8 matmul ceiling is real (1.83x), but **(a)** a ViT forward isn't only GEMM
(attention/softmax/layernorm/GELU don't accelerate) → the whole model gains far less than 1.71x;
**(b)** capturing it requires fusing quant+matmul with **`torch.compile`**, which is **broken on this
stack**: no **Triton wheel for Windows** (`import triton` → ModuleNotFoundError) and the Inductor
backend fails (`InductorError: aten.addmm.default`). Without compile, eager fp8 is *slower* than fp16
— **measured end-to-end: 0.84x** (per-layer activation quant overhead exceeds the GEMM saving).

**Verdict — Amdahl rules:** even with perfect fp8 (1.83x) on the embed, that is **~0.57s of compute
on an idle GPU** while the HDD demuxes for ~5s. The max saving is **~1–3% of wall-time/film**, with
**risk** to the cosines that uphold the zero-FP of T1/T2. Not enabled. FP4 (nvfp4) needs
TensorRT-Model-Optimizer/Transformer-Engine (no clean support here) → same verdict, more precision risk.

**IMPLEMENTED as opt-in (`--fp8-embed` / `Embedder(fp8=True)`):** swaps DINOv2's 48 `nn.Linear` to
FP8 e4m3 with `torch._scaled_mm`, behind a mandatory **FIDELITY GUARD** — it compares the fp8-vs-fp16
embed cosine on a sample batch and **reverts to fp16 if it drops below 0.99** (protects the zero-FP of
T1/T2). Measured fidelity: **cosine 0.9939 mean / 0.9905 min**. **Off by default** (eager is slower on
this stack; see table). fp8 and fp16 embeddings are interchangeable in the index (cosine 0.99) → no
cache invalidation.

**When to enable it (compute-bound regime):** Linux, NVMe storage (decode no longer the bottleneck),
and/or large batches of many frames (densely-sampled 8K), with a working `torch.compile(model)`
(essential for the real speedup). Fused-quant recipe:
```python
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
quantize_(model, Float8DynamicActivationFloat8WeightConfig())
model = torch.compile(model, mode="max-autotune")   # essential for the speedup
```
Mandatory precision guard before trusting it: compare the global descriptor's FP8-vs-FP16 cosine on a
real film and abort if < 0.99 (protects the zero-false-positives of T1/T2).

### 2. NVDEC/NVENC (hardware decode/encode)

**NVDEC is already in use:** `ffmpeg -hwaccel cuda` in [frames.py](../src/dupdetect/features/frames.py).
But, measured, **it doesn't help** for keyframe sampling:
- NVDEC 5.2s/932 MB vs CPU 4.4s/714 MB → a tie per MB. Only keyframes (sparse) are decoded; the cost
  is **demuxing the whole container linearly**, not decoding.
- NVDEC would only win if we decoded **many** high-resolution frames (4K/8K I-frames are expensive on
  CPU). With sparse sampling that isn't the case.

**NVENC: not applicable to detection** (nothing is encoded). It would only matter for re-encoding the
"keep" copy after a cluster is decided — out of the detector's scope.

**Zero-copy NVDEC→VRAM (no temp file):** the ideal version decodes keyframes straight to CUDA tensors
(no ~150 MB temp-file round-trip). Blocked today: `PyNvVideoCodec` **doesn't import on Python 3.14**
(`from ast import Str`, removed in 3.12+); `torchvision`'s Windows wheel ships without VideoReader;
`torchcodec`/DALI have no clean CUDA support here. On a Linux box with Python ≤3.12: use
`PyNvVideoCodec.SimpleDecoder(path, use_device_memory=True)` → DLPack → CUDA tensor → DINOv2, scaling
on the GPU (`scale_cuda`/NPP) so an 8K frame never crosses PCIe at full resolution.

### 3. Asynchronous CUDA streams (overlap read ∥ H2D ∥ compute)

- **H2D ∥ compute:** measured at whole-film level, the H2D of **1500 frames at 224² (903 MB) is
  56.8 ms vs ~565 ms of forward → H2D is 10% of compute**, and that 10% is itself a fraction of the
  decode-bound wall-time. Key: frames are scaled to 224² **before** crossing PCIe, so the H2D size is
  **independent of source resolution** — the "PCIe saturation" worry **doesn't occur** (an 8K keyframe
  transfers like an HD one: 150 KB). A copy-stream + pinned + double-buffer encode (H2D of batch N+1 ∥
  compute of N) **measured end-to-end: 1.02x (1.7%)** — H2D 8.2 ms of 520 ms; cosine 1.000000. Reverted:
  complexity without benefit (H2D is already marginal).
- **decode ∥ compute:** decode runs in an ffmpeg **subprocess** (not in torch's CUDA stream), so torch
  streams can't overlap it. The overlap here is at the *process* level (prefetch: read film N+1 from
  disk while embedding N) — see "Available optimizations".
- **Parallelizing decode across files: counterproductive on HDD.** Measured: 3 concurrent decodes =
  **7x slower** (head thrashing). On a 2-spindle Storage Space the disk-concurrency optimum is ~2;
  that's why `--workers 6` helps (CPU whisper provides cover) but going higher doesn't scale.

## The regime depends on storage (measured on both)

The bottleneck — and therefore the optimal configuration — **changes with where the videos live**:

| Metric | Library on **HDD** (Storage Space) | Library on **NVMe SSD** (PCIe 4) |
|---|---|---|
| 3× decode concurrency | **7x SLOWER** (head thrashing) | **1.62x faster** |
| Parallel decode + embed (vs serial) | impossible (worse) | **2.19x** |
| End-to-end `--workers 6` vs `1` | 2.0x | 1.26x (bottleneck → CPU whisper) |
| End-to-end `--decode-workers 4` (with workers=6) | counterproductive | **+1.33x extra** |
| Demux 4K 30 GB | ~worst case (mechanical) | 66s |

On **SSD the system stops being disk-bound**: parallelizing the decode DOES pay (the "overlap
read↔compute" lever) and the new bottleneck becomes **whisper (CPU)**.

## Recommended configuration

**Library on HDD** (bottleneck = disk; concurrency thrashes):
```bash
dupdetect scan "D:/Videos" --workers 6                       # serial decode on the main process
```

**Library on SSD/NVMe** (bottleneck = compute; decode parallelizes):
```bash
dupdetect scan "C:/Videos" --workers 6 --decode-workers 4    # decode on a thread pool
```
- `--decode-workers N` (>1 ONLY on SSD): moves decode to a bounded-prefetch thread pool and keeps the
  GPU embed on the main process → overlaps read↔compute. **Implemented** in
  [fullscan.py](../src/dupdetect/pipeline/fullscan.py) (`_drain_pipelined`); default 1 preserves the
  HDD-safe behavior. Measured +1.33x end-to-end (with whisper) and 2.19x on the decode+embed portion.
- Precision: **fp16** (default). FP8 still isn't worth it (eager slower; `torch.compile` broken on
  Windows) — see above; reserved for compute-bound Linux production.
- Decode: **NVDEC via ffmpeg** (already on) with CPU fallback; keyframe sampling.
- On SSD the next bottleneck is **language detection** (CPU). It is NOT solved by trusting the
  container's language tag (it often lies): language is DETECTED from the audio. The detector is robust
  (region VAD + multi-window vote, see [language.py](../src/dupdetect/quality/language.py)) and costs
  ~3–6s/film; it runs on the CPU workers in parallel. If it became the dominant bottleneck, the path is
  to move whisper to the GPU (not skip it).

## Production recipe for 8K / fast storage (compute-bound regime)

When the library has 8K **and** the storage is NVMe (the bottleneck shifts from disk to decode+compute),
the order of levers inverts:

1. **Adaptive seek sampling** for large files (>~6 GB) — **IMPLEMENTED**: `decode_frames` jumps to
   ~200 timestamps via the PyAV index instead of demuxing the whole container. Measured on a 34.8 GB
   4K file: demux **168s** → seek **54s** (**3.1x**); on NVMe the edge is smaller (66s→54s) but grows
   with size/8K. Configurable: `sampling.seek_threshold_gb`/`seek_n` in thresholds.yaml. MIXED sampling
   (4K seek vs 1080p demux) broke fine alignment (align_video aligns by frame index); resolved by
   resampling both sequences to a common TEMPORAL grid (`resample_to_grid` + persisted `frame_times`)
   before align_video. Validated: a mixed 720p↔4K pair → CERTAIN (video 0.84/cov 0.99); demux-demux no
   regression.
2. **NVDEC zero-copy to VRAM** (PyNvVideoCodec on Python ≤3.12) + `scale_cuda` on the GPU → the 8K frame
   is reduced to 224² before leaving the GPU; the embed consumes the tensor with no CPU round-trip.
3. **CUDA streams**: decode (NVDEC engine) ∥ embed (tensor cores) ∥ H2D — with decode already in the
   CUDA context (step 2), torch streams can overlap.
4. **FP8 + `torch.compile`** (Linux): ~1.5–2x the embed when the batch is large, with the cosine > 0.99
   guard.

## Available optimizations on request (this environment)

- **Embed/demux prefetch** (~1.4s/film, ~20%): overlap the GPU embed of film N with the disk demux of
  N+1. Touches the validated serial `_pass1` loop → regression risk.
- **Skip whisper using the container language tag** when present: saves ~3.9s CPU/film, but at high
  `--workers` it doesn't improve wall-time (the bottleneck is the main process).

## Appendix: how it was measured

- Decode demux vs NVDEC vs CPU, concurrency, and seek: ffmpeg `-skip_frame nokey` + PyAV seek,
  controlling the disk cache (distinct cold files per configuration).
- FP8/precision: a real DINOv2 ViT-B/14 forward (512 frames, warmup + `cuda.synchronize`) fp16 vs bf16;
  fp8 GEMM ceiling with `torch._scaled_mm` ([512·197,768]×[768,768]); `torch.compile` evidence (no
  Triton on Windows → `InductorError`). H2D: pinned→CUDA copy of 1500×3×224² fp32 (`non_blocking`).
- Disk: `Get-PhysicalDisk` (two SATA HDDs) + a ~44 TB Storage Space volume.
