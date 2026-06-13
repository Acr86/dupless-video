"""Frame decode and sampling. GPU-first (NVDEC) on an NVIDIA GPU, CPU fallback.

KEYFRAME sampling (`-skip_frame nokey`): NVDEC decodes only I-frames
(~1 every 2-5s) and SKIPS P/B frames -> ~14x faster than decoding everything at 2fps (the
old "frame-by-frame" approach). Because keyframes are irregular and at different
positions per encode, their real TIMESTAMPS (showinfo) are captured so that
derived scene cuts remain comparable across copies.

Returns (ImageNet-normalized tensor [N,C,H,W], timestamps[N] in s). Falls back to
software (CPU) if NVDEC doesn't support the codec (mpeg4/xvid). CPU-only with a warning if no GPU.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import warnings

import numpy as np

from dupdetect.quality.color import ColorStats, color_descriptor
from dupdetect.runtime import resolve_binary
from dupdetect.util import CREATE_NO_WINDOW

# ImageNet normalization — expected by DINOv2 backbones.
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)
DEFAULT_SIZE = 224                      # 16 patches of 14 px

# Adaptive sampling (defaults; the pipeline passes these from thresholds.yaml):
# files > SEEK_THRESHOLD_BYTES are sampled by sparse SEEK instead of full demux.
SEEK_THRESHOLD_BYTES = 6 * 1024 ** 3
SEEK_N = 200

# Per-file decode timeout (broken/missing index -> demux becomes very slow). When exceeded,
# the error carries 'timeout' -> the store classifies it as 'reindex' (remux fixes it).
# Legitimate giants use SEEK (fast) and never get close; a <6GB file that exceeds this has a bad index.
DECODE_TIMEOUT_S = 240

_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def sample_timestamps(duration_s: float, fps: float) -> np.ndarray:
    """Deterministic timestamps at `fps`. Utility/compatibility (the real decode uses keyframes)."""
    n = max(1, int(duration_s * fps))
    return np.linspace(0, duration_s, n, endpoint=False)


def _resolve_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    warnings.warn("CUDA not available: decode_frames falls back to CPU (slower).")
    return "cpu"


def _ffmpeg_keyframes(path: str, size: int, use_nvdec: bool,
                      timeout: int = DECODE_TIMEOUT_S) -> tuple[np.ndarray, np.ndarray]:
    """Keyframes scaled to size×size [N,H,W,3] uint8 + their timestamps[N] (s).
    `-skip_frame nokey` -> NVDEC decodes only I-frames (fast). `showinfo` (stderr,
    loglevel info) provides the pts_time of each frame. Falls back to software if NVDEC fails.

    Frames go to a TEMP FILE (not the stdout pipe): capturing hundreds of MB
    via pipe stalls on Windows; ffmpeg->file + np.fromfile is robust and fast."""
    vf = f"scale={size}:{size},showinfo"
    fb = size * size * 3
    fd, raw = tempfile.mkstemp(suffix=".raw", prefix="dupdec_")
    os.close(fd)
    # -fps_mode passthrough: do NOT duplicate keyframes to fill CFR (without this, ffmpeg
    # emits ~all frames repeated -> 32GB and very slow). This way ONLY keyframes are output.
    tail = ["-y", "-skip_frame", "nokey", "-i", path, "-an", "-sn", "-vf", vf,
            "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", raw]
    ff = resolve_binary("ffmpeg")                            # platform seam (bundle -> env -> venv -> PATH)
    attempts = []
    if use_nvdec:
        attempts.append([ff, "-v", "info", "-hwaccel", "cuda", *tail])
    attempts.append([ff, "-v", "info", *tail])               # software (fallback / no GPU)

    last_err = ""
    try:
        for cmd in attempts:
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                                      creationflags=CREATE_NO_WINDOW)  # stderr only (frames->file)
            except subprocess.TimeoutExpired:
                last_err = f"timeout (>{timeout}s) decoding (broken/missing index?)"
                continue
            if proc.returncode == 0:
                times = [float(t) for t in _PTS_RE.findall(proc.stderr.decode("utf-8", "replace"))]
                data = np.fromfile(raw, dtype=np.uint8)
                n = data.size // fb
                if n == 0:
                    return np.empty((0, size, size, 3), np.uint8), np.empty(0, np.float32)
                frames = data[:n * fb].reshape(n, size, size, 3)
                m = min(n, len(times))                       # align counts (robustness)
                return frames[:m].copy(), np.array(times[:m], dtype=np.float32)
            last_err = proc.stderr.decode("utf-8", "ignore")
        raise RuntimeError(f"ffmpeg decode: {last_err.strip()[-200:] or 'unknown error'}")
    finally:
        if os.path.exists(raw):
            os.remove(raw)


def _seek_keyframes(path: str, size: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """SPARSE seek sampling (PyAV): seeks to `n` evenly-spaced timestamps via the container
    index and decodes the keyframe for each -> does NOT read the whole file. For
    4K/8K giants it is ~3x faster than demuxing all keyframes (cost ~constant).

    Returns (frames[N,H,W,3] uint8 size×size, timestamps[N] s). Deduplicates repeated timestamps
    (multiple targets may land on the same keyframe when the GOP is long)."""
    import av

    container = av.open(path)
    try:
        vs = container.streams.video[0]
        vs.codec_context.skip_frame = "NONKEY"             # decode only I-frames
        tb = vs.time_base
        if vs.duration:
            dur = float(vs.duration * tb)
        elif container.duration:
            dur = container.duration / av.time_base
        else:
            dur = 0.0
        if dur <= 0:
            return np.empty((0, size, size, 3), np.uint8), np.empty(0, np.float32)
        frames: list[np.ndarray] = []
        times: list[float] = []
        last_pts = None
        for target in np.linspace(0.0, dur * 0.999, max(1, n)):
            try:
                container.seek(int(target / tb), stream=vs, backward=True, any_frame=False)
                for f in container.decode(vs):
                    if f.pts is not None and f.pts == last_pts:
                        break                              # same keyframe as previous -> skip
                    arr = f.reformat(width=size, height=size, format="rgb24").to_ndarray()
                    frames.append(arr)
                    times.append(float(f.pts * tb) if f.pts is not None else float(target))
                    last_pts = f.pts
                    break
            except av.AVError:
                continue
        if not frames:
            return np.empty((0, size, size, 3), np.uint8), np.empty(0, np.float32)
        return np.stack(frames), np.array(times, dtype=np.float32)
    finally:
        container.close()


def decode_frames(path: str, size: int = DEFAULT_SIZE,
                  seek_threshold_bytes: int = SEEK_THRESHOLD_BYTES, seek_n: int = SEEK_N,
                  height: int = 0, decode_timeout_s: int = DECODE_TIMEOUT_S):
    """Decodes the KEYFRAMES of `path` -> (normalized tensor [N,C,H,W], timestamps[N] s).
    Adaptive by RESOLUTION and size: 8K (height >= 4320) always sampled by sparse SEEK
    (demuxing an entire 8K file is brutal); 4K/large (> `seek_threshold_bytes`) also by seek;
    the rest (HD/SD) by keyframe demux (faster, seek has per-jump overhead). Falls back to
    demux if seek fails. Runs on CUDA if a GPU is available (NVDEC), otherwise CPU with a warning.
    Raises if the file is unreadable (corrupt) -> caught by pass 1 and written to the `problems` table."""
    import torch

    device = _resolve_device()
    try:
        big = os.path.getsize(path) > seek_threshold_bytes
    except OSError:
        big = False
    big = big or (height and height >= 4320)               # 8K -> seek even if below the threshold

    rgb, times = (np.empty((0, size, size, 3), np.uint8), np.empty(0, np.float32))
    if big:
        try:
            rgb, times = _seek_keyframes(path, size, seek_n)
        except Exception:                                  # seek not supported -> demux
            rgb = np.empty((0, size, size, 3), np.uint8)
    if rgb.shape[0] == 0:                                  # not-big, or seek empty/failed
        rgb, times = _ffmpeg_keyframes(path, size, use_nvdec=(device == "cuda"),
                                       timeout=decode_timeout_s)
    if rgb.shape[0] == 0:
        return torch.empty((0, 3, size, size), dtype=torch.float32, device=device), times, ColorStats()

    color = color_descriptor(rgb)                            # color signals from the RAW rgb (reused)
    x = torch.from_numpy(rgb.copy()).to(device)              # [N,H,W,3] uint8
    x = x.permute(0, 3, 1, 2).float().div_(255.0)            # [N,3,H,W] 0..1
    mean = torch.tensor(DINOV2_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(DINOV2_STD, device=device).view(1, 3, 1, 1)
    x.sub_(mean).div_(std)                                   # ImageNet norm
    return x, times, color
