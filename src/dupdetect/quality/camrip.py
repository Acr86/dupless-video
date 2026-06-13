"""Heuristics for detecting cam rips / low quality. Score 0..1 (not binary).
Does NOT affect identity; only decides 'keep' within a cluster (rank_cluster)."""
from __future__ import annotations

from dupdetect.models import Probe

# reference bits-per-pixel-frame. Modern codecs (x264/x265) look fine at ~0.04 bpp,
# so 0.10 was flagging legitimate efficient encodes as 'cam'. 0.04 only penalizes
# genuinely starved encodes. CALIBRATE — codec-dependent.
TARGET_BPP = 0.04
# Laplacian variance scale for mapping sharpness -> cam. CALIBRATE.
SHARP_VAR_SCALE = 0.05

# KNOWN LIMITATIONS (measured on real data, see step 10/calibration):
#  - bitrate-vs-resolution does NOT capture low resolution: a 480p screener with
#    healthy bpp scores "good" even if it's a worse copy. RESOLUTION weighs more in rank_cluster.
#  - Laplacian variance is FOOLED by grain/noise (a grainy screener looks sharp).
#    That is why cam_score is a WEAK ranking signal, not a primary one.


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _bitrate_cam(probe: Probe) -> float | None:
    """Bitrate-vs-resolution: a '1080p' with low bitrate is soft/blocky."""
    if not (probe.bitrate_kbps and probe.width and probe.height):
        return None
    bpp = (probe.bitrate_kbps * 1000) / (probe.width * probe.height * 24.0)
    return _clamp01((TARGET_BPP - bpp) / TARGET_BPP)         # low bpp -> high cam


def _sharpness_cam(frames) -> float:
    """Laplacian variance (sharpness). Cam/blurry => low variance => high cam."""
    import torch
    import torch.nn.functional as F

    x = frames.float().mean(dim=1, keepdim=True)            # approx luminance [N,1,H,W]
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                     dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    lap = F.conv2d(x, k, padding=1)
    var = float(lap.var(dim=(1, 2, 3)).mean().item())       # mean spatial variance
    return _clamp01(SHARP_VAR_SCALE / (var + SHARP_VAR_SCALE))   # high var -> cam ~0


def cam_score(probe: Probe, frames=None) -> float:
    """Combines cam/low-quality signals into [0,1]. Starts with bitrate-vs-resolution
    (free, probe only) and adds Laplacian sharpness if decoded frames are available.
    Averages the available signals. Suspicion threshold in config (cam_score_flag)."""
    parts = []
    br = _bitrate_cam(probe)
    if br is not None:
        parts.append(br)
    if frames is not None and getattr(frames, "shape", (0,))[0] > 0:
        parts.append(_sharpness_cam(frames))
    if not parts:
        return 0.0
    return float(sum(parts) / len(parts))
