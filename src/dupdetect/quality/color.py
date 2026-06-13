"""Color-quality signals to help pick the KEEP among duplicate copies (latest pipeline stage).

Measured from the SAME decoded RGB keyframes used for embeddings (no extra decode), numpy-only.

  - CLIPPING (crushed blacks / blown highlights) = destroyed detail. This is the OBJECTIVE KEEP
    signal: less clipping -> more preserved information -> better source. Validated on real
    color-corrected duplicates: the original clipped ~1% of pixels, bad auto-corrections ~27%.
  - cast / saturation / contrast describe the GRADE (look). "Better grade" is subjective, so these
    only drive a 'color differs' FLAG for the user to decide, never an automatic delete.

Detect, don't trust (§0): computed from pixels, not from color-space metadata (which lies — the
corrected re-encodes were better-tagged than the original).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# C4: bump if the color descriptor changes. feature_version incorporates it -> re-scan.
COLOR_VERSION = 1

# Grade distance above which cluster copies are flagged '⚠ color differs — pick manually'
# (UI-side; the KEEP is still suggested by least-clipping). Validated: corrected vs original ~0.34.
GRADE_DIVERGENCE = 0.15

# Extra clipping (fraction of pixels) the score-winner must have OVER the least-clipped copy before
# the color override downgrades resolution to prefer the original. Below this, a trivial clip edge
# must NOT cost a real resolution upgrade (4K @1% must beat 1080p @0%). Measured: original ~1% vs a
# bad re-grade/upscale ~26% -> a 5-point margin cleanly separates "noise" from "destroyed detail".
CLIP_DOWNGRADE_MARGIN = 0.05

# Luma thresholds (0..255) for "clipped": near-black and near-white.
_BLACK = 5.0
_WHITE = 250.0


@dataclass
class ColorStats:
    clip: float = 0.0          # fraction [0..1] of clipped pixels (black+white); less = better KEEP
    cast: float = 0.0          # color tint: spread of per-channel means (0 = neutral white balance)
    saturation: float = 0.0    # mean saturation [0..1]
    contrast: float = 0.0      # luma std [0..1] (tonal spread)

    def to_list(self) -> list[float]:
        return [self.clip, self.cast, self.saturation, self.contrast]

    @staticmethod
    def from_list(v) -> "ColorStats":
        return ColorStats(*[float(x) for x in v]) if v is not None and len(v) >= 4 else ColorStats()

    def grade_distance(self, other: "ColorStats") -> float:
        """Relative GRADE difference vs another copy (cast/saturation/contrast — NOT clipping).
        Used to flag 'color differs' so the user picks; >~0.15 means a visibly different look."""
        d = 0.0
        for x, y in ((self.cast, other.cast), (self.saturation, other.saturation),
                     (self.contrast, other.contrast)):
            d += abs(x - y) / (max(abs(x), abs(y)) + 1e-6)
        return d / 3.0


def color_descriptor(rgb) -> ColorStats:
    """`rgb`: [N,H,W,3] uint8 decoded keyframes. Aggregates objective color signals over the
    frames. Empty/odd input -> neutral zeros (no false signal). Pure numpy (no cv2)."""
    a = np.asarray(rgb)
    if a.ndim != 4 or a.shape[0] == 0 or a.shape[-1] != 3:
        return ColorStats()
    f = a.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b                      # Rec.601 luma, 0..255
    clip = float(((luma < _BLACK) | (luma > _WHITE)).mean())      # destroyed detail
    # white-balance tint: how far the per-channel means spread (neutral -> all equal)
    ch_means = np.array([r.mean(), g.mean(), b.mean()])
    cast = float((ch_means.max() - ch_means.min()) / 255.0)
    mx = f.max(axis=-1); mn = f.min(axis=-1)
    saturation = float(((mx - mn) / (mx + 1e-6)).mean())          # HSV-style S, 0..1
    contrast = float((luma / 255.0).std())                        # tonal spread
    return ColorStats(clip=clip, cast=cast, saturation=saturation, contrast=contrast)
