"""Core data types for the detector.

`Record`  -> everything analyze_file() produces for ONE file.
`AlignResult` -> output of each aligner (audio/video/scenes).
`Result`  -> match verdict between the queried file and a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from dupdetect.quality.color import ColorStats


class Verdict(str, Enum):
    CERTAIN = "CERTAIN"                    # T0/T1
    VERY_HIGH = "VERY_HIGH"               # T2 (different dub)
    HIGH = "HIGH"                         # T3
    NAME_COPY = "NAME_COPY"              # name-based copy (movie (1).avi); NOT a content tier
    PROBABLE = "PROBABLE"                 # T4 -> review queue
    DIFFERENT = "DIFFERENT"               # T5
    DIFFERENT_EDITION = "DIFFERENT_EDITION"  # contiguous superset (director's cut)


@dataclass
class AudioTrack:
    index: int
    lang_tag: Optional[str]              # declared tag (often wrong/empty)
    codec: Optional[str] = None
    channels: Optional[int] = None


@dataclass
class Probe:
    """ffprobe output. Zero decoding."""
    duration_s: float
    width: int
    height: int
    vcodec: str
    bitrate_kbps: Optional[int]
    audio_tracks: list[AudioTrack] = field(default_factory=list)


@dataclass
class Quality:
    """Quality signals — INDEPENDENT of identity.
    C3: ad_offset does NOT live here; it is a property of the PAIR (AlignResult.offset / matches)."""
    lang_detected: Optional[str] = None   # actual language (whisper-detect), not the tag
    cam_score: float = 0.0                # 0..1, heuristic
    # Audio coverage [0..1]: 1.0 = full audio; <1 = missing/cut audio (a mute copy
    # must not be blindly chosen as KEEP -> user is warned). None = not computed yet (deferred:
    # ensured on-demand for cluster members / Deep); legacy NULL loads as 1.0.
    audio_coverage: float | None = 1.0
    # Color signals (latest stage): clipping is the objective KEEP signal (less = more detail
    # preserved); cast/saturation/contrast describe the grade -> only flag 'color differs'.
    color: "ColorStats" = field(default_factory=lambda: ColorStats())


@dataclass
class Record:
    """Complete fingerprint of a file. Persisted in the store."""
    path: str
    mtime: float
    size: int

    probe: Probe
    content_hash: str                     # xxhash(head ‖ mid ‖ tail) -> T0

    # identity signals
    global_vec: np.ndarray                # [D] coarse descriptor (mean of embeddings, L2)
    window_vecs: np.ndarray               # A2: [K, D] per-temporal-window descriptors (L2)
    embeddings: np.ndarray                # [N, D] per-frame (fp16 on disk; CUDA tensor in cache)
    audio_fp: np.ndarray                  # uint32 [M] (Chromaprint raw) — C1: NOT float32
    scene_cuts: np.ndarray                # scene cut timestamps [K]

    # REAL timestamp (s) of each frame in `embeddings`. Allows alignment by TIME
    # (not by index) -> robust to mixed demux/seek sampling. Empty in legacy records.
    frame_times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    quality: Quality = field(default_factory=Quality)

    @property
    def n_frames(self) -> int:
        return int(self.embeddings.shape[0]) if self.embeddings is not None else 0


@dataclass
class AlignResult:
    """Result from one aligner. `offset` in seconds (b relative to a).
    C3: offset is a property of the pair (prepended ads = offset != 0 with full coverage)."""
    score: float                          # mean similarity along the aligned path
    offset: float = 0.0
    coverage: float = 0.0                 # fraction of the shorter file that aligns
    contiguous_superset: bool = False     # one file contains the other as a contiguous segment + extra
    superset_dir: int = 0                 # C2: +1 if a⊂b, -1 if b⊂a, 0 if neither (bidirectional)
    extra_ratio: float = 0.0              # fraction of contiguous extra runtime
    # Inserted-commercials signal: extra runtime INTERLEAVED inside the aligned span (contiguous
    # unmatched blocks), NOT contiguous at the ends (that is `contiguous_superset` = a director's
    # cut). Measured: clean dups/editions/different content = 0.0; mid-roll ads = the ad fraction.
    interleaved_ratio: float = 0.0        # fraction of the LONGER file that is interleaved extra (ads)
    ad_dir: int = 0                       # +1 if b is the ad copy (longer, interleaved), -1 if a, 0 none


@dataclass
class Result:
    """Verdict between the queried record and a candidate."""
    candidate_path: str
    verdict: Verdict
    confidence: float
    reason: str                           # tier + human-readable explanation
    audio: Optional[AlignResult] = None
    video: Optional[AlignResult] = None
    scenes: Optional[AlignResult] = None
