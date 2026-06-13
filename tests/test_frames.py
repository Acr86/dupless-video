"""Tests for decode/sampling (step 2).

Generates a synthetic video with PyAV (portable, no dependency on the real library),
decodes it, and validates shape, dtype, device, and ImageNet normalization.
"""
from __future__ import annotations

import numpy as np
import pytest

from dupdetect.features.frames import (
    DEFAULT_SIZE,
    DINOV2_MEAN,
    DINOV2_STD,
    decode_frames,
    sample_timestamps,
)

av = pytest.importorskip("av")
torch = pytest.importorskip("torch")


def _make_video(path, n_frames: int, fps: int, w: int = 256, h: int = 256) -> None:
    """Encodes `n_frames` each with a distinct color (frame 0 = black).
    Short GOP -> multiple keyframes (decode samples by keyframes)."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.gop_size = 5                                            # keyframe every ~5 frames
    for i in range(n_frames):
        arr = np.full((h, w, 3), (i * 5) % 256, dtype=np.uint8)   # i=0 -> black
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():                                # flush encoder
        container.mux(packet)
    container.close()


# --------------------------------------------------------------- sample_timestamps

def test_sample_timestamps_dense():
    ts = sample_timestamps(10.0, 2.0)
    assert len(ts) == 20
    assert ts[0] == 0.0 and ts[-1] < 10.0


def test_sample_timestamps_minimum_one_frame():
    assert len(sample_timestamps(0.0, 2.0)) == 1


# --------------------------------------------------------------- decode_frames

@pytest.fixture
def video(tmp_path):
    p = tmp_path / "synthetic.mp4"
    _make_video(p, n_frames=50, fps=10)        # 5 s
    return str(p)


def test_decode_keyframes_shape_and_timestamps(video):
    x, times, color = decode_frames(video)      # keyframe sampling + color stats (reused frames)
    assert x.ndim == 4
    assert x.shape[1:] == (3, DEFAULT_SIZE, DEFAULT_SIZE)
    assert x.shape[0] >= 1                       # at least one keyframe (frame 0)
    assert times.shape[0] == x.shape[0]          # one timestamp per frame
    assert x.dtype == torch.float32
    from dupdetect.quality.color import ColorStats
    assert isinstance(color, ColorStats) and 0.0 <= color.clip <= 1.0
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert str(x.device).startswith(expected)


def test_decode_imagenet_normalization(video):
    # frame 0 is black (keyframe): after ImageNet norm the R channel drops below -2.
    x, _, _ = decode_frames(video)
    assert x.min().item() < -2.0                       # ImageNet normalization applied


def test_decode_unreadable_file_raises(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video" * 50)
    with pytest.raises(RuntimeError):           # _pass1 catches it -> problems table
        decode_frames(str(bad))


# --------------------------------------------------------------- seek-based sampling (4K/8K)

def test_seek_keyframes_samples_and_sorts(video):
    """The seek sampler jumps to N timestamps and returns uint8 keyframes + ascending ts."""
    from dupdetect.features.frames import _seek_keyframes
    frames, ts = _seek_keyframes(video, DEFAULT_SIZE, n=8)
    assert frames.ndim == 4 and frames.shape[1:] == (DEFAULT_SIZE, DEFAULT_SIZE, 3)
    assert frames.dtype == np.uint8
    assert ts.shape[0] == frames.shape[0] >= 1
    assert list(ts) == sorted(ts)                       # ascending timestamps (temporal order)
    assert len(np.unique(ts)) == len(ts)                # dedup: no repeated keyframe


def test_decode_frames_forced_seek(video):
    """seek_threshold_bytes=0 forces the seek path; returns the SAME normalized tensor
    type as demux (the rest of the pipeline does not notice the difference)."""
    x, times, _ = decode_frames(video, seek_threshold_bytes=0, seek_n=8)
    assert x.ndim == 4 and x.shape[1:] == (3, DEFAULT_SIZE, DEFAULT_SIZE)
    assert x.shape[0] == times.shape[0] >= 1
    assert x.dtype == torch.float32


def test_decode_frames_demux_when_small(video):
    """Below the threshold (small file) uses demux: nothing breaks."""
    x, times, _ = decode_frames(video, seek_threshold_bytes=10 ** 12)   # very high threshold -> demux
    assert x.shape[0] == times.shape[0] >= 1
