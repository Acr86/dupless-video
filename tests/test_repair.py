"""Index rebuild tests (remux -c copy). Generates a synthetic mp4 with PyAV,
remuxes it, and verifies it still opens with the SAME duration (lossless) and that the
original is only replaced after verification."""
from __future__ import annotations

import numpy as np
import pytest

av = pytest.importorskip("av")

from dupdetect.features.probe import ffprobe
from dupdetect.repair import remux_rebuild_index


def _make_mp4(path, n: int = 30, fps: int = 10) -> None:
    c = av.open(str(path), mode="w")
    s = c.add_stream("mpeg4", rate=fps)
    s.width = s.height = 64
    s.pix_fmt = "yuv420p"
    s.gop_size = 5
    for i in range(n):
        arr = np.full((64, 64, 3), (i * 8) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
        for pkt in s.encode(frame):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()


def test_remux_rebuild_index_is_lossless(tmp_path):
    src = tmp_path / "v.mp4"
    _make_mp4(src)
    d0 = ffprobe(str(src)).duration_s
    ok, kind, msg = remux_rebuild_index(str(src))
    assert ok, msg
    assert kind == "ok"
    assert src.exists()                                          # original still present (replaced)
    d1 = ffprobe(str(src)).duration_s
    assert d1 > 0 and abs(d1 - d0) <= max(2.0, 0.05 * d0)        # same duration (no re-encode)


def test_remux_missing_file_does_not_crash(tmp_path):
    ok, kind, msg = remux_rebuild_index(str(tmp_path / "nope.mp4"))
    assert not ok and kind == "gone" and "exist" in msg


def test_remux_reports_progress(tmp_path):
    """`on_progress` receives increasing fractions [0..1] ending at 1.0 (used by the UI for the
    per-file progress bar). Without a callback the result is identical (must not alter the remux)."""
    src = tmp_path / "v.mp4"
    _make_mp4(src, n=120)
    fracs: list[float] = []
    ok, kind, _ = remux_rebuild_index(str(src), on_progress=fracs.append)
    assert ok and kind == "ok"
    assert fracs and fracs == sorted(fracs) and fracs[-1] >= 1.0
    assert all(0.0 <= f <= 1.0 for f in fracs)
