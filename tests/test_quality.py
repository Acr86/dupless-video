"""Quality tests (step 8): cam_score (bitrate + sharpness) and language (ISO mapping).

cam_score with bitrate is pure (Probe only). The sharpness part uses torch.
Real detect_language requires ffmpeg + whisper model -> covered in a separate demo; here
we test the ISO mapping and extraction logic.
"""
from __future__ import annotations

import numpy as np
import pytest

from dupdetect.models import AudioTrack, Probe
from dupdetect.quality.camrip import TARGET_BPP, cam_score

torch = pytest.importorskip("torch")


def _probe(bitrate_kbps, w=1920, h=1080):
    return Probe(duration_s=6000.0, width=w, height=h, vcodec="h264",
                 bitrate_kbps=bitrate_kbps, audio_tracks=[])


# --------------------------------------------------------------- cam_score (bitrate)

def test_healthy_bitrate_low_cam():
    # 1080p at ~9 Mbps -> bpp ~0.18 > target -> cam 0
    assert cam_score(_probe(9000)) == pytest.approx(0.0, abs=1e-6)


def test_poor_bitrate_high_cam():
    # 1080p at ~400 kbps -> bpp tiny (starved) -> high cam (TARGET_BPP=0.04)
    assert cam_score(_probe(400)) > 0.7


def test_no_bitrate_returns_zero():
    assert cam_score(_probe(None)) == 0.0


def test_intermediate_bitrate_monotonic():
    # more bitrate => lower cam (monotonic), in the starved zone <2 Mbps at 1080p
    assert cam_score(_probe(500)) > cam_score(_probe(1000)) > cam_score(_probe(1500))


# --------------------------------------------------------------- cam_score (sharpness)

def test_sharp_frames_lower_cam():
    probe = _probe(1000)                                  # bad bitrate -> high cam alone
    base = cam_score(probe)
    # VERY sharp frames (high-freq noise) -> high Laplacian variance -> cam~0
    sharp = torch.randn(4, 3, 64, 64)
    combined = cam_score(probe, frames=sharp)
    assert combined < base                                # sharpness averages the score down


def test_flat_frames_do_not_lower_much():
    probe = _probe(1000)
    flat = torch.zeros(4, 3, 64, 64)                      # flat image -> no sharpness -> high cam
    s = cam_score(probe, frames=flat)
    assert s > 0.7


# --------------------------------------------------------------- ISO language mapping

def test_iso_639_2_mapping():
    from dupdetect.quality.language import _ISO1_TO_ISO2B
    assert _ISO1_TO_ISO2B["es"] == "spa"
    assert _ISO1_TO_ISO2B["en"] == "eng"
    assert _ISO1_TO_ISO2B["ru"] == "rus"


def test_audio_track_dataclass_still_valid():
    t = AudioTrack(index=1, lang_tag="spa", codec="ac3", channels=6)
    assert t.lang_tag == "spa" and t.channels == 6


# --------------------------------------------------- language detection: multi-window vote

def test_vote_sums_probabilities_speech_wins():
    """The 'kor' bug: one music window classifies 'kor' with low prob; real speech windows
    classify 'eng' with high prob. Sum by language must yield 'eng'."""
    from dupdetect.quality.language import _vote
    res = [("en", 0.85), ("ko", 0.30), ("en", 0.80)]      # 2 English windows, 1 Korean noise
    lang, conf = _vote(res)
    assert lang == "en"
    assert conf == pytest.approx(0.825, abs=1e-3)          # mean of the 'en' windows


def test_vote_ignores_unknown_and_empty():
    from dupdetect.quality.language import _vote
    assert _vote([("unknown", 0.0), ("unknown", 0.0)]) == ("unknown", 0.0)
    assert _vote([]) == ("unknown", 0.0)


def test_region_starts_distributes_and_clamps():
    from dupdetect.quality.language import REGION_S, _region_starts
    # long film: 3 regions centred at 25/50/75%, all within [0, dur-REGION_S]
    dur = 3000.0
    starts = _region_starts(dur, (0.25, 0.5, 0.75))
    assert len(starts) == 3
    assert all(0.0 <= s <= dur - REGION_S for s in starts)
    assert starts == sorted(starts)                        # ascending
    # film shorter than one region -> single window from 0
    assert _region_starts(REGION_S - 10, (0.25, 0.5, 0.75)) == [0.0]


def test_speech_window_no_vad_falls_back_to_start(monkeypatch):
    """If VAD finds no speech (or is absent), use the region start (no crash)."""
    import numpy as np

    from dupdetect.quality import language as L
    monkeypatch.setattr(L, "_vad_segments", lambda audio, sr=L.SAMPLE_RATE: [])
    region = np.ones(int(L.REGION_S * L.SAMPLE_RATE), dtype=np.float32)
    win = L._speech_window(region)
    assert win.size == int(L.WINDOW_S * L.SAMPLE_RATE)     # WINDOW_S from the start


def test_speech_window_clips_to_longest_segment(monkeypatch):
    import numpy as np

    from dupdetect.quality import language as L
    # VAD reports two segments; longest wins (10-40s) -> window starts at 10s
    monkeypatch.setattr(L, "_vad_segments",
                        lambda audio, sr=L.SAMPLE_RATE: [(1.0, 3.0), (10.0, 40.0)])
    region = np.ones(int(L.REGION_S * L.SAMPLE_RATE), dtype=np.float32)
    win = L._speech_window(region)
    assert win.size == int(L.WINDOW_S * L.SAMPLE_RATE)     # 30s from the long segment (10..40)


# --------------------------------------------------------------- color descriptor
def test_color_descriptor_detects_clipping_and_grade():
    import numpy as np
    from dupdetect.quality.color import ColorStats, color_descriptor
    clean = np.full((4, 16, 16, 3), 128, np.uint8)                  # mid-gray: no clipping
    assert color_descriptor(clean).clip < 0.01
    crushed = np.zeros((4, 16, 16, 3), np.uint8); crushed[:, :8] = 255  # half black/half white
    assert color_descriptor(crushed).clip > 0.9                    # heavy clipping (lost detail)
    warm = np.full((4, 16, 16, 3), 128, np.uint8); warm[..., 0] = 210   # red tint -> cast
    assert color_descriptor(warm).grade_distance(color_descriptor(clean)) > 0.0
    assert color_descriptor(np.empty((0, 2, 2, 3), np.uint8)) == ColorStats()   # empty -> neutral
