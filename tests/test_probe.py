"""Tests for the ffprobe parser (step 1).

Uses JSON fixtures captured from REAL library files, chosen for their
metadata diversity (not their content):
  - probe_avi_screener.json : AVI mpeg4 low-quality, audio WITH NO language tag
  - probe_mp4_dual.json      : MP4 h264, 2 audio tracks (spa/eng) + subtitles to ignore
  - probe_mkv_multiaudio.json: MKV hevc 4K remux, 15 audio tracks, duration/bitrate only in format
Plus synthetic cases for missing fields.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dupdetect.features.probe import _parse_streams
from dupdetect.models import Probe

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- real fixtures

def test_avi_screener_low_quality():
    """AVI mpeg4: bitrate from the video stream, audio with no language tag -> None."""
    p = _parse_streams(_load("probe_avi_screener.json"))
    assert isinstance(p, Probe)
    assert p.duration_s == pytest.approx(5332.16)
    assert (p.width, p.height) == (820, 480)
    assert p.vcodec == "mpeg4"
    assert p.bitrate_kbps == 1558            # from the video stream's bit_rate
    assert len(p.audio_tracks) == 1
    t = p.audio_tracks[0]
    assert t.lang_tag is None                # low-quality case: no tags
    assert t.codec == "ac3" and t.channels == 2


def test_mp4_dual_filters_out_subtitles():
    """MP4 h264: 2 audio tracks (spa/eng); the 2 subtitle streams do NOT count."""
    p = _parse_streams(_load("probe_mp4_dual.json"))
    assert (p.width, p.height) == (1920, 1080)
    assert p.vcodec == "h264"
    assert p.bitrate_kbps == 1999
    assert len(p.audio_tracks) == 2          # NOT 4 (there were 2 subtitles)
    assert [t.lang_tag for t in p.audio_tracks] == ["spa", "eng"]
    assert all(t.codec == "aac" for t in p.audio_tracks)


def test_mkv_remux_15_audio_tracks():
    """MKV remux: duration/bitrate live in `format` (the video stream omits them).
    15 audio tracks, 32 subtitles ignored."""
    p = _parse_streams(_load("probe_mkv_multiaudio.json"))
    assert p.duration_s == pytest.approx(8297.504)
    assert (p.width, p.height) == (3840, 2160)
    assert p.vcodec == "hevc"
    assert p.bitrate_kbps == 78901           # fallback to format.bit_rate (total)
    assert len(p.audio_tracks) == 15
    langs = {t.lang_tag for t in p.audio_tracks}
    assert {"eng", "spa", "fre", "ger", "ita"}.issubset(langs)
    assert all(t.codec for t in p.audio_tracks)   # all tracks have a codec


# ---------------------------------------------------------------- missing fields

def test_empty_json_does_not_crash():
    p = _parse_streams({})
    assert p.duration_s == 0.0
    assert (p.width, p.height) == (0, 0)
    assert p.vcodec == ""
    assert p.bitrate_kbps is None
    assert p.audio_tracks == []


def test_no_video_stream():
    data = {"streams": [{"codec_type": "audio", "codec_name": "mp3", "index": 0}],
            "format": {"duration": "120.5"}}
    p = _parse_streams(data)
    assert p.duration_s == pytest.approx(120.5)
    assert (p.width, p.height) == (0, 0)
    assert len(p.audio_tracks) == 1
    assert p.audio_tracks[0].lang_tag is None        # no tags
    assert p.audio_tracks[0].channels is None        # absent -> None


def test_und_language_tag_treated_as_none():
    data = {"streams": [
        {"codec_type": "audio", "index": 0, "tags": {"language": "und"}},
        {"codec_type": "audio", "index": 1, "tags": {"language": "ENG"}},  # uppercase
    ]}
    p = _parse_streams(data)
    assert p.audio_tracks[0].lang_tag is None
    assert p.audio_tracks[1].lang_tag == "eng"       # normalised to lowercase


def test_duration_falls_back_to_stream_when_missing_from_format():
    data = {"streams": [{"codec_type": "video", "codec_name": "h264",
                         "width": 640, "height": 360, "duration": "99.0"}],
            "format": {}}
    p = _parse_streams(data)
    assert p.duration_s == pytest.approx(99.0)
