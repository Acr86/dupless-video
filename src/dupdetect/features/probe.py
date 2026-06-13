"""ffprobe wrapper. Zero decoding — metadata only.

Parsing lives in `_parse_streams(dict)` so it can be tested with JSON fixtures
captured from real files (see tests/fixtures/). Robust to missing fields:
low-quality rips omit per-stream duration/bitrate and language tags.
"""
from __future__ import annotations

import json
import subprocess

from dupdetect.models import AudioTrack, Probe
from dupdetect.runtime import resolve_binary
from dupdetect.util import CREATE_NO_WINDOW

# "Unknown language" codes that do NOT count as a real tag. Actual language
# detection is done by quality/language.py (whisper on the audio).
_UNKNOWN_LANGS = {"", "und", "unknown", "none", "mis", "zxx"}


def ffprobe(path: str) -> Probe:
    """Reads duration, resolution, codec, bitrate, and audio tracks with their tags.
    Uses `-v error` (not `quiet`) so that if the file is unreadable (corrupt, missing
    moov atom, truncated...), the REAL error travels in the exception -> gets logged."""
    cmd = [
        resolve_binary("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                          creationflags=CREATE_NO_WINDOW)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe: {proc.stderr.strip() or 'unknown error'}")
    return _parse_streams(json.loads(proc.stdout))


def _to_int(x) -> int | None:
    """ffprobe returns almost everything as a string ('1999491'). Cast or None."""
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _to_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _lang_tag(stream: dict) -> str | None:
    """Declared language of a stream. Returns None if missing or 'und'/empty."""
    tags = stream.get("tags") or {}
    # muxers normalize to 'language', but some MKVs use uppercase
    raw = tags.get("language") or tags.get("LANGUAGE")
    if raw is None:
        return None
    raw = raw.strip().lower()
    return None if raw in _UNKNOWN_LANGS else raw


def _parse_streams(data: dict) -> Probe:
    """Maps ffprobe JSON output to Probe. Tolerates missing fields.

    - duration: `format.duration` first (Matroska does not put it in the video stream).
    - bitrate_kbps: video stream bit_rate first (the video bitrate, useful for
      cam-detection); falls back to `format.bit_rate` (container total) if missing (remux).
    - audio_tracks: ONLY streams of type 'audio' (subtitle/attachment ignored).
    """
    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audios = [s for s in streams if s.get("codec_type") == "audio"]

    # --- duration (seconds) ---
    duration_s = _to_float(fmt.get("duration"))
    if duration_s is None and video is not None:
        duration_s = _to_float(video.get("duration"))
    duration_s = duration_s or 0.0

    # --- video resolution and codec ---
    width = (_to_int(video.get("width")) if video else None) or 0
    height = (_to_int(video.get("height")) if video else None) or 0
    vcodec = (video.get("codec_name") if video else None) or ""

    # --- bitrate (kbps): video first, then container total ---
    br_bps = _to_int(video.get("bit_rate")) if video else None
    if br_bps is None:
        br_bps = _to_int(fmt.get("bit_rate"))
    bitrate_kbps = br_bps // 1000 if br_bps is not None else None

    # --- audio tracks ---
    audio_tracks = [
        AudioTrack(
            index=_to_int(s.get("index")) or 0,
            lang_tag=_lang_tag(s),
            codec=s.get("codec_name"),
            channels=_to_int(s.get("channels")),
        )
        for s in audios
    ]

    return Probe(
        duration_s=duration_s,
        width=width,
        height=height,
        vcodec=vcodec,
        bitrate_kbps=bitrate_kbps,
        audio_tracks=audio_tracks,
    )
