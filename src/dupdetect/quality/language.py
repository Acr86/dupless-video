"""Detects the REAL audio language (NEVER from the container tag, which is often wrong
or outright lies). Language decides the cluster "keep" (retain the copy in the desired
language), so an error here can delete the correct copy.

VAD by regions + multi-window + VOTE (adapted from L:\\Work\\Code\\Video\\SpeechAnalyzer,
using only faster-whisper — no whisperx, so it runs on Python 3.14):

  - A single 30s window fails when it lands on music/silence/ads: whisper guesses
    (detected 'kor' in English films). Blind sampling also fails: with sparse dialogue
    blind windows fall on NON-speech and vote garbage.
  - Instead, at 3 positions spread across the film a REGION of 120s is decoded (-ss
    cut, cheap) and Silero VAD locates actual SPEECH inside; language is detected from
    that speech window (faster-whisper, detect only, no transcription) and VOTED by
    probability sum. Adaptive: if confidence is low, adds more regions.
  - VAD by regions instead of over the full audio: VAD over 49 min costs ~5s; over
    3×120s, ~0.6s. Same robustness, ~5x cheaper.

M3: runs on CPU (int8) in the extract_cpu_features workers (does not cross CUDA context).
"""
from __future__ import annotations

import subprocess
from typing import Optional

import numpy as np

from dupdetect.util import CREATE_NO_WINDOW

SAMPLE_RATE = 16000

# Whisper returns ISO 639-1 (2 letters); the rest of the system uses ISO 639-2/B
# (3 letters, like ffmpeg tags and wanted_langs). Map of the most common ones.
_ISO1_TO_ISO2B = {
    "es": "spa", "en": "eng", "fr": "fre", "de": "ger", "it": "ita", "pt": "por",
    "ru": "rus", "ja": "jpn", "zh": "chi", "ko": "kor", "ar": "ara", "nl": "dut",
    "pl": "pol", "tr": "tur", "sv": "swe", "cs": "cze", "hu": "hun", "hi": "hin",
    "ca": "cat", "el": "gre", "he": "heb", "th": "tha", "da": "dan", "fi": "fin",
    "no": "nor", "ro": "rum", "uk": "ukr", "vi": "vie", "id": "ind",
}

_MODEL_CACHE: dict = {}

REGION_S = 120.0          # large window decoded per position (-ss, cheap)
WINDOW_S = 30.0           # SPEECH chunk detected inside the region
_INITIAL_FRACS = (0.25, 0.5, 0.75)
_EXTRA_FRACS = (0.4, 0.6, 0.15, 0.85)


def _media_duration(path: str) -> float:
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
           "-of", "csv=p=0", path]
    try:
        return float(subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                                    check=True, creationflags=CREATE_NO_WINDOW).stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def _extract_audio(path: str, sample_s: float, start_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """`sample_s` of mono audio at `sr` (float32) starting at `start_s`, via ffmpeg (fast -ss)."""
    cmd = ["ffmpeg", "-v", "quiet", "-ss", str(start_s), "-t", str(sample_s), "-i", path,
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=False,
                         creationflags=CREATE_NO_WINDOW).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _get_model(model: str, device: str):
    from faster_whisper import WhisperModel

    key = (model, device)
    if key not in _MODEL_CACHE:
        compute = "float16" if device == "cuda" else "int8"
        _MODEL_CACHE[key] = WhisperModel(model, device=device, compute_type=compute)
    return _MODEL_CACHE[key]


def _vad_segments(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list[tuple[float, float]]:
    """Speech segments (start_s, end_s) within `audio`, via Silero VAD (bundled in
    faster-whisper). Returns [] if no speech or VAD is unavailable."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except Exception:                                   # VAD unavailable -> no segments
        return []
    opts = VadOptions(min_speech_duration_ms=250, min_silence_duration_ms=500)
    try:
        ts = get_speech_timestamps(audio, vad_options=opts)
    except TypeError:                                   # older versions: positional
        ts = get_speech_timestamps(audio, opts)
    except Exception:                                   # VAD crashed -> no segments
        return []
    return [(t["start"] / sr, t["end"] / sr) for t in ts]


def _speech_window(region: np.ndarray) -> np.ndarray:
    """A chunk of up to WINDOW_S of SPEECH within `region` (16k audio). Uses the longest
    speech segment found by VAD; falls back to the start of the region if none (or no VAD)."""
    segs = _vad_segments(region)
    if not segs:
        return region[: int(WINDOW_S * SAMPLE_RATE)]
    s0, e0 = max(segs, key=lambda x: x[1] - x[0])       # longest speech segment
    start = int(s0 * SAMPLE_RATE)
    end = min(start + int(WINDOW_S * SAMPLE_RATE), int(e0 * SAMPLE_RATE), region.size)
    return region[start:end]


def _detect_one(audio: np.ndarray, model) -> tuple[str, float]:
    """(lang_iso1, prob) for an audio chunk. Tolerant of detect_language signature changes
    across faster-whisper versions."""
    if audio.size < SAMPLE_RATE:                        # <1s: insufficient for detection
        return "unknown", 0.0
    try:
        res = model.detect_language(audio)
        if isinstance(res, tuple) and len(res) >= 2:
            return res[0], float(res[1])
        if isinstance(res, dict):
            return res.get("language", "unknown"), float(
                res.get("probability", res.get("language_probability", 0.0)))
    except Exception:                                   # bad window -> don't abort detection
        pass
    return "unknown", 0.0


def _vote(results: list[tuple[str, float]]) -> tuple[str, float]:
    """The language with the highest PROBABILITY SUM wins; confidence = its mean prob. This way
    a bad window (music -> 'kor' with low prob) cannot beat real speech windows."""
    scores: dict[str, list[float]] = {}
    for lang, prob in results:
        if lang != "unknown":
            scores.setdefault(lang, []).append(prob)
    if not scores:
        return "unknown", 0.0
    winner = max(scores, key=lambda lang: sum(scores[lang]))
    return winner, sum(scores[winner]) / len(scores[winner])


def _region_starts(dur: float, fracs: tuple[float, ...]) -> list[float]:
    """start_s for each REGION_S region, centered at frac*dur and clamped to the file."""
    if dur <= REGION_S:
        return [0.0]
    return [min(max(0.0, dur * f - REGION_S / 2), dur - REGION_S) for f in fracs]


def detect_language(path: str, model: str = "base", device: str = "cpu",
                    n_extra: int = 2, min_confidence: float = 0.6) -> Optional[str]:
    """Real audio language (ISO 639-2/B: 'spa','eng','rus',...) or None if no audio.

    VAD by regions + multi-window vote (see module). Robust to dialogue-free intros,
    sparse dialogue, and a single misclassified window (the 'kor' bug in English films).
    """
    dur = _media_duration(path)
    fw = _get_model(model, device)

    def sample(fracs):
        out = []
        for s in _region_starts(dur, fracs):
            region = _extract_audio(path, REGION_S, s)
            if region.size:
                out.append(_detect_one(_speech_window(region), fw))
        return out

    results = sample(_INITIAL_FRACS)
    if not results:
        return None
    lang, conf = _vote(results)

    if conf < min_confidence and n_extra > 0:           # uncertain -> add regions and re-vote
        results.extend(sample(_EXTRA_FRACS[:n_extra]))
        lang, conf = _vote(results)

    return None if lang == "unknown" else _ISO1_TO_ISO2B.get(lang, lang)
