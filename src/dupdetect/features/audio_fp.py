"""Audio fingerprint (Chromaprint/fpcalc). Robust to re-encode/resolution changes.
Fails on genuinely different audio (different language) and cam rips (room noise)."""
from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile

import numpy as np

from dupdetect.runtime import resolve_binary

# C4: bump if fpcalc/Chromaprint parameters change (algorithm or sample
# rate). feature_version incorporates it -> invalidates old audio_fp cache.
AUDIO_FP_VERSION = 1

# C4: incorporated into feature_version. DELIBERATELY NOT bumped for the v2 probe (5s windows,
# duration-scaled points, all-streams): a bump would invalidate feature_version for EVERY file and
# force a full re-decode+embed of the library for an audio-only change. The rollout is handled by a
# targeted store migration instead (store._init_schema, PRAGMA user_version): only rows the v1 probe
# FLAGGED lose their cache and re-measure lazily with v2.
COVERAGE_VERSION = 1

# Chromaprint emits ~8.08 items/s (sample 11025 Hz, hop 1365). Used to convert offset to s.
ITEM_RATE_HZ = 8.0

# Audio coverage (fingerprint vs video duration) BELOW which we warn: the copy
# has missing or truncated audio past a certain second. Generous (avoid crying wolf): only
# flags real gaps (clipped trailing silence, etc. don't count).
AUDIO_OK_COVERAGE = 0.85

# Copies whose audio coverage is within this band are treated as the SAME audio (same source),
# even if both are truncated -> audio is not a differentiator and must NOT block auto-KEEP / force
# review. Only a coverage DIFFERENCE above this means one copy really has better audio than another.
AUDIO_COV_TOL = 0.05


def audio_coverage(n_items: int, duration_s: float, max_length_s: int = 0) -> float:
    """Fraction [0..1] of the video that the audio (fingerprint) covers. 1.0 = full audio;
    0.0 = no audio; intermediate = audio truncated at ~(cov·duration). Measures CONTENT (number of
    items ÷ ITEM_RATE_HZ ≈ seconds of audio), not metadata (§0). Respects the opt-in cap
    `max_length_s` (if the fingerprint was intentionally capped, the reference is the cap, not the duration).
    Returns 1.0 when it cannot be judged (unknown duration) -> no warning generated."""
    if duration_s <= 0:
        return 1.0
    expected_s = min(duration_s, max_length_s) if max_length_s and max_length_s > 0 else duration_s
    if expected_s <= 0:
        return 1.0
    decoded_s = n_items / ITEM_RATE_HZ
    return max(0.0, min(1.0, decoded_s / expected_s))


def _fpcalc_bin() -> str:
    """fpcalc path via the platform seam (bundle -> $FPCALC/$DUPDETECT_FPCALC -> venv -> PATH).
    Resolution lives in runtime.resolve_binary so every external binary is found the same way."""
    return resolve_binary("fpcalc")


def audio_fingerprint(path: str, max_length_s: int = 0,
                      timeout: float | None = None) -> np.ndarray:
    """FULL fingerprint sequence (not a single hash) -> allows later offset alignment
    (catching 'glued ads'). `-length 0` = whole file; `>0` caps it
    (less disk read on giants, but limits the detectable offset range).

    Robust (a healthy video is NEVER discarded due to audio, §2):
      - fpcalc direct with exit 0 -> complete fingerprint (fast path, most cases).
      - exit≠0 (bad audio frame mid-file -> PARTIAL fingerprint, or codec that its internal
        libav doesn't support -> empty fingerprint) -> retried with the SYSTEM ffmpeg (more
        tolerant, more codecs) and the MOST COMPLETE fingerprint (most items) is kept. This
        avoids truncated fingerprints (worse matching) and FALSE 'audio truncated' warnings.
        Only if EVEN THAT yields no decodable audio does it stay empty (non-fatal: video is
        still indexed by embeddings).

    `timeout`: aborts if fpcalc/ffmpeg take too long (damaged index/seek) -> the error carries
    'timeout' so the store classifies it as 'reindex' (fixable with remux).

    C1: dtype uint32 (raw Chromaprint items are unsigned 32-bit integers;
    e.g. 3438471587 > 2^31). Never float -> store cast would truncate bits.
    """
    rc, out, _err = _run_fpcalc(
        [_fpcalc_bin(), "-raw", "-length", str(int(max_length_s)), path], timeout)
    fp = _parse_fpcalc(out)
    if rc == 0:
        return fp                                   # fpcalc decoded CLEANLY to the end
    # rc≠0 -> fpcalc had problems: either a bad audio frame mid-file (PARTIAL fingerprint: using
    # it as-is UNDERESTIMATES audio and triggers false 'audio truncated' warnings), or a codec
    # its internal libav doesn't support (empty fingerprint). The SYSTEM ffmpeg is more tolerant
    # and supports more codecs: retry and keep the MOST COMPLETE fingerprint (most items). Only
    # if even that yields no decodable audio does it stay empty (non-fatal: video indexed by embeddings).
    fp_ff = _fingerprint_via_ffmpeg(path, max_length_s, timeout)
    return fp_ff if fp_ff.size > fp.size else fp


def _run_fpcalc(cmd: list[str], timeout: float | None) -> tuple[int, str, str]:
    """Runs fpcalc. Converts timeout into a RuntimeError with 'timeout' (-> 'reindex' in the
    store). Does NOT use check=True: exit≠0 with a fingerprint present is valid (isolated bad frame)."""
    from dupdetect.util import CREATE_NO_WINDOW
    try:
        proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                              creationflags=CREATE_NO_WINDOW, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"audio_fp: timeout (>{timeout:.0f}s) — fpcalc took too long "
            f"(damaged index/audio?)") from e
    return proc.returncode, proc.stdout, proc.stderr


def _fingerprint_via_ffmpeg(path: str, max_length_s: int, timeout: float | None) -> np.ndarray:
    """Fallback: decodes audio with the SYSTEM ffmpeg to PCM mono 11025 (same
    sample rate Chromaprint uses -> fingerprint COMPATIBLE with the direct path) and pipes
    it to fpcalc. Recovers codecs that fpcalc's decoder doesn't support (adpcm_ms, etc.).
    Returns an empty fingerprint ONLY if there is no decodable audio track (-> non-fatal: video
    is still indexed and detected by embeddings)."""
    from dupdetect.util import CREATE_NO_WINDOW
    fd, wav = tempfile.mkstemp(suffix=".wav", prefix="dupfp_")
    os.close(fd)
    try:
        cmd = ["ffmpeg", "-v", "error", "-y"]
        if max_length_s and max_length_s > 0:
            cmd += ["-t", str(int(max_length_s))]   # cap the decode (input) on giants
        cmd += ["-i", path, "-vn", "-ac", "1", "-ar", "11025", "-c:a", "pcm_s16le", wav]
        try:
            proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                                  creationflags=CREATE_NO_WINDOW, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"audio_fp: timeout (>{timeout:.0f}s) — ffmpeg (audio) took too long "
                f"(damaged index/audio?)") from e
        if proc.returncode != 0 or _safe_size(wav) == 0:
            return np.empty(0, dtype=np.uint32)     # no decodable audio -> no fingerprint
        _rc, out, _err = _run_fpcalc(
            [_fpcalc_bin(), "-raw", "-length", str(int(max_length_s)), wav], timeout)
        return _parse_fpcalc(out)
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass


_MEANVOL_RE = re.compile(r"mean_volume:\s*(-?[0-9.]+|-inf) dB")

# v2 probe geometry. WINDOW: 0.5s windows fell inside dialogue pauses / quiet ambient scenes and
# read as "no audio" (mean < silence_db) — with the 12-point KEEP muted-check, TWO genuinely quiet
# windows flagged a healthy film. 5s averages across the pause (music/ambience lifts the mean);
# the probe cost is SEEK-dominated on spinning disks (§1), so decoding 5s instead of 0.5s per
# probe is nearly free. DENSITY: fixed 40 points over-probes a 22-min episode and under-probes a
# 3h movie; one probe per ~2 min of runtime (clamped) keeps the granularity uniform per TITLE.
COVERAGE_WIN_S = 5.0
COVERAGE_PROBE_EVERY_S = 120.0
COVERAGE_POINTS_MIN = 8
COVERAGE_POINTS_MAX = 48


def coverage_points(duration_s: float) -> int:
    """Duration-scaled probe count: one probe per COVERAGE_PROBE_EVERY_S of runtime, clamped to
    [MIN, MAX]. A pure function of the ffprobe duration -> same input, same probe positions (§0:
    deterministic, no RNG). 10min->8, 30min->15, 90min->45, >=96min->48."""
    if duration_s <= 0:
        return COVERAGE_POINTS_MIN
    return int(min(COVERAGE_POINTS_MAX,
                   max(COVERAGE_POINTS_MIN, math.ceil(duration_s / COVERAGE_PROBE_EVERY_S))))


def scan_audio_coverage(path: str, duration_s: float, n_points: int | None = None,
                        win_s: float = COVERAGE_WIN_S, silence_db: float = -60.0,
                        per_probe_timeout_s: float | None = 20.0) -> float:
    """Cheap WHOLE-FILE audio coverage WITHOUT reading the whole file (vs the full
    fingerprint, ~44x less I/O measured on a 54GB REMUX). Decodes a `win_s` window at
    `n_points` timestamps spread across `duration_s` (sparse SEEK; None -> duration-scaled, see
    coverage_points), measures mean volume, and returns the fraction of probes with audio above
    `silence_db`.

    ALL audio streams are probed in ONE pass (`-map 0:a?` — volumedetect instantiates per mapped
    stream and emits one mean_volume each, verified against the bundled ffmpeg): a probe counts as
    with-audio if ANY stream is above `silence_db`. The v1 first-stream-only probe (`0:a:0?`)
    false-flagged files whose track 0 is a commentary / low-level secondary language while the real
    audio lives on track 1. Same number of seeks — the real cost on a spinning disk (§1).

    Detects audio that drops out ANYWHERE (e.g. a copy that goes silent after 15 min, so the
    user doesn't keep the muted one). Measures CONTENT, not metadata (§0). Probes that time out /
    can't be judged are skipped (not counted as silent); returns 1.0 when nothing can be judged
    -> no false warning. Thresholds (AUDIO_OK_COVERAGE, silence_db) are unchanged from v1 (§0:
    never loosened)."""
    if n_points is None:
        n_points = coverage_points(duration_s)
    if duration_s <= 0 or n_points <= 0:
        return 1.0
    from dupdetect.util import CREATE_NO_WINDOW
    bin_ = _ffmpeg_bin()
    judged = 0
    with_audio = 0
    for i in range(n_points):
        ts = duration_s * (i + 0.5) / n_points         # centered, evenly spread across the file
        cmd = [bin_, "-nostdin", "-ss", f"{ts:.2f}", "-t", f"{win_s}", "-i", path,
               "-map", "0:a?", "-af", "volumedetect", "-f", "null", os.devnull]
        try:
            proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                                  creationflags=CREATE_NO_WINDOW, timeout=per_probe_timeout_s)
        except (subprocess.TimeoutExpired, OSError):
            continue                                    # not judged (broken index / hang)
        vols = _MEANVOL_RE.findall(proc.stderr or "")   # one mean_volume PER mapped audio stream
        if not vols:
            continue                                    # no audio stream at this point / unparseable
        judged += 1
        if any(v != "-inf" and float(v) > silence_db for v in vols):
            with_audio += 1                             # ANY stream with audio -> the probe has audio
    return (with_audio / judged) if judged else 1.0


def _ffmpeg_bin() -> str:
    """ffmpeg path via the platform seam (bundle -> $DUPDETECT_FFMPEG -> venv -> PATH)."""
    return resolve_binary("ffmpeg")


def _safe_size(p: str) -> int:
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def _parse_fpcalc(out: str) -> np.ndarray:
    """Extracts the FINGERPRINT=a,b,c,... line as a uint32 array. Testable without fpcalc."""
    for line in out.splitlines():
        if line.startswith("FINGERPRINT="):
            body = line[len("FINGERPRINT="):].strip()
            if not body:
                break
            return np.array([int(v) for v in body.split(",")], dtype=np.uint32)
    return np.empty(0, dtype=np.uint32)
