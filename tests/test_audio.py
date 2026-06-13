"""Audio tests (step 5): fpcalc output parsing + offset-based alignment.

_parse_fpcalc is tested with raw text (no real fpcalc). align_audio is tested
with synthetic uint32 fingerprints. Validation against a real fpcalc binary
lives in a separate demo.
"""
from __future__ import annotations

import numpy as np
import pytest

from dupdetect.align.audio import ITEM_RATE_HZ, _popcount32, align_audio
from dupdetect.features import audio_fp as afp
from dupdetect.features.audio_fp import _parse_fpcalc


# ------------------------------------------------------------- _parse_fpcalc

def test_parse_fpcalc_large_uint32_values():
    out = "DURATION=8793\nFINGERPRINT=1285749667,3438471587,4294967295,0\n"
    fp = _parse_fpcalc(out)
    assert fp.dtype == np.uint32
    # 3438471587 > 2^31 and 4294967295 = 2^32-1 must survive exactly (C1)
    assert fp.tolist() == [1285749667, 3438471587, 4294967295, 0]


def test_parse_fpcalc_empty():
    assert _parse_fpcalc("DURATION=10\nFINGERPRINT=\n").shape == (0,)
    assert _parse_fpcalc("no fingerprint here").shape == (0,)


# ------------------------------------------------- audio_fingerprint: exit≠0

def _fake_run(returncode, stdout, stderr=""):
    """Faithfully emulates subprocess.run: with check=True and returncode≠0 it raises
    CalledProcessError (just like the real thing). This ensures the fix does NOT use
    check=True but instead inspects returncode + fingerprint on its own."""
    import subprocess as sp

    def run(*a, **k):
        if k.get("check") and returncode != 0:
            raise sp.CalledProcessError(returncode, a[0] if a else "fpcalc",
                                        output=stdout, stderr=stderr)
        return sp.CompletedProcess(args=["fpcalc"], returncode=returncode,
                                   stdout=stdout, stderr=stderr)
    return run


def _cp(rc, out, err=""):
    import subprocess as sp
    return sp.CompletedProcess(args=["x"], returncode=rc, stdout=out, stderr=err)


# ----------------------------------------------------- scan_audio_coverage (cheap, whole-file)

def test_scan_audio_coverage_detects_midfile_dropout(monkeypatch):
    """Seek-sampled coverage: audio present in the first half, silent afterwards -> ~0.5, so a
    copy that loses audio partway IS flagged (the user must not keep the muted one)."""
    calls = {"i": 0}

    def fake_run(cmd, **kw):
        i = calls["i"]; calls["i"] += 1
        return _cp(0, "", f"mean_volume: {'-30.0' if i < 20 else '-inf'} dB")  # 20 audio, 20 silent

    monkeypatch.setattr(afp.subprocess, "run", fake_run)
    cov = afp.scan_audio_coverage("x.mkv", duration_s=7200, n_points=40)
    assert 0.45 <= cov <= 0.55


def test_scan_audio_coverage_all_silent_is_zero(monkeypatch):
    monkeypatch.setattr(afp.subprocess, "run",
                        lambda cmd, **kw: _cp(0, "", "mean_volume: -inf dB"))
    assert afp.scan_audio_coverage("x.mkv", duration_s=3600, n_points=10) == 0.0


def test_scan_audio_coverage_unknown_duration_no_warning(monkeypatch):
    # duration <=0 cannot be judged -> 1.0 (no false warning, §0)
    assert afp.scan_audio_coverage("x.mkv", duration_s=0, n_points=40) == 1.0


def test_scan_audio_coverage_skips_failed_probes(monkeypatch):
    # probes that hang/raise are NOT counted as silent; here all fail -> nothing judged -> 1.0
    import subprocess as sp

    def boom(cmd, **kw):
        raise sp.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(afp.subprocess, "run", boom)
    assert afp.scan_audio_coverage("x.mkv", duration_s=3600, n_points=8) == 1.0


def test_audio_fingerprint_nonzero_exit_with_fingerprint_is_used(monkeypatch):
    """fpcalc may exit with nonzero when it hits ONE bad audio frame inside a long file
    (`-length 0` on an AVI) yet still emit a usable FINGERPRINT. The video is healthy
    (ffprobe can open it) -> the partial fingerprint is used, NOT sent to `problems`."""
    monkeypatch.setattr(afp.subprocess, "run", _fake_run(
        3, "DURATION=6794\nFINGERPRINT=1398828501,325090679\n",
        "ERROR: Error decoding audio frame (Invalid data found when processing input)"))
    fp = afp.audio_fingerprint("anything.avi")
    assert fp.tolist() == [1398828501, 325090679]


def test_audio_fingerprint_ffmpeg_fallback_recovers_unsupported_codec(monkeypatch):
    """Direct fpcalc cannot decode the codec (e.g. adpcm_ms: 'Could not find any audio
    stream'), but the audio IS readable -> decoded via the system ffmpeg and the
    fingerprint is recovered. (Previously this sent a healthy video to 'corrupt'.)"""
    import os
    calls = []

    def fake_run(cmd, *a, **k):
        prog = os.path.basename(str(cmd[0])).lower()
        calls.append(prog)
        if "ffmpeg" in prog:                         # decode: write the wav destination
            with open(cmd[-1], "wb") as f:
                f.write(b"RIFF\x00\x00\x00\x00WAVE")
            return _cp(0, "")
        target = str(cmd[-1])                         # fpcalc
        if target.endswith(".wav"):                   # on the fallback PCM -> fingerprint
            return _cp(0, "DURATION=10\nFINGERPRINT=11,22,33\n")
        return _cp(2, "", "ERROR: Could not find any audio stream in the file (Decoder)")
    monkeypatch.setattr(afp.subprocess, "run", fake_run)
    fp = afp.audio_fingerprint("movie.avi")
    assert fp.tolist() == [11, 22, 33]
    assert any("ffmpeg" in c for c in calls)          # used the fallback


def test_audio_coverage_detects_truncation_and_no_audio():
    """Coverage = num items/8 / duration (measures content, not metadata). Full ~1.0; truncated
    to 120s of 1430s ~0.08; no audio 0.0; unknown duration 1.0 (no warning); respects the cap."""
    from dupdetect.features.audio_fp import AUDIO_OK_COVERAGE, audio_coverage
    assert audio_coverage(int(1430 * 8), 1430) >= AUDIO_OK_COVERAGE      # full
    assert audio_coverage(120 * 8, 1430) < AUDIO_OK_COVERAGE             # audio truncated at 120s
    assert audio_coverage(0, 1430) == 0.0                                # no audio
    assert audio_coverage(0, 0) == 1.0                                   # unknown duration: no judgment
    assert audio_coverage(300 * 8, 1430, max_length_s=300) >= AUDIO_OK_COVERAGE  # cap respected


def test_audio_fingerprint_short_partial_uses_longer_ffmpeg_result(monkeypatch):
    """fpcalc stops at a bad frame mid-file (exit≠0, SHORT PARTIAL fingerprint) but the audio
    is complete: the ffmpeg fallback yields a LONGER fingerprint which is kept (avoids a
    truncated fingerprint and a spurious 'audio truncated' warning)."""
    import os

    def fake_run(cmd, *a, **k):
        prog = os.path.basename(str(cmd[0])).lower()
        if "ffmpeg" in prog:
            with open(cmd[-1], "wb") as f:
                f.write(b"RIFF\x00\x00\x00\x00WAVE")
            return _cp(0, "")
        if str(cmd[-1]).endswith(".wav"):                 # fpcalc on full PCM -> long fingerprint
            return _cp(0, "FINGERPRINT=" + ",".join(str(i) for i in range(2000)) + "\n")
        return _cp(3, "FINGERPRINT=1,2,3\n",              # direct fpcalc: partial (bad frame)
                   "ERROR: Error decoding audio frame")
    monkeypatch.setattr(afp.subprocess, "run", fake_run)
    fp = afp.audio_fingerprint("long.mkv")
    assert fp.size == 2000                                 # kept the MORE complete one (ffmpeg)


def test_audio_fingerprint_no_audio_returns_empty_not_fatal(monkeypatch):
    """No decodable audio track: fpcalc fails and ffmpeg produces no wav -> EMPTY fingerprint,
    NO exception (the healthy video is indexed normally; not sent to 'corrupt')."""
    import os

    def fake_run(cmd, *a, **k):
        prog = os.path.basename(str(cmd[0])).lower()
        if "ffmpeg" in prog:                          # no audio -> no wav written, rc!=0
            return _cp(1, "", "Output file does not contain any stream")
        return _cp(2, "", "ERROR: Could not find any audio stream in the file")
    monkeypatch.setattr(afp.subprocess, "run", fake_run)
    fp = afp.audio_fingerprint("video_only.mp4")
    assert fp.size == 0 and fp.dtype == np.uint32


# ------------------------------------------------------------- _popcount32

def test_popcount32():
    x = np.array([0, 1, 0xFFFFFFFF, 0x80000000], dtype=np.uint32)
    assert _popcount32(x).tolist() == [0, 1, 32, 1]


# ------------------------------------------------------------- align_audio

@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_identical_audio_high_score_zero_offset(rng):
    a = rng.integers(0, 2**32, size=1000, dtype=np.uint64).astype(np.uint32)
    r = align_audio(a, a, min_overlap_s=30.0)
    assert r.score == pytest.approx(1.0)
    assert r.offset == pytest.approx(0.0)
    assert r.coverage == pytest.approx(1.0)


def test_prepended_ads_offset_detected(rng):
    a = rng.integers(0, 2**32, size=1000, dtype=np.uint64).astype(np.uint32)
    ads = rng.integers(0, 2**32, size=100, dtype=np.uint64).astype(np.uint32)
    b = np.concatenate([ads, a])                       # 100 ad items prepended
    r = align_audio(a, b, min_overlap_s=30.0)
    assert r.score == pytest.approx(1.0, abs=1e-6)     # content matches perfectly
    assert r.offset == pytest.approx(100 / ITEM_RATE_HZ, abs=0.2)   # b relative to a, +
    assert r.coverage == pytest.approx(1.0)


def test_different_audio_correlates_at_random(rng):
    a = rng.integers(0, 2**32, size=1000, dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 2**32, size=1000, dtype=np.uint64).astype(np.uint32)
    r = align_audio(a, b, min_overlap_s=30.0)
    # random bits match ~50% -> sim ~0.5 (well below theta_a=0.80)
    assert 0.4 < r.score < 0.65


def test_insufficient_overlap_returns_zero(rng):
    a = rng.integers(0, 2**32, size=100, dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 2**32, size=100, dtype=np.uint64).astype(np.uint32)
    r = align_audio(a, b, min_overlap_s=60.0)          # 60s*8 = 480 > 100 items
    assert r.score == 0.0


# --------------------------------------------- FFT align_audio == brute-force (verdict invariance §0)

def _align_audio_bruteforce(fp_a, fp_b, min_overlap_s=60.0, item_rate=ITEM_RATE_HZ, max_offset_s=300.0):
    """The original O(offsets·N) per-offset scan — kept here as the REFERENCE the FFT version must
    reproduce bit-exactly (same offset, score, coverage). If they ever diverge, the FFT rewrite
    changed a verdict and the test fails."""
    a = np.asarray(fp_a, dtype=np.uint32); b = np.asarray(fp_b, dtype=np.uint32)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return (0.0, 0, 0.0)
    min_overlap = max(1, int(min_overlap_s * item_rate))
    max_off = int(max_offset_s * item_rate)
    lo_off, hi_off = -min(max_off, na - 1), min(max_off, nb - 1)
    best_sim, best_off, best_len = 0.0, 0, 0
    for off in range(lo_off, hi_off + 1):
        lo = max(0, -off); hi = min(na, nb - off); length = hi - lo
        if length < min_overlap:
            continue
        bits = int(_popcount32(a[lo:hi] ^ b[lo + off:hi + off]).sum())
        sim = 1.0 - bits / (32.0 * length)
        if sim > best_sim:
            best_sim, best_off, best_len = sim, off, length
    if best_len == 0:
        return (0.0, 0, 0.0)
    return (float(best_sim), best_off, float(best_len / min(na, nb)))


def _make_pair(rng, na, nb, off, flip_frac):
    """b = a shifted by `off` with `flip_frac` of bits flipped (a partial/noisy match)."""
    a = rng.integers(0, 2**32, size=na, dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 2**32, size=nb, dtype=np.uint64).astype(np.uint32)
    for i in range(nb):                                    # copy a into b at the given offset
        j = i - off                                        # a[j] ~ b[i]  (b[i] = a[i-off])
        if 0 <= j < na:
            v = int(a[j])
            if flip_frac:
                mask = 0
                for bitpos in range(32):
                    if rng.random() < flip_frac:
                        mask |= (1 << bitpos)
                v ^= mask
            b[i] = v
    return a, b


def test_fft_align_audio_matches_bruteforce(rng):
    """Bit-exact equivalence on a spread of cases: clean/noisy, +/- offsets, unequal lengths."""
    cases = [
        (800, 800, 0, 0.0), (800, 800, 0, 0.10), (900, 700, 120, 0.05),
        (700, 900, -150, 0.05), (1200, 1200, 300, 0.20), (600, 1500, 50, 0.0),
        (1500, 600, -400, 0.15), (480, 480, 0, 0.30), (2000, 2000, 0, 0.02),
    ]
    for na, nb, off, flip in cases:
        a, b = _make_pair(rng, na, nb, off, flip)
        ref_sim, ref_off, ref_cov = _align_audio_bruteforce(a, b, min_overlap_s=30.0)
        r = align_audio(a, b, min_overlap_s=30.0)
        assert r.offset == ref_off / ITEM_RATE_HZ, f"offset mismatch on {(na, nb, off, flip)}"
        assert abs(r.coverage - ref_cov) < 1e-12, f"coverage mismatch on {(na, nb, off, flip)}"
        assert abs(r.score - ref_sim) < 1e-9, f"score mismatch on {(na, nb, off, flip)}"


def test_fft_align_audio_matches_bruteforce_random(rng):
    """Fuzz: many random pairs (incl. uncorrelated) must still match the reference exactly."""
    for _ in range(40):
        na = int(rng.integers(300, 1500)); nb = int(rng.integers(300, 1500))
        off = int(rng.integers(-200, 200)); flip = float(rng.choice([0.0, 0.05, 0.2, 0.5]))
        a, b = _make_pair(rng, na, nb, off, flip)
        ref = _align_audio_bruteforce(a, b, min_overlap_s=20.0)
        r = align_audio(a, b, min_overlap_s=20.0)
        assert r.offset == ref[1] / ITEM_RATE_HZ
        assert abs(r.coverage - ref[2]) < 1e-12
        assert abs(r.score - ref[0]) < 1e-9
