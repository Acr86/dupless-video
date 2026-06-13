"""Tests for the decision tree. The tree is pure logic => testable without features.

These cases encode the required semantics; they act as a safety net when
calibrating thresholds so that a tier does not break unnoticed.
"""
from __future__ import annotations

import numpy as np
import pytest

from dupdetect.config import load_thresholds
from dupdetect.models import AlignResult, Probe, Record, Verdict
from dupdetect.match.tree import decide_tree


def _rec(path: str, content_hash: str = "h", size: int = 100, n_cuts: int = 700) -> Record:
    # n_cuts default 700 over 7000 s = 0.10 cuts/s: a dense (demux-like) signature, so the
    # scenes-only T4 guard trusts it. Pass a low n_cuts to simulate a coarse SEEK signature.
    return Record(
        path=path, mtime=0.0, size=size,
        probe=Probe(duration_s=7000, width=1920, height=1080, vcodec="h264",
                    bitrate_kbps=8000, audio_tracks=[]),
        content_hash=content_hash,
        global_vec=np.zeros(8, dtype=np.float32),
        window_vecs=np.zeros((4, 8), dtype=np.float32),
        embeddings=np.zeros((1, 8), dtype=np.float32),
        audio_fp=np.zeros(1, dtype=np.float32),
        scene_cuts=np.zeros(n_cuts, dtype=np.float32),
    )


@pytest.fixture
def th():
    return load_thresholds()


def test_t0_byte_identical(th):
    a, b = _rec("a", "same", 100), _rec("b", "same", 100)
    r = decide_tree(a, b, AlignResult(0), AlignResult(0), AlignResult(0), th)
    assert r.verdict == Verdict.CERTAIN and r.confidence == 1.0


def test_t1_audio_and_video(th):
    a, b = _rec("a", "x"), _rec("b", "y")
    r = decide_tree(a, b, AlignResult(0.9), AlignResult(0.9, coverage=0.9),
                    AlignResult(0.5), th)
    assert r.verdict == Verdict.CERTAIN
    assert "T1" in r.reason


def test_t2_different_dub(th):
    # identical video, audio does NOT align -> same film, different language
    a, b = _rec("a", "x"), _rec("b", "y")
    r = decide_tree(a, b, AlignResult(0.1), AlignResult(0.9, coverage=0.9),
                    AlignResult(0.4), th)
    assert r.verdict == Verdict.VERY_HIGH
    assert "dub" in r.reason


def test_t4_scenes_only_goes_to_review(th):
    a, b = _rec("a", "x"), _rec("b", "y")           # dense cut signature (default) -> trusted
    r = decide_tree(a, b, AlignResult(0.1), AlignResult(0.3, coverage=0.3),
                    AlignResult(0.85), th)
    assert r.verdict == Verdict.PROBABLE  # review queue, NEVER auto-deletes


def test_t4_coarse_scene_signature_not_trusted(th):
    # SAME high scene score, but a COARSE cut signature (150 cuts / 7000 s = 0.021 cuts/s,
    # below min_cut_density): SEEK-sampled giants. Scenes alone are not discriminative here
    # (unrelated dense films align ~0.88), so T4 must NOT fire -> falls through to DIFFERENT.
    a, b = _rec("a", "x", n_cuts=150), _rec("b", "y", n_cuts=150)
    r = decide_tree(a, b, AlignResult(0.1), AlignResult(0.3, coverage=0.3),
                    AlignResult(0.85), th)
    assert r.verdict == Verdict.DIFFERENT  # coarse scenes-only no longer clutters review


def test_t4b_dead_zone_goes_to_review(th):
    # A4: video in band [theta_v=0.75, theta_v_high=0.85) (0.80) WITH high coverage
    # (0.85), ambiguous audio (0.5) and weak scenes (0.6). Goes to review.
    a, b = _rec("a", "x"), _rec("b", "y")
    r = decide_tree(a, b, AlignResult(0.5), AlignResult(0.80, coverage=0.85),
                    AlignResult(0.6), th)
    assert r.verdict == Verdict.PROBABLE
    assert "T4b" in r.reason


def test_high_video_score_but_negligible_coverage_does_not_align(th):
    # Real finding (a feature film vs a short clip): SW cherry-picks a spurious path -> score
    # 0.80 but coverage 0.037 between DIFFERENT films. Without coverage gate it
    # fell into T4b (review); now must land in DIFFERENT (audio also non-corroborating).
    a, b = _rec("a", "x"), _rec("b", "y")
    r = decide_tree(a, b, AlignResult(0.53), AlignResult(0.80, coverage=0.037),
                    AlignResult(0.3), th)
    assert r.verdict == Verdict.DIFFERENT


def test_t1_requires_coverage(th):
    # strong audio + high video score but negligible coverage -> NOT T1 (certain dup);
    # falls to T4b via strong audio (review), never CERTAIN with a 4% match.
    a, b = _rec("a", "x"), _rec("b", "y")
    r = decide_tree(a, b, AlignResult(0.9), AlignResult(0.9, coverage=0.04),
                    AlignResult(0.2), th)
    assert r.verdict != Verdict.CERTAIN


def test_t5_different_files(th):
    # all signals weak -> genuinely different
    a, b = _rec("a", "x"), _rec("b", "y")
    r = decide_tree(a, b, AlignResult(0.1), AlignResult(0.2), AlignResult(0.2), th)
    assert r.verdict == Verdict.DIFFERENT


def test_different_edition_is_not_duplicate(th):
    # video aligns strongly BUT b is a contiguous superset -> director's cut
    a, b = _rec("a", "x"), _rec("b", "y")
    v = AlignResult(0.9, coverage=0.8, contiguous_superset=True, superset_dir=1,
                    extra_ratio=0.15)
    r = decide_tree(a, b, AlignResult(0.2), v, AlignResult(0.5), th)
    assert r.verdict == Verdict.DIFFERENT_EDITION
