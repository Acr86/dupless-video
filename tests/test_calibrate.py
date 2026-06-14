"""Tests for calibration (step 10): confusion by tier + threshold suggestion.

Uses synthetic LabeledSignals (no real files): this exercises the threshold
sweep and the confusion matrix in isolation.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict

import numpy as np
import pytest

from dupdetect.config import load_thresholds
from dupdetect.models import AlignResult, Probe, Quality, Record, Verdict
from dupdetect.pipeline.calibrate import (
    LabeledSignal, confusion_by_genre, confusion_by_tier, labeled_signals_from_feedback,
    load_pairs, suggest_thresholds, verdict_of,
)
from dupdetect.store import FingerprintStore


@pytest.fixture
def th():
    return load_thresholds()


@pytest.fixture
def store(tmp_path):
    s = FingerprintStore(tmp_path / "calib.sqlite")
    yield s
    s.close()


def _rec(path):
    """Minimal persistable Record (no real embeddings): sufficient for store.load(with_embeddings=False)
    during recompute, which only needs audio_fp/scene_cuts/path."""
    return Record(path=path, mtime=0.0, size=1_000,
                  probe=Probe(3600.0, 1920, 1080, "h264", 5000, []),
                  content_hash="h" + path, global_vec=np.zeros(8, np.float32),
                  window_vecs=np.zeros((2, 8), np.float32), embeddings=np.zeros((3, 8), np.float16),
                  audio_fp=np.zeros(1, np.uint32), scene_cuts=np.zeros(1, np.float32),
                  quality=Quality())


def _sig(label, a, v, vcov, s):
    return LabeledSignal(label, AlignResult(a), AlignResult(v, coverage=vcov), AlignResult(s))


# --------------------------------------------------------------- pair loading

def test_load_pairs_json(tmp_path):
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps([{"path_a": "/a", "path_b": "/b", "label": "dup"}]), "utf-8")
    assert load_pairs(p) == [("/a", "/b", "dup")]


def test_load_pairs_csv(tmp_path):
    p = tmp_path / "pairs.csv"
    p.write_text("path_a,path_b,label\n/a,/b,same\n/c,/d,diff\n", "utf-8")
    assert load_pairs(p) == [("/a", "/b", "same"), ("/c", "/d", "diff")]


# --------------------------------------------------------------- is_same / verdict

def test_is_same_label():
    assert _sig("dub", 0, 0, 0, 0).is_same is True
    assert _sig("diff", 0, 0, 0, 0).is_same is False


def test_verdict_of_t1(th):
    # strong audio+video with coverage -> CERTAIN
    assert verdict_of(_sig("dup", 0.9, 0.9, 0.95, 0.5), th) == Verdict.CERTAIN


# --------------------------------------------------------------- confusion

def test_confusion_separates_same_diff(th):
    signals = [
        _sig("dup", 0.9, 0.9, 0.95, 0.5),    # CERTAIN, same
        _sig("diff", 0.1, 0.1, 0.05, 0.1),   # DIFFERENT, diff
    ]
    conf = confusion_by_tier(signals, th)
    assert conf["CERTAIN"]["same"] == 1
    assert conf["DIFFERENT"]["diff"] == 1


def _gsig(label, a, v, vcov, s, genre):
    sig = _sig(label, a, v, vcov, s)
    sig.genre = genre
    return sig


def test_confusion_by_genre_breaks_down_per_label(th):
    """Per-genre metrics evaluate the REAL tree. Genre only buckets the report (never the verdict)."""
    signals = [
        _gsig("dup", 0.9, 0.9, 0.95, 0.5, "series"),    # CERTAIN, same  -> series TP
        _gsig("dup", 0.9, 0.9, 0.95, 0.5, "series"),    # CERTAIN, same
        _gsig("diff", 0.9, 0.9, 0.95, 0.0, "series"),   # CERTAIN, DIFF  -> series strict FP
        _gsig("dup", 0.9, 0.9, 0.95, 0.5, "cinema"),    # CERTAIN, same  -> cinema clean
        _gsig("diff", 0.1, 0.1, 0.05, 0.1, "cinema"),   # DIFFERENT, diff
    ]
    by_g = confusion_by_genre(signals, th)
    assert by_g["series"]["fp_strict"] == 1            # the cross-episode FP surfaces under series
    assert by_g["series"]["precision"] == 2 / 3        # 2 same / 3 dup-tier
    assert by_g["series"]["recall"] == 1.0             # both 'same' caught
    assert by_g["cinema"]["fp_strict"] == 0
    assert by_g["cinema"]["precision"] == 1.0


def test_confusion_by_genre_none_when_no_data(th):
    """precision/recall are None (not a misleading 0%) when a genre has no dup-tier / no 'same'."""
    by_g = confusion_by_genre([_gsig("diff", 0.1, 0.1, 0.05, 0.1, "docs")], th)
    assert by_g["docs"]["precision"] is None and by_g["docs"]["recall"] is None


# --------------------------------------------------------------- suggestion

def test_suggest_achieves_zero_fp(th):
    # a 'diff' with moderate scores that would be FP at a low theta_v; the sweep raises
    # the threshold to push it out of T1/T2 while keeping the strong 'same' pairs.
    signals = [
        _sig("dup", 0.95, 0.95, 1.0, 0.6),    # same, very strong
        _sig("dup", 0.90, 0.92, 0.9, 0.6),    # same, strong
        _sig("diff", 0.82, 0.83, 0.9, 0.5),   # diff but high scores -> FP risk
    ]
    sug = suggest_thresholds(signals, base=th)
    assert sug["false_positives_T1T2"] == 0          # goal: zero FP in T1/T2
    assert sug["recall_dup"] > 0.0                   # still catches real duplicates


def test_suggest_reports_confusion_and_n(th):
    signals = [_sig("dup", 0.9, 0.9, 0.95, 0.5), _sig("diff", 0.1, 0.1, 0.0, 0.1)]
    sug = suggest_thresholds(signals, base=th)
    assert sug["n_pairs"] == 2
    assert "confusion" in sug and 0.5 <= sug["theta_v"] <= 0.95


# ----------------------------------------- feedback -> signals (fast path + orphan recompute)

def _no_align(*_a, **_k):
    raise AssertionError("_align_pair should not be called on this path")


def test_labeled_from_feedback_uses_matches_json(store, th, monkeypatch):
    """Pair with a row in `matches`: reuses its JSON scores WITHOUT recomputing (does not call _align_pair)."""
    import dupdetect.pipeline.calibrate as cal
    monkeypatch.setattr(cal, "_align_pair", _no_align)   # if called, the fast path failed
    aj = json.dumps(asdict(AlignResult(0.9)))
    store.save_match("/a", "/b", "CERTAIN", 0.99, "T1", audio_json=aj, video_json=aj, scenes_json=aj)
    store.save_feedback("/a", "/b", "same")
    sigs = labeled_signals_from_feedback(store, th=th)
    assert len(sigs) == 1 and sigs[0].is_same
    assert sigs[0].audio.score == pytest.approx(0.9)


def test_labeled_from_feedback_recomputes_orphan(store, th, monkeypatch):
    """ORPHAN pair (feedback but NO row in matches): recovered by recomputing from stored
    fingerprints. _align_pair is mocked to avoid depending on torch/real embeddings."""
    import dupdetect.pipeline.calibrate as cal
    store.save(_rec("/a"), feature_version="fv")
    store.save(_rec("/b"), feature_version="fv")
    store.save_feedback("/a", "/b", "same")              # canonical (/a,/b); NO save_match
    monkeypatch.setattr(cal, "_align_pair",
                        lambda ra, rb, cache, t: (AlignResult(0.95), AlignResult(0.95, coverage=0.9),
                                                  AlignResult(0.5)))
    sigs = labeled_signals_from_feedback(store, th=th)
    assert len(sigs) == 1 and sigs[0].is_same           # orphan recovered
    assert sigs[0].video.coverage == pytest.approx(0.9)


def test_labeled_from_feedback_skips_missing_fingerprint(store, th, monkeypatch):
    """Endpoint with no record in `files` -> the pair is SKIPPED (None), no exception."""
    import dupdetect.pipeline.calibrate as cal
    store.save(_rec("/a"), feature_version="fv")         # /b does NOT exist in files
    store.save_feedback("/a", "/b", "same")
    monkeypatch.setattr(cal, "_align_pair", _no_align)   # must not align without both records
    sigs = labeled_signals_from_feedback(store, th=th)
    assert sigs == []


def test_labeled_from_feedback_skips_lite_without_embeddings(store, th, monkeypatch):
    """LITE / 'exact-only' records (no fingerprints) -> UNRECOVERABLE: a signal with score 0 is NOT
    fabricated (would poison calibration). Skipped -> user must re-scan (full)."""
    import dupdetect.pipeline.calibrate as cal
    for p in ("/a", "/b"):
        lite = _rec(p)
        lite.embeddings = np.zeros((0, 0), np.float16)   # no embeddings (like an exact-scan)
        store.save(lite, feature_version="exact-only-v1")
    store.save_feedback("/a", "/b", "same")
    monkeypatch.setattr(cal, "_align_pair", _no_align)   # must not attempt alignment without embeddings
    sigs = labeled_signals_from_feedback(store, th=th)
    assert sigs == []


# ------------------------------------------------- apply_thresholds (frozen-app write target)

def test_apply_thresholds_frozen_writes_user_override(tmp_path, monkeypatch):
    """Frozen app: the bundled thresholds.yaml is READ-ONLY, so recalibrate must write a per-user
    OVERRIDE in the data dir (not crash with WinError 5), and load_thresholds then prefers it."""
    from pathlib import Path

    import dupdetect.config as cfg
    from dupdetect.ui import actions
    override = tmp_path / "thresholds.yaml"
    monkeypatch.setattr(cfg, "user_thresholds_path", lambda: override)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    p = actions.apply_thresholds(0.81, 0.93)             # no explicit path -> per-user override (frozen)
    assert Path(p) == override and override.exists()      # wrote the override, NOT the bundled config
    th2 = cfg.load_thresholds()                           # frozen + override exists -> prefers it
    assert th2.theta_v == 0.81 and th2.theta_a == 0.93


def test_apply_thresholds_explicit_path_is_respected(tmp_path):
    """An explicit config_path is written in place (used by tests/automation; no data-dir side effects)."""
    import yaml

    from dupdetect.ui import actions
    cfg_file = tmp_path / "t.yaml"
    cfg_file.write_text(yaml.safe_dump({"video": {"theta_v": 0.5}, "audio": {"theta_a": 0.5}}), "utf-8")
    p = actions.apply_thresholds(0.8, 0.9, config_path=str(cfg_file))
    assert p == str(cfg_file)
    raw = yaml.safe_load(cfg_file.read_text("utf-8"))
    assert raw["video"]["theta_v"] == 0.8 and raw["audio"]["theta_a"] == 0.9
