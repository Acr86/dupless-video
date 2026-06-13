"""Tests for scan optimizations: problem classification (corrupt/reindex),
store helpers, opt-in cap of audio_fp in feature_version, and storage auto-tune.
All pure (no real disk or GPU): auto-tune receives injected latency."""
from __future__ import annotations

from dupdetect.store import FingerprintStore, classify_problem
from dupdetect.tuning import autotune


# --------------------------------------------------------------- problem classification
def test_classify_problem_timeout_is_reindex():
    assert classify_problem("audio_fp: timeout (>600s) — fpcalc took too long") == "reindex"
    assert classify_problem("timeout (>240s) decoding (broken index?)") == "reindex"


def test_classify_problem_rest_is_corrupt():
    assert classify_problem("moov atom not found") == "corrupt"
    assert classify_problem("Invalid data found when processing input") == "corrupt"
    assert classify_problem("") == "corrupt"
    assert classify_problem(None) == "corrupt"


def test_store_category_filter_and_clear(tmp_path):
    s = FingerprintStore(tmp_path / "p.sqlite")
    s.save_problem("/a.mkv", "timeout (>240s) decoding")          # auto -> reindex
    s.save_problem("/b.mp4", "moov atom not found")               # auto -> corrupt
    s.save_problem("/c.avi", "slow", category="reindex")          # explicit
    assert {p for p, *_ in s.problems(category="reindex")} == {"/a.mkv", "/c.avi"}
    assert {p for p, *_ in s.problems(category="corrupt")} == {"/b.mp4"}
    assert len(s.problems()) == 3
    assert all(len(t) == 4 for t in s.problems())                 # (path, error, category, repair_note)
    assert all(len(t) == 2 for t in s.iter_problems())            # compat: 2-tuples
    s.clear_problem("/a.mkv")
    assert {p for p, *_ in s.problems(category="reindex")} == {"/c.avi"}
    s.close()


def test_mark_repair_failed_timeout_stays_reindex(tmp_path):
    """A remux that fails due to TIMEOUT is NOT declared unrecoverable (on HDD this is usually
    disk contention): stays 'reindex' (retryable), with the note from the last attempt."""
    s = FingerprintStore(tmp_path / "p.sqlite")
    s.save_problem("/a.mkv", "timeout (>900s) decoding")          # reindex
    s.mark_repair_failed("/a.mkv", "timeout", "remux timeout")
    (path, _err, cat, note), = s.problems(category="reindex")
    assert path == "/a.mkv" and cat == "reindex"
    assert note and "timeout" in note and "repairable" in note
    s.close()


def test_mark_repair_failed_hard_moves_to_corrupt(tmp_path):
    """A hard remux failure (kind!='timeout') moves the file to 'corrupt' with the reason."""
    s = FingerprintStore(tmp_path / "p.sqlite")
    s.save_problem("/a.mkv", "timeout (>900s)")                   # starts as reindex
    s.mark_repair_failed("/a.mkv", "corrupt", "ffmpeg: invalid data found")
    assert s.problems(category="reindex") == []
    (path, _err, cat, note), = s.problems(category="corrupt")
    assert path == "/a.mkv" and cat == "corrupt"
    assert note == "remux failed: ffmpeg: invalid data found"
    s.close()


def test_prune_missing_problems_forgets_deleted_not_offline(tmp_path):
    """Forgets a problem whose file no longer exists but whose FOLDER does (truly deleted); does NOT
    touch one whose folder also does not exist (possible unmounted volume)."""
    s = FingerprintStore(tmp_path / "p.sqlite")
    real = tmp_path / "deleted.mkv"                               # folder (tmp_path) exists, file does not
    s.save_problem(str(real), "moov atom not found")
    s.save_problem("/offline_disk/x.mkv", "moov atom not found")  # folder doesn't exist either
    n = s.prune_missing_problems()
    assert n == 1
    paths = {p for p, *_ in s.problems()}
    assert str(real) not in paths and "/offline_disk/x.mkv" in paths
    s.close()


def test_reclassify_stale_on_reopen(tmp_path):
    """Old rows were all left as 'corrupt' (migration default). On reopening the store the
    category is recomputed from the error: a 'timeout' becomes 'reindex'."""
    db = tmp_path / "p.sqlite"
    s = FingerprintStore(db)
    s.save_problem("/a.mkv", "timeout (>900s)")                   # would be reindex...
    s.conn.execute("UPDATE problems SET category='corrupt'")      # ...corrupt it manually (old DB)
    s.conn.commit(); s.close()
    s2 = FingerprintStore(db)                                     # reopen -> _reclassify_stale_problems
    assert {p for p, *_ in s2.problems(category="reindex")} == {"/a.mkv"}
    s2.close()


# --------------------------------------------------------------- audio_fp duration-gated cap (fork)
def test_feature_version_gated_audio_fp_invalidates_cache():
    from dupdetect.features.embeddings import Embedder
    from dupdetect.pipeline.analyze import feature_version
    e = Embedder(model="dinov2_vitb14", dim=768, fps=2.0)
    base = feature_version(e)                                     # no cap = whole file always
    assert feature_version(e, audio_fp_cap_s=0) == base           # capping disabled = no change
    gated = feature_version(e, audio_fp_cap_s=600, audio_fp_cap_above_s=3600)
    assert gated != base and "G3600C600" in gated                # gated policy -> different version


def test_audio_fp_max_for_duration_gate():
    """The cap is applied ONLY above the duration gate; short content is whole-file (0)."""
    from dupdetect.config import load_thresholds
    th = load_thresholds()                                        # fp_max_s=600, fp_cap_above_s=3600
    assert th.audio_fp_max_for(1353) == 0                         # 22min episode -> whole file
    assert th.audio_fp_max_for(3600) == 0                         # exactly at the gate -> whole file
    assert th.audio_fp_max_for(7200) == 600                       # 2h movie -> capped head
    assert th.audio_fp_max_for(None) == 600                       # unknown duration -> cap (conservative)
    assert th.audio_fp_max_for(0) == 600                          # zero/unknown -> cap


# --------------------------------------------------------------- storage-aware auto-tune
def test_autotune_hdd_lowers_workers():
    at = autotune(["x"], cpu_count=32, seek_ms=12.0)             # high latency = spinning disk
    assert at.workers == 2 and at.decode_workers == 1
    assert at.kind in ("hdd", "network-hdd") and "workers=2" in at.note


def test_autotune_ssd_raises_workers_and_decode():
    at = autotune(["x"], cpu_count=32, seek_ms=0.3)             # low latency = NVMe
    assert at.kind == "ssd" and at.workers == 12 and at.decode_workers == 4


def test_autotune_intermediate():
    at = autotune(["x"], cpu_count=32, seek_ms=3.0)
    assert at.decode_workers == 2 and at.kind in ("moderate", "network")


def test_autotune_no_probe_is_conservative():
    at = autotune([], cpu_count=8)                               # no files -> no probe taken
    assert at.workers == 2 and at.decode_workers == 1 and at.kind == "unknown"


def test_autotune_tiered_hdd_detected_by_concurrency():
    """Storage Space (HDD + SSD cache): the seek probe looks fast (cache serves tiny reads) but
    concurrent large reads thrash the mechanical tier. The scaling probe catches it -> workers=2."""
    at = autotune(["x"], cpu_count=32, seek_ms=0.3, scaling=0.3)  # fast seek BUT concurrency collapses
    assert at.kind == "hdd-tiered" and at.workers == 2 and at.decode_workers == 1
    assert "thrashes" in at.note


def test_autotune_ssd_confirmed_when_concurrency_scales():
    """A true SSD: fast seek AND concurrency holds throughput -> high workers."""
    at = autotune(["x"], cpu_count=32, seek_ms=0.3, scaling=1.2)  # concurrency holds
    assert at.kind == "ssd" and at.workers == 12 and at.decode_workers == 4
