"""Tests for scan optimizations: problem classification (corrupt/reindex),
store helpers, opt-in cap of audio_fp in feature_version, and storage auto-tune.
All pure (no real disk or GPU): auto-tune receives injected latency."""
from __future__ import annotations

import os

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


def test_prune_missing_problems_forgets_deleted_on_online_volume(tmp_path):
    """A problem row whose file is gone (folder still present) on an ONLINE volume is forgotten."""
    s = FingerprintStore(tmp_path / "p.sqlite")
    p = str(tmp_path / "Media" / "deleted.mkv")
    s.save_problem(p, "moov atom not found")
    n = s.prune_missing_problems(exists=lambda x: x != p,          # all present except the file
                                 isdir=lambda x: True, ismount=lambda x: False)
    assert n == 1 and s.problems() == []
    s.close()


def test_prune_missing_problems_mode_b_deleted_folder(tmp_path):
    """Mode B (the residual ghost bug): the file's whole FOLDER was deleted while the volume stayed
    online. The old parent-dir guard read 'parent gone' as 'volume offline' and kept the row forever
    in the Problems tab; the shared volume+mount-aware core (same as prune_missing_files) cleans it."""
    s = FingerprintStore(tmp_path / "p.sqlite")
    folder = str(tmp_path / "Media" / "OldShow")
    p = os.path.join(folder, "ep.mkv")
    s.save_problem(p, "moov atom not found")
    n = s.prune_missing_problems(
        exists=lambda x: not x.startswith(folder),                # volume + Media online; OldShow gone
        isdir=lambda x: not x.startswith(folder),                 # OldShow and below: not a dir
        ismount=lambda x: False)                                  # plain deleted folder, not a mount
    assert n == 1 and s.problems() == []
    s.close()


def test_prune_missing_problems_keeps_offline_drive(tmp_path):
    """§2 fail-safe: the whole drive reads offline -> its problem rows are never touched."""
    from dupdetect.store.store import _volume_root
    s = FingerprintStore(tmp_path / "p.sqlite")
    p = str(tmp_path / "x.mkv")
    s.save_problem(p, "moov atom not found")
    anchor = _volume_root(p)
    n = s.prune_missing_problems(exists=lambda x: x not in (anchor, p))   # drive itself unreachable
    assert n == 0 and [pp for pp, *_ in s.problems()] == [p]
    s.close()


def test_prune_missing_problems_keeps_offline_junction_subtree(tmp_path):
    """§2: an offline nested mount/junction under an online volume is an unmount, NOT a deletion."""
    s = FingerprintStore(tmp_path / "p.sqlite")
    junction = str(tmp_path / "Media" / "NAS")
    p = os.path.join(junction, "ep.mkv")
    s.save_problem(p, "moov atom not found")
    n = s.prune_missing_problems(
        exists=lambda x: not x.startswith(junction),              # volume/Media online; NAS unreachable
        isdir=lambda x: not x.startswith(junction),               # NAS doesn't resolve (offline target)
        ismount=lambda x: x == junction)                          # NAS IS a junction/mount
    assert n == 0 and [pp for pp, *_ in s.problems()] == [p]
    s.close()


# --------------------------------------------------------------- file-existence self-heal (prune_missing_files)
def _save_file(s, path):
    """Insert a minimal indexed file row (LITE: metadata only, no .npy) for the existence-sweep tests."""
    from dupdetect.models import Probe
    s.save_meta(path, 0.0, 100, "h",
                Probe(duration_s=1.0, width=0, height=0, vcodec="h264", bitrate_kbps=None), "fv")


def test_prune_missing_files_forgets_deleted_on_reachable_volume(tmp_path):
    """Mode A: a file gone from disk on a mounted volume (its folder still present) is forgotten."""
    s = FingerprintStore(tmp_path / "p.sqlite")
    p = str(tmp_path / "Media" / "gone.mkv")
    _save_file(s, p)
    n = s.prune_missing_files(exists=lambda x: x != p,            # all present except the file
                              isdir=lambda x: True, ismount=lambda x: False)
    assert n == 1 and p not in s.all_paths()
    s.close()


def test_prune_missing_files_mode_b_deleted_subfolder_volume_online(tmp_path):
    """Mode B (the reported bug): a whole SUBFOLDER was deleted while the drive stays online. Its files
    ARE cleaned even though their parent dir is gone — what orphan_paths' watched-root guard misses."""
    s = FingerprintStore(tmp_path / "p.sqlite")
    folder = str(tmp_path / "Media" / "OldShow")
    p = os.path.join(folder, "ep.mkv")
    _save_file(s, p)
    n = s.prune_missing_files(
        exists=lambda x: not x.startswith(folder),               # volume + Media online; OldShow gone
        isdir=lambda x: not x.startswith(folder),                # OldShow and below: not a dir
        ismount=lambda x: False)                                 # plain deleted folder, not a mount
    assert n == 1 and p not in s.all_paths()
    s.close()


def test_prune_missing_files_keeps_offline_junction_subtree(tmp_path):
    """§2 catastrophe PREVENTED (adversarially found): a nested junction to an OFFLINE target under a
    mounted volume must NOT be mass-forgotten — its boundary reads as a mount/reparse point -> keep."""
    s = FingerprintStore(tmp_path / "p.sqlite")
    junction = str(tmp_path / "Media" / "NAS")
    p = os.path.join(junction, "ep.mkv")
    _save_file(s, p)
    n = s.prune_missing_files(
        exists=lambda x: not x.startswith(junction),             # volume/Media online; NAS unreachable
        isdir=lambda x: not x.startswith(junction),              # NAS doesn't resolve (offline target)
        ismount=lambda x: x == junction)                         # NAS IS a junction/mount
    assert n == 0 and p in s.all_paths()                         # kept (an unmount is not a deletion)
    s.close()


def test_prune_missing_files_keeps_offline_drive(tmp_path):
    """§2: a whole offline drive is skipped wholesale by the one-probe-per-volume guard."""
    from dupdetect.store.store import _volume_root
    s = FingerprintStore(tmp_path / "p.sqlite")
    p = str(tmp_path / "x.mkv")
    _save_file(s, p)
    anchor = _volume_root(p)
    n = s.prune_missing_files(exists=lambda x: x not in (anchor, p))   # the DRIVE itself reads offline
    assert n == 0 and p in s.all_paths()
    s.close()


def test_prune_missing_files_noop_when_all_present(tmp_path):
    s = FingerprintStore(tmp_path / "p.sqlite")
    a, b = str(tmp_path / "a.mkv"), str(tmp_path / "b.mkv")
    _save_file(s, a); _save_file(s, b)
    n = s.prune_missing_files(exists=lambda x: True, isdir=lambda x: True, ismount=lambda x: False)
    assert n == 0 and set(s.all_paths()) == {a, b}
    s.close()


def test_prune_missing_files_deterministic_via_injected_exists(tmp_path):
    """§0: the decision core has no real-disk dependency — exactly the injected 'gone' subset is forgotten."""
    s = FingerprintStore(tmp_path / "p.sqlite")
    keep, drop = str(tmp_path / "keep.mkv"), str(tmp_path / "drop.mkv")
    _save_file(s, keep); _save_file(s, drop)
    n = s.prune_missing_files(exists=lambda x: x != drop, isdir=lambda x: True, ismount=lambda x: False)
    assert n == 1 and s.all_paths() == [keep]
    s.close()


def test_volume_root_rejects_degenerate_anchors():
    """Forms that would probe the WRONG location (and mass-forget) map to '' (unknown -> keep)."""
    from dupdetect.store.store import _volume_root
    if os.name == "nt":
        assert _volume_root("C:\\Media\\x.mkv") == "C:\\"                    # proper drive root
        assert _volume_root("\\\\srv\\share\\x.mkv") == "\\\\srv\\share\\"   # proper UNC share root
        assert _volume_root("C:foo.mkv") == ""                              # drive-relative -> unknown
        assert _volume_root("rel\\x.mkv") == ""                             # relative -> unknown
    else:
        assert _volume_root("/mnt/x/file.mkv") == "/"                       # POSIX single root
        assert _volume_root("rel/x.mkv") == ""                             # relative -> unknown


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


# --------------------------------------------------------------- Pass-2 BLAS thread pinning (perf §1)
def test_single_threaded_blas_pins_then_restores(monkeypatch):
    """Inside the context the BLAS/OpenMP thread vars are '1' (so the Pass-2 process pool doesn't
    oversubscribe the cores); on exit the PRIOR state is restored exactly: a var that was unset is
    removed again, and a pre-existing value is put back. Speed-only -> verdict unchanged (§0)."""
    from dupdetect.match.matcher import _BLAS_THREAD_VARS, single_threaded_blas

    # One var pre-set by the user, the rest unset -> both restore paths are exercised.
    preset = _BLAS_THREAD_VARS[0]
    for v in _BLAS_THREAD_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(preset, "8")

    with single_threaded_blas():
        assert all(os.environ[v] == "1" for v in _BLAS_THREAD_VARS)   # pinned for the pool's lifetime

    assert os.environ.get(preset) == "8"                              # pre-existing value restored
    assert all(v not in os.environ for v in _BLAS_THREAD_VARS[1:])    # previously-unset vars removed


def test_single_threaded_blas_restores_on_exception():
    """The restore runs even if the wrapped block raises (the pool can fail mid-scan, §2)."""
    from dupdetect.match.matcher import _BLAS_THREAD_VARS, single_threaded_blas

    before = {v: os.environ.get(v) for v in _BLAS_THREAD_VARS}
    try:
        with single_threaded_blas():
            raise RuntimeError("pool boom")
    except RuntimeError:
        pass
    assert {v: os.environ.get(v) for v in _BLAS_THREAD_VARS} == before


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
