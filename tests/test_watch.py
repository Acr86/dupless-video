"""Watcher: reconcile (new/changed/deleted detection, mid-copy debounce) + one-cycle orchestration."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from dupdetect import watch
from dupdetect.config import load_thresholds
from dupdetect.models import Probe, Quality, Record
from dupdetect.pipeline.analyze import feature_version
from dupdetect.store import FingerprintStore


class _DummyEmbedder:
    fps = 2.0; model_name = "m"; dim = 8; algo_version = 1
    @property
    def feature_version(self) -> str:
        return "fvtest"


def _vid(p) -> str:
    p.write_bytes(b"x" * 1024)                       # content irrelevant (no decode in these tests)
    return str(p)


def _rec(path, mtime: float = 0.0, size: int = 1) -> Record:
    return Record(path=str(path), mtime=mtime, size=size,
                  probe=Probe(10.0, 100, 100, "h264", 1000, []), content_hash="h",
                  global_vec=np.zeros(8, np.float32), window_vecs=np.zeros((0, 8), np.float32),
                  embeddings=np.zeros((0, 8), np.float16), audio_fp=np.zeros(0, np.uint32),
                  scene_cuts=np.zeros(0, np.float32), quality=Quality())


def _age(p, secs: float = 100.0) -> None:
    t = time.time() - secs
    os.utime(p, (t, t))                              # make mtime old -> 'stable'


def test_pending_files_new_changed_skips_fresh_and_midcopy(tmp_path):
    th = load_thresholds()
    fv = feature_version(_DummyEmbedder(), False,
                         audio_fp_cap_s=th.audio_fp_cap_s, audio_fp_cap_above_s=th.audio_fp_cap_above_s)
    s = FingerprintStore(tmp_path / "w.sqlite")
    new = _vid(tmp_path / "new.mp4"); _age(tmp_path / "new.mp4")
    fresh = _vid(tmp_path / "fresh.mp4"); _age(tmp_path / "fresh.mp4")
    mid = _vid(tmp_path / "copying.mp4")             # just written -> recent mtime (mid-copy)
    st = os.stat(fresh)
    s.save(_rec(fresh, st.st_mtime, st.st_size), feature_version=fv)   # mark as already indexed
    pend = watch.pending_files(str(tmp_path), s, fv, stable_s=15.0)
    assert new in pend                               # new -> pending
    assert fresh not in pend                         # already fresh -> skip
    assert mid not in pend                           # modified within stable_s -> wait a cycle
    s.close()


def test_orphan_paths_finds_deleted(tmp_path):
    s = FingerprintStore(tmp_path / "w.sqlite")
    exists = _vid(tmp_path / "a.mp4"); st = os.stat(exists)
    s.save(_rec(exists, st.st_mtime, st.st_size), feature_version="fv")
    s.save(_rec(tmp_path / "gone.mp4"), feature_version="fv")    # never created on disk
    orph = watch.orphan_paths(str(tmp_path), s)
    assert str(tmp_path / "gone.mp4") in orph and exists not in orph
    s.close()


def test_orphan_paths_normalized_root_and_boundary(tmp_path):
    """The watch root is normalized (case+separators via normcase) before comparison, so a root
    written with different separators / a trailing slash still finds orphans; and a sibling folder
    that only shares a name PREFIX ('Series2' vs root 'Series') is NOT falsely matched."""
    s = FingerprintStore(tmp_path / "w.sqlite")
    series = tmp_path / "Series"; series.mkdir()
    series2 = tmp_path / "Series2"; series2.mkdir()
    s.save(_rec(series / "gone.mp4"), feature_version="fv")        # orphan under Series
    s.save(_rec(series2 / "alive.mp4"), feature_version="fv")      # sibling sharing the name prefix
    root = str(series).replace(os.sep, "/") + "/"                 # odd-but-equivalent root form
    orph = {Path(p).name for p in watch.orphan_paths(root, s)}
    assert "gone.mp4" in orph                                      # detected despite the root form
    assert "alive.mp4" not in orph                                # sibling prefix NOT falsely matched
    s.close()


@pytest.mark.skipif(os.name != "nt", reason="paths are case-insensitive only on Windows")
def test_orphan_paths_case_insensitive_root_on_windows(tmp_path):
    """Windows: a watch root whose CASE differs from the indexed path (the FS is case-insensitive)
    must still detect orphans — otherwise a file sent to the Recycle Bin is never removed from the
    list. This was the bug: case-sensitive startswith missed the orphan."""
    s = FingerprintStore(tmp_path / "w.sqlite")
    sub = tmp_path / "Series"; sub.mkdir()
    s.save(_rec(sub / "gone.mp4"), feature_version="fv")
    orph = watch.orphan_paths(str(sub).replace("Series", "series"), s)   # lowercased watch root
    assert any(Path(p).name == "gone.mp4" for p in orph)
    s.close()


def test_watch_once_indexes_matches_notifies_and_self_heals(tmp_path, monkeypatch):
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    new = _vid(tmp_path / "movie.mp4"); _age(tmp_path / "movie.mp4")
    s.save(_rec(tmp_path / "gone.mp4"), feature_version="x")     # deleted file -> orphan to forget

    calls: dict = {}
    monkeypatch.setattr(watch, "analyze_file",
                        lambda p, *a, **k: calls.setdefault("analyzed", []).append(p))
    monkeypatch.setattr(watch, "_pass2", lambda *a, **k: calls.__setitem__("pass2", True))
    monkeypatch.setattr(watch, "_apply_name_grouping", lambda *a, **k: None)
    other = str(tmp_path / "other.mp4")
    cl = {"cluster_id": 0, "keep": new, "discard": [other],
          "evidence": {new: "KEEP", other: "discard"}}           # cluster contains the new file
    monkeypatch.setattr(watch, "_rebuild_clusters", lambda *a, **k: [cl])

    notified: list = []
    res = watch.watch_once(str(tmp_path), s, _DummyEmbedder(), th,
                           on_duplicate=notified.append, stable_s=15.0)
    assert res.indexed == 1 and res.removed == 1                 # indexed the new, forgot the orphan
    assert new in calls["analyzed"] and calls.get("pass2") is True
    assert res.dup_clusters == [cl] and notified == [[cl]]       # notified about the new duplicate
    assert s.load(str(tmp_path / "gone.mp4")) is None            # orphan actually forgotten
    s.close()


def test_watch_once_survives_corrupt_file(tmp_path, monkeypatch):
    """A file that fails analysis is skipped-and-reported, never crashes the cycle (§2)."""
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    _vid(tmp_path / "bad.mp4"); _age(tmp_path / "bad.mp4")

    def _boom(*a, **k):
        raise RuntimeError("ffprobe failed")
    monkeypatch.setattr(watch, "analyze_file", _boom)
    res = watch.watch_once(str(tmp_path), s, _DummyEmbedder(), th, stable_s=15.0)
    assert res.indexed == 0 and len(res.errors) == 1            # reported, not raised
    s.close()


def test_watch_loop_stops_and_polls(tmp_path, monkeypatch):
    """watch_loop runs cycles until stop() is True; sleep is injected (no real waiting)."""
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    cycles = {"n": 0}
    monkeypatch.setattr(watch, "watch_once",
                        lambda *a, **k: watch.CycleResult(indexed=cycles.__setitem__("n", cycles["n"] + 1) or 0))
    stops = iter([False, False, True])
    watch.watch_loop(str(tmp_path), s, _DummyEmbedder(), th, interval=0.0,
                     sleep=lambda _: None, stop=lambda: next(stops))
    assert cycles["n"] == 2                                      # two cycles, then stop
    s.close()


def test_watch_loop_backoff_grows_idle_resets_on_activity(tmp_path, monkeypatch):
    """Idle cycles back off (×backoff up to max_interval) so a static library is barely touched;
    a cycle with activity resets to the base interval."""
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    intervals: list = []
    results = iter([watch.CycleResult(), watch.CycleResult(),
                    watch.CycleResult(indexed=1), watch.CycleResult()])
    monkeypatch.setattr(watch, "watch_once", lambda *a, **k: next(results))
    stops = iter([False, False, False, False, True])
    watch.watch_loop(str(tmp_path), s, _DummyEmbedder(), th,
                     interval=10, max_interval=100, backoff=2.0,
                     sleep=intervals.append, stop=lambda: next(stops))
    assert intervals == [20, 40, 10, 20]            # idle 20, idle 40, active->reset 10, idle 20
    s.close()


def test_watch_loop_wake_event_resets_cadence(tmp_path, monkeypatch):
    """A filesystem event (wake.set) makes the loop reconcile immediately and reset to fast cadence."""
    import threading
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    monkeypatch.setattr(watch, "watch_once", lambda *a, **k: watch.CycleResult())   # always idle
    wake = threading.Event(); wake.set()            # pretend an event already arrived
    waits: list = []
    real_wait = wake.wait
    monkeypatch.setattr(wake, "wait", lambda t: (waits.append(t), real_wait(0))[1])
    stops = iter([False, False, True])
    watch.watch_loop(str(tmp_path), s, _DummyEmbedder(), th, interval=10, max_interval=100,
                     backoff=2.0, wake=wake, stop=lambda: next(stops))
    # iter1 idle -> wait 20, woken (event set) -> backoff RESETS; iter2 idle -> 20 again (not 40).
    # Without the wake reset it would grow [20, 40] -> the event keeps the cadence fast.
    assert waits == [20, 20]
    s.close()


def test_watch_loop_yields_while_scan_in_progress(tmp_path, monkeypatch):
    """A user scan has priority: while scan_in_progress() is True the loop must NOT run watch_once
    (no disk/DB contention); it resumes the cycle once the scan releases the lock."""
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    calls = {"n": 0}
    monkeypatch.setattr(watch, "watch_once",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or watch.CycleResult()))
    flags = iter([True, False])            # scan running on cycle 1, free on cycle 2
    monkeypatch.setattr(watch, "scan_in_progress", lambda: next(flags))
    stops = iter([False, False, True])
    watch.watch_loop(str(tmp_path), s, _DummyEmbedder(), th, interval=10,
                     sleep=lambda _t: None, stop=lambda: next(stops))
    assert calls["n"] == 1                 # skipped the scan cycle, ran only the free one
    s.close()


def test_start_fs_events_graceful(tmp_path):
    """watchdog is optional: returns a stop() callable if installed, else None — never raises."""
    import threading
    r = watch.start_fs_events(str(tmp_path), threading.Event())
    assert r is None or callable(r)
    if callable(r):
        r()
