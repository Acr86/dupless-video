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
    def __init__(self):
        self.freed = 0
    @property
    def feature_version(self) -> str:
        return "fvtest"
    def free_cache(self) -> None:                        # idle GPU-cache release (no-op on the stub)
        self.freed += 1


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


def _ctx(targets, store, th, embedder=None) -> "watch.WatchContext":
    """Bundle the watcher deps the way callers do now (targets/store/embedder/th)."""
    return watch.WatchContext(str(targets), store, embedder or _DummyEmbedder(), th)


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


def test_pending_files_skips_unchanged_corrupt(tmp_path):
    """A file that already FAILED analysis (corrupt) and hasn't changed must NOT be re-attempted every
    sweep (it would just fail again and never let the count complete). A re-download lifts the guard."""
    th = load_thresholds()
    fv = feature_version(_DummyEmbedder(), False,
                         audio_fp_cap_s=th.audio_fp_cap_s, audio_fp_cap_above_s=th.audio_fp_cap_above_s)
    s = FingerprintStore(tmp_path / "w.sqlite")
    bad = _vid(tmp_path / "bad.mp4"); _age(tmp_path / "bad.mp4")
    s.save_problem(bad, "moov atom not found", "corrupt")          # already attempted -> failed
    assert watch.pending_files(str(tmp_path), s, fv, stable_s=15.0) == []   # unchanged corrupt -> skip
    Path(bad).write_bytes(b"y" * 4096); _age(tmp_path / "bad.mp4")          # re-downloaded (size changed)
    assert bad in watch.pending_files(str(tmp_path), s, fv, stable_s=15.0)  # changed -> retry
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


def test_orphan_paths_skips_unreachable_root(tmp_path):
    """§2 offline-drive guard: a watched root that is currently UNREACHABLE (unmounted Storage Space
    / offline share) is skipped, so its indexed files are NOT read as a mass deletion. A momentary
    unmount must never wipe the index; once the root is reachable, missing files count as orphans."""
    s = FingerprintStore(tmp_path / "w.sqlite")
    s.save(_rec(tmp_path / "library" / "movie.mp4"), feature_version="fv")   # never created on disk
    missing_root = str(tmp_path / "library")                                 # root does not exist yet
    assert watch.orphan_paths(missing_root, s) == []                         # unreachable -> no orphans
    (tmp_path / "library").mkdir()                                           # root comes online
    orph = watch.orphan_paths(missing_root, s)
    assert str(tmp_path / "library" / "movie.mp4") in orph                   # now it IS an orphan
    s.close()


def test_full_sweep_closes_mode_b_deleted_root_volume_online(tmp_path, monkeypatch):
    """Mode B in the background: `orphan_paths` SKIPS a whole DELETED watched root (its root guard sees
    the root gone, so an unmount isn't read as a deletion), but `_full_sweep` then runs the store's
    VOLUME-guarded `prune_missing_files`, which cleans the files since the drive is still mounted —
    closing the blind spot orphan_paths leaves open. scan gate forced off for isolation."""
    monkeypatch.setattr(watch, "scan_in_progress", lambda: False)
    s = FingerprintStore(tmp_path / "w.sqlite")
    root = tmp_path / "Media"                                    # never created on disk == deleted root
    s.save(_rec(root / "ep.mkv"), feature_version="fv")          # indexed under the (now gone) root
    assert watch.orphan_paths(str(root), s) == []               # orphan_paths bows out (root unreachable)
    removed = watch._full_sweep([str(root)], s)                 # the full sweep prunes by VOLUME instead
    assert removed == 1 and str(root / "ep.mkv") not in s.all_paths()
    s.close()


def test_full_sweep_skips_prune_during_scan(tmp_path, monkeypatch):
    """§0/§1: the backstop volume-prune is SKIPPED while a scan holds the lock — don't race Pass-2's
    re-persist guard or add an O(library) stat sweep to the HDD the scan is reading. The next backstop
    sweep (or the UI prune) reconciles once the scan finishes."""
    monkeypatch.setattr(watch, "scan_in_progress", lambda: True)
    s = FingerprintStore(tmp_path / "w.sqlite")
    p = tmp_path / "Media" / "ep.mkv"
    s.save(_rec(p), feature_version="fv")                        # Mode B file, never on disk
    removed = watch._full_sweep([str(tmp_path / "Media")], s)
    assert removed == 0 and str(p) in s.all_paths()             # kept until the scan releases the lock
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
    res = watch.watch_once(_ctx(tmp_path, s, th), watch.WatchTuning(stable_s=15.0),
                           on_duplicate=notified.append)
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
    res = watch.watch_once(_ctx(tmp_path, s, th), watch.WatchTuning(stable_s=15.0))
    assert res.indexed == 0 and len(res.errors) == 1            # reported, not raised
    s.close()


def test_watch_loop_stops_and_polls(tmp_path, monkeypatch):
    """watch_loop runs cycles until stop() is True; sleep is injected (no real waiting)."""
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    monkeypatch.setattr(watch, "scan_in_progress", lambda: False)   # isolate from a stale real scan.lock
    cycles = {"n": 0}
    monkeypatch.setattr(watch, "reconcile_removals", lambda *a, **k: watch.CycleResult())
    monkeypatch.setattr(watch, "ingest_new",
                        lambda *a, **k: watch.CycleResult(indexed=cycles.__setitem__("n", cycles["n"] + 1) or 0))
    stops = iter([False, False, True])
    watch.watch_loop(_ctx(tmp_path, s, th), tuning=watch.WatchTuning(interval=0.0),
                     sleep=lambda _: None, stop=lambda: next(stops))
    assert cycles["n"] == 2                                      # two cycles, then stop
    s.close()


def test_watch_loop_backoff_grows_idle_resets_on_activity(tmp_path, monkeypatch):
    """Idle cycles back off (×backoff up to max_interval) so a static library is barely touched;
    a cycle with activity resets to the base interval."""
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    monkeypatch.setattr(watch, "scan_in_progress", lambda: False)   # isolate from a stale real scan.lock
    intervals: list = []
    results = iter([watch.CycleResult(), watch.CycleResult(),
                    watch.CycleResult(indexed=1), watch.CycleResult()])
    monkeypatch.setattr(watch, "reconcile_removals", lambda *a, **k: watch.CycleResult())
    monkeypatch.setattr(watch, "ingest_new", lambda *a, **k: next(results))
    stops = iter([False, False, False, False, True])
    watch.watch_loop(_ctx(tmp_path, s, th),
                     tuning=watch.WatchTuning(interval=10, max_interval=100, backoff=2.0),
                     sleep=intervals.append, stop=lambda: next(stops))
    assert intervals == [20, 40, 10, 20]            # idle 20, idle 40, active->reset 10, idle 20
    s.close()


def test_watch_loop_wake_event_resets_cadence(tmp_path, monkeypatch):
    """A filesystem event (wake.set) makes the loop reconcile immediately and reset to fast cadence."""
    import threading
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    monkeypatch.setattr(watch, "reconcile_removals", lambda *a, **k: watch.CycleResult())
    monkeypatch.setattr(watch, "ingest_new", lambda *a, **k: watch.CycleResult())   # always idle
    wake = threading.Event(); wake.set()            # pretend an event already arrived
    waits: list = []
    real_wait = wake.wait
    monkeypatch.setattr(wake, "wait", lambda t: (waits.append(t), real_wait(0))[1])
    stops = iter([False, False, True])
    watch.watch_loop(_ctx(tmp_path, s, th),
                     tuning=watch.WatchTuning(interval=10, max_interval=100, backoff=2.0),
                     wake=wake, stop=lambda: next(stops))
    # iter1 idle -> wait 20, woken (event set) -> backoff RESETS; iter2 idle -> 20 again (not 40).
    # Without the wake reset it would grow [20, 40] -> the event keeps the cadence fast.
    assert waits == [20, 20]
    s.close()


def test_watch_loop_reconciles_removals_during_scan_defers_ingest(tmp_path, monkeypatch):
    """Split-priority contract: while a user scan holds the lock, the loop STILL runs
    reconcile_removals every cycle (a trashed file leaves the list) but DEFERS the heavy ingest_new
    until the scan releases the lock — deletions stay responsive without disk thrashing."""
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    calls = {"reconcile": 0, "ingest": 0}
    monkeypatch.setattr(watch, "reconcile_removals",
                        lambda *a, **k: (calls.__setitem__("reconcile", calls["reconcile"] + 1)
                                         or watch.CycleResult()))
    monkeypatch.setattr(watch, "ingest_new",
                        lambda *a, **k: (calls.__setitem__("ingest", calls["ingest"] + 1)
                                         or watch.CycleResult()))
    flags = iter([True, False])            # scan running on cycle 1, free on cycle 2
    monkeypatch.setattr(watch, "scan_in_progress", lambda: next(flags))
    stops = iter([False, False, True])
    watch.watch_loop(_ctx(tmp_path, s, th), tuning=watch.WatchTuning(interval=10),
                     sleep=lambda _t: None, stop=lambda: next(stops))
    assert calls["reconcile"] == 2         # ran BOTH cycles (deletions reconciled even during a scan)
    assert calls["ingest"] == 1            # heavy indexing only on the free cycle


def test_reconcile_removals_forgets_orphans_without_decode(tmp_path, monkeypatch):
    """reconcile_removals is the cheap, decode-free half: it forgets orphans and rebuilds clusters
    but NEVER calls analyze_file -> safe to run during a scan without thrashing the disk."""
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    s.save(_rec(tmp_path / "gone.mp4"), feature_version="x")     # orphan to forget
    monkeypatch.setattr(watch, "analyze_file",
                        lambda *a, **k: pytest.fail("reconcile_removals must not decode"))
    monkeypatch.setattr(watch, "_apply_name_grouping", lambda *a, **k: None)
    monkeypatch.setattr(watch, "_rebuild_clusters", lambda *a, **k: [])
    res = watch.reconcile_removals(_ctx(tmp_path, s, th))
    assert res.removed == 1 and res.indexed == 0
    assert s.load(str(tmp_path / "gone.mp4")) is None            # orphan actually forgotten
    s.close()


def test_start_fs_events_graceful(tmp_path):
    """watchdog is optional: returns a stop() callable if installed, else None — never raises, and
    accepts the `deleted` queue without error."""
    import threading
    from collections import deque
    r = watch.start_fs_events(str(tmp_path), threading.Event(), deque())
    assert r is None or callable(r)
    if callable(r):
        r()


# ------------------------------------------------- event-driven deletion (O(borrados), no full sweep)

def test_drain_deleted_forgets_gone_skips_present(tmp_path):
    """The cheap event drain forgets only paths that truly left the disk, is idempotent on a repeated
    event, leaves present files untouched, and fully empties the queue."""
    from collections import deque
    s = FingerprintStore(tmp_path / "w.sqlite")
    present = _vid(tmp_path / "here.mp4"); st = os.stat(present)
    s.save(_rec(present, st.st_mtime, st.st_size), feature_version="fv")
    s.save(_rec(tmp_path / "gone.mp4"), feature_version="fv")        # not on disk
    gone = str(tmp_path / "gone.mp4")
    q = deque([gone, present, gone])                                 # gone twice + a present file
    removed = watch._drain_deleted(q, s)
    assert removed == 1                                              # forgotten once (idempotent)
    assert s.load(gone) is None and s.load(present) is not None      # only the gone file dropped
    assert len(q) == 0                                              # queue fully drained
    s.close()


def test_reconcile_removals_event_drain_skips_full_sweep(tmp_path, monkeypatch):
    """With a `deleted` queue and full=False, reconcile drains events and must NOT run the O(library)
    orphan_paths sweep — that is the whole point of the event-driven path."""
    from collections import deque
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    s.save(_rec(tmp_path / "gone.mp4"), feature_version="fv")
    monkeypatch.setattr(watch, "orphan_paths",
                        lambda *a, **k: pytest.fail("full sweep must not run when draining events"))
    monkeypatch.setattr(watch, "_apply_name_grouping", lambda *a, **k: None)
    monkeypatch.setattr(watch, "_rebuild_clusters", lambda *a, **k: [])
    res = watch.reconcile_removals(_ctx(tmp_path, s, th), deleted=deque([str(tmp_path / "gone.mp4")]))
    assert res.removed == 1
    s.close()


def test_reconcile_removals_full_sweep_on_backstop_and_no_watchdog(tmp_path, monkeypatch):
    """The O(library) sweep runs when full=True (startup/backstop) and when deleted is None (no
    watchdog), but not on a plain event-drain cycle."""
    from collections import deque
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    swept = {"n": 0}
    monkeypatch.setattr(watch, "orphan_paths",
                        lambda *a, **k: (swept.__setitem__("n", swept["n"] + 1) or []))
    ctx = _ctx(tmp_path, s, th)
    watch.reconcile_removals(ctx, deleted=deque(), full=True)    # backstop
    watch.reconcile_removals(ctx, deleted=None)                  # no watchdog
    watch.reconcile_removals(ctx, deleted=deque(), full=False)   # plain drain -> skip
    assert swept["n"] == 2
    s.close()


def test_watch_loop_no_full_sweep_during_scan(tmp_path, monkeypatch):
    """While a scan holds the lock, the loop runs only the cheap drain (full=False); the O(library)
    backstop sweep is reserved for idle cycles (startup sweep fires on the first free cycle)."""
    from collections import deque
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    fulls: list = []
    monkeypatch.setattr(watch, "reconcile_removals",
                        lambda *a, **k: (fulls.append(k.get("full")) or watch.CycleResult()))
    monkeypatch.setattr(watch, "ingest_new", lambda *a, **k: watch.CycleResult())
    flags = iter([True, True, False])                  # scan running cycles 1-2, free on cycle 3
    monkeypatch.setattr(watch, "scan_in_progress", lambda: next(flags))
    stops = iter([False, False, False, True])
    watch.watch_loop(_ctx(tmp_path, s, th), tuning=watch.WatchTuning(interval=10), deleted=deque(),
                     sleep=lambda _t: None, stop=lambda: next(stops))
    assert fulls == [False, False, True]               # no sweep during the scan; startup sweep when free
    s.close()


def test_watch_loop_frees_gpu_cache_when_idle(tmp_path, monkeypatch):
    """An idle cycle (nothing embedded) releases the embedder's cached GPU VRAM, so a 24/7 watcher
    stops squatting the GPU between cycles."""
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    emb = _DummyEmbedder()
    ctx = watch.WatchContext(str(tmp_path), s, emb, th)
    monkeypatch.setattr(watch, "reconcile_removals", lambda *a, **k: watch.CycleResult())
    monkeypatch.setattr(watch, "ingest_new", lambda *a, **k: watch.CycleResult())     # idle
    stops = iter([False, False, True])
    watch.watch_loop(ctx, tuning=watch.WatchTuning(interval=10),
                     sleep=lambda _t: None, stop=lambda: next(stops))
    assert emb.freed == 2                                # freed on each of the 2 idle cycles
    s.close()


def test_watch_loop_keeps_gpu_cache_while_indexing(tmp_path, monkeypatch):
    """An active cycle (files embedded) KEEPS the cache — releasing it mid-burst would force a
    re-alloc next file. Only idle releases."""
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    emb = _DummyEmbedder()
    ctx = watch.WatchContext(str(tmp_path), s, emb, th)
    monkeypatch.setattr(watch, "scan_in_progress", lambda: False)   # isolate from a stale real scan.lock
    monkeypatch.setattr(watch, "reconcile_removals", lambda *a, **k: watch.CycleResult())
    monkeypatch.setattr(watch, "ingest_new", lambda *a, **k: watch.CycleResult(indexed=3))  # active
    stops = iter([False, True])
    watch.watch_loop(ctx, tuning=watch.WatchTuning(interval=10),
                     sleep=lambda _t: None, stop=lambda: next(stops))
    assert emb.freed == 0                                # active ingest keeps the cache for throughput
    s.close()


# ------------------------------------------------- Fast-Lane scheduler (A+B): two-lane ingest order

def test_scheduler_fast_lane_first_and_capped(tmp_path):
    """FAST lane (watchdog events) is processed before the backlog, and the chunk cap bounds the
    cycle; backlog top-up is OLDEST-first."""
    from collections import deque
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    b1 = _vid(tmp_path / "b1.mp4"); _age(tmp_path / "b1.mp4", 300)        # oldest
    _vid(tmp_path / "b2.mp4"); _age(tmp_path / "b2.mp4", 200)             # newer backlog
    fnew = _vid(tmp_path / "fnew.mp4"); _age(tmp_path / "fnew.mp4", 50)   # the just-dropped file
    ctx = watch.WatchContext(str(tmp_path), s, _DummyEmbedder(), th)
    sched = watch._IngestScheduler(changed=deque([fnew]), chunk=2)
    out = sched.next_chunk(ctx, "fv", watch.WatchTuning(stable_s=15.0), now=time.time())
    assert out[0] == fnew                                # fast lane wins the front
    assert len(out) == 2 and b1 in out                   # capped at 2; topped up with the OLDEST backlog
    assert str(tmp_path / "b2.mp4") not in out           # newer backlog waits (oldest-first)
    s.close()


def test_scheduler_backlog_oldest_first(tmp_path):
    """With no events, the backlog is consumed oldest-first (monotonic march to 100%)."""
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    for name, age in [("new.mp4", 50), ("mid.mp4", 150), ("old.mp4", 400)]:
        _vid(tmp_path / name); _age(tmp_path / name, age)
    ctx = watch.WatchContext(str(tmp_path), s, _DummyEmbedder(), th)
    sched = watch._IngestScheduler(changed=None, chunk=10)
    out = [Path(p).name for p in sched.next_chunk(ctx, "fv", watch.WatchTuning(stable_s=15.0), now=time.time())]
    assert out == ["old.mp4", "mid.mp4", "new.mp4"]      # oldest -> newest
    s.close()


def test_scheduler_midcopy_requeued_not_lost(tmp_path):
    """A fast-lane file still being copied (recent mtime) is NOT read this cycle but stays queued;
    once it settles it's returned. Never dropped."""
    from collections import deque
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    f = _vid(tmp_path / "copying.mp4")                   # fresh mtime -> mid-copy
    ctx = watch.WatchContext(str(tmp_path), s, _DummyEmbedder(), th)
    sched = watch._IngestScheduler(changed=deque([f]), chunk=4)
    tun = watch.WatchTuning(stable_s=15.0)
    assert sched.next_chunk(ctx, "fv", tun, now=time.time()) == []   # mid-copy -> not ready
    assert f in sched._fast                              # but kept (re-queued)
    _age(tmp_path / "copying.mp4", 100)                  # copy finished -> settles
    assert sched.next_chunk(ctx, "fv", tun, now=time.time()) == [f]  # now indexed
    s.close()


def test_scheduler_dedups_repeated_events(tmp_path):
    """A copy fires many 'modified' events for the same path -> the fast lane processes it ONCE."""
    from collections import deque
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    f = _vid(tmp_path / "x.mp4"); _age(tmp_path / "x.mp4", 100)
    ctx = watch.WatchContext(str(tmp_path), s, _DummyEmbedder(), th)
    sched = watch._IngestScheduler(changed=deque([f, f, f]), chunk=10)
    out = sched.next_chunk(ctx, "fv", watch.WatchTuning(stable_s=15.0), now=time.time())
    assert out.count(f) == 1                             # deduped
    s.close()


def test_scheduler_discovery_decoupled_no_rewalk_per_chunk(tmp_path, monkeypatch):
    """The rglob walk runs at MOST once per discovery interval, NOT per chunk — otherwise a chunked
    backlog would thrash the disk re-walking the whole tree every few files."""
    th = load_thresholds(); s = FingerprintStore(tmp_path / "w.sqlite")
    files = [str(tmp_path / f"f{i}.mp4") for i in range(5)]
    walks = {"n": 0}
    monkeypatch.setattr(watch, "pending_files",
                        lambda *a, **k: (walks.__setitem__("n", walks["n"] + 1) or list(files)))
    # has_fresh is real (files don't exist on disk -> _ready drops them), so stub _ready to "ready"
    monkeypatch.setattr(watch._IngestScheduler, "_ready", staticmethod(lambda *a, **k: (True, True)))
    ctx = watch.WatchContext(str(tmp_path), s, _DummyEmbedder(), th)
    sched = watch._IngestScheduler(changed=None, chunk=1, discovery_interval=10_000)
    tun = watch.WatchTuning(stable_s=15.0)
    sched.next_chunk(ctx, "fv", tun, now=1000.0)         # discovers (walk 1), consumes f0
    sched.next_chunk(ctx, "fv", tun, now=1001.0)         # backlog not exhausted, interval not elapsed
    sched.next_chunk(ctx, "fv", tun, now=1002.0)         # -> still no re-walk
    assert walks["n"] == 1                               # walked ONCE, not per chunk
    s.close()
