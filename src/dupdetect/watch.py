"""Background watcher — keeps the store up to date so the user only opens the app to SEE results.

POLLING + reconcile (no filesystem-event dependency): robust on Storage Space / network shares
where FS events are flaky, and crash-safe because the store's own FRESHNESS is the queue — an
interrupted file is simply still 'pending' next cycle, no separate queue to corrupt.

Each cycle (`watch_once`):
  1. self-heal — forget files deleted from disk (no ghost clusters).
  2. find NEW/CHANGED files that are STABLE (mtime settled -> not mid-copy).
  3. Pass-1 each (skip-and-report on corrupt — never crash the loop, §2).
  4. incrementally MATCH the new files against the index + rebuild the affected clusters
     (rank_cluster re-ranks them, so the best-quality KEEP stays current — no full re-scan needed).
  5. notify on freshly-formed duplicate clusters (via the `on_duplicate` callback).

A full re-verify is ONLY for threshold/model (`feature_version`) changes or deep self-heal — NOT a
routine the user should be nagged to run.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from dupdetect.config import Thresholds
from dupdetect.features.embeddings import Embedder
from dupdetect.match.retrieval import CoarseIndex
from dupdetect.pipeline.analyze import analyze_file, feature_version
from dupdetect.pipeline.fullscan import (
    _apply_name_grouping,
    _pass2,
    _rebuild_clusters,
    collect_videos,
)
from dupdetect.runtime import scan_in_progress
from dupdetect.store import FingerprintStore

# Default: a file must be unmodified for this long before we touch it (avoid mid-copy reads).
DEFAULT_STABLE_S = 15.0
DEFAULT_INTERVAL_S = 60.0          # base poll cadence
DEFAULT_MAX_INTERVAL_S = 1800.0    # idle backoff cap (30 min) — barely touch a static library
DEFAULT_BACKOFF = 2.0              # grow the wait ×this after each idle cycle


def _as_list(targets) -> list[str]:
    return [str(t) for t in targets] if isinstance(targets, (list, tuple, set)) else [str(targets)]


def pending_files(targets, store: FingerprintStore, fv: str, recursive: bool = True,
                  stable_s: float = DEFAULT_STABLE_S, now: float | None = None) -> list[str]:
    """On-disk videos that are NEW or CHANGED vs the store AND stable. A file modified within the
    last `stable_s` seconds is skipped (still being copied) -> picked up once it settles. `now`
    injectable for tests."""
    now = time.time() if now is None else now
    out: list[str] = []
    for p in collect_videos(targets, recursive=recursive):
        try:
            st = os.stat(p)
        except OSError:
            continue                               # vanished between listing and stat -> skip
        if now - st.st_mtime < stable_s:           # still being written -> wait a cycle
            continue
        if not store.has_fresh(p, st, fv):         # new or changed (mtime/size/feature_version)
            out.append(p)
    return out


def _norm(path: str) -> str:
    """Absolute + case/separator-normalized. `os.path.normcase` folds case AND maps '/'->'\\' on
    Windows, and is the IDENTITY on case-sensitive POSIX -> path comparisons are robust to how the
    path was written without breaking Linux/macOS."""
    return os.path.normcase(os.path.abspath(path))


def orphan_paths(targets, store: FingerprintStore) -> list[str]:
    """Indexed paths under `targets` whose file is gone from disk -> to forget (self-heal).

    The root match is NORMALIZED (case + separators, via `os.path.normcase`): on Windows the
    filesystem is case-insensitive but the STORED path (from the scan) and the WATCHED root string
    can differ in case — without normalizing, a trashed file under a differently-cased root is never
    detected and lingers in the list. A trailing-separator boundary keeps a sibling folder that only
    shares a name prefix from matching (e.g. 'Series' must not swallow 'Series2')."""
    roots = [_norm(t).rstrip(os.sep) for t in _as_list(targets)]
    out: list[str] = []
    for p in store.all_paths():
        ap = _norm(p)
        if any(ap == r or ap.startswith(r + os.sep) for r in roots) and not os.path.exists(p):
            out.append(p)
    return out


@dataclass
class CycleResult:
    indexed: int = 0
    removed: int = 0
    dup_clusters: list = field(default_factory=list)   # dup clusters that include a new file
    errors: list = field(default_factory=list)         # (path, error) for skipped/corrupt files


def _cluster_members(cl: dict) -> list[str]:
    """All paths in a rebuilt cluster (rank_cluster's evidence is keyed by every member)."""
    ev = cl.get("evidence") or {}
    if ev:
        return list(ev.keys())
    keep = cl.get("keep")
    return list(cl.get("discard", [])) + ([keep] if keep else [])


def _affected_dup_clusters(new_paths, clusters: list[dict]) -> list[dict]:
    """Of the rebuilt clusters, those that include one of the just-processed files. Every cluster
    here is already a duplicate group (union-find only unions duplicate verdicts)."""
    new = {os.path.abspath(p) for p in new_paths}
    return [cl for cl in clusters
            if any(os.path.abspath(m) in new for m in _cluster_members(cl))]


def watch_once(targets, store: FingerprintStore, embedder: Embedder, th: Thresholds, *,
               on_duplicate: Optional[Callable[[list], None]] = None,
               on_detect: Optional[Callable[[int], None]] = None,
               stable_s: float = DEFAULT_STABLE_S, independent_scenes: bool = False,
               recursive: bool = True) -> CycleResult:
    """One reconcile cycle. Returns a CycleResult. Builds the coarse index ONLY when there's new
    work (idle cycles are cheap). Never raises on a single bad file (skip-and-report). `on_detect(n)`
    fires with the count of NEW/changed files found, BEFORE the (slow) indexing — lets the UI show
    'detected N — indexing…' live instead of only the post-cycle total."""
    fv = feature_version(embedder, independent_scenes,
                         audio_fp_cap_s=th.audio_fp_cap_s, audio_fp_cap_above_s=th.audio_fp_cap_above_s)
    res = CycleResult()

    gone = orphan_paths(targets, store)
    for p in gone:
        store.forget_file(p)
        res.removed += 1

    todo = pending_files(targets, store, fv, recursive=recursive, stable_s=stable_s)
    if todo and on_detect:
        on_detect(len(todo))                       # surface detection before the slow per-file work
    if not todo:
        if res.removed:                            # deletions can change/empty clusters -> rebuild
            _apply_name_grouping(store, th)
            _rebuild_clusters(store, th)
        return res

    done: list[str] = []
    for p in todo:
        try:
            analyze_file(p, store, embedder, th, independent_scenes=independent_scenes)
            done.append(p)
            res.indexed += 1
        except Exception as e:                     # noqa: BLE001 — §2 skip-and-report, keep the loop alive
            res.errors.append((p, str(e)))
    if not done:
        return res

    # Incremental match: build the index ONCE (now includes the just-indexed files, so intra-batch
    # duplicates are caught), match the new files, then rebuild the affected clusters.
    ap, gv = store.all_global_vecs()
    wo, wv = store.all_window_vecs()
    index = CoarseIndex(dim=gv.shape[1] if gv.size else th.raw["embeddings"]["dim"])
    index.build(ap, gv, window_owners=wo, window_vecs=wv)
    _pass2(done, store, index, th, False)
    _apply_name_grouping(store, th)
    clusters = _rebuild_clusters(store, th)
    res.dup_clusters = _affected_dup_clusters(done, clusters)
    if on_duplicate and res.dup_clusters:
        on_duplicate(res.dup_clusters)
    return res


def start_fs_events(targets, wake) -> Optional[Callable[[], None]]:
    """Subscribe to filesystem events under `targets` (native: ReadDirectoryChangesW on Windows,
    inotify on Linux, FSEvents on macOS — via `watchdog`) and set `wake` on ANY change so the loop
    reconciles immediately. Returns a stop() callable, or None if `watchdog` isn't installed (the
    loop then relies on backoff polling alone). Events are a TRIGGER, not the source of truth — the
    reconcile still decides what changed, so a missed/overflowed event only delays, never corrupts."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except Exception:                              # noqa: BLE001 watchdog optional -> polling only
        return None

    class _Wake(FileSystemEventHandler):
        def on_any_event(self, _event):
            wake.set()

    obs = Observer()
    handler = _Wake()
    scheduled = 0
    for t in _as_list(targets):
        try:
            obs.schedule(handler, t, recursive=True)
            scheduled += 1
        except OSError:
            continue                               # path gone / not watchable -> skip
    if not scheduled:
        return None
    obs.daemon = True
    obs.start()
    return obs.stop


def watch_loop(targets, store: FingerprintStore, embedder: Embedder, th: Thresholds, *,
               interval: float = DEFAULT_INTERVAL_S, max_interval: float = DEFAULT_MAX_INTERVAL_S,
               backoff: float = DEFAULT_BACKOFF,
               on_duplicate: Optional[Callable[[list], None]] = None,
               on_cycle: Optional[Callable[[CycleResult], None]] = None,
               on_detect: Optional[Callable[[int], None]] = None,
               stable_s: float = DEFAULT_STABLE_S, independent_scenes: bool = False,
               recursive: bool = True, sleep: Callable[[float], None] = time.sleep,
               wake=None, stop: Optional[Callable[[], bool]] = None) -> None:
    """Reconcile until `stop()` returns True (or forever). IDLE BACKOFF: after a cycle that found
    nothing, the wait grows (×`backoff`, capped at `max_interval`) so a library that rarely changes
    is barely touched (saves power / avoids waking the disk); any activity resets it to `interval`.
    `wake` (a threading.Event set by a filesystem-event source) triggers an IMMEDIATE reconcile and
    resets the cadence -> instant reaction, with polling only as the overflow/offline safety net.
    `sleep`/`stop` injectable for tests. A failing cycle is reported and does not kill the loop."""
    cur = interval
    paused = False
    while not (stop and stop()):                   # checked once per cycle (at the top)
        if scan_in_progress():                     # a user scan has ABSOLUTE priority -> yield the
            if not paused:                         # disk/DB (no thrashing, no write starvation).
                print("[watch] paused — a scan is running (priority)", flush=True)
                paused = True
            sleep(min(cur, 3.0))                   # re-check soon; the watcher is idempotent, nothing lost
            continue
        if paused:
            print("[watch] resumed", flush=True)
            paused = False
        try:
            res = watch_once(targets, store, embedder, th, on_duplicate=on_duplicate,
                             on_detect=on_detect, stable_s=stable_s,
                             independent_scenes=independent_scenes, recursive=recursive)
        except Exception as e:                     # noqa: BLE001 a cycle error must not stop the watcher
            res = CycleResult(errors=[("<cycle>", str(e))])
        if on_cycle:
            on_cycle(res)
        cur = interval if (res.indexed or res.removed or res.errors) else min(cur * backoff, max_interval)
        if wake is not None:                       # event-driven wait: reconcile early on a FS event
            if wake.wait(cur):
                cur = interval                     # a real change arrived -> back to fast cadence
            wake.clear()
        else:
            sleep(cur)
