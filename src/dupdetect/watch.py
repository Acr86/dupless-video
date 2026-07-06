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
from dupdetect.pipeline.analyze import analysis_state, analyze_file, feature_version
from dupdetect.pipeline.fullscan import (
    _apply_name_grouping,
    _pass2,
    _rebuild_clusters,
    _snapshot_clusters,
    collect_videos,
)
from dupdetect.runtime import scan_in_progress
from dupdetect.store import FingerprintStore
from dupdetect.store.store import canonical_path

# Default: a file must be unmodified for this long before we touch it (avoid mid-copy reads).
DEFAULT_STABLE_S = 15.0
DEFAULT_INTERVAL_S = 60.0          # base poll cadence
DEFAULT_MAX_INTERVAL_S = 1800.0    # idle backoff cap (30 min) — barely touch a static library
DEFAULT_BACKOFF = 2.0              # grow the wait ×this after each idle cycle
DEFERRED_POLL_S = 10.0             # no-watchdog fallback: cap the wait so a poll still sees deletions
FULL_SWEEP_EVERY = 30              # idle cycles between O(library) backstop sweeps (catch missed events)
DEFAULT_INGEST_CHUNK = 4           # files analyzed per cycle -> bounds the wait before the fast lane is
#                                  # re-checked (~N×15-23s on HDD). Keeps a just-dropped file responsive
#                                  # even behind a huge backlog. A measured lever (§3): raise to amortize
#                                  # the per-cycle index/cluster rebuild if it dominates.
DEFAULT_DISCOVERY_INTERVAL_S = 300.0   # min seconds between full rglob backlog walks — discovery is
#                                  # DECOUPLED from processing so a chunked backlog never re-walks the
#                                  # whole tree per chunk (that would thrash the very HDD the scan needs).


@dataclass
class WatchContext:
    """What the watcher operates on, as ONE unit — was the 4 args (targets/store/embedder/th) copied
    through reconcile_removals / ingest_new / watch_once / watch_loop. Threading it kills that clump."""
    targets: object
    store: FingerprintStore
    embedder: Embedder
    th: Thresholds


@dataclass
class WatchTuning:
    """Cadence + analysis knobs, with their defaults in one place (was 6 loose positional/keyword args
    on watch_loop). A measured lever each (§3): poll cadence, idle backoff, mid-copy debounce, etc."""
    interval: float = DEFAULT_INTERVAL_S
    max_interval: float = DEFAULT_MAX_INTERVAL_S
    backoff: float = DEFAULT_BACKOFF
    stable_s: float = DEFAULT_STABLE_S
    independent_scenes: bool = False
    recursive: bool = True
    ingest_chunk: int = DEFAULT_INGEST_CHUNK            # files per cycle (fast-lane responsiveness)
    discovery_interval_s: float = DEFAULT_DISCOVERY_INTERVAL_S   # min gap between backlog rglob walks


def _as_list(targets) -> list[str]:
    return [str(t) for t in targets] if isinstance(targets, (list, tuple, set)) else [str(targets)]


def pending_files(targets, store: FingerprintStore, fv: str, recursive: bool = True,
                  stable_s: float = DEFAULT_STABLE_S, now: float | None = None) -> list[str]:
    """On-disk videos that are NEW or CHANGED vs the store AND stable ('pending' under the shared
    freshness contract, analysis_state — gone/mid-copy/fresh/known-failed are all skipped; a mid-copy
    file is picked up once it settles). `now` injectable for tests."""
    now = time.time() if now is None else now
    return [p for p in collect_videos(targets, recursive=recursive)
            if analysis_state(store, p, fv, stable_s=stable_s, now=now) == "pending"]


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
    shares a name prefix from matching (e.g. 'Series' must not swallow 'Series2').

    ROOT-REACHABLE GUARD (§2): a watched root that is currently UNREACHABLE (an unmounted Storage
    Space / offline network share) is SKIPPED, so its indexed files are NOT mistaken for orphans — a
    momentary unmount must never read as a mass deletion and wipe the index. With every root
    unreachable, returns [] (assume nothing was deleted, just inaccessible)."""
    roots = [(_norm(t).rstrip(os.sep), t) for t in _as_list(targets)]
    roots = [(r, raw) for (r, raw) in roots if os.path.exists(raw)]   # drop offline roots (§2)
    if not roots:
        return []
    out: list[str] = []
    for p in store.all_paths():
        ap = _norm(p)
        if any(ap == r or ap.startswith(r + os.sep) for r, _ in roots) and not os.path.exists(p):
            out.append(p)
    return out


@dataclass
class CycleResult:
    indexed: int = 0
    removed: int = 0
    dup_clusters: list = field(default_factory=list)   # dup clusters that include a new file
    errors: list = field(default_factory=list)         # (path, error) for skipped/corrupt files

    def merge(self, other: "CycleResult") -> "CycleResult":
        """Fold the heavy ingest half into this (the removals half) and return self. `removed` is
        OWNED by reconcile_removals and never set by ingest_new, so it is intentionally not merged."""
        self.indexed += other.indexed
        self.errors += other.errors
        self.dup_clusters += other.dup_clusters
        return self


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


def _drain_deleted(deleted, store: FingerprintStore) -> int:
    """Process the queue of paths watchdog reported gone (deleted/moved). For each, ONE stat() to
    confirm it really left the disk (a delete-then-recreate / move-back is skipped), then forget it.
    O(events), NOT O(library) — the OS told us exactly what changed, so no full sweep. A path whose
    STORED form differs only in case/separators from the event (rare; a prior scan used a
    differently-cased root) won't exact-match `forget_file` here and is caught by the full sweep."""
    removed = 0
    while True:
        try:
            p = deleted.popleft()                  # deque.popleft is atomic -> no lock vs the observer
        except IndexError:
            break
        if os.path.exists(p):                      # recreated / moved back -> not a deletion
            continue
        if store.forget_file(p):                   # True only if a stored row actually matched
            removed += 1
    return removed


def _full_sweep(targets, store: FingerprintStore) -> int:
    """Robust but O(library) deletion sweep: forget every indexed file under a REACHABLE watched root
    that is gone from disk. The §2 root-reachable guard (in `orphan_paths`) keeps an offline drive
    from reading as a mass deletion. This is the STARTUP catch-up + periodic backstop for missed FS
    events + the no-watchdog fallback — NOT a per-cycle operation.

    COMPLEMENT (Mode B): `orphan_paths` skips a file whose whole watched ROOT was deleted (its root
    guard sees the root gone and bows out, so an unmounted drive isn't read as a deletion). The store's
    `prune_missing_files` catches exactly those — it guards by VOLUME + nested mount/junction instead of
    watched root, so a deleted folder on an ONLINE drive is cleaned while an offline drive/mount is left
    alone. Run it only when NO scan holds the lock: a scan is re-persisting matches (don't race its
    concurrent-deletion guard, §0) and owns the disk (don't add an O(library) stat sweep to the HDD
    contention, §1) — the next backstop sweep, or the UI's own prune, reconciles after the scan."""
    removed = 0
    for p in orphan_paths(targets, store):
        store.forget_file(p)
        removed += 1
    if not scan_in_progress():
        removed += store.prune_missing_files()         # Mode B: root gone but volume online (§0/§1 gated)
    return removed


def reconcile_removals(ctx: WatchContext, *, deleted=None, full: bool = False) -> CycleResult:
    """HIGH-PRIORITY, I/O-LIGHT half of a cycle: forget files that left the disk and rebuild the
    clusters they leave behind. NO video decode either way -> safe even while a scan holds the lock.
    Two sources, by cost:
      - EVENT DRAIN (cheap, O(deletions)): drain `deleted`, the queue watchdog fills with gone paths
        -> a trash/move is reconciled instantly without touching the rest of the library.
      - FULL SWEEP (O(library)): only when `full` (startup catch-up / periodic backstop for missed
        events) or when there is NO event source (`deleted is None` -> watchdog absent, polling only).
    Rebuilds clusters ONCE if anything was removed, REUSING the ranking of clusters that didn't change
    (only the cluster a deletion touched re-ranks — no whisper/audio on the rest of the library). Name
    grouping is NOT re-run: removing a file can't create a new '(N)' sibling pair."""
    prior = _snapshot_clusters(ctx.store)          # pre-removal ranking, captured before the forgets
    res = CycleResult()
    if deleted is not None:
        res.removed += _drain_deleted(deleted, ctx.store)
    if full or deleted is None:
        res.removed += _full_sweep(ctx.targets, ctx.store)
    if res.removed:                                # deletions can change/empty clusters -> rebuild
        _rebuild_clusters(ctx.store, ctx.th, reuse=prior)
    return res


def _safe_mtime(p: str) -> float:
    try:
        return os.stat(p).st_mtime
    except OSError:
        return 0.0


class _IngestScheduler:
    """Two-lane work source for ingest, DECOUPLING discovery (the O(library) rglob walk) from
    per-cycle processing:
      FAST lane — paths from watchdog created/modified events (`changed` deque), drained + deduped
                  each cycle, NEWEST-first: the file you just dropped is indexed within ~one chunk.
      SLOW lane — the historical backlog, discovered by an rglob walk at most every
                  `discovery_interval` seconds (NOT per chunk -> no disk thrash), consumed OLDEST-first
                  so progress to 100% is monotonic (no starvation under churn).
    Each cycle hands back at most `chunk` paths; every path is RE-VALIDATED at point of use (still on
    disk, stable past `stable_s`, not already fresh) so both lanes share ONE stability contract and a
    still-copying file is neither read early nor lost."""

    def __init__(self, changed=None, chunk: int = DEFAULT_INGEST_CHUNK,
                 discovery_interval: float = DEFAULT_DISCOVERY_INTERVAL_S):
        self.changed = changed                         # producer deque (watchdog) or None
        self.chunk = chunk
        self.discovery_interval = discovery_interval
        self._fast: dict[str, None] = {}               # consumer-side ordered dedup set
        self._backlog: list[str] = []                  # cached worklist, oldest-first
        self._cursor = 0                               # next backlog index to try
        self._last_discovery: float | None = None

    def _drain_events(self) -> None:
        """Fold the watchdog `changed` deque into the dedup set (a single copy fires many 'modified'
        events -> dedup is essential). Re-insert so the most-recent event wins ordering."""
        if self.changed is None:
            return
        while True:
            try:
                p = self.changed.popleft()             # atomic vs the observer thread
            except IndexError:
                return
            self._fast.pop(p, None)
            self._fast[p] = None

    def _discover(self, ctx: WatchContext, fv: str, tuning: WatchTuning, now: float) -> None:
        """Refresh the backlog by an rglob walk — but only when it's EXHAUSTED or the interval elapsed,
        so a chunked backlog never re-walks the whole tree per chunk. Sorted OLDEST-first."""
        due = (self._last_discovery is None
               or self._cursor >= len(self._backlog)
               or (now - self._last_discovery) >= self.discovery_interval)
        if not due:
            return
        pend = pending_files(ctx.targets, ctx.store, fv, recursive=tuning.recursive,
                             stable_s=tuning.stable_s, now=now)
        pend.sort(key=_safe_mtime)                     # oldest-first -> monotonic march to 100%
        self._backlog, self._cursor, self._last_discovery = pend, 0, now

    @staticmethod
    def _ready(p: str, store: FingerprintStore, fv: str, stable_s: float, now: float):
        """(ready, keep) — re-validate a candidate at point of use via the shared freshness contract
        (analysis_state, same rule as pending_files and the full scan): 'gone'/'done' -> drop;
        'copying' -> keep-but-not-ready (re-queue); 'pending' -> ready."""
        state = analysis_state(store, p, fv, stable_s=stable_s, now=now)
        return state == "pending", state in ("pending", "copying")

    def next_chunk(self, ctx: WatchContext, fv: str, tuning: WatchTuning, now: float) -> list[str]:
        """Up to `chunk` paths to analyze this cycle: FAST lane (newest-first) first, then OLDEST
        backlog to top up. Mid-copy fast-lane items stay queued (not popped) for the next cycle."""
        self._drain_events()
        self._discover(ctx, fv, tuning, now)
        out: list[str] = []
        for p in reversed(list(self._fast)):           # newest event first
            if len(out) >= self.chunk:
                return out
            ready, keep = self._ready(p, ctx.store, fv, tuning.stable_s, now)
            if ready or not keep:                      # consumed, or dropped (gone/fresh) -> leave the set
                self._fast.pop(p, None)
            if ready:
                out.append(p)
        picked = set(out)                              # don't pick a fast-lane file again from the backlog
        while self._cursor < len(self._backlog) and len(out) < self.chunk:
            p = self._backlog[self._cursor]; self._cursor += 1
            if p not in picked and self._ready(p, ctx.store, fv, tuning.stable_s, now)[0]:
                out.append(p)
        return out


def ingest_new(ctx: WatchContext, tuning: WatchTuning, *, scheduler: "_IngestScheduler",
               on_duplicate: Optional[Callable[[list], None]] = None,
               on_detect: Optional[Callable[[int], None]] = None,
               now: float | None = None) -> CycleResult:
    """HEAVY half of a cycle: index up to `tuning.ingest_chunk` NEW/CHANGED stable files this cycle
    (FAST lane first, then OLDEST backlog — see _IngestScheduler), match them incrementally against the
    index and rebuild the affected clusters. This is the disk-thrashing path the scan-priority lock
    protects -> the loop runs it ONLY when no scan holds the lock. CHUNKED so a just-dropped file is
    reached within ~one chunk even behind a big backlog. Never raises on a single bad file (§2).
    `on_detect(n)` fires with THIS chunk's size (not the whole backlog)."""
    now = time.time() if now is None else now
    store, embedder, th = ctx.store, ctx.embedder, ctx.th
    fv = feature_version(embedder, tuning.independent_scenes,
                         audio_fp_cap_s=th.audio_fp_cap_s, audio_fp_cap_above_s=th.audio_fp_cap_above_s)
    res = CycleResult()
    todo = scheduler.next_chunk(ctx, fv, tuning, now)
    if todo and on_detect:
        on_detect(len(todo))                       # surface detection before the slow per-file work
    if not todo:
        return res

    done: list[str] = []
    for p in todo:
        try:
            analyze_file(p, store, embedder, th, independent_scenes=tuning.independent_scenes)
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


def watch_once(ctx: WatchContext, tuning: Optional[WatchTuning] = None, *,
               on_duplicate: Optional[Callable[[list], None]] = None,
               on_detect: Optional[Callable[[int], None]] = None) -> CycleResult:
    """One FULL reconcile cycle = removals (cheap) + new-file ingest (heavy), combined. Retained for
    `watch --once` and tests; the watch LOOP runs the two halves at DIFFERENT priorities (removals
    always, even during a scan; ingest only when no scan holds the lock)."""
    tuning = tuning or WatchTuning()
    res = reconcile_removals(ctx)
    once = _IngestScheduler(changed=None, chunk=10 ** 9)   # one-shot: drain the whole backlog now
    ing = ingest_new(ctx, tuning, scheduler=once, on_duplicate=on_duplicate, on_detect=on_detect)
    return res.merge(ing)


def _route_event(event, deleted, changed) -> None:
    """Push a watchdog event's exact path onto the right fast-lane deque (consumer dedups + validates).
    A dir event is skipped (a recursive delete may report only the dir -> the backstop sweep covers it).
    Paths are CANONICALIZED: watchdog builds src_path as (watched root, as given with '/') + (relative
    with '\\') -> a '/'+'\\' MIX that, stored as-is, becomes a PHANTOM DUPLICATE of the scan's pathlib
    form. canonical_path folds it to the same key collect_videos produces, so one file = one record."""
    if event.is_directory:
        return
    et = event.event_type
    if deleted is not None and et in ("deleted", "moved"):
        deleted.append(canonical_path(event.src_path))        # left this path -> forget it
    if changed is not None:
        if et in ("created", "modified"):
            changed.append(canonical_path(event.src_path))    # new/changed -> fast-lane index
        elif et == "moved":
            changed.append(canonical_path(getattr(event, "dest_path", "") or event.src_path))  # moved-into


def start_fs_events(targets, wake, deleted=None, changed=None) -> Optional[Callable[[], None]]:
    """Subscribe to filesystem events under `targets` (native: ReadDirectoryChangesW on Windows,
    inotify on Linux, FSEvents on macOS — via `watchdog`) and set `wake` on ANY change so the loop
    reconciles immediately. Two exact-path fast-lanes (each a deque the loop drains): `deleted` gets
    delete/move-away paths (forget in O(1)); `changed` gets create/modify/move-into paths so the loop
    INDEXES the just-touched file before chewing the historical backlog. Returns a stop() callable, or
    None if `watchdog` isn't installed (the loop then relies on the periodic full sweep + discovery
    walk). Events are a TRIGGER, not the source of truth — the producer only does an atomic
    `deque.append` (never blocks the OS callback / drops events); dedup + stability are the consumer's
    job. A missed/overflowed event only delays (the backstop catches it), never corrupts."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except Exception:                              # noqa: BLE001 watchdog optional -> polling only
        return None

    class _Wake(FileSystemEventHandler):
        def on_any_event(self, event):
            _route_event(event, deleted, changed)
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


def _run_cycle(ctx: WatchContext, tuning: WatchTuning, *, busy: bool, deleted, full: bool,
               scheduler: "_IngestScheduler", on_duplicate, on_detect) -> CycleResult:
    """ONE reconcile cycle: removals ALWAYS (cheap event drain + optional `full` sweep); the HEAVY
    ingest only when no scan holds the lock (`busy` False). A failing cycle is caught here so the
    loop never dies (§2)."""
    try:
        res = reconcile_removals(ctx, deleted=deleted, full=full)
        if not busy:
            res.merge(ingest_new(ctx, tuning, scheduler=scheduler,
                                 on_duplicate=on_duplicate, on_detect=on_detect))
        if not res.indexed:                            # nothing embedded -> release cached GPU VRAM so
            ctx.embedder.free_cache()                  # an idle 24/7 watcher stops squatting the GPU
        return res
    except Exception as e:                             # noqa: BLE001 a cycle error must not stop the watcher
        return CycleResult(errors=[("<cycle>", str(e))])


def _next_cadence(res: CycleResult, cur: float, interval: float, max_interval: float,
                  backoff: float) -> float:
    """IDLE BACKOFF: a cycle that found nothing grows the wait (×`backoff`, capped at `max_interval`)
    so a static library is barely touched; any activity (indexed/removed/errors) resets to `interval`."""
    if res.indexed or res.removed or res.errors:
        return interval
    return min(cur * backoff, max_interval)


class _SweepSchedule:
    """Decides when the O(library) backstop full sweep runs: once at startup (first idle cycle), then
    every `every` idle cycles thereafter. NEVER while a scan is busy — the sweep must not compete with
    it for the disk (§1). Holds the across-cycle counter so the loop body stays flat."""
    def __init__(self, every: int = FULL_SWEEP_EVERY) -> None:
        self.every = every
        self.pending = True                        # startup catch-up
        self.idle = 0

    def due(self, busy: bool) -> bool:
        if busy:
            return False
        self.idle += 1
        if self.pending or self.idle >= self.every:
            self.pending = False
            self.idle = 0
            return True
        return False


def _wait_next(wake, wait_s: float, sleep: Callable[[float], None]) -> bool:
    """Wait before the next cycle. With a `wake` event (FS-event source) a real change reconciles
    early; returns True if the cadence should RESET (an event arrived). Without `wake` it just sleeps
    and returns False."""
    if wake is not None:
        woke = wake.wait(wait_s)
        wake.clear()
        return woke
    sleep(wait_s)
    return False


def watch_loop(ctx: WatchContext, *, tuning: Optional[WatchTuning] = None,
               on_duplicate: Optional[Callable[[list], None]] = None,
               on_cycle: Optional[Callable[[CycleResult], None]] = None,
               on_detect: Optional[Callable[[int], None]] = None,
               sleep: Callable[[float], None] = time.sleep,
               wake=None, deleted=None, changed=None, stop: Optional[Callable[[], bool]] = None) -> None:
    """Reconcile until `stop()` returns True (or forever). EVENT-DRIVEN DELETIONS: when watchdog fills
    the `deleted` queue, each trash/move is forgotten in O(1) — the expensive O(library) `orphan_paths`
    sweep is demoted to a startup catch-up + a periodic backstop (`FULL_SWEEP_EVERY` idle cycles) for
    missed events, and never runs while a scan holds the lock. Without watchdog (`deleted is None`)
    the sweep is the only option and runs every cycle. SPLIT PRIORITY: removals run every cycle; only
    the HEAVY `ingest_new` (decode+embed) yields to a scan. IDLE BACKOFF: an idle cycle grows the wait
    (×`backoff`, capped at `max_interval`); activity resets it to `interval`. `wake` triggers an
    IMMEDIATE reconcile (instant deletion latency). `sleep`/`stop` injectable for tests. A failing
    cycle is reported and does not kill the loop."""
    tuning = tuning or WatchTuning()
    cur = tuning.interval
    deferred = False
    sweep = _SweepSchedule()
    scheduler = _IngestScheduler(changed=changed, chunk=tuning.ingest_chunk,
                                 discovery_interval=tuning.discovery_interval_s)
    while not (stop and stop()):                   # checked once per cycle (at the top)
        busy = scan_in_progress()                  # a user scan holds the priority lock
        if busy != deferred:                       # announce the transition once (not every cycle)
            print("[watch] scan running — deletions still reconciled, heavy indexing deferred (priority)"
                  if busy else "[watch] resumed — indexing new files", flush=True)
            deferred = busy
        res = _run_cycle(ctx, tuning, busy=busy, deleted=deleted, full=sweep.due(busy),
                         scheduler=scheduler, on_duplicate=on_duplicate, on_detect=on_detect)
        if on_cycle:
            on_cycle(res)
        cur = _next_cadence(res, cur, tuning.interval, tuning.max_interval, tuning.backoff)
        # No watchdog + scan running: polling is the only way to see a deletion, so cap the wait.
        # With watchdog, the `wake` event drives deletion latency -> no forced short poll needed.
        wait_s = min(cur, DEFERRED_POLL_S) if (busy and wake is None) else cur
        if _wait_next(wake, wait_s, sleep):        # an FS event arrived -> back to fast cadence
            cur = tuning.interval
