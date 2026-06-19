# ENGINEERING_LOG — solved problems & non-obvious gotchas (Dupless Video)

**Purpose:** durable memory of things already figured out, so they are never re-investigated.
Grep this file (by symptom) before any non-trivial debugging or "why does X happen" exploration;
append an entry when you solve something that took real digging. A fix without a log entry is half-finished.

**Entry format:** `### <searchable symptom>` then **Symptom / Root cause / Resolution / Files-refs / Date-scope**.
Newest on top. Append-only in spirit. Deep rationale for the design invariants lives in [CLAUDE.md](../CLAUDE.md).

---

## Entries

### Watcher "Fast-Lane": a just-dropped file waits behind the whole backlog
- **Symptom:** with a large backlog (~8,760 files still needing full analysis), a newly added/changed
  file isn't indexed promptly — the watcher processes pending files in arbitrary `rglob` order and the
  fresh file queues behind thousands.
- **Root cause:** `ingest_new` took `pending_files` (a full `rglob` walk, unsorted, uncapped) and
  analyzed EVERY pending file in ONE cycle (~36-56h for 8,760 @ ~15-23s). No prioritization, and one
  giant cycle never returns to react to new events.
- **Resolution (two-lane scheduler, `_IngestScheduler`):**
  - FAST lane: the watchdog handler (`_route_event`) pushes `created`/`modified`/`moved`-dest paths to
    a `changed` deque (mirrors the `deleted` lane); the loop drains+dedups it and indexes those FIRST,
    newest-first. (A copy fires many `modified` events -> dedup on the consumer side; producer only does
    an atomic `deque.append`.)
  - SLOW lane: the historical backlog, consumed OLDEST-first (FIFO -> monotonic to 100%, no starvation
    under churn — the fast lane already covers fresh drops).
  - CHUNKED: at most `ingest_chunk` (default 4) files/cycle, so the loop returns to re-check the fast
    lane / scan-priority within ~one chunk (~60-90s). Without the cap the fast lane is meaningless.
  - DISCOVERY DECOUPLED: the `rglob` walk runs at most every `discovery_interval_s` (default 300s) or
    when the cached backlog empties — NOT per chunk. (Per-chunk re-walking would thrash the HDD the scan
    needs — the load-bearing trap the Tech Lead review caught.)
  - Index built ONCE per cycle (NOT incremental `index.add()` — that desyncs `CoarseIndex._gvecs`/`_pos`
    and silently breaks `gate_by_global`; amortize via chunk size instead).
  - ONE stability contract (`_ready`): every path is re-`stat`'d at point of use (still on disk, settled
    past `stable_s`, not already fresh); mid-copy fast-lane items are re-queued (never dropped/read early).
- **Design debated** with a Tech Lead agent; decisions: backlog OLDEST-first (vs the initial newest-first
  spec) for no-starvation; chunk N=4 (~60-90s latency); discovery decoupled. Knobs on `WatchTuning`:
  `ingest_chunk`, `discovery_interval_s`.
- **Files / refs:** `watch.py` (`_IngestScheduler`, `_route_event`, `ingest_new`, `watch_loop`,
  `start_fs_events`, `WatchTuning`); `cli.py` watch (`changed` deque); tests in `test_watch.py`.
- **Scope:** decided 2026-06-19.

### Background watcher holds ~GBs of GPU VRAM while idle ("memory not freed")
- **Symptom:** the GPU sits ~85% full with util ~1% (idle) while the long-lived watcher is up; suspicion
  that something "no se libera". On Windows/WDDM `nvidia-smi` does NOT report per-process VRAM (all
  `N/A`), so it can't be attributed directly — but a 24/7 watcher process is a real contributor.
- **Root cause:** PyTorch's caching allocator keeps freed blocks resident (does not return them to the
  driver), so after an ingest cycle the watcher holds its PEAK VRAM footprint forever between cycles —
  even when the library is quiet and the user wants the GPU for other apps.
- **Resolution:** `Embedder.free_cache()` (calls `torch.cuda.empty_cache()`, model stays resident, no-op
  on CPU). The watch loop (`_run_cycle`) calls it on any cycle that embedded NOTHING (`res.indexed == 0`)
  — i.e. when the library goes idle — returning cached VRAM to the system. Active ingest bursts
  (`indexed > 0`) KEEP the cache so throughput isn't hurt by re-allocating each file. Note: this only
  frees the WATCHER's own cache; browsers/Electron (Chrome/Edge WebView/VS Code) hold the rest.
- **Files / refs:** `features/embeddings.py` `free_cache`; `watch.py` `_run_cycle`; tests in `test_watch.py`.
- **Scope:** decided 2026-06-19.

### Canceling/stopping a scan is reported as "❌ Failed (exit 1)"
- **Symptom:** pressing Cancel/Stop on a running analysis pops the failure dialog ("The analysis failed,
  exit code 1"), even though the scan was progressing fine and the log has NO traceback (just tqdm
  progress up to where it was killed).
- **Root cause:** `ScanPanel._done` decided cancel-vs-failure by reading the STATUS LABEL TEXT
  (`if "Canceled" in self.status.text()`). `_stop` set that text, but the 1 Hz heartbeat (`_tick`→
  `_render`) or a final BUFFERED stdout line (`_read`→`_parse`→`_render`) arriving after the kill
  overwrites it with the last progress line. By the time `finished` fires, the text no longer says
  "Canceled", so `_done` falls to the `else` branch and treats the killed process's nonzero exit as a
  crash. Text is presentation, not state — using it as a state flag is the bug.
- **Resolution:** explicit `self._canceled` boolean — set in `_stop`, reset when a run launches, and
  checked in `_done` (which also re-asserts the "Canceled." label in case a late render clobbered it).
  Real failures (not canceled) still surface the dialog. Tests in `test_ui.py` cover both: a cancel
  whose label was overwritten is NOT reported as failure, and a genuine crash still is.
- **Files / refs:** `ui/scan_panel.py` (`_stop`, `_done`, `__init__`, `_toggle`).
- **Scope:** decided 2026-06-17.

### Scan crashes mid-Pass-2 with `AssertionError` (faiss `assert d == self.d`)
- **Symptom:** a full scan dies with exit 1, crash dialog "Cause: AssertionError"; the `dupdetect_scan.log`
  traceback ends in `matcher.candidate_paths` → `retrieval.query_global` → faiss
  `class_wrappers.py ... assert d == self.d`. One bad file takes the WHOLE Pass-2 batch down (§2 break).
- **Root cause:** `candidate_paths` called `index.query_global(rec.global_vec, …)` with NO guard. A
  record with an EMPTY global embedding (dim 0 — e.g. a file Pass-1 could not decode any frames from)
  reshaped to `(1, 0)`, and faiss asserts the query dim equals the index dim (768) → hard crash. The
  index itself is fine: `all_global_vecs` already skips embedding-less records, and config dim == model
  dim == 768 (so it is NOT a mixed-dimension / wrong-fallback-dim bug — verified on the real DB: every
  `global_vec` blob is exactly 768·4 bytes). It is purely the QUERY side that was unguarded.
- **Resolution:** (1) `candidate_paths` returns `set()` immediately when `rec.global_vec` is None/empty
  — an un-embeddable file simply produces no candidates (§2 skip, no matches). (2) Defense-in-depth at
  the faiss boundary: `CoarseIndex.query_global`/`query_windows` return empty when the query dim != the
  index dim, turning any future dim mismatch into skip-and-report instead of a batch-killing assert.
  ROOT FIX (the deeper issue): `build_record` now RAISES `"no decodable video frames — cannot compute
  an embedding"` when `emb` is empty, so Pass-1's caller catches it and `save_problem` surfaces the file
  in the Problems tab (classified 'corrupt') instead of persisting a useless record with an empty
  global_vec. `build_record` is the single chokepoint for both `analyze_file` and `_gpu_finish`, so the
  serial, parallel and pipelined Pass-1 paths are all covered.
- **Files / refs:** `pipeline/analyze.py` `build_record` (empty-emb guard); `match/matcher.py`
  `candidate_paths`; `match/retrieval.py` `query_global`/`query_windows` (faiss boundary guards);
  tests in `test_pipeline.py` + `test_retrieval.py`.
- **Scope:** decided 2026-06-17.

### Trashed files stay in the list while a scan runs / "the watcher-update isn't working"
- **Symptom:** files sent to the Recycle Bin (in-app or from Explorer) keep showing in the duplicates
  list, especially while "Analyzing N/M — re-encode/upgrade detection still completing" is running.
- **Root cause:** TWO distinct problems, both rooted in over-broad coupling. (A) The full scan works off
  a file-set SNAPSHOT taken at scan start (`collect_videos`) and a coarse index built ONCE in memory; a
  file trashed mid-scan was still re-matched and `save_match`'d, RESURRECTING the row the UI just forgot
  (`actions.delete_files` → `forget_file`). (B) The watcher's cheap, decode-free self-heal
  (`orphan_paths` → `forget_file`) was bundled with the heavy decode/embed ingest under ONE yield-gate,
  so `scan_in_progress()` paused BOTH — yet the scan-priority lock only ever existed to stop HDD
  thrashing / WAL write-starvation from the HEAVY path, not deletion bookkeeping.
- **Resolution (3 layers):** (1) §0 stat-guard `_both_on_disk` in `_pass2_sequential`/`_pass2_parallel`
  — never (re)persist a match whose endpoints aren't both on disk, so a concurrent delete WINS the race
  (memoized: ≤1 stat/path). (2) Split `watch_once` into `reconcile_removals` (cheap, write-only, runs
  EVERY cycle even during a scan) and `ingest_new` (heavy, still yields to the scan); `watch_loop` runs
  removals always + ingest only when no lock is held. EVENT-DRIVEN deletions: `start_fs_events` pushes
  the exact path of each watchdog `deleted`/`moved` event onto a thread-safe `deque`; `_drain_deleted`
  forgets just that path in O(1) — so a trash/move is reconciled instantly without re-deriving it. The
  O(library) `orphan_paths` sweep (`_full_sweep`) is demoted to a STARTUP catch-up + a periodic backstop
  (`_SweepSchedule`, `FULL_SWEEP_EVERY` idle cycles) for missed events, never runs while a scan holds
  the lock, and is the only mechanism when watchdog is absent (`deleted is None`). This killed a
  ~17k-row read + ~17k stat() sweep every 10s for the whole duration of a multi-hour scan (§1: don't
  thrash the disk the scan needs). (3) Optimistic in-place view update: in-app delete drops the rows via
  `model.remove_paths` (collapses singleton clusters) instead of a full `refresh()` that could race the
  scan's mid-flight cluster rebuild. §2 offline-drive guard added to `orphan_paths`: an UNREACHABLE
  watched root is skipped (a momentary unmount must never read as a mass deletion and wipe the index).
- **Note:** there is NO on-disk watcher log — the watcher only prints to stdout, captured in-memory by
  `WatchPanel` (≤800 lines). "Today's watcher log" does not exist as a file.
- **Gotcha + fix (measured 2026-06-17):** `reconcile_removals` used to re-run `_rebuild_clusters`
  (re-rank of ALL clusters) on ANY removal. `rank_cluster` lazily computes whisper (~4s/file) +
  whole-file audio coverage for any member missing its cache, so the FIRST removal on a library
  scanned at fast/standard depth paid the whole deferred bill at once (observed: forgetting 4 orphans
  → ~8 min / ~500 CPU-s re-ranking ~200 clusters). FIX: `_rebuild_clusters(reuse=<snapshot>)` keeps the
  union-find STRUCTURE rebuild full (cheap; a removed hub still splits correctly) but REUSES the
  persisted KEEP/rank_reason of any cluster whose membership is unchanged — only the cluster a deletion
  touched re-ranks. The snapshot (`_snapshot_clusters`) is captured BEFORE the forgets so an affected
  cluster's larger pre-removal signature no longer matches its rebuilt group. `reconcile_removals` also
  stopped calling `_apply_name_grouping` (a removal can't create a new '(N)' sibling pair). Measured on
  the real 202-cluster DB: a 1-file removal went from 201 → ≤1 `rank_cluster` calls (6.10s → 1.56s even
  warm; minutes → seconds cold). `reuse` is NOT passed by full_scan / ingest, where a member's data may
  have changed (re-encode) and ranking must be fresh.
- **Files / refs:** `pipeline/fullscan.py` (`_on_disk`/`_both_on_disk`, `_pass2_*`); `watch.py`
  (`reconcile_removals`, `_drain_deleted`, `_full_sweep`, `_SweepSchedule`, `ingest_new`, `watch_loop`,
  `start_fs_events`, `orphan_paths`); `cli.py` watch (`deleted` deque); `store.py` `forget_file` (now
  returns bool); `ui/model.py` `remove_paths`; `ui/main.py` `_delete_selected`; tests in
  `test_watch.py`/`test_fullscan.py`/`test_ui.py`.
- **Scope:** decided 2026-06-17.

### Make notifications silent / "the whole app should make no sound"
- **Symptom:** background notifications (new duplicates, scan finished) should be silent, or the whole
  app should never make a sound.
- **Root cause:** `QSystemTrayIcon.showMessage` (the native Windows toast) plays the OS notification
  sound and Qt exposes NO per-message way to mute it. (QMessageBox sounds are tied to its icon — already
  handled by the iconless `_mbox`; status-bar `_toast` is silent.)
- **Resolution:** the app is intentionally SILENT — there are NO `QSystemTrayIcon.showMessage` calls
  left. Background events use the silent in-app status-bar `_toast`; the "Analysis finished", "Still
  running" (close-to-tray hint) and "Running in the background" (start-hidden) OS toasts were removed.
  A truly *silent Windows toast* (keep the popup, drop the sound) is only possible via a native WinRT
  toast (`<audio silent="true"/>`, dep: windows-toasts) — deliberately NOT adopted. Don't reintroduce
  `showMessage`. App identity for toasts is the AUMID `DupDetector.VideoDedup.1` (`run()`).
- **Files / refs:** `ui/main.py` (`_on_watch_dups`, `_on_scan_finished`, `closeEvent`, `run`); `ui/actions.py` `_mbox`/`_toast`.
- **Scope:** decided 2026-06-15.

### Byte-identical files show as "Review only" instead of a certain duplicate (T0)
- **Symptom:** files that are byte-identical (or that a fast/exact-only scan grouped together) don't
  get a CERTAIN verdict; the cluster appears but its verdict is empty ("Review only").
- **Root cause:** `exact_scan` (fast mode / `--depth fast`) groups identical files by
  `content_hash`+`size` into the `clusters` table but historically did NOT write `matches`. The T0
  "CERTAIN" verdict lives inside `decide_tree` (run only during a full Pass-2), so byte-identical
  pairs never got a `matches` row → `clusters`↔`matches` drift → UI falls back to "Review only".
- **Resolution:** `exact_scan` now also emits T0 `CERTAIN` matches (shared `T0_REASON`, star topology
  keep↔copies, skips a pair that already carries a content verdict). Re-run the fast scan to backfill
  an existing DB — cheap, it reuses stored hashes. Byte-identity = exact equality of the SAMPLED
  `content_hash` (xxh3_64 of head|mid|tail) + `size`; it is not a tunable threshold.
- **Files / refs:** `pipeline/fullscan.py` `exact_scan`; `match/tree.py` `T0_REASON`; `ui/data.py` `drift_report`.
- **Scope:** fixed 2026-06-15.

### Recalibrating thresholds doesn't change already-scanned results / applying θ without a re-scan
- **Symptom:** changing θv/θa (recalibrate) only affects future scans; existing verdicts/clusters stay.
- **Root cause:** a verdict is the pure `decide_tree(rec_a, rec_b, signals, θ)`; the per-pair raw
  signals are already stored in `matches` (`audio_json/video_json/scenes_json`), but nothing re-decided
  them against the new θ.
- **Resolution:** `calibrate.apply_thresholds_to_store(store, th)` re-runs `decide_tree` over the
  stored signals with REAL records (so T0 byte-identity + T4 cut-density stay correct — unlike the
  `_mk` calibration stub), rewrites only the rows whose verdict moved, and rebuilds clusters — zero
  decode/embed/GPU. Wired into the UI recalibrate action (`ui/actions.apply_thresholds(..., store=...)`).
  Rows without signals (T0, NAME_COPY) are θ-independent → skipped. Recall ceiling: it re-judges only
  pairs Pass-2 already evaluated; it cannot surface a duplicate that retrieval never generated.
- **Files / refs:** `pipeline/calibrate.py` `apply_thresholds_to_store`; `ui/actions.py` `apply_thresholds`.
- **Scope:** added 2026-06-15.

### "How do I make the scan faster?" — tempted to add GPU FP8/FP4 or async CUDA streams
- **Symptom:** recurring urge to micro-optimize the GPU path for throughput.
- **Root cause:** the pipeline is **I/O-bound on disk**, not compute-bound.
- **Resolution:** GPU tricks (FP8/FP4, async CUDA streams) were **measured at ~zero net gain — do NOT
  re-benchmark them.** Real wins are I/O: adaptive sampling, decode↔embed overlap, avoiding disk
  contention. Any new perf change ships **off by default** until a benchmark shows it pays here.
- **Files / refs:** [CLAUDE.md](../CLAUDE.md) §1.
- **Scope:** seeded 2026-06-15 · recurring · standing answer.

### Doubt that CPU vs GPU (or fp16 vs fp32) could change which files get flagged
- **Symptom:** worry that a verdict depends on hardware or precision.
- **Resolution:** already validated — **fp32 vs fp16 = 0 verdict flips**; models + thresholds are global
  and fixed. Never lower precision for speed (breaks the zero-false-positive guarantee in T1/T2).
  Don't re-investigate hardware parity.
- **Files / refs:** [CLAUDE.md](../CLAUDE.md) §0.
- **Scope:** seeded 2026-06-15 · standing answer.

### A worker/concurrency count that's fast on one machine thrashes on another
- **Symptom:** the same concurrency setting is fast on NVMe but crawls / thrashes on a spinning HDD.
- **Root cause:** the optimum is **storage-aware** — an HDD thrashes under the concurrency an NVMe loves.
- **Resolution:** auto-tune by storage type, or expose the knob with one-line guidance. Don't hardcode a
  single "best" worker count, and don't treat an HDD regression as a bug in the concurrency code.
- **Files / refs:** [CLAUDE.md](../CLAUDE.md) §1, §3.
- **Scope:** seeded 2026-06-15 · standing answer.

### UI / full-scan tests hang or error in a headless run
- **Symptom:** tests touching the Qt UI (e.g. `tests/test_ui.py`, `tests/test_fullscan.py`) hang or fail
  with a display/platform error.
- **Resolution:** run them with the offscreen Qt platform:
  `QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m pytest tests/test_ui.py tests/test_fullscan.py -q`
- **Scope:** seeded 2026-06-15 · recurring.
