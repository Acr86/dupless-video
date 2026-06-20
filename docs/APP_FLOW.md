# App Flow — Dupless Video

> **Status:** v1 (reverse-engineered from `cli.py`, `pipeline/fullscan.py`, `pipeline/analyze.py`,
> `match/*`, `watch.py`, `runtime.py`, `ui/main.py`). Describes the **current technical flow** of the
> built app: process topology, the two scan passes, the watcher loop, and how each step touches the
> backend.
> **Companion docs:** [TECHNICAL_REQUIREMENTS.md](TECHNICAL_REQUIREMENTS.md), [BACKEND_SCHEMA.md](BACKEND_SCHEMA.md), [HPC_PIPELINE.md](HPC_PIPELINE.md).

## 1. Process topology

Three entry points share one engine and one DB:

```
        ┌─────────────────────────── dupdetect CLI (typer) ───────────────────────────┐
        │  scan │ check │ ui │ calibrate │ watch │ repair-indexes │ autotune          │
        └───────┬───────────────┬───────────────────────┬─────────────────────────────┘
                │               │                        │
         full_scan/         analyze_single          MainWindow (PySide6)
         exact_scan             (one file)           reads the DB, spawns CLI
                │               │                    subprocesses for scan/watch
                ▼               ▼                        │ QProcess
        ┌───────────────────────────────────────────────▼───────────────┐
        │  FingerprintStore (SQLite, WAL)  +  embeddings/*.npy (fp16)    │
        └───────────────────────────────────────────────────────────────┘
                ▲
         watch_loop (background) — keeps the DB current; yields to a scan via the priority lock
```

- The **UI never computes features itself**: it launches `dupdetect scan`/`watch` as **subprocesses**
  (`runtime.cli_subprocess` resolves dev `python -m dupdetect.cli …` vs frozen `<exe> scan …`) and
  parses their stdout progress lines. This keeps Qt responsive and the heavy work crash-isolated.
- **Within a scan**, work is split by resource (M3): CPU features in a `ProcessPool`; GPU decode+embed
  on the main process (single CUDA context).

## 2. Bootstrap (every CLI command)

1. `_bootstrap(db, config)` → `load_thresholds()` (per-user override if recalibrated, else bundled
   default), open `FingerprintStore` (creates data dir, applies schema + idempotent migrations, sets
   WAL), construct `Embedder` (DINOv2, lazy model load; offline via `configure_offline_model`).
2. External binaries (`ffmpeg`/`ffprobe`/`fpcalc`) resolved offline-first (`resolve_binary`).
3. The Windows console is reconfigured to UTF-8 so non-cp1252 filenames don't crash output.

## 3. Full scan flow (`dupdetect scan`, depth = standard/deep)

`cli.scan` → `scan_priority_lock()` held for the whole scan → `full_scan(...)`.

### 3.1 Discovery & filters
- `collect_videos(targets, recursive)` expands files/folders in place (no copy), filtering by
  `VIDEO_EXTS`. Warnings printed for non-existent paths / unrecognized extensions.
- Optional `--max-height`: a fast parallel ffprobe (`filter_by_height`) splits kept/excluded;
  unmeasurable files are kept (handled by Pass-1).
- `feature_version(...)` computed once (drives incrementality).

### 3.2 Pass-1 — index (features → store) · `_pass1`
For each file **not already fresh** (`has_fresh`: mtime±tol + size + feature_version + `.npy` exists):

```
extract_cpu_features (ProcessPool, no CUDA)        extract_gpu_features (main, CUDA)
  ├─ ffprobe (duration/res/codec/bitrate/tracks)     ├─ decode_frames (adaptive: demux ≤6GB / seek >6GB)
  ├─ content_hash (xxhash head|mid|tail)             │     → frames[N], frame_times[N], ColorStats
  ├─ scene_cuts (only if --independent-scenes)       └─ embedder.encode → embeddings[N,D] fp16 (L2)
  ├─ cam_score_partial
  ├─ audio_fp = EMPTY  (deferred to Pass-2)        build_record → global_vec, window_vecs,
  └─ audio_coverage = NULL (deferred)                scene_cuts(from embeddings+frame_times)
                                                   store.save(rec, feature_version)
```

- **Two execution modes:** serial decode on main (HDD default), or `--decode-workers>1` →
  `_drain_pipelined` overlaps disk decode (thread pool, bounded prefetch) with GPU embed (SSD/NVMe only;
  measured 2.19x on the decode+embed portion, thrashes on HDD).
- **Backend writes:** each `store.save` upserts the `files` row, writes the `.npy`, clears any
  `problems` row, and **deletes stale `matches`** for the file. Incremental skips re-print
  `fresh=` counts.
- **Resilience (§2):** any decode/probe error → `save_problem(path, error)` (categorized
  corrupt/reindex) and the file is skipped; the batch never crashes. A file with no decodable frames is
  surfaced as a problem, not saved as a degenerate empty-vector record.
- **Deep depth only:** `_ensure_coverage_all` fills whole-file `audio_coverage` for every file
  (incremental — only NULLs).
- `--no-match`: stop here (Pass-1 only), return empty queues.

### 3.3 Coarse index build
`store.all_global_vecs()` + `all_window_vecs()` (both **skip LITE/embedding-less records**) →
`CoarseIndex.build(...)` (FAISS, inner-product over L2-normalized = cosine). This in-memory index is a
**snapshot at scan start** (relevant to the concurrent-deletion guard, §3.5).

### 3.4 Pass-2 — match (pairs → matches) · `_pass2`
Dispatch: **parallel** (`match_pairs_parallel`) when ≥8 files and >1 core, else **sequential**
(`_pass2_sequential` with an LRU `EmbeddingCache`). Both are deterministic per pair (NN-2).

Per source file:
```
candidate_paths(rec):                       # uncapped recall, then pruned
   topk_global ∪ window_vecs ∪ (duration_blocking ±3% GATED by global cosine ≥ 0.7)
for each candidate (canonical pair, once):
   fa,fb = _ensure_audio_fp(...)            # ON-DEMAND fpcalc here (candidate pairs only), persisted
   a = align_audio(fa, fb)                  # audio-fp alignment
   v = _align_video_pair(rec, other)        # banded video DP on a common temporal grid (~89% of Pass-2)
   s = align_scenes(rec.scene_cuts, ...)    # scene-cut DTW
   res = decide_tree(rec, other, a, v, s)   # T0…T5 — first tier wins
   store.save_match(...)                     # unless DIFFERENT (T5, not persisted)
   route → review_queue (PROBABLE) | editions (DIFFERENT_EDITION)
```
- **Parallel path:** main process ensures+persists each involved file's audio-fp once, then read-only
  worker processes align pairs (numpy only — workers never import torch). Audio-fp never runs
  concurrently.
- **Concurrent-deletion guard (§0):** before persisting, `_both_on_disk(a,b)` checks the live disk; a
  file the user trashed mid-scan never resurrects its `matches` row.

### 3.5 Grouping & clustering
1. `_apply_name_grouping` — `movie (N).ext` siblings in the same folder → NAME_COPY **with a content
   veto** (`name_pair_content_differs`: skip if content says DIFFERENT or can't be verified).
2. `_rebuild_clusters` — union-find over all `DUPLICATE_VERDICTS` in `matches` → groups; each gets a
   **stable content-derived `cluster_id`**; `rank_cluster` picks the KEEP (deferred whisper +
   on-demand coverage run **here**, only for members); atomic `replace_clusters`.
3. Report JSON (clusters / review_queue / editions / skipped / excluded_by_height) written next to the
   DB; `problems` summarized.

## 4. Fast scan flow (`dupdetect scan --depth fast` / `--exact-only`) · `exact_scan`
Byte-identical detection only, ~0.1s/file vs ~12s:
```
_partition_by_hash → reuse stored hash for unchanged files (incremental); hash the rest (ProcessPool)
_hash_exact        → save_meta (LITE record: metadata+hash, no embeddings/.npy)
_build_exact_clusters → group by (hash,size); stamp each pair T0 CERTAIN in matches; atomic clusters
```
LITE records show in the UI; a later full scan re-indexes them with embeddings.

## 5. Single-file check (`dupdetect check`) · `analyze_single`
Same engine as the watcher: build the coarse index from the DB, `analyze_file` the new file, `match`
it against the index, print verdicts. Used to test one file without a full scan.

## 6. Watcher flow (`dupdetect watch`) · `watch_loop`

Goal: keep the DB current so the user opens the app only to review. **Polling + reconcile** with
optional native FS events (watchdog).

### 6.1 Startup
- Optional Phase 1: instant `exact_scan` sweep (byte-identical dups visible in minutes).
- Subscribe to FS events if `watchdog` present (`start_fs_events`): two fast-lane deques —
  `deleted` (delete/move-away) and `changed` (create/modify/move-into). Event paths are
  **canonicalized** (`canonical_path`) so a watchdog `/`+`\` mix never becomes a phantom duplicate.
  Without watchdog → backoff polling only.

### 6.2 Each cycle (`_run_cycle`)
```
busy = scan_in_progress()      # is a user scan holding the priority lock?
reconcile_removals(...)        # ALWAYS (cheap, no decode):
   ├─ _drain_deleted (O(deletions)): forget files watchdog reported gone
   └─ _full_sweep (O(library)): startup catch-up + periodic backstop (FULL_SWEEP_EVERY idle cycles)
                                — root-reachable guard: an offline drive is NOT read as a mass deletion
if not busy:                   # HEAVY half yields to a scan (priority lock):
   ingest_new(...):
      ├─ _IngestScheduler.next_chunk: FAST lane (newest events) then SLOW lane (oldest backlog),
      │     ≤ ingest_chunk files/cycle, each re-validated (on disk, stable past stable_s, not fresh)
      ├─ analyze_file each (skip-and-report on error)
      ├─ rebuild coarse index (now includes the new files → intra-batch dups caught)
      ├─ _pass2 (incremental) + _apply_name_grouping + _rebuild_clusters
      └─ on_duplicate(affected dup clusters)  → notification
if nothing indexed: embedder.free_cache()    # idle watcher releases GPU VRAM
```
- **Split priority:** deletions are reconciled even during a scan; only decode+embed ingest defers.
- **Idle backoff:** an idle cycle grows the wait (×backoff, capped at `max_interval` 30 min); any
  activity resets to `interval`. An FS-event `wake` triggers an immediate reconcile (instant deletion
  latency).
- **Discovery is decoupled** from processing: the O(library) rglob runs at most every
  `discovery_interval_s` (default 300s), not per chunk, so a chunked backlog never re-walks the tree.
- **Cluster ranking reuse:** a removal-only rebuild reuses the ranking of clusters whose membership
  didn't change (`_snapshot_clusters` + `reuse`), so deleting a file only re-ranks the clusters it
  touched (no whisper/audio on the rest).

## 7. UI flow (`dupdetect ui`) · `MainWindow`
Read/act layer over the DB — **never re-scans or retrains**:
- Loads clusters (`load_clusters`/`sort_clusters`), shows a sortable/filterable tree (sort by copies /
  reclaimable space / confidence; filter actionable / all / review).
- **Scan/Watch panels** spawn the CLI as a `QProcess`, parse progress (`tqdm`, `REPAIR_PROGRESS`,
  `on_detect`/`VIZ:` lines), and stream the "What the AI sees" live-view (on-demand via the `viz.on`
  signal file; cosmetic, never affects a verdict).
- **Actions:** open in VLC, send to Recycle Bin (`send2trash`) with a listing confirmation →
  `record_deletion` audit + optimistic `remove_paths`/`prune_singleton_clusters`. Feedback
  (same/different) → `save_feedback` → recalibration. Repair queue → `repair-indexes` subprocess.

## 8. Per-stage data shapes & measured cost (Pass-1, one HD film, reference HDD)

From [HPC_PIPELINE.md](HPC_PIPELINE.md) (Windows 11, Blackwell GPU, library on a spinning Storage
Space, `--workers 6`, ~5 s/film):

| Stage | In → Out | Resource | Cost | Bottleneck |
|-------|----------|----------|------|------------|
| ffprobe | path → `Probe` | disk (light) | 0.11s | no |
| content_hash | path → 64-bit hex | disk (24 MB sampled) | 0.12s | no |
| audio_fp `fpcalc -length 0`† | path → `uint32[M]` | full demux | 3.3s | partial |
| detect_language (whisper base)† | path → lang str | CPU | 3.9s | partial |
| **keyframe demux** | path → `uint8[N,224,224,3]` + `times[N]` | **disk (serial, main)** | **~5s** | **yes** |
| H2D | numpy → CUDA `[N,3,224,224]` | PCIe | 0.04s | no |
| embed DINOv2 fp16 | `[N,3,224,224]` → `[N,768]` fp16 | GPU | 1.4s | no |

† In the current pipeline these are **deferred / on-demand** (audio-fp and whisper run in Pass-2 /
cluster-rank, not Pass-1) — the table reflects their standalone cost. The GPU sits **idle** waiting on
the platter; this is why perf levers ship off-by-default and auto-tune caps HDD concurrency at ~2.

## 9. Pass-2 data flow for one candidate pair (exact types)

```
rec.global_vec  float32[768] ──► CoarseIndex.query_global(k=25)      ┐
rec.window_vecs float32[12,768]─► CoarseIndex.query_windows(k=10)    ├─► candidate set (paths)
duration ±3%    ──► find_by_duration ──► gate_by_global(≥0.7 cosine)  ┘
        │  for each canonical pair (a≤b), once (seen-set):
        ▼
   audio:  _ensure_audio_fp(a,b)  uint32[M] ──► align_audio (FFT xcorr) ──► AlignResult(score,offset,cov)
   video:  cache.get → fp16[N,768]; resample_to_grid(step 2s) ──► banded SW ──► AlignResult(score,cov,superset,interleaved)
   scenes: scene_cuts float32[K] ──► np.diff ──► DTW ──► AlignResult(score,cov)
        ▼
   decide_tree(a,b, audio,video,scenes, th) ──► Result(verdict ∈ {CERTAIN…DIFFERENT}, confidence, reason)
        ▼
   if verdict ≠ DIFFERENT and _both_on_disk(a,b):  store.save_match(... + ad_offset + audio/video/scenes JSON)
```
- **Parallel path** (`match_pairs_parallel`): the main process computes+persists each involved file's
  `audio_fp` once, then read-only worker processes run the three aligns on **numpy** (no torch import →
  fast spawn). One `_pass2_pair` per pair, `chunksize=4`. Deterministic per pair → verdict invariant.
- **Sequential path** (`match`): same logic with an LRU `EmbeddingCache(max_items=1500)` of resident
  fp16 tensors; video matmul + DP may run on GPU when CUDA is present.

## 10. Concurrency & safety summary (how the flows coexist)
| Mechanism | Protects against | Where |
|-----------|------------------|-------|
| WAL + `busy_timeout` | reader/writer starvation | `store.__init__` |
| scan-priority PID lock | HDD thrashing / SQLite write starvation between scan & watcher | `scan_priority_lock`, `scan_in_progress` |
| atomic `replace_clusters` | concurrent rebuilds fusing unrelated components | `store.replace_clusters`, `_stable_cluster_id` |
| `_both_on_disk` guard | resurrecting a mid-scan-deleted file's match | `_pass2_*` |
| root-reachable guard | an unmounted drive read as a mass deletion | `orphan_paths` |
| canonical paths | `/`-vs-`\` phantom duplicate records | `canonical_path`, `_route_event` |
| skip-and-report | one bad file crashing the batch/loop | `problems` table, try-blocks |
</content>
