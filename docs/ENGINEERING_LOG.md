# ENGINEERING_LOG — solved problems & non-obvious gotchas (Dupless Video)

**Purpose:** durable memory of things already figured out, so they are never re-investigated.
Grep this file (by symptom) before any non-trivial debugging or "why does X happen" exploration;
append an entry when you solve something that took real digging. A fix without a log entry is half-finished.

**Entry format:** `### <searchable symptom>` then **Symptom / Root cause / Resolution / Files-refs / Date-scope**.
Newest on top. Append-only in spirit. Deep rationale for the design invariants lives in [CLAUDE.md](../CLAUDE.md).

---

## Entries

### Mega-cluster persisted after the CONTAINS fix — a POISONED per-user thresholds override (θv 0.5)
- **Symptom:** after shipping the CONTAINS guard (entry below) a fresh Standard scan STILL produced a
  1338-member cluster (down from 1621, so CONTAINS worked — 31k CONTAINS rows emitted — but something
  else kept fusing).
- **Root cause (config, NOT code):** `%LOCALAPPDATA%\Dupless Video\thresholds.yaml` — the per-user
  override that `effective_config_path()` prefers in the frozen app — held **θv 0.5 / θa 0.55**
  (shipped: 0.75 / 0.80), written by the old degenerate recalibration (one-sided feedback, recall 0.0)
  before the MIN_PER_CLASS guard existed. Proof from the DB: **3855/4796 (80%) strong-tier rows had
  video < 0.75** (e.g. `T1 audio(0.56)+video(0.60)`). Same-genre pairs at video 0.5–0.7 flooded
  T1/T3 → union-find fused them. Timeline matched exactly: override mtime 2026-07-18, first "detection
  broke" report right after.
- **Recovery (app pipeline, no re-scan):** stored `audio_json/video_json/scenes_json` make re-deciding
  cheap → `actions.apply_thresholds(0.75, 0.80, config_path=<override>, store=…)` (the same function
  behind Recalibrate→Apply) re-decided all 344k matches in ~21 min and rebuilt clusters. Result:
  DIFFERENT 40→319,963 · PROBABLE 280k→20,665 · CONTAINS 31k→1,589 · HIGH 3,456→36 · strong rows with
  video<0.75 = **0** · biggest cluster **1338 → 4 members** (1,217 clusters / 2,506 files). DB backed
  up first (`dupdetect.sqlite.bak-20260722`).
- **Prevention (already shipped in 0.1.3):** `suggest_thresholds` refuses degenerate label sets
  (missing class → `degenerate: True`, thresholds untouched) and the UI explains instead of applying.
- **Lesson:** when detection quality "suddenly regresses", check `effective_config_path()` FIRST —
  a stale per-user override silently outranks the shipped defaults in the frozen app, and no code
  diff will explain it. The DB `reason` strings carry the scores → grep them against the shipped θ
  to prove/disprove a loose-threshold episode in minutes.

### Mega-cluster of unrelated videos — a compilation chain-fuses its neighbours (CONTAINS guard)
- **Symptom:** a Standard scan produced a 1621-member "duplicate" cluster of unrelated videos (also
  32- and 19-member ones). Quality dropped sharply vs the prior build.
- **Root cause (NOT a loose threshold — §0 held):** clusters are built by union-find over duplicate
  matches, i.e. a *partition*. But the data is a *graph*: a compilation legitimately shares scene X with
  video_02 and scene Y with video_05. Each such pair aligned as a strong tier (video≈0.9, cov≈0.99 —
  because `coverage` = path/`min(na,nb)`, it measures only the SHORTER file, so a clip that sits fully
  inside a long compilation reads cov≈1.0 and looks like a duplicate). Union-find then fuses the
  compilation with its unrelated neighbours, and each neighbour drags in *its* neighbours → one giant
  blob. The compilation is the hub.
- **Fix (a fork, approved — new relationship type, not a looser tier):** a new signal
  `coverage_long = coverage * min_dur/max_dur` = the fraction of the LONGER file that aligns (valid
  because `resample_to_grid` makes frame counts ∝ duration). When video aligns strongly and `coverage`
  is high but `coverage_long < min_coverage_long` (0.5) → `Verdict.CONTAINS` (see
  `match/tree.py::_structural_verdict`). CONTAINS is NOT in `DUPLICATE_VERDICTS` nor `REVIEW_VERDICTS`,
  so union-find never groups it and it never floods review — it is a *relationship*, surfaced (Part 2)
  as "related, not duplicates" alongside DIFFERENT_EDITION. A real duplicate (both files ~fully aligned)
  keeps `coverage_long≈coverage` → unchanged. §0 intact: the guard only *demotes* (never promotes) and
  fires before the strong tiers.
- **Ledger gotcha:** the incremental `evaluated_pairs` ledger keyed only on `feature_version + th.raw`,
  so a code-only change to the tree (this guard) would NOT invalidate cached evaluations → a normal
  re-scan would skip these pairs and never reclassify them. Added `tree.DECISION_VERSION` (bump on any
  verdict-flipping logic change) folded into `_scan_fingerprint`. To reclassify an already-scanned
  library: re-scan (the ledger now misses on the version bump), or Force-recompute to be certain.
- **Tests:** `test_tree.py::test_contains_clip_in_compilation_is_not_a_duplicate`,
  `::test_similar_duration_full_match_stays_a_duplicate`,
  `test_fullscan.py::test_contains_edge_does_not_fuse_clusters` (the CONTAINS edge must not union).

### Healthy files flagged "NO AUDIO / audio truncated" (false quality warnings) — probe v2 + user fixes
- **Symptom:** files that play fine (audio present) show in Quality warnings / carry the per-copy ⚠,
  blocking auto-KEEP on their clusters; nothing in the UI could correct it (the tab was triage-only)
  and the cached value never re-measured (only a global force re-scan reset it).
- **Root cause (three failure modes in `scan_audio_coverage` v1):** (1) 0.5s `volumedetect` windows —
  a dialogue pause / quiet ambient scene averages < −60 dB and counts as "no audio"; with the KEEP
  muted-check's 12 points, TWO genuinely quiet windows flag a healthy film. (2) Only the FIRST audio
  stream was probed (`-map 0:a:0?`): a commentary/low-level track 0 false-flags a file whose real
  audio is on track 1. (3) Fixed 40 points over-probes shorts and under-probes 3h movies. Plus:
  coverage is computed once and cached forever (NULL = not computed), with no targeted re-measure
  and no per-file correction.
- **Resolution:** probe v2 + two user-facing fixes; thresholds NOT loosened (§0 spirit), and
  audio_coverage never enters `decide_tree` (verified + regression test), so verdicts are untouched.
  1. **Probe v2** (`scan_audio_coverage`): 5s windows (averages across pauses; cost is SEEK-dominated
     so decoding 5s vs 0.5s is ~free, §1), duration-scaled density (`coverage_points`: one probe per
     ~2 min, clamp [8,48] — deterministic, §0), and ALL audio streams in one pass (`-map 0:a?` emits
     one `mean_volume` PER stream — verified against the bundled ffmpeg with a 2-track file; any
     stream above −60 dB counts). Genre-adaptivity was deliberately NOT content-based (measure,
     don't classify): wide windows + scaled density absorb the variance; the residue is what the
     manual override is for.
  2. **Rollout without re-decode:** `COVERAGE_VERSION` NOT bumped (it lives inside feature_version →
     would re-decode+embed the whole library). Instead a one-shot `PRAGMA user_version`-keyed
     migration NULLs ONLY the rows the v1 probe flagged (< 0.85) → they lazily re-measure with v2;
     clean caches survive, and a legitimately-low v2 value is never re-NULLed on reopen.
  3. **'↻ Re-check audio'** (Quality tab + duplicates-tree context menu): re-measures with full v2
     density and OVERWRITES the cache (`ensure_audio_coverage(force=True)`), off the GUI thread
     (`_AudioFixWorker(QThread)`, own store handle). **'✓ Mark audio as OK'**: per-file override in
     the new `quality_overrides` table, stamped with mtime+size so it AUTO-EXPIRES when the file
     changes (a truly muted re-download warns again); revertible ("Restore measured value" — the
     measured `files.audio_coverage` is never rewritten). Consumed at exactly three read points:
     `ensure_audio_coverage` (pipeline: KEEP muted-check, color-keep guard, rank evidence),
     `audio_warnings` (tab), `load_clusters` (audio_bad/audio_warning/is_actionable coalesce).
     After a fix, the TOUCHED clusters re-rank via the reuse-snapshot-minus-affected pattern
     (`_rebuild_clusters(reuse=…)`) so a stale "⚠ NO AUDIO" in the persisted rank_reason regenerates.
- **Files / refs:** `features/audio_fp.py` (`coverage_points`, `COVERAGE_WIN_S`, v2 probe);
  `store.py`/`schema.sql` (`quality_overrides`, `set/clear/has_quality_override`,
  `quality_overridden_paths`, user_version migration, `audio_warnings` filter, `forget_file`);
  `pipeline/analyze.py` (`ensure_audio_coverage` override/force); `ui/data.py` (`load_clusters`),
  `ui/model.py` (`AUDIO_BAD_ROLE`), `ui/main.py` (`_AudioFixWorker`, `_audio_fix`, tab buttons,
  context menu). Tests in `test_audio.py`, `test_store.py`, `test_pipeline.py`, `test_tree.py`
  (§0 invariance), `test_ui.py` (worker end-to-end + coalesce).
- **Scope:** decided 2026-07-06.

### Review sweep: Problems-tab ghosts (Mode B), POSIX prune was a silent no-op, M4 move-clone, UI prune off the GUI thread
- **Symptom:** a code review of the file-existence reconciliation found residual holes after the
  2026-06-21 fix: (a) the Problems tab kept ghost rows for files whose whole FOLDER was deleted;
  (b) `_volume_root` rejected the POSIX anchor `/`, making `prune_missing_files` a silent NO-OP on
  Linux (its own committed tests — `test_perf_opts` POSIX branch, `test_full_sweep_closes_mode_b` —
  would have failed on the ubuntu CI at the next push; the last green run predated them); (c) a MOVED
  byte-identical file paid a full re-decode+embed although `find_by_hash` (M4) existed for exactly
  that clone — it had NO caller and `analyze_file`'s docstring claimed a short-circuit that wasn't
  implemented; (d) the on-open disk reconcile ran synchronously on the Qt main thread.
- **Root causes:** (a) `prune_missing_problems` still used the parent-dir guard ('parent isdir =>
  truly deleted') — for a deleted folder the parent IS the folder, so it read as 'volume offline' and
  kept the row forever, exactly the hole the 2026-06-21 entry documents for the duplicates list.
  (b) The bare-separator rejection in `_volume_root` (meant for Windows' degenerate `\` anchor that
  resolves to the current drive) also caught POSIX `/` — but on Windows `/x` normalizes to anchor
  `\`, so `/` can ONLY mean the real POSIX root; rejecting `\` alone is sufficient and portable.
- **Resolution:** ONE shared decision core `_missing_on_online_volume` (volume probe + per-file
  `_real_deletion`) now backs BOTH `prune_missing_files` and `prune_missing_problems` (injectable
  `exists/isdir/ismount`, deterministic tests for Mode B / offline drive / offline junction on the
  problems table too). `_volume_root` accepts POSIX `/`. M4 implemented as `record_from_donor`
  (analyze.py): same sampled hash AND size (the T0 identity standard) AND same feature_version →
  clone the donor's content-derived features; identity fields (path/mtime/size/probe/hash) come from
  the fresh CpuFeatures so no stale data survives, and the donor's raw `audio_coverage` is re-read
  from the DB so NULL ('not yet computed') isn't persisted as a fake 1.0. Wired into `analyze_file`
  (watcher/serial) and `_gpu_finish`'s serial-decode path (parallel Pass-1) — the pipelined path
  already decoded, so it doesn't check. UI: `_refresh_with_prune` now paints immediately and runs
  prune+rebuild in a `_PruneWorker(QThread)` with its OWN store handle (SQLite is per-thread; WAL);
  the queued `done(n)` refreshes only when something was forgotten; failures are printed, never
  swallowed. Perf hygiene alongside: `idx_matches_b` (the `a_path=? OR b_path=?` deletes/ads-checks
  scanned the whole matches table per call), one transaction per prune batch (was one commit per
  forgotten file), and the freshness contract unified in `analysis_state` (was triplicated across
  `_needs_analysis` / `pending_files` / `_IngestScheduler._ready`, one drift away from a new
  scan-vs-watcher bug). UI tests with synthetic '/x' fixtures neutralize BOTH sweeps
  (`_quiet_reconcile`) — with `/` now a reachable volume on Linux, the background worker would
  otherwise forget the fixtures mid-test.
- **Files / refs:** `store.py` (`_missing_on_online_volume`, `_volume_root`, `prune_missing_*`,
  `forget_file(commit=)`, `find_by_hash(size=)`, `_emb_file`; `iter_problems` removed — use
  `problems()`); `schema.sql` `idx_matches_b`; `pipeline/analyze.py` (`analysis_state`,
  `record_from_donor`); `pipeline/fullscan.py` (`_needs_analysis`, `_gpu_finish`); `watch.py`
  (`pending_files`, `_IngestScheduler._ready`); `match/matcher.py` (dead duration-gate condition);
  `ui/main.py` (`_PruneWorker`, `_refresh_with_prune`, `_on_prune_done`); tests in
  `test_perf_opts.py`, `test_store.py`, `test_pipeline.py`, `test_ui.py`.
- **Scope:** decided 2026-07-06.

### Files deleted outside the app linger in the duplicates list forever ("I deleted folders, still see them")
- **Symptom:** the UI keeps showing files/clusters for videos the user deleted in Explorer. Opening or
  refreshing the app does NOT clear them.
- **Root cause:** the UI is a pure VIEW — `ui.data.load_clusters` reads `clusters ⋈ files` with NO
  existence check, and nothing prunes the `files`/`clusters` tables on open. Reconciliation lived ONLY
  in (a) the in-app delete (`actions.delete_files` → `forget_file`) and (b) the background watcher.
  The watcher's `orphan_paths` (watch.py) guards by WATCHED-ROOT reachability: delete the whole watched
  root and `os.path.exists(root)` is False → the root is dropped (its §2 offline-drive guard) → its
  files are never forgotten (**Mode B**). With no watcher running (**Mode A**) nothing prunes at all.
- **Resolution:** new `FingerprintStore.prune_missing_files()` (store.py) — a §2 self-heal the UI runs
  on open / switch_db / ↻ / a "🧹 Clean missing" button (NOT per-keystroke; SKIPPED while a scan holds
  the priority lock so it can't race Pass-2's `_both_on_disk` resurrection guard §0 or fight the HDD
  §1). The CALLER rebuilds clusters with `reuse=_snapshot_clusters()` (only touched clusters re-rank).
  **Guard design was shaped by an adversarial review that REPRODUCED mass-forget holes in the naive
  "volume-anchor only" version** — forgetting records is recoverable (re-scan rebuilds; the file on disk
  is never touched), but wiping the index on an unmount is expensive, so it must be fail-safe:
    1. VOLUME fast-skip — one `exists(anchor)` probe per drive; an unmounted drive, or a degenerate /
       unknown anchor (bare-backslash "UNC" `\\`, drive-relative `C:foo`) → skip ALL its files. The
       degenerate forms were found to otherwise resolve to the current drive root (reachable) and
       mass-forget legacy rows.
    2. MOUNT-AWARE per-file (`_real_deletion`) — a gone file counts as a real deletion only if NO
       disconnected mount/junction sits between it and the nearest present dir. An offline nested NAS
       **junction** under a mounted drive (`exists('L:\\')` True, the subtree unreachable) is KEPT
       (`Path(x).is_junction()` reads the reparse entry even when the target is offline); a plainly
       deleted SUBFOLDER on an online volume is CLEANED (Mode B). The simple parent-dir guard
       (`prune_missing_problems` style) does NOT work here — it would skip Mode B (the deleted folder
       is the parent). POSIX caveat: anchor is always `/`, so nested-mount protection there leans on
       `os.path.ismount`; documented as Windows-precise.
  `exists`/`isdir`/`ismount` injectable → the decision core is deterministic and unit-tested without
  real drives/junctions (offline-drive kept, offline-junction kept, Mode B forgotten, degenerate
  anchors kept). Verified end-to-end: open a real DB with one present + one missing file → the missing
  one is forgotten, the present one kept, the orphaned cluster collapses.
- **Files / refs:** `store.py` `prune_missing_files` + `_volume_root`/`_volume_reachable`/`_is_mount`/
  `_real_deletion`; `ui/main.py` `_refresh_with_prune` + "Clean missing" button; tests in
  `test_perf_opts.py`, `test_ui.py`. The watcher's `_full_sweep` (watch.py) now ALSO calls
  `prune_missing_files` (after `orphan_paths`, gated by `scan_in_progress` so it never races a scan /
  fights the HDD), closing Mode B in the background too — `orphan_paths` handles files under a present
  root, the volume-prune catches files whose root vanished while the drive stayed online. The feared
  normcase divergence is moot: `os.path.exists` is case-insensitive on Windows, so a differently-cased
  stored path still resolves and is not spuriously forgotten. Tests in `test_watch.py`.
- **Scope:** decided 2026-06-21.

### Pass-2 (candidate matching) ETA ~8h / "is the parallel pass even using the cores?"
- **Symptom:** a full scan sat in Pass-2 at ~59 pair/s, ETA ~8h for 1.9M candidate pairs (~19k files,
  library on a spinning HDD). The in-code comment claimed Pass-2 was "COMPUTE-bound (banded DP ~89%),
  scales across cores" — but 30 workers were yielding only 59 pair/s.
- **Root cause (MEASURED — two assumptions overturned):** (1) NOT I/O-bound: embeddings are 18 GB of
  small fp16 `.npy` on an **NVMe** (C:), with 128 GB RAM → the whole set fits in page cache, so the
  per-pair re-read is free (the HDD holds the *videos*, read in Pass-1, not the embeddings). The
  "obvious" cache/locality fix would buy nothing. (2) The real costs, by a synthetic N-sweep + a
  per-stage profile (`scripts/bench_align_split.py`, `scripts/profile_pass2.py`, BLAS pinned to 1
  thread): the per-pair **similarity matmul `a@b.T` is only ~3-5%** for typical content; the
  **banded Smith-Waterman DP (`align_video.banded_align`) is ~78-95%** of `align_video`, and the
  **scenes DTW (`align_scenes`) is a separate ~34%** of total Pass-2 — both pure-Python numeric loops.
  Plus **OpenBLAS thread oversubscription**: 30 worker processes × a multi-threaded matmul (no cap) =
  ~30×32 threads on 32 cores. The matmul/DP split FLIPS with duration: matmul is O(N²) vs the DP's
  O(N·band), so it only overtakes the DP past ~N=8000 (~4.5 h runtime) — relevant as the library
  trends to long 4K/8K (N = duration/grid, resolution-INDEPENDENT). **GPU offload of the matmul was
  rejected**: it reaches only the ~3% matmul and would need to ship sim/band matrices to the CPU DP
  workers (~12 TB IPC) — a net loss (the DP can't go on GPU; per-row GPU kernels measured ~15× slower).
- **Resolution (three levers, all §0-invariant, measured):** (1) **Pin BLAS to 1 thread/worker** around
  the Pass-2 pool (`matcher.single_threaded_blas`, env set in the parent before spawn) — parallelism
  comes from the processes. (2) **Banded matmul** for `min(na,nb) > 2400`: fill only the diagonal band
  the DP reads via row-block GEMMs (`align/video._banded_matmul`); `banded_align` is unchanged →
  identical path. N=8000 matmul 1087→255 ms (4.3×). (3) **Cython** the two hot loops
  (`align/_fastdp.pyx`: `banded_align_fast`, `scenes_dtw_final`) — typed memoryviews, pure-Python
  fallbacks kept as reference; DP 904→253 ms at N=8000 (and 61→8 ms at N=500). Combined per-pair
  video-align **3.1-4.7× faster** (giant pair 1926→477 ms), verified bit-identical (randomized
  fast==pure, 0 mismatches/200). The Cython ext is OPTIONAL (`Extension(optional=True)` + import
  fallback): no compiler → pure-Python, nothing breaks; PyInstaller bundles the `.pyd` only if built.
- **Files / refs:** `match/matcher.py` (`single_threaded_blas`, `_pass2_init`); `align/video.py`
  (`_banded_matmul`, `_BANDED_MATMUL_MIN_N`, `banded_align` dispatch); `align/scenes.py`
  (`_dtw_final_py` + dispatch); `align/_fastdp.pyx`; `setup.py`; `scripts/bench_align_split.py`,
  `scripts/profile_pass2.py`; tests `test_align_video.py`, `test_scenes.py`, `test_perf_opts.py`.
- **Scope:** decided 2026-06-21. NOT yet measured end-to-end (a real multi-hour scan); the per-stage
  microbenchmarks quantify the per-pair win. Block size 256 / gate 2400 chosen by the bench, not tuned.

### Pass-2 "Standard" sits for hours with the CPU at ~0% (the audio-fingerprint pre-pass)
- **Symptom:** a Standard scan of ~15k files on a spinning HDD showed `Pass 2 (duplicates) · 5614/15005
  · 3.62s/file · ETA 9h`, the app's CPU at **0.1%**, disk at 0.2 MB/s. Looked frozen / "why is matching
  so slow and the CPU idle?".
- **Diagnosis (what it actually was):** NOT the GPU, embeddings, or video align. The UI relabels *every*
  line containing "Pass 2" to "Pass 2 (duplicates)" (`ui/scan_panel.py:_parse`), so the text hides the
  sub-stage; the **unit `s/file`** + total = library size is the tell. It was the **audio-fingerprint
  pre-pass** inside `match_pairs_parallel`: a SERIAL main-process loop running `fpcalc` once per
  *involved* file. `involved` ≈ the whole library because faiss top-k returns candidates for *every*
  file, so the "on-demand, most films never pay" intent (matcher comment) degenerated to "all files".
  CPU ~0% because each `fpcalc` is a subprocess blocked on **HDD seek latency** decoding the audio
  stream (whole-file for ≤1h content, 600s cap above) — I/O-bound, not compute. 15005 × 3.62s ≈ 15 h
  for a pre-pass that runs *before* a single pair is compared.
- **Cost model:** the cost is **extracting** the fingerprint (whole-audio decode off the HDD), not
  comparing it (`align_audio` is fast FFT; video runs on page-cached embeddings). So "audio-first" is
  worse; the cheap coarse filter is already video (faiss over one vector/file from Pass-1).
- **Resolution — LAZY AUDIO (video-first), approved fork:** the fingerprint is now extracted on-demand
  and ONLY when the video warrants it. Tracing `decide_tree`, audio affects the verdict only in T1/T2,
  which both require `video.score >= theta_v AND coverage >= min_coverage`. The one exception was T4b's
  **audio-only OR-branch** (`or audio.score >= theta_a`), the sole tier consulting audio for a
  video-weak pair — **removed** (with the user's OK, §4). After that, gating audio on
  `_audio_warranted(v)` = `video.score >= theta_v and video.coverage >= min_coverage` is
  **verdict-identical** for every other tier (an empty `AlignResult` yields the same tier when video is
  weak). Unique films and their weak faiss neighbours never decode audio. Workers extract+persist the
  fp themselves now (WAL serializes the few writes); the serial main-process pre-pass is gone.
- **Trade-off (the fork):** a pair with matching AUDIO but DIFFERENT video no longer reaches the review
  queue. Near-empty for a *video* dedup (re-encodes keep the video alignable; cam rips kill the audio),
  and the strong tiers are untouched → zero-FP guarantee intact (§0). Coverage/"went silent" detection
  is unaffected (it's the separate sparse `scan_audio_coverage`, not the fingerprint).
- **Files / refs:** `match/tree.py` (T4b), `match/matcher.py` (`_audio_warranted`,
  `_audio_if_video_warrants`, `match`, `_pass2_pair`, `name_pair_content_differs`,
  `match_pairs_parallel` — pre-pass deleted); tests `test_tree.py`
  (`test_t4b_audio_only_no_longer_reviews`, `test_t1_requires_coverage`), `test_pipeline.py`
  (`test_lazy_audio_skips_fingerprint_for_video_weak_pairs`), `test_fullscan.py` (the two re-decide
  threshold tests updated to demote via the video-corroborated T4b).

### Pass-2 looks frozen — bar stuck at 0% for an hour while every core is pegged
- **Symptom:** a Standard scan sat ~1 h with NO visible progress ("no indicator at all, the user thinks
  it hung"), yet the scan process had **17 workers, 107 threads, ~9 cores of CPU and 0 MB/s disk** —
  i.e. genuinely crunching, just invisibly. Pass-2 align phase.
- **Root cause (two silent gaps in `match_pairs_parallel`):** (1) **Candidate enumeration** — an
  O(files) serial loop of faiss/duration queries in the MAIN process — emitted NO tqdm line, a
  multi-minute silent gap. (2) The align bar wrapped an **ORDERED `pool.map(..., chunksize=4)`**: tqdm
  advances only as results arrive *in submission order*, so a single slow giant pair at the head
  stalled the bar at 0% while the other W-1 workers had already cleared thousands of pairs — CPU
  pegged, bar frozen. Removing the old "Pass 2 (audio fp)" pre-pass bar (lazy-audio change) deleted the
  one intermediate signal that used to move, so the UI was stuck on the last line ("Building coarse
  index…"). The UI relabels every "Pass 2" line to "Pass 2 (duplicates)", hiding which sub-stage stalls.
- **Resolution (§2 — never look frozen):** `match_pairs_parallel` is now a **generator** that (1)
  shows a **"Pass 2 (candidates)"** bar over the enumeration, and (2) drains the pool with a
  **bounded window** (`_drain_pairs_bounded`): `wait(FIRST_COMPLETED)` advances the bar on EVERY pair
  that finishes, in any order — a slow giant can no longer stall it. Yielding per-completion also lets
  `_pass2_parallel` **persist each match incrementally** (`save_match` as rows arrive) instead of only
  after the whole batch, so a cancel/crash keeps finished work. Completion order ≠ submission order
  changes only the persist order, idempotent per canonical pair → verdict-invariant (§0).
- **Bounded submission (RAM):** an early `as_completed(pool.submit(...) for all pairs)` would build a
  Future per pair — **millions** on a dense library (GBs). `_drain_pairs_bounded` keeps only
  `4·workers` in flight and refills one slot per completion → memory O(workers), pool never starves.
- **Worker cap (perf, §1 — measured idle cores):** `_pass2`'s `match_workers` was hard-capped at
  `min(16, …)`, leaving **14 of 32 cores idle** on this box. Align is CPU-bound with BLAS pinned to 1
  thread/worker (no oversubscription), so it scales ~linearly → raised to `min(32, cpu-2)` ≈ 30 workers
  (~1.8× expected: ~1 h → ~35 min). Speed only, verdict unchanged (§0).
- **Files / refs:** `match/matcher.py` (`match_pairs_parallel` → generator; `_drain_pairs_bounded`);
  `pipeline/fullscan.py` (`_pass2` worker cap 16→32; `_pass2_parallel` saves incrementally); test
  `test_pipeline.py::test_drain_pairs_bounded_consumes_all_and_filters_none`.

### Cluster ranking ("Group" step) cost ~5 h, mostly wasted — and Pass-2 re-aligns everything on every run
- **Symptom:** after the align finished, the scan sat hours in the silent "Group" phase; and a second
  `Analyze` with nothing changed re-did the whole ~1 h alignment. "Why reprocess if nothing changed?"
- **Root cause (two):** (1) `rank_cluster` ran **whisper (language) + whole-file audio coverage for
  EVERY cluster member** — ~17-20 s/member off the HDD (the audio decode, not GPU). Measured: language
  only flips KEEP in a minority (resolution already dominates per the rank comment), and coverage only
  matters when copies DIFFER — so most of that work changed no decision. (2) Pass-2 **re-aligned every
  candidate pair on every run**: `match_pairs_parallel` had no memory of evaluated pairs, and DIFFERENT
  verdicts aren't persisted (they're ~all of the 2.67M pairs), so there was nothing to skip them by.
- **Resolution A — lean ranking (Phase 1):** KEEP now scored from **metadata only** (no decode):
  RESOLUTION >> no-ads >> lower-cam >> **codec-aware effective bitrate** (`_effective_bitrate`:
  bitrate × codec efficiency, so an AV1/HEVC copy isn't discarded for its lower raw bitrate) − clipping.
  Audio is reduced to one question — **is the KEEP muted?** — probed lazily DOWN the ranking
  (`_keep_by_audio`): KEEP the first non-muted copy (escalate past a muted "best"); if none is clean,
  keep the top and warn only when coverages DIFFER. (`_color_adjusted_keep` won't undo this — it never
  downgrades KEEP onto a less-clipped copy that has WORSE audio; probes that target's coverage lazily.)
  Muted-check uses few seek-probes
  (`_KEEP_COVERAGE_POINTS=12`; 40 was ~28 s on a 44 GB file, ~3 s at 8). **Language is OFF by default**
  (`rank_cluster(detect_lang=False)`), opt-in for on-demand language-preference KEEP. Measured: ranking
  ~5 h → ~15-30 min (mostly removing whisper; coverage only on KEEPs).
- **Resolution B — incremental Pass-2 (evaluated-pairs ledger):** new `evaluated_pairs(pair_hash,
  fingerprint)` table records EVERY aligned pair (match OR DIFFERENT). `fingerprint` = feature_version +
  all thresholds (`_scan_fingerprint`) — a change clears the ledger (looser θ can turn a DIFFERENT into
  a match). On a re-run, `_enumerate_pairs` SKIPS a pair whose hash is in the ledger UNLESS an endpoint
  is in `changed` (= files Pass-1 re-analyzed this run, `_changed_paths`, captured BEFORE Pass-1). A
  re-run with nothing changed aligns ZERO pairs (just the cheap enumeration). Recording happens only
  when the generator fully drains, so a cancelled scan records nothing → never wrongly skips next time.
- **Forks (approved):** language dropped from default KEEP (on-demand); the muted guard now PROMOTES the
  next-best copy with audio instead of blanket-reviewing (UI never auto-deletes, so the suggestion is
  safe; §0 strong tiers untouched). Stale-match pruning for a re-encoded file is a pre-existing gap (a
  DIFFERENT re-verdict doesn't delete the old `matches` row) — unchanged by this work, noted for later.
- **Files / refs:** `pipeline/fullscan.py` (`_effective_bitrate`/`_CODEC_EFF`, `_keep_by_audio`,
  `rank_cluster(detect_lang=…)`, `_score_member`, `_rank_evidence`, `_changed_paths`,
  `_scan_fingerprint`, `_pass2`/`_pass2_parallel` plumbing); `match/matcher.py` (`_enumerate_pairs`,
  `_pair_hash`, ledger wiring in `match_pairs_parallel`); `store/store.py`
  (`evaluated_pairs_load`/`_add`); `store/schema.sql` (`evaluated_pairs`); `pipeline/analyze.py`
  (`ensure_audio_coverage(n_points=…)`); tests in `test_fullscan.py`/`test_ui.py`/`test_pipeline.py`.
  Incremental ledger is parallel-path only (small/single-core libraries re-evaluate — cheap there).

### Background analyzer "never finishes" (stuck at 18992/19014) — re-attempts corrupt files every sweep
- **Symptom:** the tray watcher sat IDLE (0 CPU, no child procs) yet showed `18992/19014`, looking
  unfinished. The 22 "missing" were the SAME corrupt files (moov atom not found, truncated mp4s) every
  cycle — "they were already attempted; failing is a result, not pending."
- **Root cause:** every place that decides what to analyze used only `store.has_fresh(p, st, fv)`
  (`watch.pending_files`, `watch.Scheduler._ready`, `fullscan._pass1`'s `todo`). A corrupt file gets a
  LITE record (`exact-only-v1` fv, no embeddings) + a `problems` row, but `has_fresh` is False (fv
  mismatch + no emb_path), so it counts as "new/changed" → re-decoded EVERY sweep → fails → re-reported
  → still LITE → forever. Pure waste, and the "processed" count can never reach 100% (those files never
  get embeddings). §2 said "skip and report", but the report wasn't REMEMBERED across cycles.
- **Resolution:** the `problems` table now stamps the file's **mtime+size** at failure
  (`save_problem`), and a new `store.has_unchanged_problem(path, st)` returns True while the file is
  UNCHANGED. The three work-selection sites skip a file that is fresh OR a known-unchanged problem
  (`fullscan._needs_analysis`). A re-download / remux changes mtime|size → the guard lifts → it's
  retried; `--force` always re-attempts. A NULL-mtime row (file gone at failure) never matches.
- **Migration / self-heal:** existing problem rows have NULL mtime/size (added by the ALTER), so they
  get re-attempted ONCE more (which stamps them), then are skipped thereafter.
- **Files / refs:** `store/schema.sql` + `store.py` (`problems.mtime/size`, `save_problem`,
  `has_unchanged_problem`); `watch.py` (`pending_files`, `_ready`); `pipeline/fullscan.py`
  (`_needs_analysis`); tests `test_store.py::test_has_unchanged_problem_guards_known_corrupt`,
  `test_watch.py::test_pending_files_skips_unchanged_corrupt`,
  `test_fullscan.py::test_needs_analysis_skips_unchanged_corrupt`. Ships in the next build (the running
  tray watcher is the frozen .exe).

### Burst clips grouped as duplicates — different recordings of the same static scene (T2 FP)
- **Symptom:** a folder of phone clips (`Fotos/Por_fecha`, 1–3 s, 1440p) all fused into ~2 huge
  "duplicate" groups across DIFFERENT timestamps (4.09.35 ↔ 4.09.38 ↔ 4.09.43…). Measured: 730
  cross-timestamp pairs at **VERY_HIGH**, video.score 0.85–1.00, audio 0.0.
- **Root cause:** these are different recordings of a near-static scene, so their video aligns ~1.0.
  **T2** ("video identical + audio doesn't align ⇒ same video, different dub") fired — but a 2-second,
  4-frame static clip has NON-discriminative video (any two clips of the same wall align). The tree
  already guards the scenes-only tier with `min_cut_density`, but the STRONG video tiers (edition/T1/
  T2/T3) had NO discriminative-content check. NOT a clustering/id bug (union-find over real matches);
  NOT this session's lazy-audio/ranking work — the matches came from the earlier scan.
- **Resolution (approved §0 fork):** `_discriminative(a, b, th)` gates edition/T1/T2/T3 on
  `min(dur_a, dur_b) >= th.min_strong_duration_s` (default 15 s). Below it the pair falls to T4b →
  review (the doubt path), never a strong duplicate. T0 (byte-identity) is unaffected. Re-validated:
  the labeled calibration set is full-length content (durations ≫ 15 s) → 0-FP / recall unchanged; the
  calibration `_mk` stub's `Probe(0,…)` was bumped to a real duration so the guard doesn't misread it.
- **Files / refs:** `match/tree.py` (`_discriminative`, gates); `config.py` (`min_strong_duration_s`);
  `pipeline/calibrate.py` (`_mk` duration); tests `test_tree.py` (short-clip → review, byte-identical
  short → still CERTAIN). Reprocess without re-align: `apply_thresholds_to_store` re-decides stored
  signals with the new tree. NOTE: true short `-Copy(N)` copies also drop to review (also short); the
  name-grouping doesn't yet promote them back (it respects the existing PROBABLE) — open follow-up.

### "Not a duplicate" did nothing — and Recalibrate offered to LOOSEN everything
- **Symptom:** (a) marking a file "not a duplicate" left the group unchanged; (b) the Recalibrate dialog
  proposed `θv 0.5 / θa 0.7` with `recall = 0.0` from 4 labels (all not-dup, 17 orphaned).
- **Root cause (a):** the only consumer of `feedback` was `calibrate.labeled_signals_from_feedback` —
  the button just stored a row for a future GLOBAL threshold sweep. The schema's "view overrides" did
  not exist. Worse, the label is PAIR-scoped (KEEP↔file) while a cluster is a union-find COMPONENT
  (measured: a real cluster had 5 of 6 possible edges), so cutting one edge often leaves the file in
  the group via a transitive path — hence "nothing happens".
- **Root cause (b):** `suggest_thresholds` ranks by `(fewest FP, most recall, LOWEST threshold)`. With
  no POSITIVE labels, recall is 0 for every combination, so the tie-break collapses to "pick the lowest
  threshold" → a massive loosening (§0 says recall comes from the review queue, never from loosening).
- **Resolution:** (a) `store.vetoed_pairs()` (feedback rows labelled `different`) is now honoured by
  `_rebuild_clusters`, which NEVER unions a vetoed pair — so a corrected group stays corrected even
  after a re-scan re-declares the pair CERTAIN. The UI's `_split_from_group` vetoes each selected file
  against the REST of its cluster (links AMONG the selected are kept, so a fused sub-group splits off
  intact), rebuilds with `reuse=` and refreshes, so the change is visible immediately; `✓ Confirm it is
  a duplicate` upserts the label and lifts the veto. (b) `suggest_thresholds` returns `degenerate` with
  thresholds UNCHANGED unless BOTH labels are present (`MIN_PER_CLASS`); the UI explains instead of
  offering Apply.
- **UI wording:** "✗ Not a duplicate (false positive)" (a jargon LABEL) → "✗ Not the same — remove from
  this group" (states the EFFECT), plus a bulk "✂ Not the same — split off" that reuses the existing
  tick-boxes, matching the established "tick rows → bottom button" pattern of Delete/VLC.
- **Files / refs:** `store.py` (`vetoed_pairs`); `pipeline/fullscan.py` (`_rebuild_clusters` veto);
  `pipeline/calibrate.py` (`MIN_PER_CLASS`, degenerate return); `ui/main.py` (`_split_from_group`,
  menu/button); tests in `test_fullscan.py`, `test_ui.py`, `test_calibrate.py`.
- **Open:** a cluster is a connected component, so a chain A~B~C can still present as "3 copies ·
  CERTAIN" without A≁C being checked — requiring (near-)clique-ness would attack that root cause.

### Phantom duplicate records: same file under '/' and '\' paths (Windows)
- **Symptom:** the DB shows the SAME file twice (inflated index, even a file "duplicate" of itself).
  Diagnosis: some `files.path` were like `L:/Media/Adult\Flat\x.mp4` (root with '/', rest with '\')
  while others were all-'\' — two rows for one file (measured: 78 mixed-form rows, 76 distinct files).
- **Root cause:** the mixed form is `os.path.join(root, rel)` where the root was passed with forward
  slashes — i.e. WATCHDOG's `event.src_path` (watched dir as given, '/', + relative, '\'). The scan
  stores pathlib-normalized paths (all '\'), so the watcher's event paths key DIFFERENTLY → a second
  record. Windows treats '/' and '\' as the same separator on disk, but the SQLite `path` string is a
  plain key, so the two forms are distinct rows.
- **Resolution:** `canonical_path(p) = str(Path(p))` (OS-native form) applied at the PRODUCER that was
  non-canonical — `watch._route_event` (the watchdog deque feed). `collect_videos` already yields
  pathlib-canonical paths, so both producers now agree → one file, one record. Deliberately NOT applied
  inside the store: that would rewrite synthetic '/'-paths used across the test-suite to '\' on Windows
  (e.g. `cl.keep.path == "/Ik4k"` would break), and the store is fed only by these two producers.
  One-time repair: `forget_file` each '/'-form row (removes its row+matches+clusters+npy; the '\'-twin
  remains; unique ones get re-added canonically by the next scan), then rebuild clusters.
- **Files / refs:** `store.py` `canonical_path`; `watch.py` `_route_event`; tests in `test_ui.py`.
- **Scope:** decided 2026-06-19.

### Unrelated movies fused into one CERTAIN cluster (clusters table ≠ match graph)
- **Symptom:** a cluster shows totally different content as "N copies · CERTAIN" (e.g. a 2-second 1440p
  clip pair AND a 66-minute SD movie pair under one header, two ★ KEEPs). The MATCHES are correct — it's
  not a false-positive embedding/audio match.
- **Root cause:** `cluster_id` was a per-rebuild `enumerate()` index, and cluster rebuilds are NOT
  mutually exclusive across processes, NOR atomic: `_rebuild_clusters` did `clear_clusters()` (commit)
  then per-row `save_cluster()` (each its own commit). When two rebuilds run concurrently (the watcher's
  `reconcile_removals`/`ingest_new` while a scan runs, or a one-off sweep while the live watcher is up),
  both number their components 0,1,2,…; their committed writes interleave and components that land on the
  SAME index number coexist under one `cluster_id` (PK is (cluster_id, path), so different paths under one
  id just merge). Measured on the real DB: match graph had 238 correct components but the table had 169
  clusters, 88 of them internally disconnected — and the fused ids formed a contiguous LOW band (7–94)
  with HIGH ids (95–169) clean: the exact fingerprint of two `enumerate()`s overlapping in their low
  range. This is a §0 break (KEEP/reclaim span unrelated content → user could delete a unique file).
  NOTE: the split-priority change (watcher reconcile runs during a scan) widened the concurrency window.
- **Resolution:** (1) ATOMIC rebuild — `store.replace_clusters(rows)` does DELETE+INSERT in ONE
  transaction, so a concurrent rebuild can't interleave (last rebuild wins entirely). (2) STABLE,
  content-derived ids — `_stable_cluster_id(members)` = blake2b of the component's min member (56-bit),
  so independent rebuilds AGREE on a component's id and DISTINCT components never collide. Applied to
  both `_rebuild_clusters` and exact_scan's `_build_exact_clusters`. (3) One-time repair: a clean rebuild
  from `matches` split the 88 fused back into the 238 correct clusters.
- **Files / refs:** `store.py` `replace_clusters`; `pipeline/fullscan.py` `_stable_cluster_id`,
  `_rebuild_clusters`, `_build_exact_clusters`; tests in `test_fullscan.py`.
- **Scope:** decided 2026-06-19.

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
