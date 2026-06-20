# Technical Requirements — Dupless Video

> **Status:** v1 (reverse-engineered from the codebase at `version = 0.1.1`). Describes what is
> **already built**, not a wish list. Each requirement traces to source; forks (threshold/semantics
> changes) are out of scope for this document.
> **Companion docs:** [BACKEND_SCHEMA.md](BACKEND_SCHEMA.md), [APP_FLOW.md](APP_FLOW.md),
> [HPC_PIPELINE.md](HPC_PIPELINE.md) (measured performance), [DESIGN_v2_distributed.md](DESIGN_v2_distributed.md).

## 1. Purpose & scope

Dupless Video is a **content-based duplicate / upgrade detector** for video libraries. It decides,
from the **bytes** of each file (never from filename or metadata that can lie), whether two videos are
the same content — including re-encodes, different dubs, cam rips, different editions, and ad-injected
copies — and recommends which copy to KEEP. It is built to run at **tens of thousands of files** on
consumer hardware and to extract the most from it, or to surface the right knob in the UI.

In scope: indexing, matching, clustering, KEEP ranking, a desktop review UI, a background watcher,
and a recoverable delete action.
Out of scope (by design): re-encoding the kept copy (NVENC), transcoding, cloud sync, network
distribution (a distributed design is sketched in [DESIGN_v2_distributed.md](DESIGN_v2_distributed.md)
but not built).

## 2. Non-negotiable requirements (§0 — override convenience and speed)

| ID | Requirement | Where enforced |
|----|-------------|----------------|
| **NN-1** | **Zero false positives in the strong tiers (T1/T2).** Recall is recovered from a review queue, never by loosening thresholds. | `match/tree.py` (tier guards: coverage, `theta_*`); `config/thresholds.yaml` |
| **NN-2** | **A verdict must not depend on config, node, or hardware.** Same input → same result; CPU/GPU and fp16/fp32 agree (measured: 0 verdict flips). | Deterministic decision core in `decide_tree`; `_pass2_pair` deterministic per pair; fidelity guard in `Embedder` |
| **NN-3** | **Detect, don't trust.** Language, content and structure are measured from decoded bytes, not read from metadata. | `quality/language.py` (whisper-detect, not tag); `content_hash` over sampled bytes |
| **NN-4** | **Scale-resilient: skip-and-report, never crash the batch.** Corrupt/unreadable/giant/slow files are recorded in `problems` and skipped. | `_pass1`/`_pass2` try-blocks; `store.save_problem`; `candidate_paths` empty-vec guard |
| **NN-5** | **Models and thresholds are global and fixed.** Shipped defaults stay constant for anyone who has not recalibrated. | `config.effective_config_path` (read-only bundled default; per-user override only on explicit recalibration) |

## 3. Functional requirements

### 3.1 Indexing (Pass-1)
- **FR-1** Walk one or more target paths (files and/or folders), **in place** (no copy/move),
  filtering by a fixed video-extension set (`VIDEO_EXTS`, ~24 extensions). Recursive by default;
  `--no-recursive` for top level only. (`fullscan.collect_videos`, `iter_videos`)
- **FR-2** For each file compute: ffprobe metadata, a **sampled content hash** (`xxhash` of
  head‖mid‖tail), per-frame **DINOv2 embeddings** (fp16), a coarse **global descriptor**, **window
  descriptors** (multi-vector), **scene cuts**, **color stats**, and a cheap **whole-file audio
  coverage**. (`pipeline/analyze.py`)
- **FR-3** **Incrementality:** a file is recomputed only if `mtime` (±tol), `size`, or
  `feature_version` changed, or its `.npy` embeddings file is missing (self-heal).
  (`store.has_fresh`)
- **FR-4** **Resource split (M3):** CPU/IO features (probe/hash/scenes) run in a `ProcessPool`
  (fork-safe, no CUDA); GPU decode+embed runs only on the main process (single CUDA context).
  (`extract_cpu_features` vs `extract_gpu_features`)
- **FR-5** **Adaptive frame sampling:** files above `seek_threshold_gb` are sampled by sparse PyAV
  **seek** (~constant cost on 4K/8K giants); smaller files use keyframe **demux**. Both sequences are
  resampled to a common **temporal grid** so seek↔demux pairs still align by time. (`features/frames.py`,
  `align/video.resample_to_grid`, `frame_times`)
- **FR-6** **Deferred / on-demand work:** whisper language detection and the matching audio
  fingerprint are **not** computed in Pass-1 — they are computed only for files that reach a candidate
  pair / cluster, then persisted. Most unique files never pay for them. (`_ensure_audio_fp`,
  `_ensure_lang`)

### 3.2 Matching (Pass-2)
- **FR-7** **Candidate retrieval (uncapped recall):** union of (a) top-k global FAISS, (b)
  multi-vector window FAISS, (c) duration blocking ±tol **gated** by global cosine to prune the dense
  same-length dragnet. (`match/matcher.candidate_paths`, `retrieval.CoarseIndex`)
- **FR-8** **Three independent signals per pair:** audio-fp alignment, banded video-embedding
  alignment (Sakoe-Chiba), scene-cut alignment. (`align/audio.py`, `align/video.py`, `align/scenes.py`)
- **FR-9** **Staged decision tree** producing a single verdict per pair, first tier wins:
  T0 (byte-id), edition guard, T1, T2, T3, T4, T4b, T5. (`match/tree.decide_tree`) — see §5.
- **FR-10** **Pair canonicalization (C2):** each unordered pair is evaluated and stored once.
  (`canonical_pair`)
- **FR-11** **Parallel Pass-2 when it pays:** ≥8 files and >1 core → align pairs across a
  `ProcessPool` of read-only store handles; else the sequential matcher with an LRU embedding cache.
  Deterministic per pair (NN-2). (`match_pairs_parallel`, `_pass2_sequential`)

### 3.3 Clustering & KEEP
- **FR-12** Clusters are a **derived view** of the global `matches` graph via union-find over
  duplicate verdicts, rebuilt atomically with **stable content-derived cluster ids** (no cross-rebuild
  fusion of unrelated components). (`_rebuild_clusters`, `_stable_cluster_id`,
  `store.replace_clusters`)
- **FR-13** **KEEP ranking by quality (not identity):** wanted language ≫ resolution ≫ no-ads ≫ lower
  cam-score ≫ higher bitrate, minus color clipping; tiebreak = shortest path. An **audio guard** sends
  the cluster to manual review only when copies *differ* in coverage. (`rank_cluster`, `_score_member`)
- **FR-14** **Name-copy grouping (NAME_COPY)** for `movie (N).ext` siblings in the same folder,
  **with a content veto** (never grouped if content says DIFFERENT or can't be verified). (`_apply_name_grouping`,
  `name_pair_content_differs`)
- **FR-15** **Ad / color hints** steer KEEP and raise a UI flag but **never change the duplicate
  verdict**. (`_cluster_has_ads`, `_color_adjusted_keep`)

### 3.4 Modes & depth
- **FR-16** **Depth levels** (incremental — deeper reuses shallower): `fast` (byte-identical hash
  only, ~0.1s/file), `standard` (visual+audio detection, coverage on-demand), `deep` (standard +
  whole-file audio coverage for every file). (`cli.scan --depth`, `full_scan(eager_coverage=...)`)
- **FR-17** **Pass-1-only mode** (`--no-match`) to (re)index a large library cheaply before running
  Pass-2. (`full_scan(match=False)`)
- **FR-18** **Exact-only / LITE records:** byte-identical sweep that writes metadata+hash without
  embeddings; a later full scan re-indexes them. The store and all readers tolerate a mixed
  full+LITE DB. (`exact_scan`, `save_meta`, `all_global_vecs` skip-on-null)

### 3.5 Watcher
- **FR-19** **Background watcher** keeps the DB current (add/change/remove) so the user opens the app
  only to review. Two-lane scheduler: a **fast lane** (watchdog create/modify events, newest-first)
  and a **slow lane** (rglob backlog, oldest-first), chunked so a just-dropped file is indexed within
  ~one chunk even behind a large backlog. (`watch.py`, `_IngestScheduler`)
- **FR-20** **Event-driven deletions** in O(1) via a watchdog `deleted` queue; an O(library) sweep is
  demoted to a startup catch-up + periodic backstop. Optional dependency: without `watchdog`, falls
  back to backoff polling. (`reconcile_removals`, `start_fs_events`)
- **FR-21** **Scan-priority lock:** a user scan takes a cross-process PID lock; the watcher reconciles
  deletions always but **defers heavy indexing** while a scan holds the lock (avoids HDD thrashing +
  SQLite write starvation). Self-heals a stale lock. (`runtime.scan_priority_lock`, `scan_in_progress`)
- **FR-22** **Idle GPU release:** an idle watcher frees cached GPU VRAM so a 24/7 watcher stops
  squatting the GPU. (`_run_cycle` → `embedder.free_cache()`)

### 3.6 UI & actions
- **FR-23** Desktop UI (PySide6): sortable/filterable duplicate tree, open in VLC, preview, **send to
  Recycle Bin** with a confirmation that lists everything, and the "What the AI sees" live-view panel.
  Reads the existing DB; **does not re-scan or retrain.** (`ui/main.py`, `ui/*_panel.py`)
- **FR-24** **Recoverable delete only:** deletions go to Trash (`send2trash`) and are audited in
  `deletions`. (`ui/actions.py`, `store.record_deletion`)
- **FR-25** **Feedback → recalibration:** user `same`/`different` labels feed threshold
  recalibration and view overrides; they **do not retrain the network**. (`store.save_feedback`,
  `pipeline/calibrate.py`)
- **FR-26** **Repair queue:** files that failed with a slow-seek timeout are categorized `reindex`
  and can be losslessly remuxed (`remux -c copy`); truly corrupt files are `corrupt`.
  (`cli.repair-indexes`, `repair.py`, `classify_problem`)

### 3.7 Calibration
- **FR-27** `calibrate` evaluates thresholds against a hand-labeled set and suggests `theta_v`/`theta_a`
  with **zero FP in T1/T2**; recalibration writes a per-user override, never the shipped default.
  (`cli.calibrate`, `pipeline/calibrate.suggest_thresholds`)

## 4. Non-functional requirements

| ID | Requirement | Evidence / mechanism |
|----|-------------|----------------------|
| **NFR-1 Performance** | Pipeline is **I/O-bound on disk**, not compute-bound. Perf levers ship **off by default until a benchmark shows they pay** here; storage-aware auto-tune (HDD ~1–2 workers, SSD parallel decode). | [HPC_PIPELINE.md](HPC_PIPELINE.md); `tuning.autotune`; `--workers 0`/`--decode-workers -1` AUTO |
| **NFR-2 Determinism** | Decision core free of ambient time/RNG/locale/float drift; verdict invariant across CPU/GPU and fp16/fp32 (measured 0 flips). | `decide_tree` pure; FP8 behind a fidelity guard reverting <0.99 cosine |
| **NFR-3 Resilience** | Any one bad file is skipped-and-reported; unbounded work stays incremental and self-heals stale/orphaned data. | `problems` table; `prune_missing_problems`; orphan `.npy` guards |
| **NFR-4 Progress/UX** | Every long operation shows progress + ETA + live counts (never a silent freeze). | `tqdm` bars; `REPAIR_PROGRESS`/`on_detect` lines parsed by the UI |
| **NFR-5 Concurrency safety** | Scan, watcher, and UI hit the same SQLite DB; WAL + busy_timeout + priority lock + atomic cluster rebuild prevent starvation and cross-component fusion. | `PRAGMA journal_mode=WAL`, `busy_timeout=30000`; `replace_clusters` single transaction |
| **NFR-6 Portability** | Per-user data dir per OS; OS-specific seam isolated in one module; offline model load (no first-run network). | `runtime.app_data_dir`, `resolve_binary`, `configure_offline_model` |
| **NFR-7 Privacy / i18n** | All shipped strings in English; UTF-8 forced on the Windows console for non-cp1252 filenames; no personal data in code/fixtures. | `cli.py` stream reconfigure; CLAUDE.md charter |

## 5. Decision tiers (current calibration)

First tier that fires fixes verdict + confidence (`decide_tree`). Defaults from `config/thresholds.yaml`.

| Tier | Verdict | Condition (summary) | Conf | Action class |
|------|---------|---------------------|------|--------------|
| **T0** | CERTAIN | `content_hash` and `size` identical (sampled hash — verify byte-exact before delete) | 1.00 | duplicate |
| **edition guard** | DIFFERENT_EDITION | strong video + contiguous superset (director's cut, +`superset_min_extra_ratio` runtime) | ≤0.90 | related, not dup |
| **T1** | CERTAIN | `audio ≥ θ_a (0.80)` **and** `video ≥ θ_v (0.75)` **and** `coverage ≥ 0.70` | 0.99 | duplicate |
| **T2** | VERY_HIGH | `video ≥ θ_v_high (0.85)` + coverage, **and** `audio < θ_a_low (0.30)` → different dub | 0.95 | duplicate |
| **T3** | HIGH | `video ≥ θ_v` + coverage **and** `scenes ≥ θ_s (0.70)` | 0.88 | duplicate |
| **T4** | PROBABLE | `scenes ≥ θ_s_high (0.80)` only, `video < θ_v`, **and** cut-density ≥ `min_cut_density (0.04/s)` on both | 0.65 | review queue |
| **T4b** | PROBABLE | one strong modality uncorroborated (video+cov, or audio) | 0.55 | review queue |
| **T5** | DIFFERENT | no alignment | 0.0 | discarded (not persisted) |

NAME_COPY (conf 0.75) is **added post-hoc** by name grouping with a content veto — it is not emitted
by `decide_tree` and does not affect the T1/T2 zero-FP guarantee.

## 6. External dependencies & environment

- **Python ≥ 3.11.** Core deps: `typer`, `numpy`, `xxhash`, `pyyaml`, `tqdm`.
- **ML/GPU stack (per hardware, see [requirements-gpu.txt](../requirements-gpu.txt)):** `torch` +
  `torchvision` (CUDA build), `faiss-gpu`/`faiss-cpu`, `faster-whisper`, `av` (PyAV, NVDEC), DINOv2
  via `torch.hub`. **CPU fallback supported** (fidelity verified, §NN-2).
- **UI extra (`[ui]`):** `PySide6`, `send2trash`, `opencv-python-headless` (live-view JPEG encode).
- **Watch extra (`[watch]`):** `watchdog` (optional; polling fallback without it).
- **External binaries:** `ffmpeg`, `ffprobe`, `fpcalc` (Chromaprint) — resolved offline-first from the
  bundle, then env, then venv, then PATH (`runtime.resolve_binary`).
- **Reference hardware (measured):** Windows 11, Blackwell GPU (sm_120), torch 2.11+cu128, library on a
  ~44 TB multi-disk Storage Space (spinning HDD over SATA). The optimum is storage-dependent — see
  [HPC_PIPELINE.md](HPC_PIPELINE.md).

## 7. Tunable levers (each maps to a measured outcome — §3 charter)

| Lever | Default | When to change |
|-------|---------|----------------|
| `--workers` | AUTO (~2 HDD, higher SSD) | manual override per storage |
| `--decode-workers` | AUTO (1 on HDD) | >1 only on SSD/NVMe (thrashes HDD) |
| `--depth fast/standard/deep` | standard | fast=hash only; deep=audio quality for all |
| `--max-height` | off | exclude 4K/8K from a run |
| `--independent-scenes` | off (scenes from embeddings) | pixel-based scenes (slower, better for cam) |
| `--fp8-embed` | off | compute-bound Linux/NVMe only; reverts <0.99 cosine |
| `seek_threshold_gb` / `seek_n` | 6 GB / 200 | demux↔seek crossover per storage |
| `duration_tolerance` / `duration_block_cos_gate` | ±3% / 0.7 | candidate-set size vs recall |
| watcher `interval` / `stable_s` / `ingest_chunk` | 60s / 15s / 4 | responsiveness vs disk load |

## 9. Algorithmic specification

This section pins the exact algorithms, data shapes, dtypes and parameters behind the functional
requirements. `D = 768` (DINOv2 ViT-B/14), `fps = 2.0`, item rate `8.0/s` (Chromaprint).

### 9.1 Signal extraction

| Signal | Algorithm | Output shape / dtype | Key params |
|--------|-----------|----------------------|------------|
| **content_hash** (`features/hashing.py`) | `xxh3_64` over `head(8MB) ‖ mid(8MB @ size/2) ‖ tail(8MB)`; mid+tail only if `size > 16MB` | 16-hex string (64-bit) | `CHUNK = 8 MiB` |
| **keyframe decode** (`features/frames.py`) | ffmpeg `-skip_frame nokey -fps_mode passthrough` (I-frames only) scaled to 224², NVDEC `-hwaccel cuda` → software fallback; or PyAV sparse **seek** to N timestamps for giants | frames `[N,3,224,224]` fp32 (ImageNet-norm) + `times[N]` fp32 (real pts) + `ColorStats` | seek if `size > 6 GB` **or** `height ≥ 4320` (8K); `seek_n = 200`; `decode_timeout_s = 240` |
| **embeddings** (`features/embeddings.py`) | DINOv2 forward (CLS token), `autocast fp16`, batched, `F.normalize` per frame | `[N,768]` fp16 (L2-unit) | `batch = 512` |
| **global_vec** | mean-pool over frames + L2 | `[768]` fp32 | — |
| **window_vecs** | `np.array_split` into `min(K,N)` contiguous windows, mean-pool + L2 each | `[K,768]` fp32 | `K = n_window_vecs = 12` |
| **audio_fp** (`features/audio_fp.py`) | Chromaprint raw items via `fpcalc -raw -length L`; fallback decodes PCM mono 11025 Hz through system ffmpeg → fpcalc | `[M]` **uint32** (`M ≈ 8.0·seconds`) | `L = audio_fp_max_for(duration)` (whole file ≤1h, else 600s) |
| **audio_coverage** (`scan_audio_coverage`) | 40 sparse probes × 0.5s windows, ffmpeg `volumedetect`, fraction with `mean_volume > −60 dB` | float [0..1] | `n_points=40`, `win_s=0.5`, `silence_db=−60` |
| **scene_cuts (EMB)** (`features/scenes.py`) | cosine drop `< 0.6` between consecutive frame embeddings → cut at that frame's real pts | `[K]` fp32 timestamps (sorted) | `sim_threshold = 0.6` |
| **scene_cuts (PIX)** | ffmpeg `select='gt(scene,0.3)'` on a 180px downscale, `showinfo` pts | `[K]` fp32 timestamps | `threshold=0.3`, `height=180` |
| **color** (`quality/color.py`) | from decoded RGB keyframes: `clip` = frac luma `<5` or `>250` (Rec.601); `cast`/`saturation`/`contrast` describe grade | `ColorStats[4]` fp32 | `GRADE_DIVERGENCE=0.15`, `CLIP_DOWNGRADE_MARGIN=0.05` |

### 9.2 Pairwise alignment (the three independent signals)

**Video — banded Smith-Waterman** (`align/video.banded_align`)
- `Sim = emb_a @ emb_bᵀ` (cosine; inputs are L2-unit). On torch CUDA tensors the matmul runs on
  device then → numpy for the DP; Pass-2 workers stay pure-numpy.
- Local alignment with a **Sakoe-Chiba band** of radius `r = max_offset_s · fps = 300·2 = 600`
  frames; width `W = 2r+1`. Recurrence (floor 0, linear gaps), reward `s = sim[i,j] − match_threshold`:
  `H₀[d] = max(0, H[i−1,d]+s, H[i−1,d+1]−gp)`, then `H[d] = max(H₀[d], H[d−1]−gp)` (left gap resolved
  vectorized via cumulative-max). `gap_penalty = 0.3`, `match_threshold = 0.5`.
- **Complexity `O(Na · W)`** (linear in frames, not `O(Na·Nb)`) — the band makes a 14k×14k matrix
  tractable. Loop over rows only; band width vectorized. Traceback is `O(path length)`.
- Outputs: `score = mean(sim[path])`, `offset = (ib₀−ia₀)/fps`, `coverage = len(path)/min(Na,Nb)`,
  plus **superset** (`_detect_superset`: ≥90% dense containment → edition) and **interleaved-ad**
  (`_interleaved_extra`: unmatched contiguous runs ≥ `min_ad_run_s·fps` inside the span).
- **Mixed-sampling fix:** when both have `frame_times`, both sequences are `resample_to_grid`'d
  (nearest-neighbor in time, step `grid_step_s=2.0`) before alignment → seek↔demux copies align by
  time, not frame index. Band recomputed as `max(1, max_offset_s/step)`.

**Audio — FFT cross-correlation over bit-planes** (`align/audio.align_audio`)
- For offset `off`, `hamming_sum(off) = Σ popcount(a[i] ⊕ b[i+off])`. Using `x⊕y = x+y−2·x·y` per bit:
  `= (Σ pc a) + (Σ pc b) − 2·Σ_k corr_k(off)`, where linear terms come from popcount prefix-sums and
  `Σ_k corr_k` over the **32 bit-planes** comes from one batched `rfft/irfft`.
- **Complexity `O(N log N)`** vs the old `O(offsets·N)` per-offset scan (the measured 92% Pass-2
  bottleneck before the rewrite). **Bit-exact** to brute force (`np.rint` recovers integer
  correlations) → verdict invariance (§NN-2), covered by an equivalence test.
- `score = 1 − bits/(32·length)`; valid only if `length ≥ min_overlap_s·8 = 480` items; offset search
  bounded to `±max_offset_s·8`. Different-language audio decorrelates to ~0.5 (random bit agreement).

**Scenes — DTW over inter-cut intervals** (`align/scenes.align_scenes`)
- Aligns `np.diff(cuts)` (intervals, trim-invariant) — not absolute timestamps. Three constraints make
  the score *discriminate*: Sakoe-Chiba band `= max(25, ⌈0.2·max(n,m)⌉)`, `gap_penalty = 0.2`, and a
  length cutoff (`|n−m| > band` → score 0). Local cost `= min(1, |aᵢ−bⱼ|/(aᵢ+bⱼ+ε))`.
- `score = max(0, 1 − D[n,m]/max(n,m))`. **Complexity `O(n·band)`**.

### 9.3 Candidate retrieval (`match/retrieval.CoarseIndex`)
- **Dual exact FAISS** `IndexFlatIP` (inner product over L2-unit vecs = cosine): a global mean-pool
  index and a window-vec index. At ~1–2k films (×K=12 ≈ 24k window-vecs) both fit in RAM.
- Candidate set = `query_global(k=25) ∪ query_windows(window_faiss_k=10 owners) ∪ gated_duration`.
- **Duration block:** SQL `BETWEEN duration·(1∓0.03)` on `idx_files_duration`, then `gate_by_global`
  keeps only neighbors with global cosine `≥ 0.7` (vectorized `gvecs[rows] @ q`). Measured effect:
  ~17M → ~626k pairs on a dense library (Pass-2 align ~126h → ~5h), labeled dups retained (≥0.962).
- §2 boundary guards: a dim-0 / mismatched query returns empty instead of tripping faiss'
  `assert d == self.d`.

### 9.4 Embedding residency (`match/cache.EmbeddingCache`)
- `OrderedDict` LRU of resident fp16 tensors (CUDA if available, else CPU). `max_items = 1500` in the
  sequential Pass-2; loaded once per film from its `.npy`. A missing `.npy` raises `KeyError` →
  treated as "no video signal" (audio/scenes still decide), never crashes.

### 9.5 Storage-aware auto-tune (`tuning.autotune`)
Two measured signals (never the OS label, which lies for Storage Spaces / iSCSI / SMB):
1. **random-read latency** — median of 24 cold 1 MiB reads at random offsets (`buffering=0`).
2. **concurrency scaling** — aggregate throughput of 4 concurrent cold 16 MiB sequential reads ÷
   serial; catches **tiered HDD** (SSD cache serves the tiny seeks → *looks* SSD, but large reads
   thrash). Decision: `lat<6 ∧ scaling<0.6` → tiered HDD `(2,1)`; `lat≥6` → HDD `(2,1)`; `lat≤1.5` →
   SSD `(min(cpu,12), 4)`; else moderate `(min(cpu,6), 2)`. Deterministic RNG seed → reproducible probe.

## 10. Complexity & data-shape reference

| Quantity | Value / formula | Note |
|----------|-----------------|------|
| Embedding dim `D` | 768 | DINOv2 ViT-B/14 |
| Per-frame embedding | fp16, 1536 B/frame on disk | ~22 MB/film (README) |
| Window vecs/film `K` | 12 × 768 fp32 | multi-vector retrieval |
| Audio-fp items `M` | `≈ 8.0 · min(duration, cap)` uint32 | cap = whole file ≤1h else 600s |
| Video align | `O(Na · W)`, `W = 1201` | banded SW (band 600) |
| Audio align | `O(N log N)` | FFT over 32 bit-planes |
| Scene align | `O(n · band)`, `band ≥ 25` | DTW over intervals |
| Pass-1 | `O(files)`, I/O-bound (disk demux dominant) | incremental; skips fresh |
| Pass-2 candidate gen | `O(files · k)` FAISS + duration block | gated to bound the dense-library blow-up |
| Pass-2 align | `O(pairs · per-pair align)` | compute-bound (~89% banded video DP); parallel across cores |

## 11. Out of scope / known limitations
- T0 uses a **sampled** hash (head‖mid‖tail); the UI advises verifying byte-exact before deleting.
- Legacy records created before a column existed load with safe defaults (e.g. `audio_coverage` NULL→1.0)
  and need a re-scan to populate the new signal.
- The distributed/multi-node design ([DESIGN_v2_distributed.md](DESIGN_v2_distributed.md)) is a design,
  not an implementation.
- NVENC re-encoding of the kept copy is intentionally outside the detector.
</content>
</invoke>
