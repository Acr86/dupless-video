# Backend Schema — Dupless Video

> **Status:** v1 (reverse-engineered from `src/dupdetect/store/schema.sql` + the idempotent migrations
> in `store.py` and the binary-blob encodings in `models.py`/`save()`). Authoritative source is the
> code; this document explains the structure, types, and invariants.
> **Companion docs:** [TECHNICAL_REQUIREMENTS.md](TECHNICAL_REQUIREMENTS.md), [APP_FLOW.md](APP_FLOW.md).

## 1. Storage architecture

Two coordinated stores, both per-user under the OS data dir (`runtime.app_data_dir()`):

| Store | Location | Holds |
|-------|----------|-------|
| **SQLite DB** | `<data_dir>/dupdetect.sqlite` | metadata, small vectors (BLOB), matches, clusters, problems, feedback, deletions |
| **Embeddings dir** | `<data_dir>/embeddings/<sha1(path)[:16]>.npy` | per-frame DINOv2 embeddings `[N, D]` fp16, referenced by `files.emb_path` |

- **Windows:** `%LOCALAPPDATA%\Dupless Video\` · **macOS:** `~/Library/Application Support/Dupless Video\` ·
  **Linux:** `$XDG_DATA_HOME/Dupless Video`. Override with `$DUPDETECT_DATA_DIR`.
- **Why split:** per-frame embeddings are large (~22 MB/film fp16); keeping them out of SQLite keeps the
  DB small and lets the matcher `mmap`/lazy-load only the films actually compared.

### 1.1 PRAGMAs & concurrency
- `journal_mode = WAL` — readers (UI, watcher) don't block the writer (scan) and vice-versa. Set on the
  rw handle; falls back silently if the FS rejects WAL (e.g. some network shares).
- `busy_timeout = 30000` — retry on a lock instead of erroring.
- Read-only worker handles open with `init_schema=False` (no schema writes → no contention).
- Cross-process **scan-priority lock** (`scan.lock` PID file) — see [APP_FLOW.md](APP_FLOW.md) §watcher.

## 2. Entity-relationship overview

```
                    ┌──────────────┐
                    │    files     │  1 row per indexed video (FULL or LITE)
                    │  path (UQ)   │  PK id; path UNIQUE = the natural key
                    └──────┬───────┘
        emb_path ───────────┘ (→ embeddings/<id>.npy, fp16 [N,D])
                           │
     ┌─────────────────────┼──────────────────────┬───────────────────┐
     │                     │                       │                   │
┌────▼─────┐        ┌──────▼──────┐         ┌──────▼─────┐      ┌──────▼──────┐
│ matches  │        │  clusters   │         │  problems  │      │  feedback   │
│ a,b (UQ) │        │ cid+path PK │         │  path PK   │      │ a,b (UQ)    │
│ verdict  │        │ is_keep     │         │ category   │      │ label       │
└──────────┘        └─────────────┘         └────────────┘      └─────────────┘
   pair graph     derived view of the     skip-and-report       user corrections
 (canonical a≤b)  matches duplicate graph    queue              (recalibration)

                    ┌──────────────┐
                    │  deletions   │  audit log of UI trash actions
                    └──────────────┘
```

Relationships are **by path string**, not by FK (paths are the natural key; SQLite FKs are not
declared). `clusters` and `matches` are kept in sync deliberately (see §4, drift).

## 3. Tables

### 3.1 `files` — one row per indexed video
The fingerprint of a file. A **FULL** record has embeddings; a **LITE** record (exact-only mode) has
metadata+hash only (`emb_path`, `global_vec`, etc. NULL).

| Column | Type | Null | Meaning |
|--------|------|------|---------|
| `id` | INTEGER PK AUTOINC | no | surrogate key |
| `path` | TEXT **UNIQUE** | no | natural key; OS-canonical (`canonical_path`) to avoid `/`-vs-`\` phantoms |
| `mtime` | REAL | no | modification time (compared with `mtime_tol`, default 2.0s for SMB) |
| `size` | INTEGER | no | byte size (part of T0 identity + freshness) |
| `content_hash` | TEXT | no | `xxhash(head‖mid‖tail)` — **sampled**, not full-file (T0) |
| `feature_version` | TEXT | no | C4 cache key: model+fps+audio-policy+scene-mode+algo versions (invalidates on change) |
| `duration_s` | REAL | yes | from ffprobe |
| `width`, `height` | INTEGER | yes | resolution (KEEP ranking + height filter) |
| `vcodec` | TEXT | yes | video codec |
| `bitrate_kbps` | INTEGER | yes | KEEP tiebreak |
| `audio_tracks` | TEXT (JSON) | yes | `[{index,lang_tag,codec,channels}]` — declared tags (often wrong) |
| `global_vec` | BLOB | yes | float32 `[D]` mean-pooled L2 descriptor (coarse FAISS) |
| `window_vecs` | BLOB | yes | float32 `[K*D]` per-temporal-window descriptors (multi-vector retrieval) |
| `window_k` | INTEGER | yes | K = number of window descriptors |
| `emb_dim` | INTEGER | yes | D, to reshape `window_vecs`/embeddings |
| `emb_path` | TEXT | yes | filename of the `.npy` (relative to embeddings dir); NULL ⇒ LITE |
| `n_frames` | INTEGER | yes | N rows in the embeddings array |
| `audio_fp` | BLOB | yes | **uint32** `[M]` Chromaprint raw fp — computed **on-demand** in Pass-2 |
| `scene_cuts` | BLOB | yes | float32 `[K]` cut timestamps (s) |
| `frame_times` | BLOB | yes | float32 `[N]` real timestamp per frame (align by **time**, not index) |
| `lang_detected` | TEXT | yes | actual language (whisper-detect); **deferred** to cluster-rank time |
| `cam_score` | REAL | yes | 0..1 suspected-camrip heuristic |
| `audio_coverage` | REAL | yes | [0..1] whole-file coverage; NULL = not computed (loads as 1.0); <1 = muted/truncated |
| `color_stats` | BLOB | yes | float32 `[4]` = clip, cast, saturation, contrast; NULL → neutral |
| `indexed_at` | REAL | no | last write time |

**Indexes:** `idx_files_hash(content_hash)`, `idx_files_size(size)`, `idx_files_duration(duration_s)`
(duration blocking).

**BLOB encodings (round-trip in `store.save` / `_row_to_record`):**
- `global_vec` → `np.float32.tobytes()`; reload `np.frombuffer(..., float32)`.
- `window_vecs` → float32 `[K,D]` flattened; reshape with `window_k`×`emb_dim`.
- `emb_path` file → `np.save(... astype(float16))`; per-frame `[N,D]` fp16 on disk.
- `audio_fp` → **uint32** (`astype(np.uint32).tobytes()`) — NOT float32.
- `scene_cuts`, `frame_times`, `color_stats` → float32.

**Migrations (idempotent, applied at open in `_init_schema`):** `frame_times`, `problems.category`,
`problems.repair_note`, `audio_coverage`, `color_stats` are `ALTER TABLE … ADD COLUMN` wrapped in
try/except (so old DBs gain them; `CREATE TABLE IF NOT EXISTS` does not add columns). Stale `problems`
rows are reclassified from their error message on open (`_reclassify_stale_problems`). One-shot
data migrations are keyed by **`PRAGMA user_version`** (v1: the coverage-probe v2 rollout NULLs only
the rows the v1 probe flagged, so they lazily re-measure — a version bump inside `feature_version`
would have re-decoded the whole library for an audio-only change).

### 3.2 `matches` — pairwise verdicts (the duplicate graph)
| Column | Type | Null | Meaning |
|--------|------|------|---------|
| `id` | INTEGER PK AUTOINC | no | |
| `a_path`, `b_path` | TEXT | no | **canonicalized so `a ≤ b`** (C2) — one row per unordered pair |
| `verdict` | TEXT | no | `Verdict` enum value (CERTAIN/VERY_HIGH/HIGH/NAME_COPY/PROBABLE/DIFFERENT_EDITION) |
| `confidence` | REAL | no | tier confidence |
| `reason` | TEXT | yes | human-readable tier explanation |
| `ad_offset_s` | REAL | yes | C3 align offset (property of the **pair**); sign preserved through canonicalization |
| `audio_json`, `video_json`, `scenes_json` | TEXT | yes | serialized `AlignResult` (evidence; mid-roll-ad fields read from `video_json`) |
| `created_at` | REAL | no | |

**Constraint:** `UNIQUE(a_path, b_path)`; upsert on conflict.
**Invariants:** DIFFERENT (T5) is **never persisted** (filtered out). Re-indexing a file deletes its
matches (`save()` → `DELETE FROM matches WHERE a_path=? OR b_path=?`) so stale pairs can't ghost.

### 3.3 `clusters` — materialized duplicate groups (derived view)
| Column | Type | Null | Meaning |
|--------|------|------|---------|
| `cluster_id` | INTEGER | no | **stable, content-derived** 56-bit id from the lexicographically smallest member (`blake2b`) |
| `path` | TEXT | no | a member of the cluster |
| `is_keep` | INTEGER | yes (def 0) | 1 = recommended KEEP (★) |
| `rank_reason` | TEXT | yes | per-member evidence (lang/res/bitrate/cam/ads/audio) |

**PK:** `(cluster_id, path)`.
**Invariants / lifecycle:**
- Rebuilt **entirely** from `matches` on each scan via union-find over `DUPLICATE_VERDICTS`
  (`_rebuild_clusters`) — clusters are a derived view, not accumulated state.
- Rebuild is **atomic** (`replace_clusters`: single `DELETE`+`executemany` transaction) so a concurrent
  rebuild (watcher during a scan) can't interleave rows or fuse unrelated components.
- A cluster of `< MIN_CLUSTER_MEMBERS (2)` is "resolved" and pruned (`prune_singleton_clusters`).
- Stable ids mean two independent rebuilds agree, and the UI's KEEP selection survives a refresh.

### 3.4 `problems` — skip-and-report queue (§2)
| Column | Type | Null | Meaning |
|--------|------|------|---------|
| `path` | TEXT **PK** | no | the failing file |
| `error` | TEXT | yes | raw ffprobe/decode error |
| `category` | TEXT (def `'corrupt'`) | no | `'corrupt'` (data lost → delete/external tool) vs `'reindex'` (valid, slow seek → `remux -c copy` fixes) |
| `repair_note` | TEXT | yes | result of the last remux attempt; shown instead of the scan error when present |
| `last_seen` | REAL | no | |

**Lifecycle:** one row per path; deleted when the file analyzes OK on a later run
(`save`/`save_meta` → `DELETE FROM problems`). `prune_missing_problems` forgets rows whose file is
gone **but whose parent dir is reachable** (an unmounted volume is left intact — §2).

### 3.5 `feedback` — user corrections (recalibration input)
| Column | Type | Null | Meaning |
|--------|------|------|---------|
| `a_path`, `b_path` | TEXT | no | canonicalized `a ≤ b` |
| `label` | TEXT | no | `'same'` (is a duplicate) \| `'different'` (false positive) |
| `note` | TEXT | yes | optional user note |
| `created_at` | REAL | no | |

**Constraint:** `UNIQUE(a_path, b_path)`. **Does NOT retrain the network** — feeds
`calibrate.suggest_thresholds` and UI view overrides only.

### 3.6 `deletions` — audit log
| Column | Type | Null | Meaning |
|--------|------|------|---------|
| `path` | TEXT | no | deleted file |
| `dest` | TEXT | no | destination; always `'trash'` (recoverable) in current builds |
| `size` | INTEGER | yes | bytes (reclaimed-space reporting) |
| `deleted_at` | REAL | no | |

Append-only audit for traceability / undo-from-Recycle-Bin.

### 3.7 `quality_overrides` — per-file "Mark audio as OK"
| Column | Type | Null | Meaning |
|--------|------|------|---------|
| `path` | TEXT | no | file the user verified (PK with `kind`) |
| `kind` | TEXT | no | `'audio'` today; extensible (color, cam, …) |
| `mtime`, `size` | REAL / INTEGER | no | file identity at override time → the override **auto-expires** when the file changes |
| `created_at` | REAL | no | |

User correction for a FALSE quality warning (e.g. genuinely quiet content the probe reads as "no
audio"). Separate table on purpose: a `files` column would be clobbered by `save()`'s upsert on
re-analysis. Consumed at read time only (`ensure_audio_coverage` → 1.0, `audio_warnings` filter,
`load_clusters` coalesce) — the MEASURED `files.audio_coverage` is never rewritten, so "restore"
is just dropping the row. Steers warnings/KEEP only; never a verdict (§0).

## 4. Cross-table invariants & known coupling
- **C2 canonicalization:** `matches` and `feedback` both store `a ≤ b` so a pair is one row.
- **matches ⇄ clusters sync (drift):** byte-identical (T0) clusters historically built without writing
  `matches`, leaving the verdict empty ("Review only" instead of CERTAIN). Fixed: `exact_scan` stamps
  the T0 verdict for each pair (`_build_exact_clusters`); `ui.data.drift_report` detects residual drift.
- **Self-heal:** `forget_file(path)` removes the `files` row, its matches, its cluster membership, and
  deletes the `.npy` — preventing ghosts after a deletion.
- **Mixed FULL+LITE DB:** every reader skips records without embeddings (`all_global_vecs`,
  `all_window_vecs`) so a LITE record never crashes FAISS or Pass-2 (§2).
- **Concurrent-deletion guard (§0):** Pass-2 never (re)persists a match for a path that left the disk
  mid-scan (`_both_on_disk`), so the user's deletion wins the race against the in-memory index snapshot.

## 5. BLOB binary layouts (exact)

All numeric BLOBs are little-endian `np.ndarray.tobytes()`; reload with `np.frombuffer(buf, dtype)`
and reshape where noted. `D = emb_dim` (768 for the default DINOv2 ViT-B/14).

| Column | numpy dtype | Logical shape | Bytes | Reshape key |
|--------|-------------|---------------|-------|-------------|
| `global_vec` | float32 | `[D]` | `4·D` (3072) | — |
| `window_vecs` | float32 | `[K, D]` flattened | `4·K·D` (~36.9 KB at K=12) | `window_k` × `emb_dim` |
| `audio_fp` | **uint32** | `[M]` | `4·M` (`M ≈ 8·sec`) | — (uint32: raw Chromaprint items can exceed 2³¹; float would truncate bits — **C1**) |
| `scene_cuts` | float32 | `[K_cuts]` | `4·K_cuts` | — |
| `frame_times` | float32 | `[N]` | `4·N` | parallels `n_frames` |
| `color_stats` | float32 | `[4]` = clip,cast,sat,contrast | 16 | `ColorStats.from_list` |
| `emb_path` → `.npy` | **float16** | `[N, D]` | `2·N·D` (~22 MB/film) | stored out-of-DB, mmap-loadable |
| `audio_tracks` | TEXT (JSON) | `[{index,lang_tag,codec,channels}]` | var | `json.loads` |

**Why fp16 embeddings live outside SQLite:** at `2·N·D` bytes/film the per-frame array dwarfs every
other column; keeping it as a sidecar `.npy` keeps the DB small (metadata + small vectors) and lets the
matcher load only the films it actually compares. Filename = `sha1(path)[:16].npy`, stored **relative**
to the embeddings dir (portable); absolute legacy paths still resolve.

**Storage sizing (rule of thumb):** per FULL film ≈ `global_vec (3 KB) + window_vecs (37 KB) +
audio_fp (4·8·sec B ≈ 0.6 MB for a 2h film) + .npy (22 MB)` → the `.npy` dominates; ~10k films ≈
**~220 GB** of embeddings + a comparatively small SQLite DB. A LITE record is a few hundred bytes.

## 6. Access paths (how each component queries)

| Caller | Query / method | Index used |
|--------|----------------|-----------|
| Pass-1 freshness | `SELECT mtime,size,feature_version,emb_path FROM files WHERE path=?` | `path` UNIQUE |
| Exact dedup | `find_by_hash` / `content_hash_if_unchanged` | `idx_files_hash` |
| Duration block | `find_by_duration` → `WHERE duration_s BETWEEN ? AND ?` | `idx_files_duration` |
| Coarse index build | `all_global_vecs`, `all_window_vecs` (skip NULL vecs) | full scan of `files` |
| Pass-2 worker load | `SELECT * FROM files WHERE path=?` (read-only handle, `init_schema=False`) | `path` UNIQUE |
| Cluster rebuild | `all_matches` (full) → union-find; `replace_clusters` (atomic) | full scan of `matches` |
| KEEP ad detection | `SELECT a_path,b_path,ad_offset_s,video_json FROM matches WHERE a_path=? OR b_path=?` | `UNIQUE(a_path,…)` + `idx_matches_b` |
| UI list | `clusters` ⋈ `files` (membership + KEEP + metadata) | PK `(cluster_id,path)` |
| Quality-warnings tab | `audio_warnings` → `WHERE audio_coverage IS NOT NULL AND < threshold`, minus valid `quality_overrides` | seq |

Hot-path mutations and their side-effects:
- `save(rec)` — upsert `files`, write `.npy`, `DELETE FROM problems WHERE path=?`,
  **`DELETE FROM matches WHERE a_path=? OR b_path=?`** (re-index invalidates stale pairs).
- `save_match(...)` — canonicalize `a≤b`, negate `ad_offset_s` if the order flips (**C3** sign
  preservation), upsert on `UNIQUE(a_path,b_path)`.
- `forget_file(path)` — delete from `files` + `matches` + `clusters` + `quality_overrides`, unlink
  the `.npy` (no ghosts).
- `replace_clusters(rows)` — single `BEGIN; DELETE FROM clusters; executemany INSERT; COMMIT` (atomic,
  rolls back on error — no interleave with a concurrent rebuild).

## 7. `feature_version` — the cache-invalidation contract
`feature_version` (column on `files`) is the combined version of **all** signals:
`{embedder.feature_version}|afp{V}[G{gate}C{cap}]|cov{V}|clr{V}|scn{PIX|EMB}{V}`
(`analyze.feature_version`). Changing any model, fps, audio-fp policy, scene mode, or algorithm changes
the string → `has_fresh` recomputes even when mtime+size are unchanged (NN-2 / §0: thresholds and models
are global and fixed, so a deliberate change re-indexes rather than silently mixing calibrations).
The LITE sentinel is `EXACT_FV = "exact-only-v1"`.
</content>
