# Design v2.0 — distributed dup-detector (CPU farm + GPU node)

> **Status: PROPOSAL for v2.0. NOT implemented.** Reference document for when the distributed
> version is taken on. The current v1.x (single-machine, in-process) remains the supported mode and
> must stay intact.

## Motivation

The detector is a **component** of a larger tool for managing a big video library. The system is
**measured as I/O-bound** (the bottleneck is disk demux, not the GPU — see
[HPC_PIPELINE.md](HPC_PIPELINE.md)). The **CPU-bound** stages (audio fingerprint, language detection
via whisper, hash, mode-A scenes) are worth **distributing** across Linux containers on heterogeneous
CPUs (from a workstation down to small low-power machines). Only what **requires the GPU** — DINOv2 embeddings
and fine alignment — runs on the GPU node.

## The decisive environment fact: replicated storage

The library is **replicated to every node** (e.g. by a folder-sync tool): every node holds ALL files
locally; only the path prefix differs — `D:\Media\...` (Windows) vs `/Media/...` (Linux), **identical
from `Media` onward**.

Design implications:
- **No videos are moved over the network.** Each node reads its local copy. The design isn't "move
  data to the compute" but **distribute ROLES over already-replicated data**.
- The only data adjustment is **path canonicalization**: the store key is the path **relative to
  `Media`**; each node resolves it against its own `MEDIA_ROOT`.
- Only **small results** cross the network (vectors ~KB, embeddings ~22 MB/film), never the 4–30 GB
  videos.

## Quality invariant (non-negotiable)

**No detection decision — language, embeddings, thresholds, algorithm version — may vary with the node
that processes it.** The result must be identical no matter where it ran.

- Capacity routing decides only **WHERE** a job runs, **never with what quality/model**.
- Models and thresholds are **fixed globally** (already reflected in `feature_version`, which
  invalidates cache if they change).
- Whisper corollary: **one fixed model** across the whole farm. A node that can't run it at quality
  **does not take** that job (a capable node does); it is never run with a lesser model. If no capable
  node is free, the job **waits** — it is not degraded.

## The current coupling already enables the cut

The M3 split by resource is the natural seam:
- `extract_cpu_features(path)` ([analyze.py](../src/dupdetect/pipeline/analyze.py)) — probe, hash,
  `audio_fp`, `detect_language`, mode-A scenes. **Picklable, no CUDA** → a distributable CPU unit.
- `extract_gpu_features(path)` — decode (NVDEC) + DINOv2 embed → a GPU unit.
- `match()` ([matcher.py](../src/dupdetect/match/matcher.py)) + `CoarseIndex` (faiss-cpu) +
  `EmbeddingCache` ([cache.py](../src/dupdetect/match/cache.py)) → **inherently centralized** (needs
  the global index + resident embeddings) → lives on the GPU node.

## Architecture (Docker Compose)

**"Core" services (one host, e.g. the workstation):**
- **Postgres** — shared fingerprint store (replaces the local SQLite; concurrent writes from N
  workers). Metadata + global/window vecs + audio_fp + scene_cuts + frame_times + lang + status.
- **MinIO** (S3) — per-film embedding blobs (`<hash>.npy` fp16, ~22 MB/film). Alternatives: bytea in
  Postgres, or an NFS share (MinIO decouples and is the standard).
- **Redis** — per-film job queue + dedup/idempotency.
- **Coordinator** — enumerates the library (canonical paths), dedups by `content_hash` +
  `feature_version`, enqueues jobs, triggers pass 2 (matching) at the end.

**Workers (Linux containers, on any machine):**
- **`worker --role cpu`** (SLIM image: ffmpeg + fpcalc + faster-whisper CPU, **no torch**). Reads its
  LOCAL copy (resolves the canonical path via `MEDIA_ROOT`). Two job classes for the invariant:
  - **`jobs:light`** (probe, hash, `audio_fp`, mode-A scenes) — taken by **any** node (incl. low-power ones).
  - **`jobs:lang`** (whisper) — **one fixed model**; taken only by nodes with `CAN_RUN_LANG=1`.
- **`worker --role gpu`** (CUDA image: torch cu128 + ffmpeg-nvdec, `NVIDIA_DRIVER_CAPABILITIES` with
  `video` for NVDEC) on the GPU node: decode + DINOv2 → embeddings to MinIO, vecs to Postgres.

**A film's flow:** the Coordinator enqueues `{film}`. The CPU worker does CPU features → Postgres; the
GPU worker does decode+embed → MinIO+Postgres. In PARALLEL, over local copies (no video moved). The
film is "complete" when both parts are in; it enters matching.

**Matching (pass 2):** on the GPU/core node — `CoarseIndex` from Postgres, embeddings from MinIO into
the `EmbeddingCache`, retrieval + `align_video` (GPU). Centralized (needs the global index).

## Code changes (with single-machine mode intact)

1. **Path canonicalization** (foundational): a `media_relative(path)`/`resolve(rel)` helper against
   `MEDIA_ROOT`. The store key moves from absolute to **relative to `Media`**. Touches `collect_videos`/
   `iter_videos` (`fullscan.py`), `FingerprintStore` (keys), reporting. `content_hash` (head|mid|tail)
   is already stable across nodes → good cross-node dedup.
2. **Store abstraction** (`StoreBackend`): extract the interface from `FingerprintStore` (`store.py`)
   + a **Postgres** backend + embeddings in MinIO (`emb_path` is already relative). The SQLite/local
   backend stays the default (single-machine, untouched).
3. **Queue/worker layer** (`dupdetect.dist`): coordinator (enqueues) + `worker(role)` (consumes).
   Reuses `extract_cpu_features`/`extract_gpu_features` as-is. Redis (RQ or a thin protocol). Idempotent
   jobs (skip if `has_fresh`).
4. **CLI**: `dupdetect coordinator <targets>` and `dupdetect worker --role cpu|gpu` (`cli.py`). The
   current in-process `scan` stays as the local mode.
5. **Images + compose**: `Dockerfile.cpu` (slim) and `Dockerfile.gpu` (CUDA). A `docker-compose.yml`
   with the core services; each extra host runs a compose-worker pointing at the core. `MEDIA_ROOT`
   mounted read-only per node (its local replicated path).

## Phases (incremental)
- **F1 — Foundation:** path canonicalization + the `StoreBackend` interface (SQLite stays default). No
  behavior change; the existing test suite stays green.
- **F2 — Shared backends:** Postgres + MinIO. Optional migration (a re-scan repopulates).
- **F3 — Queue + workers + images + compose.**
- **F4 — Capacity routing** (`light`/`lang` queues; `CAN_RUN_LANG` per node). Decides only WHERE;
  models/thresholds global → invariant quality.

## Trade-offs and risks
- **SQLite → Postgres**: the biggest change (concurrent writes). Mitigated by the abstraction and by
  keeping SQLite for single-machine.
- **Embeddings off local disk** (MinIO): adds a dependency. Alternatives above. Do NOT use a sync'd
  folder for the writable index (concurrent-write conflicts).
- **The GPU node still decodes locally** (~5s/film, I/O-bound); distributing CPU frees it from
  audio+whisper (~7s) → ~2x its pass-1 throughput. The GPU becomes the aggregate limiter (scalable by
  adding GPU nodes later).
- **Invariant quality (resolved):** one fixed language model; incapable nodes don't take `jobs:lang`.
  A film's `lang` does NOT depend on who processed it. The price is throughput, not quality. Same rule
  for any parameter that affects a verdict.
- **Matching is not distributed** (global index) — it lives on the GPU node.
- **Failures:** Redis re-queues if a worker dies; `content_hash` + `feature_version` make jobs
  idempotent.

## Verification (when implemented)
1. **F1:** the test suite stays green unchanged; a `media_relative`/`resolve` test (round-trip
   Windows↔POSIX, different prefix, same suffix after `Media`).
2. **Store backend:** a Postgres-backend test against the SAME contract as SQLite (save/load/has_fresh/
   all_global_vecs/matches/clusters), with an ephemeral Postgres.
3. **E2E (compose):** core + 1 cpu-worker + 1 gpu-worker; `coordinator` over a test folder; compare
   clusters/verdicts with the single-machine scan (identical: same groupings, 0 FP). Confirm the video
   does NOT travel (local I/O) and that only vecs/embeddings cross to Postgres/MinIO.
4. **Heterogeneity + quality invariant:** a low-power node with `CAN_RUN_LANG=0` and a strong node with
   `CAN_RUN_LANG=1`. Verify that (a) the weak node never runs whisper, (b) each film's `lang` is identical no
   matter who processed it (same fixed model): re-queuing on another capable node gives the same `lang`.
5. **Resilience:** kill a worker mid-job → re-queue and idempotency (no duplicates, identical final
   result).

## Open question (decide at F2)
Embeddings: MinIO (recommended), bytea in Postgres, or an NFS share?
