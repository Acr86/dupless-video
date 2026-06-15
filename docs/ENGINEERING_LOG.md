# ENGINEERING_LOG — solved problems & non-obvious gotchas (Dupless Video)

**Purpose:** durable memory of things already figured out, so they are never re-investigated.
Grep this file (by symptom) before any non-trivial debugging or "why does X happen" exploration;
append an entry when you solve something that took real digging. A fix without a log entry is half-finished.

**Entry format:** `### <searchable symptom>` then **Symptom / Root cause / Resolution / Files-refs / Date-scope**.
Newest on top. Append-only in spirit. Deep rationale for the design invariants lives in [CLAUDE.md](../CLAUDE.md).

---

## Entries

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
