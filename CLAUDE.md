# CLAUDE.md — Engineering charter

The working agreement for **Dupless Video** — a content-based duplicate / upgrade detector built to
run at scale (tens of thousands of files) and to extract the most from the hardware, *or surface the
right knob to the user, clearly, in the UI*. It also served as the operating contract for the project's
AI-assisted development: every principle below is a guardrail the work was held to. Trivial edits use
judgment — the full ceremony isn't for fixing a typo.

## 0. Non-negotiables — these override convenience, and even raw speed
- **Zero false positives in the strong tiers (T1/T2).** A wrong "duplicate" can cost someone a file,
  so recall is recovered from a review queue — never from loosening thresholds.
- **A verdict must not depend on config, node, or hardware.** Same input → same result; CPU and GPU
  must agree (measured: fp32 vs fp16 = 0 verdict flips). "Maximize performance" never means "lower
  precision." Models and thresholds are global and fixed.
- **Detect, don't trust.** Anything that drives a decision — language, content, structure — is measured
  from the bytes, never read from metadata that can lie.

## 1. Measure before optimizing — then fix the *real* bottleneck
Profile first; spend effort where the data points, not where it sounds impressive (Amdahl). This
pipeline is **I/O-bound on disk**, not compute-bound: GPU tricks (FP8/FP4, async CUDA streams) measured
at ~zero net gain; the real wins were I/O — adaptive sampling, decode↔embed overlap, avoiding disk
contention. A perf change ships **off by default until a benchmark shows it pays here**; if it risks
the zero-FP precision it is gated behind a fidelity check that reverts on regression. The optimum is
storage-aware (a spinning HDD thrashes under the concurrency an NVMe loves) — auto-tune, or expose the
knob with one line of guidance.

## 2. Scale-resilient by default
At tens of thousands of files the rare case is a certainty: corrupt/unreadable files, missing data, 8K
giants, slow disks. **Skip and report — never crash the batch.** Unbounded work streams and stays
incremental (skip already-done work; self-heal stale or orphaned data instead of crashing on it). Any
long-running operation shows progress + ETA + live counts — never a silent "is it frozen?".

## 3. Tuning is a feature — surface it, don't bury it
A *measured* lever (workers, decode-workers, resolution cap, exact-only, …) ships with three things: a
sensible default, a UI control, and a one-line "when/why + expected effect." Configurability that maps
to a measured outcome is in scope; knobs that change no measurable outcome are not — drop those.

## 4. Think before coding; ask on forks
State assumptions. When several readings exist, present them — don't pick silently; push back when a
simpler or cheaper path exists. Separate **contract bugs** (just fix) from **forks** — changes to
detection semantics, calibration, or thresholds — and never change a fork silently: surface it with the
measured trade-off.

## 5. Surgical changes
Every changed line traces to the request **or a measured bottleneck**. Don't refactor what isn't broken
or "improve" adjacent code. Match the surrounding densely-commented style; all code — comments,
docstrings, identifiers, and user-facing UI/CLI strings — is written in **English**. Remove only the
orphans a change creates; flag pre-existing dead code rather than deleting it.

## 6. Verify — tests *and* numbers
Bug → write the failing test first, then fix. Feature → tests for the behavior; keep the suite green.
Every performance claim carries a before/after benchmark on representative data — not a vibe. "Done"
means verified: the tests pass **and** the app actually runs the change.

---

**This is working if:** diffs are small and trace to the request or a measured bottleneck; performance
claims come with numbers; scale failures are skipped-and-reported, not crashes; the user gets clear,
defaulted knobs for their hardware; and the zero-false-positive and hardware-invariance guarantees hold.
