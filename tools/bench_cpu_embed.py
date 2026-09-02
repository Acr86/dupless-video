#!/usr/bin/env python
"""CPU embedding benchmark — estimates the DINOv2 backfill cost on a CPU-only box (e.g. a NAS).

WHY: on a weak CPU (no GPU) the pipeline stops being I/O-bound and becomes compute-bound on the DINOv2
embedding — the single expensive step of Pass-1. This measures ONLY that step (frames/sec through the
backbone on CPU) and projects it to a whole-library backfill. Decode (ffmpeg) and the cheap
hash/probe/audio steps are NOT included; they run fine on a NAS and overlap I/O.

The verdict is identical on CPU vs GPU (§0); this is purely about HOW LONG the first full scan takes.

RUN (from the repo, in the project venv — or inside the NAS container where the app is installed):

    .venv/Scripts/python.exe tools/bench_cpu_embed.py                 # defaults: 19000 files, 32 frames/file
    python tools/bench_cpu_embed.py --n-files 19000 --frames-per-file 32 --threads 8

  --threads N   cap the CPU threads (set to the NAS's thread count, e.g. 8 on a TS-873A V1500B) to
                get a NAS-like number when running on a beefier PC. Omit to use all cores.

NOTE ON THE MODEL: the first run needs the DINOv2 checkpoint (~330 MB) — served offline from the app
bundle if present, else downloaded via torch.hub (needs network once, then cached). Loading is timed
and reported separately, so it does not pollute the throughput number.
"""
from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time

# Force CPU BEFORE torch is imported anywhere, so this measures the CPU path even on a GPU box.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _human_hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:.0f} min"
    if h < 48:
        return f"{h:.1f} h"
    return f"{h / 24:.1f} days"


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU DINOv2 embedding benchmark (backfill estimator).")
    ap.add_argument("--n-files", type=int, default=19000, help="Library size to project (default 19000).")
    ap.add_argument("--frames-per-file", type=int, default=32,
                    help="Sampled frames embedded per file (default 32; the app samples adaptively).")
    ap.add_argument("--batch", type=int, default=32, help="Embedder batch size on CPU (default 32).")
    ap.add_argument("--n-bench", type=int, default=128, help="Frames to push through for timing (default 128).")
    ap.add_argument("--size", type=int, default=224, help="Frame side in px (DINOv2 vitb14 = 224).")
    ap.add_argument("--repeats", type=int, default=3, help="Timed passes; the median is reported.")
    ap.add_argument("--threads", type=int, default=0,
                    help="Cap CPU threads (0 = use all; set to the NAS thread count for a NAS-like number).")
    args = ap.parse_args()

    import torch

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    from dupdetect.features.embeddings import Embedder

    print("=" * 72)
    print("CPU EMBEDDING BENCHMARK (DINOv2 vitb14)")
    print(f"  platform     : {platform.platform()}")
    print(f"  python       : {sys.version.split()[0]}   torch: {torch.__version__}")
    print(f"  cpu count    : {os.cpu_count()}   torch threads: {torch.get_num_threads()}")
    print(f"  cuda visible : {torch.cuda.is_available()} (forced OFF for this benchmark)")
    print(f"  frame        : {args.size}x{args.size}   batch: {args.batch}")
    print("=" * 72)

    emb = Embedder(batch=args.batch)
    t0 = time.perf_counter()
    emb._ensure()                                    # load the backbone to CPU (times the one-off cost)
    load_s = time.perf_counter() - t0
    print(f"model load (CPU): {load_s:.1f}s   device={emb._device}")
    if emb._device != "cpu":
        print("WARNING: model did not land on CPU — the number below is NOT a CPU measurement.")

    # Synthetic frames: the throughput of the backbone is input-content-independent, so random tensors
    # at the real shape give the same per-frame cost as decoded frames. fp32 (CPU path uses no autocast).
    frames = torch.randn(args.n_bench, 3, args.size, args.size, dtype=torch.float32)

    emb.encode(frames[: args.batch])                 # warmup (allocates buffers, primes caches)

    rates = []
    for r in range(args.repeats):
        t = time.perf_counter()
        out = emb.encode(frames)
        dt = time.perf_counter() - t
        fps = args.n_bench / dt
        rates.append(fps)
        print(f"  pass {r + 1}/{args.repeats}: {args.n_bench} frames in {dt:.2f}s -> {fps:.1f} frames/s "
              f"({1000 * dt / args.n_bench:.1f} ms/frame)   out={out.shape}")

    fps = statistics.median(rates)
    print("-" * 72)
    print(f"MEDIAN THROUGHPUT: {fps:.1f} frames/s  ({1000 / fps:.1f} ms/frame)")

    # Backfill projection (EMBED ONLY — decode/hash/audio not included).
    print("-" * 72)
    print(f"Projected EMBED-ONLY backfill for {args.n_files:,} files "
          f"(decode & I/O NOT included — add margin):")
    for fpf in sorted({16, 32, 64, args.frames_per_file}):
        total_frames = args.n_files * fpf
        hours = total_frames / fps / 3600
        mark = "  <- given" if fpf == args.frames_per_file else ""
        print(f"  {fpf:>3} frames/file : {total_frames:>12,} frames  ~ {_human_hours(hours):>9}{mark}")
    print("=" * 72)
    print("Reading: if this reads as DAYS, the NAS is impractical for the one-time backfill -> keep the")
    print("PC (GPU) as compute and the NAS as storage; the weak CPU is still fine for incremental watch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
