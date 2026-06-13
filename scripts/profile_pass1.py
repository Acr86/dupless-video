"""Profiler for Pass-1: WHERE does the ~30s/file go? (decode vs embed vs full-file hash vs audio
coverage vs probe). Wraps the REAL stage functions (production code NOT modified) and runs the
CPU + GPU extraction on a sample of files, then prints a per-stage breakdown.

Run on a folder of REAL files; it re-extracts from scratch (heavy disk I/O) so run it alone (no
scan/watcher competing). Privacy: aggregate timings only, no file names.

Usage:
  python scripts/profile_pass1.py "D:\\Videos\\Some Folder" [--db DB] [--limit N]
Portable: --db defaults to the per-user data dir; override with $DUPDETECT_DB.
"""
import argparse
import time
from collections import defaultdict

from dupdetect.cli import _bootstrap
from dupdetect.pipeline import analyze
from dupdetect.pipeline.fullscan import collect_videos
from dupdetect.runtime import default_db_path

_T: dict = defaultdict(lambda: [0.0, 0])          # stage -> [total_seconds, n_calls]


def _timed(name: str, fn):
    def wrap(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            d = _T[name]
            d[0] += time.perf_counter() - t0
            d[1] += 1
    return wrap


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-stage Pass-1 profiler.")
    ap.add_argument("target", help="folder to sample real files from")
    ap.add_argument("--db", default=str(default_db_path()))
    ap.add_argument("--limit", type=int, default=8, help="number of files to profile")
    args = ap.parse_args()

    th, store, embedder = _bootstrap(args.db, None)
    files = collect_videos([args.target], recursive=True)[: args.limit]
    print(f"profiling Pass-1 on {len(files)} files under: {args.target}")
    if not files:
        store.close()
        return

    # Wrap the real stage functions (names as used inside analyze.extract_*).
    analyze.ffprobe = _timed("probe (ffprobe)", analyze.ffprobe)
    analyze.content_hash = _timed("content_hash (full-file read)", analyze.content_hash)
    analyze.scan_audio_coverage = _timed("audio_coverage (seek-sampled)", analyze.scan_audio_coverage)
    analyze.cam_score = _timed("cam_score (partial)", analyze.cam_score)
    analyze.decode_frames = _timed("decode_frames (ffmpeg)", analyze.decode_frames)
    analyze.scene_cuts_from_embeddings = _timed("scene_cuts (from emb)", analyze.scene_cuts_from_embeddings)
    embedder.encode = _timed("embed (DINOv2 GPU)", embedder.encode)

    t0 = time.perf_counter()
    ok = 0
    for p in files:
        try:
            cpu = analyze.extract_cpu_features(p)
            emb, times, color = analyze.extract_gpu_features(p, cpu.probe, embedder, th)
            analyze.build_record(cpu, emb, times, color, embedder, th)
            ok += 1
        except Exception as e:                    # noqa: BLE001 skip-and-report
            print(f"  skip {type(e).__name__}: {str(e)[:60]}")
    total = time.perf_counter() - t0

    print(f"\n=== Pass-1 profile: {ok}/{len(files)} files, {total:.1f}s total ===")
    print(f"{'stage':34} {'total_s':>9} {'%':>6} {'calls':>7} {'s/call':>9}")
    for name, (secs, calls) in sorted(_T.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:34} {secs:9.2f} {100 * secs / max(total, 1e-9):6.1f} "
              f"{calls:7} {secs / max(calls, 1):9.2f}")
    print(f"\nper file: {total / max(ok, 1):.2f}s")
    store.close()


if __name__ == "__main__":
    main()
