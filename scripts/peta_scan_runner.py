"""Privacy-safe scan runner: scans the given target dirs into the FINAL library DB and prints
ONLY aggregate counts + timing — never file/folder names (full_scan progress disabled, skipped
files reported as a COUNT only). Targets are passed as argv. The DB/report keep the paths for the
user to review privately in the UI.

NOTE: the `if __name__ == "__main__"` guard is REQUIRED — full_scan uses ProcessPoolExecutor and
on Windows (spawn) child processes re-import this module; without the guard they'd re-run the scan."""
import json
import sys
import time
from pathlib import Path

from dupdetect.cli import _bootstrap, _resolve_auto_workers
from dupdetect.pipeline.fullscan import collect_videos, full_scan

import os

from dupdetect.runtime import app_data_dir, default_db_path

DB = os.environ.get("DUPDETECT_DB") or str(default_db_path())   # portable; override with $DUPDETECT_DB


def main() -> int:
    targets = sys.argv[1:]
    if not targets:
        print("no targets"); return 2

    th, store, embedder = _bootstrap(DB, None)
    files = collect_videos(targets, recursive=True)
    print(f"[scan] target videos found: {len(files)}", flush=True)
    print(f"[scan] DB: {DB} (existing files stay; only NEW targets are processed)", flush=True)

    workers, decode_workers = _resolve_auto_workers(0, -1, files)   # auto-tune (HDD -> 2/1)
    t0 = time.perf_counter()
    report = full_scan(files, store, embedder, th, force=False, workers=workers,
                       recursive=True, progress=False, decode_workers=decode_workers)
    elapsed = time.perf_counter() - t0

    # report JSON keeps paths for the UI; written but NOT printed here.
    Path(os.environ.get("DUPDETECT_REPORT") or str(app_data_dir() / "peta_scan_report.json")).write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    print("\n==================== RESULT (counts only) ====================", flush=True)
    print(f"  scanned files      : {len(files)}", flush=True)
    print(f"  elapsed            : {elapsed/60:.1f} min ({elapsed:.0f}s) = "
          f"{elapsed/max(len(files),1):.1f}s/file", flush=True)
    print(f"  clusters (dup grps): {len(report.get('clusters', []))}", flush=True)
    print(f"  review queue       : {len(report.get('review_queue', []))}", flush=True)
    print(f"  editions (related) : {len(report.get('editions', []))}", flush=True)
    print(f"  skipped (unreadable): {len(report.get('skipped', []))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
