"""Fast path for calibration: re-index ONLY the files referenced in labels.csv (regenerate their
.npy), so the threshold sweep can recompute signals. ~100 files, workers=2 (HDD-safe: the autotune
mis-reads this Storage Space's SSD cache as NVMe and would pick 12 -> thrashing). Privacy: counts only.

Windows: full_scan uses a ProcessPool (spawn) -> run under __main__."""
import csv
import os
import time

from dupdetect.cli import _bootstrap
from dupdetect.pipeline.analyze import analyze_file

from dupdetect.runtime import app_data_dir, default_db_path

# Portable: default to the per-user data dir; override with $DUPDETECT_DB / $DUPDETECT_LABELS.
DB = os.environ.get("DUPDETECT_DB") or str(default_db_path())
CSV = os.environ.get("DUPDETECT_LABELS") or str(app_data_dir() / "labels.csv")


def main() -> None:
    rows = list(csv.DictReader(open(CSV, newline="", encoding="utf-8")))
    files = sorted({r["path_a"] for r in rows} | {r["path_b"] for r in rows})
    present = [p for p in files if os.path.exists(p)]
    print(f"files in CSV: {len(files)} | exist on disk: {len(present)}")
    # Pass-1 ONLY (regenerate embeddings/.npy). NOT full_scan: matching the labeled files against
    # the WHOLE indexed library would enumerate ~150k candidate pairs (hours). We only need the
    # .npy back so calibration's signal_for_pair can recompute the labeled pairs' signals.
    th, store, embedder = _bootstrap(DB, None)
    t0, done, failed = time.time(), 0, 0
    for i, p in enumerate(present, 1):
        try:
            analyze_file(p, store, embedder, th, force=True)
            done += 1
        except Exception as e:                          # noqa: BLE001  skip-and-report (§2)
            failed += 1
            print(f"  [{i}] skip {type(e).__name__}")
        if i % 10 == 0:
            print(f"  {i}/{len(present)}  ({(time.time()-t0)/60:.1f} min)")
    print(f"\ndone in {(time.time()-t0)/60:.1f} min | regenerated={done} failed={failed}")
    store.close()


if __name__ == "__main__":
    main()
