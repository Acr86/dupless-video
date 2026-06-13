"""Phase B: re-index ONLY the currently full-indexed files (global_vec NOT NULL) to regenerate the
~64% of missing .npy embeddings, restoring matching + unblocking calibration. Privacy: counts only,
no file names. force=True recomputes; the bumped feature_version (clr1) would force it anyway.

Windows note: full_scan uses a ProcessPool (spawn) -> the run MUST be under __main__ or each worker
re-executes the module (RuntimeError)."""
import time

from dupdetect.cli import _bootstrap
from dupdetect.pipeline.fullscan import full_scan

import os

from dupdetect.runtime import default_db_path

DB = os.environ.get("DUPDETECT_DB") or str(default_db_path())   # portable; override with $DUPDETECT_DB


def main() -> None:
    th, store, embedder = _bootstrap(DB, None)
    paths = [r[0] for r in store.conn.execute("SELECT path FROM files WHERE global_vec IS NOT NULL")]
    print(f"re-indexing {len(paths)} full-indexed files (force, workers=2) ...")
    # workers=2: HDD-safe. The autotune mis-reads this Storage Space's SSD cache as NVMe (0.34ms
    # probe) and would pick 12 workers -> the real reads hit the mechanical tier -> thrashing (~10x).
    t0 = time.time()
    rep = full_scan(paths, store, embedder, th, force=True, workers=2, progress=True)
    dt = time.time() - t0
    print(f"\ndone in {dt/60:.1f} min | clusters={len(rep['clusters'])} "
          f"review={len(rep['review_queue'])} skipped={len(rep['skipped'])}")
    store.close()


if __name__ == "__main__":
    main()
