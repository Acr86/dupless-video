"""Generate a labeling CSV for calibration. Pairs are drawn from FULL-indexed files (with
embeddings, so calibrate can recompute the signals). Buckets:
  dup      - system says DUPLICATE (CERTAIN/VERY_HIGH/HIGH/NAME_COPY) -> precision / zero-FP check
  review   - system says PROBABLE (review queue, NOT a dup claim) -> measures review-queue noise
  recall   - same clean_title, NOT matched -> suspected dups the system may have MISSED
  negative - plausible-but-different (different title, similar duration)
The user fills `label`: same / upgrade / dub / cam / different (blank = different).
"""
import csv
import itertools
import os
import sqlite3
from collections import Counter

from dupdetect.store.store import canonical_pair
from dupdetect.ui.data import clean_title

from dupdetect.runtime import app_data_dir, default_db_path

# Genre is a DEV measurement label ONLY (never a runtime input, §0). We infer an EDITABLE default
# from the folder layout so the user just tweaks it; it is used solely to break the precision/recall
# report down per content type (see calibrate.confusion_by_genre).
_GENRE_KEYS = ("series", "movies", "movie", "cine", "cinema", "documental", "documentary",
               "docs", "anime", "concierto", "concert", "music", "ai", "shorts")


def _infer_genre(path: str) -> str:
    """Best-effort genre from the path: the component right after a 'Media'-like root, else a
    recognized keyword found in the path, else the file's grandparent folder. Always editable."""
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    low = [p.lower() for p in parts]
    for i, p in enumerate(low):                         # component after a 'media'/'library' root
        if p in ("media", "library", "videos") and i + 1 < len(parts):
            return parts[i + 1]
    for p in low:                                       # any recognized genre keyword in the path
        if p in _GENRE_KEYS:
            return p
    return parts[-3] if len(parts) >= 3 else (parts[0] if parts else "unknown")

# Portable: default to the per-user data dir; override with $DUPDETECT_DB / $DUPDETECT_LABELS.
DB = os.environ.get("DUPDETECT_DB") or str(default_db_path())
OUT = os.environ.get("DUPDETECT_LABELS") or str(app_data_dir() / "labels.csv")
N_DUP, N_REVIEW, N_RECALL, N_NEG = 20, 15, 12, 6
DUP_VERDS = {"CERTAIN", "VERY_HIGH", "HIGH", "NAME_COPY"}

conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row
full = {r["path"]: r for r in conn.execute(
    "SELECT path, duration_s FROM files WHERE global_vec IS NOT NULL")}  # full-indexed only
matched = set()
verd = {}
for r in conn.execute("SELECT a_path, b_path, verdict FROM matches"):
    pr = canonical_pair(r["a_path"], r["b_path"]); matched.add(pr); verd[pr] = r["verdict"]

rows = []  # (bucket, system_verdict, a, b)

# dup: the system's DUPLICATE claims (must be precise) ; review: PROBABLE (borderline queue)
both = [(a, b) for (a, b) in matched if a in full and b in full]
for a, b in [p for p in both if verd[p] in DUP_VERDS][:N_DUP]:
    rows.append(("dup", verd[(a, b)], a, b))
for a, b in [p for p in both if verd[p] == "PROBABLE"][:N_REVIEW]:
    rows.append(("review", "PROBABLE", a, b))

# recall: same clean_title, full-indexed, NOT matched, small groups (likely real dup sets)
by_title: dict = {}
for p in full:
    by_title.setdefault(clean_title(os.path.basename(p)).lower(), []).append(p)
recall = []
for title, ps in by_title.items():
    if not title or len(ps) < 2 or len(ps) > 4:   # skip empties and huge generic groups
        continue
    for a, b in itertools.combinations(sorted(ps), 2):
        pr = canonical_pair(a, b)
        if pr not in matched:
            recall.append(pr)
for a, b in recall[:N_RECALL]:
    rows.append(("recall", "(none)", a, b))

# negatives: different clean_title but similar duration (plausible-but-different), not matched
flist = sorted(full)
negs = []
for a, b in itertools.combinations(flist, 2):
    if len(negs) >= N_NEG * 4:
        break
    da, db = full[a]["duration_s"] or 0, full[b]["duration_s"] or 0
    if da <= 0 or db <= 0:
        continue
    if abs(da - db) / max(da, db) < 0.02 and canonical_pair(a, b) not in matched \
       and clean_title(os.path.basename(a)).lower() != clean_title(os.path.basename(b)).lower():
        negs.append((a, b))
for a, b in negs[:N_NEG]:
    rows.append(("negative", "(none)", a, b))

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["bucket", "system_verdict", "genre", "path_a", "path_b", "label"])
    for bucket, sv, a, b in rows:
        # genre: inferred default from BOTH paths (use a's; note if they disagree) -> user edits.
        ga, gb = _infer_genre(a), _infer_genre(b)
        genre = ga if ga == gb else f"{ga}?"            # '?' flags a cross-folder pair to review
        w.writerow([bucket, sv, genre, a, b, ""])

c = Counter(r[0] for r in rows)
print(f"labels.csv -> {OUT}")
print(f"  total pairs: {len(rows)} | dup={c['dup']} review={c['review']} "
      f"recall={c['recall']} negative={c['negative']}")
print("  fill the 'label' column: same / upgrade / dub / cam / different (blank = different)")
