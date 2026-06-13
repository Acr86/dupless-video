"""Run calibration on the hand-labeled CSV: recompute each pair's signals from the STORED
fingerprints (no re-decode), then report precision (zero-FP of dup tiers), review-queue noise,
recall, and suggested thresholds. Privacy: prints counts/verdicts only, never file names."""
import csv

from dupdetect.config import load_thresholds
from dupdetect.match.cache import EmbeddingCache
from dupdetect.pipeline.calibrate import (
    DUP_VERDICTS, STRONG_VERDICTS, confusion_by_genre, confusion_by_tier, signal_for_pair,
    suggest_thresholds, verdict_of,
)
from dupdetect.store import FingerprintStore

import os

from dupdetect.runtime import app_data_dir, default_db_path

# Portable: default to the per-user data dir; override with $DUPDETECT_DB / $DUPDETECT_LABELS.
DB = os.environ.get("DUPDETECT_DB") or str(default_db_path())
CSV = os.environ.get("DUPDETECT_LABELS") or str(app_data_dir() / "labels.csv")

rows = list(csv.DictReader(open(CSV, newline="", encoding="utf-8")))
th = load_thresholds()
store = FingerprintStore(DB, init_schema=False)
cache = EmbeddingCache(store)

signals, skipped = [], []
for r in rows:
    label = (r.get("label") or "").strip()
    try:
        sig = signal_for_pair(r["path_a"], r["path_b"], label or "different", store, cache, th,
                              genre=(r.get("genre") or "").strip())   # DEV label; never enters tree (§0)
    except (OSError, KeyError):                             # orphaned/missing .npy -> skip-and-report
        sig = None
    if sig is None:
        skipped.append(r["bucket"]); continue
    sig._bucket = r["bucket"]                               # attach for reporting
    signals.append(sig)

from collections import Counter
print(f"pairs in CSV: {len(rows)} | evaluated: {len(signals)} | "
      f"skipped (LITE / .npy missing or moved): {len(skipped)} {dict(Counter(skipped))}\n")

# --- with CURRENT thresholds
print("=== Recomputed verdict vs label (CURRENT thresholds) ===")
print(f"{'verdict':14s} {'same':>5s} {'diff':>5s}")
conf = confusion_by_tier(signals, th)
for v, cell in sorted(conf.items()):
    print(f"  {v:12s} {cell['same']:5d} {cell['diff']:5d}")

# precision of dup tiers (CERTAIN/VERY_HIGH/HIGH) + the critical T1/T2 false positives
dup_same = sum(c.get("same", 0) for v, c in conf.items() if v in {x.value for x in DUP_VERDICTS})
dup_diff = sum(c.get("diff", 0) for v, c in conf.items() if v in {x.value for x in DUP_VERDICTS})
fp_strong = sum(conf.get(v.value, {}).get("diff", 0) for v in STRONG_VERDICTS)
prec = dup_same / (dup_same + dup_diff) if (dup_same + dup_diff) else 0.0
same_total = sum(1 for s in signals if s.is_same)
recall = sum(1 for s in signals if s.is_same and verdict_of(s, th) in DUP_VERDICTS) / max(same_total, 1)
print(f"\nPRECISION dup-tiers (CERTAIN/VERY_HIGH/HIGH): {prec:.1%}  ({dup_same} same / {dup_diff} diff)")
print(f"FALSE POSITIVES in T1/T2 (zero-FP target = 0): {fp_strong}")
print(f"RECALL (of {same_total} 'same' pairs, how many land in a dup-tier): {recall:.1%}")

# review-queue noise: PROBABLE pairs that are actually different
prob = conf.get("PROBABLE", {"same": 0, "diff": 0})
print(f"\nREVIEW-queue noise (PROBABLE): {prob['same']} same / {prob['diff']} diff "
      f"-> {prob['diff']/max(prob['same']+prob['diff'],1):.0%} were actually different")

# any T1/T2 false positive? surface the pair (path basenames only, for the user to inspect)
import os
for s in signals:
    if not s.is_same and verdict_of(s, th) in STRONG_VERDICTS:
        print(f"  !! FP {verdict_of(s, th).value}: ...{os.path.basename(s.a_path)[-34:]}  vs  "
              f"...{os.path.basename(s.b_path)[-34:]}")

# --- per-genre breakdown: WHERE the single global logic leaks (genre never enters the verdict, §0)
print("\n=== Precision / recall BY GENRE (one global logic; genre = measurement label only) ===")
print(f"{'genre':16s} {'n':>4s} {'prec':>6s} {'recall':>7s} {'FP-T1/T2':>8s}  (dup same/diff, same total)")
for g, m in confusion_by_genre(signals, th).items():
    pr = f"{m['precision']:.0%}" if m["precision"] is not None else "  -"
    rc = f"{m['recall']:.0%}" if m["recall"] is not None else "  -"
    flag = "  <-- !!" if m["fp_strict"] else ""
    print(f"{g[:16]:16s} {m['n']:4d} {pr:>6s} {rc:>7s} {m['fp_strict']:8d}  "
          f"({m['dup_same']}/{m['dup_diff']}, {m['same_total']}){flag}")

# --- suggested thresholds (sweep theta_v/theta_a for zero-FP then max recall)
print("\n=== Threshold suggestion (sweep) ===")
sug = suggest_thresholds(signals, base=th)
print(f"  theta_v={sug['theta_v']} theta_a={sug['theta_a']}  "
      f"(current: theta_v={th.theta_v} theta_a={th.theta_a})")
print(f"  -> FP T1/T2={sug['false_positives_T1T2']}  recall_dup={sug['recall_dup']}  n={sug['n_pairs']}")
store.close()
