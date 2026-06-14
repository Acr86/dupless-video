"""CLI. `dupdetect scan <lib>`, `dupdetect check <file>`, `dupdetect calibrate`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

# Windows console uses cp1252 and filenames may contain characters outside that set
# (Chinese titles, unusual accents...). Forcing UTF-8 prevents a typer.echo of such
# a name from crashing the CLI.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from dupdetect.config import load_thresholds
from dupdetect.features.embeddings import Embedder
from dupdetect.match.cache import EmbeddingCache
from dupdetect.match.retrieval import CoarseIndex
from dupdetect.pipeline.fullscan import full_scan
from dupdetect.pipeline.single import analyze_single
from dupdetect.runtime import default_db_path, scan_priority_lock
from dupdetect.store import FingerprintStore

app = typer.Typer(add_completion=False, help="Duplicate/upgrade detector for movies")

# Per-user, persistent, cross-platform (see runtime.app_data_dir). Override with --db or
# $DUPDETECT_DATA_DIR. No machine-specific path baked in -> portable for every user.
DEFAULT_DB = default_db_path()


def _bootstrap(db: Path, config: Path | None):
    th = load_thresholds(config)
    store = FingerprintStore(db)
    embedder = Embedder(
        model=th.raw["embeddings"]["model"],
        dim=th.raw["embeddings"]["dim"],
        batch=th.raw["embeddings"]["batch"],
        fps=th.fps_sample,          # C4: actual fps must feed into feature_version
    )
    return th, store, embedder


def _resolve_auto_workers(workers: int, decode_workers: int,
                          files: list[str]) -> tuple[int, int]:
    """0/-1 = AUTO: probes storage (random-read latency) and returns concrete
    (workers, decode_workers), printing the rationale note. Values >0/>=0 are
    respected as-is (manual override). Verdict does NOT depend on this -> speed only."""
    if workers > 0 and decode_workers >= 0:
        return workers, decode_workers
    from dupdetect.tuning import autotune
    at = autotune(files)
    typer.echo(at.note, err=True)
    return (at.workers if workers <= 0 else workers,
            at.decode_workers if decode_workers < 0 else decode_workers)


@app.command()
def scan(
    targets: list[str] = typer.Argument(..., help="Paths: files and/or folders (in place, without copying)"),
    db: Path = typer.Option(DEFAULT_DB, help="SQLite store path"),
    config: Path = typer.Option(None, help="thresholds.yaml (default: config/)"),
    force: bool = typer.Option(False, help="Recompute even if already in the store"),
    workers: int = typer.Option(
        0, help="CPU workers for features. 0=AUTO (probes storage: ~2 on HDD, "
                "higher on SSD/NVMe). >0 sets the value manually."),
    decode_workers: int = typer.Option(
        -1, help="Threads for parallel video decode (GPU prefetch). -1=AUTO (follows "
                 "auto-tune). >=1 sets manually; >1 ONLY on SSD/NVMe (causes thrashing on HDD)."),
    fp8_embed: bool = typer.Option(
        False, help="FP8 inference (Blackwell) for DINOv2 embed. Opt-in: only pays off in "
                    "compute-bound scenarios (NVMe/8K) with fused quant; fidelity guard reverts "
                    "to fp16 if cosine drops <0.99. Default fp16."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive",
                                   help="--no-recursive: top level only, no subfolders"),
    independent_scenes: bool = typer.Option(
        False, "--independent-scenes/--fast-scenes",
        help="--independent-scenes: scenes by pixels (slow, video-independent, "
             "better for cam). Default (--fast-scenes): derived from embeddings (fast)."),
    max_height: int = typer.Option(0, help="Exclude videos with height > N px (e.g. 1080 skips 4K). 0=no filter"),
    depth: str = typer.Option(
        "standard", "--depth",
        help="Analysis depth (incremental — deeper levels reuse what shallower ones computed): "
             "'fast' = byte-identical only (hash, ~0.1s/file); 'standard' = visual+audio duplicate "
             "detection (audio quality computed on-demand for duplicates); 'deep' = standard + "
             "whole-file audio quality for EVERY file (slower)."),
    exact_only: bool = typer.Option(
        False, "--exact-only/--full",
        help="Alias of --depth fast (BYTE-IDENTICAL only). Kept for back-compat."),
    no_match: bool = typer.Option(
        False, "--no-match", help="Pass-1 ONLY: compute+persist features (embeddings/audio/scenes), "
                                  "skip matching. Use to (re)index a large library cheaply, then "
                                  "run a normal scan to do Pass-2. Avoids the O(N^2) candidate "
                                  "blow-up until you're ready."),
    out: Path = typer.Option(None, help="Cluster report JSON (default: a 'reports' folder next to the DB)"),
):
    """Indexes paths (files and/or folders, in place, without copying) and reports clusters + review queue."""
    from dupdetect.pipeline.fullscan import collect_videos, filter_by_height

    # Report goes NEXT TO THE DB (always writable — we open the DB there). A CWD-relative default like
    # 'reports/scan.json' crashed the frozen app: its CWD is the read-only install dir -> WinError 5.
    out = Path(out) if out is not None else Path(db).parent / "reports" / "scan.json"
    th, store, embedder = _bootstrap(db, config)
    if fp8_embed:
        embedder.fp8 = True                            # applied when the model is loaded (lazy)
    # Warn before enumeration: on large libraries tree traversal takes a while and without
    # this the UI would go silent. flush so the panel receives it immediately.
    typer.echo(f"Searching videos in {len(targets)} path(s) (recursive={recursive})…", err=True)
    import sys as _sys; _sys.stderr.flush()
    files = collect_videos(targets, recursive=recursive)
    typer.echo(f"Searching videos: {len(files)} found.", err=True)
    collected = set(files)
    for t in targets:                                  # warn about ignored/non-existent paths
        p = Path(t)
        if not p.exists():
            typer.echo(f"WARNING: does not exist -> {t}")
        elif p.is_file() and str(p) not in collected:
            typer.echo(f"WARNING: unrecognized extension, ignored -> {p.name} ({p.suffix})")

    excluded_h: list[str] = []
    if max_height:                                     # resolution filter (fast probe, ffprobe)
        files, excluded_h = filter_by_height(files, max_height, workers if workers > 0 else 8)
        typer.echo(f"Resolution filter height<={max_height}px: {len(files)} pass, "
                   f"{len(excluded_h)} excluded (4K/tall)")

    depth = (depth or "standard").lower()
    if depth not in ("fast", "standard", "deep"):
        raise typer.BadParameter("--depth must be fast | standard | deep")
    is_fast = exact_only or depth == "fast"            # --exact-only is an alias of fast
    eager_coverage = depth == "deep"                   # whole-file audio quality for ALL files
    typer.echo(f"Analyzing {len(files)} file(s) -> DB {db}  (depth={'fast' if is_fast else depth})")
    if len(files) < 2:
        typer.echo("(with <2 files there are no pairs to compare: 0 duplicates is expected)")
    # The user-initiated scan takes priority: the background watcher yields the disk/DB while this
    # lock is held (avoids HDD thrashing + SQLite write starvation — see runtime.scan_priority_lock).
    with scan_priority_lock():
        if is_fast:
            from dupdetect.pipeline.fullscan import exact_scan
            ew = workers if workers > 1 else 8         # 1 worker hashing a huge library is pointless
            typer.echo(f"Fast mode (hash, no embeddings) — {ew} workers")
            report = exact_scan(files, store, th, workers=ew, recursive=recursive, progress=True)
        else:
            workers, decode_workers = _resolve_auto_workers(workers, decode_workers, files)
            report = full_scan(files, store, embedder, th, force=force, workers=workers,
                               recursive=recursive, independent_scenes=independent_scenes,
                               progress=True, decode_workers=decode_workers, match=not no_match,
                               eager_coverage=eager_coverage)
    report["excluded_by_height"] = excluded_h          # already filtered in the CLI
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False),
                   encoding="utf-8")
    skipped = report.get("skipped", [])
    if skipped:
        typer.echo(f"Skipped {len(skipped)} unreadable file(s) (corrupt/no moov):")
        for pth, err in skipped:
            typer.echo(f"  - {Path(pth).name}: {err[:70]}")
    typer.echo(f"Clusters: {len(report['clusters'])} | "
               f"review: {len(report['review_queue'])} | "
               f"editions: {len(report['editions'])} | "
               f"skipped: {len(skipped)} | -> {out}")
    allprobs = store.iter_problems()
    if allprobs:
        typer.echo(f"DB: {len(allprobs)} problematic file(s) in table 'problems' "
                   f"(query: SELECT * FROM problems)")
    store.close()


@app.command()
def check(
    file: str = typer.Argument(..., help="New file to check"),
    db: Path = typer.Option(DEFAULT_DB),
    config: Path = typer.Option(None),
):
    """Checks ONE file against the library (same engine as the watcher)."""
    th, store, embedder = _bootstrap(db, config)
    paths, gvecs = store.all_global_vecs()
    w_owners, wvecs = store.all_window_vecs()
    index = CoarseIndex(dim=gvecs.shape[1] if gvecs.size else th.raw["embeddings"]["dim"])
    index.build(paths, gvecs, window_owners=w_owners, window_vecs=wvecs)
    cache = EmbeddingCache(store)
    cache.warm(paths)

    results = analyze_single(file, store, embedder, index, th, cache=cache)
    if not results:
        typer.echo("No duplicates: appears unique in the library.")
    for r in results:
        typer.echo(f"[{r.verdict}] {r.confidence:.2f}  {r.reason}")
        typer.echo(f"        vs {r.candidate_path}")
    store.close()


@app.command()
def ui(db: Path = typer.Option(DEFAULT_DB, help="SQLite store path to explore"),
       tray: bool = typer.Option(False, "--tray",
                                 help="Start hidden in the system tray (used by the run-at-login entry).")):
    """Desktop UI: sortable duplicate tree, preview in VLC, send to Trash with
    confirmation, and feedback that recalibrates thresholds. Reads the existing DB; does not re-scan."""
    from dupdetect.ui.main import run
    raise typer.Exit(run(str(db), start_hidden=tray))


@app.command()
def calibrate(
    pairs: Path = typer.Argument(..., help="Hand-labeled pairs (CSV or JSON)"),
    db: Path = typer.Option(DEFAULT_DB),
    config: Path = typer.Option(None),
):
    """Evaluates thresholds against a labeled set and suggests theta_v/theta_a with ZERO FP in T1/T2."""
    from dupdetect.pipeline.calibrate import (
        compute_signals, confusion_by_tier, load_pairs, suggest_thresholds,
    )

    th, store, embedder = _bootstrap(db, config)
    labeled = load_pairs(pairs)
    typer.echo(f"Calibrating with {len(labeled)} pairs...")
    signals = compute_signals(labeled, store, embedder, th)

    typer.echo("\nConfusion matrix (current thresholds):")
    for verdict, cell in confusion_by_tier(signals, th).items():
        typer.echo(f"  {verdict:18} same={cell['same']:3}  diff={cell['diff']:3}")

    sug = suggest_thresholds(signals, base=th)
    typer.echo(f"\nSugerencia (cero FP en T1/T2): theta_v={sug['theta_v']} "
               f"theta_a={sug['theta_a']} | FP_T1T2={sug['false_positives_T1T2']} "
               f"recall_dup={sug['recall_dup']}")
    store.close()


@app.command(name="repair-indexes")
def repair_indexes(
    db: Path = typer.Option(DEFAULT_DB, help="SQLite store path"),
    apply: bool = typer.Option(
        False, "--apply", help="Rebuilds the indexes (remux -c copy, lossless). Without this, "
                               "only LISTS. Replaces the original atomically."),
):
    """Lists (or rebuilds with --apply) the indexes of VALID files with slow seek
    (category 'reindex', those skipped due to timeout). Remux does not re-encode -> the
    verdict does not change; after repair, the NEXT run checks them as duplicates."""
    store = FingerprintStore(db)
    items = store.problems(category="reindex")
    if not items:
        typer.echo("No files with an index to rebuild.")
        store.close()
        raise typer.Exit()
    typer.echo(f"{len(items)} file(s) with an index to rebuild:")
    for p, _err, _cat, _note in items:
        typer.echo(f"  - {p}")
    if not apply:
        typer.echo("\n(list only; use --apply to rebuild with remux -c copy)")
        store.close()
        raise typer.Exit()

    import os as _os
    import sys as _sys
    import time as _time

    from dupdetect.repair import remux_rebuild_index

    # Weight by bytes -> realistic ETA (files range from MB to tens of GB; an ETA by
    # file count would lie). total_bytes is the sum of existing files.
    sizes = {p: (_os.path.getsize(p) if _os.path.exists(p) else 0) for p, *_ in items}
    total_bytes = sum(sizes.values()) or 1
    t0 = _time.monotonic()
    done_bytes = 0
    last_emit = [0.0]

    def _emit(idx: int, path: str, frac: float, force: bool = False) -> None:
        """Progress line parseable by the UI (REPAIR_PROGRESS …), throttled ~0.5s. flush=True
        because stdout is not a TTY under QProcess (otherwise it would not arrive live)."""
        now = _time.monotonic()
        if not force and now - last_emit[0] < 0.5:
            return
        last_emit[0] = now
        processed = done_bytes + frac * sizes.get(path, 0)
        overall = processed / total_bytes * 100
        elapsed = now - t0
        rate = processed / elapsed if elapsed > 0 else 0          # bytes/s
        eta = int((total_bytes - processed) / rate) if rate > 0 else -1
        print(f"REPAIR_PROGRESS idx={idx} total={len(items)} file_pct={int(frac * 100)} "
              f"overall_pct={int(overall)} eta_s={eta} done_gb={processed / 1e9:.1f} "
              f"total_gb={total_bytes / 1e9:.1f} file={Path(path).name}", flush=True)

    ok = fail = gone = 0
    for idx, (p, _err, _cat, _note) in enumerate(items, start=1):
        name = Path(p).name
        _emit(idx, p, 0.0, force=True)                            # marks the start of this file
        good, kind, msg = remux_rebuild_index(
            p, on_progress=lambda f, _i=idx, _p=p: _emit(_i, _p, f))
        done_bytes += sizes.get(p, 0)
        if good:
            store.clear_problem(p)                 # success -> next run checks it
            ok += 1
            typer.echo(f"  ✓ {name}")
        elif kind == "gone":
            store.clear_problem(p)                 # no longer exists -> forget it (self-heal)
            gone += 1
            typer.echo(f"  · {name}: {msg} (olvidado)")
        else:
            # PERSIST the result: 'timeout' remains repairable (retryable), any other kind
            # becomes unrecoverable with the reason -> stops being retried blindly.
            store.mark_repair_failed(p, kind, msg)
            fail += 1
            typer.echo(f"  ✗ {name}: {msg}  [{kind}]")
        _sys.stdout.flush()
    extra = f", {gone} olvidado(s)" if gone else ""
    typer.echo(f"\nReconstruidos {ok}, fallaron {fail}{extra}. "
               "Re-scan to check the rebuilt files as duplicates.")
    store.close()


@app.command()
def autotune(targets: list[str] = typer.Argument(..., help="Paths to probe (where the videos live)")):
    """Shows what workers/decode-workers auto-tune would choose for that storage
    (random-read latency probe), without scanning. Useful for HDD vs SSD/iSCSI/network share."""
    from dupdetect.pipeline.fullscan import collect_videos
    from dupdetect.tuning import autotune as _autotune
    files = collect_videos(targets, recursive=True)
    if not files:
        typer.echo("No videos found to probe.")
        raise typer.Exit()
    at = _autotune(files)
    typer.echo(at.note)


def _fmt_dup(cl: dict) -> str:
    """One-line, human notification for a duplicate cluster."""
    import os
    members = list((cl.get("evidence") or {}).keys()) or list(cl.get("discard", []))
    keep = cl.get("keep")
    title = os.path.basename(keep) if keep else (os.path.basename(members[0]) if members else "?")
    tail = (f"keep “{os.path.basename(keep)}”" if keep else "pick KEEP manually (review)")
    return f"  🔁 {len(members)} copies — {title}  ·  {tail}"


@app.command()
def watch(
    targets: list[str] = typer.Argument(..., help="Folders to watch (in place, no copying)"),
    db: Path = typer.Option(DEFAULT_DB, help="SQLite store path"),
    config: Path = typer.Option(None, help="thresholds.yaml (default: config/)"),
    interval: float = typer.Option(60.0, help="Seconds between reconcile cycles"),
    stable_s: float = typer.Option(
        15.0, help="A file must be unmodified this long before processing (avoid reading a file "
                   "still being copied)."),
    exact_first: bool = typer.Option(
        True, "--exact-first/--no-exact-first",
        help="On start, sweep byte-identical duplicates instantly (fast hash) before the deep "
             "background pass — gives value in minutes instead of hours."),
    workers: int = typer.Option(8, help="Workers for the initial exact sweep (hash, light I/O)."),
    once: bool = typer.Option(False, "--once", help="Run a single reconcile cycle and exit (cron)."),
    independent_scenes: bool = typer.Option(False, "--independent-scenes/--fast-scenes"),
):
    """Background watcher: keeps the DB current as files are added/changed/removed, matching new
    files incrementally and notifying on duplicates. Open the app only to REVIEW results — no full
    re-scan needed day to day (the affected cluster is re-ranked, so the best-quality KEEP stays
    current). A full re-verify is only for threshold/model changes."""
    import threading

    from dupdetect.pipeline.fullscan import exact_scan
    from dupdetect.watch import CycleResult, start_fs_events, watch_loop, watch_once

    th, store, embedder = _bootstrap(db, config)
    tlist = [str(t) for t in targets]
    typer.echo("Dupless Video watcher — usable right away.")
    typer.echo("  Exact (byte-identical) duplicates show immediately; re-encode/upgrade detection")
    typer.echo("  completes in the background as files are analyzed. Leave it running; open the app")
    typer.echo("  only to review. Ctrl+C to stop.\n")

    if exact_first and not once:
        typer.echo("Phase 1/2 — instant exact-duplicate sweep…")
        rep = exact_scan(tlist, store, th, workers=max(workers, 2), progress=True)
        typer.echo(f"  done: {len(rep.get('clusters', []))} byte-identical cluster(s) ready to review.\n")

    def _notify(clusters: list) -> None:
        typer.echo(f"\n🔔 {len(clusters)} duplicate cluster(s) just detected:")
        for cl in clusters:
            typer.echo(_fmt_dup(cl))

    def _cycle(res: "CycleResult") -> None:
        if res.indexed or res.removed or res.errors:
            from time import strftime
            typer.echo(f"[{strftime('%H:%M:%S')}] indexed={res.indexed} removed={res.removed} "
                       f"new_dups={len(res.dup_clusters)} errors={len(res.errors)}")

    def _detect(n: int) -> None:                           # fires BEFORE the slow indexing -> live UI
        from time import strftime
        typer.echo(f"[{strftime('%H:%M:%S')}] detecting={n} new file(s)…")

    if once:
        res = watch_once(tlist, store, embedder, th, on_duplicate=_notify,
                         stable_s=stable_s, independent_scenes=independent_scenes)
        _cycle(res)
        typer.echo(f"One cycle: indexed={res.indexed} removed={res.removed} "
                   f"dups={len(res.dup_clusters)}")
        return

    wake = threading.Event()
    stop_events = start_fs_events(tlist, wake)              # native FS events (watchdog) if available
    if stop_events:
        typer.echo("Filesystem events: subscribed — instant reaction; polling backs off when idle.")
    else:
        typer.echo("Filesystem events: watchdog not installed — polling only "
                   "(pip install watchdog for instant reaction).")
    typer.echo("Phase 2/2 — background watch (deep). Ctrl+C to stop.")
    try:
        watch_loop(tlist, store, embedder, th, interval=interval, on_duplicate=_notify,
                   on_cycle=_cycle, on_detect=_detect, stable_s=stable_s,
                   independent_scenes=independent_scenes, wake=wake)
    except KeyboardInterrupt:
        typer.echo("\nWatcher stopped. The DB is up to date through the last completed cycle.")
    finally:
        if stop_events:
            stop_events()


if __name__ == "__main__":
    app()
