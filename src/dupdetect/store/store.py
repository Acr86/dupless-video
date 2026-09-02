"""SQLite store + memmap embeddings access.

Incrementality lives here: has_fresh(path, st) decides whether to recompute.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from dupdetect.features.audio_fp import AUDIO_OK_COVERAGE
from dupdetect.models import AudioTrack, Probe, Quality, Record
from dupdetect.quality.color import ColorStats

SCHEMA = Path(__file__).with_name("schema.sql")

# A cluster needs at least this many members to be a duplicate GROUP; one copy left is 'resolved'.
# Shared so the DB prune (prune_singleton_clusters) and the optimistic UI prune (ui.model.remove_paths)
# can't silently disagree on the rule.
MIN_CLUSTER_MEMBERS = 2


class FingerprintStore:
    def __init__(self, db_path: str | Path, emb_dir: str | Path | None = None,
                 mtime_tol: float = 2.0, init_schema: bool = True):
        self.db_path = Path(db_path)
        self.mtime_tol = mtime_tol            # M2: SMB shares have ~1-2s mtime resolution
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.emb_dir = Path(emb_dir) if emb_dir else self.db_path.parent / "embeddings"
        self.emb_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout=30000")    # retry instead of erroring on a lock
        # WAL: readers don't block the writer and vice versa -> the UI can refresh and the watcher
        # can read while a scan writes, instead of starving (rollback-journal made the scan stall
        # until the watcher was stopped). Persistent per-DB; set on the rw handle (workers inherit).
        if init_schema:
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass                                       # e.g. DB on a network FS that rejects WAL
        # `init_schema=False`: read-only worker handles (Pass-2 parallel) skip schema writes
        # -> concurrent openers don't contend on the DB.
        if init_schema:
            self._init_schema()

    def _init_schema(self) -> None:
        with open(SCHEMA, encoding="utf-8") as f:
            self.conn.executescript(f.read())
        # Idempotent migration: DBs created before `frame_times` gain the column
        # (CREATE TABLE IF NOT EXISTS does not add columns to existing tables).
        try:
            self.conn.execute("ALTER TABLE files ADD COLUMN frame_times BLOB")
        except sqlite3.OperationalError:
            pass                                   # already exists
        # Same for `problems.category` (corrupt vs reindex) in old DBs.
        try:
            self.conn.execute(
                "ALTER TABLE problems ADD COLUMN category TEXT NOT NULL DEFAULT 'corrupt'")
        except sqlite3.OperationalError:
            pass                                   # already exists
        # Same for `problems.repair_note` (result of the last remux attempt).
        try:
            self.conn.execute("ALTER TABLE problems ADD COLUMN repair_note TEXT")
        except sqlite3.OperationalError:
            pass                                   # already exists
        # Same for `problems.mtime`/`size` (skip re-attempting a known-corrupt file that didn't change).
        for _col, _type in (("mtime", "REAL"), ("size", "INTEGER")):
            try:
                self.conn.execute(f"ALTER TABLE problems ADD COLUMN {_col} {_type}")
            except sqlite3.OperationalError:
                pass                               # already exists
        # Same for `files.audio_coverage` (audio coverage; NULL in old records -> 1.0).
        try:
            self.conn.execute("ALTER TABLE files ADD COLUMN audio_coverage REAL")
        except sqlite3.OperationalError:
            pass                                   # already exists
        # Same for `files.color_stats` (4 float32 blob: clip, cast, saturation, contrast; NULL -> neutral).
        try:
            self.conn.execute("ALTER TABLE files ADD COLUMN color_stats BLOB")
        except sqlite3.OperationalError:
            pass                                   # already exists
        # One-time v2-probe rollout (audio coverage): rows the v1 probe FLAGGED lose their cached
        # value and re-measure lazily with the v2 probe (5s windows, all audio streams) — the false
        # positives live below the threshold by definition. Rows above it keep their cache. Keyed by
        # PRAGMA user_version so a legitimately-low v2 measurement is never re-NULLed on reopen.
        # Deliberately NOT a COVERAGE_VERSION bump: that lives inside feature_version and would force
        # a full re-decode+embed of the library for an audio-only change.
        if self.conn.execute("PRAGMA user_version").fetchone()[0] < 1:
            self.conn.execute("UPDATE files SET audio_coverage=NULL WHERE audio_coverage < ?",
                              (AUDIO_OK_COVERAGE,))
            self.conn.execute("PRAGMA user_version = 1")
        self.conn.commit()
        self._reclassify_stale_problems()

    def _reclassify_stale_problems(self) -> None:
        """The `category` migration left ALL old rows as 'corrupt' (the default).
        Recomputes the category from the error via classify_problem (e.g. a 'timeout' becomes
        'reindex'), except rows that already have a remux attempt recorded (repair_note), whose
        category is final. Idempotent: only writes if something changes."""
        rows = list(self.conn.execute(
            "SELECT path, error, category FROM problems WHERE repair_note IS NULL"))
        changed = 0
        for r in rows:
            want = classify_problem(r["error"])
            if want != r["category"]:
                self.conn.execute("UPDATE problems SET category=? WHERE path=?",
                                  (want, r["path"]))
                changed += 1
        if changed:
            self.conn.commit()

    def _emb_file(self, ep: str) -> Path:
        """Resolve a stored `emb_path` to the on-disk `.npy`: a plain filename lives in `emb_dir`
        (current records), a legacy absolute path is used as-is. ONE rule for has_fresh /
        _row_to_record / forget_file — they must never disagree on where the embeddings are."""
        return Path(ep) if os.path.isabs(ep) else self.emb_dir / ep

    # ---- incrementalidad -------------------------------------------------
    def has_fresh(self, path: str, st: os.stat_result, feature_version: str) -> bool:
        """True if there is a record for `path` with matching mtime, size AND feature_version.
        C4: if model/fps/algorithm changed, feature_version differs -> recompute.
        M2: mtime tolerance is configurable (SMB shares have coarse mtime resolution)."""
        row = self.conn.execute(
            "SELECT mtime, size, feature_version, emb_path FROM files WHERE path = ?", (str(path),)
        ).fetchone()
        if row is None:
            return False
        if not (
            abs(row["mtime"] - st.st_mtime) <= self.mtime_tol
            and row["size"] == st.st_size
            and row["feature_version"] == feature_version
        ):
            return False
        # Self-heal: if the .npy embeddings file is gone (moved/deleted), NOT fresh
        # -> re-scan rebuilds it. Prevents records that would break in pass 2.
        ep = row["emb_path"]
        if not ep:
            return False
        return self._emb_file(ep).exists()

    def content_hash_if_unchanged(self, path: str, st: os.stat_result) -> str | None:
        """Stored content_hash if the file has NOT changed (same mtime+size), regardless of
        feature_version. 'Exact-only' mode: reuses the hash from already-indexed records (does not
        re-hash or clobber FULL records), and is incremental across runs."""
        row = self.conn.execute(
            "SELECT mtime, size, content_hash FROM files WHERE path = ?", (str(path),)
        ).fetchone()
        if row and abs(row["mtime"] - st.st_mtime) <= self.mtime_tol and row["size"] == st.st_size:
            return row["content_hash"]
        return None

    # ---- escritura -------------------------------------------------------
    def save_meta(self, path: str, mtime: float, size: int, content_hash: str,
                  probe: Probe, feature_version: str) -> None:
        """Saves ONLY metadata + hash ('exact-only' mode): no embeddings or .npy. emb_path
        stays NULL -> a FULL scan (different feature_version) re-indexes it entirely. NOT used
        on already fully-indexed files (the caller reuses their hash without touching them)."""
        tracks = json.dumps([t.__dict__ for t in probe.audio_tracks])
        self.conn.execute(
            """INSERT INTO files (path, mtime, size, content_hash, feature_version,
                   duration_s, width, height, vcodec, bitrate_kbps, audio_tracks, emb_path, indexed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET
                   mtime=excluded.mtime, size=excluded.size, content_hash=excluded.content_hash,
                   feature_version=excluded.feature_version, duration_s=excluded.duration_s,
                   width=excluded.width, height=excluded.height, vcodec=excluded.vcodec,
                   bitrate_kbps=excluded.bitrate_kbps, audio_tracks=excluded.audio_tracks,
                   emb_path=excluded.emb_path, indexed_at=excluded.indexed_at""",
            (str(path), mtime, size, content_hash, feature_version, probe.duration_s, probe.width,
             probe.height, probe.vcodec, probe.bitrate_kbps, tracks, None, time.time()),
        )
        self.conn.execute("DELETE FROM problems WHERE path = ?", (str(path),))
        self.conn.commit()

    def save(self, rec: Record, feature_version: str) -> None:
        emb_path = self.emb_dir / f"{_safe_key(rec.path)}.npy"
        np.save(emb_path, rec.embeddings.astype(np.float16))   # A1: fp16 on disk
        tracks = json.dumps([t.__dict__ for t in rec.probe.audio_tracks])
        wv = rec.window_vecs.astype(np.float32)
        wk = int(wv.shape[0]) if wv.ndim == 2 else 0
        emb_dim = int(rec.global_vec.shape[0])
        self.conn.execute(
            """
            INSERT INTO files (path, mtime, size, content_hash, feature_version,
                duration_s, width, height, vcodec, bitrate_kbps, audio_tracks,
                global_vec, window_vecs, window_k, emb_dim, emb_path, n_frames,
                audio_fp, scene_cuts, frame_times, lang_detected, cam_score,
                audio_coverage, color_stats, indexed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET
                mtime=excluded.mtime, size=excluded.size, content_hash=excluded.content_hash,
                feature_version=excluded.feature_version,
                duration_s=excluded.duration_s, width=excluded.width, height=excluded.height,
                vcodec=excluded.vcodec, bitrate_kbps=excluded.bitrate_kbps,
                audio_tracks=excluded.audio_tracks, global_vec=excluded.global_vec,
                window_vecs=excluded.window_vecs, window_k=excluded.window_k,
                emb_dim=excluded.emb_dim, emb_path=excluded.emb_path,
                n_frames=excluded.n_frames, audio_fp=excluded.audio_fp,
                scene_cuts=excluded.scene_cuts, frame_times=excluded.frame_times,
                lang_detected=excluded.lang_detected,
                cam_score=excluded.cam_score, audio_coverage=excluded.audio_coverage,
                color_stats=excluded.color_stats,
                indexed_at=excluded.indexed_at
            """,
            (
                rec.path, rec.mtime, rec.size, rec.content_hash, feature_version,
                rec.probe.duration_s, rec.probe.width, rec.probe.height,
                rec.probe.vcodec, rec.probe.bitrate_kbps, tracks,
                rec.global_vec.astype(np.float32).tobytes(),
                wv.tobytes(), wk, emb_dim, emb_path.name, rec.n_frames,   # relative: filename only
                rec.audio_fp.astype(np.uint32).tobytes(),   # C1: uint32
                rec.scene_cuts.astype(np.float32).tobytes(),
                np.asarray(rec.frame_times, dtype=np.float32).tobytes(),  # per-frame timestamps
                rec.quality.lang_detected, rec.quality.cam_score,
                rec.quality.audio_coverage,
                np.asarray(rec.quality.color.to_list(), dtype=np.float32).tobytes(),
                time.time(),
            ),
        )
        self.conn.execute("DELETE FROM problems WHERE path = ?", (rec.path,))  # no longer failing
        # Re-indexing invalidates old matches for this file: its features
        # changed, so previously computed pairs are stale. Pass 2
        # recomputes them fresh. Prevents "ghost" matches against features that no longer exist.
        self.conn.execute("DELETE FROM matches WHERE a_path = ? OR b_path = ?",
                           (rec.path, rec.path))
        self.conn.commit()

    # ---- problematic files ----------------------------------------------
    def save_problem(self, path: str, error: str, category: str | None = None) -> None:
        """Records a file that failed analysis. `category`: 'corrupt' (data lost
        -> delete/external tool) | 'reindex' (valid but missing index/slow seek -> a
        remux -c copy fixes it). If not given, inferred from the error message.
        Stamps the file's mtime+size so a later scan/sweep can SKIP it while unchanged
        (see has_unchanged_problem) instead of re-decoding a known-corrupt file every cycle."""
        cat = category or classify_problem(error)
        try:
            st = os.stat(path)
            mtime, size = st.st_mtime, st.st_size
        except OSError:                            # file vanished -> no stamp (will retry if it reappears)
            mtime, size = None, None
        self.conn.execute(
            """INSERT INTO problems (path, error, category, mtime, size, last_seen) VALUES (?,?,?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET
                   error=excluded.error, category=excluded.category,
                   mtime=excluded.mtime, size=excluded.size, last_seen=excluded.last_seen""",
            (str(path), error, cat, mtime, size, time.time()),
        )
        self.conn.commit()

    def has_unchanged_problem(self, path: str, st: os.stat_result) -> bool:
        """True if `path` already FAILED analysis and has NOT changed since (same mtime+size). Such a
        file was already attempted, so skip it: a corrupt file would just fail again, wasting a decode
        every sweep, and it would also never let the 'processed' count complete. A re-download / remux
        changes mtime|size -> the guard lifts and it's retried. A NULL-mtime row (file was gone at
        failure) never matches -> retried if it reappears."""
        row = self.conn.execute(
            "SELECT mtime, size FROM problems WHERE path = ?", (str(path),)).fetchone()
        if row is None or row["mtime"] is None:
            return False
        return abs(row["mtime"] - st.st_mtime) <= self.mtime_tol and row["size"] == st.st_size

    def problems(self, category: str | None = None) -> list[tuple[str, str, str, str | None]]:
        """(path, error, category, repair_note). Filters by 'corrupt'|'reindex' if given.
        `repair_note` is the result of the last remux attempt (None if never attempted);
        the UI shows it as the 'why' instead of the scan error when present."""
        sql = "SELECT path, error, category, repair_note FROM problems"
        args: tuple = ()
        if category is not None:
            sql += " WHERE category=?"; args = (category,)
        sql += " ORDER BY path"
        return [(r["path"], r["error"], r["category"], r["repair_note"])
                for r in self.conn.execute(sql, args)]

    def clear_problem(self, path: str) -> None:
        """Forgets a problem (e.g. after rebuilding its index): the next run treats it
        as a new file (mtime/size changed) and checks it for duplicates."""
        self.conn.execute("DELETE FROM problems WHERE path=?", (str(path),))
        self.conn.commit()

    def audio_warnings(self, threshold: float = AUDIO_OK_COVERAGE) -> list[tuple[str, float, float]]:
        """(path, audio_coverage, duration_s) for files with missing/truncated audio (coverage
        < `threshold`). For the 'Quality warnings' tab and to avoid losing the copy with audio in
        a deletion. Legacy records (audio_coverage NULL) do NOT appear -> re-scan to measure them.
        Files with a VALID user override ('Mark audio as OK') are excluded — the user already
        verified them; the measured value stays stored (restore = drop the override)."""
        rows = [(r["path"], r["audio_coverage"], r["duration_s"] or 0.0)
                for r in self.conn.execute(
                    "SELECT path, audio_coverage, duration_s FROM files "
                    "WHERE audio_coverage IS NOT NULL AND audio_coverage < ? "
                    "ORDER BY audio_coverage", (threshold,))]
        overridden = self.quality_overridden_paths("audio")
        return [t for t in rows if t[0] not in overridden]

    # ---- per-file quality overrides ("Mark audio as OK") ------------------
    def set_quality_override(self, path: str, kind: str = "audio") -> bool:
        """User correction of a FALSE quality warning (e.g. genuinely quiet content the probe reads
        as 'no audio'). Stamps the file's mtime+size so the override AUTO-EXPIRES when the file
        changes (a truly muted re-download must warn again — the problems.mtime/size pattern).
        Steers warnings and KEEP only, never a verdict (§0: audio_coverage does not enter
        decide_tree). Returns False when the file can't be statted (an override stamps a real file)."""
        try:
            st = os.stat(path)
        except OSError:
            return False
        self.conn.execute(
            """INSERT INTO quality_overrides (path, kind, mtime, size, created_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(path, kind) DO UPDATE SET
                   mtime=excluded.mtime, size=excluded.size, created_at=excluded.created_at""",
            (str(path), kind, st.st_mtime, st.st_size, time.time()))
        self.conn.commit()
        return True

    def clear_quality_override(self, path: str, kind: str = "audio") -> None:
        """Drops the override -> the MEASURED value (still stored in files.audio_coverage) rules
        again. Also the first step of a re-check: a fresh measurement supersedes the user's mark."""
        self.conn.execute("DELETE FROM quality_overrides WHERE path=? AND kind=?",
                          (str(path), kind))
        self.conn.commit()

    def _override_valid(self, row, path: str) -> bool:
        """An override counts only while the file is UNCHANGED since it was set (same mtime+size,
        within the store's tolerance). Changed or unstattable -> expired (fail-safe: warn again)."""
        try:
            st = os.stat(path)
        except OSError:
            return False
        return abs(row["mtime"] - st.st_mtime) <= self.mtime_tol and row["size"] == st.st_size

    def has_quality_override(self, path: str, kind: str = "audio") -> bool:
        """True iff a currently-VALID override exists for `path` (see _override_valid)."""
        row = self.conn.execute(
            "SELECT mtime, size FROM quality_overrides WHERE path=? AND kind=?",
            (str(path), kind)).fetchone()
        return row is not None and self._override_valid(row, str(path))

    def quality_overridden_paths(self, kind: str = "audio") -> set[str]:
        """Paths with a currently-VALID override. O(overrides) stats — the table holds only the
        files the user marked, not the library."""
        return {r["path"] for r in self.conn.execute(
                    "SELECT path, mtime, size FROM quality_overrides WHERE kind=?", (kind,))
                if self._override_valid(r, r["path"])}

    def prune_missing_problems(self, *, exists=os.path.exists, isdir=os.path.isdir,
                               ismount=None) -> int:
        """Self-heal (§2): forgets problems whose file is REALLY gone, with the SAME volume+mount-aware
        guard as prune_missing_files (shared decision core, `_missing_on_online_volume`): a whole
        deleted SUBFOLDER on an ONLINE volume is cleaned (Mode B), while an offline drive or a
        disconnected nested mount/junction is never mistaken for a deletion. The old parent-dir guard
        ('parent isdir => truly deleted') skipped Mode B — the deleted folder IS the parent — so
        problems of deleted folders lingered forever in the Problems tab. `exists`/`isdir`/`ismount`
        injectable for deterministic tests (§0). Returns how many were forgotten."""
        paths = [r["path"] for r in self.conn.execute("SELECT path FROM problems")]
        gone = _missing_on_online_volume(paths, exists, isdir, ismount or _is_mount)
        with self.conn:                            # one transaction, not one commit per row
            self.conn.executemany("DELETE FROM problems WHERE path=?", [(p,) for p in gone])
        return len(gone)

    def prune_missing_files(self, *, exists=os.path.exists, isdir=os.path.isdir,
                            ismount=None) -> int:
        """Self-heal (§2) for the DUPLICATES list — the UI counterpart of prune_missing_problems:
        forget indexed files whose bytes are gone from disk, with a VOLUME + MOUNT-AWARE guard so an
        offline drive or a disconnected nested mount/junction is NEVER mistaken for a mass deletion
        (§0/§2), while a whole deleted SUBFOLDER on an ONLINE volume IS cleaned — the 'I deleted folders
        but still see them' bug that orphan_paths' watched-ROOT guard misses (Mode B).

        Two guards (both fail-SAFE — on doubt, keep the record; the file on disk is never touched and a
        re-scan rebuilds a wrongly-forgotten record):
          1. VOLUME fast-skip: group by volume anchor (`_volume_root`); one `exists(anchor)` probe per
             drive/share. An unmounted drive (or an unknown/degenerate anchor) -> skip ALL its files.
          2. MOUNT-AWARE per-file (`_real_deletion`): on a reachable volume, a gone file counts as a
             real deletion only if NO disconnected mount/junction sits between it and the nearest present
             directory; an offline sub-volume (its boundary reads as a mount/reparse point) is kept.

        `exists`/`isdir`/`ismount` are injectable so the decision core is deterministic and unit-testable
        without real drives/junctions (§0). Does NOT rebuild clusters (the caller owns that, like
        actions.delete_files) and never touches the `problems` table. Returns how many were forgotten."""
        gone = _missing_on_online_volume(self.all_paths(), exists, isdir, ismount or _is_mount)
        with self.conn:                                   # ONE transaction: a folder prune is N files,
            for p in gone:                                # N commits (fsyncs) would pay per row
                self.forget_file(p, commit=False)         # row + matches + cluster + .npy
        return len(gone)

    def mark_repair_failed(self, path: str, kind: str, reason: str) -> None:
        """Persists the result of a FAILED remux attempt (previously the CLI only printed it
        -> the file would retry forever without ever explaining why). `kind='timeout'`
        stays 'reindex' (retryable: on HDD/Storage Space a timeout is usually disk contention,
        not corruption); any other `kind` becomes 'corrupt' (unrecoverable)."""
        if kind == "timeout":
            note = f"last attempt: timeout — {reason} · still repairable (retry with free disk)"
            self.conn.execute(
                "UPDATE problems SET repair_note=?, last_seen=? WHERE path=?",
                (note, time.time(), str(path)))
        else:
            self.conn.execute(
                "UPDATE problems SET category='corrupt', repair_note=?, last_seen=? WHERE path=?",
                (f"remux failed: {reason}", time.time(), str(path)))
        self.conn.commit()

    # ---- lectura ---------------------------------------------------------
    def load(self, path: str, with_embeddings: bool = True) -> Record | None:
        row = self.conn.execute(
            "SELECT * FROM files WHERE path = ?", (str(path),)
        ).fetchone()
        return self._row_to_record(row, with_embeddings) if row else None

    def iter_records(self, with_embeddings: bool = False) -> Iterator[Record]:
        for row in self.conn.execute("SELECT * FROM files"):
            yield self._row_to_record(row, with_embeddings)

    def all_global_vecs(self) -> tuple[list[str], np.ndarray]:
        """For building the coarse FAISS index (mean-pool, first pass). SKIPS records without
        embeddings (LITE / exact-only have global_vec NULL) -> they can't be video-matched; a
        mixed DB (full + exact-only) must not crash here (§2: skip, don't crash)."""
        paths, vecs = [], []
        for row in self.conn.execute("SELECT path, global_vec FROM files"):
            gv = row["global_vec"]
            if not gv:                                 # LITE/exact-only record: no embeddings
                continue
            paths.append(row["path"])
            vecs.append(np.frombuffer(gv, dtype=np.float32))
        arr = np.vstack(vecs) if vecs else np.empty((0, 0), dtype=np.float32)
        return paths, arr

    def all_window_vecs(self) -> tuple[list[str], np.ndarray]:
        """A2: multi-vector. Returns (paths_repetidos, [sum_K, D]); each window-vec
        maps to its file. For the second FAISS index and candidate union."""
        owners, vecs = [], []
        for row in self.conn.execute(
            "SELECT path, window_vecs, window_k, emb_dim FROM files"
        ):
            wk, d = row["window_k"] or 0, row["emb_dim"] or 0
            if not (wk and d):
                continue
            wv = np.frombuffer(row["window_vecs"], dtype=np.float32).reshape(wk, d)
            owners.extend([row["path"]] * wk)
            vecs.append(wv)
        arr = np.vstack(vecs) if vecs else np.empty((0, 0), dtype=np.float32)
        return owners, arr

    def find_by_duration(self, duration_s: float, tol: float) -> list[str]:
        """A2: safety net — paths with duration within ±tol (fraction)."""
        lo, hi = duration_s * (1 - tol), duration_s * (1 + tol)
        rows = self.conn.execute(
            "SELECT path FROM files WHERE duration_s BETWEEN ? AND ?", (lo, hi)
        ).fetchall()
        return [r["path"] for r in rows]

    def find_by_hash(self, content_hash: str, feature_version: str,
                     size: int | None = None) -> str | None:
        """M4: is there a byte-identical file already indexed with the same feature_version?
        If so, cloning its features avoids re-decoding+embedding (the expensive step).
        `size` narrows to true byte-identity candidates: the hash is SAMPLED (head|mid|tail), so
        equal size is required to claim identity — the same standard the T0 tier applies (§0)."""
        sql = "SELECT path FROM files WHERE content_hash = ? AND feature_version = ?"
        args: list = [content_hash, feature_version]
        if size is not None:
            sql += " AND size = ?"
            args.append(size)
        row = self.conn.execute(sql + " LIMIT 1", args).fetchone()
        return row["path"] if row else None

    def _row_to_record(self, row: sqlite3.Row, with_embeddings: bool) -> Record:
        tracks = [AudioTrack(**t) for t in json.loads(row["audio_tracks"] or "[]")]
        probe = Probe(
            duration_s=row["duration_s"], width=row["width"], height=row["height"],
            vcodec=row["vcodec"], bitrate_kbps=row["bitrate_kbps"], audio_tracks=tracks,
        )
        # Tolerant of LITE records ('exact-only' mode): emb_path and the BLOBs for
        # embeddings/audio/scenes may be NULL -> empty arrays, no crash.
        def _buf(col, dtype):
            b = row[col]
            return np.frombuffer(b, dtype=dtype) if b else np.empty(0, dtype=dtype)

        ep = row["emb_path"]
        if with_embeddings and ep:
            emb = np.load(self._emb_file(ep), mmap_mode=None)
        else:
            emb = np.empty((0, 0))
        d = row["emb_dim"] or 0
        wk = row["window_k"] or 0
        wv = (np.frombuffer(row["window_vecs"], dtype=np.float32).reshape(wk, d)
              if wk and d else np.empty((0, d), dtype=np.float32))
        return Record(
            path=row["path"], mtime=row["mtime"], size=row["size"],
            probe=probe, content_hash=row["content_hash"],
            global_vec=_buf("global_vec", np.float32),
            window_vecs=wv,
            embeddings=emb,                                            # A1: fp16
            audio_fp=_buf("audio_fp", np.uint32),                     # C1: uint32
            scene_cuts=_buf("scene_cuts", np.float32),
            frame_times=_buf("frame_times", np.float32),
            quality=Quality(
                lang_detected=row["lang_detected"], cam_score=row["cam_score"] or 0.0,
                audio_coverage=(row["audio_coverage"] if row["audio_coverage"] is not None else 1.0),
                color=ColorStats.from_list(_buf("color_stats", np.float32)),
            ),
        )

    def set_audio_fp(self, path: str, fp: np.ndarray) -> None:
        """Persists a computed audio fingerprint for an already-indexed file. Used by the
        ON-DEMAND fingerprinting in Pass-2: the fp is computed only for candidate pairs, then
        cached here so future runs and cluster ranking reuse it without recomputing."""
        self.conn.execute("UPDATE files SET audio_fp=? WHERE path=?",
                          (fp.astype(np.uint32).tobytes(), str(path)))
        self.conn.commit()

    def set_audio_coverage(self, path: str, coverage: float) -> None:
        """Persists a computed whole-file audio coverage. Like set_audio_fp, this backs the
        ON-DEMAND coverage: NULL means 'not computed', filled here for cluster members (Standard)
        or all files (Deep) so future runs reuse it (incremental — no recompute)."""
        self.conn.execute("UPDATE files SET audio_coverage=? WHERE path=?",
                          (float(coverage), str(path)))
        self.conn.commit()

    def set_lang(self, path: str, lang: str) -> None:
        """Persists a detected language for an already-indexed file. Used by DEFERRED language
        detection (whisper runs only for cluster members at rank time, not in Pass-1, since
        lang_detected is consumed only for KEEP selection -> most unique files never need it)."""
        self.conn.execute("UPDATE files SET lang_detected=? WHERE path=?", (lang, str(path)))
        self.conn.commit()

    # ---- matches / clusters ---------------------------------------------
    def save_match(self, a: str, b: str, verdict: str, conf: float, reason: str,
                   ad_offset_s: float | None = None,
                   audio_json: str = "", video_json: str = "", scenes_json: str = "") -> None:
        # C2: canonicalize (a_path <= b_path) for a single row per pair.
        # C3: ad_offset_s arrives as "b relative to a" in the CALLER's order. If
        # canonicalization reverses the order, negate the sign so it reads
        # "b_canon relative to a_canon" -> preserves WHICH copy carries the offset
        # (prepended ads), which is what rank_cluster needs.
        if a <= b:
            ca, cb, coff = a, b, ad_offset_s
        else:
            ca, cb, coff = b, a, (-ad_offset_s if ad_offset_s is not None else None)
        self.conn.execute(
            """INSERT INTO matches (a_path,b_path,verdict,confidence,reason,ad_offset_s,
                   audio_json,video_json,scenes_json,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(a_path,b_path) DO UPDATE SET
                   verdict=excluded.verdict, confidence=excluded.confidence,
                   reason=excluded.reason, ad_offset_s=excluded.ad_offset_s,
                   audio_json=excluded.audio_json, video_json=excluded.video_json,
                   scenes_json=excluded.scenes_json, created_at=excluded.created_at""",
            (ca, cb, verdict, conf, reason, coff,
             audio_json, video_json, scenes_json, time.time()),
        )
        self.conn.commit()

    def has_match(self, a: str, b: str) -> bool:
        """Does a match (any verdict) already exist for this pair? To avoid overwriting a
        CONTENT verdict with a name-based one (content takes precedence / can veto)."""
        ca, cb = canonical_pair(a, b)
        return self.conn.execute(
            "SELECT 1 FROM matches WHERE a_path=? AND b_path=? LIMIT 1", (ca, cb)
        ).fetchone() is not None

    def all_paths(self) -> list[str]:
        """All indexed paths (full or lite). For name-based grouping."""
        return [r["path"] for r in self.conn.execute("SELECT path FROM files")]

    def relocate_path(self, old: str, new: str) -> None:
        """Rename a file's path old->new across the store, KEEPING its analysis (the features and their
        .npy survive — emb_path is unchanged). For a MOVED or RE-MOUNTED file (same content, new mount
        prefix like M:\\ -> /share): the ONE record follows the active path instead of being orphaned —
        an orphan under the same content_hash+size would be a T0 'duplicate of itself'. The file's own
        MATCHES are dropped (its duplicate edges are re-derived cheaply by the next Pass-2 in the same
        scan); clusters/problems/overrides/feedback are re-pointed. OR REPLACE guards the rare case where
        `new` already carried a (stale) row."""
        if old == new:
            return
        c = self.conn
        c.execute("UPDATE OR REPLACE files SET path=? WHERE path=?", (new, old))
        c.execute("DELETE FROM matches WHERE a_path=? OR b_path=?", (old, old))
        c.execute("DELETE FROM clusters WHERE path=?", (old,))
        c.execute("UPDATE OR REPLACE problems SET path=? WHERE path=?", (new, old))
        c.execute("UPDATE OR REPLACE quality_overrides SET path=? WHERE path=?", (new, old))
        c.execute("UPDATE OR REPLACE feedback SET a_path=? WHERE a_path=?", (new, old))
        c.execute("UPDATE OR REPLACE feedback SET b_path=? WHERE b_path=?", (new, old))
        c.commit()

    def delete_match(self, a: str, b: str, commit: bool = True) -> None:
        """Remove ONE pair's row (canonicalized). Used when a pair RE-DECIDES to DIFFERENT: DIFFERENT is
        never kept in `matches` (schema), so a prior duplicate/review row must be dropped, not left to
        keep fusing a cluster. Idempotent (deleting an absent pair is a no-op)."""
        ca, cb = canonical_pair(a, b)
        self.conn.execute("DELETE FROM matches WHERE a_path=? AND b_path=?", (ca, cb))
        if commit:
            self.conn.commit()

    def delete_matches(self, pairs) -> int:
        """Batch-delete pairs (each canonicalized) in ONE transaction. Returns how many rows were
        removed. For the scan's delete-on-DIFFERENT sweep, where per-pair commits would be far too many."""
        n = 0
        for a, b in pairs:
            ca, cb = canonical_pair(a, b)
            n += self.conn.execute("DELETE FROM matches WHERE a_path=? AND b_path=?", (ca, cb)).rowcount
        self.conn.commit()
        return n

    def matched_pairs(self) -> set[tuple[str, str]]:
        """Set of canonical (a_path, b_path) that currently HAVE a row. Small (DIFFERENT is not kept),
        so Pass-2 can preload it and delete-on-DIFFERENT only the few pairs that actually flipped —
        never touching the millions of never-matched candidate pairs."""
        return {(r["a_path"], r["b_path"])
                for r in self.conn.execute("SELECT a_path, b_path FROM matches")}

    def prune_matches_by_verdict(self, verdict: str) -> int:
        """Delete every row with this verdict (one transaction). Maintenance for DIFFERENT rows that an
        older re-decide upserted in place instead of dropping — they bloat every clusters/verdict read
        without ever being used (DIFFERENT never clusters). Returns rows removed."""
        cur = self.conn.execute("DELETE FROM matches WHERE verdict=?", (verdict,))
        self.conn.commit()
        return cur.rowcount

    def all_matches(self) -> list[tuple[str, str, str]]:
        """(a_path, b_path, verdict) for ALL persisted pairs. Clusters are
        derived from here (global graph), not from a single run's yields -> so a
        re-scan does not leave stale clusters hanging."""
        return [(r["a_path"], r["b_path"], r["verdict"])
                for r in self.conn.execute("SELECT a_path, b_path, verdict FROM matches")]

    # ---- evaluated-pairs ledger (incremental Pass-2) -------------------
    def evaluated_pairs_load(self, fingerprint: str) -> set[str]:
        """Pair-hashes already ALIGNED under THIS fingerprint (feature_version + thresholds), so the
        next scan can skip re-aligning them. Prunes rows from any OTHER fingerprint first: an algorithm
        or θ change invalidates them (a pair that was DIFFERENT could become a match under looser θ)."""
        self.conn.execute("DELETE FROM evaluated_pairs WHERE fingerprint != ?", (fingerprint,))
        self.conn.commit()
        return {r[0] for r in self.conn.execute(
            "SELECT pair_hash FROM evaluated_pairs WHERE fingerprint = ?", (fingerprint,))}

    def evaluated_pairs_add(self, pair_hashes, fingerprint: str) -> None:
        """Record aligned pairs (match OR DIFFERENT) under `fingerprint`. Idempotent per pair_hash."""
        rows = [(h, fingerprint) for h in pair_hashes]
        if not rows:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO evaluated_pairs(pair_hash, fingerprint) VALUES (?, ?)", rows)
        self.conn.commit()

    def clear_clusters(self) -> None:
        """Clears the clusters table. Rebuilt entirely from `matches` on each
        full-scan: clusters are a DERIVED VIEW, not accumulated state. Without this,
        a path that changes cluster_id between runs would leave its old row (the same
        file would appear in two clusters)."""
        self.conn.execute("DELETE FROM clusters")
        self.conn.commit()

    def prune_singleton_clusters(self) -> None:
        """After deleting members, removes clusters left with <2 files: a cluster of
        1 is NO LONGER a duplicate group (considered 'resolved'). The remaining file (the keep)
        stays in `files` and on disk; it simply stops appearing in the duplicates list."""
        self.conn.execute(
            "DELETE FROM clusters WHERE cluster_id IN ("
            "  SELECT cluster_id FROM clusters GROUP BY cluster_id HAVING COUNT(*) < ?)",
            (MIN_CLUSTER_MEMBERS,))
        self.conn.commit()

    def replace_clusters(self, rows) -> None:
        """ATOMICALLY replace the ENTIRE clusters table in ONE transaction: a rebuild is
        all-or-nothing, so a CONCURRENT rebuild (e.g. the watcher while a scan runs) can't interleave
        its per-row writes with ours and FUSE distinct match-components under a shared cluster_id.
        `rows`: iterable of (cluster_id, path, is_keep, rank_reason). Combined with content-derived
        (stable) cluster ids, two independent rebuilds agree on ids and unrelated components never
        collide. Replaces the old clear_clusters()+per-row save_cluster() (each its own commit)."""
        rows = [(int(cid), str(p), 1 if keep else 0, reason or "") for cid, p, keep, reason in rows]
        with self.conn:                                    # single BEGIN..COMMIT (rolls back on error)
            self.conn.execute("DELETE FROM clusters")
            self.conn.executemany(
                "INSERT INTO clusters (cluster_id, path, is_keep, rank_reason) VALUES (?,?,?,?)", rows)

    def save_cluster(self, cluster_id: int, path: str, is_keep: bool,
                     rank_reason: str = "") -> None:
        """A5: persists the cluster/keep decision — source of truth for actions."""
        self.conn.execute(
            """INSERT INTO clusters (cluster_id, path, is_keep, rank_reason)
               VALUES (?,?,?,?)
               ON CONFLICT(cluster_id, path) DO UPDATE SET
                   is_keep=excluded.is_keep, rank_reason=excluded.rank_reason""",
            (cluster_id, path, 1 if is_keep else 0, rank_reason),
        )
        self.conn.commit()

    def set_keep(self, cluster_id: int, keep_path: str) -> None:
        """Manual UI action: marks `keep_path` as the KEEP (★) for the cluster and un-marks
        the rest. Does NOT delete anything — only changes which copy is kept (the others become
        selectable for deletion)."""
        self.conn.execute("UPDATE clusters SET is_keep=0 WHERE cluster_id=?", (cluster_id,))
        self.conn.execute("UPDATE clusters SET is_keep=1 WHERE cluster_id=? AND path=?",
                          (cluster_id, str(keep_path)))
        self.conn.commit()

    # ---- feedback / UI actions ----------------------------------
    def save_feedback(self, a: str, b: str, label: str, note: str = "") -> None:
        """User correction label for a pair (canonicalized a<=b). Does NOT retrain
        the network: feeds threshold recalibration and view overrides. label =
        'same' | 'different'."""
        ca, cb = (a, b) if a <= b else (b, a)
        self.conn.execute(
            """INSERT INTO feedback (a_path, b_path, label, note, created_at) VALUES (?,?,?,?,?)
               ON CONFLICT(a_path, b_path) DO UPDATE SET
                   label=excluded.label, note=excluded.note, created_at=excluded.created_at""",
            (ca, cb, label, note, time.time()),
        )
        self.conn.commit()

    def iter_feedback(self) -> list[tuple[str, str, str]]:
        """(a_path, b_path, label) for all user corrections."""
        return [(r["a_path"], r["b_path"], r["label"])
                for r in self.conn.execute("SELECT a_path, b_path, label FROM feedback")]

    def vetoed_pairs(self) -> set[tuple[str, str]]:
        """Canonical pairs the USER marked 'different' — a human veto that outranks the content
        verdict: cluster building must NEVER union them, so a corrected group stays corrected across
        re-scans (a re-scan re-aligns and would otherwise re-declare them duplicates). Marking the
        pair 'same' again upserts the label and lifts the veto."""
        return {(r["a_path"], r["b_path"]) for r in self.conn.execute(
            "SELECT a_path, b_path FROM feedback WHERE label = 'different'")}

    def record_deletion(self, path: str, dest: str, size: int) -> None:
        """Audits a deletion made from the UI (traceability / undo from Recycle Bin)."""
        self.conn.execute(
            "INSERT INTO deletions (path, dest, size, deleted_at) VALUES (?,?,?,?)",
            (path, dest, int(size or 0), time.time()),
        )
        self.conn.commit()

    def forget_file(self, path: str, commit: bool = True) -> bool:
        """Forgets a deleted file: removes its `files` row, its matches and cluster membership,
        and deletes its `.npy` embeddings file. Prevents 'ghosts' after deletion. Returns True if a
        `files` row actually existed (let the event-drain count only real removals and ignore a path
        whose stored form didn't match — it falls through to the periodic full sweep).
        `commit=False` lets a batch caller (prune_missing_files) fold N forgets into ONE transaction."""
        row = self.conn.execute("SELECT emb_path FROM files WHERE path=?", (str(path),)).fetchone()
        if row and row["emb_path"]:
            try:
                self._emb_file(row["emb_path"]).unlink(missing_ok=True)
            except OSError:
                pass
        self.conn.execute("DELETE FROM files WHERE path=?", (str(path),))
        self.conn.execute("DELETE FROM matches WHERE a_path=? OR b_path=?", (str(path), str(path)))
        self.conn.execute("DELETE FROM clusters WHERE path=?", (str(path),))
        self.conn.execute("DELETE FROM quality_overrides WHERE path=?", (str(path),))
        if commit:
            self.conn.commit()
        return row is not None

    def close(self) -> None:
        self.conn.close()


# Tokens indicating a VALID file with a bad/missing index (seek so slow it
# aborts on timeout) -> a remux -c copy fixes it. Everything else is treated as corrupt
# (missing moov, truncated, lost data): no index to rebuild.
_REINDEX_TOKENS = ("timeout", "timed out")


def classify_problem(error: str | None) -> str:
    """'reindex' (fixable with remux) if the file aborted due to seek/decode too
    slow; 'corrupt' in all other cases (data lost)."""
    e = (error or "").lower()
    return "reindex" if any(t in e for t in _REINDEX_TOKENS) else "corrupt"


def _safe_key(path: str) -> str:
    import hashlib
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def canonical_path(p: str) -> str:
    """Canonical key for a file path: the OS-native form via pathlib, so the SAME file maps to ONE
    record no matter how the path was built. On Windows this folds a '/'+'\\' MIX (e.g. a watchdog
    event path 'L:/Media\\sub\\x.mp4' vs a scan's pathlib 'L:\\Media\\sub\\x.mp4') to a single string;
    without it the two are stored as DIFFERENT files -> phantom duplicates of the same video.
    Applied at the PRODUCERS (watchdog events; collect_videos is already pathlib-canonical) rather than
    in the store, so synthetic '/'-paths in tests are stored verbatim and stay portable."""
    return str(Path(p))


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    """C2: stable ordering of a pair so it is evaluated and stored ONCE."""
    return (a, b) if a <= b else (b, a)


# --------------------------------------------------------------------------- file-existence self-heal
# Helpers for prune_missing_files (the UI duplicates-list §2 self-heal). Kept module-level + injectable
# so the decision core is deterministic and unit-testable without real drives/junctions (§0). The forms
# rejected here were ADVERSARIALLY found to otherwise mass-forget records on offline storage.

def _volume_root(path: str) -> str:
    """The mountable VOLUME a path lives on, as a fail-safe anchor: a Windows DRIVE root ('L:\\') or a
    UNC SHARE root ('\\\\srv\\share\\'), or POSIX '/'. Returns '' (UNKNOWN -> never touched) for forms
    that would probe the WRONG thing and mass-forget: a relative/unanchored path, a drive-RELATIVE
    anchor ('C:foo' -> 'C:', no root), or a degenerate bare-backslash 'UNC' ('\\x' / '/x' on Windows,
    whose anchor '\\' os.path.exists resolves to the current drive). POSIX '/' is the real, single
    volume root (always present; Path normalizes it to '\\' on Windows, so it never means POSIX there)
    — rejecting it would make the whole sweep a silent no-op on Linux; nested-mount protection there
    leans on the per-file `_real_deletion` walk (os.path.ismount). Detect, don't trust."""
    a = Path(path).anchor
    if not a or a == "\\":
        return ""                                         # relative, or bare-backslash (degenerate UNC)
    if len(a) >= 2 and a[1] == ":" and not a.endswith(("\\", "/")):
        return ""                                         # drive-relative 'C:foo' -> 'C:' (no root)
    return a


def _volume_reachable(anchor: str, exists) -> bool:
    """Reachable iff the anchor is a KNOWN form AND present. Empty/unknown anchor is forced unreachable
    so its files are never forgotten (fail-safe). One cheap probe per distinct drive/share."""
    return bool(anchor) and bool(exists(anchor))


def _is_mount(path: str) -> bool:
    """True if `path` is a mount point or a (possibly disconnected) Windows junction/reparse point.
    Never raises. is_junction reads the reparse ENTRY (in the still-online parent), so it flags a
    junction even when its TARGET is offline -> that subtree is an unmount, NOT a deletion (§2)."""
    try:
        if os.path.ismount(path):
            return True
    except OSError:
        pass
    try:
        return Path(path).is_junction()                   # Python 3.12+
    except (OSError, AttributeError):
        return False


def _missing_on_online_volume(paths, exists, isdir, ismount) -> list[str]:
    """Shared decision core of prune_missing_files / prune_missing_problems (§2 self-heals): of
    `paths`, those whose bytes are REALLY gone — the volume is online AND no disconnected
    mount/junction sits between the path and the nearest present directory. Fail-SAFE both ways:
    an unknown/degenerate anchor or an offline volume keeps ALL its paths (an unmount must never
    read as a mass deletion), and any OSError on a probe keeps that path."""
    by_anchor: dict[str, list[str]] = {}
    for p in paths:
        by_anchor.setdefault(_volume_root(p), []).append(p)
    gone: list[str] = []
    for anchor, ps in by_anchor.items():
        if not _volume_reachable(anchor, exists):         # unknown form or offline drive -> keep all (§2)
            continue
        for p in ps:
            try:
                if not exists(p) and _real_deletion(p, anchor, isdir, ismount):
                    gone.append(p)                        # gone, volume online, no offline mount above
            except OSError:
                pass                                      # dead UNC / weird path -> leave it (be safe)
    return gone


def _real_deletion(path: str, anchor: str, isdir, ismount) -> bool:
    """On a reachable `anchor` volume, is a gone `path` a REAL deletion (forget it)? True only if every
    still-missing ancestor between it and the nearest present directory is a PLAIN deleted folder —
    never a disconnected mount/junction. A missing level that is a mount/reparse point is an OFFLINE
    sub-volume whose whole subtree must be KEPT (§2); a plain missing folder was deleted on an online
    volume (Mode B) -> forget. Walks up at most to `anchor`."""
    cur = os.path.dirname(path)
    while cur and cur != anchor and not isdir(cur):
        if ismount(cur):
            return False                                  # offline mount/junction above -> keep subtree
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break                                         # can't ascend further (defensive)
        cur = nxt
    return True
