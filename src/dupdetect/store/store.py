"""SQLite store + memmap embeddings access.

Incrementality lives here: has_fresh(path, st) decides whether to recompute.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from dupdetect.features.audio_fp import AUDIO_OK_COVERAGE
from dupdetect.models import AudioTrack, Probe, Quality, Record
from dupdetect.quality.color import ColorStats

SCHEMA = Path(__file__).with_name("schema.sql")


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
        with open(SCHEMA, "r", encoding="utf-8") as f:
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
        emb_file = Path(ep) if os.path.isabs(ep) else self.emb_dir / ep
        return emb_file.exists()

    def content_hash_if_unchanged(self, path: str, st: os.stat_result) -> Optional[str]:
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
        remux -c copy fixes it). If not given, inferred from the error message."""
        cat = category or classify_problem(error)
        self.conn.execute(
            """INSERT INTO problems (path, error, category, last_seen) VALUES (?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET
                   error=excluded.error, category=excluded.category, last_seen=excluded.last_seen""",
            (str(path), error, cat, time.time()),
        )
        self.conn.commit()

    def iter_problems(self) -> list[tuple[str, str]]:
        """(path, error) for ALL problems. Stable signature; use problems() for the category."""
        return [(r["path"], r["error"]) for r in
                self.conn.execute("SELECT path, error FROM problems ORDER BY path")]

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
        a deletion. Legacy records (audio_coverage NULL) do NOT appear -> re-scan to measure them."""
        return [(r["path"], r["audio_coverage"], r["duration_s"] or 0.0)
                for r in self.conn.execute(
                    "SELECT path, audio_coverage, duration_s FROM files "
                    "WHERE audio_coverage IS NOT NULL AND audio_coverage < ? "
                    "ORDER BY audio_coverage", (threshold,))]

    def prune_missing_problems(self) -> int:
        """Self-heal (§2): forgets problems whose file NO LONGER exists but whose parent folder IS
        reachable. Distinguishes 'deleted/moved' from 'volume unmounted': if even the parent
        directory is gone, the disk may be offline -> do NOT touch (don't delete 69 rows because L:
        is not mounted). Returns how many were forgotten."""
        gone = []
        for r in self.conn.execute("SELECT path FROM problems"):
            p = r["path"]
            try:
                parent = os.path.dirname(p) or "."
                if not os.path.exists(p) and os.path.isdir(parent):
                    gone.append(p)
            except OSError:
                pass                               # weird path (dead UNC): leave it alone to be safe
        for p in gone:
            self.conn.execute("DELETE FROM problems WHERE path=?", (p,))
        if gone:
            self.conn.commit()
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
    def load(self, path: str, with_embeddings: bool = True) -> Optional[Record]:
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

    def find_by_hash(self, content_hash: str, feature_version: str) -> str | None:
        """M4: is there a byte-identical file already indexed with the same feature_version?
        If so, cloning its features avoids re-decoding+embedding (the expensive step)."""
        row = self.conn.execute(
            "SELECT path FROM files WHERE content_hash = ? AND feature_version = ? LIMIT 1",
            (content_hash, feature_version),
        ).fetchone()
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
            emb_file = Path(ep) if os.path.isabs(ep) else self.emb_dir / ep
            emb = np.load(emb_file, mmap_mode=None)
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

    def all_matches(self) -> list[tuple[str, str, str]]:
        """(a_path, b_path, verdict) for ALL persisted pairs. Clusters are
        derived from here (global graph), not from a single run's yields -> so a
        re-scan does not leave stale clusters hanging."""
        return [(r["a_path"], r["b_path"], r["verdict"])
                for r in self.conn.execute("SELECT a_path, b_path, verdict FROM matches")]

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
            "  SELECT cluster_id FROM clusters GROUP BY cluster_id HAVING COUNT(*) < 2)")
        self.conn.commit()

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

    def record_deletion(self, path: str, dest: str, size: int) -> None:
        """Audits a deletion made from the UI (traceability / undo from Recycle Bin)."""
        self.conn.execute(
            "INSERT INTO deletions (path, dest, size, deleted_at) VALUES (?,?,?,?)",
            (path, dest, int(size or 0), time.time()),
        )
        self.conn.commit()

    def forget_file(self, path: str) -> None:
        """Forgets a deleted file: removes its `files` row, its matches and cluster membership,
        and deletes its `.npy` embeddings file. Prevents 'ghosts' after deletion."""
        row = self.conn.execute("SELECT emb_path FROM files WHERE path=?", (str(path),)).fetchone()
        if row and row["emb_path"]:
            ep = row["emb_path"]
            emb_file = Path(ep) if os.path.isabs(ep) else self.emb_dir / ep
            try:
                emb_file.unlink(missing_ok=True)
            except OSError:
                pass
        self.conn.execute("DELETE FROM files WHERE path=?", (str(path),))
        self.conn.execute("DELETE FROM matches WHERE a_path=? OR b_path=?", (str(path), str(path)))
        self.conn.execute("DELETE FROM clusters WHERE path=?", (str(path),))
        self.conn.commit()

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


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    """C2: stable ordering of a pair so it is evaluated and stored ONCE."""
    return (a, b) if a <= b else (b, a)
