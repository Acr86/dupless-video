-- Fingerprint store. Metadata + global_vec in SQLite; per-frame embeddings in a
-- separate memmap (data/embeddings/<id>.npy) referenced by emb_path.

CREATE TABLE IF NOT EXISTS files (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT UNIQUE NOT NULL,
    mtime         REAL NOT NULL,
    size          INTEGER NOT NULL,
    content_hash  TEXT NOT NULL,          -- xxhash head|mid|tail (T0, SAMPLED hash)

    -- C4: invalidates the cache when the model/fps/feature algorithm changes.
    -- has_fresh requires a match; otherwise it recomputes even if mtime+size are unchanged.
    feature_version TEXT NOT NULL,

    duration_s    REAL,
    width         INTEGER,
    height        INTEGER,
    vcodec        TEXT,
    bitrate_kbps  INTEGER,
    audio_tracks  TEXT,                    -- JSON: [{index,lang_tag,codec,channels}]

    global_vec    BLOB,                    -- float32 [D] (np.tobytes)
    window_vecs   BLOB,                    -- A2: float32 [K*D] (np.tobytes; K in window_k)
    window_k      INTEGER,                 -- number of descriptors per window
    emb_dim       INTEGER,                 -- D, to reconstruct window_vecs/embeddings
    emb_path      TEXT,                    -- A1: memmap [N, D] fp16 on disk
    n_frames      INTEGER,
    audio_fp      BLOB,                    -- C1: uint32 [M] (Chromaprint raw) — NOT float32
    scene_cuts    BLOB,                    -- float32 [K] timestamps
    frame_times   BLOB,                    -- float32 [N] real ts per frame (align by TIME)

    -- quality (independent of identity). C3: ad_offset does NOT live here — it is a
    -- property of the PAIR (depends on what you align against), read from matches.
    lang_detected TEXT,
    cam_score     REAL,
    audio_coverage REAL,                   -- [0..1] audio coverage: <1 = missing/truncated audio

    indexed_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_size ON files(size);
CREATE INDEX IF NOT EXISTS idx_files_duration ON files(duration_s);  -- A2: duration blocking

-- Persisted match results (for reports and to avoid re-evaluating pairs).
CREATE TABLE IF NOT EXISTS matches (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    a_path         TEXT NOT NULL,          -- C2: canonicalized (a_path < b_path)
    b_path         TEXT NOT NULL,
    verdict        TEXT NOT NULL,
    confidence     REAL NOT NULL,
    reason         TEXT,
    ad_offset_s    REAL,                   -- C3: align offset (property of the pair)
    audio_json     TEXT,                   -- AlignResult serialized
    video_json     TEXT,
    scenes_json    TEXT,
    created_at     REAL NOT NULL,
    UNIQUE(a_path, b_path)
);

-- Full-scan clusters (materialized union-find).
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id     INTEGER NOT NULL,
    path           TEXT NOT NULL,
    is_keep        INTEGER DEFAULT 0,      -- 1 = copy recommended to keep
    rank_reason    TEXT,                   -- why it is keep / why it is discard
    PRIMARY KEY (cluster_id, path)
);

-- Files that failed analysis (corrupt, no moov atom, truncated...) + the real
-- ffprobe/decode error. To decide whether to rebuild the index or re-encode.
-- One row per path; deleted when it analyzes OK on a later run.
-- category: 'corrupt' (data lost -> delete/external tool) vs 'reindex'
-- (valid but no index/slow seek -> remux -c copy fixes it for the next run).
CREATE TABLE IF NOT EXISTS problems (
    path        TEXT PRIMARY KEY,
    error       TEXT,
    category    TEXT NOT NULL DEFAULT 'corrupt',
    -- repair_note: result of the LAST remux attempt (NULL if not yet attempted).
    -- When present it is the "why" the UI shows instead of the scan error
    -- ('remux failed: …' -> unrecoverable; 'timeout …' -> still repairable, retriable).
    repair_note TEXT,
    last_seen   REAL NOT NULL
);

-- User feedback from the UI (detection correction). Does NOT retrain the network:
-- it feeds threshold recalibration (calibrate.suggest_thresholds) and view overrides.
-- label: 'same' (is a duplicate) | 'different' (NOT a duplicate, false positive).
CREATE TABLE IF NOT EXISTS feedback (
    a_path     TEXT NOT NULL,          -- canonicalized (a_path <= b_path), like matches
    b_path     TEXT NOT NULL,
    label      TEXT NOT NULL,
    note       TEXT,
    created_at REAL NOT NULL,
    UNIQUE(a_path, b_path)
);

-- Audit of deletions made from the UI (traceability / possible undo from Trash).
CREATE TABLE IF NOT EXISTS deletions (
    path        TEXT NOT NULL,
    dest        TEXT NOT NULL,         -- destination; always 'trash' (recoverable) in current builds
    size        INTEGER,
    deleted_at  REAL NOT NULL
);
