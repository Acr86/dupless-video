"""Store round-trip. M5: this test alone would have caught C1 (audio_fp dtype).

Validates that what goes in comes out bit-for-bit, especially audio_fp uint32.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from dupdetect.models import Probe, Quality, Record
from dupdetect.store import FingerprintStore

FV = "dinov2_vitb14|d768|fps2.0|algo1"  # feature_version for testing


def _rec(path: str) -> Record:
    return Record(
        path=path, mtime=123.456, size=999,
        probe=Probe(duration_s=7000.0, width=1920, height=1080, vcodec="h264",
                    bitrate_kbps=8000, audio_tracks=[]),
        content_hash="abc123",
        global_vec=np.random.rand(8).astype(np.float32),
        window_vecs=np.random.rand(12, 8).astype(np.float32),   # A2
        embeddings=np.random.rand(5, 8).astype(np.float32),
        # large values that float32 cannot represent exactly -> catches C1
        audio_fp=np.array([0, 1, 2**31, 2**32 - 1, 3_000_000_017], dtype=np.uint32),
        scene_cuts=np.array([1.0, 2.5, 60.0], dtype=np.float32),
        quality=Quality(lang_detected="spa", cam_score=0.3),
    )


@pytest.fixture
def store(tmp_path):
    s = FingerprintStore(tmp_path / "t.sqlite")
    yield s
    s.close()


def test_missing_embedding_npy_does_not_crash(store):
    """Regression: if the embeddings .npy was deleted/moved, pass 2 must NOT crash.
    has_fresh stops treating it as fresh (triggers re-index) and EmbeddingCache.get raises
    KeyError (which `warm` skips), instead of an uncaught FileNotFoundError."""
    from dupdetect.match.cache import EmbeddingCache
    rec = _rec("/lib/huerfano.mkv")
    store.save(rec, feature_version=FV)
    st = type("S", (), {"st_mtime": rec.mtime, "st_size": rec.size})()
    assert store.has_fresh(rec.path, st, FV) is True              # with .npy: fresh
    for f in store.emb_dir.glob("*.npy"):                         # clear embeddings
        f.unlink()
    assert store.has_fresh(rec.path, st, FV) is False             # without .npy: NOT fresh -> re-index
    cache = EmbeddingCache(store)
    with pytest.raises(KeyError):                                 # not FileNotFoundError
        cache.get(rec.path)
    cache.warm([rec.path])                                        # warm skips it without crashing


def test_audio_fp_uint32_exact_roundtrip(store):
    """C1: large uint32 values must survive the round-trip without losing bits."""
    rec = _rec("/lib/movie.mkv")
    store.save(rec, feature_version=FV)
    back = store.load(rec.path, with_embeddings=True)
    assert back is not None
    assert back.audio_fp.dtype == np.uint32
    np.testing.assert_array_equal(back.audio_fp, rec.audio_fp)


def test_embeddings_fp16_and_metadata_roundtrip(store):
    rec = _rec("/lib/movie.mkv")
    store.save(rec, feature_version=FV)
    back = store.load(rec.path, with_embeddings=True)
    # A1: embeddings are persisted as fp16 -> compare against the fp16 version
    assert back.embeddings.dtype == np.float16
    np.testing.assert_allclose(back.embeddings, rec.embeddings.astype(np.float16), rtol=1e-2)
    np.testing.assert_allclose(back.global_vec, rec.global_vec, rtol=1e-6)  # global_vec fp32
    assert back.probe.duration_s == rec.probe.duration_s
    assert back.quality.lang_detected == "spa"


def test_window_vecs_roundtrip(store):
    """A2: multi-vector descriptors survive with shape [K, D]."""
    rec = _rec("/lib/movie.mkv")
    store.save(rec, feature_version=FV)
    back = store.load(rec.path, with_embeddings=True)
    assert back.window_vecs.shape == (12, 8)
    np.testing.assert_allclose(back.window_vecs, rec.window_vecs, rtol=1e-6)


def test_duration_blocking(store):
    """A2: duration safety net finds similar-length candidates."""
    a = _rec("/lib/a.mkv"); a.probe.duration_s = 7000.0
    b = _rec("/lib/b.mkv"); b.probe.duration_s = 7200.0   # +2.9%, within ±12%
    c = _rec("/lib/c.mkv"); c.probe.duration_s = 3000.0   # very different
    for r in (a, b, c):
        store.save(r, feature_version=FV)
    hits = set(store.find_by_duration(7000.0, tol=0.12))
    assert "/lib/a.mkv" in hits and "/lib/b.mkv" in hits
    assert "/lib/c.mkv" not in hits


def test_find_by_hash_respects_feature_version(store):
    """M4: the short-circuit only applies with the SAME feature_version, and (when given) the SAME
    size — the hash is SAMPLED, so equal size is required to claim byte-identity (the T0 standard)."""
    rec = _rec("/lib/movie.mkv")
    store.save(rec, feature_version=FV)
    assert store.find_by_hash("abc123", FV) == "/lib/movie.mkv"
    assert store.find_by_hash("abc123", "OTHER") is None
    assert store.find_by_hash("abc123", FV, size=rec.size) == "/lib/movie.mkv"
    assert store.find_by_hash("abc123", FV, size=rec.size + 1) is None


def test_has_fresh_invalidated_by_feature_version(store, tmp_path):
    """C4: same mtime+size but different feature_version => NOT fresh."""
    import os
    f = tmp_path / "movie.mkv"
    f.write_bytes(b"x" * 999)
    st = os.stat(f)
    rec = _rec(str(f))
    rec.mtime, rec.size = st.st_mtime, st.st_size
    store.save(rec, feature_version=FV)

    assert store.has_fresh(str(f), st, feature_version=FV) is True
    assert store.has_fresh(str(f), st, feature_version="OTHER_MODEL") is False


def test_problems_persistence_and_auto_cleanup(store):
    """The problems table stores (path, error) and is CLEARED when that path is analyzed OK."""
    store.save_problem("/lib/broken.mkv", "moov atom not found")
    probs = [(p, e) for p, e, *_ in store.problems()]
    assert probs == [("/lib/broken.mkv", "moov atom not found")]

    # if that same path is later saved OK (Record), it disappears from problems
    rec = _rec("/lib/broken.mkv")
    store.save(rec, feature_version=FV)
    assert store.problems() == []


def test_delete_match_and_matched_pairs(store):
    """delete_match drops ONE pair (canonicalized, order-independent); matched_pairs is the small set of
    pairs that currently have a row (preloaded by the scan's delete-on-DIFFERENT guard)."""
    store.save_match("/a", "/b", "CERTAIN", 0.99, "T1")
    store.save_match("/c", "/d", "HIGH", 0.88, "T3")
    assert store.matched_pairs() == {("/a", "/b"), ("/c", "/d")}
    store.delete_match("/b", "/a")                         # reversed order -> still canonical (/a,/b)
    assert store.has_match("/a", "/b") is False
    assert store.matched_pairs() == {("/c", "/d")}


def test_delete_matches_batch_and_prune_by_verdict(store):
    """delete_matches removes many pairs in one transaction; prune_matches_by_verdict drops every row of
    a verdict (the maintenance sweep for DIFFERENT rows an old re-decide upserted in place)."""
    store.save_match("/a", "/b", "DIFFERENT", 0.0, "T5")
    store.save_match("/a", "/c", "DIFFERENT", 0.0, "T5")
    store.save_match("/a", "/d", "CERTAIN", 0.99, "T1")
    assert store.prune_matches_by_verdict("DIFFERENT") == 2
    assert store.matched_pairs() == {("/a", "/d")}
    assert store.delete_matches([("/d", "/a")]) == 1       # reversed -> canonical (/a,/d)
    assert store.matched_pairs() == set()


def test_has_unchanged_problem_guards_known_corrupt(tmp_path):
    """save_problem stamps mtime+size; has_unchanged_problem is True only while the file is unchanged
    -> the scan/watcher skips a known-corrupt file instead of re-decoding it every cycle."""
    s = FingerprintStore(tmp_path / "p.sqlite")
    f = tmp_path / "broken.mp4"; f.write_bytes(b"x" * 100)
    assert s.has_unchanged_problem(str(f), os.stat(f)) is False    # no problem recorded yet
    s.save_problem(str(f), "moov atom not found", "corrupt")
    assert s.has_unchanged_problem(str(f), os.stat(f)) is True     # known + unchanged -> skip
    f.write_bytes(b"x" * 200)                                      # re-downloaded (size changed)
    assert s.has_unchanged_problem(str(f), os.stat(f)) is False    # changed -> retry
    s.close()


def test_emb_path_relative_relocatable(tmp_path):
    """emb_path is stored as relative => the store is relocatable (moving the folder does not break it)."""
    s = FingerprintStore(tmp_path / "a" / "db.sqlite")
    rec = _rec("/lib/movie.mkv")
    s.save(rec, feature_version=FV)
    row = s.conn.execute("SELECT emb_path FROM files").fetchone()
    assert not os.path.isabs(row["emb_path"])            # just the filename, not an absolute path
    back = s.load("/lib/movie.mkv", with_embeddings=True)  # resolved against emb_dir
    assert back.embeddings.shape == (5, 8)
    s.close()


def test_canonical_pair_symmetric():
    """C2: the pair is ordered the same regardless of input order."""
    from dupdetect.store.store import canonical_pair
    assert canonical_pair("b", "a") == canonical_pair("a", "b") == ("a", "b")


def _match_row(store, a, b):
    return store.conn.execute(
        "SELECT a_path, b_path, ad_offset_s FROM matches WHERE a_path=? AND b_path=?",
        (a, b),
    ).fetchone()


def test_save_match_normalizes_offset_sign(store):
    """C3: the offset is directional ('b relative to a'). When canonicalizing a reversed
    pair, the sign must be negated to preserve WHICH copy carries the ads."""
    # already canonical order: a<=b -> offset unchanged
    store.save_match("/lib/a.mkv", "/lib/b.mkv", "CERTAIN", 0.99, "T1", ad_offset_s=45.0)
    row = _match_row(store, "/lib/a.mkv", "/lib/b.mkv")
    assert row is not None and row["ad_offset_s"] == 45.0

    # reversed order (caller passes b,a): stored as (a,b) and the offset is negated
    store.save_match("/lib/d.mkv", "/lib/c.mkv", "CERTAIN", 0.99, "T1", ad_offset_s=45.0)
    row = _match_row(store, "/lib/c.mkv", "/lib/d.mkv")
    assert row is not None and row["ad_offset_s"] == -45.0


def test_save_match_offset_none_does_not_crash(store):
    """C3: ad_offset_s can be None (no video track) even with reversed order."""
    store.save_match("/lib/z.mkv", "/lib/y.mkv", "PROBABLE", 0.5, "T4", ad_offset_s=None)
    row = _match_row(store, "/lib/y.mkv", "/lib/z.mkv")
    assert row is not None and row["ad_offset_s"] is None


def test_feature_version_includes_audio_and_scenes():
    """C4: the combiner versions ALL signals, not just embeddings. A change in
    Chromaprint/PySceneDetect must change the string -> invalidates the cache."""
    from dupdetect.features.audio_fp import AUDIO_FP_VERSION
    from dupdetect.features.embeddings import Embedder
    from dupdetect.features.scenes import SCENE_ALGO_VERSION
    from dupdetect.pipeline.analyze import feature_version

    fv = feature_version(Embedder(model="dinov2_vitb14", dim=768, fps=2.0))
    assert f"afp{AUDIO_FP_VERSION}" in fv
    assert f"scnEMB{SCENE_ALGO_VERSION}" in fv          # mode B (default) in the version string
    assert "fps2.0" in fv


def test_feature_version_changes_with_fps():
    """C4-bug (cli): the actual fps must be included in the version; two different fps values
    produce different versions -> cache is invalidated when fps_sample changes."""
    from dupdetect.features.embeddings import Embedder
    from dupdetect.pipeline.analyze import feature_version

    assert feature_version(Embedder(fps=2.0)) != feature_version(Embedder(fps=3.0))


def test_all_global_vecs_skips_lite_records(store):
    """Regression (real DB exposed it): a store mixing full records with exact-only LITE records
    (global_vec NULL) must NOT crash in all_global_vecs -> LITE rows are skipped (no embeddings
    to index). Any full scan into a DB that contains exact-only sweeps relies on this."""
    full = _rec("/full.mkv")
    store.save(full, feature_version=FV)
    store.save_meta("/lite.mkv", 1.0, 100, "h", full.probe, feature_version="exact-only-v1")
    paths, vecs = store.all_global_vecs()            # would TypeError on np.frombuffer(None) before
    assert paths == ["/full.mkv"]                    # only the full record, LITE skipped
    assert vecs.shape == (1, 8)


# ----------------------------------------------------- quality overrides ("Mark audio as OK")

def _flagged_file(store, tmp_path, name: str):
    """A real on-disk file, indexed and MEASURED below the audio threshold (shows as a warning)."""
    f = tmp_path / name
    f.write_bytes(b"x" * 100)
    p = str(f)
    store.save(_rec(p), feature_version=FV)
    store.set_audio_coverage(p, 0.2)
    return f, p


def test_quality_override_hides_warning_and_is_revertible(store, tmp_path):
    """'Mark audio as OK': the warning disappears but the MEASURED value stays stored, so
    'restore' is just dropping the override — no data is lost."""
    _f, p = _flagged_file(store, tmp_path, "quiet.mkv")
    assert [w[0] for w in store.audio_warnings()] == [p]
    assert store.set_quality_override(p) is True
    assert store.has_quality_override(p)
    assert store.audio_warnings() == []                            # hidden while overridden
    got = store.conn.execute("SELECT audio_coverage FROM files WHERE path=?", (p,)).fetchone()[0]
    assert got == 0.2                                              # measurement untouched
    store.clear_quality_override(p)
    assert not store.has_quality_override(p)
    assert [w[0] for w in store.audio_warnings()] == [p]           # measured value rules again


def test_quality_override_expires_when_file_changes(store, tmp_path):
    """The override stamps mtime+size: a changed file (truly muted re-download) EXPIRES it and
    the warning comes back — fail-safe, the user is warned again."""
    f, p = _flagged_file(store, tmp_path, "q2.mkv")
    store.set_quality_override(p)
    assert store.has_quality_override(p) and store.audio_warnings() == []
    f.write_bytes(b"y" * 999)                                      # size (and mtime) change
    assert not store.has_quality_override(p)
    assert [w[0] for w in store.audio_warnings()] == [p]


def test_quality_override_needs_real_file_and_dies_with_it(store, tmp_path):
    """An override stamps a real file (unstattable -> no row) and is removed by forget_file."""
    assert store.set_quality_override(str(tmp_path / "ghost.mkv")) is False
    _f, p = _flagged_file(store, tmp_path, "q3.mkv")
    store.set_quality_override(p)
    store.forget_file(p)
    assert store.conn.execute("SELECT COUNT(*) FROM quality_overrides").fetchone()[0] == 0


def test_coverage_v2_migration_nulls_only_flagged_once(tmp_path):
    """v2-probe rollout: reopening a pre-v2 DB NULLs ONLY the rows the v1 probe FLAGGED (they
    re-measure lazily with v2); clean caches are kept, and the user_version marker guarantees a
    legitimately-low v2 value is never re-NULLed on a later reopen."""
    db = tmp_path / "mig.sqlite"
    s = FingerprintStore(db)
    for p, cov in (("/lib/ok.mkv", 0.95), ("/lib/flagged.mkv", 0.30)):
        s.save(_rec(p), feature_version=FV)
        s.set_audio_coverage(p, cov)
    s.conn.execute("PRAGMA user_version = 0")                      # simulate a pre-v2 DB
    s.conn.commit()
    s.close()
    s2 = FingerprintStore(db)                                      # reopen -> migration runs
    rows = dict(s2.conn.execute("SELECT path, audio_coverage FROM files"))
    assert rows["/lib/ok.mkv"] == 0.95                             # clean cache kept
    assert rows["/lib/flagged.mkv"] is None                        # flagged -> re-measure with v2
    s2.set_audio_coverage("/lib/flagged.mkv", 0.30)                # a legit LOW v2 measurement...
    s2.close()
    s3 = FingerprintStore(db)                                      # ...survives the next reopen
    assert s3.conn.execute("SELECT audio_coverage FROM files WHERE path=?",
                           ("/lib/flagged.mkv",)).fetchone()[0] == 0.30
    s3.close()
