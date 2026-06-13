"""Tests for full_scan (step 9): rank_cluster, _cluster_has_ads, UnionFind.

rank_cluster is tested with synthetic Records against a real store (no video I/O).
"""
from __future__ import annotations

import numpy as np
import pytest

from dupdetect.config import load_thresholds
from dupdetect.models import Probe, Quality, Record
from dupdetect.pipeline.fullscan import (
    UnionFind, _cluster_has_ads, _rebuild_clusters, exact_scan, full_scan, rank_cluster,
)
from dupdetect.store import FingerprintStore

FV = "test|v1"


def _rec(path, w=1920, h=1080, br=8000, lang="eng", cam=0.1) -> Record:
    return Record(
        path=path, mtime=0.0, size=100,
        probe=Probe(duration_s=6000.0, width=w, height=h, vcodec="h264",
                    bitrate_kbps=br, audio_tracks=[]),
        content_hash="h" + path,
        global_vec=np.zeros(8, np.float32), window_vecs=np.zeros((4, 8), np.float32),
        embeddings=np.zeros((1, 8), np.float16), audio_fp=np.zeros(1, np.uint32),
        scene_cuts=np.zeros(1, np.float32),
        quality=Quality(lang_detected=lang, cam_score=cam),
    )


@pytest.fixture
def th():
    return load_thresholds()       # wanted_langs = [spa, eng]


@pytest.fixture
def store(tmp_path):
    s = FingerprintStore(tmp_path / "fs.sqlite")
    yield s
    s.close()


# --------------------------------------------------------------- UnionFind

def test_unionfind_groups_transitively():
    uf = UnionFind()
    uf.union("a", "b"); uf.union("b", "c"); uf.union("x", "y")
    groups = sorted(sorted(g) for g in uf.groups().values())
    assert ["a", "b", "c"] in groups and ["x", "y"] in groups


# --------------------------------------------------------------- rank_cluster

def test_rank_prefers_higher_resolution(th, store):
    for r in (_rec("/1080.mkv", 1920, 1080), _rec("/720.mkv", 1280, 720),
              _rec("/480.mkv", 854, 480)):
        store.save(r, feature_version=FV)
    out = rank_cluster(["/1080.mkv", "/720.mkv", "/480.mkv"], store, th)
    assert out["keep"] == "/1080.mkv"
    assert set(out["discard"]) == {"/720.mkv", "/480.mkv"}


def test_rank_wanted_lang_beats_resolution(th, store):
    # 4K in Russian (unwanted) vs 1080p in Spanish (wanted) -> Spanish wins
    store.save(_rec("/4k_ru.mkv", 3840, 2160, lang="rus"), feature_version=FV)
    store.save(_rec("/1080_es.mkv", 1920, 1080, lang="spa"), feature_version=FV)
    out = rank_cluster(["/4k_ru.mkv", "/1080_es.mkv"], store, th)
    assert out["keep"] == "/1080_es.mkv"


def test_rank_bitrate_breaks_tie_at_same_resolution(th, store):
    store.save(_rec("/hi.mkv", 1920, 1080, br=12000), feature_version=FV)
    store.save(_rec("/lo.mkv", 1920, 1080, br=3000), feature_version=FV)
    out = rank_cluster(["/hi.mkv", "/lo.mkv"], store, th)
    assert out["keep"] == "/hi.mkv"


def test_cluster_has_ads_reads_canonical_offset(th, store):
    store.save(_rec("/a.mkv"), feature_version=FV)
    store.save(_rec("/b.mkv"), feature_version=FV)
    # canonical pair (a<b); offset +30s => 'b' has ads at the start
    store.save_match("/a.mkv", "/b.mkv", "CERTAIN", 0.99, "T1", ad_offset_s=30.0)
    assert _cluster_has_ads(store, "/b.mkv", {"/a.mkv", "/b.mkv"}, th) is True
    assert _cluster_has_ads(store, "/a.mkv", {"/a.mkv", "/b.mkv"}, th) is False


def test_rank_penalizes_ads_at_same_quality(th, store):
    store.save(_rec("/clean.mkv"), feature_version=FV)
    store.save(_rec("/ads.mkv"), feature_version=FV)
    # caller (query=clean, candidate=ads): offset +40 = 'ads' starts 40s later
    # (has ads at the start). save_match normalizes to canonical orientation.
    store.save_match("/clean.mkv", "/ads.mkv", "CERTAIN", 0.99, "T1", ad_offset_s=40.0)
    assert _cluster_has_ads(store, "/ads.mkv", {"/clean.mkv", "/ads.mkv"}, th) is True
    out = rank_cluster(["/clean.mkv", "/ads.mkv"], store, th)
    assert out["keep"] == "/clean.mkv"


def test_cluster_has_midroll_ads_via_interleaved_ratio(th, store):
    """Mid-roll commercials: video_json.interleaved_ratio >= threshold with ad_dir pointing at the
    LONGER (ad) copy. The ad copy is flagged and KEEP prefers the clean one (verdict untouched)."""
    import json
    store.save(_rec("/clean.mkv"), feature_version=FV)
    store.save(_rec("/withads.mkv"), feature_version=FV)
    # canonical pair a<b: '/clean.mkv' < '/withads.mkv'; ad_dir=+1 => b ('/withads') is the ad copy
    vj = json.dumps({"score": 0.99, "coverage": 1.0, "interleaved_ratio": 0.09, "ad_dir": 1})
    store.save_match("/clean.mkv", "/withads.mkv", "CERTAIN", 0.99, "T1", video_json=vj)
    assert _cluster_has_ads(store, "/withads.mkv", {"/clean.mkv", "/withads.mkv"}, th) is True
    assert _cluster_has_ads(store, "/clean.mkv", {"/clean.mkv", "/withads.mkv"}, th) is False
    out = rank_cluster(["/clean.mkv", "/withads.mkv"], store, th)
    assert out["keep"] == "/clean.mkv"                       # KEEP the copy WITHOUT commercials
    assert ", ads" in out["evidence"]["/withads.mkv"]        # UI marks which copy has ads


# ---------------------------------------------- clusters = derived view (does not accumulate)

def _all_cluster_rows(store):
    return store.conn.execute("SELECT cluster_id, path FROM clusters").fetchall()


def test_rebuild_clusters_from_global_graph(th, store):
    for n in ("/a.mkv", "/b.mkv", "/c.mkv", "/d.mkv"):
        store.save(_rec(n), feature_version=FV)
    store.save_match("/a.mkv", "/b.mkv", "CERTAIN", 0.99, "T1")
    store.save_match("/c.mkv", "/d.mkv", "CERTAIN", 0.99, "T1")
    out = _rebuild_clusters(store, th)
    groups = sorted(sorted([c["keep"], *c["discard"]]) for c in out)
    assert groups == [["/a.mkv", "/b.mkv"], ["/c.mkv", "/d.mkv"]]


def test_rebuild_clusters_leaves_no_stale_rows(th, store):
    """Regression: a re-scan that changes membership must NOT leave a file in two
    clusters. The entire table is rebuilt from the global match graph."""
    for n in ("/a.mkv", "/b.mkv", "/c.mkv"):
        store.save(_rec(n), feature_version=FV)
    # run 1: a-b are dups, c is standalone -> 1 cluster {a,b}
    store.save_match("/a.mkv", "/b.mkv", "CERTAIN", 0.99, "T1")
    _rebuild_clusters(store, th)
    # run 2: graph changes -> now b-c are dups, a-b no longer. (re-indexing b deleted its
    # old match; the matcher persists the new one)
    store.conn.execute("DELETE FROM matches"); store.conn.commit()
    store.save_match("/b.mkv", "/c.mkv", "CERTAIN", 0.99, "T1")
    _rebuild_clusters(store, th)
    rows = _all_cluster_rows(store)
    paths = [r["path"] for r in rows]
    assert sorted(paths) == ["/b.mkv", "/c.mkv"]          # a is gone; no stale rows
    assert len(paths) == len(set(paths))                  # no path appears in two clusters
    assert "/a.mkv" not in paths


def test_save_invalidates_stale_matches_on_reindex(store):
    """Re-indexing a file deletes its previous matches (stale features)."""
    store.save(_rec("/x.mkv"), feature_version=FV)
    store.save(_rec("/y.mkv"), feature_version=FV)
    store.save_match("/x.mkv", "/y.mkv", "CERTAIN", 0.99, "T1")
    assert len(store.all_matches()) == 1
    store.save(_rec("/x.mkv"), feature_version=FV)        # re-index x
    assert store.all_matches() == []                      # its old match is gone


# ------------------------------------------------ parallel decode (SSD pipeline)

def test_drain_pipelined_processes_all_and_is_resilient(monkeypatch, store):
    """The decode-prefetch scheduler processes ALL files, keeps the pipeline full,
    and routes corrupt ones to `skipped`/problems without crashing the rest. No GPU:
    decode_frames and _gpu_finish are mocked."""
    from concurrent.futures import Future
    from dupdetect.pipeline import fullscan as fs

    class _Cpu:
        def __init__(self, p): self.path = p

    def _fut(val=None, exc=None):
        f = Future()
        f.set_exception(exc) if exc else f.set_result(val)
        return f

    cpu_futs = {_fut(_Cpu("/a")): "/a", _fut(_Cpu("/b")): "/b",
                _fut(exc=RuntimeError("corrupt")): "/c", _fut(_Cpu("/d")): "/d"}
    monkeypatch.setattr(fs, "decode_frames", lambda p: ("FRAMES", "TIMES"))
    done = []
    monkeypatch.setattr(fs, "_gpu_finish",
                        lambda cpu, ft, *a, **k: done.append((cpu.path, ft)))
    skipped, marks = [], []
    fs._drain_pipelined(cpu_futs, store, None, None, "fv", False, 2, skipped, marks.append)

    assert sorted(p for p, _ in done) == ["/a", "/b", "/d"]      # good files are embedded
    assert all(ft == ("FRAMES", "TIMES") for _, ft in done)      # with pre-decoded frames
    assert [p for p, _ in skipped] == ["/c"]                     # corrupt one, isolated
    assert sorted(marks) == ["/a", "/b", "/c", "/d"]             # all advance the progress bar


# --------------------------------------------------------------- exact_scan (exact-only mode)

def test_exact_scan_detects_byte_identical_files(th, store, tmp_path):
    """Exact-only mode: groups byte-identical files by hash, saves LITE records (no
    embeddings) that the UI can display, and excludes differing files."""
    import shutil

    av = pytest.importorskip("av")

    def _mk(p, val):
        c = av.open(str(p), "w"); s = c.add_stream("mpeg4", rate=10)
        s.width = s.height = 64; s.pix_fmt = "yuv420p"
        for _ in range(10):
            fr = av.VideoFrame.from_ndarray(np.full((64, 64, 3), val, np.uint8),
                                            format="rgb24").reformat(format="yuv420p")
            for pk in s.encode(fr):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
        c.close()

    a = tmp_path / "a.mp4"; _mk(a, 50)
    b = tmp_path / "b.mp4"; shutil.copyfile(a, b)         # byte-identical copy of a
    d = tmp_path / "d.mp4"; _mk(d, 200)                   # different content

    rep = exact_scan([str(tmp_path)], store, th, workers=1, recursive=True)
    assert len(rep["clusters"]) == 1                      # exactly one identical group
    members = {rep["clusters"][0]["keep"], *rep["clusters"][0]["discard"]}
    assert members == {str(a), str(b)} and str(d) not in members
    # LITE record: has hash + probe but NO embeddings (the expensive pass was skipped)
    rec = store.load(str(a), with_embeddings=False)
    assert rec is not None and rec.content_hash and rec.embeddings.size == 0
    # incremental: re-running does not re-hash (reuses stored hash), same result
    rep2 = exact_scan([str(tmp_path)], store, th, workers=1, recursive=True)
    assert len(rep2["clusters"]) == 1


# --------------------------------------------------------------- full_scan empty dir

def test_full_scan_skips_corrupt_file(th, store, tmp_path):
    """Resilience: an unreadable .mp4 (garbage bytes) is SKIPPED and reported, not a crash."""
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"this is not a valid video" * 100)   # ffprobe will fail

    class _DummyEmbedder:
        fps = 2.0; model_name = "m"; dim = 8; algo_version = 1
        @property
        def feature_version(self): return FV
    report = full_scan(str(bad), store, _DummyEmbedder(), th)
    assert len(report["skipped"]) == 1
    assert "corrupt.mp4" in report["skipped"][0][0]
    assert report["clusters"] == []
    # the problem is PERSISTED in the DB with its error (for index reconstruction)
    probs = store.iter_problems()
    assert len(probs) == 1 and "corrupt.mp4" in probs[0][0]
    assert probs[0][1]                                   # error message is present


def test_filter_by_height(tmp_path):
    """filter_by_height splits by height; unmeasurable (corrupt) files are KEPT."""
    import av
    from dupdetect.pipeline.fullscan import filter_by_height

    def _mk(name, h):
        p = tmp_path / name
        c = av.open(str(p), mode="w"); s = c.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt = int(h * 16 / 9), h, "yuv420p"
        for _ in range(3):
            fr = av.VideoFrame.from_ndarray(np.zeros((h, int(h * 16 / 9), 3), np.uint8),
                                            format="rgb24").reformat(format="yuv420p")
            for pk in s.encode(fr):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
        c.close()
        return str(p)

    hd = _mk("hd.mp4", 720)
    big = _mk("big.mp4", 1440)
    bad = tmp_path / "broken.mp4"; bad.write_bytes(b"garbage" * 100)
    kept, excluded = filter_by_height([hd, big, str(bad)], max_height=1080)
    assert hd in kept and str(bad) in kept       # HD passes; corrupt kept (_pass1 handles it)
    assert big in excluded                       # 1440 > 1080 -> excluded


def test_full_scan_empty_dir(th, store, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    class _DummyEmbedder:
        def __init__(self): self.fps = th.fps_sample; self.model_name="m"; self.dim=8; self.algo_version=1
        @property
        def feature_version(self): return FV
    report = full_scan(str(empty), store, _DummyEmbedder(), th)
    assert report["clusters"] == [] and report["review_queue"] == [] and report["editions"] == []


def test_full_scan_no_match_skips_pass2(th, store, tmp_path, monkeypatch):
    """`match=False` -> Pass-1 only: matching (and the coarse index) are NOT run. Lets a large
    library be (re)indexed cheaply before the O(N^2) Pass-2."""
    import dupdetect.pipeline.fullscan as fs

    def _boom(*a, **k):
        raise AssertionError("Pass-2 must not run when match=False")
    monkeypatch.setattr(fs, "_pass2", _boom)
    empty = tmp_path / "empty"; empty.mkdir()

    class _DummyEmbedder:
        fps = 2.0; model_name = "m"; dim = 8; algo_version = 1
        @property
        def feature_version(self): return FV
    rep = fs.full_scan(str(empty), store, _DummyEmbedder(), th, match=False)
    assert rep["clusters"] == [] and rep["review_queue"] == [] and rep["editions"] == []
