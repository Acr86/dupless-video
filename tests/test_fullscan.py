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
from dupdetect.store import FingerprintStore, canonical_pair

FV = "test|v1"


def _rec(path, w=1920, h=1080, br=8000, lang="eng", cam=0.1, codec="h264") -> Record:
    return Record(
        path=path, mtime=0.0, size=100,
        probe=Probe(duration_s=6000.0, width=w, height=h, vcodec=codec,
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


def test_rank_wanted_lang_beats_resolution_when_opted_in(th, store):
    # Language KEEP is OPT-IN now (whisper is expensive; resolution dominates by default). With
    # detect_lang=True the wanted language still dominates: 1080p Spanish beats 4K Russian.
    store.save(_rec("/4k_ru.mkv", 3840, 2160, lang="rus"), feature_version=FV)
    store.save(_rec("/1080_es.mkv", 1920, 1080, lang="spa"), feature_version=FV)
    out = rank_cluster(["/4k_ru.mkv", "/1080_es.mkv"], store, th, detect_lang=True)
    assert out["keep"] == "/1080_es.mkv"


def test_rank_ignores_language_by_default(th, store):
    # Default (detect_lang=False): language is NOT consulted -> the 4K copy wins on resolution even
    # though the 1080p is in the wanted language. (Phase-1 lean ranking; language is on-demand.)
    store.save(_rec("/4k_ru.mkv", 3840, 2160, lang="rus"), feature_version=FV)
    store.save(_rec("/1080_es.mkv", 1920, 1080, lang="spa"), feature_version=FV)
    out = rank_cluster(["/4k_ru.mkv", "/1080_es.mkv"], store, th)
    assert out["keep"] == "/4k_ru.mkv"


def test_rank_bitrate_breaks_tie_at_same_resolution(th, store):
    store.save(_rec("/hi.mkv", 1920, 1080, br=12000), feature_version=FV)
    store.save(_rec("/lo.mkv", 1920, 1080, br=3000), feature_version=FV)
    out = rank_cluster(["/hi.mkv", "/lo.mkv"], store, th)
    assert out["keep"] == "/hi.mkv"


def test_rank_codec_aware_bitrate_keeps_efficient_av1(th, store):
    # Same resolution: an AV1 at 3000 kbps vs H.264 at 5000 kbps. Raw bitrate would keep the H.264,
    # but AV1 reaches the same quality at ~half the bitrate -> effective 6000 > 5000 -> KEEP the AV1.
    store.save(_rec("/av1.mkv", br=3000, codec="av1"), feature_version=FV)
    store.save(_rec("/h264.mkv", br=5000, codec="h264"), feature_version=FV)
    out = rank_cluster(["/av1.mkv", "/h264.mkv"], store, th)
    assert out["keep"] == "/av1.mkv"


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


# ---------------------------------------------- incremental Pass-2 (evaluated-pairs ledger)

def test_needs_analysis_skips_unchanged_corrupt(tmp_path):
    """Pass-1 must not re-decode a known-corrupt file that didn't change (it would fail again and sit
    'unprocessed' forever). Skipped while unchanged; retried after force or a content change."""
    import os

    from dupdetect.pipeline.fullscan import _needs_analysis
    from dupdetect.store import FingerprintStore
    s = FingerprintStore(tmp_path / "n.sqlite")
    f = tmp_path / "bad.mp4"; f.write_bytes(b"x" * 50)
    assert _needs_analysis(s, str(f), "fv1", False) is True       # unseen -> analyze
    s.save_problem(str(f), "moov atom not found", "corrupt")
    assert _needs_analysis(s, str(f), "fv1", False) is False      # known corrupt, unchanged -> skip
    assert _needs_analysis(s, str(f), "fv1", True) is True        # force overrides
    f.write_bytes(b"x" * 99)                                      # re-downloaded
    assert _needs_analysis(s, str(f), "fv1", False) is True       # changed -> retry
    s.close()


def test_scan_fingerprint_changes_with_fv_and_thresholds(th):
    import copy

    from dupdetect.config import Thresholds
    from dupdetect.pipeline.fullscan import _scan_fingerprint
    base = _scan_fingerprint("fv1", th)
    assert _scan_fingerprint("fv1", th) == base                  # stable: same inputs -> same key
    assert _scan_fingerprint("fv2", th) != base                  # algorithm change -> invalidates
    raw = copy.deepcopy(th.raw); raw["video"]["theta_v"] = 0.999
    assert _scan_fingerprint("fv1", Thresholds(raw=raw)) != base  # θ change -> invalidates (looser θ
    #                                                              # could turn a DIFFERENT into a match)


def test_evaluated_pairs_ledger_roundtrip_and_invalidation(store):
    store.evaluated_pairs_add(["aa", "bb", "cc"], "fp1")
    assert store.evaluated_pairs_load("fp1") == {"aa", "bb", "cc"}
    # loading under a DIFFERENT fingerprint sees nothing AND prunes the now-stale rows
    assert store.evaluated_pairs_load("fp2") == set()
    assert store.evaluated_pairs_load("fp1") == set()            # fp1 rows were pruned


def test_enumerate_pairs_skips_evaluated_unless_changed(th, store, monkeypatch):
    from dupdetect.match import matcher
    from dupdetect.match.matcher import _enumerate_pairs, _pair_hash
    from dupdetect.store.store import canonical_pair
    for p in ("/a.mkv", "/b.mkv", "/c.mkv"):
        store.save(_rec(p), feature_version=FV)
    cand = {"/a.mkv": {"/b.mkv", "/c.mkv"}}                       # candidate graph: a~b, a~c
    monkeypatch.setattr(matcher, "candidate_paths", lambda rec, s, i, t: cand.get(rec.path, set()))
    paths = ["/a.mkv", "/b.mkv", "/c.mkv"]
    ab, ac = canonical_pair("/a.mkv", "/b.mkv"), canonical_pair("/a.mkv", "/c.mkv")

    # run 1: nothing evaluated -> all pairs enumerated
    pairs1 = _enumerate_pairs(paths, store, None, th, set(), set(), progress=False)
    assert set(pairs1) == {ab, ac}
    evaluated = {_pair_hash(pr) for pr in pairs1}

    # run 2: all evaluated, nothing changed -> ZERO pairs (the incremental win)
    assert _enumerate_pairs(paths, store, None, th, evaluated, set(), progress=False) == []

    # run 2b: /c re-analyzed -> only pairs TOUCHING /c are re-enumerated
    pairs2 = _enumerate_pairs(paths, store, None, th, evaluated, {"/c.mkv"}, progress=False)
    assert set(pairs2) == {ac}


def test_match_pairs_parallel_records_every_aligned_pair(th, store, monkeypatch):
    """The ledger records EVERY aligned pair (match OR DIFFERENT), so the next run can skip it. Stubs
    the pool/drain so no real multiprocessing runs; the pair here decides DIFFERENT (row None) yet is
    still recorded."""
    import contextlib

    from dupdetect.match import matcher
    from dupdetect.store.store import canonical_pair
    for p in ("/a.mkv", "/b.mkv"):
        store.save(_rec(p), feature_version=FV)
    monkeypatch.setattr(matcher, "candidate_paths",
                        lambda rec, s, i, t: {"/b.mkv"} if rec.path == "/a.mkv" else set())
    monkeypatch.setattr(matcher, "single_threaded_blas", contextlib.nullcontext)
    monkeypatch.setattr(matcher, "ProcessPoolExecutor", lambda **k: contextlib.nullcontext())
    # drain yields (pair, None) for each pair -> DIFFERENT, nothing to save, but still "aligned"
    monkeypatch.setattr(matcher, "_drain_pairs_bounded",
                        lambda pool, pl, w, pr: iter([(x, None) for x in pl]))
    rows = list(matcher.match_pairs_parallel(["/a.mkv", "/b.mkv"], store, None, th,
                                             workers=2, progress=False, fingerprint="fp1"))
    assert rows == []                                            # DIFFERENT -> no rows yielded
    assert store.evaluated_pairs_load("fp1") == {matcher._pair_hash(canonical_pair("/a.mkv", "/b.mkv"))}


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


def test_contains_edge_does_not_fuse_clusters(th, store):
    """Root fix for compilation chain-fusion: a CONTAINS edge (a clip is a segment inside a long
    compilation) must NOT union files. So a compilation that contains clip C, while also being a real
    duplicate of A, does NOT drag C into A's group."""
    for n in ("/a.mkv", "/b_comp.mkv", "/c_clip.mkv"):
        store.save(_rec(n), feature_version=FV)
    store.save_match("/a.mkv", "/b_comp.mkv", "CERTAIN", 0.99, "T1")            # real duplicate
    store.save_match("/b_comp.mkv", "/c_clip.mkv", "CONTAINS", 0.85, "contains")  # b contains c
    groups = sorted(sorted([c["keep"], *c["discard"]]) for c in _rebuild_clusters(store, th))
    assert groups == [["/a.mkv", "/b_comp.mkv"]]                    # c_clip NOT fused via the CONTAINS edge


def test_user_veto_splits_cluster_and_survives_rescan(th, store):
    """A user 'not a duplicate' veto OUTRANKS the content verdict: the vetoed pair is never unioned,
    so the group splits — and it STAYS split when a later scan re-declares the pair CERTAIN (the veto
    lives in `feedback`, re-applied on every rebuild). Marking it 'same' again lifts the veto."""
    for n in ("/a.mkv", "/b.mkv", "/c.mkv"):
        store.save(_rec(n), feature_version=FV)
    store.save_match("/a.mkv", "/b.mkv", "CERTAIN", 0.99, "T1")
    store.save_match("/b.mkv", "/c.mkv", "CERTAIN", 0.99, "T1")     # chain -> a,b,c one component
    assert len(_rebuild_clusters(store, th)) == 1                   # one group of 3

    store.save_feedback("/b.mkv", "/c.mkv", "different")            # user: c doesn't belong
    out = _rebuild_clusters(store, th)
    groups = sorted(sorted([c["keep"], *c["discard"]]) for c in out)
    assert groups == [["/a.mkv", "/b.mkv"]]                         # c split off (singleton dropped)

    store.save_match("/b.mkv", "/c.mkv", "CERTAIN", 0.99, "T1")     # a re-scan re-declares it
    groups = sorted(sorted([c["keep"], *c["discard"]]) for c in _rebuild_clusters(store, th))
    assert groups == [["/a.mkv", "/b.mkv"]]                         # veto still holds

    store.save_feedback("/b.mkv", "/c.mkv", "same")                 # user changes their mind
    assert len(_rebuild_clusters(store, th)[0]["discard"]) == 2     # regrouped: 3 members again


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
    embeddings) that the UI can display, and excludes differing files. It must ALSO stamp
    the T0 verdict in `matches` so the pair reads as CERTAIN, not just 'Review only'
    (otherwise clusters and matches drift — see ui.data.drift_report)."""
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
    # M1: the byte-identical pair must be stamped CERTAIN (T0) in `matches`, not left
    # verdict-less ("Review only"). Without this, clusters and matches drift.
    assert store.has_match(str(a), str(b))
    ca, cb = canonical_pair(str(a), str(b))
    row = store.conn.execute(
        "SELECT verdict, confidence, reason FROM matches WHERE a_path=? AND b_path=?", (ca, cb)
    ).fetchone()
    assert row is not None and row["verdict"] == "CERTAIN"
    assert row["confidence"] == pytest.approx(1.0)
    assert row["reason"].startswith("T0")
    assert not store.has_match(str(a), str(d))           # the different file is not a duplicate
    # LITE record: has hash + probe but NO embeddings (the expensive pass was skipped)
    rec = store.load(str(a), with_embeddings=False)
    assert rec is not None and rec.content_hash and rec.embeddings.size == 0
    # incremental: re-running does not re-hash (reuses stored hash), same result
    rep2 = exact_scan([str(tmp_path)], store, th, workers=1, recursive=True)
    assert len(rep2["clusters"]) == 1


# --------------------------------------------------------- apply_thresholds_to_store

def test_apply_thresholds_to_store_redecides_from_stored_signals(th, store):
    """Re-applies thresholds to existing matches WITHOUT decoding: re-runs decide_tree over
    the signals already in `matches`. Tightening a threshold flips a content verdict; the SAME
    thresholds are a no-op; signal-less rows (T0 / NAME_COPY) are threshold-independent -> kept."""
    import copy
    import json as _json

    from dupdetect.config import Thresholds
    from dupdetect.pipeline.calibrate import apply_thresholds_to_store

    store.save(_rec("/a.mkv"), feature_version=FV)
    store.save(_rec("/b.mkv"), feature_version=FV)
    # content pair that clears T1 under default thresholds (audio+video agree, good coverage)
    store.save_match(
        "/a.mkv", "/b.mkv", "CERTAIN", 0.99, "T1 placeholder",
        audio_json=_json.dumps({"score": 0.85}),
        video_json=_json.dumps({"score": 0.80, "coverage": 0.90, "contiguous_superset": False}),
        scenes_json=_json.dumps({"score": 0.0}))
    # signal-less NAME_COPY row: threshold-independent, must never be re-decided
    store.save(_rec("/n1.mkv"), feature_version=FV)
    store.save(_rec("/n2.mkv"), feature_version=FV)
    store.save_match("/n1.mkv", "/n2.mkv", "NAME_COPY", 0.75, "same name except (N)")

    def _verdict(a):
        return store.conn.execute("SELECT verdict FROM matches WHERE a_path=?", (a,)).fetchone()[0]

    # same thresholds -> idempotent no-op (no verdict changes)
    rep0 = apply_thresholds_to_store(store, th)
    assert rep0["changed"] == 0 and _verdict("/a.mkv") == "CERTAIN"
    assert rep0["skipped_no_signals"] >= 1                          # the NAME_COPY row

    # tighten the AUDIO threshold above the stored audio score (0.85) so the pair no longer clears
    # T1, but the video still corroborates (0.80 >= theta_v, cov 0.90) -> demotes to T4b review.
    # (With lazy audio the audio-only review path is gone, so demotion to PROBABLE now requires the
    # surviving video-corroborated T4b branch.)
    tight_raw = copy.deepcopy(th.raw)
    tight_raw["audio"]["theta_a"] = 0.90
    tight = Thresholds(raw=tight_raw)
    rep = apply_thresholds_to_store(store, tight)
    assert rep["changed"] == 1
    assert rep["transitions"].get("CERTAIN->PROBABLE") == 1
    assert _verdict("/a.mkv") == "PROBABLE"
    assert _verdict("/n1.mkv") == "NAME_COPY"                       # signal-less row untouched

    # counter-check (new semantics): tightening VIDEO below the stored score now drops a pair with
    # no scene corroboration straight to DIFFERENT — the audio-only T4b review path was removed.
    store.save_match(
        "/a.mkv", "/b.mkv", "CERTAIN", 0.99, "T1 placeholder",      # reset the row to CERTAIN
        audio_json=_json.dumps({"score": 0.85}),
        video_json=_json.dumps({"score": 0.80, "coverage": 0.90, "contiguous_superset": False}),
        scenes_json=_json.dumps({"score": 0.0}))
    tv_raw = copy.deepcopy(th.raw)
    tv_raw["video"]["theta_v"] = 0.95
    apply_thresholds_to_store(store, Thresholds(raw=tv_raw))
    assert _verdict("/a.mkv") == "DIFFERENT"


def test_apply_thresholds_action_reapplies_to_store(store, tmp_path):
    """The recalibrate action re-applies the new θ to ALREADY-scanned results when a store is
    passed: it writes the config AND re-decides existing matches (no decode)."""
    import json as _json
    import shutil

    import yaml

    from dupdetect.config import effective_config_path
    from dupdetect.ui import actions

    store.save(_rec("/a.mkv"), feature_version=FV)
    store.save(_rec("/b.mkv"), feature_version=FV)
    store.save_match(
        "/a.mkv", "/b.mkv", "CERTAIN", 0.99, "T1 placeholder",
        audio_json=_json.dumps({"score": 0.85}),
        video_json=_json.dumps({"score": 0.80, "coverage": 0.90, "contiguous_superset": False}),
        scenes_json=_json.dumps({"score": 0.0}))

    cfg = tmp_path / "th.yaml"
    shutil.copyfile(effective_config_path(), cfg)                   # an existing config to base on
    # Raise theta_a above the stored audio (0.85) while keeping theta_v below the video (0.80): T1
    # no longer clears but the video still corroborates -> T4b review. (Lazy audio removed the
    # audio-only review path, so demotion to PROBABLE goes through the video-corroborated T4b.)
    out = actions.apply_thresholds(0.78, 0.90, config_path=str(cfg), store=store)

    assert out == str(cfg)
    assert yaml.safe_load(cfg.read_text(encoding="utf-8"))["video"]["theta_v"] == pytest.approx(0.78)
    v = store.conn.execute("SELECT verdict FROM matches WHERE a_path='/a.mkv'").fetchone()[0]
    assert v == "PROBABLE"                                          # T1 no longer clears -> review


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
    probs = store.problems()
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


# --------------------------------------------------------------- concurrent-deletion guard (§0)

def test_pass2_does_not_resurrect_deleted_candidate(tmp_path, th, store, monkeypatch):
    """§0 concurrent-deletion guard: a candidate trashed MID-SCAN (gone from disk) must NOT be
    re-persisted to `matches`, so the scan can't resurrect the row the UI just forgot (forget_file).
    The coarse index is an in-memory snapshot, so without the guard a deleted file's vector still
    yields a candidate pair and the deletion would silently bounce back into the list."""
    from dupdetect.models import Verdict
    from dupdetect.pipeline import fullscan
    src = tmp_path / "src.mp4"; src.write_bytes(b"x")            # on disk
    gone = str(tmp_path / "gone.mp4")                            # candidate NOT on disk
    store.save(_rec(str(src)), feature_version=FV)

    class _Res:                                                 # a fake match() hit at the gone file
        candidate_path = gone
        verdict = Verdict.CERTAIN
        confidence = 1.0
        reason = "x"
        audio = video = scenes = None
    monkeypatch.setattr(fullscan, "match", lambda *a, **k: [_Res()])
    saved: list = []
    monkeypatch.setattr(store, "save_match", lambda *a, **k: saved.append(a))
    fullscan._pass2_sequential([str(src)], store, object(), th, cache=None, progress=False)
    assert saved == []                                          # nothing persisted for the gone file


# --------------------------------------------------------------- removal reuse (only re-rank changed)

def test_rebuild_clusters_reuse_skips_unchanged(store, th, monkeypatch):
    """On a removal, `_rebuild_clusters(reuse=...)` re-ranks ONLY the cluster whose membership changed;
    untouched clusters keep their persisted KEEP/rank_reason — no whisper/audio on the whole library."""
    from dupdetect.pipeline import fullscan
    for p in ("/A", "/B", "/C", "/X", "/Y"):
        store.save(_rec(p), feature_version=FV)
    for a, b in [("/A", "/B"), ("/B", "/C"), ("/A", "/C")]:        # cluster {A,B,C}
        store.save_match(a, b, "CERTAIN", 0.99, "T1")
    store.save_match("/X", "/Y", "CERTAIN", 0.99, "T1")           # cluster {X,Y}
    fullscan._rebuild_clusters(store, th)                          # initial build (ranks both)

    prior = fullscan._snapshot_clusters(store)                    # snapshot BEFORE the removal
    store.forget_file("/C")                                       # C leaves {A,B,C} -> {A,B}

    calls = {"n": 0}
    real = fullscan.rank_cluster
    monkeypatch.setattr(fullscan, "rank_cluster",
                        lambda m, s, t: (calls.__setitem__("n", calls["n"] + 1) or real(m, s, t)))
    fullscan._rebuild_clusters(store, th, reuse=prior)
    assert calls["n"] == 1                                        # only the changed {A,B} re-ranked
    keeps = {r["path"] for r in store.conn.execute("SELECT path FROM clusters WHERE is_keep=1")}
    assert "/X" in keeps                                          # untouched cluster kept its KEEP


# --------------------------------------------------------------- cluster fusion regression (§0)

def test_rebuild_clusters_no_cross_component_fusion(store, th):
    """REGRESSION: distinct match-components must NEVER share a cluster_id (the bug fused 88/169
    clusters under colliding enumerate() indices during concurrent rebuilds). Two DISCONNECTED
    duplicate pairs must yield two clusters, each = exactly one component."""
    from dupdetect.pipeline import fullscan
    for p in ("/a", "/b", "/c", "/d"):
        store.save(_rec(p), feature_version=FV)
    store.save_match("/a", "/b", "CERTAIN", 0.99, "T1")        # component 1
    store.save_match("/c", "/d", "CERTAIN", 0.99, "T1")        # component 2 (no edge to comp 1)
    fullscan._rebuild_clusters(store, th)
    by_cid = {}
    for r in store.conn.execute("SELECT cluster_id, path FROM clusters"):
        by_cid.setdefault(r["cluster_id"], set()).add(r["path"])
    assert len(by_cid) == 2                                    # two clusters, NOT one fused
    assert sorted(sorted(s) for s in by_cid.values()) == [["/a", "/b"], ["/c", "/d"]]


def test_stable_cluster_id_distinct_and_order_independent():
    """Content-derived ids: same component -> same id (order-independent via min); distinct components
    -> distinct ids. This is what stops concurrent rebuilds from reusing index 0,1,2.. for unrelated
    components and fusing them."""
    from dupdetect.pipeline.fullscan import _stable_cluster_id
    assert _stable_cluster_id(["/a", "/b"]) == _stable_cluster_id(["/b", "/a"])
    assert _stable_cluster_id(["/a", "/b"]) != _stable_cluster_id(["/c", "/d"])


def test_rebuild_clusters_ids_stable_across_runs(store, th):
    """The same component keeps the SAME cluster_id across independent rebuilds (so two concurrent
    rebuilds agree, and the UI/KEEP survives a refresh)."""
    from dupdetect.pipeline import fullscan
    for p in ("/a", "/b"):
        store.save(_rec(p), feature_version=FV)
    store.save_match("/a", "/b", "CERTAIN", 0.99, "T1")
    fullscan._rebuild_clusters(store, th)
    cid1 = store.conn.execute("SELECT DISTINCT cluster_id FROM clusters").fetchone()["cluster_id"]
    fullscan._rebuild_clusters(store, th)                      # rebuild again from scratch
    cid2 = store.conn.execute("SELECT DISTINCT cluster_id FROM clusters").fetchone()["cluster_id"]
    assert cid1 == cid2


def test_replace_clusters_is_full_atomic_replace(store):
    """replace_clusters wipes + rewrites in ONE transaction (last rebuild wins entirely) — no leftover
    rows from a previous rebuild can linger to fuse a cluster."""
    store.replace_clusters([(10, "/a", True, ""), (10, "/b", False, "")])
    store.replace_clusters([(20, "/c", True, ""), (20, "/d", False, "")])
    rows = {(r["cluster_id"], r["path"]) for r in store.conn.execute("SELECT cluster_id, path FROM clusters")}
    assert rows == {(20, "/c"), (20, "/d")}                    # first replace fully gone
