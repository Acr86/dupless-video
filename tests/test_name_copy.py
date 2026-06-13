"""Name-copy detection (N): normalization, same-folder grouping with content veto,
keep by shortest path, and set_keep (manual master)."""
from __future__ import annotations

import numpy as np

from dupdetect.config import load_thresholds
from dupdetect.models import Probe, Quality, Record
from dupdetect.names import base_stem, name_sibling_pairs
from dupdetect.pipeline.fullscan import _apply_name_grouping, _rebuild_clusters, rank_cluster
from dupdetect.store import FingerprintStore


def _rec(p, h=1080, sz=1_000_000_000, br=5000, lang="eng"):
    return Record(path=p, mtime=0.0, size=sz,
                  probe=Probe(3600.0, h * 16 // 9, h, "h264", br, []),
                  content_hash="h" + p, global_vec=np.zeros(8, np.float32),
                  window_vecs=np.zeros((2, 8), np.float32),
                  embeddings=np.zeros((3, 8), np.float16), audio_fp=np.zeros(1, np.uint32),
                  scene_cuts=np.zeros(1, np.float32), quality=Quality(lang_detected=lang))


# --------------------------------------------------------------- normalization
def test_base_stem_strips_copy_markers_not_years_or_parts():
    assert base_stem("movie (1)") == "movie"
    assert base_stem("movie(2)") == "movie"
    assert base_stem("movie (1) (1)") == "movie"      # repeated copy markers
    assert base_stem("Movie (2009)") == "Movie (2009)"   # year (4 digits) is NOT a copy marker
    assert base_stem("CD1") == "CD1"                 # no parens -> part label, not copy
    assert base_stem("movie") == "movie"


def test_name_sibling_pairs_same_folder_and_base_only():
    paths = [r"C:\m\movie.avi", r"C:\m\movie (1).avi", r"C:\m\movie(2).avi",  # copies
             r"C:\m\other.avi",                                           # different base
             r"C:\b\movie (1).avi",                                        # different folder
             r"C:\m\Movie (2009).avi", r"C:\m\Movie (2010).avi"]          # years, not copies
    pairs = name_sibling_pairs(paths)
    assert sorted(pairs) == sorted([(r"C:\m\movie.avi", r"C:\m\movie (1).avi"),
                                    (r"C:\m\movie.avi", r"C:\m\movie(2).avi")])


# --------------------------------------------------------------- grouping + veto
def _rec_emb(p, emb):
    r = _rec(p); r.embeddings = emb
    return r


def test_name_grouping_creates_cluster_and_keep_is_shortest_path(tmp_path):
    s = FingerprintStore(tmp_path / "n.sqlite")
    a = str(tmp_path / "movie.avi"); b = str(tmp_path / "movie (1).avi")
    emb = np.zeros((5, 8), np.float16); emb[:, 0] = 1.0   # identical content -> veto passes
    s.save(_rec_emb(a, emb), "fv"); s.save(_rec_emb(b, emb), "fv")
    th = load_thresholds()
    _apply_name_grouping(s, th)
    assert s.has_match(a, b)                          # NAME_COPY added (content corroborates)
    cl = _rebuild_clusters(s, th)
    assert len(cl) == 1
    assert cl[0]["keep"] == a                         # shortest path = the original (no (N))
    assert set(cl[0]["discard"]) == {b}
    s.close()


def test_name_grouping_respects_content_veto(tmp_path):
    s = FingerprintStore(tmp_path / "v.sqlite")
    a = str(tmp_path / "movie.avi"); b = str(tmp_path / "movie (1).avi")
    s.save(_rec(a), "fv"); s.save(_rec(b), "fv")
    s.save_match(a, b, "DIFFERENT", 0.0, "T5 no alignment")   # content says DIFFERENT
    th = load_thresholds()
    _apply_name_grouping(s, th)
    assert _rebuild_clusters(s, th) == []            # content veto: not grouped
    s.close()


def test_name_grouping_can_be_disabled(tmp_path):
    s = FingerprintStore(tmp_path / "off.sqlite")
    a = str(tmp_path / "movie.avi"); b = str(tmp_path / "movie (1).avi")
    s.save(_rec(a), "fv"); s.save(_rec(b), "fv")
    th = load_thresholds()
    th.raw["quality"]["name_copy_grouping"] = False
    _apply_name_grouping(s, th)
    assert not s.has_match(a, b)
    s.close()


# --------------------------------------------------------------- shortest path as keep
def test_rank_cluster_tiebreaks_by_shortest_path(tmp_path):
    s = FingerprintStore(tmp_path / "r.sqlite")
    short = "/m/movie.avi"; long = "/m/movie (1).avi"
    s.save(_rec(short), "fv"); s.save(_rec(long), "fv")   # identical except for path
    ranked = rank_cluster([long, short], s, load_thresholds())
    assert ranked["keep"] == short                    # among equivalent copies, the shorter path wins
    s.close()


# --------------------------------------------------------------- manual master
def test_set_keep_moves_the_star(tmp_path):
    s = FingerprintStore(tmp_path / "k.sqlite")
    a, b = "/m/movie.avi", "/m/movie (1).avi"
    s.save(_rec(a), "fv"); s.save(_rec(b), "fv")
    s.save_cluster(0, a, is_keep=True); s.save_cluster(0, b, is_keep=False)
    s.set_keep(0, b)                                   # user promotes the copy to master
    keeps = {r["path"]: r["is_keep"] for r in
             s.conn.execute("SELECT path, is_keep FROM clusters WHERE cluster_id=0")}
    assert keeps[b] == 1 and keeps[a] == 0
    s.close()


def test_rank_cluster_prefers_least_clipped(tmp_path):
    """Color (latest stage): among same-resolution copies the KEEP is the LEAST-clipped one (the
    original); a heavily color-corrected copy (crushed blacks ~27%) is not chosen as KEEP."""
    from dupdetect.quality.color import ColorStats
    s = FingerprintStore(tmp_path / "t.sqlite")
    orig = _rec(r"C:\m\v aaa.mp4"); orig.quality.color = ColorStats(clip=0.01)
    corr = _rec(r"C:\m\v bbb.mp4"); corr.quality.color = ColorStats(clip=0.27)
    s.save(orig, "fv"); s.save(corr, "fv")
    ranked = rank_cluster([orig.path, corr.path], s, load_thresholds())
    assert ranked["keep"] == orig.path                             # least-clipped wins (same res)
    s.close()


def test_rank_cluster_color_divergence_prefers_original_over_higher_res(tmp_path):
    """When a copy was re-graded (color diverges) AND a corrected copy is higher-res, KEEP must be
    the LEAST-CLIPPED (preserved original), not the higher-res clipped re-grade. Mirrors a real
    library case: a clean 1440x1080 original vs a 1920x1080 color-corrected re-grade, blacks crushed."""
    from dupdetect.quality.color import ColorStats
    s = FingerprintStore(tmp_path / "t.sqlite")
    orig = _rec(r"C:\m\v aaa.mp4", h=720)          # lower res, clean
    orig.quality.color = ColorStats(clip=0.004, cast=0.42, saturation=0.62, contrast=0.20)
    corr = _rec(r"C:\m\v bbb.mp4", h=1080)         # higher res, but heavily clipped + re-graded
    corr.quality.color = ColorStats(clip=0.26, cast=0.18, saturation=0.32, contrast=0.34)
    s.save(orig, "fv"); s.save(corr, "fv")
    ranked = rank_cluster([orig.path, corr.path], s, load_thresholds())
    assert ranked["keep"] == orig.path             # least-clipped wins on color divergence
    s.close()


def test_name_grouping_vetoes_unsaved_different(tmp_path):
    """The real bug: name-siblings whose content is DIFFERENT (and NOT persisted, since match()
    drops DIFFERENT verdicts) must NOT be grouped. NAME_COPY re-verifies content -> different
    videos that reuse a '(N)' name stay separate (zero-FP, §0)."""
    s = FingerprintStore(tmp_path / "d.sqlite")
    a = str(tmp_path / "movie.avi"); b = str(tmp_path / "movie (1).avi")
    ea = np.zeros((5, 8), np.float16); ea[:, 0] = 1.0
    eb = np.zeros((5, 8), np.float16); eb[:, 1] = 1.0     # orthogonal embeddings -> content DIFFERENT
    s.save(_rec_emb(a, ea), "fv"); s.save(_rec_emb(b, eb), "fv")
    _apply_name_grouping(s, load_thresholds())
    assert not s.has_match(a, b)                          # vetoed: not grouped
    s.close()


def test_name_grouping_skips_lite_without_embeddings(tmp_path):
    """LITE/exact-only siblings (no embeddings) can't be content-verified -> NOT grouped (blind
    NAME_COPY on reused names caused false positives). They group only after a full re-index."""
    s = FingerprintStore(tmp_path / "l.sqlite")
    a = str(tmp_path / "movie.avi"); b = str(tmp_path / "movie (1).avi")
    empty = np.empty((0, 8), np.float16)
    s.save(_rec_emb(a, empty), "fv"); s.save(_rec_emb(b, empty), "fv")
    _apply_name_grouping(s, load_thresholds())
    assert not s.has_match(a, b)                          # unverifiable -> not grouped
    s.close()
