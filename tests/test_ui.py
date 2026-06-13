"""UI tests: data logic (sort/reclaimable/KEEP), feedback/deletion store, and the
feedback->recalibration bridge. Logic is pure (no Qt); optional Qt smoke tests are at the end.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np
import pytest

from dupdetect.models import AlignResult, Probe, Quality, Record
from dupdetect.store import FingerprintStore
from dupdetect.ui.data import ClusterRow, FileRow, clean_title, is_actionable, load_clusters, sort_clusters


def test_clean_title_strips_release_tags():
    assert clean_title("Sample.Movie.1999.1080p.BluRay.x264-GROUP.mkv") == "Sample Movie 1999"
    assert clean_title("Another Film (2002) [1080p] WEB-DL DTS.mp4") == "Another Film 2002"
    assert clean_title("movie_x265_HDR_2160p.mkv").lower().startswith("movie")


@pytest.fixture
def store(tmp_path):
    s = FingerprintStore(tmp_path / "ui.sqlite")
    yield s
    s.close()


def _rec(path, h=1080, size=1_000_000_000, br=5000, lang="eng", audio_cov=1.0):
    return Record(path=path, mtime=0.0, size=size,
                  probe=Probe(3600.0, h * 16 // 9, h, "h264", br, []),
                  content_hash="h" + path, global_vec=np.zeros(8, np.float32),
                  window_vecs=np.zeros((2, 8), np.float32), embeddings=np.zeros((3, 8), np.float16),
                  audio_fp=np.zeros(1, np.uint32), scene_cuts=np.zeros(1, np.float32),
                  quality=Quality(lang_detected=lang, audio_coverage=audio_cov))


# --------------------------------------------------------------- error summary (scan failure UI)

# The actual Rich-rendered traceback the scan subprocess emitted on the empty-embeddings crash.
_RICH_TRACEBACK = [
    "Pass 2 (duplicates):   0%|          | 0/7 [00:02<?, ?film/s, review=0 | sample_movie_1080p.mp4]",
    "┌───────────────────── Traceback (most recent call last) ─────────────────────┐",
    "│ C:\\...\\src\\dupdetect\\cli.py:129 in scan                                   │",
    "│ > 129 │   │   report = full_scan(files, store, embedder, th, force=force,       │",
    "│ C:\\...\\align\\video.py:63 in align_video                                       │",
    "│ >  63 │   │   sim = (emb_a.float() @ emb_b.t().float()).cpu().numpy()           │",
    "└─────────────────────────────────────────────────────────────────────────────┘",
    "RuntimeError: mat1 and mat2 shapes cannot be multiplied (465x768 and 0x0)",
]


def test_summarize_error_extracts_exception_from_rich_traceback():
    """A normal user must see the REAL cause, not 'exit 1': the exception line is dug out of the
    Rich traceback (box characters and source frames around it are ignored)."""
    from dupdetect.util import summarize_error
    assert (summarize_error(_RICH_TRACEBACK)
            == "RuntimeError: mat1 and mat2 shapes cannot be multiplied (465x768 and 0x0)")


def test_summarize_error_keeps_the_raised_not_inner_exception():
    """With a chained traceback, the LAST exception (the one actually raised) wins."""
    from dupdetect.util import summarize_error
    log = ["KeyError: 'a'", "During handling of the above exception, another occurred:",
           "ValueError: bad value"]
    assert summarize_error(log) == "ValueError: bad value"


def test_summarize_error_framed_exception_line():
    """Some Rich versions wrap the exception INSIDE the panel: un-frame both ends."""
    from dupdetect.util import summarize_error
    assert summarize_error(["│ FileNotFoundError: model.bin is missing │"]) \
        == "FileNotFoundError: model.bin is missing"


def test_summarize_error_fallbacks_when_no_exception():
    """No recognizable exception -> last meaningful line, never an empty/code-only message."""
    from dupdetect.util import summarize_error
    assert summarize_error(["loading…", "ffmpeg: Invalid data found"]) == "ffmpeg: Invalid data found"
    assert summarize_error([]) == "the process exited without output"


# --------------------------------------------------------------- sort / reclaimable / KEEP

def _cl(cid, members, verdict="CERTAIN", conf=0.99):
    return ClusterRow(cid, members, verdict, conf)


def test_sort_by_copies_space_confidence():
    big_space = _cl(0, [FileRow("/k", 2160, size=24_000_000_000, is_keep=True),
                        FileRow("/d", 1080, size=6_000_000_000)])
    many = _cl(1, [FileRow("/k2", 1080, size=2_000_000_000, is_keep=True),
                   FileRow("/a", 720, size=1_000_000_000),
                   FileRow("/b", 480, size=500_000_000)])
    review = _cl(2, [FileRow("/k3", 1080, size=9_000_000_000, is_keep=True),
                     FileRow("/c", 720, size=3_000_000_000)], verdict="PROBABLE", conf=0.65)
    cs = [big_space, many, review]
    assert [c.cluster_id for c in sort_clusters(cs, "copies")][0] == 1        # 3 copies
    assert [c.cluster_id for c in sort_clusters(cs, "space")][0] == 0         # 6 GB reclaimable
    assert [c.cluster_id for c in sort_clusters(cs, "confidence")][0] in (0, 1)  # CERTAIN before PROBABLE
    assert sort_clusters(cs, "confidence")[-1].cluster_id == 2               # PROBABLE at the end


def test_reclaimable_and_deletable_excludes_keep():
    c = _cl(0, [FileRow("/k", 2160, size=24_000_000_000, is_keep=True),
                FileRow("/d1", 1080, size=6_000_000_000),
                FileRow("/d2", 720, size=2_000_000_000)])
    assert c.reclaimable_bytes == 8_000_000_000           # non-keep only
    assert {f.path for f in c.deletable()} == {"/d1", "/d2"}   # KEEP is never deletable


def test_is_actionable():
    assert is_actionable(_cl(0, [], "CERTAIN"))
    assert not is_actionable(_cl(0, [], "PROBABLE"))


def test_drift_report_detects_desync(store):
    """clusters and matches over DISJOINT paths (e.g. exact_scan after full_scan) -> drifted.
    If they share paths (clusters derived from matches) -> not drifted."""
    from dupdetect.ui.data import drift_report

    # matches over (/m1,/m2); clusters over (/c1,/c2): zero overlap -> drifted
    store.save_match("/m1", "/m2", "CERTAIN", 0.99, "T1")
    store.save_cluster(0, "/c1", is_keep=True)
    store.save_cluster(0, "/c2", is_keep=False)
    rep = drift_report(store)
    assert rep["drifted"] is True and rep["orphan_paths"] == 2

    # clusters consistent with matches -> not drifted
    store.clear_clusters()
    store.save_cluster(0, "/m1", is_keep=True)
    store.save_cluster(0, "/m2", is_keep=False)
    assert drift_report(store)["drifted"] is False


def test_load_clusters_from_store(store):
    for p, h, sz in [("/Ik4k", 2160, 24_000_000_000), ("/Ik1080", 1080, 6_000_000_000),
                     ("/Ik720", 720, 1_900_000_000)]:
        store.save(_rec(p, h, sz, lang="jpn"), feature_version="fv")
    for p in ("/Ik4k", "/Ik1080", "/Ik720"):
        store.save_cluster(0, p, is_keep=(p == "/Ik4k"), rank_reason="")
    store.save_match("/Ik4k", "/Ik1080", "CERTAIN", 0.99, "T1")
    cl = load_clusters(store)[0]
    assert cl.n_copies == 3 and cl.keep.path == "/Ik4k"
    assert cl.verdict == "CERTAIN"
    assert cl.reclaimable_bytes == 7_900_000_000          # 6.0 + 1.9 GB (non-keep)


# --------------------------------------------------------------- store: feedback / deletion

def test_save_feedback_canonicalizes(store):
    store.save_feedback("/z.mkv", "/a.mkv", "different")   # stored as (a, z)
    assert store.iter_feedback() == [("/a.mkv", "/z.mkv", "different")]


def test_forget_file_clears_record_matches_cluster_and_npy(store):
    store.save(_rec("/v.mkv"), feature_version="fv")
    store.save(_rec("/w.mkv"), feature_version="fv")
    store.save_match("/v.mkv", "/w.mkv", "CERTAIN", 0.99, "T1")
    store.save_cluster(0, "/v.mkv", is_keep=False)
    npy = list((store.emb_dir).glob("*.npy"))
    assert len(npy) == 2
    store.forget_file("/v.mkv")
    assert store.load("/v.mkv") is None
    assert store.all_matches() == []                      # its match was removed
    assert store.conn.execute("SELECT COUNT(*) FROM clusters WHERE path='/v.mkv'").fetchone()[0] == 0
    assert len(list((store.emb_dir).glob("*.npy"))) == 1  # its .npy deleted


# --------------------------------------------------------------- feedback -> recalibration

def test_labeled_signals_and_recalibration(store):
    from dupdetect.pipeline.calibrate import labeled_signals_from_feedback, suggest_thresholds

    def mj(score, cov=1.0):
        return json.dumps(asdict(AlignResult(score=score, coverage=cov)))

    # 3 real duplicates (high scores) + 1 false positive (low audio, mediocre video)
    pairs = [("/a1", "/b1", 0.95, "same"), ("/a2", "/b2", 0.93, "same"),
             ("/a3", "/b3", 0.97, "same"), ("/x", "/y", 0.30, "different")]
    for a, b, sc, label in pairs:
        store.save_match(a, b, "CERTAIN", 0.99, "T1",
                         audio_json=mj(sc), video_json=mj(sc), scenes_json=mj(sc))
        store.save_feedback(a, b, label)
    sigs = labeled_signals_from_feedback(store)
    assert len(sigs) == 4 and sum(s.is_same for s in sigs) == 3
    sug = suggest_thresholds(sigs)
    assert sug["false_positives_T1T2"] == 0               # goal: zero FP in strong tiers
    assert sug["n_pairs"] == 4


# --------------------------------------------------------------- Qt smoke tests (optional, offscreen)

def test_smoke_gui_offscreen(store, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from dupdetect.ui.model import build_model, checked_files
    for p, keep in [("/k", True), ("/d", False)]:
        store.save(_rec(p), feature_version="fv")
        store.save_cluster(0, p, is_keep=keep)
    store.save_match("/k", "/d", "CERTAIN", 0.99, "T1")
    QApplication.instance() or QApplication([])
    cls = sort_clusters(load_clusters(store), "copies")
    m = build_model(cls)
    cl = m.invisibleRootItem().child(0)
    for j in range(cl.rowCount()):
        it = cl.child(j)
        if it.isCheckable():
            it.setCheckState(Qt.Checked)
    assert len(checked_files(m)) == 1                     # only non-keep (KEEP is locked)


def test_build_model_preserves_selection(store, monkeypatch):
    """Rebuilding the tree (e.g. after 'Set as master') must preserve checked state:
    paths in `checked` remain checked; other non-keep entries are unchecked."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from dupdetect.ui.model import build_model, checked_files
    for p, keep in [("/k", True), ("/d1", False), ("/d2", False)]:
        store.save(_rec(p), feature_version="fv")
        store.save_cluster(0, p, is_keep=keep)
    store.save_match("/k", "/d1", "CERTAIN", 0.99, "T1")
    QApplication.instance() or QApplication([])
    cls = sort_clusters(load_clusters(store), "copies")
    m = build_model(cls, checked={"/d1"})                  # /d1 was previously checked
    assert {p for p, _ in checked_files(m)} == {"/d1"}     # preserved; /d2 remains unchecked


def test_switch_db_reloads_tree(tmp_path, monkeypatch):
    """Selecting a different DB in the panel reloads the tree immediately (no scan needed)."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from dupdetect.ui.main import MainWindow

    def mkdb(name, n):
        st = FingerprintStore(tmp_path / name)
        for cid in range(n):
            for p, keep in ((f"/c{cid}_k", True), (f"/c{cid}_d", False)):
                st.save(_rec(p), feature_version="fv")
                st.save_cluster(cid, p, is_keep=keep)
            st.save_match(f"/c{cid}_k", f"/c{cid}_d", "CERTAIN", 0.99, "T1")
        st.close()
        return str(tmp_path / name)

    a, b = mkdb("A.sqlite", 1), mkdb("B.sqlite", 3)
    QApplication.instance() or QApplication([])
    w = MainWindow(a)
    assert w.model.invisibleRootItem().rowCount() == 1
    w.switch_db(b)                                         # DB switch -> reload
    assert w.model.invisibleRootItem().rowCount() == 3 and w._db_path == b


# ----------------------------------------------------- problem tree (reindex / corrupt)

def test_build_problem_model_detail_not_checkable(monkeypatch):
    """Each file is a checkable node (☑ for VLC/delete) with ONE child detail row (the reason)
    that is NEVER checkable; repair_note overrides the scan error when present."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from dupdetect.ui.model import KIND_ROLE, build_problem_model, checked_problems
    QApplication.instance() or QApplication([])
    items = [("/x/big.mkv", "timeout (>900s)", "reindex", None)]
    m = build_problem_model(items, repairable=True)
    node = m.invisibleRootItem().child(0)
    assert node.isCheckable() and node.data(KIND_ROLE) == "problem"
    detail = node.child(0)
    assert detail.data(KIND_ROLE) == "detail" and not detail.isCheckable()
    # corrupt: reason says 'irrecoverable'; repair_note (if present) replaces the error
    mc = build_problem_model(
        [("/y/a.mp4", "moov atom not found", "corrupt", None),
         ("/y/b.mkv", "x", "corrupt", "remux failed: invalid data")], repairable=False)
    n0, n1 = (mc.invisibleRootItem().child(i) for i in range(2))
    assert "irrecoverable" in n0.child(0).text()
    assert n1.child(0).text().endswith("remux failed: invalid data")
    from PySide6.QtCore import Qt
    node.setCheckState(Qt.Checked)
    assert [p for p, _ in checked_problems(m)] == ["/x/big.mkv"]


def test_mainwindow_populates_problem_tabs(tmp_path, monkeypatch):
    """The window distributes problems into its two trees, sets the count in the tab title,
    and enables Rebuild only when there are repairable entries."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from dupdetect.ui.main import MainWindow
    db = tmp_path / "p.sqlite"
    st = FingerprintStore(db)
    st.save_problem("/r/slow.mkv", "timeout (>900s)")             # reindex candidate
    st.save_problem("/c/dead.mp4", "moov atom not found")         # corrupt candidate
    st.close()
    QApplication.instance() or QApplication([])
    w = MainWindow(str(db))
    assert w.reindex_model.invisibleRootItem().rowCount() == 1
    assert w.corrupt_model.invisibleRootItem().rowCount() == 1
    assert "(1)" in w.tabs.tabText(1) and "(1)" in w.tabs.tabText(2)
    assert w.btn_repair.isEnabled()


# ----------------------------------------------------- audio quality warning (silent copy)

def test_cluster_with_poor_audio_is_not_actionable(store):
    """If a copy has missing/clipped audio, the cluster is NOT actionable (goes to Review) and
    NO KEEP is auto-selected -> the user decides (avoid losing the copy with audio)."""
    store.save(_rec("/full.mkv", audio_cov=1.0), feature_version="fv")
    store.save(_rec("/mute.mkv", audio_cov=0.0), feature_version="fv")     # silent copy
    store.save_cluster(0, "/full.mkv", is_keep=0)                          # rank did not choose keep
    store.save_cluster(0, "/mute.mkv", is_keep=0)
    store.save_match("/full.mkv", "/mute.mkv", "CERTAIN", 0.99, "T1")
    (cl,) = load_clusters(store)
    assert cl.audio_warning and not is_actionable(cl)
    assert {m.path for m in cl.members if m.audio_bad} == {"/mute.mkv"}
    assert cl.keep is None                                                 # no auto-KEEP


def test_rank_cluster_does_not_choose_keep_with_poor_audio(store):
    """rank_cluster returns keep=None (and audio_warning) when any copy has bad audio."""
    from dupdetect.config import load_thresholds
    from dupdetect.pipeline.fullscan import rank_cluster
    store.save(_rec("/a.mkv", h=2160, audio_cov=0.1), feature_version="fv")  # 4K but clipped audio
    store.save(_rec("/b.mkv", h=1080, audio_cov=1.0), feature_version="fv")  # 1080p with audio
    ranked = rank_cluster(["/a.mkv", "/b.mkv"], store, load_thresholds())
    assert ranked["keep"] is None and ranked["audio_warning"] is True


def test_cluster_same_truncated_audio_is_actionable(store):
    """Regression (real pair): BOTH copies share the SAME truncated coverage (same source). Audio
    is not a differentiator -> the cluster must be actionable and KEEP auto-selected, NOT hidden in
    Review. Before: any low coverage forced Review and the obvious dup was invisible by default."""
    store.save(_rec("/copy_a.mkv", h=720, br=781, audio_cov=0.65), feature_version="fv")
    store.save(_rec("/copy_b.mkv", h=720, br=783, audio_cov=0.65), feature_version="fv")
    store.save_cluster(0, "/copy_b.mkv", is_keep=1)            # rank picks one (higher bitrate)
    store.save_cluster(0, "/copy_a.mkv", is_keep=0)
    store.save_match("/copy_a.mkv", "/copy_b.mkv", "CERTAIN", 0.99, "T1")
    (cl,) = load_clusters(store)
    assert not cl.audio_warning and is_actionable(cl)         # same coverage -> not a warning
    assert cl.keep is not None


def test_rank_cluster_picks_keep_when_audio_equal_low(store):
    """rank_cluster auto-picks KEEP when every copy has the SAME (low) coverage: nothing to lose
    audio-wise, so it ranks by quality (here: higher bitrate) instead of forcing Review."""
    from dupdetect.config import load_thresholds
    from dupdetect.pipeline.fullscan import rank_cluster
    store.save(_rec("/x.mkv", h=720, br=783, audio_cov=0.65), feature_version="fv")
    store.save(_rec("/y.mkv", h=720, br=781, audio_cov=0.65), feature_version="fv")
    ranked = rank_cluster(["/x.mkv", "/y.mkv"], store, load_thresholds())
    assert ranked["audio_warning"] is False and ranked["keep"] == "/x.mkv"   # higher bitrate wins


def test_rank_keeps_4k_over_1080p_despite_color_clip(store):
    """Regression (real bug): a 1080p with 0% clip must NOT be picked KEEP over a 4K with 1% clip.
    The color-clip override must never downgrade a real resolution upgrade."""
    from dupdetect.config import load_thresholds
    from dupdetect.models import Probe, Quality, Record
    from dupdetect.pipeline.fullscan import rank_cluster
    from dupdetect.quality.color import ColorStats

    def _crec(path, h, clip, grade):                     # grade differs -> _color_diverges fires
        return Record(path=path, mtime=0.0, size=1,
                      probe=Probe(600.0, h * 16 // 9, h, "h264", 8000, []),
                      content_hash=path, global_vec=np.zeros(8, np.float32),
                      window_vecs=np.zeros((0, 8), np.float32), embeddings=np.zeros((0, 8), np.float16),
                      audio_fp=np.zeros(0, np.uint32), scene_cuts=np.zeros(0, np.float32),
                      quality=Quality(lang_detected="eng", audio_coverage=1.0,
                                      color=ColorStats(clip=clip, cast=grade, saturation=grade,
                                                       contrast=grade)))
    store.save(_crec("/a_4k.mp4", 2160, 0.01, 0.10), feature_version="fv")
    store.save(_crec("/b_4k.mp4", 2160, 0.01, 0.60), feature_version="fv")   # divergent grade
    store.save(_crec("/c_1080.mp4", 1080, 0.00, 0.95), feature_version="fv")  # least clip, lower res
    ranked = rank_cluster(["/a_4k.mp4", "/b_4k.mp4", "/c_1080.mp4"], store, load_thresholds())
    assert ranked["keep"] in ("/a_4k.mp4", "/b_4k.mp4")  # a 4K wins, never the 1080p
    assert ranked["keep"] != "/c_1080.mp4"


def test_cluster_tooltip_explains_the_warning(store=None):
    """The ⚠ needs hover text so a first-time user knows what it means. cluster_tooltip is pure."""
    from dupdetect.ui.data import cluster_tooltip
    ready = _cl(0, [FileRow("/k", 1080, is_keep=True), FileRow("/d", 720)], verdict="CERTAIN")
    assert "ready to act" in cluster_tooltip(ready).lower()           # actionable -> KEEP explained
    aud = _cl(1, [FileRow("/k", 1080, audio_coverage=1.0, is_keep=True),
                  FileRow("/d", 1080, audio_coverage=0.2)], verdict="CERTAIN")
    tip = cluster_tooltip(aud)
    assert "review" in tip.lower() and "audio" in tip.lower()         # audio reason listed
    prob = _cl(2, [FileRow("/k", 1080, is_keep=True), FileRow("/d", 1080)], verdict="PROBABLE")
    assert "PROBABLE" in cluster_tooltip(prob)                        # low-confidence reason listed


def test_store_audio_warnings_lists_deficient(store):
    store.save(_rec("/ok.mkv", audio_cov=1.0), feature_version="fv")
    store.save(_rec("/cut.mkv", audio_cov=0.3), feature_version="fv")
    paths = {p for p, _cov, _dur in store.audio_warnings()}
    assert paths == {"/cut.mkv"}


def test_mainwindow_tab_quality_warnings(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from dupdetect.ui.main import MainWindow
    db = tmp_path / "q.sqlite"
    st = FingerprintStore(db)
    st.save(_rec("/mute.mkv", audio_cov=0.0), feature_version="fv")
    st.close()
    QApplication.instance() or QApplication([])
    w = MainWindow(str(db))
    assert w.audio_model.invisibleRootItem().rowCount() == 1
    assert "(1)" in w.tabs.tabText(3)


def test_close_to_tray_toggle_persists(tmp_path, monkeypatch):
    """The tray toggle that switches the X between 'minimize to tray' and 'exit for real' must be a
    checkable action and persist its choice across sessions (read by closeEvent)."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from dupdetect.ui.main import MainWindow
    db = tmp_path / "c.sqlite"
    FingerprintStore(db).close()
    QApplication.instance() or QApplication([])
    w = MainWindow(str(db))
    assert w._act_close_tray.isCheckable()
    w._set_close_to_tray(False)                          # X exits for real
    assert w._settings.value("close_to_tray", True, bool) is False
    w._set_close_to_tray(True)                           # back to minimize-to-tray (default)
    assert w._settings.value("close_to_tray", True, bool) is True


def test_set_as_master_keeps_cluster_visible(tmp_path, monkeypatch):
    """Regression: 'Set as Master' on a 2-member cluster must NOT remove it from the list. Sorting
    by reclaimable space re-orders it when the KEEP changes (reclaimable = Σ non-keep sizes per
    cluster), so it jumps position; _reveal_cluster keeps it in view + selected."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from dupdetect.store import FingerprintStore
    from dupdetect.ui.model import PATH_ROLE

    db = tmp_path / "t.sqlite"
    st = FingerprintStore(db)
    for p, keep, size in [("/milk_k.mp4", True, 1105), ("/milk_d.mp4", False, 555)]:
        r = _rec(p); r.size = size
        st.save(r, feature_version="fv"); st.save_cluster(0, p, is_keep=keep)
    st.save_match("/milk_k.mp4", "/milk_d.mp4", "CERTAIN", 0.99, "T1")
    st.close()
    QApplication.instance() or QApplication([])
    from dupdetect.ui.main import MainWindow
    w = MainWindow(str(db))
    w.sort.setCurrentIndex(1)            # Most reclaimable space
    w.filt.setCurrentIndex(0)            # CERTAIN + HIGH
    w.refresh()
    root = w.model.invisibleRootItem()
    assert root.rowCount() == 1 and root.child(0).rowCount() == 2     # cluster + 2 members
    # 'Set as Master' on the copy -> moves ★; cluster must stay present and get revealed
    w.store.set_keep(0, "/milk_d.mp4"); w.refresh(); w._reveal_cluster(0)
    root = w.model.invisibleRootItem()
    assert root.rowCount() == 1 and root.child(0).rowCount() == 2     # still here (not removed)
    assert w.tree.currentIndex().isValid()                            # revealed/selected
    assert w.model.itemFromIndex(w.tree.currentIndex().siblingAtColumn(0)).data(PATH_ROLE) == 0


def test_cluster_color_warning_on_grade_divergence():
    """ClusterRow flags 'color differs' when copies' grade (cast/sat/contrast) diverges; identical
    grades do not flag. KEEP is still suggested elsewhere (least-clipped) — this is just a flag."""
    from dupdetect.quality.color import ColorStats
    from dupdetect.ui.data import ClusterRow, FileRow
    diverge = ClusterRow(0, members=[
        FileRow("/a", color=ColorStats(clip=0.01, cast=0.42, saturation=0.63, contrast=0.20)),
        FileRow("/b", color=ColorStats(clip=0.27, cast=0.18, saturation=0.32, contrast=0.34))])
    assert diverge.color_warning is True
    same = ClusterRow(0, members=[
        FileRow("/a", color=ColorStats(0.0, 0.30, 0.50, 0.20)),
        FileRow("/b", color=ColorStats(0.2, 0.30, 0.50, 0.20))])   # only clip differs (not grade)
    assert same.color_warning is False


def test_watch_panel_parses_dup_and_cycle_lines(monkeypatch):
    """WatchPanel turns watcher output lines into signals + the composite status state (self._last,
    self._mode): '🔔 N duplicate' -> duplicate_detected; 'detecting=N' -> live 'detected…'; a cycle
    line -> _last + activity when something changed; the events line -> watchdog/polling mode."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from dupdetect.ui.watch_panel import WatchPanel
    QApplication.instance() or QApplication([])
    wp = WatchPanel("/db.sqlite")
    dups, acts, cycles = [], [], []
    wp.duplicate_detected.connect(dups.append)
    wp.activity.connect(lambda: acts.append(1))
    wp.cycle.connect(lambda i, r, n: cycles.append((i, r, n)))
    wp._handle_line("Filesystem events: subscribed — instant reaction; polling backs off when idle.")
    assert "watchdog" in wp._mode                               # instant mode detected
    wp._handle_line("🔔 3 duplicate cluster(s) just detected:")
    assert dups == [3] and acts == [1]
    wp._handle_line("[12:00:00] detecting=4 new file(s)…")
    assert "detected 4" in wp._last and len(acts) == 2          # live pre-index feedback
    wp._handle_line("[12:00:05] indexed=5 removed=1 new_dups=0 errors=0")
    assert "5 indexed" in wp._last and cycles == [(5, 1, 0)] and len(acts) == 3   # indexed>0 -> activity
    wp._handle_line("[12:01:00] indexed=0 removed=0 new_dups=0 errors=0")
    assert len(acts) == 3                                        # idle cycle -> no extra refresh


def test_main_window_tray_session_count(tmp_path, monkeypatch):
    """Tray tooltip accumulates how many files were analyzed since the watcher started."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from dupdetect.ui.main import MainWindow
    w = MainWindow(str(tmp_path / "u.sqlite"))
    w._on_watch_cycle(5, 1, 0)
    w._on_watch_cycle(3, 0, 2)
    assert w._session_indexed == 8 and "8" in w._tray.toolTip()
    w._really_quit = True
    w.close()


def test_tray_watch_action_reflects_real_state(tmp_path, monkeypatch):
    """Regression: the tray 'Start/Pause watching' label must match the REAL watch state, so
    'Start watching' never stops an already-running watcher (e.g. started from the panel)."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from dupdetect.ui.main import MainWindow
    w = MainWindow(str(tmp_path / "u.sqlite"))
    monkeypatch.setattr(w.watch_panel, "is_watching", lambda: True)
    w._refresh_watch_action()
    assert "Pause" in w._act_watch.text()           # running -> the menu offers Pause
    monkeypatch.setattr(w.watch_panel, "is_watching", lambda: False)
    w._refresh_watch_action()
    assert "Start" in w._act_watch.text()            # stopped -> the menu offers Start
    w._really_quit = True
    w.close()
