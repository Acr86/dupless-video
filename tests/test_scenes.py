"""Scene tests (step 6): cut detection + DTW over intervals.

align_scenes (DTW) is tested with synthetic cut arrays. scene_cuts is tested by
generating a video with a hard black->white cut (PySceneDetect must detect it).
"""
from __future__ import annotations

import numpy as np
import pytest

from dupdetect.align.scenes import align_scenes

torch = pytest.importorskip("torch")
av = pytest.importorskip("av")


# --------------------------------------------------------------- align_scenes

def test_identical_cuts_high_score():
    cuts = np.array([10.0, 25.0, 40.0, 80.0, 95.0], dtype=np.float32)
    r = align_scenes(cuts, cuts)
    assert r.score == pytest.approx(1.0, abs=1e-3)


def test_invariant_to_initial_trim():
    # same INTERVALS, shifted +100s (trim/intro): signature does not change
    cuts_a = np.array([10.0, 25.0, 40.0, 80.0], dtype=np.float32)
    cuts_b = cuts_a + 100.0
    r = align_scenes(cuts_a, cuts_b)
    assert r.score == pytest.approx(1.0, abs=1e-3)


def test_different_films_low_score():
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)        # uniform 10s intervals
    b = np.array([3.0, 40.0, 47.0, 95.0, 110.0], dtype=np.float32)        # irregular
    r = align_scenes(a, b)
    assert r.score < 0.7


def test_robust_to_spurious_cut():
    # b = a with one extra spurious cut (splits an interval in two) -> still reasonably high
    a = np.array([0.0, 30.0, 60.0, 90.0, 120.0], dtype=np.float32)        # 30s intervals
    b = np.array([0.0, 30.0, 45.0, 60.0, 90.0, 120.0], dtype=np.float32)  # spurious cut at 45
    r = align_scenes(a, b)
    assert r.score > 0.6


def test_too_few_cuts_returns_zero():
    assert align_scenes(np.array([5.0]), np.array([5.0, 10.0])).score == 0.0


# --------------------------------------------------------------- mode B (from emb)

def test_scene_cuts_from_embeddings_detects_change():
    from dupdetect.features.scenes import scene_cuts_from_embeddings
    # 3 frames scene A + 3 scene B (orthogonal vectors) -> 1 cut at frame 3
    a = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    b = np.tile([0.0, 1.0, 0.0, 0.0], (3, 1))
    emb = np.vstack([a, b]).astype(np.float16)
    times = np.array([0.0, 3.1, 6.0, 9.4, 12.0, 15.2], dtype=np.float32)   # irregular keyframes
    cuts = scene_cuts_from_embeddings(emb, times, sim_threshold=0.6)
    assert len(cuts) == 1 and cuts[0] == pytest.approx(9.4)   # REAL timestamp of frame 3


def test_scene_cuts_from_embeddings_no_changes():
    from dupdetect.features.scenes import scene_cuts_from_embeddings
    emb = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)).astype(np.float16)
    times = np.arange(10, dtype=np.float32)
    assert len(scene_cuts_from_embeddings(emb, times)) == 0


def test_feature_version_scene_mode_differs():
    from dupdetect.features.embeddings import Embedder
    from dupdetect.pipeline.analyze import feature_version
    e = Embedder(fps=2.0)
    assert feature_version(e, independent_scenes=True) != feature_version(e, independent_scenes=False)


# --------------------------------------------------------------- fix DTW (band)

def test_very_different_cut_count_gives_zero():
    # a real pair with VERY different cut counts -> incompatible -> score 0
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.uniform(2, 60, size=200))
    b = np.cumsum(rng.uniform(2, 60, size=300))         # +50% cuts
    assert align_scenes(a, b).score == 0.0


def test_robust_to_camrip_on_long_sequence():
    # cam rip: +15% spurious cuts but same edit -> still HIGH (band 0.2 tolerates it)
    rng = np.random.default_rng(1)
    iv = rng.uniform(2, 60, size=200)
    same = np.cumsum(iv)
    cam = np.cumsum(np.concatenate([iv, rng.uniform(2, 60, 30)]))
    assert align_scenes(same, same).score > 0.95
    assert align_scenes(same, cam).score > 0.85         # cam rip not missed


def test_align_scenes_bounded_with_degenerate_cuts():
    # non-monotonic / negative cuts must not produce a score outside [0,1] (keyframe bug)
    a = np.array([0.0, 50.0, 10.0, 80.0, 30.0], dtype=np.float32)   # out of order
    b = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)            # duplicates (zero intervals)
    r = align_scenes(a, b)
    assert 0.0 <= r.score <= 1.0


def test_scene_cuts_from_embeddings_output_sorted():
    from dupdetect.features.scenes import scene_cuts_from_embeddings
    e = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float16)   # alternating -> 3 cuts
    t = np.array([10.0, 2.0, 8.0, 4.0], dtype=np.float32)              # unsorted timestamps
    cuts = scene_cuts_from_embeddings(e, t, sim_threshold=0.5)
    assert list(cuts) == sorted(cuts)                                  # output must be ascending


def test_unrelated_same_length_scores_lower_than_self():
    # same length, different content: with band + gaps scores NOTABLY lower
    # than self-match (unconstrained warping used to leave it at ~0.83). The final
    # fine separation is set by theta_s at calibration time.
    rng = np.random.default_rng(2)
    a = np.cumsum(rng.gamma(2.0, 8.0, size=200))
    b = np.cumsum(rng.gamma(2.0, 8.0, size=200))
    s_same = align_scenes(a, a).score
    s_diff = align_scenes(a, b).score
    assert s_same > 0.95
    assert s_diff < s_same - 0.25                        # clear separation


# --------------------------------------------------------------- scene_cuts (real)

def _make_two_scene_video(path, fps=10, w=128, h=128):
    """First half black, second half white -> one hard cut in the middle."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for i in range(50):
        val = 0 if i < 25 else 255
        arr = np.full((h, w, 3), val, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_scene_cuts_detecta_corte_duro(tmp_path):
    from dupdetect.features.scenes import scene_cuts
    p = tmp_path / "twoscene.mp4"
    _make_two_scene_video(p)
    cuts = scene_cuts(str(p), threshold=0.3)           # ffmpeg scene in [0,1]
    assert cuts.dtype == np.float32
    assert len(cuts) >= 1                              # detects the black->white cut
    assert any(2.0 < c < 3.0 for c in cuts)            # cut is near 2.5s


def test_scene_cuts_non_cp1252_filename_does_not_crash(tmp_path):
    """Regression: a filename with bytes outside cp1252 (Cyrillic -> 0x81 in UTF-8)
    appeared in the ffmpeg log and broke decoding with text=True. Now uses UTF-8."""
    from dupdetect.features.scenes import scene_cuts
    p = tmp_path / "тест_видео.mp4"                    # Cyrillic
    _make_two_scene_video(p)
    cuts = scene_cuts(str(p), threshold=0.3)           # must not raise UnicodeDecodeError
    assert cuts.dtype == np.float32


def test_scene_cuts_unreadable_file_raises(tmp_path):
    from dupdetect.features.scenes import scene_cuts
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video" * 50)
    import pytest as _pytest
    with _pytest.raises(RuntimeError):                 # _pass1 catches it -> problems table
        scene_cuts(str(bad))
