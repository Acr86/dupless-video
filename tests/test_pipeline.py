"""Integration tests for the match() orchestration with heavy leaves STUBBED OUT.

Does not run real features/align/retrieval (still NotImplementedError, no GPU).
Validates the WIRING touched by the fixes:
  - C2: pair dedup shared ACROSS calls (no re-aligning (A,B) in both match(A) and match(B))
  - the tree receives the correct signals and filters out DIFFERENT pairs
  - offset (C3) travels in the Result

Trick: each record embeds an integer id in audio_fp/scene_cuts, and the fake
cache also returns it, so align_* stubs can return a canonical result per pair.
"""
from __future__ import annotations

import numpy as np
import pytest

from dupdetect.config import load_thresholds
from dupdetect.match import matcher
from dupdetect.models import AlignResult, Probe, Quality, Record, Verdict
from dupdetect.store import FingerprintStore

FV = "test|v1"
IDS = {"/A.mkv": 0, "/B.mkv": 1, "/C.mkv": 2}

# Canonical signals per unordered pair (frozenset of ids):
#   A,B -> T1 (strong audio+video) ; any pair with C -> nothing aligns (DIFFERENT)
CANNED = {
    frozenset({0, 1}): dict(
        audio=AlignResult(0.9),
        video=AlignResult(0.9, offset=45.0, coverage=0.9),
        scenes=AlignResult(0.5),
    ),
}


def _rec(path: str) -> Record:
    i = IDS[path]
    return Record(
        path=path, mtime=0.0, size=100,
        probe=Probe(duration_s=7000.0, width=1920, height=1080, vcodec="h264",
                    bitrate_kbps=8000, audio_tracks=[]),
        content_hash="h_" + path,
        global_vec=np.zeros(8, np.float32),
        window_vecs=np.zeros((4, 8), np.float32),
        embeddings=np.zeros((2, 8), np.float16),
        audio_fp=np.array([i], np.uint32),     # id embedded for the stub
        scene_cuts=np.array([float(i)], np.float32),
        quality=Quality(),
    )


class _FakeCache:
    """A1 stub: returns [id] as the 'embedding' so the align_video stub can identify the pair."""
    def get(self, path: str):
        return np.array([float(IDS[path])], np.float32)


def _canned(ka: int, kb: int):
    return CANNED.get(frozenset({ka, kb}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = FingerprintStore(tmp_path / "t.sqlite")
    for p in IDS:
        store.save(_rec(p), feature_version=FV)

    # candidates: all-vs-all, no FAISS
    monkeypatch.setattr(matcher, "candidate_paths",
                        lambda rec, store, index, th: {p for p in IDS if p != rec.path})

    def fake_audio(fa, fb, **k):
        c = _canned(int(fa[0]), int(fb[0]));  return c["audio"] if c else AlignResult(0.0)

    def fake_video(ea, eb, **k):
        c = _canned(int(ea[0]), int(eb[0]));  return c["video"] if c else AlignResult(0.0)

    def fake_scenes(ca, cb, **k):
        c = _canned(int(ca[0]), int(cb[0]));  return c["scenes"] if c else AlignResult(0.0)

    monkeypatch.setattr(matcher, "align_audio", fake_audio)
    monkeypatch.setattr(matcher, "align_video", fake_video)
    monkeypatch.setattr(matcher, "align_scenes", fake_scenes)

    th = load_thresholds()
    yield store, th, _FakeCache()
    store.close()


def test_match_detects_dup_and_filters_different(env):
    store, th, cache = env
    a = store.load("/A.mkv", with_embeddings=False)
    res = matcher.match(a, store, index=None, th=th, cache=cache)
    # only B (T1 CERTAIN); C is filtered out as DIFFERENT
    assert [r.candidate_path for r in res] == ["/B.mkv"]
    assert res[0].verdict == Verdict.CERTAIN
    assert res[0].video.offset == 45.0          # C3: offset travels in the Result


def test_shared_seen_prevents_double_evaluation(env):
    """C2: with a shared `seen` set, pair (A,B) is evaluated in match(A) and NOT
    re-evaluated in match(B). Without sharing, match(B) would return A."""
    store, th, cache = env
    a = store.load("/A.mkv", with_embeddings=False)
    b = store.load("/B.mkv", with_embeddings=False)

    shared: set[tuple[str, str]] = set()
    ra = matcher.match(a, store, index=None, th=th, cache=cache, seen=shared)
    rb = matcher.match(b, store, index=None, th=th, cache=cache, seen=shared)
    assert [r.candidate_path for r in ra] == ["/B.mkv"]
    assert rb == []                              # (A,B) already evaluated -> not repeated

    # counter-check: without shared seen, match(B) does see A
    rb_solo = matcher.match(b, store, index=None, th=th, cache=cache)
    assert [r.candidate_path for r in rb_solo] == ["/A.mkv"]


def test_pass2_load_safe_tolerates_missing_npy(tmp_path):
    """Regression (real DB exposed it): the parallel Pass-2 worker loads embeddings directly; a
    missing/moved .npy must NOT crash it -> _load_safe falls back to a record with empty
    embeddings (video contributes 0; audio/scenes still decide), like the sequential cache path."""
    import numpy as np
    from dupdetect.match.matcher import _load_safe
    from dupdetect.models import Probe, Quality, Record
    from dupdetect.store import FingerprintStore
    s = FingerprintStore(tmp_path / "t.sqlite")
    rec = Record(path="/x.mkv", mtime=1.0, size=10,
                 probe=Probe(duration_s=100.0, width=1920, height=1080, vcodec="h264",
                             bitrate_kbps=8000, audio_tracks=[]),
                 content_hash="h", global_vec=np.random.rand(8).astype(np.float32),
                 window_vecs=np.random.rand(4, 8).astype(np.float32),
                 embeddings=np.random.rand(5, 8).astype(np.float32),
                 audio_fp=np.zeros(1, np.uint32), scene_cuts=np.zeros(1, np.float32),
                 quality=Quality())
    s.save(rec, feature_version="fv")
    # delete the .npy on disk -> store.load(with_embeddings=True) would raise
    for npy in (tmp_path / "embeddings").glob("*.npy"):
        npy.unlink()
    out = _load_safe(s, "/x.mkv")                     # must NOT raise
    assert out is not None and out.embeddings.size == 0   # fell back to empty embeddings
    s.close()


def test_align_video_pair_empty_embeddings_with_frame_times_no_crash():
    """§2 regression: a missing/orphaned .npy loads as EMPTY embeddings while frame_times still
    come from the DB row (non-empty). _align_video_pair must return a zero AlignResult, not crash
    (resample_to_grid would index an empty array -> IndexError)."""
    import numpy as np

    from dupdetect.config import load_thresholds
    from dupdetect.match.matcher import _DictCache, _align_video_pair
    from dupdetect.models import Probe, Quality, Record

    def _rec(p):
        return Record(path=p, mtime=0.0, size=1, probe=Probe(10.0, 100, 100, "h264", 1000, []),
                      content_hash=p, global_vec=np.zeros(8, np.float32),
                      window_vecs=np.zeros((0, 8), np.float32),
                      embeddings=np.empty((0, 8), np.float16), audio_fp=np.zeros(0, np.uint32),
                      scene_cuts=np.zeros(0, np.float32),
                      frame_times=np.array([0.0, 2.0, 4.0], np.float32), quality=Quality())
    ra, rb = _rec("/a"), _rec("/b")
    cache = _DictCache({"/a": np.empty((0, 0), np.float32), "/b": np.empty((0, 0), np.float32)})
    res = _align_video_pair(ra, rb, cache, load_thresholds())
    assert res.score == 0.0                              # no video signal, no crash


def test_ensure_audio_coverage_is_incremental(tmp_path, monkeypatch):
    """Deferred coverage: computed-if-missing (NULL) then REUSED — running Deep after Standard must
    not recompute. Mirrors the on-demand audio-fp pattern."""
    import numpy as np

    from dupdetect.models import Probe, Quality, Record
    from dupdetect.pipeline import analyze
    from dupdetect.store import FingerprintStore
    s = FingerprintStore(tmp_path / "cov.sqlite")
    rec = Record(path="/v.mkv", mtime=0.0, size=1, probe=Probe(120.0, 1920, 1080, "h264", 5000, []),
                 content_hash="h", global_vec=np.zeros(8, np.float32),
                 window_vecs=np.zeros((0, 8), np.float32), embeddings=np.zeros((0, 8), np.float16),
                 audio_fp=np.zeros(0, np.uint32), scene_cuts=np.zeros(0, np.float32),
                 quality=Quality(audio_coverage=None))      # Standard Pass-1 leaves it NULL
    s.save(rec, feature_version="fv")
    assert s.conn.execute("SELECT audio_coverage FROM files").fetchone()[0] is None
    calls = {"n": 0}
    monkeypatch.setattr(analyze, "scan_audio_coverage",
                        lambda p, d, **k: (calls.__setitem__("n", calls["n"] + 1) or 0.6))
    cov1 = analyze.ensure_audio_coverage("/v.mkv", s, 120.0, has_audio=True)
    cov2 = analyze.ensure_audio_coverage("/v.mkv", s, 120.0, has_audio=True)
    assert cov1 == 0.6 and cov2 == 0.6
    assert calls["n"] == 1                                # computed once, reused after (incremental)
    s.close()


def test_emit_viz_burst_writes_jpeg_markers(capsys):
    """The live-view emits one 'VIZ:<base64 jpeg>|<filename>' line PER frame of the burst (playback);
    every payload is a real in-RAM JPEG (no files written)."""
    import base64

    import numpy as np

    from dupdetect.pipeline.analyze import _emit_viz_burst
    batch = np.random.default_rng(0).integers(0, 255, (5, 120, 160, 3), dtype=np.uint8)
    _emit_viz_burst("movie.mp4", batch)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("VIZ:")]
    assert len(lines) == 5                                # one line per frame -> playback sequence
    for ln in lines:
        b64, name = ln[4:].rsplit("|", 1)
        assert name == "movie.mp4"
        assert base64.b64decode(b64)[:2] == b"\xff\xd8"   # JPEG magic bytes


def test_frames_to_rgb_uint8_denormalizes_real_tensor():
    """Regression: real decode_frames output is an ImageNet-normalized torch BATCH [K,3,H,W], NOT
    numpy uint8 — the live-view must de-normalize it (this was why the panel showed nothing)."""
    import numpy as np
    import torch

    from dupdetect.features.frames import DINOV2_MEAN, DINOV2_STD
    from dupdetect.pipeline.analyze import _frames_to_rgb_uint8
    rgb = np.random.default_rng(0).integers(0, 255, (2, 4, 4, 3), dtype=np.uint8)
    x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
    norm = (x - torch.tensor(DINOV2_MEAN).view(1, 3, 1, 1)) / torch.tensor(DINOV2_STD).view(1, 3, 1, 1)
    out = _frames_to_rgb_uint8(norm)
    assert out.shape == (2, 4, 4, 3) and out.dtype == np.uint8
    assert np.abs(out.astype(int) - rgb.astype(int)).max() <= 1   # recovers the original (± rounding)


def test_maybe_emit_viz_silent_when_panel_closed(capsys, monkeypatch, tmp_path):
    """ON-DEMAND: with no live-view panel open (no signal file), the scan emits NOTHING (zero cost)."""
    import numpy as np

    from dupdetect.pipeline import analyze
    monkeypatch.setenv("DUPDETECT_DATA_DIR", str(tmp_path))   # no viz.on -> disabled
    analyze.maybe_emit_viz("/m.mp4", np.zeros((4, 8, 8, 3), np.uint8))
    assert capsys.readouterr().out == ""


class _FakeTorchEmpty:
    """Reproduces the torch-tensor surface that broke the guard: `.size` is a METHOD (so the old
    numpy-style `e.size == 0` compared a bound method to 0 -> always False), `.numel()` reports 0.
    EmbeddingCache hands back exactly these (CUDA tensors), so the guard must be count-agnostic."""
    def numel(self):
        return 0

    def size(self):
        return (0, 0)


def test_emb_is_empty_handles_torch_size_method():
    """Unit regression: the emptiness guard must treat a torch-like empty tensor as empty even
    though its `.size` is a method, while still working for numpy arrays (int `.size`) and None."""
    import numpy as np

    from dupdetect.match.matcher import _emb_is_empty
    assert _emb_is_empty(_FakeTorchEmpty()) is True      # the bug: `.size == 0` was always False
    assert _emb_is_empty(None) is True
    assert _emb_is_empty(np.empty((0, 8), np.float32)) is True
    assert _emb_is_empty(np.zeros((3, 8), np.float32)) is False


def test_align_video_pair_empty_torch_tensor_no_crash():
    """§2 regression (the real crash): EmbeddingCache returns torch tensors, whose `.size` is a
    METHOD, so an orphaned .npy that loads as an EMPTY CUDA tensor slipped past the guard and
    crashed align_video's matmul (mat1 465x768 @ mat2 0x0). Must skip-and-report, not crash."""
    import numpy as np

    from dupdetect.config import load_thresholds
    from dupdetect.match.matcher import _DictCache, _align_video_pair
    from dupdetect.models import Probe, Quality, Record

    def _rec(p):
        return Record(path=p, mtime=0.0, size=1, probe=Probe(10.0, 100, 100, "h264", 1000, []),
                      content_hash=p, global_vec=np.zeros(8, np.float32),
                      window_vecs=np.zeros((0, 8), np.float32),
                      embeddings=np.empty((0, 8), np.float16), audio_fp=np.zeros(0, np.uint32),
                      scene_cuts=np.zeros(0, np.float32),
                      frame_times=np.array([0.0, 2.0, 4.0], np.float32), quality=Quality())
    ra, rb = _rec("/a"), _rec("/b")
    cache = _DictCache({"/a": _FakeTorchEmpty(), "/b": _FakeTorchEmpty()})
    res = _align_video_pair(ra, rb, cache, load_thresholds())
    assert res.score == 0.0                              # guard catches torch empties -> no matmul
