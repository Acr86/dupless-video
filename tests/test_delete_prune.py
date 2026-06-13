"""Deleting the worse copy in a duplicate pair resolves the cluster: the group (1 member) is removed
from the duplicates list, but the remaining KEEP is NOT touched (still in `files` and on disk)."""
from __future__ import annotations

import numpy as np

from dupdetect.models import Probe, Quality, Record
from dupdetect.store import FingerprintStore
from dupdetect.ui.data import load_clusters


def _rec(p, sz=1_000_000_000):
    return Record(path=p, mtime=0.0, size=sz,
                  probe=Probe(3600.0, 1920, 1080, "h264", 5000, []),
                  content_hash="h" + p, global_vec=np.zeros(8, np.float32),
                  window_vecs=np.zeros((2, 8), np.float32),
                  embeddings=np.zeros((3, 8), np.float16), audio_fp=np.zeros(1, np.uint32),
                  scene_cuts=np.zeros(1, np.float32), quality=Quality(lang_detected="eng"))


def test_prune_singleton_clusters(tmp_path):
    s = FingerprintStore(tmp_path / "x.sqlite")
    for p in ("/K", "/D"):
        s.save(_rec(p), "fv")
    s.save_cluster(0, "/K", is_keep=True)
    s.save_cluster(0, "/D", is_keep=False)
    s.forget_file("/D")                       # remove D from cluster -> leaves {K}
    s.prune_singleton_clusters()
    assert load_clusters(s) == []             # cluster resolved -> off the list
    assert s.load("/K") is not None           # the keep is still intact
    s.close()


def test_delete_files_prunes_and_preserves_keep(tmp_path, monkeypatch):
    from dupdetect.ui import actions

    keep = tmp_path / "K.mkv"; disc = tmp_path / "D.mkv"
    keep.write_bytes(b"k"); disc.write_bytes(b"d")
    s = FingerprintStore(tmp_path / "y.sqlite")
    s.save(_rec(str(keep)), "fv"); s.save(_rec(str(disc)), "fv")
    s.save_cluster(0, str(keep), is_keep=True)
    s.save_cluster(0, str(disc), is_keep=False)

    monkeypatch.setattr("send2trash.send2trash", lambda p: None)   # don't touch the real Trash
    res = actions.delete_files(s, [(str(disc), 10)], dest="trash")
    assert res.deleted == [str(disc)]
    assert load_clusters(s) == []             # resolved pair is removed from the duplicates list
    assert keep.exists()                       # the KEEP is NOT deleted from disk
    assert s.load(str(keep)) is not None       # nor from the store
    s.close()
