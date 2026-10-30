"""Deleting the worse copy in a duplicate pair resolves the cluster: the group (1 member) is removed
from the duplicates list, but the remaining KEEP is NOT touched (still in `files` and on disk)."""
from __future__ import annotations

import os

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
    res = actions.delete_files(s, [(str(disc), 10)])   # always to Trash (recoverable)
    assert res.deleted == [str(disc)]
    assert load_clusters(s) == []             # resolved pair is removed from the duplicates list
    assert keep.exists()                       # the KEEP is NOT deleted from disk
    assert s.load(str(keep)) is not None       # nor from the store
    s.close()


def _pair(tmp_path):
    """A KEEP + discard pair in a fresh store; returns (store, keep, disc)."""
    keep = tmp_path / "K.mkv"; disc = tmp_path / "D.mkv"
    keep.write_bytes(b"k"); disc.write_bytes(b"d")
    s = FingerprintStore(tmp_path / "z.sqlite")
    s.save(_rec(str(keep)), "fv"); s.save(_rec(str(disc)), "fv")
    s.save_cluster(0, str(keep), is_keep=True)
    s.save_cluster(0, str(disc), is_keep=False)
    return s, keep, disc


def test_delete_files_forgets_a_file_already_gone(tmp_path, monkeypatch):
    """A listed file that is ALREADY gone from an online volume (deleted outside the app, or by an
    earlier attempt that errored after the bytes were removed) must leave the list — not fail forever
    with '[Errno 2] ... \\\\?\\M:\\...' on every retry (send2trash's long-path form)."""
    from dupdetect.ui import actions

    s, keep, disc = _pair(tmp_path)
    disc.unlink()
    calls = []
    monkeypatch.setattr("send2trash.send2trash", calls.append)
    res = actions.delete_files(s, [(str(disc), 10)])
    assert res.already_gone == [str(disc)] and res.errors == [] and res.deleted == []
    assert calls == []                                    # nothing to send to the Trash
    assert load_clusters(s) == []                         # ghost row gone, pair resolved
    assert s.conn.execute("SELECT COUNT(*) FROM deletions").fetchone()[0] == 0   # not OUR deletion
    s.close()


def test_delete_files_keeps_a_missing_file_on_an_offline_volume(tmp_path, monkeypatch):
    """Fail-SAFE: if the file's volume is unreachable (NAS disconnected), a missing file is NOT
    forgotten — it reports an error and stays listed (same guard as prune_missing_files)."""
    from dupdetect.ui import actions

    s, keep, disc = _pair(tmp_path)
    monkeypatch.setattr("send2trash.send2trash", lambda p: (_ for _ in ()).throw(OSError("offline")))
    res = actions.delete_files(s, [(str(disc), 10)], exists=lambda p: False)   # whole volume offline
    assert res.deleted == [] and res.already_gone == [] and len(res.errors) == 1
    assert s.load(str(disc)) is not None                  # still indexed
    s.close()


def test_delete_files_counts_a_shell_error_when_the_file_is_gone(tmp_path, monkeypatch):
    """The shell can remove the bytes and still report an error (seen on network shares). The file is
    gone either way: record + forget it instead of stranding a row that errors on every retry."""
    from dupdetect.ui import actions

    s, keep, disc = _pair(tmp_path)

    def _removes_then_fails(p):
        os.remove(p)
        raise OSError(120, "reported failure")
    monkeypatch.setattr("send2trash.send2trash", _removes_then_fails)
    res = actions.delete_files(s, [(str(disc), 10)])
    assert res.deleted == [str(disc)] and res.errors == []
    assert s.conn.execute("SELECT COUNT(*) FROM deletions").fetchone()[0] == 1
    assert load_clusters(s) == []
    s.close()


def test_is_network_path(monkeypatch):
    """UNC shares and mapped network drives have NO Recycle Bin on Windows (send2trash deletes them
    permanently), so the UI must know which paths those are. Local drives are not network."""
    from dupdetect.ui import actions

    monkeypatch.setattr(actions, "_drive_type", lambda root: 4 if root.upper() == "M:\\" else 3)
    monkeypatch.setattr(actions.sys, "platform", "win32")
    assert actions.is_network_path(r"\\nas\share\a.mp4")
    assert actions.is_network_path(r"M:\Media\a.mp4")
    assert not actions.is_network_path(r"L:\Media\a.mp4")
    assert not actions.is_network_path(r"\\?\L:\Media\a.mp4")       # long-path LOCAL form, not UNC


def test_delete_files_audits_network_deletion_as_permanent(tmp_path, monkeypatch):
    """The audit row must tell the truth: a network delete is PERMANENT, not 'trash'."""
    from dupdetect.ui import actions

    s, keep, disc = _pair(tmp_path)
    monkeypatch.setattr("send2trash.send2trash", lambda p: os.remove(p))
    monkeypatch.setattr(actions, "is_network_path", lambda p: True)
    res = actions.delete_files(s, [(str(disc), 10)])
    assert res.deleted == [str(disc)]
    assert s.conn.execute("SELECT dest FROM deletions").fetchone()[0] == "permanent"
    s.close()
