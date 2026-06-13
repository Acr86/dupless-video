"""Platform/install seam: external-binary resolution precedence.

Pure (no torch / no Qt): only os/sys/fs. Each test clears resolve_binary's cache because it is
process-cached. Files are created on disk because resolution checks os.path.isfile / .is_file.
"""
from __future__ import annotations

import os
from pathlib import Path

from dupdetect import runtime
from dupdetect.runtime import _exe, app_data_dir, default_db_path, os_key, resolve_binary


def _make(path) -> str:
    path.write_text("binary")                    # content irrelevant; resolution only checks existence
    return str(path)


def test_env_override_wins(tmp_path, monkeypatch):
    resolve_binary.cache_clear()
    exe = _make(tmp_path / _exe("ffmpeg"))
    monkeypatch.setenv("DUPDETECT_FFMPEG", exe)
    assert resolve_binary("ffmpeg") == exe


def test_env_override_ignored_if_missing_file(tmp_path, monkeypatch):
    """A stale override path must NOT win -> falls through to the next source (bare name here)."""
    resolve_binary.cache_clear()
    monkeypatch.setenv("DUPDETECT_FFMPEG", str(tmp_path / "does_not_exist"))
    monkeypatch.setenv("DUPDETECT_BUNDLE", str(tmp_path))           # empty bundle dir: isolate from a
    monkeypatch.setattr(runtime.shutil, "which", lambda _n: None)   # dev box that vendored real binaries
    monkeypatch.setattr(runtime, "_next_to_python", lambda _e: None)   # force the bare-name fallback
    assert resolve_binary("ffmpeg") == "ffmpeg"


def test_bundle_used_when_present(tmp_path, monkeypatch):
    """The shipped binary under bundle/bin/<os>/ is the offline 'just works' path."""
    resolve_binary.cache_clear()
    monkeypatch.delenv("DUPDETECT_FFPROBE", raising=False)
    bindir = tmp_path / "bin" / os_key()
    bindir.mkdir(parents=True)
    exe = _make(bindir / _exe("ffprobe"))
    monkeypatch.setenv("DUPDETECT_BUNDLE", str(tmp_path))
    assert resolve_binary("ffprobe") == exe


def test_legacy_fpcalc_env_still_works(tmp_path, monkeypatch):
    resolve_binary.cache_clear()
    monkeypatch.delenv("DUPDETECT_FPCALC", raising=False)
    monkeypatch.delenv("DUPDETECT_BUNDLE", raising=False)
    exe = _make(tmp_path / _exe("fpcalc"))
    monkeypatch.setenv("FPCALC", exe)
    assert resolve_binary("fpcalc") == exe


def test_falls_back_to_path(tmp_path, monkeypatch):
    """No override, no bundle, not next to python -> PATH (shutil.which)."""
    resolve_binary.cache_clear()
    monkeypatch.delenv("DUPDETECT_FFPROBE", raising=False)
    monkeypatch.setenv("DUPDETECT_BUNDLE", str(tmp_path / "empty"))   # nonexistent bundle
    monkeypatch.setattr(runtime, "_next_to_python", lambda _e: None)
    found = str(tmp_path / _exe("ffprobe"))
    monkeypatch.setattr(runtime.shutil, "which", lambda n: found if n == "ffprobe" else None)
    assert resolve_binary("ffprobe") == found


def test_os_key_known_platforms(monkeypatch):
    for plat, key in (("win32", "win"), ("linux", "linux"), ("darwin", "macos")):
        monkeypatch.setattr(runtime.sys, "platform", plat)
        assert os_key() == key


# ------------------------------------------------------ per-user data dir (DEFAULT_DB)

def test_app_data_dir_windows(monkeypatch):
    monkeypatch.delenv("DUPDETECT_DATA_DIR", raising=False)
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\x\AppData\Local")
    assert app_data_dir() == Path(r"C:\Users\x\AppData\Local") / "Dupless Video"


def test_app_data_dir_linux_xdg(monkeypatch):
    monkeypatch.delenv("DUPDETECT_DATA_DIR", raising=False)
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "/home/x/.local/share")
    assert app_data_dir() == Path("/home/x/.local/share/Dupless Video")


def test_app_data_dir_macos_structure(monkeypatch):
    monkeypatch.delenv("DUPDETECT_DATA_DIR", raising=False)
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    assert app_data_dir().parts[-3:] == ("Library", "Application Support", "Dupless Video")


def test_data_dir_env_override_and_default_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DUPDETECT_DATA_DIR", str(tmp_path))
    assert app_data_dir() == tmp_path
    assert default_db_path() == tmp_path / "dupdetect.sqlite"   # no hardcoded machine path


def test_app_data_dir_is_pure_no_mkdir(monkeypatch, tmp_path):
    """Computing the path must NOT create anything (it's used at import as the CLI default)."""
    target = tmp_path / "not_created_yet"
    monkeypatch.setenv("DUPDETECT_DATA_DIR", str(target))
    _ = default_db_path()
    assert not target.exists()


# ------------------------------------------------------ scan-priority lock (watcher yields to scan)

def test_scan_lock_held_then_released(monkeypatch, tmp_path):
    monkeypatch.setenv("DUPDETECT_DATA_DIR", str(tmp_path))
    assert runtime.scan_in_progress() is False
    with runtime.scan_priority_lock():
        assert runtime.scan_in_progress() is True          # watcher must yield while held
    assert runtime.scan_in_progress() is False             # released on exit
    assert not runtime.scan_lock_path().exists()


def test_scan_lock_self_heals_stale_pid(monkeypatch, tmp_path):
    """A lock left by a crashed scan (dead PID) must not freeze the watcher forever."""
    monkeypatch.setenv("DUPDETECT_DATA_DIR", str(tmp_path))
    runtime.scan_lock_path().parent.mkdir(parents=True, exist_ok=True)
    runtime.scan_lock_path().write_text("999999999")       # PID that isn't running
    assert runtime.scan_in_progress() is False             # treated as stale
    assert not runtime.scan_lock_path().exists()           # and cleaned up


def test_viz_signal_toggle(monkeypatch, tmp_path):
    """Live-view streaming follows the panel: on when open, off (no signal) when closed."""
    monkeypatch.setenv("DUPDETECT_DATA_DIR", str(tmp_path))
    assert runtime.viz_enabled() is False                  # closed by default -> no streaming
    runtime.set_viz(True)
    assert runtime.viz_enabled() is True
    runtime.set_viz(False)
    assert runtime.viz_enabled() is False


def test_configure_offline_model(monkeypatch, tmp_path):
    """No bundled model -> online load (False). Vendored repo present -> offline source='local' (True)
    and TORCH_HOME pointed at the bundle so the checkpoint is found without downloading."""
    monkeypatch.setenv("DUPDETECT_BUNDLE", str(tmp_path))
    saved = os.environ.pop("TORCH_HOME", None)             # configure_* mutates env globally -> restore
    try:
        assert runtime.configure_offline_model() is False  # dev box (no bundle/models) -> online
        (tmp_path / "models" / "hub" / "facebookresearch_dinov2_main").mkdir(parents=True)
        assert runtime.configure_offline_model() is True   # vendored repo -> load source='local'
        assert os.environ["TORCH_HOME"] == str(tmp_path / "models")
    finally:
        os.environ.pop("TORCH_HOME", None)
        if saved is not None:
            os.environ["TORCH_HOME"] = saved
