"""Run-at-login helper: command building + OS-support gating (no registry writes in tests)."""
import os

from dupdetect.ui import startup


def test_launch_command_opens_ui_with_db():
    cmd = startup.launch_command(r"C:\Users\x\db.sqlite")
    assert "ui" in cmd and "--db" in cmd and "db.sqlite" in cmd
    assert "--tray" in cmd                              # login entry boots hidden into the system tray


def test_is_supported_matches_os():
    assert startup.is_supported() == (os.name == "nt")


def test_set_enabled_raises_when_unsupported(monkeypatch):
    monkeypatch.setattr(startup, "is_supported", lambda: False)
    import pytest
    with pytest.raises(RuntimeError):
        startup.set_enabled(True, "db.sqlite")
    assert startup.is_enabled() is False
