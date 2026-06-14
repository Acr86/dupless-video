"""'Run at login' helper — isolated, per-OS (keeps the rest of the app cross-platform).

Windows: a value under HKCU\\…\\Run. macOS (launchd plist) and Linux (XDG autostart .desktop) are
not wired yet — `is_supported()` returns False there and the UI disables the toggle, so nothing
breaks; only this small glue is platform-specific.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_APP_NAME = "Dupless Video"


def is_supported() -> bool:
    return os.name == "nt"


def launch_command(db: str) -> str:
    """Command that opens the app at login, HIDDEN in the system tray (`--tray`), where it auto-resumes
    the watcher — no window pops up on every boot. Frozen (.exe): just the installed app. Source: the
    installed `dupdetect` console script (no PYTHONPATH needed), else pythonw -m (no console window)."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --tray'
    exe = shutil.which("dupdetect")
    if exe:
        return f'"{exe}" ui --db "{db}" --tray'
    pyw = Path(sys.executable).with_name("pythonw.exe")
    runner = str(pyw) if pyw.exists() else sys.executable
    return f'"{runner}" -m dupdetect.cli ui --db "{db}" --tray'


def is_enabled() -> bool:
    if not is_supported():
        return False
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as k:
            winreg.QueryValueEx(k, _APP_NAME)
            return True
    except OSError:
        return False


def set_enabled(enabled: bool, db: str) -> None:
    """Add/remove the login entry. Raises if the OS isn't wired (caller shows the message)."""
    if not is_supported():
        raise RuntimeError("Run-at-login is only wired for Windows so far.")
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        if enabled:
            winreg.SetValueEx(k, _APP_NAME, 0, winreg.REG_SZ, launch_command(db))
        else:
            try:
                winreg.DeleteValue(k, _APP_NAME)
            except OSError:
                pass                                   # already absent -> fine
