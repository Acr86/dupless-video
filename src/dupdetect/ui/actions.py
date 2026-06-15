"""UI actions: open in VLC and delete files (always to the system Trash — recoverable).
Logic separated from Qt to keep it testable. KEEP never reaches here (filtered by the model)."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

from dupdetect.store import FingerprintStore


def find_vlc() -> str | None:
    """Locates vlc.exe: DUPDETECT_VLC env var -> PATH -> typical Windows paths."""
    env = os.environ.get("DUPDETECT_VLC")
    if env and os.path.isfile(env):
        return env
    which = shutil.which("vlc")
    if which:
        return which
    for p in (r"C:\Program Files\VideoLAN\VLC\vlc.exe",
              r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"):
        if os.path.isfile(p):
            return p
    return None


def open_in_vlc(path: str) -> None:
    """Opens `path` in VLC; if VLC is not found, falls back to the OS default player."""
    vlc = find_vlc()
    if vlc:
        subprocess.Popen([vlc, os.path.normpath(path)])
    else:
        os.startfile(os.path.normpath(path))            # type: ignore[attr-defined]  # Windows


def find_brave() -> str | None:
    """Locate brave.exe: DUPDETECT_BROWSER -> PATH -> typical Windows install paths."""
    env = os.environ.get("DUPDETECT_BROWSER")
    if env and os.path.isfile(env):
        return env
    which = shutil.which("brave")
    if which:
        return which
    for p in (r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
              r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
              os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe")):
        if os.path.isfile(p):
            return p
    return None


def _private_flag(exe: str) -> str:
    """'Private window' flag for the browser family (inferred from the exe name).
    Firefox and Edge use their own flags; the rest (Brave/Chrome/Chromium/Vivaldi) are Chromium."""
    name = os.path.basename(exe).lower()
    if "firefox" in name:
        return "-private-window"
    if "msedge" in name or name == "edge.exe":
        return "--inprivate"
    if "opera" in name:
        return "--private"
    return "--incognito"                                 # brave, chrome, chromium, vivaldi…


def _default_browser() -> str | None:
    """Path to the user's default browser, read from the Windows registry (the user's https
    choice). None if it can't be determined (non-Windows, missing key, odd ProgId)."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                r"SOFTWARE\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice") as k:
            progid = winreg.QueryValueEx(k, "ProgId")[0]
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, progid + r"\shell\open\command") as k:
            cmd = winreg.QueryValueEx(k, None)[0]         # e.g.  "C:\...\app.exe" -- "%1"
    except (OSError, ImportError):
        return None
    import shlex
    try:
        exe = shlex.split(cmd, posix=False)[0].strip('"')  # exe = first token (quoted)
    except ValueError:
        return None
    return exe if os.path.isfile(exe) else None


def web_search(query: str) -> None:
    """Opens a Google search for `query` in a PRIVATE WINDOW: Brave if installed, otherwise the
    OS default browser (also private). Last resort if none is detected: default browser without
    private mode (webbrowser). Only opens a normal web search."""
    from urllib.parse import quote_plus
    url = "https://www.google.com/search?q=" + quote_plus(query)
    exe = find_brave() or _default_browser()
    if exe:
        subprocess.Popen([exe, _private_flag(exe), url])
    else:
        import webbrowser
        webbrowser.open(url)                             # no private mode: no browser detected


@dataclass
class DeleteResult:
    deleted: list[str]
    errors: list[tuple[str, str]]                       # (path, message)
    freed_bytes: int


def delete_files(store: FingerprintStore, files: list[tuple[str, int]]) -> DeleteResult:
    """Sends `files` = [(path, size)] to the system Trash (Recycle Bin on Windows, XDG Trash on
    Linux, Trash on macOS) via send2trash — ALWAYS recoverable; this app never deletes permanently
    (a dedupe tool must not be a one-click way to lose a file). After each: audits it
    (`record_deletion`) and REMOVES it from the index (`forget_file`: row + matches + cluster + .npy)
    so no ghost lingers. Must NEVER receive a KEEP (the model blocks it)."""
    from send2trash import send2trash

    deleted: list[str] = []
    errors: list[tuple[str, str]] = []
    freed = 0
    for path, size in files:
        try:
            send2trash(os.path.normpath(path))
            store.record_deletion(path, "trash", size)
            store.forget_file(path)
            deleted.append(path)
            freed += int(size or 0)
        except Exception as e:                          # noqa: BLE001
            errors.append((path, str(e)))
    if deleted:
        # A cluster left with 1 file is no longer a duplicate group: prune it so
        # it does not appear in the list (the remaining keep is intact in `files` and on disk).
        store.prune_singleton_clusters()
    return DeleteResult(deleted=deleted, errors=errors, freed_bytes=freed)


def apply_thresholds(theta_v: float, theta_a: float, config_path: str | None = None,
                     store=None) -> str:
    """Writes the recalibrated thresholds (only θv/θa, which is what suggest_thresholds sweeps).
    With no explicit path: in the FROZEN app the bundled config is READ-ONLY, so it writes a per-user
    OVERRIDE in the data dir (load_thresholds prefers it) instead of crashing with WinError 5; in dev
    it writes the repo config as before. When `store` is given, the new θ are ALSO re-applied to the
    already-scanned results (re-decides existing matches from their stored signals + rebuilds clusters,
    cheap: no decode), so recalibration affects completed detection too, not only future scans.
    Returns the config path."""
    import sys
    from pathlib import Path

    import yaml

    from dupdetect.config import DEFAULT_CONFIG, effective_config_path, user_thresholds_path
    src = Path(config_path) if config_path else effective_config_path()    # base on the current config
    with open(src, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["video"]["theta_v"] = float(theta_v)
    raw["audio"]["theta_a"] = float(theta_a)
    if config_path:
        dst = Path(config_path)
    elif getattr(sys, "frozen", False):
        dst = user_thresholds_path()                  # bundled config is read-only -> per-user override
    else:
        dst = Path(DEFAULT_CONFIG)                     # dev: repo config (writable), as before
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
    if store is not None:                             # re-apply the new θ to ALREADY-scanned results too
        from dupdetect.config import Thresholds
        from dupdetect.pipeline.calibrate import apply_thresholds_to_store
        apply_thresholds_to_store(store, Thresholds(raw=raw))
    return str(dst)
