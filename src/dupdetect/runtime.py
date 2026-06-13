"""Platform / install seam — the ONE module that knows about the host OS and where the app's
external pieces live. The core (pipeline / match / align / the features' logic) stays
OS-agnostic and only calls in here. Porting to another OS = drop a `bundle/bin/<os>/` folder and
add an install script; this module already branches on `sys.platform`, so the core never changes.

Owns today:
  - resolve_binary(): locate ffmpeg / ffprobe / fpcalc (bundle -> env -> venv -> PATH).

Will own next (same seam, kept here on purpose so the boundary stays in one place):
  - offline model cache: point TORCH_HOME at a bundled DINOv2 repo + checkpoint (no first-run network).
  - per-OS app-data / cache dirs (DB, logs) — %LOCALAPPDATA% / ~/.local/share / ~/Library.
"""
from __future__ import annotations

import os
import shutil
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

APP_NAME = "Dupless Video"          # user-facing product name (data dir, installer, window title)

# sys.platform -> bundle/bin/<os> subfolder name. Unknown platforms fall back to the raw value.
_OS_DIR = {"win32": "win", "linux": "linux", "darwin": "macos"}


def app_data_dir() -> Path:
    """Per-user PERSISTENT data dir (index DB, embeddings, logs) — deliberately NOT a temp dir:
    the index is expensive to rebuild and must survive reboots/cleanup. Cross-platform:
      Windows -> %LOCALAPPDATA%\\Dupless Video
      macOS   -> ~/Library/Application Support/Dupless Video
      Linux   -> $XDG_DATA_HOME/Dupless Video  (or ~/.local/share/Dupless Video)
    Overridable with $DUPDETECT_DATA_DIR. Pure (no mkdir): FingerprintStore creates it on open."""
    env = os.environ.get("DUPDETECT_DATA_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:                                                   # linux / other POSIX -> XDG
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def default_db_path() -> Path:
    """Default SQLite index location inside the per-user data dir (the embeddings/ dir lands
    next to it). Replaces any machine-specific hardcoded path so the app is portable per user."""
    return app_data_dir() / "dupdetect.sqlite"


# --------------------------------------------------------------------------- offline model cache
# A frozen/installed app must run with NO first-run network: the DINOv2 repo + checkpoint are
# vendored into bundle/models and torch.hub loads them locally (validated: source='local' loads in
# ~1.3s, no GitHub). configure_offline_model() points TORCH_HOME at the bundle so the checkpoint is
# found cached; embeddings.py then loads with source='local'. Dev (no bundle) -> normal online load.

def model_dir() -> Path:
    """Bundled model cache = TORCH_HOME for offline loading (bundle/models, holding hub/...)."""
    return bundle_root() / "models"


def model_repo_dir() -> Path:
    """Vendored DINOv2 repo for torch.hub(source='local') — exactly how torch.hub names the clone."""
    return model_dir() / "hub" / "facebookresearch_dinov2_main"


def configure_offline_model() -> bool:
    """If the model is vendored in the bundle, set TORCH_HOME to it (so the checkpoint is found
    locally, no download) and return True -> the caller loads DINOv2 with source='local' (no GitHub).
    No bundle (dev box) -> returns False -> caller does the normal online torch.hub load. Idempotent."""
    md = model_dir()
    if (md / "hub").is_dir():
        os.environ["TORCH_HOME"] = str(md)
        return model_repo_dir().is_dir()
    return False


# --------------------------------------------------------------------------- scan-priority lock
# A user-initiated scan and the background watcher both hit the SAME disk + SQLite DB. On a
# spinning disk that means seek thrashing, and in rollback-journal mode the watcher's writes can
# starve the scan's reads (observed: the scan only finished once the watcher was stopped). So the
# scan takes a cross-process priority lock (a PID lockfile) and the watcher yields while it's held.

def scan_lock_path() -> Path:
    return app_data_dir() / "scan.lock"


def _pid_alive(pid: int) -> bool:
    """Best-effort, cross-platform 'is this PID running?'. On any doubt returns True (conservative:
    the watcher keeps yielding rather than risk thrashing a live scan)."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        os.kill(pid, 0)                                     # POSIX: signal 0 = liveness probe
        return True
    except (OSError, ValueError):
        return False


def scan_in_progress() -> bool:
    """True if a user scan currently holds the priority lock -> the watcher must yield. Self-heals a
    stale lock left by a crashed scan: if the holder PID is gone, the lock is removed and ignored."""
    p = scan_lock_path()
    try:
        pid = int(p.read_text().strip() or "0")
    except (OSError, ValueError):
        return False                                       # no lock (or unreadable) -> free
    if not _pid_alive(pid):
        try:
            p.unlink()
        except OSError:
            pass
        return False
    return True


# --------------------------------------------------------------------------- live-view signal
# The "What the AI sees" panel is ON-DEMAND: the scan emits a frame thumbnail ONLY while the panel
# is open (a signal file exists), so a closed panel costs nothing. File-based cross-process signal,
# same idea as the scan lock. Purely cosmetic (§0: never affects a verdict).

def viz_signal_path() -> Path:
    return app_data_dir() / "viz.on"


def viz_enabled() -> bool:
    """True while the live-view panel is open -> the scan streams frame thumbnails."""
    return viz_signal_path().exists()


def set_viz(on: bool) -> None:
    """UI toggles this when the live-view panel opens/closes."""
    p = viz_signal_path()
    if on:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("1")
    else:
        try:
            p.unlink()
        except OSError:
            pass


@contextmanager
def scan_priority_lock():
    """Held for the whole duration of a user scan (CLI or UI subprocess). The watcher checks
    scan_in_progress() each cycle and skips its disk work while this is held, giving the scan the
    disk/DB uncontended. Released (file deleted) on exit, even on error."""
    p = scan_lock_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(os.getpid()))
    try:
        yield
    finally:
        try:
            p.unlink()
        except OSError:
            pass


def os_key() -> str:
    """Short OS folder name used under bundle/bin/<os>."""
    return _OS_DIR.get(sys.platform, sys.platform)


def bundle_root() -> Path:
    """Root of the assets shipped with the app (`bundle/`). Resolves for BOTH layouts:
      - frozen (PyInstaller): next to the executable (or the extracted sys._MEIPASS dir).
      - source / editable install: the repo root (…/src/dupdetect/runtime.py -> parents[2]).
    Overridable with $DUPDETECT_BUNDLE so an installer can place the assets anywhere.
    The dir need NOT exist (a plain `pip install` dev box has no bundle) -> callers degrade."""
    env = os.environ.get("DUPDETECT_BUNDLE")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):                       # PyInstaller / cx_Freeze
        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
        return Path(base) / "bundle"
    return Path(__file__).resolve().parents[2] / "bundle"   # …/src/dupdetect/runtime.py -> repo root


def cli_subprocess(cli_args: list[str]) -> tuple[str, list[str], str | None]:
    """How the GUI should spawn a dupdetect CLI command (scan/watch) as a SUBPROCESS, working in
    BOTH layouts:
      - FROZEN (.exe): sys.executable is the bundled app, not python, and there is no `src/` to put on
        PYTHONPATH. Launch the exe itself with the bare subcommand (`<exe> scan ...`); the frozen entry
        point routes a leading subcommand to the Typer CLI. Returns pythonpath=None.
      - DEV (source): launch `python -m dupdetect.cli ...` with `src/` on PYTHONPATH.
    Returns (program, argv, pythonpath_or_None). Centralized here so panels stay layout-agnostic."""
    if getattr(sys, "frozen", False):
        return sys.executable, list(cli_args), None
    src = str(Path(__file__).resolve().parents[1])         # …/src/dupdetect/runtime.py -> …/src
    return sys.executable, ["-u", "-m", "dupdetect.cli", *cli_args], src   # -u: unbuffered -> live progress


def _exe(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def _next_to_python(exe: str) -> str | None:
    """pip drops console binaries (e.g. fpcalc) beside the interpreter; find them WITHOUT
    needing the venv activated (Scripts on Windows, bin on POSIX)."""
    base = os.path.dirname(sys.executable)
    for d in (base, os.path.join(base, "Scripts"), os.path.join(os.path.dirname(base), "Scripts")):
        p = os.path.join(d, exe)
        if os.path.isfile(p):
            return p
    return None


@lru_cache(maxsize=None)
def resolve_binary(name: str) -> str:
    """Absolute path (or bare name as a last resort) of an external binary: ffmpeg/ffprobe/fpcalc.

    Precedence — offline-FIRST so a shipped app uses its OWN known-good binaries regardless of the
    user's PATH, while still honoring an explicit override and a plain dev machine:
      1. env override  $DUPDETECT_<NAME>  (e.g. DUPDETECT_FFMPEG); also legacy $FPCALC for fpcalc
      2. bundle/bin/<os>/<exe>            — shipped with the app (the "just works" / offline path)
      3. next to the python executable    — pip-installed console scripts (fpcalc via pip)
      4. PATH                             — system install (winget / apt / brew)
      5. bare name                        — last resort; the OS raises a clear "not found" if missing

    Cached: resolution is process-stable and these binaries are invoked hundreds of times per scan.
    Tests that vary the environment must call `resolve_binary.cache_clear()` between cases.
    """
    exe = _exe(name)
    env = os.environ.get(f"DUPDETECT_{name.upper()}")
    if not env and name == "fpcalc":
        env = os.environ.get("FPCALC")                      # legacy override kept working
    if env and os.path.isfile(env):
        return env
    cand = bundle_root() / "bin" / os_key() / exe
    if cand.is_file():
        return str(cand)
    nxt = _next_to_python(exe)
    if nxt:
        return nxt
    return shutil.which(name) or name
