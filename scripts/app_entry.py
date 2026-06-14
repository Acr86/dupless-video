r"""Frozen-app entry point (the PyInstaller target). Mirrors `dupdetect ui` but is a plain module
with no Typer/CLI layer, so the .exe boots straight into the desktop app.

Boot order matters:
  1. configure_offline_model() -> point torch.hub at the BUNDLED DINOv2 (no first-run GitHub fetch).
     resolve_binary() already prefers bundle/bin, so ffmpeg/ffprobe/fpcalc are offline too.
  2. run() the Qt UI on the per-user DB (%LOCALAPPDATA%\Dupless Video on Windows).

Also runnable from a dev checkout: `python scripts/app_entry.py` (the src path insert is a no-op
when frozen, since dupdetect is collected into the executable)."""
import sys
from pathlib import Path

if not getattr(sys, "frozen", False):                   # dev checkout: make `src/` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dupdetect import runtime                            # noqa: E402
from dupdetect.ui.main import run                        # noqa: E402


def selftest() -> int:
    """Headless OFFLINE self-check (run: `Dupless Video.exe --selftest`). Confirms the FROZEN app
    resolves its BUNDLED model + binaries (no network, no PATH) — the part the GUI startup doesn't
    exercise. Writes a report to the per-user data dir AND stdout, and returns 0 (ok) / 1 (problem),
    so it doubles as a diagnostic to send back if a scan ever fails on the user's machine."""
    import subprocess

    lines, ok = [], True
    lines.append(f"frozen={getattr(sys, 'frozen', False)}  bundle_root={runtime.bundle_root()}")
    configured = runtime.configure_offline_model()
    repo = runtime.model_repo_dir()
    ok &= configured and repo.exists()
    lines.append(f"offline model: configured={configured} repo={repo} exists={repo.exists()}")
    for name in ("ffmpeg", "ffprobe", "fpcalc"):
        runtime.resolve_binary.cache_clear()
        path = runtime.resolve_binary(name)
        bundled = "bundle" in path.lower()
        try:
            r = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=20)
            head = (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else "(no output)"
            ok &= (r.returncode == 0) and bundled
            lines.append(f"{name}: {'BUNDLED' if bundled else 'EXTERNAL'} rc={r.returncode} :: {head[:60]}")
        except Exception as e:  # noqa: BLE001
            ok = False
            lines.append(f"{name}: FAILED to run from {path} :: {e}")
    try:
        import torch
        lines.append(f"torch {torch.__version__} cuda_build={torch.version.cuda} "
                     f"gpu_available={torch.cuda.is_available()}")
    except Exception as e:  # noqa: BLE001
        ok = False
        lines.append(f"torch import FAILED :: {e}")
    # Open a throwaway DB: exercises store._init_schema -> reads the BUNDLED schema.sql. This is the
    # exact path that crashed when schema.sql wasn't collected; the GUI startup smoke test misses it
    # (a crash dialog keeps the process alive), so the self-check must hit it explicitly.
    try:
        import tempfile
        from dupdetect.store import FingerprintStore
        probe_db = os.path.join(tempfile.gettempdir(), "dupless_selftest.sqlite")
        FingerprintStore(probe_db).close()
        try:
            os.remove(probe_db)
        except OSError:
            pass
        lines.append("store schema init: OK (schema.sql found + applied)")
    except Exception as e:  # noqa: BLE001
        ok = False
        lines.append(f"store schema init FAILED :: {e}")
    # Load the global thresholds: exercises config.DEFAULT_CONFIG -> the BUNDLED config/thresholds.yaml
    # (the path that crashed the first scan when it wasn't frozen-aware). The GUI startup doesn't hit it.
    try:
        from dupdetect.config import load_thresholds
        th = load_thresholds()
        lines.append(f"thresholds.yaml: OK (theta_v={th.theta_v} theta_a={th.theta_a})")
    except Exception as e:  # noqa: BLE001
        ok = False
        lines.append(f"thresholds.yaml load FAILED :: {e}")
    # Watchdog: bundled -> the watcher reacts to file changes INSTANTLY (not backoff polling). Not
    # fatal if absent (the watcher degrades to polling), but we report it so a missing bundle shows.
    try:
        from watchdog.observers import Observer  # noqa: F401
        lines.append("watchdog: OK (instant file-change events available)")
    except Exception as e:  # noqa: BLE001
        lines.append(f"watchdog: ABSENT -> watcher will poll only ({e})")
    lines.append(f"SELFTEST {'OK' if ok else 'FAILED'}")
    report = "\n".join(lines)
    print(report)
    try:                                                # also persist (windowed exe has no console)
        out = runtime.app_data_dir() / "selftest.log"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
    except OSError:
        pass
    return 0 if ok else 1


_APP_MUTEX = None  # kept alive for the app's lifetime so the installer/uninstaller can detect us


def _hold_app_mutex(name: str = "DuplessVideoSetupMutex") -> None:
    """Create a named Windows mutex so Inno Setup's `AppMutex` sees the app is RUNNING and asks the
    user to close it before installing/uninstalling (avoids replacing files in use). Held for the
    process lifetime via a module global; harmless no-op off Windows. Only the GUI holds it — the
    transient scan/watch worker subprocesses must NOT (the installer cares about the open window)."""
    global _APP_MUTEX
    if sys.platform != "win32":
        return
    import ctypes
    _APP_MUTEX = ctypes.windll.kernel32.CreateMutexW(None, False, name)


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    # Frozen worker mode: the GUI's scan/watch panels spawn THIS exe (sys.executable is the app, not
    # python) with a bare CLI subcommand — `<exe> scan <folder> ...`. Route it to the Typer CLI instead
    # of opening another window. A leading non-flag token = a subcommand (the GUI launch has no args;
    # --selftest is handled above). See runtime.cli_subprocess.
    if getattr(sys, "frozen", False) and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        from dupdetect.cli import app
        app()                                           # Typer parses sys.argv[1:]
        return 0
    _hold_app_mutex()                                   # let the installer/uninstaller detect us (AppMutex)
    runtime.configure_offline_model()                   # offline DINOv2 (no network on first scan)
    # `--tray` (written by the run-at-login entry): boot straight into the system tray, no window.
    return run(str(runtime.default_db_path()), start_hidden="--tray" in sys.argv)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()                    # frozen Pass-1 ProcessPool workers (Windows spawn)
    raise SystemExit(main())
