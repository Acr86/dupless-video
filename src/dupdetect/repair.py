"""Index reconstruction: remux `-c copy` (LOSSLESS) to fix VALID files with a broken/missing
index (category 'reindex' -> extremely slow seek). Does NOT re-encode: frames and audio are
byte-for-byte identical -> embeddings/fingerprint and therefore the VERDICT do not change
(§0). Deferred and opt-in (user triggers it at the end, never during scan). ATOMIC: remux
to temp -> verify -> replace original (os.replace).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

from dupdetect.features.probe import ffprobe
from dupdetect.util import CREATE_NO_WINDOW

# Containers where +faststart (moving the moov atom to the front) applies; skipped for others.
_FASTSTART_EXTS = {".mp4", ".m4v", ".mov", ".m4a"}


def remux_rebuild_index(path: str, timeout: int = 1800,
                        on_progress: Callable[[float], None] | None = None) -> tuple[bool, str, str]:
    """Rebuilds the index of `path` via lossless remux. Returns (ok, kind, message)
    where `kind` ∈ {'ok','gone','timeout','corrupt'}: caller uses it to FORGET the issue
    ('ok'/'gone'), leave it retryable ('timeout', likely disk contention), or mark it
    unrecoverable ('corrupt'). The original is only replaced if the temp file passes verification.

    `on_progress(frac)`: callback with progress [0..1] for THIS file (from `ffmpeg -progress`,
    out_time/duration) -> UI renders a per-file bar. Optional; without it behavior is identical
    to before (a giant file without this would appear frozen, §2)."""
    src = Path(path)
    if not src.exists():
        return (False, "gone", "file no longer exists")
    try:
        old = ffprobe(str(src))                    # original header (valid even if seek is slow)
    except Exception as e:                          # noqa: BLE001
        return (False, "corrupt", f"original does not open ({e}); corrupt, unrecoverable?")

    tmp = src.with_name(src.stem + ".reindex_tmp" + src.suffix)
    # -progress pipe:1 -nostats: ffmpeg emits parseable `out_time_us=…` blocks (~every 0.5s);
    # these drive the per-file progress %. Errors still go to stderr (captured separately).
    cmd = ["ffmpeg", "-v", "error", "-y", "-fflags", "+genpts", "-i", str(src),
           "-map", "0", "-c", "copy"]
    if src.suffix.lower() in _FASTSTART_EXTS:
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", "-nostats", str(tmp)]

    ok, kind, msg = _run_ffmpeg_streaming(cmd, old.duration_s or 0.0, timeout, on_progress)
    if not ok:
        _unlink(tmp)
        return (False, kind, msg)
    if not tmp.exists():
        return (False, "corrupt", "ffmpeg exited without producing the temp file")

    # Verification: temp file opens and its duration matches the original (±2s or ±2%).
    try:
        new = ffprobe(str(tmp))
    except Exception as e:                          # noqa: BLE001
        _unlink(tmp)
        return (False, "corrupt", f"remux does not verify ({e})")
    tol = max(2.0, 0.02 * (old.duration_s or 0.0))
    if new.duration_s <= 0 or (old.duration_s and abs(new.duration_s - old.duration_s) > tol):
        _unlink(tmp)
        return (False, "corrupt", f"duration mismatch ({new.duration_s:.0f}s vs {old.duration_s:.0f}s)")

    os.replace(str(tmp), str(src))                  # atomic on the same volume
    if on_progress:
        on_progress(1.0)                            # 100% upon confirming the replacement
    return (True, "ok", "index rebuilt")


def _run_ffmpeg_streaming(cmd: list[str], duration: float, timeout: int,
                          on_progress: Callable[[float], None] | None) -> tuple[bool, str, str]:
    """Runs ffmpeg reading its `-progress` output live. A threading.Timer kills the process on
    timeout (covers hangs with no output); stderr goes to a temp file to avoid pipe deadlock.
    Returns (ok, kind, msg) for the ffmpeg stage only."""
    errf = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, text=True,
                                encoding="utf-8", errors="replace",
                                creationflags=CREATE_NO_WINDOW)
    except OSError as e:
        errf.close()
        return (False, "corrupt", f"could not launch ffmpeg ({e})")

    timed_out = {"v": False}

    def _kill():
        timed_out["v"] = True
        try:
            proc.kill()
        except OSError:
            pass

    timer = threading.Timer(timeout, _kill)
    timer.start()
    try:
        for line in proc.stdout:                    # -progress blocks; ends when ffmpeg closes stdout
            if on_progress and duration > 0 and line.startswith("out_time_us="):
                try:
                    us = int(line.split("=", 1)[1])
                    on_progress(max(0.0, min(0.999, us / 1e6 / duration)))
                except ValueError:
                    pass                            # 'N/A' at startup
        proc.wait()
    finally:
        timer.cancel()
        proc.stdout.close()
        errf.seek(0)
        err = errf.read().decode("utf-8", "replace")
        errf.close()

    if timed_out["v"]:
        return (False, "timeout", "remux timed out")
    if proc.returncode != 0:
        return (False, "corrupt", f"ffmpeg: {err.strip()[-200:] or 'error'}")
    return (True, "ok", "")


def _unlink(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


def _unlink(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass
