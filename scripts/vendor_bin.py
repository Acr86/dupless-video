"""Vendor the external binaries (ffmpeg, ffprobe, fpcalc) into `bundle/bin/<os>/` so a frozen /
installed app runs OFFLINE with its OWN known-good tools (resolve_binary precedence #2), regardless
of the user's PATH. Mirrors scripts/vendor_model.py (which vendors the DINOv2 cache).

Strategy: COPY the binaries this dev machine already resolves (the same ones the app has been
validated against) — no network download, no version drift. Verifies each copy actually EXECUTES
from the bundle location (catches a dynamically-linked build that would fail without its DLLs).

Usage:  python scripts/vendor_bin.py            # copy + verify
        python scripts/vendor_bin.py --check     # only verify what is already vendored
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dupdetect import runtime  # noqa: E402

BINARIES = ("ffmpeg", "ffprobe", "fpcalc")
VERSION_FLAG = {"ffmpeg": "-version", "ffprobe": "-version", "fpcalc": "-version"}


def _dest_dir() -> Path:
    d = runtime.bundle_root() / "bin" / runtime.os_key()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _verify(path: Path, name: str) -> str:
    """Run `<bin> -version` so a missing-DLL / wrong-arch copy fails LOUDLY here, not on a user's PC."""
    out = subprocess.run([str(path), VERSION_FLAG[name]], capture_output=True, text=True, timeout=30)
    first = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "(no output)"
    if out.returncode != 0:
        raise RuntimeError(f"{name} exited {out.returncode}: {first}")
    return first


def check_only() -> int:
    dest = _dest_dir()
    missing = 0
    for name in BINARIES:
        p = dest / runtime._exe(name)
        if not p.is_file():
            print(f"  MISSING  {p}")
            missing += 1
            continue
        try:
            print(f"  OK       {p.name:14} -> {_verify(p, name)}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL     {p.name}: {e}")
            missing += 1
    return missing


def vendor() -> int:
    dest = _dest_dir()
    print(f"vendoring binaries -> {dest}")
    failed = 0
    for name in BINARIES:
        exe = runtime._exe(name)
        # find the source the app currently uses (PATH / next-to-python / explicit), NOT the bundle
        src = shutil.which(name) or runtime._next_to_python(exe)
        if not src:
            print(f"  SKIP     {name}: not found on this machine (install it, then re-run)")
            failed += 1
            continue
        target = dest / exe
        shutil.copy2(src, target)
        try:
            ver = _verify(target, name)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL     {name}: copied but does NOT run from the bundle ({e}). "
                  "Likely a dynamically-linked build — vendor a STATIC build instead.")
            failed += 1
            continue
        size_mb = target.stat().st_size / 1e6
        print(f"  COPIED   {exe:14} {size_mb:6.1f} MB  from {src}\n           -> {ver}")
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="only verify already-vendored binaries")
    args = ap.parse_args()
    failed = check_only() if args.check else vendor()
    if failed:
        print(f"\n{failed} binar{'y' if failed == 1 else 'ies'} missing/failed. The app still falls "
              "back to PATH, but the OFFLINE/frozen build needs all of them vendored.")
        return 1
    print("\nAll binaries vendored + verified. bundle/bin is .gitignored (fetched per build).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
