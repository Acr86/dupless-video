# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Dupless Video (Windows desktop .exe).

Flavor-agnostic: builds from EITHER a CPU-only or a CUDA venv (it bundles whatever torch the build
venv has). Verdicts are identical on CPU and GPU (measured: fp32 vs fp16 = 0 flips), so the choice
is only speed + installer size. See docs/BUILD_WINDOWS.md for picking the venv.

  pip install pyinstaller
  python scripts/vendor_model.py     # DINOv2 repo + checkpoint  -> bundle/models
  python scripts/vendor_bin.py       # ffmpeg/ffprobe/fpcalc     -> bundle/bin/win
  pyinstaller --noconfirm dupless-video.spec

Ships fully OFFLINE: the bundle (model + binaries) is added as data; runtime.configure_offline_model
points torch.hub at it, and resolve_binary prefers bundle/bin. No first-run network.
"""
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# C-extension packages (binaries + data + dynamic submodules) -> collect EVERYTHING they ship.
# cv2 is imported lazily (inside a thread) for the live-view JPEG encode, so PyInstaller's static
# analysis misses it -> collect it explicitly or the "What the AI sees" panel is silently blank.
datas, binaries, hidden = [], [], []
for pkg in ("faiss", "av", "faster_whisper", "ctranslate2", "cv2"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hidden += h

# torch / torchvision load submodules dynamically; PySide6 has an official hook (Qt plugins handled).
hidden += collect_submodules("torch") + collect_submodules("torchvision")
# watchdog picks its OS observer (ReadDirectoryChangesW on Windows) dynamically -> collect submodules
# so the bundled watcher reacts to file changes INSTANTLY instead of falling back to backoff polling.
hidden += collect_submodules("watchdog")
hidden += ["numpy", "yaml", "send2trash",
           "PySide6.QtNetwork", "PySide6.QtWidgets", "PySide6.QtGui", "PySide6.QtCore"]

# The bundle is loaded from the FILESYSTEM at runtime (torch.hub source='local' reads the DINOv2 .py
# from bundle/models/hub/...; resolve_binary execs bundle/bin/win/*.exe), so ship it as plain data.
datas += [
    ("bundle", "bundle"),                         # DINOv2 model cache + ffmpeg/ffprobe/fpcalc
    ("config/thresholds.yaml", "config"),         # global thresholds (loaded by config.py)
    ("src/dupdetect/store/schema.sql", "dupdetect/store"),  # DB schema (store.py loads it via __file__)
    ("src/dupdetect/ui/icon.ico", "dupdetect/ui"),
    ("src/dupdetect/ui/icon.png", "dupdetect/ui"),
]
# Catch-all for any other package data (non-.py files loaded relative to __file__).
datas += collect_data_files("dupdetect")

a = Analysis(
    ["scripts/app_entry.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    excludes=["tkinter", "matplotlib", "pytest", "IPython"],   # not used by the app -> shrink
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Dupless Video",
    console=False,                                # GUI app: no console window
    icon="src/dupdetect/ui/icon.ico",
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,                       # UPX can trip AV false-positives on Windows
    name="Dupless Video",
)
