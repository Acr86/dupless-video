# Building the Windows installer (Dupless Video)

Goal: a double-click `DuplessVideoSetup.exe` a **non-technical** user runs with no Python, no PATH,
no admin, and **no first-run network** (model + ffmpeg are bundled).

## Which flavor: CPU vs GPU (CUDA)

The **verdicts are identical on CPU and GPU** (measured: CPU-fp32 vs GPU-fp16 = 0 flips, see
`memory/cpu-gpu-fidelity-s0`), so the choice is purely **speed + size**, never correctness:

| Flavor | For | Installer | Notes |
|--------|-----|-----------|-------|
| **GPU (CUDA)** | the user has an NVIDIA GPU | ~2 GB | uses the GPU (much faster scan); **auto-falls back to CPU** if none. Most capable. |
| **CPU-only** | no GPU / universal | ~600 MB | always works; slower without a GPU. |

A CUDA build runs everywhere (GPU when present, CPU otherwise), so when in doubt and size is not a
concern, **ship the GPU build**. Pick the venv accordingly:

## 0. One-time: the build venv

GPU build — reuse the dev `.venv` (it already has CUDA torch + all deps), just add PyInstaller:

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller
```

CPU-only build — a SEPARATE venv with CPU torch:

```powershell
py -3.14 -m venv .venv-build
.\.venv-build\Scripts\Activate.ps1
pip install -e ".[ui]"                     # the app + PySide6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu av faster-whisper pyinstaller
```

Build the spec with the venv you chose (`.venv\Scripts\python.exe` for GPU, `.venv-build\...` for CPU).

## 1. Vendor the offline assets into `bundle/`

```powershell
python scripts/vendor_model.py             # DINOv2 repo + checkpoint  -> bundle/models  (~350 MB)
python scripts/vendor_bin.py               # ffmpeg/ffprobe/fpcalc      -> bundle/bin/win
python scripts/vendor_bin.py --check       # verify each binary RUNS from the bundle
```

`bundle/` is `.gitignore`d (fetched per build, never committed). `runtime.configure_offline_model()`
points `torch.hub` at `bundle/models`; `runtime.resolve_binary()` prefers `bundle/bin/win` — so the
frozen app uses its OWN known-good model + binaries regardless of the user's machine.

> ffmpeg/ffprobe from the Gyan **full** build are ~194 MB each. To shrink the installer, vendor a
> static **essentials** build instead (still single-exe, no DLLs) before running `vendor_bin.py`.

## 2. Freeze with PyInstaller

```powershell
pyinstaller --noconfirm dupless-video.spec     # -> dist\Dupless Video\  (the COLLECT folder)
```

Smoke-test the frozen app before packaging:

```powershell
".\dist\Dupless Video\Dupless Video.exe"        # must open the UI with NO console + NO network
```

The spec ships `scripts/app_entry.py` (boots offline model -> UI), the whole `bundle/`,
`config/thresholds.yaml`, and the icons. Hidden imports cover torch/torchvision/faiss/av/PySide6.

## 3. Wrap in an installer (Inno Setup)

Install [Inno Setup](https://jrsoftware.org/isdl.php), then pass the flavor so the output is named for
it (the CPU and GPU installers can then coexist as separate downloads):

```powershell
iscc /DFlavor=GPU install\windows\dupless-video.iss   # -> Output\DuplessVideoSetup-GPU.exe
iscc /DFlavor=CPU install\windows\dupless-video.iss   # -> Output\DuplessVideoSetup-CPU.exe
```

Omit `/DFlavor` for a plain `DuplessVideoSetup.exe`. The flavor changes only the installer filename
and the bundled torch — same AppId/AppName, so installing one upgrades the other in place.

Per-user install (`PrivilegesRequired=lowest`) — no UAC prompt. The user's **data** (DB, embeddings)
lives in `%LOCALAPPDATA%\Dupless Video` and survives uninstall.

## 4. Test on a CLEAN machine (the real gate)

The only way to catch a missing DLL / first-run network / AV false-positive is a Windows box (or VM)
with **no Python and no GPU**:

1. Run `DuplessVideoSetup.exe`, accept defaults, launch.
2. Scan a small folder **offline** (disconnect the network) — embeddings + ffmpeg must work.
3. Confirm a known duplicate is found and **deletion asks for confirmation** (never auto-deletes).

Only after this passes is the build ready to hand to a non-technical user as a BETA.

## Notes
- **AV false-positives:** unsigned PyInstaller exes sometimes trip SmartScreen/Defender. Code-signing
  the exe (or submitting to Microsoft) removes the warning; otherwise tell the friend "More info -> Run".
- **Updates:** a new build re-installs over the old one; user data is untouched. `feature_version`
  changes trigger an incremental re-scan (not a full re-decode) — keep it stable between friend builds.
- **CPU scan speed:** without a GPU, embedding is slower. Set expectations; the pipeline is still I/O-bound.
