<#
.SYNOPSIS
  dup-detector — Windows bootstrap (v1). Creates the venv, picks the right torch (GPU vs CPU),
  installs the app + heavy features, drops ffmpeg/ffprobe/fpcalc into bundle/bin/win, and warms
  the DINOv2 model cache so later runs work offline.

.DESCRIPTION
  Idempotent: re-running skips what's already in place. Nothing outside the repo is touched except
  the per-user torch hub cache (the model download). Mirrors the manual steps in INSTALL_RECORD.md.

  GPU is autodetected via `nvidia-smi`: present -> torch CUDA (cu128, for Blackwell/RTX 50xx);
  absent -> torch CPU. The verdict is identical either way (§0) — GPU only makes it faster.

.PARAMETER Cpu          Force the CPU torch build even if an NVIDIA GPU is present.
.PARAMETER SkipBinaries Don't download ffmpeg/fpcalc (use whatever is already on PATH/bundle).
.PARAMETER SkipModel    Don't pre-download the DINOv2 weights (will download on first real use).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File install\windows\install.ps1
#>
[CmdletBinding()]
param(
    [switch]$Cpu,
    [switch]$SkipBinaries,
    [switch]$SkipModel
)

$ErrorActionPreference = 'Stop'

# --- pinned sources (bump here when they move) ---------------------------------------------------
$TORCH_CUDA_INDEX = 'https://download.pytorch.org/whl/cu128'   # Blackwell sm_120 needs CUDA 12.8+
$TORCH_CPU_INDEX  = 'https://download.pytorch.org/whl/cpu'
$FFMPEG_ZIP_URL   = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
$FPCALC_ZIP_URL   = 'https://github.com/acoustid/chromaprint/releases/download/v1.5.1/chromaprint-fpcalc-1.5.1-windows-x86_64.zip'

# --- paths ---------------------------------------------------------------------------------------
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Venv     = Join-Path $RepoRoot '.venv'
$VenvPy   = Join-Path $Venv 'Scripts\python.exe'
$BinDir   = Join-Path $RepoRoot 'bundle\bin\win'

function Info($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  ok  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  !!  $m" -ForegroundColor Yellow }

Info "Repo: $RepoRoot"

# --- 1) base interpreter (>=3.11) ---------------------------------------------------------------
function Find-BasePython {
    foreach ($try in @(@('py', '-3.13'), @('py', '-3.12'), @('py', '-3.11'), @('py', '-3'), @('python'))) {
        $exe = $try[0]; $rest = @($try[1..($try.Count - 1)])
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        $v = & $exe @rest -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and [version]$v -ge [version]'3.11') { return ($exe + ' ' + ($rest -join ' ')).Trim() }
    }
    throw "No Python >= 3.11 found. Install Python 3.11+ (winget install Python.Python.3.13) and re-run."
}

# --- 2) venv -------------------------------------------------------------------------------------
if (Test-Path $VenvPy) {
    Ok ".venv already exists"
} else {
    $base = Find-BasePython
    Info "Creating .venv with: $base"
    Invoke-Expression "$base -m venv `"$Venv`""
    Ok ".venv created"
}
& $VenvPy -m pip install --upgrade pip --quiet
Ok "pip upgraded"

# --- 3) torch (GPU vs CPU) ----------------------------------------------------------------------
$hasGpu = -not $Cpu -and [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
& $VenvPy -c "import torch" 2>$null
$alreadyTorch = ($LASTEXITCODE -eq 0)
if ($alreadyTorch) {
    Ok "torch already installed (skipping; re-create .venv to switch CPU<->GPU)"
} elseif ($hasGpu) {
    Info "NVIDIA GPU detected -> installing torch CUDA (cu128)"
    & $VenvPy -m pip install torch torchvision --index-url $TORCH_CUDA_INDEX
    Ok "torch (CUDA) installed"
} else {
    Info "No GPU (or -Cpu) -> installing torch CPU"
    & $VenvPy -m pip install torch torchvision --index-url $TORCH_CPU_INDEX
    Ok "torch (CPU) installed"
}

# --- 4) app + heavy features --------------------------------------------------------------------
Info "Installing app (editable) + UI/watch + heavy features"
& $VenvPy -m pip install -e "$RepoRoot[ui,watch]"
# faiss-gpu has no Windows wheel; faiss-cpu is plenty for 1-2k vectors (IndexFlatIP).
& $VenvPy -m pip install faiss-cpu "scenedetect[opencv]" faster-whisper av
Ok "app + features installed"

# --- 5) external binaries into bundle/bin/win ---------------------------------------------------
function Get-ZipExe([string]$url, [string[]]$exes) {
    # Download $url to a temp zip, extract, and copy the named .exe files into $BinDir.
    New-Item -ItemType Directory -Force $BinDir | Out-Null
    $tmp = Join-Path $env:TEMP ("dd_" + [IO.Path]::GetRandomFileName())
    New-Item -ItemType Directory -Force $tmp | Out-Null
    try {
        $zip = Join-Path $tmp 'dl.zip'
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        Expand-Archive -Path $zip -DestinationPath $tmp -Force
        foreach ($e in $exes) {
            $found = Get-ChildItem -Path $tmp -Recurse -Filter $e -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($found) { Copy-Item $found.FullName (Join-Path $BinDir $e) -Force; Ok "bundled $e" }
            else { Warn "could not find $e inside $url" }
        }
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
}

if ($SkipBinaries) {
    Warn "skipping binary download (-SkipBinaries)"
} else {
    foreach ($pair in @(@('ffmpeg.exe', $FFMPEG_ZIP_URL, @('ffmpeg.exe', 'ffprobe.exe')),
                        @('fpcalc.exe', $FPCALC_ZIP_URL, @('fpcalc.exe')))) {
        $marker = Join-Path $BinDir $pair[0]
        if (Test-Path $marker) { Ok "$($pair[0]) already bundled"; continue }
        try { Get-ZipExe $pair[1] $pair[2] }
        catch { Warn "download failed for $($pair[1]): $($_.Exception.Message) (the app will fall back to PATH)" }
    }
}

# --- 6) warm the DINOv2 model cache (so later runs need no network) ------------------------------
if ($SkipModel) {
    Warn "skipping model pre-download (-SkipModel); it will download on first scan"
} else {
    Info "Pre-downloading DINOv2 (repo + ~330 MB checkpoint) into the torch hub cache"
    & $VenvPy -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitb14', trust_repo=True); print('dinov2 cached')"
    if ($LASTEXITCODE -eq 0) { Ok "DINOv2 cached (offline-ready on this machine)" }
    else { Warn "model pre-download failed; it will retry on first scan" }
}

# --- 7) smoke check ------------------------------------------------------------------------------
Info "Smoke check"
& $VenvPy -c @"
import sys
from dupdetect.runtime import resolve_binary
mods = []
for m in ('torch','faiss','av','scenedetect','faster_whisper','PySide6'):
    try: __import__(m); mods.append(m)
    except Exception as e: print(f'  MISSING {m}: {e}')
import torch
print('  torch', torch.__version__, '| cuda', torch.cuda.is_available())
for b in ('ffmpeg','ffprobe','fpcalc'):
    print(f'  {b} ->', resolve_binary(b))
print('  imported:', ', '.join(mods))
"@

Write-Host ""
Ok "Done. Launch with:  $VenvPy -m dupdetect.cli ui"
