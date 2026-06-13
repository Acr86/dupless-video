"""Vendor the DINOv2 model into bundle/models so the frozen/installed app runs OFFLINE (no torch.hub
download on first run). Copies the torch.hub cache — the cloned repo + the dinov2 checkpoint — which
must exist already (download it once via a normal scan or install.ps1). Privacy: only model files,
no user data. bundle/models is gitignored (big), so this is a local build step before packaging.

Usage:  python scripts/vendor_model.py
"""
import os
import shutil
from pathlib import Path

from dupdetect.runtime import model_dir

DINOV2_CKPT = "dinov2_vitb14_pretrain.pth"


def _src_hub() -> Path:
    base = Path(os.environ["TORCH_HOME"]) if os.environ.get("TORCH_HOME") else Path.home() / ".cache" / "torch"
    return base / "hub"


def main() -> None:
    src = _src_hub()
    repo = src / "facebookresearch_dinov2_main"
    ckpt = src / "checkpoints" / DINOV2_CKPT
    if not repo.is_dir() or not ckpt.is_file():
        raise SystemExit(f"DINOv2 not cached at {src}. Load it once first (run a scan or install.ps1).")
    dst = model_dir() / "hub"
    (dst / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo, dst / "facebookresearch_dinov2_main", dirs_exist_ok=True)
    shutil.copy2(ckpt, dst / "checkpoints" / DINOV2_CKPT)
    print(f"vendored DINOv2 -> {dst}")
    print(f"  repo:       {repo}")
    print(f"  checkpoint: {ckpt.stat().st_size / 1e6:.0f} MB")
    print("the frozen/installed app will now load DINOv2 with source='local' (no network).")


if __name__ == "__main__":
    main()
