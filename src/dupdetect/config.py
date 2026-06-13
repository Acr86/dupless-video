"""Loads thresholds from config/thresholds.yaml."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# config/thresholds.yaml lives at the repo root in source, but a frozen build (PyInstaller) ships it
# under the bundle dir (sys._MEIPASS/config). Resolve per layout so the .exe finds it (else FileNotFound
# on the first scan — the schema.sql sibling bug). parents[2]: …/src/dupdetect/config.py -> repo root.
if getattr(sys, "frozen", False):
    DEFAULT_CONFIG = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "config" / "thresholds.yaml"
else:
    DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "thresholds.yaml"


@dataclass
class Thresholds:
    raw: dict[str, Any]

    # convenient accessors for the most-used values in the decision tree
    @property
    def theta_a(self) -> float: return self.raw["audio"]["theta_a"]
    @property
    def theta_a_low(self) -> float: return self.raw["audio"]["theta_a_low"]
    @property
    def theta_v(self) -> float: return self.raw["video"]["theta_v"]
    @property
    def theta_v_high(self) -> float: return self.raw["video"]["theta_v_high"]
    @property
    def min_coverage(self) -> float: return self.raw["video"]["min_coverage"]
    @property
    def theta_s(self) -> float: return self.raw["scenes"]["theta_s"]
    @property
    def theta_s_high(self) -> float: return self.raw["scenes"]["theta_s_high"]
    @property
    def min_cut_density(self) -> float:
        """T4 (scenes only): minimum cut density (cuts/s) required on both files for the
        scene-cut signature to be trusted. Coarse signatures (SEEK-sampled giants) align
        spuriously -> excluded from T4. See scenes.min_cut_density in thresholds.yaml."""
        return float(self.raw["scenes"].get("min_cut_density", 0.04))
    @property
    def superset_min_extra_ratio(self) -> float:
        return self.raw["edition"]["superset_min_extra_ratio"]
    @property
    def fps_sample(self) -> float: return self.raw["video"]["fps_sample"]
    @property
    def band_radius_frames(self) -> int:
        """A3: Sakoe-Chiba band width in frames = max_offset_s * fps."""
        return int(self.raw["video"]["max_offset_s"] * self.raw["video"]["fps_sample"])
    @property
    def faiss_k(self) -> int: return self.raw["retrieval"]["faiss_k"]
    @property
    def n_window_vecs(self) -> int: return self.raw["retrieval"]["n_window_vecs"]
    @property
    def duration_tolerance(self) -> float:
        return self.raw["retrieval"]["duration_tolerance"]

    @property
    def duration_block_cos_gate(self) -> float:
        """Only align duration-neighbors whose GLOBAL embedding cosine to the query >= this — prunes
        the same-length-but-different videos that explode Pass-2 on a dense library. Prunes only the
        duration safety net (top-k/window retrieval untouched). 0 disables. Default 0.7."""
        return float(self.raw["retrieval"].get("duration_block_cos_gate", 0.7))

    # --- adaptive frame sampling (demux vs seek) + alignment grid ---
    @property
    def seek_threshold_bytes(self) -> int:
        return int(self.raw.get("sampling", {}).get("seek_threshold_gb", 6.0) * 1024 ** 3)
    @property
    def seek_n(self) -> int:
        return int(self.raw.get("sampling", {}).get("seek_n", 200))
    @property
    def grid_step_s(self) -> float:
        return float(self.raw.get("sampling", {}).get("grid_step_s", 2.0))
    @property
    def max_offset_s(self) -> float:
        return float(self.raw["video"]["max_offset_s"])

    # --- audio_fp: length + per-file timeouts (avoid waste on large/broken files) ---
    @property
    def audio_fp_cap_s(self) -> int:
        """Fingerprint length (s) used ONLY for content longer than `audio_fp_cap_above_s`
        (movie giants): a small head that bounds disk reads. 0 = never cap (whole file always)."""
        return int(self.raw.get("audio", {}).get("fp_max_s", 600))

    @property
    def audio_fp_cap_above_s(self) -> int:
        """Duration gate (s): content at/below this is fingerprinted WHOLE (the cap is a dangerous
        fraction of short content — shared intros/recaps in series, near-equal-length AI clips —
        and whole-file fpcalc is cheap there, ~2s/min); longer content uses `audio_fp_cap_s`
        (the cap is a safe small head AND whole-file is ~13x the disk cost on UHD remuxes).
        MEASURED (2026-06-12): a cross-episode pair (same series, 22min) with a shared intro scored
        audio=0.94 capped-600 -> 0.77 whole-file (divergent dialogue), so different episodes leave T1;
        true dups (same content) stay ~0.98. One global, content-derived rule (§0)."""
        return int(self.raw.get("audio", {}).get("fp_cap_above_s", 3600))

    def audio_fp_max_for(self, duration_s: float | None) -> int:
        """Per-file fingerprint length (s): whole file (0) for content at/below the gate, else the
        cap. Deterministic from the measured duration (§0). Unknown/zero duration -> cap (conservative
        on disk cost: an unprobed file could be a giant); such files are rare and re-probed on index."""
        cap = self.audio_fp_cap_s
        if cap <= 0:
            return 0                                       # capping disabled globally -> whole file
        if duration_s and duration_s <= self.audio_fp_cap_above_s:
            return 0                                       # short content -> whole file (FP-risk zone)
        return cap

    @property
    def decode_timeout_s(self) -> int:
        """Aborts the decode of ONE file that takes too long (broken/missing index -> very
        slow seek). Marked 'reindex' (a remux fixes it), does not abort the scan. Legitimate
        large files use SEEK (fast), so a file that exceeds this almost always has a bad index."""
        return int(self.raw.get("limits", {}).get("decode_timeout_s", 240))

    @property
    def audio_fp_timeout_s(self) -> int:
        """Safety net if fpcalc hangs (reads the entire file; generous for 4K/8K)."""
        return int(self.raw.get("limits", {}).get("audio_fp_timeout_s", 600))

    @property
    def name_copy_grouping(self) -> bool:
        """Group `(N)` name copies in the same folder (NAME_COPY cluster, probable)."""
        return bool(self.raw.get("quality", {}).get("name_copy_grouping", True))

    @property
    def color_clip_keep_weight(self) -> float:
        """KEEP ranking penalty per unit of color clipping (crushed blacks/blown highlights =
        destroyed detail). Tiebreaks toward the least-clipped copy among same-resolution dups,
        without overriding a real resolution jump. 0 disables the color signal in KEEP."""
        return float(self.raw.get("quality", {}).get("color_clip_keep_weight", 1_000_000))

    @property
    def ad_interleaved_min(self) -> float:
        """Flag a copy as ad-injected when its AlignResult.interleaved_ratio >= this (KEEP hint +
        UI warning only; never changes the duplicate verdict). 0 disables. See thresholds.yaml."""
        return float(self.raw.get("quality", {}).get("ad_interleaved_min", 0.02))

    @property
    def min_ad_run_s(self) -> float:
        """Minimum contiguous unmatched run (s) counted as an ad break (filters encode jitter)."""
        return float(self.raw.get("quality", {}).get("min_ad_run_s", 10))


def load_thresholds(path: Path | str | None = None) -> Thresholds:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        return Thresholds(raw=yaml.safe_load(f))
