"""Feature extraction. M3: SPLIT BY RESOURCE.

  extract_cpu_features(path)  -> probe, hash, audio_fp, scene_cuts
       CPU/IO-bound, picklable, runs in ProcessPool (forked workers).
       Does NOT touch CUDA -> safe to fork.

  extract_gpu_features(...)   -> frames(NVDEC) + embeddings + descriptors
       GPU. ALWAYS runs in the main process, serialized, large batches.
       NEVER in a forked worker (do not cross CUDA context).

  analyze_file(...)           -> orchestrates both for ONE file (watcher).
       full_scan() instead schedules CPU and GPU separately (see fullscan.py).
"""
from __future__ import annotations

import base64
import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from dupdetect.config import Thresholds
from dupdetect.features.audio_fp import AUDIO_FP_VERSION, COVERAGE_VERSION, scan_audio_coverage
from dupdetect.features.embeddings import Embedder, global_descriptor, window_descriptors
from dupdetect.features.frames import DINOV2_MEAN, DINOV2_STD, decode_frames
from dupdetect.features.hashing import content_hash
from dupdetect.features.probe import ffprobe
from dupdetect.features.scenes import SCENE_ALGO_VERSION, scene_cuts_from_embeddings
from dupdetect.features.scenes import scene_cuts as detect_scene_cuts
from dupdetect.models import Probe, Quality, Record
from dupdetect.quality.camrip import cam_score
from dupdetect.quality.color import COLOR_VERSION, ColorStats
from dupdetect.store import FingerprintStore


def feature_version(embedder: Embedder, independent_scenes: bool = False,
                    audio_fp_cap_s: int = 0, audio_fp_cap_above_s: int = 0) -> str:
    """C4: combined version of ALL signals (embed + audio + scenes).
    Changing any algorithm's implementation invalidates the cache. Scene mode
    (PIX=independent ffmpeg / EMB=derived from embeddings) is included in the version,
    so A and B don't share cache. The audio fingerprint POLICY (duration-gated cap) is encoded
    too: a cap applied above a duration gate changes the fingerprint of long content -> invalidates
    old audio_fp cache (§4). Tag `G{gate}C{cap}` (gated) vs nothing (whole file always)."""
    mode = "PIX" if independent_scenes else "EMB"
    afp = f"afp{AUDIO_FP_VERSION}"
    if audio_fp_cap_s and audio_fp_cap_s > 0:               # capping active above the gate
        afp += f"G{int(audio_fp_cap_above_s)}C{int(audio_fp_cap_s)}"
    return (f"{embedder.feature_version}|{afp}|cov{COVERAGE_VERSION}|clr{COLOR_VERSION}"
            f"|scn{mode}{SCENE_ALGO_VERSION}")


@dataclass
class CpuFeatures:
    """Picklable result from the ProcessPool (M3). No CUDA involved."""
    path: str
    mtime: float
    size: int
    probe: Probe
    content_hash: str
    audio_fp: np.ndarray          # uint32 (C1). Empty in Pass-1: the matching fingerprint is
                                  # computed ON-DEMAND in Pass-2 only for candidate pairs.
    scene_cuts: np.ndarray
    lang_detected: str | None
    cam_score_partial: float      # cam portion that doesn't require decoded frames
    audio_coverage: float | None  # None = deferred (ensured on-demand); else whole-file coverage


def extract_cpu_features(path: str, independent_scenes: bool = False) -> CpuFeatures:
    """M3: all CPU/IO-bound work. Safe to run in a forked worker.
    Mode B (default): does NOT detect scenes here (derived from embeddings in the
    main process) -> avoids the expensive decode. Mode A (independent_scenes): ffmpeg by pixels.

    Audio: computes only the CHEAP whole-file coverage here (seek-sampled, ~44x less I/O than a
    full fingerprint). The matching fingerprint is computed ON-DEMAND in Pass-2, and only for
    candidate pairs -> most files (unique movies) never pay for it. A file with no audio stream
    -> coverage 0.0 (flagged), so the user never keeps a silent copy by mistake."""
    st = os.stat(path)
    probe = ffprobe(path)
    cuts = detect_scene_cuts(path) if independent_scenes else np.empty(0, dtype=np.float32)
    return CpuFeatures(
        path=path, mtime=st.st_mtime, size=st.st_size,
        probe=probe, content_hash=content_hash(path),
        audio_fp=np.empty(0, dtype=np.uint32),       # on-demand in Pass-2 (candidate pairs only)
        scene_cuts=cuts,
        lang_detected=None,                          # DEFERRED: whisper runs only for cluster
                                                     # members at rank time (rank_cluster) -> most
                                                     # unique files never pay for it. Same output.
        cam_score_partial=cam_score(probe, None),    # portion without frames
        audio_coverage=None,                         # DEFERRED (measured ~37% of Pass-1): whole-file
                                                     # coverage is ensured on-demand for cluster
                                                     # members (Standard) or for ALL files (Deep).
    )


def ensure_audio_coverage(path: str, store: FingerprintStore, duration_s: float,
                          has_audio: bool) -> float:
    """ON-DEMAND whole-file audio coverage (muted/truncated-copy signal). Computed-if-missing and
    persisted, mirroring the audio-fp pattern: NULL in the DB = not computed yet. Checks the RAW
    column (not the NULL->1.0 loaded value) so 'already computed as 1.0' and 'not computed' don't
    collapse. Used by rank_cluster (cluster members) and by the Deep depth (all files)."""
    row = store.conn.execute("SELECT audio_coverage FROM files WHERE path=?", (str(path),)).fetchone()
    if row is not None and row[0] is not None:
        return float(row[0])                         # already computed -> reuse (incremental)
    cov = scan_audio_coverage(path, duration_s) if has_audio else 0.0
    store.set_audio_coverage(str(path), cov)
    return cov


def build_record(cpu: CpuFeatures, emb: np.ndarray, times: np.ndarray, color: ColorStats,
                 embedder: Embedder, th: Thresholds, independent_scenes: bool = False) -> Record:
    """Combines CPU features + GPU embeddings into a Record. In mode B derives scene cuts
    from embeddings using the REAL keyframe timestamps (`times`). `color` is measured from the
    same decoded keyframes (reused) -> helps pick the KEEP / flag color divergence."""
    quality = Quality(lang_detected=cpu.lang_detected, cam_score=cpu.cam_score_partial,
                      audio_coverage=cpu.audio_coverage,   # cheap whole-file coverage (Pass-1)
                      color=color)
    cuts = cpu.scene_cuts if independent_scenes else scene_cuts_from_embeddings(emb, times)
    return Record(
        path=cpu.path, mtime=cpu.mtime, size=cpu.size,
        probe=cpu.probe, content_hash=cpu.content_hash,
        global_vec=global_descriptor(emb),
        window_vecs=window_descriptors(emb, th.n_window_vecs),   # A2
        embeddings=emb, audio_fp=cpu.audio_fp, scene_cuts=cuts,
        frame_times=np.asarray(times, dtype=np.float32),         # per-frame ts (align by time)
        quality=quality,
    )


def extract_gpu_features(path: str, probe: Probe, embedder: Embedder,
                         th: Thresholds) -> tuple[np.ndarray, np.ndarray, ColorStats]:
    """M3: decode keyframes (NVDEC) + embed. MAIN process ONLY. Returns (emb fp16 [N,D],
    timestamps[N] s, color stats). Adaptive sampling (demux vs seek) by file size, from config.
    Color stats are measured from the same decoded frames (reused, no extra decode)."""
    frames, times, color = decode_frames(path, seek_threshold_bytes=th.seek_threshold_bytes,
                                         seek_n=th.seek_n, height=probe.height or 0,
                                         decode_timeout_s=th.decode_timeout_s)  # 8K -> seek
    maybe_emit_viz(path, frames)                      # live-view (only if the panel is open)
    return embedder.encode(frames), times, color      # [N, D] L2-norm fp16, ts[N], color


# --------------------------------------------------------------------------- live-view ("what the AI sees")
# DaVinci-style: per video, emit a SAMPLED SEQUENCE of its keyframes (a burst) so the UI can play it
# back as the model processes it. Everything is in RAM — JPEG is encoded in-memory (cv2.imencode) and
# sent inline as base64 over stdout; NO image files are written. Reuses the frames already decoded
# (no extra decode/disk), on a background thread, on-demand (only while the panel is open). Cosmetic
# (§0): never affects a verdict, and wrapped so it can never break a scan.
_VIZ_LAST = [0.0]              # last burst time (rate bound); a list so the closure can mutate it
_VIZ_MIN_INTERVAL = 0.4        # min seconds between bursts -> bounds pipe traffic if videos are tiny
_VIZ_FRAMES_PER_BURST = 36     # keyframes sampled per video (the UI plays them at ~12 fps)


def maybe_emit_viz(path: str, frames) -> None:
    """Emit a short playback burst of the current video's keyframes if the live-view panel is open."""
    try:
        from dupdetect.runtime import viz_enabled
        n = 0 if frames is None else len(frames)
        if n == 0 or not viz_enabled():
            return
        now = time.time()
        if now - _VIZ_LAST[0] < _VIZ_MIN_INTERVAL:
            return
        _VIZ_LAST[0] = now
        k = min(_VIZ_FRAMES_PER_BURST, n)
        idx = np.linspace(0, n - 1, k).round().astype(int)       # evenly spaced across the video
        sel = frames[idx] if hasattr(frames, "dim") else np.asarray(frames)[idx]
        rgb = _frames_to_rgb_uint8(sel)                          # GPU->CPU + de-normalize (one batched op)
        threading.Thread(target=_emit_viz_burst, args=(os.path.basename(path), rgb),
                         daemon=True).start()
    except Exception:                                  # noqa: BLE001 — never let the live-view break a scan
        pass


def _frames_to_rgb_uint8(batch) -> np.ndarray:
    """Recover viewable [K,H,W,3] uint8 RGB from the MODEL INPUT. decode_frames yields an
    ImageNet-normalized CHW batch ([K,3,224,224], usually CUDA) -> undo (x·std+mean)·255 so the panel
    shows what the model actually ingests; a raw numpy [K,H,W,3] uint8 batch is returned as-is."""
    if hasattr(batch, "dim"):                          # torch tensor (the real decode_frames output)
        import torch
        mean = torch.tensor(DINOV2_MEAN, device=batch.device).view(1, 3, 1, 1)
        std = torch.tensor(DINOV2_STD, device=batch.device).view(1, 3, 1, 1)
        img = (batch.float() * std + mean).clamp(0, 1).mul(255).round().byte()   # [K,3,H,W]
        return img.permute(0, 2, 3, 1).contiguous().cpu().numpy()                # [K,H,W,3] uint8 RGB
    return np.asarray(batch, dtype=np.uint8)


def _emit_viz_burst(name: str, rgb_batch: np.ndarray) -> None:
    """Encode each frame to JPEG IN RAM and write it inline as a 'VIZ:<base64>|<name>' line. No files."""
    try:
        import cv2
        lines = []
        for frame in rgb_batch:
            h, w = frame.shape[:2]
            tw = 280
            if w > tw:
                frame = cv2.resize(frame, (tw, max(1, int(h * tw / w))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1],    # RGB -> BGR for correct JPEG colors
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                lines.append("VIZ:" + base64.b64encode(buf.tobytes()).decode("ascii") + "|" + name)
        if lines:
            sys.stdout.write("\n".join(lines) + "\n")            # one write -> frames stay contiguous
            sys.stdout.flush()
    except Exception:                                  # noqa: BLE001 cosmetic only
        pass


def analyze_file(path: str, store: FingerprintStore, embedder: Embedder,
                 th: Thresholds, force: bool = False,
                 independent_scenes: bool = False) -> Record:
    """Single-file path (watcher). Incremental + short-circuit by hash (M4)."""
    st = os.stat(path)
    fv = feature_version(embedder, independent_scenes,
                         audio_fp_cap_s=th.audio_fp_cap_s, audio_fp_cap_above_s=th.audio_fp_cap_above_s)
    if not force and store.has_fresh(path, st, fv):
        cached = store.load(path, with_embeddings=True)
        if cached is not None:
            return cached

    cpu = extract_cpu_features(path, independent_scenes=independent_scenes)
    emb, times, color = extract_gpu_features(path, cpu.probe, embedder, th)   # keyframes NVDEC (fast)
    rec = build_record(cpu, emb, times, color, embedder, th, independent_scenes)
    store.save(rec, feature_version=fv)
    return rec
