"""Scene-cut fingerprint. The interval pattern between cuts is an edit signature,
invariant to almost everything and even robust to cam rips.

Detection via ffmpeg (filter select='gt(scene,T)') on DOWNSCALED frames:
downscaling cheapens score computation and ffmpeg decodes natively -> ~9x real-time
even on 4K on CPU (PySceneDetect+OpenCV was ~realtime or worse, the scan bottleneck).
ffmpeg's `scene` score is in [0,1] (content change between frames).
"""
from __future__ import annotations

import os
import re
import subprocess

import numpy as np

from dupdetect.runtime import resolve_binary

# C4: bump if detector/threshold changes. v3: two modes (see feature_version):
#   EMB = cuts derived from embeddings (fast, default); PIX = ffmpeg pixel-based.
SCENE_ALGO_VERSION = 3

_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def scene_cuts_from_embeddings(emb: np.ndarray, timestamps: np.ndarray,
                               sim_threshold: float = 0.6) -> np.ndarray:
    """Mode B: cuts derived from per-frame embeddings, WITHOUT decoding again.
    A cut = drop in cosine similarity between consecutive frames below the threshold
    (content change), timestamped with the REAL frame TIMESTAMP (key with keyframe
    sampling, which is irregular -> intervals are comparable across encodes).
    FAST but NOT video-independent (affects the cam case in T4)."""
    e = np.asarray(emb, dtype=np.float32)
    t = np.asarray(timestamps, dtype=np.float32)
    if e.shape[0] < 2 or t.shape[0] != e.shape[0]:
        return np.empty(0, dtype=np.float32)
    e = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-8, None)   # L2 just in case
    sims = (e[1:] * e[:-1]).sum(axis=1)                  # cosine between consecutive frames
    cut_idx = np.nonzero(sims < sim_threshold)[0] + 1    # frame where it drops
    # sort: keyframe pts can arrive out of order -> monotonic cuts
    # (np.diff intervals >= 0; otherwise DTW relative distance explodes)
    return np.sort(t[cut_idx]).astype(np.float32)        # real timestamps, ascending


def scene_cuts(path: str, threshold: float = 0.3, height: int = 180) -> np.ndarray:
    """Scene-cut timestamps (s). ffmpeg passes through `select='gt(scene,T)'` only
    frames with content change > threshold, and `showinfo` prints their pts_time.

    `height`: downscale height (width auto-adjusted); 180 px is enough to detect
    cuts (global changes) and much cheaper. `threshold` in [0,1] (TUNABLE).
    The alignment SIGNATURE is the INTERVALS between cuts (np.diff), invariant to
    leading trims; align/scenes.py runs DTW over those intervals.
    """
    cmd = [
        resolve_binary("ffmpeg"), "-i", path, "-an", "-sn",
        "-vf", f"scale=-2:{height},select='gt(scene,{threshold})',showinfo",
        "-f", "null", os.devnull,
    ]
    from dupdetect.util import CREATE_NO_WINDOW
    proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                          creationflags=CREATE_NO_WINDOW)
    if proc.returncode != 0:                          # unreadable file -> caught by _pass1
        raise RuntimeError(f"ffmpeg scene: {proc.stderr.strip()[-200:] or 'unknown error'}")
    # showinfo writes to stderr one line per frame that passed the select, with pts_time
    times = sorted(float(t) for t in _PTS_RE.findall(proc.stderr))
    return np.array(times, dtype=np.float32)
