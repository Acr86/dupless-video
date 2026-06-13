# Dupless Video

[![tests](https://github.com/Acr86/dupless-video/actions/workflows/tests.yml/badge.svg)](https://github.com/Acr86/dupless-video/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

**Find true duplicate videos by what's *in* them — not by filename.**

Dupless Video is a content-based duplicate & upgrade detector for large video libraries, built
around one uncompromising goal: **never flag the wrong file as a duplicate.**

If you've ever ended up with three copies of the same movie — a 4K remux, a 1080p re-encode, and a
dubbed version — and your library tool grouped them wrong or missed them entirely, that's the problem
this solves. It's exactly where filename matchers and tools like Plex fall short.

## Why it's different

Most "duplicate finders" compare filenames or file hashes. That misses the cases that actually
matter, because the same film can look completely different on disk:

- **Re-encodes** — same movie, different codec / resolution / bitrate → different hash, often a
  different name.
- **Different dubs** — identical video, different audio language → should be flagged as the same
  film, not treated as unrelated.
- **Cam rips** — degraded audio and video, but the *scene structure* survives.
- **Inserted commercials** — the same film with ad breaks spliced in → still a duplicate (and you'd
  want to keep the clean copy).
- **Director's cuts** — a longer, *legitimately different* edition → must **not** be deleted as junk.

Dupless Video decides from the **content itself**: deep visual embeddings, an audio fingerprint, and
a scene-cut signature — three signals that fail in independent ways.

## The core idea: agreement across independent signals

Precision comes from a simple principle — **require independent signals to agree** before calling two
files duplicates:

- **Video** — per-frame DINOv2 embeddings, temporally aligned.
- **Audio** — a Chromaprint fingerprint, offset-aligned (which also catches prepended ads).
- **Scenes** — the sequence of scene cuts, compared with dynamic time warping.

Two *independent* modalities matching by chance across genuinely different films is astronomically
unlikely — that is the guarantee. It drives a **staged decision tree** where the strongest verdicts
require corroboration:

| Tier | Condition | Verdict |
|------|-----------|---------|
| T0 | byte-identical sampled hash | CERTAIN |
| T1 | audio **and** video agree | CERTAIN |
| T2 | video identical, audio doesn't align | VERY_HIGH — *different dub* |
| T3 | partial video + scene structure | HIGH |
| T4 | scene structure only | PROBABLE → review queue |
| — | strong match, but a contiguous superset | DIFFERENT_EDITION (e.g. director's cut) |

**Zero false positives is the contract.** Recall is recovered from a review queue — never by loosening
thresholds. Verdicts are also **hardware-invariant**: CPU and GPU produce identical decisions
(verified), so a result never depends on the machine it ran on.

## Built to run at scale

Profiling this pipeline showed it is **I/O-bound on disk**, not compute-bound — so the optimizations
target I/O, not GPU tricks:

- **Coarse → fine retrieval.** One global embedding per video feeds a FAISS index to generate
  candidate pairs (avoiding O(n²)); the expensive frame-level alignment runs only on candidates.
- **Storage-aware.** Worker concurrency auto-tunes for HDD vs SSD — concurrency thrashes a spinning
  disk but helps an NVMe.
- **Incremental & resilient.** Keyed by (path, mtime, size): re-runs skip unchanged files, and a
  corrupt or unreadable file is skipped and reported, never crashing the batch.
- **Adaptive sampling.** Giant 4K/8K files are sampled by sparse seek; HD by keyframe demux.

A background **watcher** keeps the library current as files arrive — using native filesystem events
for instant reaction, with a backoff that barely touches the disk when nothing changes.

## Choosing which copy to keep

Identity ("is it the same film?") and quality ("is it the better copy?") are kept on separate rails —
mixing them is what makes other tools brittle. Once files are confirmed as the same content, a quality
stage picks the keeper by resolution then bitrate, with signals for:

- **Color clipping** — crushed blacks / blown highlights mean destroyed detail.
- **Muted or cut audio** — so the silent copy is never kept by mistake.
- **Inserted commercials** — prefers the clean copy and flags which one carries the ads.

## Tech stack

Python · PySide6 (desktop UI) · PyTorch + DINOv2 (visual embeddings) · FAISS (retrieval) ·
Chromaprint / fpcalc (audio fingerprint) · faster-whisper (language detection, used only to choose a
keeper) · FFmpeg / PyAV (decode) · SQLite (store). Runs on CPU or NVIDIA GPU with identical results.

## Getting started

```bash
pip install -e ".[ui]"
# then, per your hardware: torch, faiss, av, faster-whisper

dupdetect scan "D:/Movies"     # index a library and find duplicates
dupdetect ui                    # review and act on results in the desktop app
dupdetect watch "D:/Movies"     # keep it current in the background
```

On Windows you can build a standalone installer — no Python required — from the bundled PyInstaller
spec; see [docs/BUILD_WINDOWS.md](docs/BUILD_WINDOWS.md).

## Project layout

```
src/dupdetect/
  models.py      Record, Result, AlignResult, Verdict
  features/      probe · hashing · frames (decode) · embeddings (DINOv2) · audio_fp · scenes
  align/         audio · video (local alignment) · scenes (DTW)
  match/         retrieval (FAISS) · tree (the decision tree) · matcher
  quality/       language (whisper) · camrip · color
  pipeline/      analyze · fullscan · calibrate
  store/         SQLite metadata + memmapped embeddings
  ui/            PySide6 desktop app
config/thresholds.yaml   calibrated detection thresholds
```

## Tests

```bash
pytest -q        # 249 passing
```

## Development

Built with AI-assisted development. The architecture, the zero-false-positive guarantee, and every
design trade-off were driven and reviewed by me; an AI pair-programmer accelerated the implementation.
The engineering principles that governed the work — zero false positives, measure-before-optimize,
surgical changes, ask-on-forks — are written down in [CLAUDE.md](CLAUDE.md).

## License

[MIT](LICENSE) © 2026 Adrian Cortes Reyes
