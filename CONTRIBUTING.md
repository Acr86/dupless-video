# Contributing

Thanks for your interest in Dupless Video. This is a focused project with one non-negotiable rule, so
please read the short charter in [`CLAUDE.md`](CLAUDE.md) before a substantial change — especially:

> **Zero false positives in the strong tiers.** A wrong "duplicate" can cost someone a file. Recall is
> recovered from a review queue, never by loosening a threshold. The same input must produce the same
> verdict regardless of hardware (CPU and GPU agree).

Changes that touch detection semantics, thresholds, or calibration are **forks**: open an issue first
with the measured trade-off, rather than changing them silently.

## Development setup

Use the project virtualenv, never the system Python.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     |  Linux/macOS: source .venv/bin/activate

python -m pip install --upgrade pip
# CPU stack (the verdict is identical to GPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu av
pip install -e ".[dev,ui,watch]"

pre-commit install     # runs ruff + gitleaks + basic hooks on every commit
```

External binaries `ffmpeg`, `ffprobe` and `fpcalc` must be resolvable (on `PATH` or under
`bundle/bin/`). On Windows, `install/windows/install.ps1` bootstraps everything.

## Before you push

The CI gate mirrors these — run them locally to get a green PR:

```bash
ruff check .                 # style/lint — BLOCKING (must be clean)
mypy src/dupdetect           # types — advisory (won't fail CI yet, but keep it from growing)
pytest -q                    # full suite — must pass
```

- **Fixing a bug?** Add the failing test first, then the fix.
- **Adding behavior?** Add tests for it and keep the suite green.
- **Performance claim?** Ship a before/after number on representative data — not a vibe.
- Keep diffs surgical: every changed line should trace to the request or a measured bottleneck. Match
  the surrounding densely-commented style. All code, comments and UI strings are written in **English**.

## Pull requests

- Branch from `main`; keep the PR focused on one thing.
- Use a clear, imperative commit subject (`fix(detect): …`, `perf(pass2): …`, `docs: …`).
- The PR description should say **what** changed and **why**, and include the verification you ran
  (tests passing, benchmark numbers, or the app actually running the change).
- CI must be green: `lint`, the `test` matrix (3.11 / 3.12), `codeql`, and the `security` scans.

## Reporting bugs & security issues

- Functional bugs: open a GitHub issue with steps to reproduce and the file/verdict involved.
- Security vulnerabilities: **do not** open a public issue — see [`SECURITY.md`](SECURITY.md).
