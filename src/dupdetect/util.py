"""Shared package utilities."""
from __future__ import annotations

import os
import re
import subprocess

# On Windows, prevents EACH console subprocess (ffmpeg/ffprobe/fpcalc/taskkill) from opening its
# own window when the parent process has NO console (e.g. the UI launched with pythonw).
# Without this, a scan invoking hundreds of ffmpeg processes would flash hundreds of console windows.
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

# Rich renders tracebacks inside box-drawing characters; strip them (plus padding) from a line's
# ends so the exception line can be matched whether or not it sits inside a panel.
_BOX_CHARS = "│┌┐└┘─╭╮╰╯├┤┬┴┼ \t"
# An exception summary at the START of an (un-framed) line: 'RuntimeError: …', 'KeyError', etc.
# Anchored on purpose: source frames like 'raise RuntimeError("x")' must NOT be mistaken for it.
_EXC_RE = re.compile(r"([A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit))\b\s*:?[ \t]*(.*)")


def summarize_error(log_lines: list[str]) -> str:
    """Best one-line reason a subprocess run failed, dug out of its captured output.

    A normal user must never be shown just 'exit 1': this pulls the Python exception line out of
    the (Rich-rendered) traceback — e.g. 'RuntimeError: mat1 and mat2 shapes cannot be multiplied
    (465x768 and 0x0)' — un-framing box-drawing characters first. Keeps the LAST match (the actually
    raised exception, not an inner 'During handling of the above…'). Falls back to the last
    Error/Exception/Traceback line, then the last non-empty line, so the result is always actionable."""
    best = ""
    for ln in log_lines:
        m = _EXC_RE.match(ln.strip(_BOX_CHARS))
        if m:
            msg = m.group(2).strip(_BOX_CHARS).strip()
            best = f"{m.group(1)}: {msg}" if msg else m.group(1)
    if best:
        return best
    for ln in reversed(log_lines):
        s = ln.strip(_BOX_CHARS)
        if any(k in s for k in ("Error", "Exception", "Traceback")):
            return s
    for ln in reversed(log_lines):
        s = ln.strip(_BOX_CHARS)
        if s:
            return s
    return "the process exited without output"
