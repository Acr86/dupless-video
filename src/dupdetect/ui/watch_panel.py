"""Background-watch launcher for the UI.

Runs `dupdetect.cli watch` as a SUBPROCESS (like ScanPanel) so new/changed files are indexed and
matched WHILE the app is open, without mixing the GPU/ProcessPool with the GUI event loop. Parses
the watcher's output and emits:
  - `duplicate_detected(n)` when a cycle reports new duplicate clusters -> the window toasts + refreshes
  - `activity()` when anything changed (indexed/removed) -> the window refreshes the tree
The user opens the app only to REVIEW; the watcher keeps the DB current in the background.
"""
from __future__ import annotations

import re
import subprocess
import time

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
)

from dupdetect import runtime
from dupdetect.util import summarize_error

_DUP_LINE = re.compile(r"(\d+)\s+duplicate cluster")        # "🔔 N duplicate cluster(s) just detected"
_EVENTS_ON = re.compile(r"Filesystem events: subscribed")   # watchdog active -> instant reaction
_EVENTS_OFF = re.compile(r"watchdog not installed|polling only")
_DETECT = re.compile(r"detecting=(\d+)")                    # N new files found this cycle (pre-index)
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"                                       # heartbeat spinner -> visibly alive
_CYCLE = re.compile(r"indexed=(\d+) removed=(\d+) new_dups=(\d+)")
_START = "👁 Start watching"
_IDLE = "Idle — not watching."


class WatchPanel(QGroupBox):
    """Start/stop a background watcher on a folder. Emits signals for the main window to react."""
    duplicate_detected = Signal(int)                        # n new duplicate clusters this cycle
    activity = Signal()                                     # indexed/removed something -> refresh
    cycle = Signal(int, int, int)                           # (indexed, removed, new_dups) per cycle

    def __init__(self, db_path: str, parent=None):
        super().__init__("Keep updated — background watch", parent)
        self.proc: QProcess | None = None
        self._db = db_path
        self._log: list[str] = []                           # captured output (shown if it crashes)
        self._stopping = False                              # True => exit was user-requested, not a crash
        # First-class watcher feedback: a heartbeat (proves it's alive even when idle) + a composite
        # status line built from state (mode, last activity, uptime).
        self._t0 = 0.0
        self._mode = ""                                     # "⚡ instant (watchdog)" / "⏱ polling"
        self._last = ""                                     # latest activity message
        self._spin = 0
        self._hb = QTimer(self)
        self._hb.setInterval(2000)                          # 2s tick: spinner + uptime advance
        self._hb.timeout.connect(self._render_status)

        self.folder = QLineEdit(placeholderText="Folder to watch — new/changed files auto-indexed…")
        self.folder.setToolTip("The watcher indexes new/changed files and matches them incrementally, "
                               "so you only open the app to review. No full re-scan needed day to day.")
        browse = QPushButton("Browse…"); browse.clicked.connect(self._browse)
        self.toggle = QPushButton(_START); self.toggle.clicked.connect(self._toggle)
        self.status = QLabel(_IDLE)
        row = QHBoxLayout()
        row.addWidget(QLabel("Watch:")); row.addWidget(self.folder, 1)
        row.addWidget(browse); row.addWidget(self.toggle)
        lay = QVBoxLayout(self); lay.addLayout(row); lay.addWidget(self.status)

    # ------------------------------------------------------------------ api
    def set_db(self, db: str) -> None:
        """Point the watcher at a different DB (when the window switches DB). Stops if running."""
        if self._db != db:
            self.stop()
            self._db = db

    def is_watching(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    # ------------------------------------------------------------------ ui
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Folder to watch", self.folder.text() or "")
        if d:
            self.folder.setText(d)

    def _toggle(self):
        self.stop() if self.is_watching() else self.start()

    def start(self):
        """Launch the background watcher (no-op if already running or no folder set)."""
        if self.is_watching():
            return
        folder = self.folder.text().strip()
        if not folder:
            self.status.setText("Pick a folder to watch first (Browse…).")
            return
        # --no-exact-first: the app already shows existing dups; the watcher only reacts to CHANGES
        # (re-hashing the whole library on every app open would be wasteful).
        args = ["watch", folder, "--db", self._db, "--no-exact-first"]
        self._log = []
        self._stopping = False
        prog, argv, pythonpath = runtime.cli_subprocess(args)   # frozen .exe vs python -m (dev)
        self.proc = QProcess(self)
        self.proc.setProgram(prog)
        self.proc.setArguments(argv)
        env = QProcessEnvironment.systemEnvironment()
        if pythonpath:
            env.insert("PYTHONPATH", pythonpath)
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.proc.start()
        self.toggle.setText("⏸ Stop watching")
        self._t0 = time.monotonic()
        self._mode = "starting…"
        self._last = "waiting for changes"
        self._hb.start()
        self._render_status()

    def stop(self):
        if self.is_watching():
            self._stopping = True                          # user-requested -> taskkill's nonzero exit isn't a crash
            pid = int(self.proc.processId())
            from dupdetect.util import CREATE_NO_WINDOW
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],     # kill the whole tree
                           capture_output=True, creationflags=CREATE_NO_WINDOW)
        self.toggle.setText(_START)
        self.status.setText(_IDLE)

    # ------------------------------------------------------------------ output
    def _read(self):
        raw = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for ln in re.split(r"[\r\n]", raw):
            s = ln.strip()
            if s:
                self._log.append(s)                        # kept so a crash can report the REAL cause
            self._handle_line(s)
        if len(self._log) > 800:
            del self._log[:-800]                           # bound memory (the watcher is long-running)

    def _handle_line(self, ln: str) -> None:
        """Parse ONE watcher output line -> update state/signals. Extracted for testing."""
        if not ln:
            return
        if _EVENTS_ON.search(ln):
            self._mode = "⚡ instant (watchdog)"; self._render_status(); return
        if _EVENTS_OFF.search(ln):
            self._mode = "⏱ polling (no watchdog)"; self._render_status(); return
        d = _DETECT.search(ln)
        if d:
            self._last = f"detected {d.group(1)} new file(s) — indexing…"
            self.activity.emit(); self._render_status(); return
        m = _DUP_LINE.search(ln)
        if m:
            self.duplicate_detected.emit(int(m.group(1)))
            self.activity.emit()
            return
        c = _CYCLE.search(ln)
        if c:
            idx, rem, nd = int(c.group(1)), int(c.group(2)), int(c.group(3))
            self._last = f"last cycle: {idx} indexed, {rem} removed, {nd} new dup(s)"
            self.cycle.emit(idx, rem, nd)
            if idx or rem:
                self.activity.emit()
            self._render_status()

    def _render_status(self) -> None:
        """Compose the live status (spinner + mode + last activity + uptime). Driven by the heartbeat
        timer AND on each parsed line, so the panel visibly stays alive even when the library is idle."""
        if not self.is_watching():
            return
        self._spin = (self._spin + 1) % len(_SPIN)
        mm, ss = divmod(int(time.monotonic() - self._t0), 60)
        self.status.setText(
            f"{_SPIN[self._spin]} Watching · {self._mode} · {self._last} · alive {mm:02d}:{ss:02d}")

    def _finished(self, code: int = 0, _status=None):
        self._hb.stop()                                    # stop the heartbeat; the watcher is no longer alive
        self.toggle.setText(_START)
        if self._stopping or code == 0:                    # user stop (taskkill exits nonzero) or clean exit
            if "Watching" in self.status.text():
                self.status.setText("Watcher stopped.")
            return
        # Crashed on its own: a background watcher dying silently is worse than "exit 1". Surface the
        # REAL cause inline AND in a one-time dialog (the panel has no live console / details button).
        cause = summarize_error(self._log)
        self.status.setText(f"❌ Watcher crashed (exit {code}) — {cause[:140]}")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.NoIcon)                  # no icon -> no Windows system sound
        box.setWindowTitle("Background watch failed")
        box.setText(f"The background watcher stopped unexpectedly (exit code {code}).\n\n"
                    f"Cause: {cause}")
        box.exec()
