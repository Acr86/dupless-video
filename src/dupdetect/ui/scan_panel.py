"""Panel for launching the analyzer (scan) from the UI with live progress.

Exposes the SAME options as `dupdetect scan` on the terminal: database, folder, max
resolution (--max-height, up to 8K), workers, decode-workers, recursive, scene mode, and
force recompute. Runs the scan as a SUBPROCESS (QProcess) — does not block the event loop or
mix the scan's ProcessPool/GPU with the GUI — and parses tqdm output (percentage, items, phase,
current file). On cancel it kills the ENTIRE process tree (including workers).
"""
from __future__ import annotations

import base64
import os
import re
import subprocess

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
)

from dupdetect import runtime
from dupdetect.util import summarize_error

# Pipeline stages per depth — shown as an animated stepper so a LONG scan reads as clear forward
# motion (which step, what it does, what's next) instead of one opaque bar. (short label, what it does)
_STAGES = {
    "fast": [("Find", "finding video files"), ("Hash", "byte-identical check"),
             ("Group", "grouping duplicates")],
    "standard": [("Find", "finding video files"), ("Analyze", "decoding + AI fingerprint — the longest step"),
                 ("Match", "comparing files for duplicates"), ("Group", "grouping + picking the best copy")],
    "deep": [("Find", "finding video files"), ("Analyze", "decoding + AI fingerprint — the longest step"),
             ("Match", "comparing files for duplicates"), ("Audio", "scoring audio quality on every file"),
             ("Group", "grouping + picking the best copy")],
}

# Max-resolution filter (--max-height EXCLUDES height > N). Thresholds are generous so the
# named tier is fully included (e.g. 6K at 6016×3384 fits within ≤3456).
_RES = [
    ("All (includes 8K/6K)", 0),
    ("Up to 8K (≤4320p)", 4320),
    ("Up to 6K (≤3456p)", 3456),
    ("Up to 4K (≤2160p)", 2160),
    ("Up to 1440p", 1440),
    ("Up to 1080p", 1080),
    ("Up to 720p", 720),
]
_PCT = re.compile(r"(\d+)%\|")
_CNT = re.compile(r"(\d+)/(\d+)")
_ETA_RATE = re.compile(r"<([0-9:]+),\s*([^,\]]+)")       # tqdm "[elapsed<ETA, RATE, …]"
_FRESH = re.compile(r"fresh=(\d+)")                      # fullscan postfix (in English)
_SKIP = re.compile(r"skipped=(\d+)")
_SKIP_TOTAL = re.compile(r"Skipped (\d+) file")         # resumen final del CLI


class ScanPanel(QGroupBox):
    """Scan launcher with progress. Emits `scan_finished(db)` on completion (so the window
    refreshes, and switches DB if a different one was selected)."""
    scan_finished = Signal(str)
    db_changed = Signal(str)                             # when a different DB is selected/typed -> reload tree
    frame_preview = Signal(str, bytes)                   # (filename, jpeg) for the live-view panel

    def __init__(self, db_path: str, parent=None):
        super().__init__("Analyze library", parent)
        self.proc: QProcess | None = None
        self._viz = None                                 # lazy "What the AI sees" window
        self._elapsed = 0
        self._phase_text = "Listo."                      # last known status (the clock is appended to it)
        self._log: list[str] = []                        # captured output (shown if the run fails)
        self._heartbeat = QTimer(self)
        self._heartbeat.setInterval(1000)
        self._heartbeat.timeout.connect(self._tick)

        # --- row 1: DB + folder ---
        self.db = QLineEdit(db_path)
        self.db.setToolTip("SQLite database where the index is stored (same as --db).")
        self.db.editingFinished.connect(self._on_db_edited)
        db_browse = QPushButton("DB…"); db_browse.clicked.connect(self._browse_db)
        self.folder = QLineEdit(placeholderText="Folder to analyze (in place, without copying)…")
        fld_browse = QPushButton("Browse…"); fld_browse.clicked.connect(self._browse_folder)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("DB:")); row1.addWidget(self.db, 1); row1.addWidget(db_browse)
        row1.addWidget(QLabel("Folder:")); row1.addWidget(self.folder, 1); row1.addWidget(fld_browse)

        # --- row 2: scan options + launch ---
        self.res = QComboBox()
        for t, v in _RES:
            self.res.addItem(t, v)
        self.workers = QSpinBox(); self.workers.setRange(0, 32); self.workers.setValue(0)
        self.workers.setSpecialValueText("Auto")         # 0 = AUTO: probes storage and chooses
        self.workers.setToolTip("--workers (CPU). Auto (0) = probe the storage and choose "
                                "(~2 on HDD, high on SSD/NVMe). Set a number to force it.")
        self.dworkers = QSpinBox(); self.dworkers.setRange(-1, 16); self.dworkers.setValue(-1)
        self.dworkers.setSpecialValueText("Auto")        # -1 = AUTO: follows the auto-tune
        self.dworkers.setToolTip("--decode-workers. Auto (-1) = follow the auto-tune. >1 ONLY on "
                                 "SSD/NVMe (on HDD disk concurrency thrashes).")
        self.recursive = QCheckBox("Recursive"); self.recursive.setChecked(True)
        self.recursive.setToolTip("--recursive/--no-recursive: descend into subfolders.")
        self.indep = QCheckBox("Independent scenes")
        self.indep.setToolTip("--independent-scenes: pixel-based scenes (slow, better for cam). "
                              "Default: derived from embeddings (fast).")
        self.force = QCheckBox("Force recompute")
        self.force.setToolTip("--force: re-analyze even if already in the index.")
        # Analysis depth — incremental pipeline (deeper levels REUSE shallower ones' work).
        self.depth = QComboBox()
        self.depth.addItem("Standard — find re-encodes & upgrades (recommended)", "standard")
        self.depth.addItem("Fast — byte-identical only (hash, no AI)", "fast")
        self.depth.addItem("Deep — Standard + audio quality on every file", "deep")
        self.depth.setToolTip(
            "How deep to analyze:\n"
            "• Fast: quick byte-identical sweep (hash only, ~0.1s/file). Misses re-encodes.\n"
            "• Standard (recommended): detects duplicates even across formats/resolutions; audio "
            "quality is measured only for the duplicates it finds.\n"
            "• Deep: Standard + whole-file audio quality for EVERY file (slower).\n"
            "Levels are incremental — running Deep after Standard reuses the analysis, it doesn't redo it.")
        self.start = QPushButton("▶ Analyze"); self.start.clicked.connect(self._toggle)
        row2 = QHBoxLayout()                             # numeric controls + button
        row2.addWidget(QLabel("Max resolution:")); row2.addWidget(self.res)
        row2.addWidget(QLabel("Workers:")); row2.addWidget(self.workers)
        row2.addWidget(QLabel("Decode:")); row2.addWidget(self.dworkers)
        row2.addStretch(1); row2.addWidget(self.start)

        self.viz_btn = QPushButton("👁 What the AI sees")
        self.viz_btn.setToolTip("Open a live preview of the frames the AI is analyzing. On-demand: "
                                "streaming stops when you close it, so it costs nothing otherwise.")
        self.viz_btn.clicked.connect(self._open_viz)
        row3 = QHBoxLayout()                             # options + depth, own row -> always visible
        row3.addWidget(self.recursive); row3.addWidget(self.indep); row3.addWidget(self.force)
        row3.addWidget(self.viz_btn)
        row3.addStretch(1)
        row3.addWidget(QLabel("Depth:")); row3.addWidget(self.depth)

        # Animated stage stepper: a long scan must read as clear forward motion, not a frozen bar.
        self.stepper = QLabel(""); self.stepper.setTextFormat(Qt.RichText); self.stepper.setWordWrap(True)
        self.stage_desc = QLabel(""); self.stage_desc.setWordWrap(True)
        self.stage_desc.setStyleSheet("color: gray;")
        self._stages: list = []                          # (label, what-it-does) for the current depth
        self._stage_idx = -1                             # active stage (len(stages) = all done)
        self._pulse = 0                                  # heartbeat-driven dots on the active stage

        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0)
        self.status = QLabel("Ready.")
        self.detail_btn = QPushButton("📄 View details")
        self.detail_btn.setToolTip("Full output of the last run (skipped, warnings, log).")
        self.detail_btn.setEnabled(False)
        self.detail_btn.clicked.connect(lambda: self._show_log("Analysis details"))
        row4 = QHBoxLayout()
        row4.addWidget(self.status, 1); row4.addWidget(self.detail_btn)
        lay = QVBoxLayout(self)
        lay.addLayout(row1); lay.addLayout(row2); lay.addLayout(row3)
        lay.addWidget(self.stepper); lay.addWidget(self.stage_desc)
        lay.addWidget(self.bar); lay.addLayout(row4)
        self._inputs = (self.db, db_browse, self.folder, fld_browse, self.res, self.workers,
                        self.dworkers, self.recursive, self.indep, self.force, self.depth)

    # ------------------------------------------------------------------ ui
    def _browse_db(self):
        dlg = QFileDialog(self, "SQLite database")
        dlg.setNameFilter("SQLite (*.sqlite *.db);;All (*)")
        dlg.setFileMode(QFileDialog.AnyFile)             # allows existing or new file
        dlg.setOption(QFileDialog.DontConfirmOverwrite, True)
        dlg.setAcceptMode(QFileDialog.AcceptOpen)
        if self.db.text():
            dlg.selectFile(self.db.text())
        if dlg.exec() and dlg.selectedFiles():
            self.db.setText(dlg.selectedFiles()[0])
            self.db_changed.emit(self.db.text().strip())     # reload the selected DB immediately

    def _on_db_edited(self):
        """If the user types an EXISTING DB path, reload it (don't create one from a typo)."""
        p = self.db.text().strip()
        if p and os.path.isfile(p):
            self.db_changed.emit(p)

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder to analyze", self.folder.text() or "")
        if d:
            self.folder.setText(d)

    def _toggle(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self._stop()
            return
        folder = self.folder.text().strip()
        if not folder:
            self.status.setText("Pick a folder first (Browse…).")
            return
        args = ["scan", folder,
                "--db", self.db.text().strip(),
                f"--workers={self.workers.value()}",            # 0 = Auto; '=' prevents -1 (decode)
                f"--decode-workers={self.dworkers.value()}",    # from being parsed as another option
                "--recursive" if self.recursive.isChecked() else "--no-recursive",
                "--independent-scenes" if self.indep.isChecked() else "--fast-scenes"]
        mh = self.res.currentData()
        if mh:
            args += ["--max-height", str(mh)]
        if self.force.isChecked():
            args += ["--force"]
        depth = self.depth.currentData() or "standard"
        args += ["--depth", depth]

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
        self.proc.finished.connect(self._done)
        self._elapsed = 0
        self._log = []
        self._phase_text = "Starting… (searching files; may take a while on large libraries)"
        self._stages = _STAGES.get(depth, _STAGES["standard"])
        self._stage_idx, self._pulse = 0, 0              # light up the first stage right away
        self._render_stepper()
        self.bar.setRange(0, 0)                          # "busy" (marquee) until the first %
        self._render()
        self._heartbeat.start()                          # clock ALWAYS visible -> never looks frozen
        self.detail_btn.setEnabled(False)
        self.start.setText("✗ Cancel")
        for w in self._inputs:
            w.setEnabled(False)
        self.proc.start()

    def _tick(self):
        self._elapsed += 1
        self._pulse += 1
        self._render_stepper()                           # animate the active stage (moving dots)
        self._render()

    def _render_stepper(self):
        """Renders the stage row: done (✓), active (▸ bold + pulsing dots), pending (grey). Makes a
        long scan read as 'step 2 of 4, here's what it does, here's what's next' — not a frozen bar."""
        if not self._stages:
            self.stepper.setText(""); self.stage_desc.setText(""); return
        chips = []
        for i, (label, _what) in enumerate(self._stages):
            if i < self._stage_idx:
                chips.append(f'<span style="color:#2e9e4f">✓ {label}</span>')
            elif i == self._stage_idx:
                chips.append(f'<b style="color:#1e88e5">▸ {label}{"." * (self._pulse % 4)}</b>')
            else:
                chips.append(f'<span style="color:#999">{label}</span>')
        self.stepper.setText("&nbsp;&nbsp;→&nbsp;&nbsp;".join(chips))
        if self._stage_idx >= len(self._stages):
            self.stage_desc.setText("Done.")
        elif 0 <= self._stage_idx < len(self._stages):
            self.stage_desc.setText(f"Now: {self._stages[self._stage_idx][1]}")

    def _advance_to(self, label: str):
        """Move the stepper to the stage with this short label (monotonic — never goes backward)."""
        for i, (lbl, _what) in enumerate(self._stages):
            if lbl == label and i > self._stage_idx:
                self._stage_idx = i
                self._render_stepper()
                return

    def _render(self):
        """Shows the last known status + elapsed clock, so activity is ALWAYS visible even
        when the current phase emits no lines (e.g. enumerating millions of files)."""
        mm, ss = divmod(self._elapsed, 60)
        self.status.setText(f"{self._phase_text}   ·   ⏱ {mm:d}:{ss:02d}")

    # ------------------------------------------------------------------ progreso
    def _read(self):
        raw = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        out = []
        for ln in (s.strip() for s in re.split(r"[\r\n]", raw) if s.strip()):
            if ln.startswith("VIZ:"):                    # live-view thumbnail -> panel, not log/progress
                self._emit_frame(ln)
            else:
                out.append(ln)
        self._log.extend(out)                            # save ALL non-VIZ output (shown if it fails)
        if len(self._log) > 800:
            del self._log[:-800]                         # cap memory
        if out:
            self._parse(out[-1])                         # parse the most recent line (tqdm uses \r)

    def _emit_frame(self, line: str):
        """Decode a 'VIZ:<base64-jpeg>|<filename>' line -> hand the JPEG to the live-view panel."""
        try:
            b64, name = line[4:].rsplit("|", 1)
            self.frame_preview.emit(name, base64.b64decode(b64))
        except Exception:                                # noqa: BLE001 cosmetic only
            pass

    def _open_viz(self):
        """Open (lazily create) the 'What the AI sees' window. Opening it makes the scan start
        streaming frames; closing it stops the stream (see runtime.set_viz)."""
        if self._viz is None:
            from dupdetect.ui.viz_panel import VizWindow
            self._viz = VizWindow(self)
            self.frame_preview.connect(self._viz.show_frame)
        self._viz.show(); self._viz.raise_(); self._viz.activateWindow()

    def _parse(self, line: str):
        m = _PCT.search(line)
        if m:
            if self.bar.maximum() == 0:                  # exit "busy" -> determinate
                self.bar.setRange(0, 100)
            self.bar.setValue(int(m.group(1)))
        if "Clusters:" in line:
            self.bar.setRange(0, 100); self.bar.setValue(100)
            self._stage_idx = len(self._stages)          # all stages done
            self._render_stepper()
            self._phase_text = line
        elif "Audio quality" in line:                    # Deep: whole-file coverage step
            self._advance_to("Audio")
            self._phase_text = self._progress_text("Audio quality", line)
        elif "Pass 1" in line:
            self._advance_to("Analyze")
            self._phase_text = self._progress_text("Pass 1 (analysis)", line)
        elif "Pass 2" in line or "coarse index" in line:
            self._advance_to("Match")
            self._phase_text = self._progress_text("Pass 2 (duplicates)", line)
        elif "Measuring" in line:
            self._phase_text = self._progress_text("Measuring resolution", line, with_file=False)
        elif line.startswith("Hash"):
            self._advance_to("Hash")
            self._phase_text = self._progress_text("Hashing (fast)", line, with_file=False)
        elif line.startswith(("Analyzing", "Searching", "Resolution filter", "Skipped", "Exact-only", "Fast mode")):
            self._phase_text = line
        else:
            return                                       # non-informative line: skip re-render
        self._render()

    def _progress_text(self, name: str, line: str, with_file: bool = True) -> str:
        """Builds the status from a tqdm line with ALL useful info the terminal shows:
        n/total, rate, ETA (remaining time), and fresh/skipped counters."""
        bits = [name]
        c = _CNT.search(line)
        if c:
            bits.append(f"{c.group(1)}/{c.group(2)}")
        er = _ETA_RATE.search(line)                      # "<ETA, RATE>" (absent at start: '<?,')
        if er:
            rate = er.group(2).strip()
            if rate:
                bits.append(rate)                        # e.g. 9.7s/film
            bits.append(f"ETA {er.group(1)}")            # remaining time
        fr, sk = _FRESH.search(line), _SKIP.search(line)
        if fr and fr.group(1) != "0":
            bits.append(f"fresh {fr.group(1)}")
        if sk and sk.group(1) != "0":
            bits.append(f"⚠ {sk.group(1)} skipped")
        text = " · ".join(bits)
        if with_file:                                    # filename is the last '|' segment
            tail = line.split("|")[-1].strip().rstrip("]") if "|" in line else ""
            if tail and "=" not in tail:                 # skip postfix without filename ("frescos=N")
                text += f" — {tail}"
        return text

    # ------------------------------------------------------------------ control
    def _stop(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            pid = int(self.proc.processId())
            from dupdetect.util import CREATE_NO_WINDOW
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, creationflags=CREATE_NO_WINDOW)
            self.status.setText("Canceled.")

    def _done(self, code, _status):
        self._heartbeat.stop()
        self.bar.setRange(0, 100)                         # exit "busy" state if still in it
        self.start.setText("▶ Analyze")
        for w in self._inputs:
            w.setEnabled(True)
        self.detail_btn.setEnabled(bool(self._log))       # run log available
        skipn = self._skipped_count()
        if "Canceled" in self.status.text():
            pass                                          # canceled by the user
        elif code == 0:
            self._stage_idx = len(self._stages)           # mark every stage done (✓)
            self._render_stepper()
            base = self._phase_text if "Clusters:" in self._phase_text else "Finished."
            if skipn:
                base += f"   ·   ⚠ {skipn} unreadable (View details)"
            self.status.setText(base)
        else:
            # Failed: surface the REAL cause inline AND in the dialog -> a normal user never sees
            # just "exit 1". The full traceback stays in 'View details' / the log file.
            cause = summarize_error(self._log)
            self.status.setText(f"❌ Failed (exit {code}) — {cause[:140]}")
            self._show_log("Analysis failed", code)
        self.scan_finished.emit(self.db.text().strip())

    def _skipped_count(self) -> int:
        """Number of skipped files (unreadable/corrupt), from the CLI's final summary in the log."""
        n = 0
        for ln in self._log:
            mo = _SKIP_TOTAL.search(ln)
            if mo:
                n = int(mo.group(1))
        return n

    def _show_log(self, title: str, code: int | None = None):
        """Shows captured output from the last run (the panel has no visible console):
        highlights skipped files/warnings and puts the full log in 'Show details' + a file."""
        import os
        import tempfile
        full = "\n".join(self._log) or "(no output)"
        logpath = os.path.join(tempfile.gettempdir(), "dupdetect_scan.log")
        try:
            with open(logpath, "w", encoding="utf-8") as f:
                f.write(full)
        except OSError:
            logpath = "(could not write the log)"
        # highlight actionable items: skipped (corrupt) files and warnings
        flagged = [ln for ln in self._log
                   if ln.startswith(("- ", "WARNING", "Skipped ")) or "problematic" in ln]
        info = "\n".join(flagged[-18:]) if flagged else "No skipped files or warnings."
        box = QMessageBox(self)
        box.setIcon(QMessageBox.NoIcon)                   # no icon -> no Windows system sound
        box.setWindowTitle(title)
        if code:                                          # failure: LEAD with the real cause, not just the code
            box.setText(f"The analysis failed (exit code {code}).")
            parts = [f"Cause: {summarize_error(self._log)}"]
            if flagged:                                   # skipped/warnings, if any, after the cause
                parts.append(info)
            parts.append(f"Full log: {logpath}")
            box.setInformativeText("\n\n".join(parts))
        else:
            box.setText("Analysis details.")
            box.setInformativeText(f"{info}\n\nFull log: {logpath}")
        box.setDetailedText(full)                         # 'Show details' button = the full console
        box.exec()
