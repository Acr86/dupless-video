"""Main window of the duplicates UI (PySide6). View + action layer over the DB: reorderable
tree, open in VLC, send to Recycle Bin with a confirmation that lists everything, and feedback
that recalibrates thresholds. Does not re-scan or retrain the network."""
from __future__ import annotations

import os
import re
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMenu,
    QMessageBox, QProgressBar, QPushButton, QRadioButton, QSystemTrayIcon, QTabWidget,
    QTreeView, QVBoxLayout, QWidget,
)

from dupdetect import runtime
from dupdetect.store import FingerprintStore
from dupdetect.ui import actions, startup
from dupdetect.ui.data import clean_title, drift_report, is_actionable, load_clusters, sort_clusters
from dupdetect.ui.model import (
    KIND_ROLE, PATH_ROLE, build_audio_warning_model, build_model, build_problem_model,
    checked_files, checked_problems, problem_paths,
)
from dupdetect.ui.scan_panel import ScanPanel
from dupdetect.ui.watch_panel import WatchPanel

_SORTS = [("Most copies", "copies"), ("Most reclaimable space", "space"),
          ("Highest confidence", "confidence")]
_FILTERS = [("CERTAIN + HIGH", "actionable"), ("All", "all"), ("Review only ⚠", "review")]


def _gb(n: int) -> str:
    return f"{n/1e9:.1f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


def _fmt_eta(seconds: int) -> str:
    """Human-readable ETA. -1 (no rate yet) -> '—'. Includes hours if applicable."""
    if seconds < 0:
        return "—"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


class MainWindow(QMainWindow):
    def __init__(self, db_path: str):
        super().__init__()
        self._db_path = db_path
        self.store = FingerprintStore(db_path)
        self._settings = QSettings("Dupless Video", "Dupless Video")  # cross-session persistence
        self._hdr_ready: set = set()                    # trees whose header already has defaults applied
        self._persisted: set = set()                    # trees restored from a previous session
        self.setWindowTitle(f"Dupless Video · Duplicates — {db_path}")
        _icon = Path(__file__).with_name("icon.ico")
        if _icon.exists():
            self.setWindowIcon(QIcon(str(_icon)))
        self.resize(1100, 700)

        self.search = QLineEdit(placeholderText="Search…")
        self.sort = QComboBox(); [self.sort.addItem(t, k) for t, k in _SORTS]
        self.filt = QComboBox(); [self.filt.addItem(t, k) for t, k in _FILTERS]
        refresh = QPushButton("↻")
        bar = QHBoxLayout()
        for w in (QLabel("Sort:"), self.sort, QLabel("Show:"), self.filt, self.search, refresh):
            bar.addWidget(w)

        self.tree = QTreeView()
        self.tree.setObjectName("dup")                  # key used to persist its header
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.doubleClicked.connect(self._on_double)

        self.sel_lbl = QLabel("Selected: 0 · 0 GB")
        btn_recal = QPushButton("⚙ Recalibrate")
        btn_vlc = QPushButton("▶ Open in VLC")
        btn_del = QPushButton("🗑 Delete selection…")
        bottom = QHBoxLayout()
        bottom.addWidget(self.sel_lbl); bottom.addStretch(1)
        for b in (btn_recal, btn_vlc, btn_del):
            bottom.addWidget(b)

        self.scan_panel = ScanPanel(db_path)
        self.scan_panel.scan_finished.connect(self._on_scan_finished)
        self.scan_panel.db_changed.connect(self.switch_db)

        # Background watcher (keeps the DB current while the app is open) + native tray notifications.
        self.watch_panel = WatchPanel(db_path)
        self.watch_panel.folder.setText(self._settings.value("watch_folder", "", str))
        self.watch_panel.duplicate_detected.connect(self._on_watch_dups)
        self.watch_panel.activity.connect(self.refresh)
        self.watch_panel.cycle.connect(self._on_watch_cycle)
        self._session_indexed = 0
        self._really_quit = False
        self._tray = QSystemTrayIcon(self.windowIcon(), self)
        self._tray.setToolTip("Dupless Video — idle")
        tray_menu = QMenu()
        tray_menu.addAction("Open Dupless Video").triggered.connect(self._restore)
        self._act_watch = tray_menu.addAction("▶ Start watching")
        self._act_watch.triggered.connect(self._toggle_watch_from_tray)
        tray_menu.addSeparator()
        self._act_startup = tray_menu.addAction("Start with Windows")
        self._act_startup.setCheckable(True)
        self._act_startup.setEnabled(startup.is_supported())
        self._act_startup.setChecked(startup.is_enabled())
        self._act_startup.toggled.connect(self._set_startup)
        # Toggle: should the window's X hide to the tray (default, watcher keeps running) or exit for
        # real? Checked = minimize to tray. Persisted. Only meaningful when a system tray exists.
        self._act_close_tray = tray_menu.addAction("Minimize to tray on close")
        self._act_close_tray.setCheckable(True)
        self._act_close_tray.setEnabled(self._tray.isSystemTrayAvailable())
        self._act_close_tray.setChecked(
            self._tray.isSystemTrayAvailable() and self._settings.value("close_to_tray", True, bool))
        self._act_close_tray.toggled.connect(self._set_close_to_tray)
        tray_menu.addSeparator()
        tray_menu.addAction("Quit").triggered.connect(self._quit)
        # Always reflect the REAL watch state when the menu opens — otherwise the label goes stale
        # (idle cycles print nothing) and "Start watching" would actually STOP an active watcher.
        tray_menu.aboutToShow.connect(self._refresh_watch_action)
        self._tray.setContextMenu(tray_menu)
        self._tray.activated.connect(self._tray_activated)
        self._tray.show()
        # Closing the window hides to the tray (the watcher keeps running) — only if a tray exists,
        # else keep the normal close-quits behaviour so the app can't get stuck invisible.
        if self._tray.isSystemTrayAvailable():
            QApplication.instance().setQuitOnLastWindowClosed(False)

        # --- Tab 1: Duplicates (the main view) ---
        dup_tab = QWidget(); dup_lay = QVBoxLayout(dup_tab)
        dup_lay.addLayout(bar); dup_lay.addWidget(self.tree); dup_lay.addLayout(bottom)

        # --- Tab 2: Indexes to rebuild (valid but with slow seek) ---
        # Tree (consistent with Duplicates): one node per file (☑ for VLC/delete) + a child
        # detail row with the reason. Model is populated in _refresh_problems.
        self.reindex_model = None
        self.reindex_tree = self._make_problem_tree(); self.reindex_tree.setObjectName("reindex")  # header persist key
        self.rx_sel = QLabel("Checked: 0 · 0 GB")
        self.btn_repair = QPushButton("🔧 Rebuild indexes")
        self.btn_repair.clicked.connect(self._repair_indexes)
        rx_vlc = QPushButton("▶ Open in VLC")
        rx_vlc.clicked.connect(lambda: self._vlc_paths(self._checked_problem_paths(self.reindex_model)))
        rx_del = QPushButton("🗑 Delete")
        rx_del.clicked.connect(lambda: self._delete_problem_paths(self._checked_problem_paths(self.reindex_model)))
        rx_google = QPushButton("🔍 Google")
        rx_google.clicked.connect(lambda: self._google_checked(self.reindex_model))
        rx_export = QPushButton("📤 Export")
        rx_export.clicked.connect(lambda: self._export_problem("reindex"))
        rx_tab = QWidget(); rx_lay = QVBoxLayout(rx_tab)
        rx_lay.addWidget(QLabel(
            "VALID files but with a broken/missing index (SKIPPED because seeking was too slow). "
            "Check (☑) to open/delete; “Rebuild indexes” fixes them ALL (remux -c copy, no "
            "re-encode) so the NEXT run analyzes them as duplicates."))
        rx_lay.addWidget(self.reindex_tree)
        # Rebuild progress (hidden until launched): bar + status with ETA, same as the scan
        # (§2: a long op must never appear frozen). Heartbeat -> clock always alive.
        self.rx_progress = QProgressBar(); self.rx_progress.setRange(0, 100); self.rx_progress.hide()
        self.rx_status = QLabel(""); self.rx_status.hide()
        self._rx_hb = QTimer(self); self._rx_hb.setInterval(1000); self._rx_hb.timeout.connect(self._repair_tick)
        self._rx_elapsed = 0
        self._repair_log: list[str] = []
        rx_lay.addWidget(self.rx_progress)
        rx_bar = QHBoxLayout(); rx_bar.addWidget(self.rx_sel); rx_bar.addWidget(self.rx_status, 1)
        rx_bar.addStretch(1)
        for b in (rx_google, rx_export, rx_vlc, rx_del, self.btn_repair):
            rx_bar.addWidget(b)
        rx_lay.addLayout(rx_bar)

        # --- Tab 3: Corrupted (lost data, not repairable here) ---
        self.corrupt_model = None
        self.corrupt_tree = self._make_problem_tree(); self.corrupt_tree.setObjectName("corrupt")
        self.cx_sel = QLabel("Checked: 0 · 0 GB")
        cx_vlc = QPushButton("▶ Open in VLC")
        cx_vlc.clicked.connect(lambda: self._vlc_paths(self._checked_problem_paths(self.corrupt_model)))
        cx_del = QPushButton("🗑 Delete")
        cx_del.clicked.connect(lambda: self._delete_problem_paths(self._checked_problem_paths(self.corrupt_model)))
        btn_open = QPushButton("📂 Open folder"); btn_open.clicked.connect(self._open_corrupt_folder)
        cx_google = QPushButton("🔍 Google")
        cx_google.clicked.connect(lambda: self._google_checked(self.corrupt_model))
        cx_export = QPushButton("📤 Export")
        cx_export.clicked.connect(lambda: self._export_problem("corrupt"))
        cx_tab = QWidget(); cx_lay = QVBoxLayout(cx_tab)
        cx_lay.addWidget(QLabel(
            "⛔ Irrecoverable: lost data (missing moov, truncated…). The reason is shown under each "
            "file; they CAN'T be repaired here (not repair-selectable) — open them in VLC or delete "
            "them. Export the list / Google a title to re-acquire from a source you're licensed for."))
        cx_lay.addWidget(self.corrupt_tree)
        c_bar = QHBoxLayout(); c_bar.addWidget(self.cx_sel); c_bar.addStretch(1)
        for b in (cx_google, cx_export, cx_vlc, cx_del, btn_open):
            c_bar.addWidget(b)
        cx_lay.addLayout(c_bar)

        # --- Tab 4: Quality warnings (missing/clipped audio — video is fine) ---
        self.audio_model = None
        self.audio_tree = self._make_problem_tree(); self.audio_tree.setObjectName("audio")
        self.aw_sel = QLabel("Checked: 0 · 0 GB")
        aw_vlc = QPushButton("▶ Open in VLC")
        aw_vlc.clicked.connect(lambda: self._vlc_paths(self._checked_problem_paths(self.audio_model)))
        aw_open = QPushButton("📂 Open folder")
        aw_open.clicked.connect(lambda: self._open_folders(self._checked_problem_paths(self.audio_model)))
        aw_google = QPushButton("🔍 Google")
        aw_google.clicked.connect(lambda: self._google_checked(self.audio_model))
        aw_export = QPushButton("📤 Export")
        aw_export.clicked.connect(lambda: self._export_problem("audio"))
        aw_tab = QWidget(); aw_lay = QVBoxLayout(aw_tab)
        aw_lay.addWidget(QLabel(
            "⚠ Audio quality: these videos play fine but a copy has NO audio or the audio is "
            "cut/corrupted from some point on. NOT deleted and NOT auto-kept — verify in VLC and "
            "decide which copy to keep, so you don't end up keeping the muted one."))
        aw_lay.addWidget(self.audio_tree)
        aw_bar = QHBoxLayout(); aw_bar.addWidget(self.aw_sel); aw_bar.addStretch(1)
        for b in (aw_google, aw_export, aw_vlc, aw_open):
            aw_bar.addWidget(b)
        aw_lay.addLayout(aw_bar)

        self.tabs = QTabWidget()
        self.tabs.addTab(dup_tab, "Duplicates")
        self.tabs.addTab(rx_tab, "Indexes to rebuild")
        self.tabs.addTab(cx_tab, "Corrupted")
        self.tabs.addTab(aw_tab, "Quality warnings")

        lay = QVBoxLayout()
        lay.addWidget(self.tabs)
        lay.addWidget(self.scan_panel)                  # analyzer launcher (bottom, shared)
        lay.addWidget(self.watch_panel)                 # background watcher (keep updated)
        central = QWidget(); central.setLayout(lay); self.setCentralWidget(central)

        refresh.clicked.connect(self.refresh)
        self.sort.currentIndexChanged.connect(self.refresh)
        self.filt.currentIndexChanged.connect(self.refresh)
        self.search.textChanged.connect(self.refresh)
        btn_del.clicked.connect(self._delete_selected)
        btn_vlc.clicked.connect(self._vlc_selected)
        btn_recal.clicked.connect(self._recalibrate)
        self._coverage = QLabel("")                     # permanent badge: library analysis coverage
        self.statusBar().addPermanentWidget(self._coverage)
        self.refresh()
        QTimer.singleShot(0, self._maybe_onboard)       # one-time intro (after the window is shown)
        if startup.is_enabled() and self.watch_panel.folder.text().strip():
            QTimer.singleShot(1500, self._autostart_watch)   # resume watching when launched at login

    def switch_db(self, db: str):
        """Switches the active DB in the UI and reloads the tree. If it's the same, just refreshes.
        Used both when selecting a different DB in the panel and when a scan finishes."""
        if db and db != self._db_path:
            self.store.close()
            self.store = FingerprintStore(db)
            self._db_path = db
            self.watch_panel.set_db(db)                  # watcher follows the active DB
            self.setWindowTitle(f"Dupless Video · Duplicates — {db}")
        self.refresh()

    def _on_watch_dups(self, n: int):
        """The background watcher reported new duplicate clusters: native toast + refresh the view."""
        self._tray.showMessage(
            "Duplicates detected",
            f"{n} new duplicate group(s) found. Open Dupless Video to review and reclaim space.",
            QSystemTrayIcon.Information, 8000)
        self.refresh()

    def _on_watch_cycle(self, indexed: int, removed: int, _new_dups: int):
        """Accumulate a session total and surface it in the tray tooltip ('how many since start')."""
        self._session_indexed += indexed
        self._refresh_watch_action()

    def _refresh_watch_action(self):
        watching = self.watch_panel.is_watching()
        self._act_watch.setText("⏸ Pause watching" if watching else "▶ Start watching")
        self._tray.setToolTip(
            f"Dupless Video — {'watching' if watching else 'idle'} · "
            f"{self._session_indexed:,} analyzed this session")

    def _restore(self):
        self.showNormal(); self.raise_(); self.activateWindow()

    def _tray_activated(self, reason):
        if reason in (QSystemTrayIcon.DoubleClick, QSystemTrayIcon.Trigger):
            self._restore()

    def _toggle_watch_from_tray(self):
        self.watch_panel.stop() if self.watch_panel.is_watching() else self.watch_panel.start()
        self._refresh_watch_action()

    def _autostart_watch(self):
        self.watch_panel.start()
        self._refresh_watch_action()

    def _set_startup(self, enabled: bool):
        try:
            startup.set_enabled(enabled, self._db_path)
        except Exception as e:                          # noqa: BLE001 — surface, don't crash
            _mbox(self, "Start with Windows", f"Couldn't change the login entry:\n{e}")
            self._act_startup.setChecked(startup.is_enabled())
            return
        if enabled and not self.watch_panel.folder.text().strip():
            _mbox(
                self, "Start with Windows",
                "Enabled. Tip: set a folder in “Keep updated — background watch” so the watcher "
                "resumes automatically when the app starts at login.")

    def _set_close_to_tray(self, enabled: bool):
        """Tray toggle: ON (default) = the window's X hides to the tray (the watcher keeps running);
        OFF = the X exits the app for real. Persisted across sessions; read in closeEvent."""
        self._settings.setValue("close_to_tray", enabled)

    def _quit(self):
        """Real exit from the tray (the window's X only hides to the tray)."""
        self._really_quit = True
        self.close()
        QApplication.instance().quit()

    def _maybe_onboard(self):
        """One-time intro explaining the initial scan + background watch (no forced wait)."""
        if self._settings.value("onboarded", False, bool):
            return
        _mbox(
            self, "Welcome to Dupless Video",
            "How it works:\n\n"
            "• Run the analyzer (below) on your library ONCE. You can use the app right away — "
            "exact (byte-identical) duplicates show in minutes, and re-encode/upgrade detection "
            "completes in the background. No need to wait for it to finish.\n\n"
            "• Turn on “Keep updated — background watch” to auto-index new/changed files. You'll get "
            "a notification when a duplicate appears; open the app only to review.\n\n"
            "• Closing the window keeps it running in the tray (next to the clock). Right-click the "
            "tray icon to Quit, or to enable “Start with Windows” so it watches from login.\n\n"
            "• Acting on what's shown is safe (you're deleting a confirmed copy of a kept file). A "
            "full re-verify is only needed after changing the model/thresholds.")
        self._settings.setValue("onboarded", True)

    def _on_scan_finished(self, db: str):
        self.switch_db(db)
        self._tray.showMessage(                          # notify even if minimized to the tray
            "Analysis finished",
            "Your library scan is done. Open Dupless Video to review duplicates and reclaim space.",
            QSystemTrayIcon.Information, 8000)

    def _update_coverage(self):
        """Permanent badge: how much of the library is FULLY analyzed (has embeddings) vs exact-only.
        Tells the user when the 'best copy of everything' guarantee fully holds (100%), without
        forcing a wait — partial results are already safe to act on."""
        try:
            total = self.store.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            full = self.store.conn.execute(
                "SELECT COUNT(*) FROM files WHERE global_vec IS NOT NULL").fetchone()[0]
        except Exception:                               # noqa: BLE001 — badge is best-effort
            return
        if total == 0:
            self._coverage.setText("Library empty — run the analyzer below to start.")
            return
        if full >= total:
            self._coverage.setText(f"✓ Library fully analyzed ({total:,} files)")
        else:
            self._coverage.setText(
                f"Analyzing: {full:,}/{total:,} ({100 * full // total}%) — re-encode/upgrade "
                "detection still completing in the background")

    # ----------------------------------------------------------------- vista
    def refresh(self):
        clusters = load_clusters(self.store)
        f = self.filt.currentData()
        if f == "actionable":
            clusters = [c for c in clusters if is_actionable(c)]
        elif f == "review":
            clusters = [c for c in clusters if not is_actionable(c)]
        q = self.search.text().strip().lower()
        if q:
            clusters = [c for c in clusters if q in c.title.lower()
                        or any(q in m.path.lower() for m in c.members)]
        clusters = sort_clusters(clusters, self.sort.currentData())
        # Preserve checked state (☑) across refresh: otherwise actions like
        # 'Set as master' that rebuild the tree would uncheck everything else.
        prev_checked = {p for p, _ in checked_files(self.model)} if getattr(self, "model", None) else set()
        self.model = build_model(clusters, checked=prev_checked)
        self.model.itemChanged.connect(self._update_sel)
        self._bind_tree_model(self.tree, self.model, range(1, self.model.columnCount()))
        self._update_sel()
        self._refresh_problems()
        self._update_coverage()
        # Warn about cluster<->match desync: without it, orphaned clusters show without a verdict
        # (everything in 'Review only') and feedback won't recalibrate -> would look normal.
        drift = drift_report(self.store)
        if drift["drifted"]:
            self.statusBar().showMessage(
                f"⚠ {len(clusters)} clusters · cluster/match desync — re-scan (full) to refresh verdicts", 0)
            if not getattr(self, "_warned_drift", False):
                self._warned_drift = True
                _mbox(
                    self, "Clusters out of sync",
                    "The clusters shown are out of sync with the match scores (likely an exact-only "
                    "scan ran after a full scan). Verdicts show empty and feedback won't recalibrate "
                    "until you run a FULL scan to rebuild both from the same data.")
        else:
            self.statusBar().showMessage(f"{len(clusters)} clusters")

    def _bind_tree_model(self, tree: QTreeView, model, fit_cols) -> None:
        """Sets `model` on `tree` PRESERVING whatever the user changed in the header (column order
        and widths): `setModel` resets them on every refresh (and refresh fires on every keystroke
        in the search box), so we save the header state and restore it afterwards. The FIRST time
        for this tree it applies the defaults: movable columns, `fit_cols` sized to content and
        col 0 (title / File-reason) at 30% of the width — after that the user is in charge."""
        h = tree.header()
        first = tree not in self._hdr_ready
        state = None if first else h.saveState()        # snapshot of the user's current order+widths
        tree.setModel(model)
        tree.expandAll()
        if state is not None:
            h.restoreState(state)                        # refresh: keep what the user left
            return
        h.setSectionsMovable(True)                       # drag headers to reorder
        for col in fit_cols:
            tree.resizeColumnToContents(col)
        saved = self._settings.value(self._hdr_key(tree))
        if saved:                                        # order+widths saved from another session
            h.restoreState(saved)
            self._persisted.add(tree)                    # already has widths -> showEvent skips the 30%
        # Without saved state, showEvent sets col 0 to 30% of the real window width.
        self._hdr_ready.add(tree)

    @staticmethod
    def _hdr_key(tree: QTreeView) -> str:
        """QSettings key for a tree's header state (by its objectName)."""
        return f"headers/{tree.objectName()}"

    def _all_trees(self) -> tuple[QTreeView, ...]:
        return (self.tree, self.reindex_tree, self.corrupt_tree, self.audio_tree)

    def _save_headers(self) -> None:
        """Persists each header's state (column order + widths) so it can be restored on the next
        launch -> §3: configuration belongs to the user, it is respected."""
        for t in self._all_trees():
            self._settings.setValue(self._hdr_key(t), t.header().saveState())

    def showEvent(self, event):
        """First paint of the window: for trees WITHOUT saved state from a previous session, gives
        col 0 (title / File-reason) 30% of the real window width. Only once; after that the user is
        in charge (their widths/order are preserved across refreshes and sessions)."""
        super().showEvent(event)
        if getattr(self, "_cols_init", False):
            return
        self._cols_init = True
        w0 = max(200, int(self.width() * 0.30))
        for t in self._all_trees():
            if t not in self._persisted:                 # keep what was restored from a previous session
                t.setColumnWidth(0, w0)

    def _make_problem_tree(self) -> QTreeView:
        """Problem tree with the same look as the Duplicates tree."""
        t = QTreeView()
        t.setSelectionMode(QAbstractItemView.ExtendedSelection)
        t.doubleClicked.connect(self._on_problem_double)
        return t

    def _refresh_problems(self):
        """Populates the 'Indexes to rebuild' (↻) and 'Corrupted' (⛔) tabs from the problems
        table, as a TREE: file node (☑) + detail row with the reason. First self-heals
        orphans (file deleted but folder still present) -> §2."""
        healed = self.store.prune_missing_problems()
        reindex = self.store.problems(category="reindex")
        corrupt = self.store.problems(category="corrupt")
        audio = self.store.audio_warnings()
        self.reindex_model = build_problem_model(reindex, repairable=True)
        self.corrupt_model = build_problem_model(corrupt, repairable=False)
        self.audio_model = build_audio_warning_model(audio)
        for tree, model in ((self.reindex_tree, self.reindex_model),
                            (self.corrupt_tree, self.corrupt_model),
                            (self.audio_tree, self.audio_model)):
            model.itemChanged.connect(self._update_problem_sel)
            self._bind_tree_model(tree, model, (1, 2))   # Size, Folder to content; col 0 = 30%
        self.tabs.setTabText(1, f"Indexes to rebuild ({len(reindex)})")
        self.tabs.setTabText(2, f"Corrupted ({len(corrupt)})")
        self.tabs.setTabText(3, f"Quality warnings ({len(audio)})")
        self.btn_repair.setEnabled(bool(reindex))
        self._update_problem_sel()
        if healed:
            self.statusBar().showMessage(f"Self-healed {healed} stale problem(s) (file deleted).", 4000)

    @staticmethod
    def _checked_problem_paths(model) -> list[str]:
        """Checked (☑) paths in a problem tree; detail rows are never counted."""
        return [p for p, _s in checked_problems(model)] if model is not None else []

    def _update_problem_sel(self, *_):
        for model, lbl in ((self.reindex_model, self.rx_sel), (self.corrupt_model, self.cx_sel),
                           (self.audio_model, self.aw_sel)):
            files = checked_problems(model) if model is not None else []
            total = sum(s for _, s in files)
            lbl.setText(f"Checked: {len(files)} · {_gb(total)}")

    def _on_problem_double(self, idx):
        """Double-click on a problem file node -> open in VLC (detail rows are ignored)."""
        it = idx.model().itemFromIndex(idx.siblingAtColumn(0))
        if it is not None and it.data(KIND_ROLE) == "problem":
            actions.open_in_vlc(it.data(PATH_ROLE))

    def _vlc_paths(self, paths: list[str]) -> None:
        if not paths:
            self._toast("Check ☑ a video first to open it in VLC.")
            return
        for p in paths[:8]:                              # cap to avoid opening dozens of windows
            actions.open_in_vlc(p)

    def _google_checked(self, model) -> None:
        """Opens a Google search (in a private window) for the CLEAN TITLE of the checked files.
        It's just a normal web search to help you locate/re-acquire the title on your own."""
        paths = self._checked_problem_paths(model)
        if not paths:
            self._toast("Check ☑ a file first to search it on Google.")
            return
        for p in paths[:6]:                              # cap: don't open dozens of tabs
            actions.web_search(clean_title(p))

    def _export_problem(self, kind: str) -> None:
        """Exports the tab list (CSV or JSON) with clean title + reason: replacement list."""
        import csv
        import json
        if kind == "audio":
            src = [(p, "no audio" if cov <= 0.001 else f"incomplete audio (~{cov*100:.0f}%)")
                   for p, cov, _dur in self.store.audio_warnings()]
        else:
            src = [(p, (note or err or "")) for p, err, _cat, note in self.store.problems(category=kind)]
        if not src:
            self._toast("Nothing to export in this tab.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export list", f"{kind}_list.csv",
                                            "CSV (*.csv);;JSON (*.json)")
        if not fn:
            return
        rows = []
        for p, reason in src:
            try:
                sz = os.path.getsize(p)
            except OSError:
                sz = 0
            rows.append({"title": clean_title(p), "name": os.path.basename(p), "path": p,
                         "size_mb": round(sz / 1e6, 1), "reason": reason})
        cols = ["title", "name", "path", "size_mb", "reason"]
        if fn.lower().endswith(".json"):
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
        else:
            with open(fn, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        self._toast(f"Exported {len(rows)} item(s) to {fn}")

    def _delete_problem_paths(self, paths: list[str]) -> None:
        """Sends files from the reindex/corrupt tabs to the Recycle Bin (they are not in clusters):
        send2trash + clears their problem record. Same behaviour as deleting duplicates."""
        if not paths:
            self._toast("Check ☑ the videos you want to delete.")
            return
        if _mbox(
                self, "Delete",
                f"Send {len(paths)} file(s) to the Recycle Bin?",
                QMessageBox.Yes | QMessageBox.Cancel) != QMessageBox.Yes:
            return
        from send2trash import send2trash
        ok = 0
        for p in paths:
            try:
                send2trash(os.path.normpath(p))
                self.store.record_deletion(p, "trash", 0)
                self.store.clear_problem(p)
                ok += 1
            except Exception as e:                       # noqa: BLE001
                self.statusBar().showMessage(f"Error deleting {Path(p).name}: {e}", 4000)
        self._toast(f"Sent {ok} to the Recycle Bin.")
        self.refresh()

    def _update_sel(self, *_):
        files = checked_files(self.model)
        total = sum(s for _, s in files)
        self.sel_lbl.setText(f"Selected: {len(files)} · {_gb(total)}")

    def _toast(self, text: str, ms: int = 5000) -> None:
        """Transient, NON-MODAL, SILENT notification in the status bar — replaces noisy info pop-ups
        (a QMessageBox plays a Windows system sound per icon; the status bar makes none and doesn't
        interrupt the user with a click-to-dismiss dialog)."""
        self.statusBar().showMessage(text, ms)

    # ----------------------------------------------------------------- actions
    def _vlc_selected(self):
        files = checked_files(self.model) or self._focused_file()
        for path, _ in files[:8]:                       # cap to avoid opening 50 windows
            actions.open_in_vlc(path)

    def _on_double(self, idx):
        if self.model.itemFromIndex(idx).data(KIND_ROLE) == "file":
            p = self.model.itemFromIndex(idx.siblingAtColumn(0)).data(PATH_ROLE)
            if p:
                actions.open_in_vlc(p)

    def _focused_file(self):
        idx = self.tree.currentIndex()
        if not idx.isValid():
            return []
        it = self.model.itemFromIndex(idx.siblingAtColumn(0))
        if it and it.data(KIND_ROLE) == "file":
            return [(it.data(PATH_ROLE), int(it.data(Qt.UserRole + 2) or 0))]
        return []

    def _delete_selected(self):
        files = checked_files(self.model)
        if not files:
            self._toast("Check the files to delete (the KEEP ★ can't be).")
            return
        dlg = _ConfirmDelete(self, files)
        if dlg.exec() != QDialog.Accepted:
            return
        res = actions.delete_files(self.store, files, dest=dlg.dest(),
                                   quarantine_dir=None)
        msg = f"Deleted {len(res.deleted)} · freed {_gb(res.freed_bytes)}"
        if res.errors:
            msg += f"\n{len(res.errors)} errors:\n" + "\n".join(f"· {p}: {e}" for p, e in res.errors[:8])
        self._toast(msg)                                 # silent, non-modal — the list refresh shows the result
        self.refresh()

    def _context_menu(self, pos):
        idx = self.tree.indexAt(pos)
        if not idx.isValid():
            return
        it = self.model.itemFromIndex(idx.siblingAtColumn(0))
        if it.data(KIND_ROLE) != "file":
            return
        path = it.data(PATH_ROLE)
        keep = self._cluster_keep(it)
        is_master = keep is not None and path == keep               # click on ★ itself: no pair to label
        head = it.parent()
        cid = head.data(PATH_ROLE) if head is not None else None    # on the head, PATH_ROLE holds the cluster_id
        menu = QMenu(self)
        menu.setToolTipsVisible(True)                               # QMenu hides action tooltips by default
        a_master = menu.addAction("★ Set as master (keep this)")
        a_master.setEnabled(cid is not None and path != keep)       # already master -> nothing to do
        menu.addSeparator()
        a_diff = menu.addAction("✗ Not a duplicate (false positive)")
        a_same = menu.addAction("✓ Confirm it is a duplicate")
        # Feedback labels the pair (★ master, COPY). On the ★ itself there is no pair ->
        # disable (previously a SILENT no-op: appeared to save but saved nothing).
        for a in (a_diff, a_same):
            a.setEnabled(not is_master)
            if is_master:
                a.setToolTip("Labels the COPY, not the ★ (master). Right-click on a copy.")
        act = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if act is a_master and cid is not None:
            self.store.set_keep(int(cid), path)                     # moves the ★; others become selectable
            self.refresh()
            self._reveal_cluster(int(cid))                          # sort by reclaimable changes when
            #                                                       # the KEEP moves -> the cluster jumps
            #                                                       # position; keep it in view (not "gone")
        elif act in (a_diff, a_same) and keep and keep != path:
            self.store.save_feedback(keep, path, "different" if act is a_diff else "same")
            self.statusBar().showMessage(
                f"Feedback saved ({'not-dup' if act is a_diff else 'dup'}). Use ⚙ Recalibrate to apply.", 5000)

    def _cluster_keep(self, file_item):
        parent = file_item.parent()
        for j in range(parent.rowCount()):
            ch = parent.child(j)
            if ch.data(Qt.UserRole + 3):                # KEEP_ROLE
                return ch.data(PATH_ROLE)
        return None

    def _reveal_cluster(self, cluster_id) -> None:
        """Scroll to + select the cluster the user just acted on. Sorting by reclaimable space
        re-orders a cluster when its KEEP changes (reclaimable_bytes = Σ sizes of non-keep
        members, computed per cluster), so without this the cluster appears to 'vanish' from the
        list — it only jumped position. The 2 rows are never removed."""
        root = self.model.invisibleRootItem()
        for i in range(root.rowCount()):
            head = root.child(i)
            if head is not None and head.data(PATH_ROLE) == cluster_id:
                idx = head.index()
                self.tree.expand(idx)
                self.tree.scrollTo(idx, QAbstractItemView.PositionAtCenter)
                self.tree.setCurrentIndex(idx)
                return

    def _recalibrate(self):
        from dupdetect.config import load_thresholds
        from dupdetect.pipeline.calibrate import labeled_signals_from_feedback, suggest_thresholds
        sigs = labeled_signals_from_feedback(self.store)
        n_feedback = len(self.store.iter_feedback())
        n_usable = len(sigs)
        if n_usable < 4:
            # Don't mislead with "0": separates how many labels exist from how many are usable
            # (orphans missing fingerprints/.npy or torch can't be recovered -> re-scan).
            msg = f"{n_feedback} feedback / {n_usable} usable. Need ≥4 usable to recalibrate."
            if n_usable < n_feedback:
                msg += (f"\n\n{n_feedback - n_usable} pair(s) couldn't be recovered "
                        "(missing fingerprints/.npy, or torch unavailable). Re-scan to restore them.")
            _mbox(self, "Recalibrate", msg)
            return
        sug = suggest_thresholds(sigs, base=load_thresholds())
        n_same = sum(1 for s in sigs if s.is_same)
        orphan_note = (f"\n({n_feedback - n_usable} unrecoverable — re-scan to restore)"
                       if n_usable < n_feedback else "")
        text = (f"{n_usable} usable of {n_feedback} feedback ({n_same} dup, {n_usable-n_same} not-dup)."
                f"{orphan_note}\n\n"
                f"Suggested:  θv → {sug['theta_v']}    θa → {sug['theta_a']}\n"
                f"Result: T1/T2 false positives = {sug['false_positives_T1T2']}  ·  recall = {sug['recall_dup']}\n\n"
                "Doesn't touch detections already made; applies to future scans.")
        if _mbox(self, "Recalibrate thresholds (from your feedback)", text,
                                QMessageBox.Apply | QMessageBox.Cancel) == QMessageBox.Apply:
            p = actions.apply_thresholds(sug["theta_v"], sug["theta_a"])
            self._toast(f"Thresholds updated — written to {p}")

    # --------------------------------------------------- indexes to rebuild / corrupted
    def _repair_indexes(self):
        """Rebuilds listed indexes via remux (through the CLI, in a subprocess, without freezing
        the UI). No re-encode -> verdict unchanged; afterwards, re-scan picks them up.
        Acts on the ENTIRE repairable group (CLI iterates category='reindex'); irrecoverables
        live in the other tab -> never selectable for repair."""
        n = len(problem_paths(self.reindex_model)) if self.reindex_model is not None else 0
        if not n:
            return
        if _mbox(
                self, "Rebuild indexes",
                f"Rebuild the index of {n} file(s) via remux (no re-encode). Replaces the original "
                "ATOMICALLY (remux→verify→replace). Continue?",
                QMessageBox.Yes | QMessageBox.Cancel) != QMessageBox.Yes:
            return
        self.btn_repair.setEnabled(False); self.btn_repair.setText("Rebuilding…")
        self._repair_log = []
        self._rx_elapsed = 0
        self.rx_progress.setRange(0, 0)                  # "busy" (marquee) until the first %
        self.rx_progress.show()
        self.rx_status.setText("Starting…"); self.rx_status.show()
        self._rx_hb.start()                              # clock -> never appears frozen (§2)
        prog, argv, pythonpath = runtime.cli_subprocess(
            ["repair-indexes", "--db", self._db_path, "--apply"])    # frozen .exe vs python -m (dev)
        proc = QProcess(self)
        proc.setProgram(prog)
        proc.setArguments(argv)
        env = QProcessEnvironment.systemEnvironment()
        if pythonpath:
            env.insert("PYTHONPATH", pythonpath)
        env.insert("PYTHONIOENCODING", "utf-8")
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._repair_read)
        proc.finished.connect(self._repair_done)
        self._repair_proc = proc                         # prevent GC
        proc.start()

    _RX_PROGRESS = re.compile(
        r"REPAIR_PROGRESS idx=(\d+) total=(\d+) file_pct=(\d+) overall_pct=(\d+) "
        r"eta_s=(-?\d+) done_gb=([\d.]+) total_gb=([\d.]+) file=(.*)")

    def _repair_read(self):
        """Parses REPAIR_PROGRESS lines from the CLI -> bar (overall) + status (k/N, %, GB, ETA).
        The rest (✓/✗, summary) is saved for the final message."""
        raw = bytes(self._repair_proc.readAllStandardOutput()).decode("utf-8", "replace")
        for ln in (s.strip() for s in re.split(r"[\r\n]", raw) if s.strip()):
            m = self._RX_PROGRESS.match(ln)
            if not m:
                self._repair_log.append(ln)             # ✓/✗/summary -> for the final message
                continue
            idx, total, fpct, opct, eta, dgb, tgb, fname = m.groups()
            if self.rx_progress.maximum() == 0:
                self.rx_progress.setRange(0, 100)
            self.rx_progress.setValue(int(opct))
            self._rx_status_base = (
                f"Rebuilding {idx}/{total} · {fpct}% {fname} · {dgb}/{tgb} GB · "
                f"ETA {_fmt_eta(int(eta))}")
            self._render_repair()

    def _repair_tick(self):
        self._rx_elapsed += 1
        self._render_repair()

    def _render_repair(self):
        base = getattr(self, "_rx_status_base", "Working…")
        mm, ss = divmod(self._rx_elapsed, 60)
        self.rx_status.setText(f"{base}   ·   ⏱ {mm:d}:{ss:02d}")

    def _repair_done(self, *_):
        self._rx_hb.stop()
        self.rx_progress.hide(); self.rx_status.hide()
        self.btn_repair.setText("🔧 Rebuild indexes")
        out = "\n".join(self._repair_log[-40:]) or "Done."
        _mbox(self, "Index rebuild", out)
        self.refresh()                                  # repaint tabs (repaired items disappear)

    def _open_corrupt_folder(self):
        self._open_folders(self._checked_problem_paths(self.corrupt_model))

    def _open_folders(self, paths: list[str]) -> None:
        """Opens the folders of checked files in the file explorer (cap 8)."""
        if not paths:
            self._toast("Check ☑ a video first to open its folder.")
            return
        import subprocess
        for p in paths[:8]:
            folder = str(Path(p).parent)
            try:
                if os.name == "nt":
                    os.startfile(folder)                 # type: ignore[attr-defined]  # Windows
                else:
                    subprocess.Popen(["xdg-open", folder])
            except OSError:
                pass

    def closeEvent(self, e):
        self._settings.setValue("watch_folder", self.watch_panel.folder.text().strip())
        # X on the window = hide to the tray (the watcher keeps running in the background) WHEN the
        # "Minimize to tray on close" tray toggle is on (default). Turn it off -> the X exits for real.
        # Real exit is also always available via the tray's Quit. No tray -> always close+quit.
        close_to_tray = self._settings.value("close_to_tray", True, bool)
        if not self._really_quit and close_to_tray and self._tray.isSystemTrayAvailable():
            e.ignore()
            self.hide()
            if not self._settings.value("tray_hinted", False, bool):
                self._tray.showMessage(
                    "Still running",
                    "Dupless Video keeps watching in the background. Right-click the tray icon to Quit.",
                    QSystemTrayIcon.Information, 5000)
                self._settings.setValue("tray_hinted", True)
            return
        self._save_headers()                            # remember column order+widths across sessions
        self.watch_panel.stop()                         # don't leave the watcher subprocess running
        self.store.close()
        super().closeEvent(e)
        # setQuitOnLastWindowClosed(False) (set when a tray exists) keeps the app alive after the
        # window closes -> quit explicitly so closing the X with the toggle OFF really exits.
        QApplication.instance().quit()


def _mbox(parent, title: str, text: str, buttons=QMessageBox.Ok, default=QMessageBox.NoButton) -> int:
    """A message box with NO icon -> no Windows 'ding' (the system sound is tied to the icon).
    Returns the clicked StandardButton. Use for modal confirmations / notices that must be seen."""
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(buttons)
    if default != QMessageBox.NoButton:
        box.setDefaultButton(default)
    box.setIcon(QMessageBox.NoIcon)                      # the icon is what triggers the system sound
    return box.exec()


class _ConfirmDelete(QDialog):
    """Confirmation dialog: lists EVERYTHING being deleted + total, and the destination."""
    def __init__(self, parent, files: list[tuple[str, int]]):
        super().__init__(parent)
        self.setWindowTitle("Confirm deletion")
        self.resize(720, 420)
        total = sum(s for _, s in files)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"You are about to delete {len(files)} files · {_gb(total)}.  "
                             "The KEEP ★ are NOT touched."))
        lst = QListWidget()
        for path, size in files:
            lst.addItem(f"{_gb(size):>9}   {path}")
        lay.addWidget(lst)
        self._trash = QRadioButton("Recycle Bin"); self._trash.setChecked(True)
        self._quar = QRadioButton("Quarantine folder (.dup_trash)")
        self._perm = QRadioButton("Delete permanently")
        self._ack = QCheckBox("I understand this is irreversible")
        for w in (self._trash, self._quar, self._perm, self._ack):
            lay.addWidget(w)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText(f"Delete {len(files)} ({_gb(total)})")
        bb.accepted.connect(self._accept); bb.rejected.connect(self.reject)
        self._bb = bb
        lay.addWidget(bb)

    def dest(self) -> str:
        return "trash" if self._trash.isChecked() else "quarantine" if self._quar.isChecked() else "permanent"

    def _accept(self):
        if self.dest() == "permanent" and not self._ack.isChecked():
            _mbox(self, "Confirmation required",
                                "Check 'I understand this is irreversible' to delete permanently.")
            return
        self.accept()


_SINGLETON = "dupdetect-ui-singleton"


def run(db_path: str) -> int:
    try:                                                # own identity in the taskbar
        import ctypes                                   # AUMID in Company.Product.Version format
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DupDetector.VideoDedup.1")
    except Exception:                                   # non-Windows / no shell32
        pass
    from PySide6.QtNetwork import QLocalServer, QLocalSocket

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Dupless Video")
    _icon = Path(__file__).with_name("icon.ico")
    if _icon.exists():
        app.setWindowIcon(QIcon(str(_icon)))           # taskbar / title bar / Alt+Tab icon

    # SINGLE instance: if one is already open, ask it to focus and exit (don't open another).
    probe = QLocalSocket()
    probe.connectToServer(_SINGLETON)
    if probe.waitForConnected(200):
        probe.write(b"raise"); probe.flush(); probe.waitForBytesWritten(300)
        probe.disconnectFromServer()
        return 0                                        # one already running -> don't duplicate window
    QLocalServer.removeServer(_SINGLETON)               # clean up orphaned socket from a crash

    win = MainWindow(db_path)
    server = QLocalServer()
    server.listen(_SINGLETON)

    def _focus():
        server.nextPendingConnection()
        win.setWindowState((win.windowState() & ~Qt.WindowMinimized) | Qt.WindowActive)
        win.show(); win.raise_(); win.activateWindow()

    server.newConnection.connect(_focus)
    win._single_server = server                         # prevent GC of server
    win.show()
    return app.exec()
