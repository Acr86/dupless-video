"""Qt model for the duplicate tree (QStandardItemModel). Cluster -> files; files are
checkable except the KEEP (safety lock: can never be marked for deletion)."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QStandardItem, QStandardItemModel

from dupdetect.ui.data import ClusterRow, cluster_tooltip, is_actionable

PATH_ROLE = Qt.UserRole + 1
SIZE_ROLE = Qt.UserRole + 2
KEEP_ROLE = Qt.UserRole + 3
KIND_ROLE = Qt.UserRole + 4                         # 'cluster' | 'file' | 'problem' | 'detail'

HEADERS = ["Movie / file", "Res", "Size", "Codec", "Language", "Path"]
PROBLEM_HEADERS = ["File / reason", "Size", "Folder"]


def _gb(n: int) -> str:
    return f"{n/1e9:.1f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


def build_model(clusters: list[ClusterRow],
                checked: set[str] | None = None) -> QStandardItemModel:
    """Builds the tree model. `clusters` arrive already sorted/filtered.
    `checked`: paths that should be checked (☑) after rebuilding -> preserves the
    user's selection across a refresh (e.g. after 'Set as master')."""
    checked = checked or set()
    model = QStandardItemModel()
    model.setHorizontalHeaderLabels(HEADERS)
    root = model.invisibleRootItem()
    for cl in clusters:
        warn = "" if is_actionable(cl) else "⚠ "
        audio_tag = "  ·  ⚠ audio differs — pick KEEP manually" if cl.audio_warning else ""
        color_tag = "  ·  ⚠ color differs — verify (KEEP = least clipped)" if cl.color_warning else ""
        head = QStandardItem(
            f"{warn}{cl.title}   ·   {cl.n_copies} copies · reclaim {_gb(cl.reclaimable_bytes)} "
            f"· {cl.verdict}{audio_tag}{color_tag}")
        head.setEditable(False)
        head.setData("cluster", KIND_ROLE)
        head.setData(cl.cluster_id, PATH_ROLE)
        tip = cluster_tooltip(cl)                        # explains the ⚠ on hover (first-time users)
        cells = [head] + [QStandardItem("") for _ in range(len(HEADERS) - 1)]
        for c in cells:
            c.setEditable(False)
            c.setToolTip(tip)                            # whole header row -> same hover text
        for m in cl.members:
            star = "★ " if m.is_keep else ""
            name = QStandardItem(f"{star}{m.name}")
            name.setEditable(False)
            name.setData(m.path, PATH_ROLE)
            name.setData(m.size, SIZE_ROLE)
            name.setData(m.is_keep, KEEP_ROLE)
            name.setData("file", KIND_ROLE)
            mark = f"⚠ {m.name}  ({m.audio_note})" if m.audio_bad else m.name
            if m.audio_bad:                             # explain the per-file ⚠ on hover
                name.setToolTip(f"⚠ {m.audio_note} — likely a muted or truncated rip; "
                                "prefer keeping a copy with full audio.")
            if cl.color_warning:                        # show clipping so the user sees the suggestion
                mark = f"{mark}  · clip {m.color.clip * 100:.0f}%"
            if m.is_keep:
                name.setCheckable(False)                # KEEP: never deletable
                name.setText(f"★ {mark}  (KEEP)")
            else:
                name.setText(f"{star}{mark}")
                name.setCheckable(True)
                name.setCheckState(Qt.Checked if m.path in checked else Qt.Unchecked)
            cols = [name,
                    _ro(m.res), _ro(_gb(m.size)), _ro(m.vcodec), _ro(m.lang), _ro(m.path)]
            head.appendRow(cols)
        root.appendRow(cells)
    return model


def _ro(text: str) -> QStandardItem:
    it = QStandardItem(str(text))
    it.setEditable(False)
    return it


def checked_files(model: QStandardItemModel) -> list[tuple[str, int]]:
    """[(path, size)] of checked files (KEEPs can never be checked)."""
    out: list[tuple[str, int]] = []
    root = model.invisibleRootItem()
    for i in range(root.rowCount()):
        cl = root.child(i)
        for j in range(cl.rowCount()):
            it = cl.child(j)
            if it.isCheckable() and it.checkState() == Qt.Checked:
                out.append((it.data(PATH_ROLE), int(it.data(SIZE_ROLE) or 0)))
    return out


# ------------------------------------------------- problem tree (reindex / corrupt)

def _problem_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0                                    # no longer exists / unreachable


def _why(error: str | None, repair_note: str | None, repairable: bool) -> str:
    """Reason shown BELOW the file. `repair_note` (result of the last remux) overrides
    the scan error. Repairable -> what fixes it; unrecoverable -> 'irrecoverable: …'."""
    if repair_note:
        return repair_note
    e = (error or "").strip().replace("\n", " ")[:200]
    if repairable:
        base = "slow/missing index — a remux -c copy will rebuild it (re-scan afterwards)"
        return f"{base}  ·  {e}" if e else base
    return f"irrecoverable: {e}" if e else "irrecoverable (unknown reason)"


def build_problem_model(items: list[tuple[str, str, str, str | None]],
                        repairable: bool,
                        checked: set[str] | None = None) -> QStandardItemModel:
    """Tree for the problems tab: one node per file (checkable, for VLC/delete) with a
    child detail row (the 'why', not checkable, in grey). `items`: (path, error,
    category, repair_note). `repairable`: 'reindex' tab (↻) vs 'corruptos' tab (⛔). The
    unrecoverable ones are NOT rebuilt -> Rebuild button only acts on the repairable tree."""
    checked = checked or set()
    model = QStandardItemModel()
    model.setHorizontalHeaderLabels(PROBLEM_HEADERS)
    root = model.invisibleRootItem()
    grey = QBrush(QColor(128, 128, 128))
    for path, error, _cat, note in items:
        size = _problem_size(path)
        name = QStandardItem(("↻ " if repairable else "⛔ ") + os.path.basename(path))
        name.setEditable(False)
        name.setData(path, PATH_ROLE)
        name.setData(size, SIZE_ROLE)
        name.setData("problem", KIND_ROLE)
        name.setCheckable(True)                     # checkbox = selection for VLC/delete (any file)
        name.setCheckState(Qt.Checked if path in checked else Qt.Unchecked)
        # detail row (child): the reason, in grey/italic; NOT checkable -> never triggers anything
        detail = QStandardItem("└ " + _why(error, note, repairable))
        detail.setEditable(False)
        detail.setData("detail", KIND_ROLE)
        detail.setForeground(grey)
        f = detail.font(); f.setItalic(True); detail.setFont(f)
        name.appendRow([detail, _ro(""), _ro("")])
        root.appendRow([name, _ro(_gb(size) if size else "—"), _ro(os.path.dirname(path))])
    return model


def checked_problems(model: QStandardItemModel) -> list[tuple[str, int]]:
    """[(path, size)] of checked problem files (detail rows are not counted)."""
    out: list[tuple[str, int]] = []
    root = model.invisibleRootItem()
    for i in range(root.rowCount()):
        it = root.child(i)
        if it.isCheckable() and it.checkState() == Qt.Checked:
            out.append((it.data(PATH_ROLE), int(it.data(SIZE_ROLE) or 0)))
    return out


def problem_paths(model: QStandardItemModel) -> list[str]:
    """All paths in the problem tree (for 'Rebuild' which acts on the whole group)."""
    root = model.invisibleRootItem()
    return [root.child(i).data(PATH_ROLE) for i in range(root.rowCount())]


def _hms(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 60}:{s % 60:02d}"


def _audio_warning_text(coverage: float, duration_s: float) -> str:
    """Human-readable reason for the audio warning shown below the file."""
    if coverage <= 0.001:
        return "no audio (this copy is silent — don't discard it if another copy has audio)"
    cut = coverage * duration_s
    return (f"audio ends ~{_hms(cut)} of {_hms(duration_s)} (~{coverage * 100:.0f}"
            "%) — may be cut/corrupt from that point")


def build_audio_warning_model(items: list[tuple[str, float, float]],
                              checked: set[str] | None = None) -> QStandardItemModel:
    """Tree for the 'Quality warnings' tab: one node per file (☑ to open in VLC) with a
    child detail row for the audio warning. `items`: (path, audio_coverage, duration_s)."""
    checked = checked or set()
    model = QStandardItemModel()
    model.setHorizontalHeaderLabels(PROBLEM_HEADERS)
    root = model.invisibleRootItem()
    grey = QBrush(QColor(128, 128, 128))
    for path, coverage, duration in items:
        size = _problem_size(path)
        name = QStandardItem("⚠ " + os.path.basename(path))
        name.setEditable(False)
        name.setData(path, PATH_ROLE)
        name.setData(size, SIZE_ROLE)
        name.setData("problem", KIND_ROLE)
        name.setCheckable(True)
        name.setCheckState(Qt.Checked if path in checked else Qt.Unchecked)
        detail = QStandardItem("└ " + _audio_warning_text(coverage, duration))
        detail.setEditable(False)
        detail.setData("detail", KIND_ROLE)
        detail.setForeground(grey)
        f = detail.font(); f.setItalic(True); detail.setFont(f)
        name.appendRow([detail, _ro(""), _ro("")])
        root.appendRow([name, _ro(_gb(size) if size else "—"), _ro(os.path.dirname(path))])
    return model
