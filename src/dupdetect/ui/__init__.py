"""Desktop UI (PySide6) for reviewing and acting on duplicates against an existing DB.

The UI is a VISUALIZATION + ACTION layer: reads `clusters`/`matches`/`files` and allows viewing in
VLC, deleting (to Trash by default), and giving feedback that recalibrates thresholds. Does NOT
re-scan or retrain the network. The data logic (`data`) and feedback->recalibration are testable
without Qt.
"""
