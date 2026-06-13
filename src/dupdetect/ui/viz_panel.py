"""Live-view window — "What the AI sees" during the visual (Analyze) step.

Engagement for new, visual users on a LONG scan: it shows the actual keyframes the model is
fingerprinting, in real time. ON-DEMAND by design: opening the window signals the scan to start
streaming thumbnails (runtime.set_viz); closing it stops the stream -> zero cost when not watching
(important once the app lives mostly in the system tray). Purely cosmetic — never affects a verdict.
"""
from __future__ import annotations

from collections import deque

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from dupdetect.runtime import set_viz


class VizWindow(QWidget):
    """Standalone window that PLAYS BACK the keyframes the model is fingerprinting, DaVinci-style:
    the scan streams a burst of each video's frames (in RAM, no files), and a timer here pops them
    at a steady fps so the panel looks like the video scrubbing by at processing speed. The queue is
    bounded -> it stays near-live (drops old frames). Streaming follows the window's visibility."""

    _FPS_MS = 80                                         # ~12 fps playback
    _QUEUE_MAX = 48                                      # bounded -> near-live; drops oldest on overflow

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("👁 What the AI sees")
        self.resize(440, 360)
        self.img = QLabel("Waiting for the analysis to reach the visual step…", alignment=Qt.AlignCenter)
        self.img.setMinimumSize(380, 250)
        self.img.setStyleSheet("background:#111; color:#888; border-radius:6px;")
        self.caption = QLabel("", alignment=Qt.AlignCenter)
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("color: gray;")
        hint = QLabel("Live playback of the frames being fingerprinted. Close to stop the stream.",
                      alignment=Qt.AlignCenter)
        hint.setStyleSheet("color:#999; font-size:11px;")
        lay = QVBoxLayout(self)
        lay.addWidget(self.img, 1)
        lay.addWidget(self.caption)
        lay.addWidget(hint)
        self._queue: deque = deque(maxlen=self._QUEUE_MAX)   # (name, jpeg bytes), all in RAM
        self._timer = QTimer(self)
        self._timer.setInterval(self._FPS_MS)
        self._timer.timeout.connect(self._next_frame)

    # ON-DEMAND: streaming + playback follow the window's visibility.
    def showEvent(self, e):
        set_viz(True)
        self._timer.start()
        super().showEvent(e)

    def closeEvent(self, e):
        self._stop()
        super().closeEvent(e)

    def hideEvent(self, e):
        self._stop()                                    # minimized/hidden also stops the stream
        super().hideEvent(e)

    def _stop(self):
        set_viz(False)
        self._timer.stop()
        self._queue.clear()

    def show_frame(self, name: str, jpeg: bytes):
        """Slot: a JPEG (bytes) + filename arrives from the scan -> queue it for playback (the bounded
        deque drops the oldest if it overflows, keeping the view near-live)."""
        self._queue.append((name, jpeg))

    def _next_frame(self):
        """Timer tick: pop one queued frame and display it; if empty, hold the last frame (paused)."""
        if not self._queue:
            return
        name, jpeg = self._queue.popleft()
        pix = QPixmap()
        if not pix.loadFromData(jpeg, "JPG"):
            return
        self.img.setPixmap(pix.scaled(self.img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.caption.setText(name)
