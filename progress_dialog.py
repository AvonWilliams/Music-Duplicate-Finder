# Music Duplicate Finder — progress_dialog.py
# Modal dialog shown while the scan worker is running.

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from .scan_worker import ScanResult


class ProgressDialog(QDialog):
    """
    Shows scan progress and hands off to a callback when done.
    The callback receives a ScanResult and is called on the main thread.
    """

    def __init__(
        self,
        parent,
        worker,
        on_finished: Callable,
        on_error: Callable[[str], None],
    ):
        super().__init__(parent)
        self._worker     = worker
        self._on_finished = on_finished
        self._on_error   = on_error

        self.setWindowTitle("Music Duplicate Finder — Scanning…")
        self.setMinimumWidth(530)
        self.setMinimumHeight(320)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        self._title_label = QLabel("<b>Scanning your library for duplicates…</b>")
        layout.addWidget(self._title_label)

        self._status_label = QLabel("Initialising…")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)   # indeterminate until we know total
        self._bar.setTextVisible(True)
        layout.addWidget(self._bar)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self._cancel)
        layout.addWidget(buttons)

        # Wire worker signals
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._handle_finished)
        worker.error.connect(self._handle_error)

        worker.start()

    # ── Worker signal handlers ─────────────────────────────────────────────

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self._status_label.setText(message)
        if total > 0:
            self._bar.setRange(0, total)
            self._bar.setValue(current)
            self._bar.setFormat(f"{current} / {total}  (%p%)")
        else:
            self._bar.setRange(0, 0)

    def _handle_finished(self, result: "ScanResult") -> None:
        self.accept()
        self._on_finished(result)

    def _handle_error(self, message: str) -> None:
        self.reject()
        self._on_error(message)

    def _cancel(self) -> None:
        self._worker.abort()
        self._worker.wait(3000)
        self.reject()
