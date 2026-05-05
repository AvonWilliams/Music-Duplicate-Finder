# Music Duplicate Finder — missing_fingerprints_dialog.py
# V2.0: alert shown before an AcoustID scan if some selected files have
# no acoustid_fingerprint metadata. Offers three choices:
#   - Scan anyway (skip unfingerprinted files)
#   - Cancel
#   - Show list (expands the dialog with a scrollable list of affected paths)

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)


class MissingFingerprintsDialog(QDialog):
    """
    Displays counts + optionally a list. Returns:
      QDialog.Accepted      → user chose "Scan Anyway"
      QDialog.Rejected      → user chose "Cancel"
      GENERATE_FINGERPRINTS → user chose "Generate Fingerprints"
    """

    GENERATE_FINGERPRINTS  = 2
    PARALLEL_GENERATE      = 3

    def __init__(
        self,
        total_files: int,
        with_fp_count: int,
        missing_paths: list[str],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Missing AcoustID Fingerprints")
        self.setModal(True)
        self.resize(600, 200)

        self._missing_paths = list(missing_paths)
        self._list_expanded = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        missing_count = len(missing_paths)

        # Header
        header = QLabel(
            "<b>{0}</b> of <b>{1}</b> files have no AcoustID fingerprint "
            "and will be skipped.".format(missing_count, total_files)
        )
        header.setTextFormat(Qt.TextFormat.RichText)
        header.setWordWrap(True)
        layout.addWidget(header)

        # Explanation
        explainer = QLabel(
            "AcoustID duplicate detection requires each file to have a "
            "chromaprint fingerprint calculated. Picard can compute these for "
            "you via <b>Tools → Scan</b> (or <b>Lookup</b>). Files already "
            "matched to MusicBrainz recordings will already have fingerprints; "
            "unmatched or freshly-added files may not.\n\n"
            "You can proceed now and the scan will cover only the {0} files "
            "that do have fingerprints. Files without fingerprints will be "
            "silently skipped and not counted in the results."
            .format(with_fp_count)
        )
        explainer.setTextFormat(Qt.TextFormat.RichText)
        explainer.setWordWrap(True)
        explainer.setStyleSheet("color: #555;")
        layout.addWidget(explainer)

        # List (hidden until user clicks "Show List")
        self._list_widget = QListWidget()
        self._list_widget.setVisible(False)
        self._list_widget.setAlternatingRowColors(True)
        layout.addWidget(self._list_widget, stretch=1)

        # Buttons
        btn_row = QHBoxLayout()
        self._show_btn = QPushButton("Show List of {0} Files".format(missing_count))
        self._show_btn.clicked.connect(self._toggle_list)
        btn_row.addWidget(self._show_btn)

        gen_btn = QPushButton("⚙ Generate Fingerprints for {0} Files".format(missing_count))
        gen_btn.setToolTip(
            "Ask Picard to compute AcoustID fingerprints for the files that are missing them. "
            "Re-run Find Duplicates once it finishes."
        )
        gen_btn.clicked.connect(lambda: self.done(self.GENERATE_FINGERPRINTS))
        btn_row.addWidget(gen_btn)

        fast_btn = QPushButton("⚡ Fast Fingerprint — Parallel")
        fast_btn.setToolTip(
            "Compute fingerprints for all {0} missing files in parallel using all CPU cores "
            "(via fpcalc). Much faster than Picard's built-in sequential scan for large "
            "libraries. Does not perform an AcoustID lookup — only computes the local "
            "fingerprint needed for duplicate detection. Re-run Find Duplicates once done."
            .format(missing_count)
        )
        fast_btn.setStyleSheet("color: #007700; font-weight: bold;")
        fast_btn.clicked.connect(lambda: self.done(self.PARALLEL_GENERATE))
        btn_row.addWidget(fast_btn)

        btn_row.addStretch()

        buttons = QDialogButtonBox()
        self._scan_btn   = buttons.addButton(
            "Scan {0} Files Anyway".format(with_fp_count),
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self._cancel_btn = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_row.addWidget(buttons)

        layout.addLayout(btn_row)

        # Defaults: Cancel (safest — user has to consciously opt in)
        self._cancel_btn.setDefault(True)
        self._cancel_btn.setFocus()

        # If zero files have fingerprints, don't offer "Scan Anyway" at all
        if with_fp_count == 0:
            self._scan_btn.setEnabled(False)
            self._scan_btn.setToolTip(
                "No files have fingerprints — nothing to scan. "
                "Run Picard's Tools → Scan first to compute them."
            )

    def _toggle_list(self) -> None:
        if self._list_expanded:
            self._list_widget.setVisible(False)
            self._show_btn.setText(
                "Show List of {0} Files".format(len(self._missing_paths))
            )
            self.resize(self.width(), 200)
        else:
            if self._list_widget.count() == 0:
                # Populate lazily on first expand
                for p in self._missing_paths:
                    self._list_widget.addItem(p)
            self._list_widget.setVisible(True)
            self._show_btn.setText("Hide List")
            self.resize(self.width(), 500)
        self._list_expanded = not self._list_expanded


class CpuFallbackWarningDialog(QDialog):
    """
    Shown when the AcoustID engine is about to run on CPU (no GPU available).
    Gives a library-size-adjusted time estimate and makes the user confirm.
    """

    # Rough wall-clock estimates at N=50k for each alignment preset,
    # scaled linearly-quadratically based on library size.
    _CPU_BENCH_AT_50K = {
        "off":          60,    # seconds
        "narrow":      180,
        "standard":   1200,
        "wide":       4800,
        "exhaustive": 12000,
    }

    def __init__(
        self,
        library_size: int,
        alignment: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("GPU Not Available")
        self.setModal(True)
        self.resize(550, 260)

        bench = self._CPU_BENCH_AT_50K.get(alignment, 1200)
        # O(N²) work scaling
        factor = (library_size / 50_000.0) ** 2
        est_seconds = max(30, int(bench * factor))
        est_str = _format_duration(est_seconds)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        header = QLabel("<b>No CUDA GPU detected.</b>")
        header.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(header)

        body = QLabel(
            "The AcoustID duplicate engine uses CUDA for fast fingerprint "
            "comparison. CUDA isn't available on this machine (no NVIDIA GPU, "
            "or torch wasn't installed with CUDA support).\n\n"
            "CPU fallback works but is significantly slower, especially with "
            "higher alignment settings.\n\n"
            "<b>Library size:</b> {0:,} files<br>"
            "<b>Alignment:</b> {1}<br>"
            "<b>Estimated CPU time:</b> ≈ {2}\n\n"
            "The scan can be cancelled at any time.".format(
                library_size, alignment, est_str,
            )
        )
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setWordWrap(True)
        layout.addWidget(body)

        btns = QDialogButtonBox()
        self._run_btn = btns.addButton(
            "Run on CPU Anyway",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        cancel = btns.addButton(QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        cancel.setDefault(True)
        cancel.setFocus()


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return "{0} seconds".format(seconds)
    if seconds < 3600:
        return "{0} minutes".format(seconds // 60)
    if seconds < 86400:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        if m == 0:
            return "{0} hours".format(h)
        return "{0} hours {1} min".format(h, m)
    return "{0}+ hours".format(seconds // 3600)
