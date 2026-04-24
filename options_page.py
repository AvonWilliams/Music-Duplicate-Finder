# Music Duplicate Finder — options_page.py
# Settings page shown in Picard's Options dialog.
# V1.1: added Local GPU inference mode.

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from picard.plugin3.api import OptionsPage

from .local_inference import available_gpus, check_dependencies
from .server_client import ServerClient
from .config_util import cfg_get


# ── Reusable widgets ───────────────────────────────────────────────────────

class ThresholdRow(QWidget):
    """Colour-badged label + synced slider + spinbox for a single threshold."""

    def __init__(self, label: str, color: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        badge = QLabel(label)
        badge.setFixedWidth(60)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"background:{color}; color:#fff; border-radius:4px; "
            "padding:2px 6px; font-weight:bold; font-size:11px;"
        )
        layout.addWidget(badge)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(50, 100)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(5)
        layout.addWidget(self.slider, stretch=1)

        self.spinbox = QSpinBox()
        self.spinbox.setRange(50, 100)
        self.spinbox.setSuffix(" %")
        self.spinbox.setFixedWidth(72)
        layout.addWidget(self.spinbox)

        self.slider.valueChanged.connect(self.spinbox.setValue)
        self.spinbox.valueChanged.connect(self.slider.setValue)

    def value(self) -> int:
        return self.spinbox.value()

    def set_value(self, v: int) -> None:
        self.spinbox.setValue(v)


# ── Remote panel ───────────────────────────────────────────────────────────

class RemotePanel(QWidget):
    """Settings for remote GPU inference server mode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        server_box = QGroupBox("GPU Inference Server")
        server_form = QFormLayout(server_box)
        server_form.setSpacing(8)

        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("10.0.0.69")
        server_form.addRow("Host:", self.host_edit)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setGroupSeparatorShown(False)
        server_form.addRow("Port:", self.port_spin)

        # API key (password-masked by default, with a show/hide toggle)
        key_row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("X-API-Key sent with every request")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_row.addWidget(self.api_key_edit, stretch=1)

        self._show_key_btn = QPushButton("👁 Show")
        self._show_key_btn.setFixedWidth(70)
        self._show_key_btn.setCheckable(True)
        self._show_key_btn.setToolTip("Toggle API key visibility")
        self._show_key_btn.toggled.connect(self._toggle_key_visible)
        key_row.addWidget(self._show_key_btn)
        server_form.addRow("API key:", key_row)

        test_row = QHBoxLayout()
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.setFixedWidth(140)
        self.test_btn.clicked.connect(self._test_connection)
        self.test_status = QLabel("")
        self.test_status.setStyleSheet("font-size: 11px;")
        test_row.addWidget(self.test_btn)
        test_row.addWidget(self.test_status)
        test_row.addStretch()
        server_form.addRow("", test_row)
        layout.addWidget(server_box)

        path_box = QGroupBox("Path Remapping  (Windows → LXC)")
        path_form = QFormLayout(path_box)
        path_form.setSpacing(8)

        self.win_root_edit = QLineEdit()
        self.win_root_edit.setPlaceholderText(r"Z:\Multimedia\Audio\Music")
        path_form.addRow("Windows root:", self.win_root_edit)

        self.lxc_root_edit = QLineEdit()
        self.lxc_root_edit.setPlaceholderText("/mnt/music")
        path_form.addRow("LXC root:", self.lxc_root_edit)

        path_note = QLabel(
            "Both machines must mount the library at these paths so the "
            "server can locate files by their remapped path."
        )
        path_note.setStyleSheet("color: #666; font-size: 11px;")
        path_note.setWordWrap(True)
        path_form.addRow("", path_note)
        layout.addWidget(path_box)
        layout.addStretch()

    def _toggle_key_visible(self, visible: bool) -> None:
        self.api_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        )
        self._show_key_btn.setText("🙈 Hide" if visible else "👁 Show")

    def _test_connection(self) -> None:
        host    = self.host_edit.text().strip() or "10.0.0.69"
        port    = self.port_spin.value()
        api_key = self.api_key_edit.text()
        self.test_btn.setEnabled(False)
        self.test_status.setText("Testing…")
        self.test_status.setStyleSheet("color: #888; font-size: 11px;")
        ok, msg = ServerClient.ping(host, port, api_key=api_key, timeout=5)
        if ok:
            self.test_status.setText(f"✓  {msg}")
            self.test_status.setStyleSheet("color: #1a7f37; font-size: 11px;")
        else:
            self.test_status.setText(f"✗  {msg}")
            self.test_status.setStyleSheet("color: #cf222e; font-size: 11px;")
        self.test_btn.setEnabled(True)


# ── Local GPU panel ────────────────────────────────────────────────────────

class LocalGpuPanel(QWidget):
    """Settings for local GPU inference mode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Dependency status ─────────────────────────────────────────────────
        self._dep_box = QGroupBox("Dependency Status")
        dep_layout = QVBoxLayout(self._dep_box)
        dep_layout.setSpacing(8)

        self._dep_label = QLabel()
        self._dep_label.setWordWrap(True)
        dep_layout.addWidget(self._dep_label)

        self._dep_install_label = QLabel()
        self._dep_install_label.setWordWrap(True)
        self._dep_install_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._dep_install_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; "
            "background: #f6f8fa; color: #24292f; "
            "padding: 8px; border-radius: 4px; border: 1px solid #d0d7de;"
        )
        dep_layout.addWidget(self._dep_install_label)

        action_row = QHBoxLayout()
        self._copy_btn = QPushButton("📋  Copy command")
        self._copy_btn.setFixedWidth(150)
        self._copy_btn.setToolTip("Copy the install command to the clipboard")
        self._copy_btn.clicked.connect(self._copy_install_cmd)
        action_row.addWidget(self._copy_btn)

        dep_refresh_btn = QPushButton("↻  Re-check")
        dep_refresh_btn.setFixedWidth(100)
        dep_refresh_btn.clicked.connect(self._refresh_deps)
        action_row.addWidget(dep_refresh_btn)

        pytorch_link = QLabel(
            '<a href="https://pytorch.org/get-started/locally/" '
            'style="color:#0969da; text-decoration:none;">'
            'PyTorch install guide  ↗</a>'
        )
        pytorch_link.setTextFormat(Qt.TextFormat.RichText)
        pytorch_link.setOpenExternalLinks(True)
        pytorch_link.setStyleSheet("font-size: 11px; padding-left: 8px;")
        pytorch_link.setToolTip(
            "Open pytorch.org to pick the right torch build "
            "for your CUDA version"
        )
        action_row.addWidget(pytorch_link)
        action_row.addStretch()

        self._copy_confirm = QLabel("")
        self._copy_confirm.setStyleSheet(
            "color: #1a7f37; font-size: 11px; font-weight: bold;"
        )
        action_row.addWidget(self._copy_confirm)
        dep_layout.addLayout(action_row)

        dep_note = QLabel(
            "The command above targets Picard's own Python environment.  "
            "For NVIDIA GPU support, follow the PyTorch guide to pick the right "
            "CUDA wheel instead of the default CPU-only build."
        )
        dep_note.setStyleSheet("color: #666; font-size: 11px;")
        dep_note.setWordWrap(True)
        dep_layout.addWidget(dep_note)

        self._refresh_deps()
        layout.addWidget(self._dep_box)

        # GPU selection ─────────────────────────────────────────────────────
        gpu_box = QGroupBox("GPU Selection")
        gpu_form = QFormLayout(gpu_box)
        gpu_form.setSpacing(8)

        gpu_row = QHBoxLayout()
        self.gpu_combo = QComboBox()
        self.gpu_combo.setMinimumWidth(280)
        gpu_row.addWidget(self.gpu_combo)

        gpu_refresh_btn = QPushButton("↻ Refresh")
        gpu_refresh_btn.setFixedWidth(90)
        gpu_refresh_btn.clicked.connect(self._refresh_gpus)
        gpu_row.addWidget(gpu_refresh_btn)
        gpu_row.addStretch()
        gpu_form.addRow("Device:", gpu_row)

        gpu_note = QLabel(
            "If no CUDA GPUs appear, inference will fall back to CPU "
            "(significantly slower for large libraries)."
        )
        gpu_note.setStyleSheet("color: #666; font-size: 11px;")
        gpu_note.setWordWrap(True)
        gpu_form.addRow("", gpu_note)
        layout.addWidget(gpu_box)
        self._refresh_gpus()

        # Model cache ───────────────────────────────────────────────────────
        model_box = QGroupBox("CLAP Model Cache")
        model_form = QFormLayout(model_box)
        model_form.setSpacing(8)

        cache_row = QHBoxLayout()
        self.cache_edit = QLineEdit()
        self.cache_edit.setPlaceholderText(
            "Leave blank to use HuggingFace default  (~/.cache/huggingface)"
        )
        cache_row.addWidget(self.cache_edit, stretch=1)

        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._browse_cache)
        cache_row.addWidget(browse_btn)
        model_form.addRow("Cache dir:", cache_row)

        model_note = QLabel(
            "laion/clap-htsat-unfused will be downloaded on first use "
            "(~1 GB). Subsequent scans reuse the cached weights."
        )
        model_note.setStyleSheet("color: #666; font-size: 11px;")
        model_note.setWordWrap(True)
        model_form.addRow("", model_note)

        warm_row = QHBoxLayout()
        self._warm_btn = QPushButton("Pre-load Model Now")
        self._warm_btn.setFixedWidth(160)
        self._warm_btn.setToolTip(
            "Download and load the model weights now so the first scan is fast"
        )
        self._warm_btn.clicked.connect(self._prewarm_model)
        self._warm_status = QLabel("")
        self._warm_status.setStyleSheet("font-size: 11px;")
        self._warm_status.setWordWrap(True)
        warm_row.addWidget(self._warm_btn)
        warm_row.addWidget(self._warm_status, stretch=1)
        model_form.addRow("", warm_row)
        layout.addWidget(model_box)

        layout.addStretch()

    # ── GPU helpers ────────────────────────────────────────────────────────

    def _refresh_gpus(self) -> None:
        current = self.gpu_combo.currentIndex()
        self.gpu_combo.clear()
        gpus = available_gpus()
        if gpus:
            for name in gpus:
                self.gpu_combo.addItem(name)
            if 0 <= current < len(gpus):
                self.gpu_combo.setCurrentIndex(current)
        else:
            self.gpu_combo.addItem("No CUDA GPUs detected — will use CPU")

    def selected_gpu_index(self) -> int:
        return max(0, self.gpu_combo.currentIndex())

    def set_gpu_index(self, idx: int) -> None:
        if 0 <= idx < self.gpu_combo.count():
            self.gpu_combo.setCurrentIndex(idx)

    # ── Dependency helpers ─────────────────────────────────────────────────

    def _refresh_deps(self) -> None:
        import sys

        # Import lazily so this page doesn't blow up if local_inference has
        # problems that only surface when its module body runs.
        from .local_inference import check_dependencies, check_ffmpeg

        missing = check_dependencies()
        have_ffmpeg = check_ffmpeg()
        py_exe  = sys.executable or "python"

        # Combine into one status line covering both Python deps and ffmpeg.
        problems: list[str] = []
        if missing:
            problems.append("missing Python packages: " + ", ".join(missing))
        if not have_ffmpeg:
            problems.append("ffmpeg not on PATH (needed for .mka / Matroska)")

        if not problems:
            self._dep_label.setText(
                "✓  All required dependencies are installed  "
                "(numpy, torch, transformers, soundfile, scipy, ffmpeg)."
            )
            self._dep_label.setTextFormat(Qt.TextFormat.PlainText)
            self._dep_label.setStyleSheet("color: #1a7f37; font-size: 12px;")
            self._dep_install_label.setText("")
            self._dep_install_label.hide()
            self._copy_btn.setEnabled(False)
        else:
            # Python install command targets Picard's actual interpreter.
            if missing:
                pkgs = " ".join(missing)
                if " " in py_exe:
                    cmd = f'"{py_exe}" -m pip install {pkgs}'
                else:
                    cmd = f'{py_exe} -m pip install {pkgs}'
            else:
                cmd = ""

            status_html = "✗  " + "<br>✗  ".join(problems)
            if not have_ffmpeg:
                status_html += (
                    "<br><br>Install ffmpeg:<br>"
                    "&nbsp;&nbsp;Linux:&nbsp;&nbsp;<code>sudo apt install ffmpeg</code><br>"
                    "&nbsp;&nbsp;macOS:&nbsp;&nbsp;<code>brew install ffmpeg</code><br>"
                    "&nbsp;&nbsp;Windows: download from https://ffmpeg.org/ "
                    "and add to PATH"
                )
            if cmd:
                status_html += (
                    "<br><br>Install Python packages (in a terminal), "
                    "then click <b>Re-check</b>:"
                )
            self._dep_label.setText(status_html)
            self._dep_label.setTextFormat(Qt.TextFormat.RichText)
            self._dep_label.setStyleSheet("color: #cf222e; font-size: 12px;")
            self._dep_install_label.setText(cmd)
            if cmd:
                self._dep_install_label.show()
                self._copy_btn.setEnabled(True)
            else:
                self._dep_install_label.hide()
                self._copy_btn.setEnabled(False)

    def _copy_install_cmd(self) -> None:
        cmd = self._dep_install_label.text()
        if not cmd:
            return
        QApplication.clipboard().setText(cmd)
        self._copy_confirm.setText("✓ Copied!")
        QTimer.singleShot(2500, lambda: self._copy_confirm.setText(""))

    # ── Cache browser ──────────────────────────────────────────────────────

    def _browse_cache(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select model cache directory")
        if d:
            self.cache_edit.setText(d)

    # ── Model pre-warm ─────────────────────────────────────────────────────

    def _prewarm_model(self) -> None:
        missing = check_dependencies()
        if missing:
            self._warm_status.setText(
                f"✗  Install missing packages first: {' '.join(missing)}"
            )
            self._warm_status.setStyleSheet("color: #cf222e; font-size: 11px;")
            return

        from .local_inference import LocalInferenceEngine

        self._warm_btn.setEnabled(False)
        self._warm_status.setText("Loading…")
        self._warm_status.setStyleSheet("color: #888; font-size: 11px;")

        def _progress(msg: str) -> None:
            self._warm_status.setText(msg)
            self._warm_status.repaint()

        try:
            engine = LocalInferenceEngine(
                gpu_index       = self.selected_gpu_index(),
                model_cache_dir = self.cache_edit.text().strip() or None,
                progress_cb     = _progress,
            )
            engine.load_model()
            self._warm_status.setText("✓  Model loaded and ready")
            self._warm_status.setStyleSheet("color: #1a7f37; font-size: 11px;")
        except Exception as exc:  # noqa: BLE001
            self._warm_status.setText(f"✗  {exc}")
            self._warm_status.setStyleSheet("color: #cf222e; font-size: 11px;")
        finally:
            self._warm_btn.setEnabled(True)


# ══════════════════════════════════════════════════════════════════════════
# Chromaprint (AcoustID) panel — V2.0
# ══════════════════════════════════════════════════════════════════════════

class ChromaprintPanel(QWidget):
    """
    Settings for the AcoustID / Chromaprint duplicate engine.
    Sits in its own QGroupBox on the main options page alongside the
    existing CLAP panels.
    """

    # Combo box display strings and their stored config values
    _ALIGNMENT_CHOICES = [
        ("Off — exact match only  (~0s tolerance)",       "off"),
        ("Narrow — ±0.5s tolerance",                       "narrow"),
        ("Standard — ±2.5s tolerance  (recommended)",      "standard"),
        ("Wide — ±10s tolerance  (slower)",                "wide"),
        ("Exhaustive — ±25s tolerance  (much slower)",     "exhaustive"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── Thresholds ────────────────────────────────────────────────────
        thresh_box = QGroupBox("AcoustID Thresholds")
        thresh_form = QFormLayout(thresh_box)
        thresh_form.setSpacing(10)

        self.certain_row = ThresholdRow("CERTAIN", "#1a7f37")
        self.likely_row  = ThresholdRow("LIKELY",  "#9a6700")
        self.unsure_row  = ThresholdRow("UNSURE",  "#cf222e")

        thresh_form.addRow("Certain (≥):", self.certain_row)
        thresh_form.addRow("Likely  (≥):", self.likely_row)
        thresh_form.addRow("Unsure  (≥):", self.unsure_row)

        note = QLabel(
            "AcoustID similarities are sharp: true duplicates land at 0.95+, "
            "unrelated tracks typically &lt; 0.30. The recommended starting "
            "point is 95 / 85 / 75."
        )
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setStyleSheet("color: #666; font-size: 11px;")
        note.setWordWrap(True)
        thresh_form.addRow("", note)
        layout.addWidget(thresh_box)

        # ── Alignment tolerance ───────────────────────────────────────────
        align_box = QGroupBox("Alignment Tolerance")
        align_layout = QVBoxLayout(align_box)
        align_layout.setSpacing(6)

        self.alignment_combo = QComboBox()
        for label, _ in self._ALIGNMENT_CHOICES:
            self.alignment_combo.addItem(label)
        align_layout.addWidget(self.alignment_combo)

        align_note = QLabel(
            "Controls how many offsets the fingerprint comparison tries when "
            "looking for a match. Higher settings catch duplicates with "
            "different leading silence or padding (e.g. between rippers) at "
            "the cost of longer scan time. <b>Standard</b> matches AcoustID's "
            "own behaviour and is the right choice for most libraries."
        )
        align_note.setTextFormat(Qt.TextFormat.RichText)
        align_note.setStyleSheet("color: #666; font-size: 11px;")
        align_note.setWordWrap(True)
        align_layout.addWidget(align_note)
        layout.addWidget(align_box)

        # ── Hardware ──────────────────────────────────────────────────────
        hw_box = QGroupBox("Hardware")
        hw_layout = QVBoxLayout(hw_box)
        hw_layout.setSpacing(6)

        from PyQt6.QtWidgets import QCheckBox
        self.use_gpu_check = QCheckBox(
            "Use GPU (CUDA) for fingerprint comparison when available"
        )
        hw_layout.addWidget(self.use_gpu_check)

        hw_note = QLabel(
            "With GPU, a 50,000-file library scans in seconds to a few "
            "minutes depending on alignment. CPU is supported as a fallback "
            "but can take hours on large libraries — you'll be warned before "
            "a scan starts."
        )
        hw_note.setStyleSheet("color: #666; font-size: 11px;")
        hw_note.setWordWrap(True)
        hw_layout.addWidget(hw_note)
        layout.addWidget(hw_box)

        # ── Requirements note ─────────────────────────────────────────────
        req = QLabel(
            "<b>Setup:</b> AcoustID detection requires fingerprints stored "
            "on each file. In Picard, select your files and run "
            "<b>Tools → Scan</b> (or <b>Lookup</b>) to compute them. Files "
            "already matched to MusicBrainz recordings will already have "
            "fingerprints.  Dependencies: <code>pyacoustid</code>, "
            "<code>numpy</code>, and <code>torch</code> (for GPU)."
        )
        req.setTextFormat(Qt.TextFormat.RichText)
        req.setStyleSheet(
            "color: #555; font-size: 11px; padding: 6px; "
            "background: #f6f8fa; border-radius: 4px;"
        )
        req.setWordWrap(True)
        layout.addWidget(req)

        layout.addStretch()

    # Helpers for main page load/save
    def selected_alignment(self) -> str:
        return self._ALIGNMENT_CHOICES[self.alignment_combo.currentIndex()][1]

    def set_alignment(self, value: str) -> None:
        for i, (_, v) in enumerate(self._ALIGNMENT_CHOICES):
            if v == value:
                self.alignment_combo.setCurrentIndex(i)
                return
        # Unknown value — default to Standard
        for i, (_, v) in enumerate(self._ALIGNMENT_CHOICES):
            if v == "standard":
                self.alignment_combo.setCurrentIndex(i)
                return


# ══════════════════════════════════════════════════════════════════════════
# Main options page
# ══════════════════════════════════════════════════════════════════════════

class DuplicateFinderOptionsPage(OptionsPage):
    NAME   = "music_duplicate_finder"
    TITLE  = "Music Duplicate Finder"
    PARENT = "plugins"

    def __init__(self):
        super().__init__()

        # Wrap everything in a scroll area so the page never overflows the
        # Options dialog on smaller screens.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        _inner = QWidget()
        root = QVBoxLayout(_inner)
        root.setSpacing(16)

        scroll.setWidget(_inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ── Header ────────────────────────────────────────────────────────
        intro = QLabel(
            "<b>Two independent engines</b> are available under the Tools menu:<br>"
            "• <b>Find Duplicates (AcoustID)</b> — true duplicate detection via "
            "chromaprint fingerprints. Fast, precise, finds the same master recording.<br>"
            "• <b>Find Similar Songs (CLAP)</b> — neural similarity; clusters by "
            "genre, mood, and production. Finds songs that <i>sound alike</i>."
        )
        intro.setTextFormat(Qt.TextFormat.RichText)
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "color: #333; font-size: 11px; padding: 8px; "
            "background: #f6f8fa; border-radius: 4px;"
        )
        root.addWidget(intro)

        # ── AcoustID panel (V2.0 — primary duplicate detection) ───────────
        acid_box = QGroupBox("AcoustID  /  Chromaprint Engine")
        acid_layout = QVBoxLayout(acid_box)
        self._cp_panel = ChromaprintPanel()
        acid_layout.addWidget(self._cp_panel)
        root.addWidget(acid_box)

        # ── CLAP thresholds (shared between remote and local CLAP) ────────
        thresh_box = QGroupBox("CLAP Similarity Thresholds")
        thresh_form = QFormLayout(thresh_box)
        thresh_form.setSpacing(10)

        self.certain_row = ThresholdRow("CERTAIN", "#1a7f37")
        self.likely_row  = ThresholdRow("LIKELY",  "#9a6700")
        self.unsure_row  = ThresholdRow("UNSURE",  "#cf222e")

        thresh_form.addRow("Certain (≥):", self.certain_row)
        thresh_form.addRow("Likely  (≥):", self.likely_row)
        thresh_form.addRow("Unsure  (≥):", self.unsure_row)

        note = QLabel(
            "CLAP similarities have a narrow distribution (mean ~0.55 on real "
            "libraries). These sliders only affect the \"Find Similar Songs\" "
            "menu item. AcoustID has its own thresholds above."
        )
        note.setStyleSheet("color: #666; font-size: 11px;")
        note.setWordWrap(True)
        thresh_form.addRow("", note)
        root.addWidget(thresh_box)

        # ── CLAP inference mode selector ──────────────────────────────────
        mode_box = QGroupBox("CLAP Inference Mode")
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.setSpacing(6)

        self._mode_group = QButtonGroup(self)
        self._remote_radio = QRadioButton(
            "Remote GPU Server  —  send files to FastAPI server on another machine"
        )
        self._local_radio = QRadioButton(
            "Local GPU  —  embed and compare audio directly on this machine"
        )
        self._mode_group.addButton(self._remote_radio, 0)
        self._mode_group.addButton(self._local_radio,  1)

        mode_layout.addWidget(self._remote_radio)
        mode_layout.addWidget(self._local_radio)
        root.addWidget(mode_box)

        # ── Stacked panel (switches with radio buttons) ───────────────────
        self._stack = QStackedWidget()
        self._remote_panel = RemotePanel()
        self._local_panel  = LocalGpuPanel()
        self._stack.addWidget(self._remote_panel)  # index 0
        self._stack.addWidget(self._local_panel)   # index 1
        root.addWidget(self._stack)

        self._mode_group.idToggled.connect(self._on_mode_changed)

        # ── Diagnostics ────────────────────────────────────────────────────
        diag_box = QGroupBox("Diagnostics")
        diag_layout = QVBoxLayout(diag_box)
        diag_layout.setSpacing(6)

        from .diag import log_file_path
        self._log_path = str(log_file_path())

        path_label = QLabel(f"Log file:  <code>{self._log_path}</code>")
        path_label.setTextFormat(Qt.TextFormat.RichText)
        path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        path_label.setWordWrap(True)
        diag_layout.addWidget(path_label)

        diag_row = QHBoxLayout()

        self._open_log_btn = QPushButton("📂  Open Log Folder")
        self._open_log_btn.setFixedWidth(180)
        self._open_log_btn.setToolTip(
            "Open the folder containing the diagnostic log file in your file manager"
        )
        self._open_log_btn.clicked.connect(self._open_log_folder)
        diag_row.addWidget(self._open_log_btn)

        self._copy_log_path_btn = QPushButton("📋  Copy Path")
        self._copy_log_path_btn.setFixedWidth(130)
        self._copy_log_path_btn.setToolTip("Copy the log file path to the clipboard")
        self._copy_log_path_btn.clicked.connect(self._copy_log_path)
        diag_row.addWidget(self._copy_log_path_btn)

        self._log_copy_confirm = QLabel("")
        self._log_copy_confirm.setStyleSheet(
            "color: #1a7f37; font-size: 11px; font-weight: bold;"
        )
        diag_row.addWidget(self._log_copy_confirm)
        diag_row.addStretch()
        diag_layout.addLayout(diag_row)

        diag_note = QLabel(
            "When reporting a problem, run a scan and share the log file above. "
            "It contains file counts, server responses, and similarity statistics — "
            "nothing from your music tags or personal data."
        )
        diag_note.setStyleSheet("color: #666; font-size: 11px;")
        diag_note.setWordWrap(True)
        diag_layout.addWidget(diag_note)

        root.addWidget(diag_box)
        root.addStretch()

    # ── OptionsPage interface ──────────────────────────────────────────────

    def load(self) -> None:
        cfg = self.api.plugin_config

        # CLAP thresholds
        self.certain_row.set_value(cfg_get(cfg, "certain_threshold", 95))
        self.likely_row.set_value(cfg_get(cfg, "likely_threshold",  85))
        self.unsure_row.set_value(cfg_get(cfg, "unsure_threshold",  70))

        # AcoustID / Chromaprint panel
        self._cp_panel.certain_row.set_value(cfg_get(cfg, "cp_certain_threshold", 95))
        self._cp_panel.likely_row.set_value (cfg_get(cfg, "cp_likely_threshold",  85))
        self._cp_panel.unsure_row.set_value (cfg_get(cfg, "cp_unsure_threshold",  75))
        self._cp_panel.set_alignment(cfg_get(cfg, "cp_alignment",   "standard"))
        self._cp_panel.use_gpu_check.setChecked(bool(cfg_get(cfg, "cp_use_gpu", True)))

        # CLAP inference mode
        mode = cfg_get(cfg, "inference_mode", "remote")
        if mode == "local":
            self._local_radio.setChecked(True)
            self._stack.setCurrentIndex(1)
        else:
            self._remote_radio.setChecked(True)
            self._stack.setCurrentIndex(0)

        # Remote panel
        self._remote_panel.host_edit.setText(cfg_get(cfg, "server_host", "10.0.0.69"))
        self._remote_panel.port_spin.setValue(cfg_get(cfg, "server_port", 8765))
        self._remote_panel.api_key_edit.setText(cfg_get(cfg, "api_key", ""))
        self._remote_panel.win_root_edit.setText(
            cfg_get(cfg, "win_music_root", r"Z:\Multimedia\Audio\Music")
        )
        self._remote_panel.lxc_root_edit.setText(
            cfg_get(cfg, "lxc_music_root", "/mnt/music")
        )
        self._remote_panel.test_status.setText("")

        # Local panel
        self._local_panel.set_gpu_index(cfg_get(cfg, "local_gpu_index", 0))
        self._local_panel.cache_edit.setText(cfg_get(cfg, "model_cache_dir", ""))

    def save(self) -> None:
        cfg = self.api.plugin_config

        # CLAP thresholds
        cfg["certain_threshold"] = self.certain_row.value()
        cfg["likely_threshold"]  = self.likely_row.value()
        cfg["unsure_threshold"]  = self.unsure_row.value()

        # AcoustID thresholds + engine settings
        cfg["cp_certain_threshold"] = self._cp_panel.certain_row.value()
        cfg["cp_likely_threshold"]  = self._cp_panel.likely_row.value()
        cfg["cp_unsure_threshold"]  = self._cp_panel.unsure_row.value()
        cfg["cp_alignment"]         = self._cp_panel.selected_alignment()
        cfg["cp_use_gpu"]           = self._cp_panel.use_gpu_check.isChecked()

        # CLAP inference mode
        cfg["inference_mode"] = "local" if self._local_radio.isChecked() else "remote"

        # Remote
        cfg["server_host"]    = self._remote_panel.host_edit.text().strip()
        cfg["server_port"]    = self._remote_panel.port_spin.value()
        cfg["api_key"]        = self._remote_panel.api_key_edit.text()
        cfg["win_music_root"] = self._remote_panel.win_root_edit.text().strip()
        cfg["lxc_music_root"] = self._remote_panel.lxc_root_edit.text().strip()

        # Local
        cfg["local_gpu_index"] = self._local_panel.selected_gpu_index()
        cfg["model_cache_dir"] = self._local_panel.cache_edit.text().strip()

    # ── Mode switching ─────────────────────────────────────────────────────

    def _on_mode_changed(self, btn_id: int, checked: bool) -> None:
        if checked:
            self._stack.setCurrentIndex(btn_id)

    # ── Diagnostics helpers ────────────────────────────────────────────────

    def _open_log_folder(self) -> None:
        import os
        import subprocess
        import sys
        folder = os.path.dirname(self._log_path)
        try:
            if sys.platform == "win32":
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:  # noqa: BLE001
            self._log_copy_confirm.setText(f"Could not open: {exc}")
            self._log_copy_confirm.setStyleSheet(
                "color: #cf222e; font-size: 11px; font-weight: bold;"
            )

    def _copy_log_path(self) -> None:
        QApplication.clipboard().setText(self._log_path)
        self._log_copy_confirm.setText("✓ Copied!")
        self._log_copy_confirm.setStyleSheet(
            "color: #1a7f37; font-size: 11px; font-weight: bold;"
        )
        QTimer.singleShot(2500, lambda: self._log_copy_confirm.setText(""))
