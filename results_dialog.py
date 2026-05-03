# Music Duplicate Finder — results_dialog.py
# The main results window shown after a successful scan.
# Groups are displayed as cards, files sorted by quality descending.

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .results_io import save_result

if TYPE_CHECKING:
    from .quality import FileQuality
    from .scan_worker import DuplicateGroup, ScanResult


# ── Palette helpers ────────────────────────────────────────────────────────

CONFIDENCE_STYLES = {
    #                icon   border/text  card bg    header bg
    "certain": ("●", "#1a7f37", "#d4edda", "#c3e6cb"),
    "likely":  ("◑", "#9a6700", "#fff3cd", "#ffeeba"),
    "unsure":  ("○", "#cf222e", "#f8d7da", "#f5c6cb"),
}
LIVE_STYLE = "background:#6f42c1; color:#fff; border-radius:3px; padding:2px 6px; font-size:10px; min-width:44px; text-align:left;"
BEST_STYLE = "background:#0d6efd; color:#fff; border-radius:3px; padding:2px 6px; font-size:10px; min-width:44px; text-align:left;"


# ══════════════════════════════════════════════════════════════════════════
# Tags viewer dialog
# ══════════════════════════════════════════════════════════════════════════

class TagsDialog(QDialog):
    def __init__(self, fq: "FileQuality", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Tags — {os.path.basename(fq.path)}")
        self.resize(560, 420)

        layout = QVBoxLayout(self)

        path_label = QLabel(f"<b>Path:</b> {fq.path}")
        path_label.setWordWrap(True)
        path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(path_label)

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Tag", "Value"])
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)

        tags = fq.tags_dict
        if not tags:
            from .quality import analyse_file
            reread = analyse_file(fq.path)
            if reread:
                tags = reread.tags_dict

        rows = sorted(tags.items())
        table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(str(k)))
            table.setItem(i, 1, QTableWidgetItem(str(v)))

        layout.addWidget(table)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


# ══════════════════════════════════════════════════════════════════════════
# Inline mini audio player
# ══════════════════════════════════════════════════════════════════════════

class MiniPlayer(QWidget):
    """
    Compact play/pause + seek slider.  Lives inside each file row.
    One shared QMediaPlayer per ResultsDialog is reused; clicking play
    on a new file stops the previous one.
    """

    def __init__(self, file_path: str, player: QMediaPlayer, parent=None):
        super().__init__(parent)
        self._path   = file_path
        self._player = player
        self._active = False  # True when THIS file is loaded into the player

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(26, 26)
        self._play_btn.setToolTip("Play / Pause")
        self._play_btn.clicked.connect(self._toggle)
        layout.addWidget(self._play_btn)

        self._seek = QSlider(Qt.Orientation.Horizontal)
        self._seek.setRange(0, 1000)
        self._seek.setFixedWidth(90)
        self._seek.setEnabled(False)
        self._seek.sliderMoved.connect(self._seek_to)
        layout.addWidget(self._seek)

        self._time_label = QLabel("0:00")
        self._time_label.setFixedWidth(38)
        self._time_label.setStyleSheet("font-size: 10px; color: #666;")
        layout.addWidget(self._time_label)

        # Connect player signals
        self._player.positionChanged.connect(self._on_position)
        self._player.playbackStateChanged.connect(self._on_state_changed)

    def deactivate(self) -> None:
        """Called when another file starts playing."""
        self._active = False
        self._play_btn.setText("▶")
        self._seek.setEnabled(False)
        self._seek.setValue(0)
        self._time_label.setText("0:00")

    def _toggle(self) -> None:
        if not self._active:
            # Stop any currently playing file
            self._player.stop()
            self._player.setSource(QUrl.fromLocalFile(self._path))
            self._player.play()
            self._active = True
            self._seek.setEnabled(True)
            self._play_btn.setText("⏸")
        else:
            state = self._player.playbackState()
            if state == QMediaPlayer.PlaybackState.PlayingState:
                self._player.pause()
                self._play_btn.setText("▶")
            else:
                self._player.play()
                self._play_btn.setText("⏸")

    def _seek_to(self, value: int) -> None:
        if self._active:
            duration = self._player.duration()
            if duration > 0:
                self._player.setPosition(int(duration * value / 1000))

    def _on_position(self, pos_ms: int) -> None:
        if not self._active:
            return
        duration = self._player.duration()
        if duration > 0:
            self._seek.setValue(int(pos_ms * 1000 / duration))
        secs = pos_ms // 1000
        m, s = divmod(secs, 60)
        self._time_label.setText(f"{m}:{s:02d}")

    def _on_state_changed(self, state) -> None:
        if not self._active:
            return
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self._play_btn.setText("▶")
            self._seek.setValue(0)
            self._time_label.setText("0:00")


# ══════════════════════════════════════════════════════════════════════════
# File row widget
# ══════════════════════════════════════════════════════════════════════════

class FileRow(QWidget):
    """A single file entry inside a group card."""

    deleted = pyqtSignal()

    def __init__(
        self,
        fq: "FileQuality",
        is_best: bool,
        player: QMediaPlayer,
        all_players: list,   # list of MiniPlayer instances for deactivation
        parent=None,
    ):
        super().__init__(parent)
        self._fq = fq
        self._all_players = all_players
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Root: optional anomaly banner above the content row
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._anomaly_lbl = QLabel()
        self._anomaly_lbl.setVisible(False)
        self._in_anomalous_group = False
        root.addWidget(self._anomaly_lbl)

        self._content = QWidget()
        layout = QHBoxLayout(self._content)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)
        root.addWidget(self._content)

        # ── Checkbox ──────────────────────────────────────────────────────
        self._checkbox = QCheckBox()
        self._checkbox.setToolTip("Select for batch move / delete")
        self._checkbox.setStyleSheet(
            "QCheckBox::indicator { width: 20px; height: 20px; }"
            "QCheckBox { padding: 20px; }"
        )
        layout.addWidget(self._checkbox)

        # ── Badges ────────────────────────────────────────────────────────
        badge_col = QVBoxLayout()
        badge_col.setSpacing(2)
        if is_best:
            best_lbl = QLabel("BEST")
            best_lbl.setStyleSheet(BEST_STYLE)
            best_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge_col.addWidget(best_lbl)
        if fq.is_live:
            live_lbl = QLabel("LIVE")
            live_lbl.setStyleSheet(LIVE_STYLE)
            live_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge_col.addWidget(live_lbl)
        badge_col.addStretch()
        badge_w = QWidget()
        badge_w.setLayout(badge_col)
        badge_w.setFixedWidth(56)
        layout.addWidget(badge_w)

        # ── Mini player ────────────────────────────────────────────────────
        self._mini = MiniPlayer(fq.path, player)
        all_players.append(self._mini)
        layout.addWidget(self._mini)

        # ── Quality score + action buttons ────────────────────────────────────
        btn_col = QVBoxLayout()
        btn_col.setSpacing(3)

        score_lbl = QLabel(f"{fq.score:.1f}")
        score_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #333;")
        score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_col.addWidget(score_lbl)
        q_note = QLabel("quality")
        q_note.setStyleSheet("color: #555; font-size: 9px;")
        q_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_col.addWidget(q_note)
        if fq.is_live:
            pen_lbl = QLabel("−50% live")
            pen_lbl.setStyleSheet("color: #6f42c1; font-size: 9px;")
            pen_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn_col.addWidget(pen_lbl)
        btn_col.addSpacing(4)

        tags_btn = QPushButton("🏷 Tags")
        tags_btn.setFixedWidth(72)
        tags_btn.setToolTip("View all metadata tags")
        tags_btn.clicked.connect(self._show_tags)
        btn_col.addWidget(tags_btn)

        folder_btn = QPushButton("📂 Open")
        folder_btn.setFixedWidth(72)
        folder_btn.setToolTip("Reveal file in Explorer")
        folder_btn.clicked.connect(self._open_folder)
        btn_col.addWidget(folder_btn)

        move_btn = QPushButton("📦 Move")
        move_btn.setFixedWidth(72)
        move_btn.setToolTip("Move file to another folder")
        move_btn.clicked.connect(self._move_file)
        btn_col.addWidget(move_btn)

        del_btn = QPushButton("🗑 Delete")
        del_btn.setFixedWidth(72)
        del_btn.setToolTip("Permanently delete this file")
        del_btn.setStyleSheet("color: #cf222e;")
        del_btn.clicked.connect(self._delete_file)
        btn_col.addWidget(del_btn)

        btn_col.addStretch()
        btn_w = QWidget()
        btn_w.setLayout(btn_col)
        btn_w.setFixedWidth(82)
        layout.addWidget(btn_w)

        # ── File info ──────────────────────────────────────────────────────
        info_col = QVBoxLayout()
        info_col.setSpacing(1)

        filename = os.path.basename(fq.path)
        name_lbl = QLabel(f"<b>{filename}</b>")
        name_lbl.setStyleSheet("color: #111;")
        name_lbl.setToolTip(fq.path)
        name_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        info_col.addWidget(name_lbl)

        path_lbl = QLabel(self._wrap_path(os.path.dirname(fq.path)))
        path_lbl.setStyleSheet("color: #444; font-size: 10px; font-family: monospace;")
        path_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        path_lbl.setWordWrap(True)
        path_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        info_col.addWidget(path_lbl)

        meta_parts = []
        if fq.title and fq.title != filename:
            meta_parts.append(fq.title)
        if fq.artist:
            meta_parts.append(fq.artist)
        if fq.album:
            meta_parts.append(fq.album)
        if meta_parts:
            meta_lbl = QLabel("  ".join(meta_parts))
            meta_lbl.setStyleSheet("color: #333; font-size: 11px;")
            info_col.addWidget(meta_lbl)

        detail_parts = []
        if fq.format_name:
            detail_parts.append(fq.format_name)
        if fq.bitrate_kbps:
            detail_parts.append(f"{fq.bitrate_kbps:.0f} kbps")
        if fq.sample_rate_hz:
            detail_parts.append(f"{fq.sample_rate_hz // 1000} kHz")
        detail_parts.append(f"{fq.file_size_mb:.1f} MB")
        if fq.duration_sec:
            detail_parts.append(fq.duration_str)

        detail_lbl = QLabel("  ·  ".join(detail_parts))
        detail_lbl.setStyleSheet("color: #555; font-size: 10px;")
        info_col.addWidget(detail_lbl)

        info_col.addStretch()

        info_w = QWidget()
        info_w.setLayout(info_col)
        info_w.setMaximumWidth(620)
        info_w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(info_w, stretch=1)

        if is_best:
            pal = self._content.palette()
            pal.setColor(QPalette.ColorRole.Window,     QColor("#e8f8ed"))
            pal.setColor(QPalette.ColorRole.WindowText, QColor("#111111"))
            self._content.setPalette(pal)
            self._content.setAutoFillBackground(True)

    @staticmethod
    def _wrap_path(path: str) -> str:
        return path.replace("/", "/\u200b").replace("\\", "\\\u200b")

    # ── Public interface ───────────────────────────────────────────────────

    def is_checked(self) -> bool:
        return self._checkbox.isChecked()

    def set_checked(self, value: bool) -> None:
        self._checkbox.setChecked(value)

    def is_size_anomaly(self) -> bool:
        return self._in_anomalous_group

    def flag_anomalous_group(self) -> None:
        self._in_anomalous_group = True

    def mark_missing(self) -> None:
        self.setEnabled(False)
        self.setStyleSheet("QWidget { color: #aaa; text-decoration: line-through; }")

    def mark_size_anomaly(self) -> None:
        self._in_anomalous_group = True
        self._anomaly_lbl.setText(
            "⚠  This group has size anomalies — "
            "verify before deleting"
        )
        self._anomaly_lbl.setStyleSheet(
            "background: #e36209; color: #fff; font-weight: bold; "
            "font-size: 10px; padding: 3px 10px;"
        )
        self._anomaly_lbl.setVisible(True)
        pal = self._content.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#fff3cd"))
        self._content.setPalette(pal)
        self._content.setAutoFillBackground(True)

    @property
    def path(self) -> str:
        return self._fq.path

    @path.setter
    def path(self, value: str) -> None:
        self._fq.path = value

    # ── Actions ────────────────────────────────────────────────────────────

    def _show_tags(self) -> None:
        dlg = TagsDialog(self._fq, self)
        dlg.exec()

    def _open_folder(self) -> None:
        path = self._fq.path
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])

    def _move_file(self) -> None:
        dest_dir = QFileDialog.getExistingDirectory(
            self, "Move file to…", os.path.dirname(self._fq.path)
        )
        if not dest_dir:
            return
        src  = self._fq.path
        dest = os.path.join(dest_dir, os.path.basename(src))
        if os.path.exists(dest):
            reply = QMessageBox.question(
                self,
                "File exists",
                f"A file named '{os.path.basename(src)}' already exists in the "
                f"destination.\n\nOverwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        try:
            shutil.move(src, dest)
            QMessageBox.information(self, "Moved", f"File moved to:\n{dest}")
            self._fq.path = dest
            self.setEnabled(False)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Could not move file:\n{exc}")

    def _delete_file(self) -> None:
        reply = QMessageBox.warning(
            self,
            "Delete file",
            f"Permanently delete:\n\n{self._fq.path}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            os.remove(self._fq.path)
            self.setEnabled(False)
            self.setStyleSheet("QWidget { color: #aaa; text-decoration: line-through; }")
            # Stop player if this file was playing
            for mp in self._all_players:
                mp.deactivate()
            self.deleted.emit()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Could not delete file:\n{exc}")


# ══════════════════════════════════════════════════════════════════════════
# Group card widget
# ══════════════════════════════════════════════════════════════════════════

class GroupCard(QFrame):
    """Displays one duplicate group with all its file rows."""

    def __init__(
        self,
        group: "DuplicateGroup",
        index: int,
        player: QMediaPlayer,
        all_players: list,
        parent=None,
    ):
        super().__init__(parent)
        self._group = group
        icon, color, bg, header_bg = CONFIDENCE_STYLES.get(
            group.confidence, ("○", "#555", "#f5f5f5", "#e8e8e8")
        )
        best = group.best

        title = f"  {icon}  GROUP {index}   —   {group.confidence.upper()}   " \
                f"({group.similarity * 100:.1f}% similarity)"
        if best and best.title:
            title += f"   |   \"{best.title}\""
            if best.artist:
                title += f"  — {best.artist}"

        self.setObjectName("GroupCard")
        self.setStyleSheet(
            f"QFrame#GroupCard {{ background: {bg}; border: 1px solid {color}50; "
            f"border-radius: 6px; }}"
        )

        outer = QVBoxLayout(self)
        outer.setSpacing(0)
        outer.setContentsMargins(0, 0, 0, 0)

        # Title bar — visibly inside the card border
        header = QLabel(title)
        header.setStyleSheet(
            f"QLabel {{ color: #111; font-weight: bold; font-size: 12px; "
            f"background: {header_bg}; padding: 5px 10px; "
            f"border-bottom: 1px solid {color}; }}"
        )
        header.setWordWrap(True)
        header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        outer.addWidget(header)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)

        self._file_rows: list[FileRow] = []
        for i, fq in enumerate(group.files):
            row = FileRow(fq, is_best=(i == 0), player=player,
                          all_players=all_players)
            self._file_rows.append(row)
            layout.addWidget(row)
            if i < len(group.files) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet("color: #ddd;")
                layout.addWidget(line)

        outer.addWidget(body)

        # Pre-select all losers; leave unchecked and flag winner if anomalous.
        self._apply_auto_check()

    def file_rows(self) -> list[FileRow]:
        return self._file_rows

    def has_active_duplicates(self) -> bool:
        """True if 2+ file rows are still enabled (not yet deleted/moved)."""
        return sum(1 for row in self._file_rows if row.isEnabled()) >= 2

    def has_size_anomaly(self) -> bool:
        return any(row.is_size_anomaly() for row in self._file_rows)

    def has_checked_files(self) -> bool:
        return any(row.is_checked() for row in self._file_rows)

    def apply_auto_check(self) -> None:
        for row in self._file_rows:
            row.set_checked(False)
        self._apply_auto_check()

    def _apply_auto_check(self) -> None:
        if len(self._file_rows) < 2:
            return
        winner_size = self._group.files[0].file_size_bytes
        other_sizes = [f.file_size_bytes for f in self._group.files[1:]]
        mean_other  = sum(other_sizes) / len(other_sizes)
        anomaly = (
            (mean_other > 0 and winner_size > 1.7 * mean_other)
            or winner_size > int(7.5 * 1024 * 1024)
        )
        if anomaly:
            self._file_rows[0].mark_size_anomaly()
            for row in self._file_rows[1:]:
                row.flag_anomalous_group()
        else:
            for row in self._file_rows[1:]:
                row.set_checked(True)


# ══════════════════════════════════════════════════════════════════════════
# Main results dialog
# ══════════════════════════════════════════════════════════════════════════

class ResultsDialog(QDialog):

    def __init__(self, result: "ScanResult", parent=None, *, loaded_from_file: bool = False):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self._result  = result
        self._groups  = result.groups
        self._players: list[MiniPlayer] = []
        self._all_file_rows: list[FileRow] = []
        self._loaded_from_file = loaded_from_file
        self._hide_resolved = False

        self.setWindowTitle("Music Duplicate Finder — Results")
        self.setMinimumSize(820, 640)
        self.resize(1000, 720)

        # ── Shared media player ────────────────────────────────────────────
        self._audio_out = QAudioOutput()
        self._audio_out.setVolume(0.8)
        self._player = QMediaPlayer()
        self._player.setAudioOutput(self._audio_out)
        # When a new source loads, deactivate all others
        self._player.sourceChanged.connect(self._deactivate_others)

        # ── Root layout ────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Summary bar ────────────────────────────────────────────────────
        summary_row = QHBoxLayout()
        n_certain = sum(1 for g in self._groups if g.confidence == "certain")
        n_likely  = sum(1 for g in self._groups if g.confidence == "likely")
        n_unsure  = sum(1 for g in self._groups if g.confidence == "unsure")
        n_files   = sum(len(g.files) for g in self._groups)

        summary = QLabel(
            f"<b>{len(self._groups)} duplicate groups</b> found across "
            f"<b>{n_files} files</b>  "
            f"(scanned {result.total_files_scanned} files in "
            f"{result.elapsed_seconds:.1f}s)  —  "
            f"<span style='color:#1a7f37'>● {n_certain} certain</span>  "
            f"<span style='color:#9a6700'>◑ {n_likely} likely</span>  "
            f"<span style='color:#cf222e'>○ {n_unsure} unsure</span>"
        )
        summary.setTextFormat(Qt.TextFormat.RichText)
        summary_row.addWidget(summary)
        summary_row.addStretch()

        # Filter combo
        summary_row.addWidget(QLabel("Show:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "Certain", "Likely", "Unsure", "Size Anomalies", "Search Results"])
        self._filter_combo.currentTextChanged.connect(self._apply_filter)
        summary_row.addWidget(self._filter_combo)

        self._resolved_btn = QPushButton("Hide Resolved Groups")
        self._resolved_btn.setToolTip(
            "Hide groups where all duplicates have been deleted or moved"
        )
        self._resolved_btn.clicked.connect(self._toggle_resolved_filter)
        summary_row.addWidget(self._resolved_btn)

        root.addLayout(summary_row)

        # ── Path search / select bar ───────────────────────────────────────
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Select by path:"))
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Type a phrase to select matching files…")
        self._search_box.setMinimumWidth(300)
        self._search_box.returnPressed.connect(self._select_by_search)
        search_row.addWidget(self._search_box)
        select_btn = QPushButton("Select Matching")
        select_btn.setToolTip("Uncheck all, then check files whose path contains the phrase above")
        select_btn.clicked.connect(self._select_by_search)
        search_row.addWidget(select_btn)
        search_row.addStretch()
        auto_check_btn = QPushButton("✔ Auto-Check")
        auto_check_btn.setToolTip("Re-apply automatic pre-selection of losing files")
        auto_check_btn.clicked.connect(self._auto_check_all)
        search_row.addWidget(auto_check_btn)
        uncheck_btn = QPushButton("✖ Uncheck All")
        uncheck_btn.setToolTip("Uncheck all selected files")
        uncheck_btn.clicked.connect(self._uncheck_all)
        search_row.addWidget(uncheck_btn)
        root.addLayout(search_row)

        # ── Scroll area with group cards ───────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._cards_container = QWidget()
        self._cards_layout    = QVBoxLayout(self._cards_container)
        self._cards_layout.setSpacing(10)
        self._cards_layout.setContentsMargins(4, 4, 4, 4)
        self._card_widgets: list[tuple[str, QWidget]] = []  # (confidence, widget)

        self._build_cards()
        self._cards_layout.addStretch()
        scroll.setWidget(self._cards_container)
        root.addWidget(scroll, stretch=1)

        if self._loaded_from_file:
            QTimer.singleShot(0, self._prompt_check_deletions)

        # ── Bottom bar ─────────────────────────────────────────────────────
        bottom = QHBoxLayout()

        save_btn = QPushButton("💾  Save Results…")
        save_btn.setToolTip("Save these results to a file so you can reload them later without rescanning")
        save_btn.clicked.connect(self._save_results)
        bottom.addWidget(save_btn)
        bottom.addStretch()

        batch_move_btn = QPushButton("📦 Move Selected")
        batch_move_btn.setToolTip("Move all checked files to a chosen folder")
        batch_move_btn.clicked.connect(self._batch_move)
        bottom.addWidget(batch_move_btn)

        batch_del_btn = QPushButton("🗑 Delete Selected")
        batch_del_btn.setStyleSheet("color: #cf222e;")
        batch_del_btn.setToolTip("Permanently delete all checked files")
        batch_del_btn.clicked.connect(self._batch_delete)
        bottom.addWidget(batch_del_btn)
        bottom.addStretch()

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self._close)
        bottom.addWidget(btns)

        root.addLayout(bottom)

    # ── Card building ──────────────────────────────────────────────────────

    def _build_cards(self) -> None:
        for i, group in enumerate(self._groups, start=1):
            card = GroupCard(
                group, i,
                player     = self._player,
                all_players = self._players,
            )
            self._card_widgets.append((group.confidence, card))
            for row in card.file_rows():
                self._all_file_rows.append(row)
                row.deleted.connect(lambda: self._apply_filter(self._filter_combo.currentText()))
            self._cards_layout.addWidget(card)
            if i % 10 == 0:
                QApplication.processEvents()

    def _prompt_check_deletions(self) -> None:
        reply = QMessageBox.question(
            self,
            "Check for deleted files",
            "Would you like to check whether all files in these results still exist on disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        missing = [row for row in self._all_file_rows if not os.path.exists(row.path)]
        for row in missing:
            row.mark_missing()
        if missing:
            QMessageBox.information(
                self,
                "Missing files",
                f"{len(missing)} file(s) no longer exist on disk and have been marked.",
            )
        else:
            QMessageBox.information(
                self,
                "All files present",
                "All files in these results still exist on disk.",
            )

    def _apply_filter(self, text: str) -> None:
        text = text.lower()
        for confidence, card in self._card_widgets:
            if text == "all":
                visible = True
            elif text == "size anomalies":
                visible = card.has_size_anomaly()
            elif text == "search results":
                visible = card.has_checked_files()
            else:
                visible = (text == confidence)
            if visible and self._hide_resolved:
                visible = card.has_active_duplicates()
            card.setVisible(visible)

    def _toggle_resolved_filter(self) -> None:
        self._hide_resolved = not self._hide_resolved
        self._resolved_btn.setText(
            "Show Resolved Groups" if self._hide_resolved else "Hide Resolved Groups"
        )
        self._apply_filter(self._filter_combo.currentText())

    # ── Player helpers ─────────────────────────────────────────────────────

    def _deactivate_others(self) -> None:
        """When a new source is loaded, visually deactivate all mini-players."""
        # We can't know which one loaded, so deactivate all — the active one
        # will re-activate itself via playbackStateChanged.
        for mp in self._players:
            mp.deactivate()

    # ── Check helpers ──────────────────────────────────────────────────────

    def _select_by_search(self) -> None:
        phrase = self._search_box.text().strip()
        if not phrase:
            return
        phrase_lower = phrase.lower()
        for row in self._all_file_rows:
            row.set_checked(False)
        matched = 0
        anomaly_skipped = 0
        for row in self._all_file_rows:
            if phrase_lower in row.path.lower():
                if row.is_size_anomaly():
                    anomaly_skipped += 1
                else:
                    row.set_checked(True)
                    matched += 1

        anomaly_note = (
            f"\n\n⚠ {anomaly_skipped} matching file(s) were NOT selected — "
            f"they are flagged as size anomalies and require manual review."
            if anomaly_skipped else ""
        )
        if self._filter_combo.currentText() == "Search Results":
            self._apply_filter("Search Results")

        if matched == 0 and anomaly_skipped == 0:
            QMessageBox.information(
                self, "No matches", f"No files found whose path contains:\n\n{phrase}"
            )
        elif matched == 0:
            QMessageBox.information(
                self, "No files selected",
                f"No files selected — all {anomaly_skipped} match(es) are size anomalies "
                f"and were skipped.{anomaly_note}"
            )
        else:
            QMessageBox.information(
                self, "Files selected",
                f"{matched} file(s) selected whose path contains:\n\n{phrase}{anomaly_note}"
            )

    def _uncheck_all(self) -> None:
        for row in self._all_file_rows:
            row.set_checked(False)

    def _auto_check_all(self) -> None:
        for _, card in self._card_widgets:
            card.apply_auto_check()

    # ── Batch operations ───────────────────────────────────────────────────

    def _checked_rows(self) -> list[FileRow]:
        return [r for r in self._all_file_rows if r.is_checked() and r.isEnabled()]

    def _batch_delete(self) -> None:
        targets = self._checked_rows()
        if not targets:
            QMessageBox.information(self, "Nothing selected", "Check at least one file first.")
            return

        # ── Confirmation dialog with optional file list ────────────────────
        confirm = QDialog(self)
        confirm.setWindowTitle("Delete files")
        confirm_layout = QVBoxLayout(confirm)
        confirm_layout.addWidget(QLabel(
            f"Permanently delete {len(targets)} file(s)?\n\nThis cannot be undone."
        ))
        confirm_btns = QHBoxLayout()

        show_btn = QPushButton("Show Files…")
        def _show_files() -> None:
            detail = QDialog(confirm)
            detail.setWindowTitle(f"Files to delete ({len(targets)})")
            dl = QVBoxLayout(detail)
            te = QTextEdit()
            te.setReadOnly(True)
            te.setPlainText("\n".join(r.path for r in targets))
            dl.addWidget(te)
            close_btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            close_btns.rejected.connect(detail.reject)
            dl.addWidget(close_btns)
            detail.resize(640, 360)
            detail.exec()
        show_btn.clicked.connect(_show_files)
        confirm_btns.addWidget(show_btn)
        confirm_btns.addStretch()

        del_btn = QPushButton("Delete")
        del_btn.setStyleSheet("color: #cf222e;")
        del_btn.clicked.connect(confirm.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(confirm.reject)
        confirm_btns.addWidget(del_btn)
        confirm_btns.addWidget(cancel_btn)
        confirm_layout.addLayout(confirm_btns)
        confirm.resize(380, 110)

        if confirm.exec() != QDialog.DialogCode.Accepted:
            return

        errors = []
        for row in targets:
            try:
                os.remove(row.path)
                row.setEnabled(False)
                row.setStyleSheet("QWidget { color: #aaa; text-decoration: line-through; }")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{row.path}: {exc}")
        for mp in self._players:
            mp.deactivate()
        self._apply_filter(self._filter_combo.currentText())
        if errors:
            QMessageBox.critical(self, "Delete errors", "\n".join(errors))

    def _batch_move(self) -> None:
        targets = self._checked_rows()
        if not targets:
            QMessageBox.information(self, "Nothing selected", "Check at least one file first.")
            return
        dest_dir = QFileDialog.getExistingDirectory(
            self, f"Move {len(targets)} file(s) to…", ""
        )
        if not dest_dir:
            return
        errors = []
        moved = 0
        for row in targets:
            src  = row.path
            dest = os.path.join(dest_dir, os.path.basename(src))
            if os.path.exists(dest):
                reply = QMessageBox.question(
                    self,
                    "File exists",
                    f"'{os.path.basename(src)}' already exists in the destination.\nOverwrite?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    continue
            try:
                shutil.move(src, dest)
                row.path = dest
                row.setEnabled(False)
                moved += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{src}: {exc}")
        if errors:
            QMessageBox.critical(self, "Move errors", "\n".join(errors))
        elif moved:
            QMessageBox.information(self, "Moved", f"{moved} file(s) moved to:\n{dest_dir}")

    # ── Save ───────────────────────────────────────────────────────────────

    def _save_results(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Duplicate Results",
            "",
            "Duplicate results (*.mdupe);;All files (*)",
        )
        if not path:
            return
        if not path.endswith(".mdupe"):
            path += ".mdupe"
        try:
            save_result(self._result, path)
            QMessageBox.information(
                self, "Saved", f"Results saved to:\n{path}"
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Error", f"Could not save results:\n{exc}")

    # ── Close ──────────────────────────────────────────────────────────────

    def _close(self) -> None:
        self._player.stop()
        self.accept()

    def closeEvent(self, event) -> None:
        self._player.stop()
        super().closeEvent(event)
