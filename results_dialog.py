# Music Duplicate Finder — results_dialog.py
# The main results window shown after a successful scan.
# Groups are displayed as cards, files sorted by quality descending.

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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

        rows = sorted(fq.tags_dict.items())
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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        # ── Checkbox ──────────────────────────────────────────────────────
        self._checkbox = QCheckBox()
        self._checkbox.setToolTip("Select for batch move / delete")
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
        path_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
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
        layout.addLayout(info_col, stretch=1)

        # ── Quality score ──────────────────────────────────────────────────
        q_col = QVBoxLayout()
        q_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_lbl = QLabel(f"{fq.score:.1f}")
        score_lbl.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #333;"
        )
        score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        q_col.addWidget(score_lbl)
        q_note = QLabel("quality")
        q_note.setStyleSheet("color: #555; font-size: 9px;")
        q_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        q_col.addWidget(q_note)
        if fq.is_live:
            pen_lbl = QLabel("−50% live")
            pen_lbl.setStyleSheet("color: #6f42c1; font-size: 9px;")
            pen_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            q_col.addWidget(pen_lbl)
        q_w = QWidget()
        q_w.setLayout(q_col)
        q_w.setFixedWidth(76)
        layout.addWidget(q_w)

        # ── Action buttons ─────────────────────────────────────────────────
        btn_col = QVBoxLayout()
        btn_col.setSpacing(3)

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
        layout.addLayout(btn_col)

        if is_best:
            pal = self.palette()
            pal.setColor(QPalette.ColorRole.Window,     QColor("#e8f8ed"))
            pal.setColor(QPalette.ColorRole.WindowText, QColor("#111111"))
            self.setPalette(pal)
            self.setAutoFillBackground(True)

    @staticmethod
    def _wrap_path(path: str) -> str:
        return path.replace("/", "/\u200b").replace("\\", "\\\u200b")

    # ── Public interface ───────────────────────────────────────────────────

    def is_checked(self) -> bool:
        return self._checkbox.isChecked()

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

    def file_rows(self) -> list[FileRow]:
        return self._file_rows


# ══════════════════════════════════════════════════════════════════════════
# Main results dialog
# ══════════════════════════════════════════════════════════════════════════

class ResultsDialog(QDialog):

    def __init__(self, result: "ScanResult", parent=None):
        super().__init__(parent)
        self._result  = result
        self._groups  = result.groups
        self._players: list[MiniPlayer] = []
        self._all_file_rows: list[FileRow] = []

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
        self._filter_combo.addItems(["All", "Certain", "Likely", "Unsure"])
        self._filter_combo.currentTextChanged.connect(self._apply_filter)
        summary_row.addWidget(self._filter_combo)

        root.addLayout(summary_row)

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
            self._all_file_rows.extend(card.file_rows())
            self._cards_layout.addWidget(card)

    def _apply_filter(self, text: str) -> None:
        text = text.lower()
        for confidence, card in self._card_widgets:
            visible = (text == "all" or text == confidence)
            card.setVisible(visible)

    # ── Player helpers ─────────────────────────────────────────────────────

    def _deactivate_others(self) -> None:
        """When a new source is loaded, visually deactivate all mini-players."""
        # We can't know which one loaded, so deactivate all — the active one
        # will re-activate itself via playbackStateChanged.
        for mp in self._players:
            mp.deactivate()

    # ── Batch operations ───────────────────────────────────────────────────

    def _checked_rows(self) -> list[FileRow]:
        return [r for r in self._all_file_rows if r.is_checked() and r.isEnabled()]

    def _batch_delete(self) -> None:
        targets = self._checked_rows()
        if not targets:
            QMessageBox.information(self, "Nothing selected", "Check at least one file first.")
            return
        paths_text = "\n".join(r.path for r in targets)
        reply = QMessageBox.warning(
            self,
            "Delete files",
            f"Permanently delete {len(targets)} file(s):\n\n{paths_text}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
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
