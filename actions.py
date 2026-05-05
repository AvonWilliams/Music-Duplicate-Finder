# Music Duplicate Finder — actions.py
# V2.0: adds a second set of actions for the AcoustID / Chromaprint engine.
# The original actions (renamed here to the "SimilarSongs" family) still
# drive the CLAP engine; the new "Duplicate" family drives chromaprint.
#
# Both action families share the same scan-launcher plumbing. The engine
# choice is made by passing a different inference_mode to the ScanWorker.

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog
from picard.plugin3.api import BaseAction

from .config_util import cfg_get
from .file_collector import collect_files_with_fingerprints
from .missing_fingerprints_dialog import (
    CpuFallbackWarningDialog,
    MissingFingerprintsDialog,
)
from .progress_dialog import ProgressDialog
from .results_dialog import ResultsDialog
from .scan_worker import ScanResult, ScanWorker


# ══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════

def _extract_files_from_objs(objs) -> list[str]:
    """Walk Picard objects and return a flat de-duplicated list of file paths."""
    seen:  set[str] = set()
    paths: list[str] = []

    def _add_file(f) -> None:
        try:
            p = f.filename
        except AttributeError:
            return
        if p and p not in seen:
            seen.add(p)
            paths.append(p)

    for obj in (objs or []):
        if hasattr(obj, "filename") and not hasattr(obj, "tracks") and not hasattr(obj, "files"):
            _add_file(obj)
            continue
        if hasattr(obj, "files") and not hasattr(obj, "tracks"):
            for f in obj.files:
                _add_file(f)
            continue
        if hasattr(obj, "tracks"):
            for track in obj.tracks:
                for f in getattr(track, "files", []):
                    _add_file(f)
            for f in getattr(obj, "unmatched_files", []):
                _add_file(f)
            continue
        if hasattr(obj, "files"):
            for f in obj.files:
                _add_file(f)

    return paths


# ══════════════════════════════════════════════════════════════════════════
# CLAP (Similar Songs) launcher — V1.9 semantics preserved
# ══════════════════════════════════════════════════════════════════════════

def _start_clap_scan(
    api,
    window,
    restrict_paths,
    source_label: str,
) -> None:
    cfg = api.plugin_config

    mode               = cfg_get(cfg, "inference_mode",    "remote")
    certain_threshold  = cfg_get(cfg, "certain_threshold", 95)
    likely_threshold   = cfg_get(cfg, "likely_threshold",  85)
    unsure_threshold   = cfg_get(cfg, "unsure_threshold",  70)

    server_host        = cfg_get(cfg, "server_host",    "10.0.0.69")
    server_port        = cfg_get(cfg, "server_port",    8765)
    api_key            = cfg_get(cfg, "api_key",        "")
    win_root           = cfg_get(cfg, "win_music_root", "")
    lxc_root           = cfg_get(cfg, "lxc_music_root", "")

    local_gpu_index    = cfg_get(cfg, "local_gpu_index", 0)
    model_cache_dir    = cfg_get(cfg, "model_cache_dir", "")

    anomaly_multiplier = cfg_get(cfg, "anomaly_size_multiplier", 1.7)
    anomaly_size_mb    = cfg_get(cfg, "anomaly_size_mb", 7.5)

    if restrict_paths is not None and not restrict_paths:
        QMessageBox.information(
            window,
            "Nothing to Scan",
            f"The {source_label} you right-clicked doesn't contain any "
            "audio files to scan.",
        )
        return

    worker = ScanWorker(
        api                = api,
        inference_mode     = mode,
        certain_threshold  = certain_threshold,
        likely_threshold   = likely_threshold,
        unsure_threshold   = unsure_threshold,
        server_host        = server_host,
        server_port        = server_port,
        api_key            = api_key,
        win_root           = win_root,
        lxc_root           = lxc_root,
        local_gpu_index    = local_gpu_index,
        model_cache_dir    = model_cache_dir,
        restrict_paths     = restrict_paths,
    )

    def on_finished(result: ScanResult) -> None:
        if not result.groups:
            QMessageBox.information(
                window,
                "No Similar Songs Found",
                f"Scanned {result.total_files_scanned} files from "
                f"{source_label} in {result.elapsed_seconds:.1f}s "
                f"(engine: CLAP {result.mode}).\n\n"
                "No similar songs were detected at the current thresholds.\n\n"
                "Try lowering the Unsure threshold in "
                "Options → Plugins → Music Duplicate Finder.",
            )
            return
        loading = QProgressDialog(f"Building display for {len(result.groups)} groups…", "", 0, 0, window)
        loading.setWindowTitle("Music Duplicate Finder")
        loading.setWindowModality(Qt.WindowModality.ApplicationModal)
        loading.setMinimumDuration(0)
        loading.setMinimumWidth(530)
        loading.setCancelButton(None)
        loading.show()
        QApplication.processEvents()
        dlg = ResultsDialog(result, window, anomaly_multiplier=anomaly_multiplier, anomaly_size_mb=anomaly_size_mb)
        loading.close()
        dlg.exec()

    def on_error(message: str) -> None:
        if mode == "local":
            hint = (
                "\n\nEnsure torch, transformers, soundfile, and scipy are "
                "installed in Picard's Python environment, and that ffmpeg "
                "is on PATH."
            )
        else:
            hint = (
                f"\n\nEnsure the GPU inference server is running at "
                f"{server_host}:{server_port} and the API key matches."
            )
        QMessageBox.critical(
            window,
            "Scan Failed",
            f"The similar-songs scan encountered an error:\n\n{message}{hint}",
        )

    progress = ProgressDialog(
        parent      = window,
        worker      = worker,
        on_finished = on_finished,
        on_error    = on_error,
    )
    progress.exec()


# ══════════════════════════════════════════════════════════════════════════
# AcoustID (Duplicate) launcher
# ══════════════════════════════════════════════════════════════════════════

def _start_chromaprint_scan(
    api,
    window,
    restrict_paths,
    source_label: str,
) -> None:
    cfg = api.plugin_config

    # Chromaprint-specific thresholds (fall back to CLAP thresholds if the
    # user hasn't set them explicitly — same slider UI, separate values).
    certain_threshold  = cfg_get(cfg, "cp_certain_threshold",
                                 cfg_get(cfg, "certain_threshold", 95))
    likely_threshold   = cfg_get(cfg, "cp_likely_threshold",
                                 cfg_get(cfg, "likely_threshold",  80))
    unsure_threshold   = cfg_get(cfg, "cp_unsure_threshold",
                                 cfg_get(cfg, "unsure_threshold",  65))

    alignment          = cfg_get(cfg, "cp_alignment",  "standard")
    use_gpu            = bool(cfg_get(cfg, "cp_use_gpu", True))
    local_gpu_index    = cfg_get(cfg, "local_gpu_index", 0)

    anomaly_multiplier = cfg_get(cfg, "anomaly_size_multiplier", 1.7)
    anomaly_size_mb    = cfg_get(cfg, "anomaly_size_mb", 7.5)

    # Harvest fingerprints on the main thread (Picard's File.metadata is
    # not guaranteed thread-safe).
    all_paths, fp_map = collect_files_with_fingerprints(
        api,
        restrict_paths=restrict_paths,
    )

    if not all_paths:
        msg = (
            "No audio files are currently loaded in Picard."
            if restrict_paths is None
            else "The {0} doesn't contain any audio files.".format(source_label)
        )
        QMessageBox.information(window, "Nothing to Scan", msg)
        return

    with_fp = len(fp_map)
    missing = [p for p in all_paths if p not in fp_map]

    # ── Missing-fingerprint alert ─────────────────────────────────────────
    if missing:
        dlg = MissingFingerprintsDialog(
            total_files    = len(all_paths),
            with_fp_count  = with_fp,
            missing_paths  = missing,
            parent         = window,
        )
        result = dlg.exec()
        if result == MissingFingerprintsDialog.GENERATE_FINGERPRINTS:
            tagger = api.tagger
            file_objs = [tagger.files[p] for p in missing if p in tagger.files]
            if file_objs:
                tagger.generate_fingerprints(file_objs)
                QMessageBox.information(
                    window,
                    "Generating Fingerprints",
                    f"Generating AcoustID fingerprints for {len(file_objs)} file(s) "
                    f"in the background.\n\nRun Find Duplicates again once Picard "
                    f"finishes processing them."
                )
            return
        if result != dlg.DialogCode.Accepted:
            return
        if with_fp == 0:
            return

    # ── CPU-fallback warning ──────────────────────────────────────────────
    if use_gpu:
        try:
            from .chromaprint_engine import torch_available
            has_gpu = torch_available()
        except Exception:  # noqa: BLE001
            has_gpu = False
        if not has_gpu:
            dlg = CpuFallbackWarningDialog(
                library_size = with_fp,
                alignment    = alignment,
                parent       = window,
            )
            if dlg.exec() != dlg.DialogCode.Accepted:
                return
            use_gpu = False

    worker = ScanWorker(
        api                      = api,
        inference_mode           = "chromaprint",
        certain_threshold        = certain_threshold,
        likely_threshold         = likely_threshold,
        unsure_threshold         = unsure_threshold,
        local_gpu_index          = local_gpu_index,
        chromaprint_alignment    = alignment,
        chromaprint_use_gpu      = use_gpu,
        chromaprint_fingerprints = fp_map,
        restrict_paths           = restrict_paths,
    )

    def on_finished(result: ScanResult) -> None:
        if not result.groups:
            QMessageBox.information(
                window,
                "No Duplicates Found",
                "Scanned {0} files from {1} in {2:.1f}s "
                "(engine: AcoustID).\n\n"
                "No duplicates detected at the current thresholds.\n\n"
                "Try lowering the Unsure threshold in "
                "Options → Plugins → Music Duplicate Finder.".format(
                    result.total_files_scanned,
                    source_label,
                    result.elapsed_seconds,
                )
            )
            return
        loading = QProgressDialog(f"Building display for {len(result.groups)} groups…", "", 0, 0, window)
        loading.setWindowTitle("Music Duplicate Finder")
        loading.setWindowModality(Qt.WindowModality.ApplicationModal)
        loading.setMinimumDuration(0)
        loading.setMinimumWidth(530)
        loading.setCancelButton(None)
        loading.show()
        QApplication.processEvents()
        dlg = ResultsDialog(result, window, anomaly_multiplier=anomaly_multiplier, anomaly_size_mb=anomaly_size_mb)
        loading.close()
        dlg.exec()

    def on_error(message: str) -> None:
        QMessageBox.critical(
            window,
            "AcoustID Scan Failed",
            "The AcoustID duplicate scan encountered an error:\n\n{0}\n\n"
            "If the error mentions pyacoustid, install it with:\n"
            "    pip install pyacoustid\n\n"
            "See the diagnostic log for more detail.".format(message),
        )

    progress = ProgressDialog(
        parent      = window,
        worker      = worker,
        on_finished = on_finished,
        on_error    = on_error,
    )
    progress.exec()


# ══════════════════════════════════════════════════════════════════════════
# Tools-menu actions
# ══════════════════════════════════════════════════════════════════════════

class _LoadThread(QThread):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self) -> None:
        from .results_io import load_result
        try:
            self.finished.emit(load_result(self._path))
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class LoadResultsAction(BaseAction):
    """Reload a previously saved .mdupe results file without rescanning."""
    TITLE = "Load Duplicate Results…"

    def callback(self, objs) -> None:
        from PyQt6.QtWidgets import QFileDialog

        window = self.api.tagger.window
        path, _ = QFileDialog.getOpenFileName(
            window,
            "Load Duplicate Finder Results",
            "",
            "Duplicate results (*.mdupe);;All files (*)",
        )
        if not path:
            return

        progress = QProgressDialog("Loading results file…", "", 0, 0, window)
        progress.setWindowTitle("Music Duplicate Finder")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(530)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()  # force paint before thread starts

        thread = _LoadThread(path, window)

        def on_finished(result) -> None:
            thread.deleteLater()
            if not result.groups:
                progress.close()
                QMessageBox.information(
                    window, "No Results", "The file contained no duplicate groups."
                )
                return
            progress.setLabelText(f"Building display for {len(result.groups)} groups…")
            QApplication.processEvents()
            cfg = self.api.plugin_config
            anomaly_multiplier = cfg_get(cfg, "anomaly_size_multiplier", 1.7)
            anomaly_size_mb    = cfg_get(cfg, "anomaly_size_mb", 7.5)
            dlg = ResultsDialog(result, window, loaded_from_file=True,
                                anomaly_multiplier=anomaly_multiplier, anomaly_size_mb=anomaly_size_mb)
            progress.close()
            dlg.exec()

        def on_error(msg: str) -> None:
            progress.close()
            thread.deleteLater()
            QMessageBox.critical(window, "Load Error", f"Could not load results:\n{msg}")

        thread.finished.connect(on_finished)
        thread.error.connect(on_error)
        thread.start()


class FindDuplicatesAcoustIDAction(BaseAction):
    """Primary duplicate-detection action — chromaprint-based."""
    TITLE = "Find Duplicates (AcoustID)…  [Ctrl+Shift+D]"

    def callback(self, objs) -> None:
        _start_chromaprint_scan(
            api            = self.api,
            window         = self.api.tagger.window,
            restrict_paths = None,
            source_label   = "your library",
        )


class FindSimilarSongsClapAction(BaseAction):
    """Similar-songs action — CLAP genre/mood based."""
    TITLE = "Find Similar Songs (CLAP, GPU)…"

    def callback(self, objs) -> None:
        _start_clap_scan(
            api            = self.api,
            window         = self.api.tagger.window,
            restrict_paths = None,
            source_label   = "your library",
        )


# ══════════════════════════════════════════════════════════════════════════
# Right-click actions — AcoustID family
# ══════════════════════════════════════════════════════════════════════════

class AcoustIDFilesAction(BaseAction):
    TITLE = "Find Duplicates — AcoustID (selected files)  [Ctrl+Shift+D = full scan]"

    def callback(self, objs) -> None:
        paths = _extract_files_from_objs(objs)
        n = len(paths)
        label = f"the selected {n} file{'s' if n != 1 else ''}"
        _start_chromaprint_scan(self.api, self.api.tagger.window, paths, label)


class AcoustIDClusterAction(BaseAction):
    TITLE = "Find Duplicates — AcoustID (this cluster)  [Ctrl+Shift+D = full scan]"

    def callback(self, objs) -> None:
        paths = _extract_files_from_objs(objs)
        n_clusters = len(objs) if objs else 0
        label = (
            "the selected cluster"
            if n_clusters == 1
            else f"the {n_clusters} selected clusters"
        )
        _start_chromaprint_scan(self.api, self.api.tagger.window, paths, label)


class AcoustIDAlbumAction(BaseAction):
    TITLE = "Find Duplicates — AcoustID (this album)  [Ctrl+Shift+D = full scan]"

    def callback(self, objs) -> None:
        paths = _extract_files_from_objs(objs)
        n_albums = len(objs) if objs else 0
        label = (
            "the selected album"
            if n_albums == 1
            else f"the {n_albums} selected albums"
        )
        _start_chromaprint_scan(self.api, self.api.tagger.window, paths, label)


# ══════════════════════════════════════════════════════════════════════════
# Right-click actions — CLAP family
# ══════════════════════════════════════════════════════════════════════════

class ClapFilesAction(BaseAction):
    TITLE = "Find Similar Songs — CLAP (selected files)"

    def callback(self, objs) -> None:
        paths = _extract_files_from_objs(objs)
        n = len(paths)
        label = f"the selected {n} file{'s' if n != 1 else ''}"
        _start_clap_scan(self.api, self.api.tagger.window, paths, label)


class ClapClusterAction(BaseAction):
    TITLE = "Find Similar Songs — CLAP (this cluster)"

    def callback(self, objs) -> None:
        paths = _extract_files_from_objs(objs)
        n_clusters = len(objs) if objs else 0
        label = (
            "the selected cluster"
            if n_clusters == 1
            else f"the {n_clusters} selected clusters"
        )
        _start_clap_scan(self.api, self.api.tagger.window, paths, label)


class ClapAlbumAction(BaseAction):
    TITLE = "Find Similar Songs — CLAP (this album)"

    def callback(self, objs) -> None:
        paths = _extract_files_from_objs(objs)
        n_albums = len(objs) if objs else 0
        label = (
            "the selected album"
            if n_albums == 1
            else f"the {n_albums} selected albums"
        )
        _start_clap_scan(self.api, self.api.tagger.window, paths, label)
