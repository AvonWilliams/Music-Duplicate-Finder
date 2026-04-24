# Music Duplicate Finder — scan_worker.py
# QThread that drives the full scan pipeline. Two modes:
#   REMOTE: collect → remap paths → POST to FastAPI server → enrich with mutagen
#   LOCAL:  collect → embed locally on GPU → cluster → enrich with mutagen
# Both modes emit the same ScanResult shape to the main thread.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

from .diag import get_logger, log_section
from .file_collector import collect_files, remap_to_lxc
from .quality import FileQuality, analyse_file
from .server_client import ServerClient

if TYPE_CHECKING:
    from picard.plugin3.api import PluginApi


_log = get_logger("scan_worker")


@dataclass
class DuplicateGroup:
    """One group of duplicate files, enriched with per-file quality data."""
    confidence: str                                     # certain | likely | unsure
    similarity: float                                   # 0.0 – 1.0
    files: list[FileQuality] = field(default_factory=list)

    @property
    def best(self) -> FileQuality | None:
        return self.files[0] if self.files else None


@dataclass
class ScanResult:
    groups: list[DuplicateGroup]
    total_files_scanned: int
    elapsed_seconds: float
    mode: str = "remote"   # "remote" or "local"


class ScanWorker(QThread):
    """
    Worker thread. Emits:
      progress(current, total, message)
      finished(ScanResult)
      error(message)
    """

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(
        self,
        api: "PluginApi",
        inference_mode: str,          # "remote" | "local" | "chromaprint"
        # ── shared ────────────────────────────────────────────────────────
        certain_threshold: int,
        likely_threshold: int,
        unsure_threshold: int,
        # ── remote mode ───────────────────────────────────────────────────
        server_host: str = "",
        server_port: int = 0,
        api_key:     str = "",
        win_root:    str = "",
        lxc_root:    str = "",
        # ── local mode ────────────────────────────────────────────────────
        local_gpu_index: int  = 0,
        model_cache_dir: str  = "",
        # ── chromaprint mode ──────────────────────────────────────────────
        chromaprint_alignment: str  = "standard",
        chromaprint_use_gpu:   bool = True,
        # Pre-harvested fingerprints for chromaprint mode. The caller reads
        # file.metadata on the main thread (Picard requirement) and passes
        # the results in.
        chromaprint_fingerprints: "dict[str, str] | None" = None,
        # ── scope ─────────────────────────────────────────────────────────
        restrict_paths: "list[str] | None" = None,
    ):
        super().__init__()
        self._api     = api
        self._mode    = inference_mode
        self._certain = certain_threshold / 100.0
        self._likely  = likely_threshold  / 100.0
        self._unsure  = unsure_threshold  / 100.0

        # Remote
        self._host     = server_host
        self._port     = server_port
        self._api_key  = api_key
        self._win_root = win_root
        self._lxc_root = lxc_root

        # Local CLAP
        self._local_gpu_index = local_gpu_index
        self._model_cache_dir = model_cache_dir or None

        # Chromaprint
        self._cp_alignment        = chromaprint_alignment
        self._cp_use_gpu          = chromaprint_use_gpu
        self._cp_fingerprints     = chromaprint_fingerprints or {}

        # Scope
        self._restrict_paths = restrict_paths

        self._abort = False

    def abort(self) -> None:
        self._abort = True

    # ── QThread entry point ────────────────────────────────────────────────

    def run(self) -> None:
        try:
            log_section(f"SCAN STARTED  (mode={self._mode})")
            _log.info(
                "Config: certain=%.3f likely=%.3f unsure=%.3f  "
                "host=%s:%s  win_root=%r  lxc_root=%r  "
                "local_gpu_index=%d  cp_alignment=%s  cp_use_gpu=%s  "
                "cp_fingerprints=%s  restrict_paths=%s",
                self._certain, self._likely, self._unsure,
                self._host, self._port, self._win_root, self._lxc_root,
                self._local_gpu_index,
                self._cp_alignment, self._cp_use_gpu,
                len(self._cp_fingerprints) if self._cp_fingerprints else 0,
                (f"{len(self._restrict_paths)} paths" if self._restrict_paths
                 is not None else "None (all loaded)"),
            )
            if self._mode == "chromaprint":
                self._run_chromaprint()
            elif self._mode == "local":
                self._run_local()
            else:
                self._run_remote()
        except Exception as exc:  # noqa: BLE001
            _log.exception("Scan failed with unhandled exception")
            self._api.logger.error(f"MusicDupeFinder scan error: {exc}")
            self.error.emit(str(exc))

    # ── Path collection (Tools-menu vs right-click) ────────────────────────

    def _collect_paths(self) -> list[str]:
        """
        Return the file paths this scan should cover.

        If the worker was created with `restrict_paths` (right-click), use
        those directly.  Otherwise (Tools-menu) walk Picard's left + right
        panels and collect every loaded audio file.
        """
        if self._restrict_paths is not None:
            # Deduplicate while preserving order
            seen: set[str] = set()
            out: list[str] = []
            for p in self._restrict_paths:
                if p and p not in seen:
                    seen.add(p)
                    out.append(p)
            return out
        return collect_files(self._api)

    def _empty_msg(self) -> str:
        if self._restrict_paths is not None:
            return "No audio files were found in the right-click selection."
        return "No audio files are currently loaded in Picard."

    # ── REMOTE mode ────────────────────────────────────────────────────────

    def _run_remote(self) -> None:
        self.progress.emit(0, 0, "Collecting files from Picard…")
        win_paths = self._collect_paths()
        if not win_paths:
            self.error.emit(self._empty_msg())
            return
        if self._abort:
            return

        total = len(win_paths)
        self.progress.emit(0, total, f"Found {total} files — remapping paths…")

        lxc_paths  = [remap_to_lxc(p, self._win_root, self._lxc_root) for p in win_paths]
        lxc_to_win = dict(zip(lxc_paths, win_paths))

        # Log a sample so we can verify the mapping is sane
        _log.info("Path remap check — first 3 mappings:")
        for i in range(min(3, len(win_paths))):
            unchanged = win_paths[i] == lxc_paths[i]
            _log.info("  [%d] win=%r  →  lxc=%r%s",
                      i, win_paths[i], lxc_paths[i],
                      "  (UNCHANGED — remap did not apply)" if unchanged else "")

        if self._abort:
            return

        self.progress.emit(0, total, f"Sending {total} files to GPU server…")
        raw = ServerClient.scan(
            host              = self._host,
            port              = self._port,
            api_key           = self._api_key,
            lxc_paths         = lxc_paths,
            certain_threshold = self._certain,
            likely_threshold  = self._likely,
            unsure_threshold  = self._unsure,
        )
        if self._abort:
            return

        raw_groups = raw.get("groups", [])
        scanned    = raw.get("scanned", total)
        elapsed    = raw.get("elapsed_seconds", 0.0)
        _log.info(
            "Server returned %d raw groups; reported scanned=%s elapsed=%ss",
            len(raw_groups), scanned, elapsed,
        )
        if scanned == 0:
            _log.warning(
                "Server reported 0 files scanned — this usually means the "
                "server could not access the files at the LXC paths. "
                "Check that lxc_root (%r) points to the same content as "
                "win_root (%r) and that the server process can read those files.",
                self._lxc_root, self._win_root,
            )
        if len(raw_groups) == 0 and scanned > 1:
            _log.warning(
                "Server scanned %s files but returned 0 groups. "
                "Either nothing similar was found, or thresholds are too high "
                "(current unsure=%.3f). If you know duplicates exist, lower "
                "the Unsure threshold to ~0.50 and retry.",
                scanned, self._unsure,
            )

        self._finalise(
            raw_groups = raw_groups,
            scanned    = scanned,
            elapsed    = elapsed,
            path_map   = lxc_to_win,
            mode       = "remote",
        )

    # ── LOCAL mode ─────────────────────────────────────────────────────────

    def _run_local(self) -> None:
        # Import lazily so missing deps don't break Picard startup
        from .local_inference import LocalInferenceEngine, MissingDependencyError

        self.progress.emit(0, 0, "Collecting files from Picard…")
        win_paths = self._collect_paths()
        if not win_paths:
            self.error.emit(self._empty_msg())
            return
        if self._abort:
            return

        total = len(win_paths)
        self.progress.emit(0, total, f"Found {total} files — loading CLAP model…")

        # Bridge engine progress strings → our progress signal
        state = {"current": 0}
        def _emit_progress(msg: str) -> None:
            # Try to parse "Embedding 42 / 512  —  file.mp3" for a numeric count
            import re
            m = re.match(r"Embedding (\d+) / (\d+)", msg)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                state["current"] = cur
                self.progress.emit(cur, tot, msg)
            else:
                self.progress.emit(state["current"], total, msg)

        try:
            engine = LocalInferenceEngine(
                gpu_index       = self._local_gpu_index,
                model_cache_dir = self._model_cache_dir,
                progress_cb     = _emit_progress,
            )
            engine.load_model()
        except MissingDependencyError as exc:
            self.error.emit(str(exc))
            return

        if self._abort:
            return

        raw = engine.find_duplicates(
            win_paths         = win_paths,
            certain_threshold = self._certain,
            likely_threshold  = self._likely,
            unsure_threshold  = self._unsure,
            abort_flag        = lambda: self._abort,
        )
        if self._abort:
            return

        raw_groups = raw.get("groups", [])
        scanned    = raw.get("scanned", total)
        embedded   = raw.get("embedded", -1)        # V1.4+; -1 = unknown
        fails      = raw.get("embed_failures", -1)
        elapsed    = raw.get("elapsed_seconds", 0.0)
        _log.info(
            "Local engine returned %d raw groups; scanned=%s embedded=%s "
            "embed_failures=%s elapsed=%.2fs",
            len(raw_groups), scanned, embedded, fails, elapsed,
        )

        # Surface a clear error if every file failed to embed (otherwise the
        # UI just shows a confusing "no duplicates found" — see HANDOFF §3.4).
        if embedded == 0 and total > 0:
            self.error.emit(
                "Failed to decode any audio files (%d of %d).\n\n"
                "The plugin tried soundfile first, then ffmpeg as a fallback. "
                "Check the diagnostic log for per-file error details:\n\n"
                "    ~/.cache/music-duplicate-finder/scan.log\n\n"
                "Most likely causes:\n"
                "  • ffmpeg not on PATH (required for .mka / Matroska)\n"
                "  • soundfile (libsndfile) package missing\n"
                "  • the files themselves are corrupt or not audio"
                % (fails if fails >= 0 else total, total)
            )
            return

        if len(raw_groups) == 0 and scanned > 1:
            _log.warning(
                "Local mode scanned %s files but produced 0 groups. "
                "Unsure threshold=%.3f — if duplicates are known-present, "
                "lower it to ~0.50 to see what the similarity matrix looks like.",
                scanned, self._unsure,
            )

        # Local mode returns Windows paths already — identity map
        self._finalise(
            raw_groups = raw_groups,
            scanned    = scanned,
            elapsed    = elapsed,
            path_map   = None,
            mode       = "local",
        )

    # ── CHROMAPRINT mode ───────────────────────────────────────────────────

    def _run_chromaprint(self) -> None:
        from .chromaprint_engine import (
            ChromaprintEngine,
            MissingDependencyError as CpMissingDep,
            NoFingerprintsError,
        )

        self.progress.emit(0, 0, "Preparing AcoustID scan…")

        # Fingerprints were harvested on the main thread; we just use them.
        fp_map = dict(self._cp_fingerprints)
        if not fp_map:
            self.error.emit(
                "No files with AcoustID fingerprints were provided. "
                "Run Picard's Tools → Scan first to calculate them."
            )
            return
        if self._abort:
            return

        total = len(fp_map)
        self.progress.emit(
            0, total,
            "Found {0} files with fingerprints — comparing…".format(total),
        )

        def _emit(msg: str) -> None:
            self.progress.emit(0, total, msg)

        try:
            engine = ChromaprintEngine(
                alignment   = self._cp_alignment,
                use_gpu     = self._cp_use_gpu,
                gpu_index   = self._local_gpu_index,
                progress_cb = _emit,
            )
        except CpMissingDep as exc:
            self.error.emit(str(exc))
            return

        try:
            raw = engine.find_duplicates(
                fingerprint_map   = fp_map,
                certain_threshold = self._certain,
                likely_threshold  = self._likely,
                unsure_threshold  = self._unsure,
                abort_flag        = lambda: self._abort,
            )
        except CpMissingDep as exc:
            self.error.emit(str(exc))
            return
        except NoFingerprintsError as exc:
            self.error.emit(str(exc))
            return

        if self._abort:
            return

        raw_groups = raw.get("groups", [])
        scanned    = raw.get("scanned", total)
        fails      = raw.get("embed_failures", 0)
        elapsed    = raw.get("elapsed_seconds", 0.0)
        _log.info(
            "Chromaprint engine returned %d raw groups; scanned=%s "
            "decode_failures=%s elapsed=%.2fs",
            len(raw_groups), scanned, fails, elapsed,
        )

        if scanned < 2:
            self.error.emit(
                "Only {0} file(s) had decodable fingerprints — need at least 2."
                .format(scanned)
            )
            return

        if len(raw_groups) == 0 and scanned > 1:
            _log.info(
                "Chromaprint mode scanned %s files and found 0 duplicate "
                "groups. Unsure threshold=%.3f — if duplicates are known-"
                "present, check that the files actually share recordings "
                "(chromaprint identifies the same master, not cover versions).",
                scanned, self._unsure,
            )

        self._finalise(
            raw_groups = raw_groups,
            scanned    = scanned,
            elapsed    = elapsed,
            path_map   = None,
            mode       = "chromaprint",
        )

    # ── Shared finalisation ────────────────────────────────────────────────

    def _finalise(
        self,
        raw_groups: list,
        scanned: int,
        elapsed: float,
        path_map: dict | None,
        mode: str,
    ) -> None:
        n_groups = len(raw_groups)
        self.progress.emit(0, n_groups, "Analysing file quality…")
        _log.info("Finalising %d raw groups (mode=%s)…", n_groups, mode)

        groups: list[DuplicateGroup] = []
        analyse_failures = 0
        dropped_too_small = 0

        for idx, rg in enumerate(raw_groups):
            if self._abort:
                return
            self.progress.emit(idx, n_groups, f"Analysing group {idx + 1} / {n_groups}…")

            confidence = rg.get("confidence", "unsure")
            similarity = float(rg.get("similarity", 0.0))
            file_paths = rg.get("files", [])

            qualities: list[FileQuality] = []
            for p in file_paths:
                # Remote returns LXC paths — remap back to Windows for disk I/O
                display_path = path_map.get(p, p) if path_map else p
                fq = analyse_file(display_path)
                if fq:
                    qualities.append(fq)
                else:
                    analyse_failures += 1
                    _log.warning(
                        "analyse_file returned None for %r "
                        "(group %d, confidence=%s) — file will be dropped",
                        display_path, idx, confidence,
                    )

            qualities.sort(key=lambda q: q.score, reverse=True)
            if len(qualities) >= 2:
                groups.append(DuplicateGroup(
                    confidence = confidence,
                    similarity = similarity,
                    files      = qualities,
                ))
            else:
                dropped_too_small += 1
                _log.warning(
                    "Group %d dropped: had %d file paths, only %d analysed "
                    "successfully (need ≥ 2). paths=%s",
                    idx, len(file_paths), len(qualities),
                    [path_map.get(p, p) if path_map else p for p in file_paths],
                )

        # Sort groups: certain first, then by similarity descending
        order = {"certain": 0, "likely": 1, "unsure": 2}
        groups.sort(key=lambda g: (order.get(g.confidence, 9), -g.similarity))

        _log.info(
            "Finalise complete: %d → %d groups after filtering  "
            "(analyse_failures=%d, groups_dropped_lt_2_files=%d)",
            n_groups, len(groups), analyse_failures, dropped_too_small,
        )
        log_section(f"SCAN FINISHED  ({len(groups)} groups, {scanned} files)")

        self.progress.emit(n_groups, n_groups, "Done!")
        self.finished.emit(ScanResult(
            groups              = groups,
            total_files_scanned = scanned,
            elapsed_seconds     = elapsed,
            mode                = mode,
        ))
