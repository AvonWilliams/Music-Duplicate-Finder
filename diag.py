# Music Duplicate Finder — diag.py
# Diagnostic logger. Every module uses get_logger() to obtain a logger that
# writes to Picard's log (via api.logger when available) AND to a rotating
# plain-text file the user can send back for debugging.
#
# Log file path:
#   Linux/macOS : ~/.cache/music-duplicate-finder/scan.log
#   Windows     : %LOCALAPPDATA%\music-duplicate-finder\scan.log
# Rotates at 2 MB, keeps last 3 files.

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

_LOGGER_NAME    = "music_duplicate_finder"
_FILE_LOGGER    = None           # set on first call to _configure()
_LOG_PATH: Path | None = None
_PICARD_API     = None


# ── Log-file location ──────────────────────────────────────────────────────

def _default_log_dir() -> Path:
    """Per-OS writable directory for the plugin log file."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return Path(base) / "music-duplicate-finder"
    # Linux / macOS
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(cache) / "music-duplicate-finder"


def log_file_path() -> Path:
    """Return the current log file path (configures the logger if needed)."""
    _configure()
    return _LOG_PATH  # type: ignore[return-value]


# ── One-time setup ─────────────────────────────────────────────────────────

def _configure() -> logging.Logger:
    global _FILE_LOGGER, _LOG_PATH
    if _FILE_LOGGER is not None:
        return _FILE_LOGGER

    log_dir = _default_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Fall back to /tmp if the preferred dir is unwritable
        log_dir = Path("/tmp") if os.name != "nt" else Path(os.environ.get("TEMP", "."))

    _LOG_PATH = log_dir / "scan.log"

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False   # don't double-echo via root logger

    # Avoid duplicate handlers if _configure() is called twice
    if not any(isinstance(h, logging.handlers.RotatingFileHandler)
               for h in logger.handlers):
        try:
            fh = logging.handlers.RotatingFileHandler(
                _LOG_PATH,
                maxBytes    = 2 * 1024 * 1024,   # 2 MB
                backupCount = 3,
                encoding    = "utf-8",
            )
            fmt = logging.Formatter(
                "%(asctime)s  %(levelname)-7s  %(name)s:%(lineno)d  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fh.setFormatter(fmt)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        except OSError as exc:
            # Last resort — stderr only. We still return a working logger.
            sys.stderr.write(f"[MusicDupeFinder] could not open log file: {exc}\n")

    _FILE_LOGGER = logger
    return logger


# ── Public API ─────────────────────────────────────────────────────────────

def set_picard_api(api) -> None:
    """
    Remember Picard's PluginApi so messages can also be forwarded to its
    own logger. Called once from __init__.enable().
    """
    global _PICARD_API
    _PICARD_API = api
    _configure()
    _FILE_LOGGER.info("=" * 72)  # type: ignore[union-attr]
    _FILE_LOGGER.info("Plugin enabled — log file: %s", _LOG_PATH)  # type: ignore[union-attr]


class _DualLogger:
    """
    Wrapper that writes to our file logger and ALSO forwards to Picard's
    api.logger if one has been registered. Supports the usual debug/info/
    warning/error/exception methods.
    """

    __slots__ = ("_module",)

    def __init__(self, module: str):
        self._module = module

    def _emit(self, level: str, msg: str, *args, **kwargs) -> None:
        logger = _configure()
        fn = getattr(logger, level)
        # Prefix every line with the module so multiple components are
        # trivially greppable in the combined log.
        fn(f"[{self._module}] {msg}", *args, **kwargs)

        if _PICARD_API is not None:
            try:
                p_fn = getattr(_PICARD_API.logger, level, None)
                if p_fn is not None:
                    # Picard's logger doesn't support %-style args reliably,
                    # so we pre-format.
                    try:
                        formatted = msg % args if args else msg
                    except Exception:
                        formatted = msg
                    p_fn(f"[MusicDupe/{self._module}] {formatted}")
            except Exception:
                pass  # never let logging crash the scan

    def debug(self,   msg, *a, **kw): self._emit("debug",   msg, *a, **kw)
    def info(self,    msg, *a, **kw): self._emit("info",    msg, *a, **kw)
    def warning(self, msg, *a, **kw): self._emit("warning", msg, *a, **kw)
    def error(self,   msg, *a, **kw): self._emit("error",   msg, *a, **kw)
    def exception(self, msg, *a, **kw): self._emit("exception", msg, *a, **kw)


def get_logger(module: str) -> _DualLogger:
    """Obtain a logger for `module` (e.g. 'scan_worker', 'server_client')."""
    return _DualLogger(module)


# ── Convenience: section headers inside the log ───────────────────────────

def log_section(title: str) -> None:
    logger = _configure()
    logger.info("")
    logger.info("─" * 72)
    logger.info("▶  %s  —  %s", title, time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("─" * 72)
