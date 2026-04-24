# Music Duplicate Finder — quality.py
# Computes a composite quality score for each audio file and detects
# live recordings, which are penalised by 50 %.

from __future__ import annotations

import os
import re

from .diag import get_logger

_log = get_logger("quality")

# mutagen is bundled with Picard — safe to import.
try:
    from mutagen import File as MutagenFile
    from mutagen.mp3 import BitrateMode, MP3
    _MUTAGEN = True
except ImportError:
    _MUTAGEN = False


# ── Live-recording heuristics ──────────────────────────────────────────────

_LIVE_TITLE_RE = re.compile(
    r"\blive\b|\bconcert\b|\bstage\b|\bunplugged\b|\bacoustic\b|\btour\b"
    r"|\blive at\b|\brecorded at\b|\bin concert\b|\bfrom the\b.*\btour\b",
    re.IGNORECASE,
)
_LIVE_FILENAME_RE = re.compile(
    r"[-_\s](live|concert|tour|unplugged|acoustic)[-_\s\.]",
    re.IGNORECASE,
)
_LIVE_TAG_VALUES = frozenset({"live", "concert", "live performance", "live recording"})


def is_live_recording(file_path: str, mutagen_tags) -> bool:
    """
    Return True if the file appears to be a live recording.
    Checks (in order): explicit tags → title → album → filename.
    """
    if mutagen_tags:
        # Check common live-indicator tags
        for key in ("comment", "description", "contentgroup", "tit3",
                    "grouping", "TIT3", "COMM"):
            val = _tag_str(mutagen_tags, key)
            if val and val.lower() in _LIVE_TAG_VALUES:
                return True
        # Title check
        title = _tag_str(mutagen_tags, "title") or _tag_str(mutagen_tags, "TIT2")
        if title and _LIVE_TITLE_RE.search(title):
            return True
        # Album check
        album = _tag_str(mutagen_tags, "album") or _tag_str(mutagen_tags, "TALB")
        if album and _LIVE_TITLE_RE.search(album):
            return True

    # Fallback: filename
    basename = os.path.basename(file_path)
    if _LIVE_FILENAME_RE.search(basename):
        return True

    return False


def _tag_str(tags, key: str) -> str:
    """Safely extract a string value from mutagen tags."""
    try:
        v = tags[key]
        if hasattr(v, "text"):          # ID3 frame
            return str(v.text[0]) if v.text else ""
        if isinstance(v, list):
            return str(v[0]) if v else ""
        return str(v)
    except (KeyError, IndexError, TypeError):
        return ""


# ── Quality scoring ────────────────────────────────────────────────────────

class FileQuality:
    """
    Holds quality metrics for a single audio file.

    score = bitrate_kbps × file_size_mb
    If the file is a live recording the score is halved.
    """

    __slots__ = (
        "path", "bitrate_kbps", "file_size_bytes", "format_name",
        "sample_rate_hz", "channels", "duration_sec",
        "is_live", "raw_score", "score",
        "title", "artist", "album", "year", "tags_dict",
    )

    def __init__(
        self,
        path: str,
        bitrate_kbps: float,
        file_size_bytes: int,
        format_name: str,
        sample_rate_hz: int,
        channels: int,
        duration_sec: float,
        is_live: bool,
        title: str,
        artist: str,
        album: str,
        year: str,
        tags_dict: dict,
    ):
        self.path           = path
        self.bitrate_kbps   = bitrate_kbps
        self.file_size_bytes = file_size_bytes
        self.format_name    = format_name
        self.sample_rate_hz = sample_rate_hz
        self.channels       = channels
        self.duration_sec   = duration_sec
        self.is_live        = is_live
        self.title          = title
        self.artist         = artist
        self.album          = album
        self.year           = year
        self.tags_dict      = tags_dict  # raw {key: str_value} for tags viewer

        file_size_mb = file_size_bytes / (1024 * 1024)
        self.raw_score = bitrate_kbps * file_size_mb
        self.score = self.raw_score * (0.5 if is_live else 1.0)

    @property
    def file_size_mb(self) -> float:
        return self.file_size_bytes / (1024 * 1024)

    @property
    def duration_str(self) -> str:
        m, s = divmod(int(self.duration_sec), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"


def analyse_file(path: str) -> FileQuality | None:
    """
    Read audio metadata with mutagen and return a FileQuality object.
    Returns None if the file cannot be read.
    """
    if not _MUTAGEN:
        _log.debug("analyse_file(%r): mutagen not available → fallback", path)
        return _fallback_quality(path)

    if not os.path.exists(path):
        _log.warning("analyse_file(%r): file does not exist at this path", path)
        return _fallback_quality(path)

    try:
        audio = MutagenFile(path, easy=False)
    except Exception as exc:  # noqa: BLE001
        _log.warning("analyse_file(%r): MutagenFile raised %s: %s → fallback",
                     path, type(exc).__name__, exc)
        return _fallback_quality(path)

    if audio is None:
        _log.warning("analyse_file(%r): MutagenFile returned None → fallback", path)
        return _fallback_quality(path)

    try:
        file_size = os.path.getsize(path)
    except OSError:
        file_size = 0

    info = audio.info
    bitrate_kbps = getattr(info, "bitrate", 0) / 1000 if hasattr(info, "bitrate") else 0.0
    sample_rate  = getattr(info, "sample_rate", 0)
    channels     = getattr(info, "channels", 2)
    duration     = getattr(info, "length", 0.0)
    fmt          = type(audio).__name__.replace("Easy", "").upper()

    tags = audio.tags or {}
    live = is_live_recording(path, tags)

    # ── Extract common tag strings ────────────────────────────────────────
    def t(key, *alts):
        for k in (key, *alts):
            v = _tag_str(tags, k)
            if v:
                return v
        return ""

    title  = t("title",  "TIT2")
    artist = t("artist", "TPE1", "albumartist", "TPE2")
    album  = t("album",  "TALB")
    year   = t("date",   "TDRC", "year", "TYER")

    # Build flat tags dict for the viewer
    tags_flat: dict[str, str] = {}
    for k in tags.keys():
        try:
            tags_flat[str(k)] = _tag_str(tags, k)
        except Exception:  # noqa: BLE001
            pass

    return FileQuality(
        path           = path,
        bitrate_kbps   = bitrate_kbps,
        file_size_bytes = file_size,
        format_name    = fmt,
        sample_rate_hz = sample_rate,
        channels       = channels,
        duration_sec   = duration,
        is_live        = live,
        title          = title or os.path.splitext(os.path.basename(path))[0],
        artist         = artist,
        album          = album,
        year           = year,
        tags_dict      = tags_flat,
    )


def _fallback_quality(path: str) -> FileQuality | None:
    """Minimal quality object when mutagen is unavailable."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        _log.warning("_fallback_quality(%r): os.path.getsize failed: %s → returning None",
                     path, exc)
        return None
    return FileQuality(
        path           = path,
        bitrate_kbps   = 0.0,
        file_size_bytes = size,
        format_name    = os.path.splitext(path)[1].lstrip(".").upper(),
        sample_rate_hz = 0,
        channels       = 0,
        duration_sec   = 0.0,
        is_live        = is_live_recording(path, None),
        title          = os.path.splitext(os.path.basename(path))[0],
        artist         = "",
        album          = "",
        year           = "",
        tags_dict      = {},
    )
