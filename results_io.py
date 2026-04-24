# Music Duplicate Finder — results_io.py
# Serialise / deserialise ScanResult to JSON so results can be saved and
# reloaded without running another scan.

from __future__ import annotations

import json

_FORMAT_VERSION = 1


def save_result(result, path: str) -> None:
    """Write a ScanResult to a JSON file."""
    data = {
        "version":             _FORMAT_VERSION,
        "mode":                result.mode,
        "total_files_scanned": result.total_files_scanned,
        "elapsed_seconds":     result.elapsed_seconds,
        "groups":              [_group_to_dict(g) for g in result.groups],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_result(path: str):
    """Read a ScanResult from a JSON file."""
    from .scan_worker import ScanResult, DuplicateGroup  # noqa: F401 (used below)

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    version = data.get("version", 1)
    if version > _FORMAT_VERSION:
        raise ValueError(
            f"Results file was saved by a newer version of the plugin "
            f"(format v{version}). Please update the plugin."
        )

    groups = [_dict_to_group(d) for d in data.get("groups", [])]
    return ScanResult(
        groups              = groups,
        total_files_scanned = data.get("total_files_scanned", 0),
        elapsed_seconds     = data.get("elapsed_seconds", 0.0),
        mode                = data.get("mode", "chromaprint"),
    )


# ── Serialisation helpers ──────────────────────────────────────────────────

def _group_to_dict(group) -> dict:
    return {
        "confidence": group.confidence,
        "similarity": group.similarity,
        "files":      [_fq_to_dict(fq) for fq in group.files],
    }


def _dict_to_group(d: dict):
    from .scan_worker import DuplicateGroup

    group = DuplicateGroup(
        confidence = d["confidence"],
        similarity = d["similarity"],
    )
    group.files = [_dict_to_fq(f) for f in d.get("files", [])]
    return group


def _fq_to_dict(fq) -> dict:
    return {
        "path":            fq.path,
        "bitrate_kbps":    fq.bitrate_kbps,
        "file_size_bytes": fq.file_size_bytes,
        "format_name":     fq.format_name,
        "sample_rate_hz":  fq.sample_rate_hz,
        "channels":        fq.channels,
        "duration_sec":    fq.duration_sec,
        "is_live":         fq.is_live,
        "title":           fq.title,
        "artist":          fq.artist,
        "album":           fq.album,
        "year":            fq.year,
        "tags_dict":       fq.tags_dict,
    }


def _dict_to_fq(d: dict):
    from .quality import FileQuality

    return FileQuality(
        path            = d.get("path",            ""),
        bitrate_kbps    = d.get("bitrate_kbps",    0.0),
        file_size_bytes = d.get("file_size_bytes",  0),
        format_name     = d.get("format_name",      ""),
        sample_rate_hz  = d.get("sample_rate_hz",   0),
        channels        = d.get("channels",          0),
        duration_sec    = d.get("duration_sec",     0.0),
        is_live         = d.get("is_live",          False),
        title           = d.get("title",             ""),
        artist          = d.get("artist",            ""),
        album           = d.get("album",             ""),
        year            = d.get("year",              ""),
        tags_dict       = d.get("tags_dict",         {}),
    )
