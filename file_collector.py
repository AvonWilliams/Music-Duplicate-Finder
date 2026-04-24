# Music Duplicate Finder — file_collector.py
# Walks every Picard container (unmatched files, clusters, albums/tracks)
# and returns a flat list of absolute file paths for all loaded audio files.

from __future__ import annotations

from typing import TYPE_CHECKING

from .diag import get_logger

if TYPE_CHECKING:
    from picard.plugin3.api import PluginApi


_log = get_logger("file_collector")


AUDIO_EXTENSIONS = frozenset(
    {".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wav", ".aiff",
     ".wv", ".ape", ".mpc", ".wma", ".dsf", ".dff"}
)


# Candidate attribute names for the "files not yet clustered or matched"
# container on the Tagger. Picard V2 used `unclustered_files` (a Cluster-
# like object with `.files`). V3 may have renamed it. We try these in
# priority order and dump `dir(tagger)` if none work so the next iteration
# can learn the right name from Avon's log.
_UNCLUSTERED_CANDIDATE_ATTRS = (
    "unclustered_files",   # V2 canonical name
    "unmatched_files",     # legacy V1.3 code used this, may be a V3 alias
    "file_list",           # plausible V3 rename
    "files",               # fallback if Tagger exposes a flat dict/list
)


def _collect_unclustered(tagger, _add) -> bool:
    """
    Walk the unclustered/unmatched files container. Returns True if any
    known attribute was found (regardless of whether files came out of it).
    Logs dir(tagger) once if every candidate failed.
    """
    for attr in _UNCLUSTERED_CANDIDATE_ATTRS:
        container = getattr(tagger, attr, None)
        if container is None:
            continue
        # Container could be a FileList/Cluster-like object with .files,
        # OR a dict/list that's already iterable of File objects.
        files_iter = getattr(container, "files", None)
        if files_iter is None:
            # Try iterating directly (dict values or list)
            try:
                if hasattr(container, "values"):
                    files_iter = list(container.values())
                elif hasattr(container, "__iter__"):
                    files_iter = list(container)
            except Exception as exc:  # noqa: BLE001
                _log.debug("could not iterate tagger.%s: %s: %s",
                           attr, type(exc).__name__, exc)
                continue
        if files_iter is None:
            _log.debug("tagger.%s exists but has no .files / is not iterable "
                       "(type=%s)", attr, type(container).__name__)
            continue
        count = 0
        for f in files_iter:
            _add(f, "unmatched")
            count += 1
        _log.info("Found %d unclustered files via tagger.%s (type=%s)",
                  count, attr, type(container).__name__)
        return True

    # Nothing worked — dump attributes so the next session can fix this.
    public_attrs = sorted(a for a in dir(tagger) if not a.startswith("_"))
    _log.warning(
        "No unclustered-files attribute on Tagger. Tried: %s. "
        "Public attributes available on tagger (for next-session debugging): %s",
        list(_UNCLUSTERED_CANDIDATE_ATTRS), public_attrs,
    )
    return False


def collect_files(api: "PluginApi") -> list[str]:
    """
    Return absolute paths of every audio file currently loaded in Picard.
    Covers: unmatched_files, clusters, and matched albums → tracks → files.
    """
    tagger = api.tagger
    seen:  set[str] = set()
    paths: list[str] = []

    counts = {"unmatched": 0, "clusters": 0, "albums": 0, "cluster_count": 0, "album_count": 0}

    def _add(file_obj, bucket: str) -> None:
        try:
            path = file_obj.filename
        except AttributeError:
            _log.warning("file object has no .filename attribute in bucket=%s (type=%s)",
                         bucket, type(file_obj).__name__)
            return
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
            counts[bucket] = counts.get(bucket, 0) + 1

    # 1. Unmatched / unclustered (left panel, grey)
    _collect_unclustered(tagger, _add)

    # 2. Clusters (left panel, yellow folder)
    try:
        for cluster in tagger.clusters:
            counts["cluster_count"] += 1
            for f in cluster.files:
                _add(f, "clusters")
    except AttributeError as exc:
        _log.debug("clusters not accessible: %s", exc)

    # 3. Albums → tracks → files (right panel, green)
    try:
        for album in tagger.albums.values():
            counts["album_count"] += 1
            for track in album.tracks:
                for f in track.files:
                    _add(f, "albums")
            # Some albums also carry unmatched-per-album files
            album_unmatched = getattr(album, "unmatched_files", None) \
                           or getattr(album, "unclustered_files", None)
            if album_unmatched is not None:
                inner = getattr(album_unmatched, "files", album_unmatched)
                try:
                    for f in inner:
                        _add(f, "albums")
                except TypeError:
                    pass
    except AttributeError as exc:
        _log.debug("albums not accessible: %s", exc)

    _log.info(
        "Collected %d unique files from Picard — unmatched=%d, clusters=%d "
        "(from %d clusters), albums=%d (from %d albums)",
        len(paths), counts["unmatched"], counts["clusters"],
        counts["cluster_count"], counts["albums"], counts["album_count"],
    )
    # Sample the first few paths so we can eyeball them
    for i, p in enumerate(paths[:5]):
        _log.info("  sample path [%d]: %r", i, p)
    if len(paths) > 5:
        _log.info("  … and %d more", len(paths) - 5)

    return paths


def collect_files_with_fingerprints(
    api: "PluginApi",
    restrict_paths: "list[str] | None" = None,
) -> tuple[list[str], dict[str, str]]:
    """
    Walk Picard's file objects and return:
        (all_paths, fingerprint_map)
    where fingerprint_map is {path: compressed_fingerprint_string} for
    every file that has file.metadata['acoustid_fingerprint'] set.

    Files without a fingerprint still appear in all_paths (so the caller
    can count and report on them) but not in the map.

    If restrict_paths is provided, only paths in that list are considered.
    """
    tagger = api.tagger
    restrict_set: "set[str] | None" = (
        set(restrict_paths) if restrict_paths is not None else None
    )

    seen: set[str] = set()
    all_paths: list[str] = []
    fp_map: dict[str, str] = {}

    # Track where fingerprints came from so we can log a breakdown — this
    # is the key diagnostic for "why does Picard claim it fingerprinted
    # these files but our plugin can't see it": fresh scans live on the
    # File attribute, not in the metadata tag.
    fp_sources = {"attribute": 0, "metadata": 0}

    def _consider(f) -> None:
        try:
            path = f.filename
        except AttributeError:
            return
        if not path or path in seen:
            return
        if restrict_set is not None and path not in restrict_set:
            return
        seen.add(path)
        all_paths.append(path)

        # Two places Picard stores chromaprint fingerprints:
        #
        #   1. file.acoustid_fingerprint (attribute on the File object)
        #      Set in-memory after Tools → Scan. This is what freshly-
        #      fingerprinted files have. Does NOT require saving.
        #
        #   2. file.metadata['acoustid_fingerprint'] (tag in the metadata)
        #      Only populated after saving with "Save AcoustID fingerprints
        #      to file tags" enabled in Picard options (off by default),
        #      OR when a file was loaded with the tag already embedded.
        #
        # Try the attribute first because freshly-computed fingerprints
        # live there; fall back to the metadata tag for persisted ones.
        fp = ""
        fp_source = None

        attr_fp = getattr(f, "acoustid_fingerprint", None)
        if attr_fp:
            if isinstance(attr_fp, (list, tuple)):
                attr_fp = attr_fp[0] if attr_fp else ""
            if isinstance(attr_fp, bytes):
                try:
                    attr_fp = attr_fp.decode("ascii")
                except Exception:  # noqa: BLE001
                    attr_fp = ""
            if attr_fp:
                fp = str(attr_fp).strip()
                fp_source = "attribute"

        if not fp:
            md = getattr(f, "metadata", None)
            if md is not None:
                try:
                    md_fp = md.get("acoustid_fingerprint", "") or md["acoustid_fingerprint"]
                except (KeyError, TypeError):
                    md_fp = ""
                except Exception:  # noqa: BLE001
                    md_fp = ""
                if md_fp:
                    if isinstance(md_fp, (list, tuple)):
                        md_fp = md_fp[0] if md_fp else ""
                    if md_fp:
                        fp = str(md_fp).strip()
                        fp_source = "metadata"

        if fp:
            fp_map[path] = fp
            if fp_source:
                fp_sources[fp_source] += 1

    # 1. Unclustered / unmatched
    for attr in _UNCLUSTERED_CANDIDATE_ATTRS:
        container = getattr(tagger, attr, None)
        if container is None:
            continue
        files_iter = getattr(container, "files", None)
        if files_iter is None:
            try:
                if hasattr(container, "values"):
                    files_iter = list(container.values())
                elif hasattr(container, "__iter__"):
                    files_iter = list(container)
            except Exception:  # noqa: BLE001
                continue
        if files_iter is None:
            continue
        for f in files_iter:
            _consider(f)
        break

    # 2. Clusters
    try:
        for cluster in tagger.clusters:
            for f in cluster.files:
                _consider(f)
    except AttributeError:
        pass

    # 3. Albums → tracks → files
    try:
        for album in tagger.albums.values():
            for track in album.tracks:
                for f in track.files:
                    _consider(f)
            album_unmatched = (
                getattr(album, "unmatched_files", None)
                or getattr(album, "unclustered_files", None)
            )
            if album_unmatched is not None:
                inner = getattr(album_unmatched, "files", album_unmatched)
                try:
                    for f in inner:
                        _consider(f)
                except TypeError:
                    pass
    except AttributeError:
        pass

    _log.info(
        "Collected %d files for AcoustID scan  (with fingerprint: %d, missing: %d)  "
        "— source breakdown: attribute=%d, metadata=%d",
        len(all_paths), len(fp_map), len(all_paths) - len(fp_map),
        fp_sources["attribute"], fp_sources["metadata"],
    )
    return all_paths, fp_map


def remap_to_lxc(win_path: str, win_root: str, lxc_root: str) -> str:
    """
    Convert a client-side absolute path to its server-side counterpart.
    e.g. Z:\\Multimedia\\Audio\\Music\\foo.mp3 → /mnt/music/foo.mp3

    If either root is empty (no remap configured), returns the path
    unchanged. This is the normal case when client and server see the
    library at the same path (e.g. both on Linux with a shared mount).
    """
    if not win_root or not lxc_root:
        return win_path
    norm      = win_path.replace("\\", "/")
    norm_root = win_root.replace("\\", "/")
    if norm.startswith(norm_root):
        rel = norm[len(norm_root):].lstrip("/")
        result = f"{lxc_root.rstrip('/')}/{rel}"
        return result
    return norm  # already POSIX or unrecognised — return as-is


def remap_to_win(lxc_path: str, win_root: str, lxc_root: str) -> str:
    """Reverse of remap_to_lxc. No-op if either root is empty."""
    if not win_root or not lxc_root:
        return lxc_path
    lxc_root = lxc_root.rstrip("/")
    if lxc_path.startswith(lxc_root):
        rel = lxc_path[len(lxc_root):].lstrip("/")
        win_rel = rel.replace("/", "\\")
        return f"{win_root.rstrip(chr(92))}\\{win_rel}"
    return lxc_path
