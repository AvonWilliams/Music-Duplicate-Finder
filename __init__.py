# Music Duplicate Finder — __init__.py
# V2.0: two independent detection engines registered side-by-side.
#   - AcoustID / Chromaprint  →  true duplicate detection (same master recording)
#   - CLAP neural embeddings  →  "similar songs" (genre / mood / production)

from picard.plugin3.api import PluginApi

from .actions import (
    # AcoustID family (primary duplicate detection)
    FindDuplicatesAcoustIDAction,
    AcoustIDFilesAction,
    AcoustIDClusterAction,
    AcoustIDAlbumAction,
    # CLAP family (similar-songs clustering)
    FindSimilarSongsClapAction,
    ClapFilesAction,
    ClapClusterAction,
    ClapAlbumAction,
)
from .diag import get_logger, log_file_path, set_picard_api
from .options_page import DuplicateFinderOptionsPage


def enable(api: PluginApi) -> None:
    """Called by Picard when the plugin is enabled."""
    set_picard_api(api)
    log = get_logger("plugin")
    log.info("Music Duplicate Finder v2.0.2 loaded  —  diagnostic log: %s", log_file_path())
    api.logger.info(f"Music Duplicate Finder v2.0.2 loaded — log: {log_file_path()}")

    # ── Thresholds (stored as int 0-100) ───────────────────────────────────
    # CLAP thresholds (legacy; kept for compatibility with V1.9 configs)
    api.plugin_config.register_option("certain_threshold", 95)
    api.plugin_config.register_option("likely_threshold",  85)
    api.plugin_config.register_option("unsure_threshold",  70)

    # AcoustID / Chromaprint thresholds — separate from CLAP because the
    # similarity distributions differ. Chromaprint real-duplicate pairs
    # sit at 0.95+ with a wide empty gap down to ~0.30 for unrelated.
    api.plugin_config.register_option("cp_certain_threshold", 95)
    api.plugin_config.register_option("cp_likely_threshold",  85)
    api.plugin_config.register_option("cp_unsure_threshold",  75)

    # ── CLAP inference mode (unchanged from V1.9) ──────────────────────────
    # "remote" = FastAPI server  |  "local" = CLAP on local GPU
    api.plugin_config.register_option("inference_mode", "remote")

    api.plugin_config.register_option("server_host", "10.0.0.69")
    api.plugin_config.register_option("server_port", 8765)
    api.plugin_config.register_option("api_key", "changeme_homelab_key")

    api.plugin_config.register_option("win_music_root", "")
    api.plugin_config.register_option("lxc_music_root", "")

    api.plugin_config.register_option("local_gpu_index", 0)
    api.plugin_config.register_option("model_cache_dir", "")

    # ── AcoustID / Chromaprint engine settings ─────────────────────────────
    # Alignment window: off / narrow / standard / wide / exhaustive.
    # "standard" = ±20 timesteps (~2.5s of audio) = what AcoustID itself uses.
    api.plugin_config.register_option("cp_alignment", "standard")
    # GPU acceleration for chromaprint comparison (with CPU fallback warning).
    api.plugin_config.register_option("cp_use_gpu", True)

    # ── Register UI ────────────────────────────────────────────────────────
    api.register_options_page(DuplicateFinderOptionsPage)

    # Tools menu: two separate entries, one per engine
    api.register_tools_menu_action(FindDuplicatesAcoustIDAction)
    api.register_tools_menu_action(FindSimilarSongsClapAction)

    # Right-click context menus: two entries per container type (files,
    # clusters, albums) — one per engine
    api.register_file_action(AcoustIDFilesAction)
    api.register_file_action(ClapFilesAction)

    api.register_cluster_action(AcoustIDClusterAction)
    api.register_cluster_action(ClapClusterAction)

    api.register_album_action(AcoustIDAlbumAction)
    api.register_album_action(ClapAlbumAction)


def disable() -> None:
    """Called by Picard when the plugin is disabled."""
    pass
