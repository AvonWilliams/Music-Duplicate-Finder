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
    # Fast fingerprint generation
    GenerateFingerprintsFilesAction,
    GenerateFingerprintsClusterAction,
    GenerateFingerprintsAlbumAction,
    # Saved results
    LoadResultsAction,
)
from .diag import get_logger, log_file_path, set_picard_api
from .options_page import DuplicateFinderOptionsPage


def enable(api: PluginApi) -> None:
    """Called by Picard when the plugin is enabled."""
    set_picard_api(api)
    log = get_logger("plugin")
    log.info("Music Duplicate Finder v2.3.12 loaded  —  diagnostic log: %s", log_file_path())
    api.logger.info(f"Music Duplicate Finder v2.3.12 loaded — log: {log_file_path()}")

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

    api.plugin_config.register_option("server_host", "10.0.0.1")
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

    # ── Size anomaly detection thresholds ─────────────────────────────────
    api.plugin_config.register_option("anomaly_size_multiplier", 1.7)
    api.plugin_config.register_option("anomaly_size_mb", 7.5)

    # ── CuPy install hint (logged once at load if CUDA is available but CuPy isn't) ──
    try:
        import torch
        if torch.cuda.is_available():
            try:
                import cupy  # noqa: F401
            except ModuleNotFoundError:
                from .chromaprint_engine import _cupy_install_hint
                hint = _cupy_install_hint()
                log.info(
                    "CuPy not found — AcoustID scan will use the slower PyTorch path. "
                    "Install CuPy for the fast fused kernel:  %s", hint,
                )
                api.logger.info(
                    f"Music Duplicate Finder: CuPy not found — AcoustID scans will be slower. "
                    f"Install CuPy to speed them up:  {hint}"
                )
    except Exception:  # noqa: BLE001
        pass

    # ── Register UI ────────────────────────────────────────────────────────
    api.register_options_page(DuplicateFinderOptionsPage)

    # Tools menu: two separate entries, one per engine
    api.register_tools_menu_action(FindDuplicatesAcoustIDAction)
    api.register_tools_menu_action(FindSimilarSongsClapAction)
    api.register_tools_menu_action(LoadResultsAction)

    # Right-click context menus: two entries per container type (files,
    # clusters, albums) — one per engine
    api.register_file_action(AcoustIDFilesAction)
    api.register_file_action(ClapFilesAction)
    api.register_file_action(GenerateFingerprintsFilesAction)

    api.register_cluster_action(AcoustIDClusterAction)
    api.register_cluster_action(ClapClusterAction)
    api.register_cluster_action(GenerateFingerprintsClusterAction)

    api.register_album_action(AcoustIDAlbumAction)
    api.register_album_action(ClapAlbumAction)
    api.register_album_action(GenerateFingerprintsAlbumAction)

    # ── Keyboard shortcut: Ctrl+Shift+D → full AcoustID scan ──────────────
    from PyQt6.QtGui import QKeySequence, QShortcut
    _shortcut = QShortcut(QKeySequence("Ctrl+Shift+D"), api.tagger.window)
    _shortcut.activated.connect(lambda: FindDuplicatesAcoustIDAction().callback([]))


def disable() -> None:
    """Called by Picard when the plugin is disabled."""
    pass
