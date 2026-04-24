# Music Duplicate Finder — config_util.py
# Small compatibility shim for Picard's plugin_config.
#
# Some V3 builds expose ConfigSection.get(key, default); others only
# support subscript access cfg[key] (which returns the registered default
# if nothing has been saved yet, or raises KeyError if unregistered).
# This helper tries both, so the plugin works on either implementation.

def cfg_get(cfg, key, default=None):
    """Safely read a plugin-config value with a fallback default."""
    try:
        return cfg.get(key, default)          # newer V3 API
    except AttributeError:
        pass
    try:
        return cfg[key]                       # older V3 API (subscript)
    except (KeyError, AttributeError):
        return default
