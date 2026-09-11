"""Persistent defaults for user-editable fields, shared across tabs.

A tiny JSON store next to the source (same convention as
timewalk.DEFAULT_CORRECTION_PATH), so any tab can persist a field's last
value across restarts with a couple of lines, and so "similar" fields on
different tabs can be pointed at the same key to share a default.

Usage:
    import app_settings

    value = app_settings.get("xps_zero_position", 0.0)
    app_settings.set("xps_zero_position", 12.5)

Or, to back a Tk variable with a persisted value (loads the last value on
creation, and saves on every change):

    var = app_settings.persistent_var(parent, tk.StringVar, "frame_time", "1")
"""
import json
import pathlib

SETTINGS_PATH = pathlib.Path(__file__).with_name("app_settings.json")

_cache = None


def _load_from_disk():
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _cache_ref():
    global _cache
    if _cache is None:
        _cache = _load_from_disk()
    return _cache


def _write_to_disk():
    try:
        SETTINGS_PATH.write_text(json.dumps(_cache_ref(), indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def get(key, default=None):
    return _cache_ref().get(key, default)


def set(key, value):
    _cache_ref()[key] = value
    _write_to_disk()


def update(mapping):
    _cache_ref().update(mapping)
    _write_to_disk()


def persistent_var(master, var_class, key, default):
    """A Tk variable pre-populated from the persisted value for `key`
    (falling back to `default`), that writes back to the settings file on
    every change."""
    var = var_class(master, value=get(key, default))

    def _on_write(*_):
        set(key, var.get())

    var.trace_add("write", _on_write)
    return var
