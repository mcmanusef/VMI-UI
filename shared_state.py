"""Live-shared Tk variable bundles for fields that should be the *same*
control across multiple tabs -- not just "same default on next launch", but
literally the same Tk variable, so editing it in one tab updates every tab
that shares it immediately. Each variable is also persisted (via
app_settings) so the shared value survives a restart too.

- make_plot_shared_vars(): the histogram/plot settings HistogramPlotPanel
  needs (log toggles, gamma, per-plot bins/min/max) -- shared between the
  Diagnostics and Monitored Acquisition tabs.
- make_acquisition_shared_vars(): frame time / run duration / save folder /
  metadata -- shared between the Collection parameters and Monitored
  Acquisition tabs (and the metadata subset is reused by the Parameter
  Sweep tab too).

Notes text can't be wired to a Tk variable the way Entry/Scale fields can
(tk.Text has no textvariable), so it's handled separately: load_notes() /
save_notes() persist the last text, but each tab keeps its own Text widget
rather than being live-mirrored keystroke for keystroke.
"""
import datetime
import tkinter as tk

import app_settings
from plot_panel import DEFAULT_HIST_SETTINGS

_LOG_KEYS = (
    "log_pixel_var",
    "log_cluster_var",
    "log_counts_var",
    "log_itof_var",
    "log_cluster_t_var",
    "log_etof_var",
)


def make_plot_shared_vars(master):
    shared = {
        name: app_settings.persistent_var(master, tk.BooleanVar, f"plot.{name}", False)
        for name in _LOG_KEYS
    }
    shared["gamma_var"] = app_settings.persistent_var(master, tk.DoubleVar, "plot.gamma", 1.0)
    shared["gamma_label_var"] = tk.StringVar(master, value=f"{shared['gamma_var'].get():.2f}")

    hist_vars = {}
    for key, cfg in DEFAULT_HIST_SETTINGS.items():
        hist_vars[key] = {
            "bins": app_settings.persistent_var(master, tk.StringVar, f"plot.{key}.bins", str(int(cfg["bins"]))),
            "min": app_settings.persistent_var(master, tk.StringVar, f"plot.{key}.min", f"{cfg['min']:g}"),
            "max": app_settings.persistent_var(master, tk.StringVar, f"plot.{key}.max", f"{cfg['max']:g}"),
        }
    shared["hist_vars"] = hist_vars
    return shared


def _default_shared_save_folder():
    return rf"C:\DATA\{datetime.date.today().strftime('%Y%m%d')}\acquisition"


def make_acquisition_shared_vars(master):
    return {
        "frame_time_var": app_settings.persistent_var(master, tk.StringVar, "acq.frame_time", "1"),
        # 0 means "run until stopped" on Monitored Acquisition, but Collection
        # parameters requires a positive duration -- default to a value valid
        # for both rather than either tab's own default.
        "run_duration_var": app_settings.persistent_var(master, tk.StringVar, "acq.run_duration", "10"),
        "save_folder_var": app_settings.persistent_var(
            master, tk.StringVar, "acq.save_folder", _default_shared_save_folder()
        ),
        "target_var": app_settings.persistent_var(master, tk.StringVar, "acq.target", ""),
        "target_pressure_var": app_settings.persistent_var(master, tk.StringVar, "acq.target_pressure", ""),
        "background_pressure_var": app_settings.persistent_var(master, tk.StringVar, "acq.background_pressure", ""),
        "power_var": app_settings.persistent_var(master, tk.StringVar, "acq.power", ""),
        "spot_size_var": app_settings.persistent_var(master, tk.StringVar, "acq.spot_size", ""),
        "polarization_var": app_settings.persistent_var(master, tk.StringVar, "acq.polarization", ""),
        "wavelength_var": app_settings.persistent_var(master, tk.StringVar, "acq.wavelength", ""),
    }


def load_notes():
    return app_settings.get("acq.notes", "")


def save_notes(text):
    app_settings.set("acq.notes", text)


def wire_notes_widget(text_widget):
    """Load the persisted notes into `text_widget` and save back to
    app_settings whenever it changes (debounced via <FocusOut> and a Modified
    flag so we don't write on every keystroke)."""
    text_widget.insert("1.0", load_notes())

    def _on_modified(_event=None):
        if text_widget.edit_modified():
            save_notes(text_widget.get("1.0", "end-1c"))
            text_widget.edit_modified(False)

    text_widget.bind("<FocusOut>", _on_modified)
