"""Reusable "histogram plots + plot options" panel.

Shared by the Diagnostics tab and the Monitored Acquisition tab so both get
identical plots, zoom/focus behavior, log-scale toggles, gamma control, and
bin/range settings without duplicating the code.
"""
import tkinter as tk
from tkinter import ttk

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.widgets import RectangleSelector

try:
    import cmasher as cmr
except Exception:
    cmr = None

from tpx_processing import hist_args, make_counts_per_pixel_hist, make_hist_1d

PLOT_SPECS = [
    ("pixel", "Pixel histogram"),
    ("cluster", "Cluster histogram"),
    ("counts", "Counts / pixel"),
    ("itof", "i-ToF histogram"),
    ("cluster_t", "Cluster t histogram"),
    ("etof", "e-ToF histogram"),
]
PLOT_LABELS = {key: label for key, label in PLOT_SPECS}
PLOT_KEYS_BY_LABEL = {label: key for key, label in PLOT_SPECS}

# "counts" uses max <= min to mean "scale to the data", and bins <= 0 to mean
# "one bin per integer count value, scaled to the range".
DEFAULT_HIST_SETTINGS = {
    "pixel": {"bins": 256, "min": 0.0, "max": 256.0},
    "cluster": {"bins": 256, "min": 0.0, "max": 256.0},
    "counts": {"bins": 0, "min": 0.0, "max": 0.0},
    "itof": {"bins": 2000, "min": 0.0, "max": 20000.0},
    "cluster_t": {"bins": 1000, "min": 0.0, "max": 1000.0},
    "etof": {"bins": 2000, "min": 0.0, "max": 1000.0},
}


def apply_log_scale(ax, counts, enabled: bool):
    if not enabled:
        ax.set_yscale("linear")
        ax.set_ylim(*_auto_hist_ylim(counts))
        return
    positive = counts[counts > 0]
    if positive.size == 0:
        ax.set_yscale("linear")
        ax.set_ylim(*_auto_hist_ylim(counts))
        return
    min_val = float(positive.min())
    max_val = float(positive.max())
    min_val = max(min_val, 1e-3)
    max_val = max(max_val, min_val * 1.1)
    ax.set_yscale("log")
    ax.set_ylim(min_val, max_val * 1.1)


def _auto_hist_ylim(counts):
    max_val = counts.max() if len(counts) else 0
    if max_val <= 0:
        max_val = 1
    return 0, 1.1 * max_val


def build_log_norm(hist_2d):
    if hist_2d is None or hist_2d.size == 0:
        return None
    positive = hist_2d[hist_2d > 0]
    if positive.size == 0:
        return None
    vmin = max(float(positive.min()), 1e-3)
    vmax = max(float(positive.max()), vmin * 1.1)
    return LogNorm(vmin=vmin, vmax=vmax)


def build_power_norm(hist_2d, gamma: float):
    """Gamma correction for the linear-scale 2D maps. gamma == 1.0 is plain
    linear scaling, so we skip PowerNorm entirely in that common case."""
    if abs(gamma - 1.0) < 1e-6:
        return None
    if hist_2d is None or hist_2d.size == 0:
        return None
    vmax = float(hist_2d.max())
    if vmax <= 0:
        return None
    return PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)


class HistogramPlotPanel(ttk.Frame):
    """The 6-plot grid (pixel/cluster/counts/itof/cluster_t/etof), its zoom
    and single-plot-focus behavior, log-scale + gamma controls, and the
    bins/bounds settings table.

    The host widget owns the actual data (accumulation, processing) and
    feeds it in via `update_plots(data)`. `data` is a dict with keys
    pixel_hist, cluster_hist, itof_hist, etof_hist, cluster_t_hist, and
    optionally settings (defaults to the panel's own current settings).
    """

    def __init__(self, parent, on_settings_changed=None, plots_first=False, shared_vars=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_settings_changed = on_settings_changed
        self._plots_first = plots_first
        self._last_data = None

        # `shared_vars` (see shared_state.make_plot_shared_vars) lets two
        # panels -- e.g. Diagnostics and Monitored Acquisition -- use the
        # exact same Tk variables, so editing settings in one updates the
        # other live. Without it, each panel gets its own (unshared,
        # unpersisted) variables like before.
        if shared_vars is not None:
            self.log_pixel_var = shared_vars["log_pixel_var"]
            self.log_cluster_var = shared_vars["log_cluster_var"]
            self.log_counts_var = shared_vars["log_counts_var"]
            self.log_itof_var = shared_vars["log_itof_var"]
            self.log_cluster_t_var = shared_vars["log_cluster_t_var"]
            self.log_etof_var = shared_vars["log_etof_var"]
            self.gamma_var = shared_vars["gamma_var"]
            self.gamma_label_var = shared_vars["gamma_label_var"]
            self._hist_vars = shared_vars["hist_vars"]
        else:
            self.log_pixel_var = tk.BooleanVar(self, value=False)
            self.log_cluster_var = tk.BooleanVar(self, value=False)
            self.log_counts_var = tk.BooleanVar(self, value=False)
            self.log_itof_var = tk.BooleanVar(self, value=False)
            self.log_cluster_t_var = tk.BooleanVar(self, value=False)
            self.log_etof_var = tk.BooleanVar(self, value=False)
            self.gamma_var = tk.DoubleVar(self, value=1.0)
            self.gamma_label_var = tk.StringVar(self, value="1.00")
            self._hist_vars = {
                key: {
                    "bins": tk.StringVar(self, value=str(int(cfg["bins"]))),
                    "min": tk.StringVar(self, value=f"{cfg['min']:g}"),
                    "max": tk.StringVar(self, value=f"{cfg['max']:g}"),
                }
                for key, cfg in DEFAULT_HIST_SETTINGS.items()
            }

        # Base to fall back on for any field that fails to parse below, then
        # immediately replaced with whatever the (possibly shared/persisted)
        # variables actually hold.
        self._hist_settings = {key: dict(cfg) for key, cfg in DEFAULT_HIST_SETTINGS.items()}
        self._hist_settings = self._read_hist_settings()

        self.focus_choice_var = tk.StringVar(self, value=PLOT_SPECS[0][1])
        self.focus_button_var = tk.StringVar(self, value="Focus")
        self._focus_key = None

        self._zoom_selectors = []
        self._base_limits = {}
        self._axes = {}
        self._press_cid = None

        self._build_ui()
        self._wire_hist_traces()
        self.update_plots(self.empty_plot_data())

    # ---- public API -----------------------------------------------------

    def hist_snapshot(self):
        return {key: dict(cfg) for key, cfg in self._hist_settings.items()}

    def empty_plot_data(self):
        settings = self.hist_snapshot()
        pixel_bins = int(settings["pixel"]["bins"])
        cluster_bins = int(settings["cluster"]["bins"])
        return {
            "pixel_hist": np.zeros((pixel_bins, pixel_bins), dtype=np.float64),
            "cluster_hist": np.zeros((cluster_bins, cluster_bins), dtype=np.float64),
            "itof_hist": make_hist_1d([], **hist_args(settings["itof"])),
            "etof_hist": make_hist_1d([], **hist_args(settings["etof"])),
            "cluster_t_hist": make_hist_1d([], **hist_args(settings["cluster_t"])),
            "settings": settings,
        }

    def update_plots(self, data):
        self._last_data = data
        settings = data.get("settings", self.hist_snapshot())

        for ax in self._axes.values():
            ax.clear()

        self._draw_map("pixel", data["pixel_hist"], settings["pixel"], self.log_pixel_var.get())
        self._draw_map("cluster", data["cluster_hist"], settings["cluster"], self.log_cluster_var.get())
        self._draw_counts(data["pixel_hist"])
        self._draw_line("itof", data["itof_hist"], "Time (ns)", "Counts", self.log_itof_var.get())
        self._draw_line("cluster_t", data["cluster_t_hist"], "Time (ns)", "Counts", self.log_cluster_t_var.get())
        self._draw_line("etof", data["etof_hist"], "Time (ns)", "Counts", self.log_etof_var.get())

        self._canvas.draw_idle()

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=1)

        self._opts_bar = ttk.Frame(self)
        self._build_opts_bar(self._opts_bar)

        self._hist_frame = ttk.LabelFrame(self, text="Histogram bins / bounds")
        self._build_hist_frame(self._hist_frame)

        self._plot_frame = ttk.Frame(self)
        self._plot_frame.rowconfigure(0, weight=1)
        self._plot_frame.columnconfigure(0, weight=1)
        self._figure = Figure(figsize=(8, 6), tight_layout=True)
        self._canvas = FigureCanvasTkAgg(self._figure, master=self._plot_frame)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._build_axes()

        if self._plots_first:
            self._plot_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
            self._opts_bar.grid(row=1, column=0, sticky="w", pady=(0, 6))
            self._hist_frame.grid(row=2, column=0, sticky="ew")
            self.rowconfigure(0, weight=1)
        else:
            self._opts_bar.grid(row=0, column=0, sticky="w", pady=(0, 6))
            self._hist_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
            self._plot_frame.grid(row=2, column=0, sticky="nsew")
            self.rowconfigure(2, weight=1)

    def _build_opts_bar(self, plot_opts):
        ttk.Label(plot_opts, text="Log scale:").grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(plot_opts, text="Pixel", variable=self.log_pixel_var, command=self._on_log_toggle).grid(
            row=0, column=1, padx=(6, 0)
        )
        ttk.Checkbutton(plot_opts, text="Cluster", variable=self.log_cluster_var, command=self._on_log_toggle).grid(
            row=0, column=2, padx=(6, 0)
        )
        ttk.Checkbutton(plot_opts, text="Counts/pixel", variable=self.log_counts_var, command=self._on_log_toggle).grid(
            row=0, column=3, padx=(6, 0)
        )
        ttk.Checkbutton(plot_opts, text="i-ToF", variable=self.log_itof_var, command=self._on_log_toggle).grid(
            row=0, column=4, padx=(6, 0)
        )
        ttk.Checkbutton(plot_opts, text="Cluster t", variable=self.log_cluster_t_var, command=self._on_log_toggle).grid(
            row=0, column=5, padx=(6, 0)
        )
        ttk.Checkbutton(plot_opts, text="e-ToF", variable=self.log_etof_var, command=self._on_log_toggle).grid(
            row=0, column=6, padx=(6, 0)
        )

        ttk.Separator(plot_opts, orient="vertical").grid(row=0, column=7, sticky="ns", padx=12)
        ttk.Label(plot_opts, text="Single plot:").grid(row=0, column=8, sticky="w")
        focus_box = ttk.Combobox(
            plot_opts,
            textvariable=self.focus_choice_var,
            values=[label for _, label in PLOT_SPECS],
            state="readonly",
            width=18,
        )
        focus_box.grid(row=0, column=9, padx=(6, 6))
        focus_box.bind("<<ComboboxSelected>>", self._on_focus_choice)
        ttk.Button(plot_opts, textvariable=self.focus_button_var, command=self._toggle_focus).grid(row=0, column=10)
        ttk.Label(plot_opts, text="(double-click a plot to focus / restore)").grid(row=0, column=11, padx=(10, 0))

        ttk.Label(plot_opts, text="Gamma (pixel/cluster maps):").grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        gamma_scale = ttk.Scale(
            plot_opts,
            from_=0.1,
            to=3.0,
            orient="horizontal",
            variable=self.gamma_var,
            length=160,
            command=self._on_gamma_change,
        )
        gamma_scale.grid(row=1, column=2, columnspan=3, sticky="w", padx=(6, 6), pady=(6, 0))
        ttk.Label(plot_opts, textvariable=self.gamma_label_var, width=5).grid(row=1, column=5, sticky="w", pady=(6, 0))
        ttk.Button(plot_opts, text="Reset", command=self._reset_gamma).grid(row=1, column=6, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Label(
            plot_opts, text="(1.0 = linear; applies when log scale is off)"
        ).grid(row=1, column=7, columnspan=4, sticky="w", padx=(10, 0), pady=(6, 0))

    def _build_hist_frame(self, hist_frame):
        groups = [PLOT_SPECS[:3], PLOT_SPECS[3:]]
        for group_idx, group in enumerate(groups):
            base_col = group_idx * 5
            ttk.Label(hist_frame, text="Plot").grid(row=0, column=base_col, sticky="w", padx=(6, 4), pady=(4, 2))
            for offset, header in enumerate(("Bins", "Min", "Max"), start=1):
                ttk.Label(hist_frame, text=header).grid(row=0, column=base_col + offset, pady=(4, 2))
            for row_idx, (key, label) in enumerate(group, start=1):
                ttk.Label(hist_frame, text=label).grid(row=row_idx, column=base_col, sticky="w", padx=(6, 4), pady=1)
                for offset, field in enumerate(("bins", "min", "max"), start=1):
                    ttk.Entry(hist_frame, textvariable=self._hist_vars[key][field], width=9).grid(
                        row=row_idx, column=base_col + offset, padx=2, pady=1
                    )
            hist_frame.columnconfigure(base_col + 4, minsize=20)

        ttk.Label(
            hist_frame,
            text="Pixel/cluster bounds apply to both axes. Counts/pixel max <= min autoscales the range, "
                 "and bins <= 0 autoscales the bin count to match it. "
                 "Cluster t bounds also gate which clusters enter the cluster map. "
                 "Changes take effect on the next frame.",
        ).grid(row=4, column=0, columnspan=10, sticky="w", padx=6, pady=(4, 4))

    def _build_axes(self):
        self._figure.clear()
        self._axes = {}
        self._base_limits = {}

        if self._focus_key is not None:
            self._axes[self._focus_key] = self._figure.add_subplot(1, 1, 1)
        else:
            grid = self._figure.subplots(2, 3)
            for idx, (key, _) in enumerate(PLOT_SPECS):
                self._axes[key] = grid[idx // 3, idx % 3]

        self._wire_zoom()

    def _wire_zoom(self):
        self._zoom_selectors = []
        for ax in self._axes.values():
            selector = RectangleSelector(
                ax,
                self._on_zoom_select,
                useblit=True,
                button=[1],
                interactive=False,
            )
            self._zoom_selectors.append(selector)

        if self._press_cid is not None:
            self._canvas.mpl_disconnect(self._press_cid)
        self._press_cid = self._canvas.mpl_connect("button_press_event", self._on_button_press)

    def _on_zoom_select(self, eclick, erelease):
        ax = eclick.inaxes
        if ax is None or erelease.inaxes != ax:
            return
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if x0 is None or x1 is None or y0 is None or y1 is None:
            return
        if abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6:
            return
        ax.set_xlim(min(x0, x1), max(x0, x1))
        ax.set_ylim(min(y0, y1), max(y0, y1))
        self._canvas.draw_idle()

    def _on_button_press(self, event):
        ax = event.inaxes
        if ax is None:
            return

        if getattr(event, "dblclick", False) and event.button == 1:
            key = self._key_for_axis(ax)
            if key is not None:
                self._set_focus(None if self._focus_key is not None else key)
            return

        if event.button != 3:
            return
        limits = self._base_limits.get(ax)
        if limits is None:
            return
        xlim, ylim = limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self._canvas.draw_idle()

    def _key_for_axis(self, ax):
        for key, candidate in self._axes.items():
            if candidate is ax:
                return key
        return None

    def _toggle_focus(self):
        if self._focus_key is not None:
            self._set_focus(None)
        else:
            self._set_focus(PLOT_KEYS_BY_LABEL.get(self.focus_choice_var.get()))

    def _on_focus_choice(self, *_):
        if self._focus_key is not None:
            self._set_focus(PLOT_KEYS_BY_LABEL.get(self.focus_choice_var.get()))

    def _set_focus(self, key):
        self._focus_key = key
        if key is not None:
            self.focus_choice_var.set(PLOT_LABELS[key])
            self.focus_button_var.set("Show all")
        else:
            self.focus_button_var.set("Focus")
        self._build_axes()
        if self._last_data is not None:
            self.update_plots(self._last_data)

    def _on_log_toggle(self):
        if self._last_data is not None:
            self.update_plots(self._last_data)

    def _on_gamma_change(self, _value=None):
        self.gamma_label_var.set(f"{self.gamma_var.get():.2f}")
        if self._last_data is not None:
            self.update_plots(self._last_data)

    def _reset_gamma(self):
        self.gamma_var.set(1.0)
        self._on_gamma_change()

    def _wire_hist_traces(self):
        for key, _ in PLOT_SPECS:
            for field in ("bins", "min", "max"):
                self._hist_vars[key][field].trace_add("write", self._on_hist_setting_change)

    def _read_hist_settings(self):
        settings = {}
        for key, _ in PLOT_SPECS:
            cfg = dict(self._hist_settings.get(key, DEFAULT_HIST_SETTINGS[key]))
            entries = self._hist_vars[key]
            try:
                bins = int(float(entries["bins"].get()))
                if bins > 0:
                    cfg["bins"] = min(bins, 8192)
                elif key == "counts":
                    # "counts" treats bins <= 0 as "autoscale the bin count".
                    cfg["bins"] = 0
            except ValueError:
                pass
            try:
                low = float(entries["min"].get())
                high = float(entries["max"].get())
                if key == "counts" or high > low:
                    cfg["min"] = low
                    cfg["max"] = high
            except ValueError:
                pass
            settings[key] = cfg
        return settings

    def _on_hist_setting_change(self, *_):
        new_settings = self._read_hist_settings()
        if new_settings == self._hist_settings:
            return
        self._hist_settings = new_settings
        if self._on_settings_changed is not None:
            self._on_settings_changed()
        elif self._last_data is not None:
            self.update_plots(self._last_data)

    # ---- drawing ----------------------------------------------------------

    def _draw_map(self, key, hist, cfg, log_enabled):
        ax = self._axes.get(key)
        if ax is None:
            return
        low = float(cfg["min"])
        high = float(cfg["max"])
        cmap = cmr.rainforest if cmr is not None else "viridis"
        norm = build_log_norm(hist) if log_enabled else build_power_norm(hist, self.gamma_var.get())
        ax.set_title(PLOT_LABELS[key])
        ax.imshow(
            hist.T,
            origin="lower",
            extent=(low, high, low, high),
            cmap=cmap,
            norm=norm,
        )
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        self._base_limits[ax] = (ax.get_xlim(), ax.get_ylim())

    def _draw_counts(self, pixel_hist):
        ax = self._axes.get("counts")
        if ax is None:
            return
        cfg = self._hist_settings["counts"]
        counts_hist = make_counts_per_pixel_hist(
            pixel_hist,
            bins=int(cfg["bins"]),
            range_min=float(cfg["min"]),
            range_max=float(cfg["max"]),
        )
        ax.set_title(PLOT_LABELS["counts"])
        ax.plot(counts_hist["bins"], counts_hist["counts"])
        ax.set_xlim(counts_hist["range"])
        apply_log_scale(ax, counts_hist["counts"], self.log_counts_var.get())
        ax.set_xlabel("Counts per pixel")
        ax.set_ylabel("Pixels")
        self._base_limits[ax] = (ax.get_xlim(), ax.get_ylim())

    def _draw_line(self, key, hist, xlabel, ylabel, log_enabled):
        ax = self._axes.get(key)
        if ax is None:
            return
        ax.set_title(PLOT_LABELS[key])
        ax.plot(hist["bins"], hist["counts"])
        ax.set_xlim(hist["range"])
        apply_log_scale(ax, hist["counts"], log_enabled)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self._base_limits[ax] = (ax.get_xlim(), ax.get_ylim())
