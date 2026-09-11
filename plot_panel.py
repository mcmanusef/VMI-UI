"""Reusable "histogram plots + plot options" panel.

Shared by the Diagnostics tab and the Monitored Acquisition tab so both get
identical plots, zoom/focus behavior, log-scale toggles, gamma control, and
bin/range settings without duplicating the code.
"""
import qtk as tk
from qtk import grid_into, ttk

import numpy as np

import pyqtgraph as pg
from PyQt5 import QtCore

from qt_plots import hist_to_rgba, ZoomFocusViewBox

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

        # Populated by _build_axes(): PlotItems, and either an ImageItem
        # (pixel/cluster) or a PlotDataItem (everything else) per key.
        self._plots = {}
        self._images = {}
        self._curves = {}

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

        self._draw_map("pixel", data["pixel_hist"], settings["pixel"], self.log_pixel_var.get())
        self._draw_map("cluster", data["cluster_hist"], settings["cluster"], self.log_cluster_var.get())
        self._draw_counts(data["pixel_hist"])
        self._draw_line("itof", data["itof_hist"], "Time (ns)", "Counts", self.log_itof_var.get())
        self._draw_line("cluster_t", data["cluster_t_hist"], "Time (ns)", "Counts", self.log_cluster_t_var.get())
        self._draw_line("etof", data["etof_hist"], "Time (ns)", "Counts", self.log_etof_var.get())

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
        self._glw = pg.GraphicsLayoutWidget()
        grid_into(self._glw, self._plot_frame, row=0, column=0, sticky="nsew")
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
        self._glw.clear()
        self._plots = {}
        self._images = {}
        self._curves = {}

        if self._focus_key is not None:
            layout = [(self._focus_key, PLOT_LABELS[self._focus_key], 0, 0)]
        else:
            layout = [(key, label, idx // 3, idx % 3) for idx, (key, label) in enumerate(PLOT_SPECS)]

        for key, label, row, col in layout:
            vb = ZoomFocusViewBox(on_focus=lambda k=key: self._on_plot_double_click(k))
            plot_item = self._glw.addPlot(row=row, col=col, viewBox=vb)
            plot_item.setTitle(label)
            self._plots[key] = plot_item
            if key in ("pixel", "cluster"):
                img = pg.ImageItem()
                plot_item.addItem(img)
                vb.setAspectLocked(True)
                self._images[key] = img
            else:
                self._curves[key] = plot_item.plot([], [])

    def _on_plot_double_click(self, key):
        self._set_focus(None if self._focus_key is not None else key)

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
        img = self._images.get(key)
        plot_item = self._plots.get(key)
        if img is None or plot_item is None:
            return
        low = float(cfg["min"])
        high = float(cfg["max"])
        if high <= low:
            high = low + 1.0
        rgba = hist_to_rgba(hist, log=log_enabled, gamma=self.gamma_var.get())
        img.setImage(rgba, autoLevels=False)
        img.setRect(QtCore.QRectF(low, low, high - low, high - low))
        plot_item.setLabel("bottom", "X")
        plot_item.setLabel("left", "Y")
        plot_item.getViewBox().setRange(xRange=(low, high), yRange=(low, high), padding=0)

    def _draw_counts(self, pixel_hist):
        curve = self._curves.get("counts")
        plot_item = self._plots.get("counts")
        if curve is None or plot_item is None:
            return
        cfg = self._hist_settings["counts"]
        counts_hist = make_counts_per_pixel_hist(
            pixel_hist,
            bins=int(cfg["bins"]),
            range_min=float(cfg["min"]),
            range_max=float(cfg["max"]),
        )
        plot_item.setLogMode(y=self.log_counts_var.get())
        curve.setData(counts_hist["bins"], counts_hist["counts"])
        plot_item.setLabel("bottom", "Counts per pixel")
        plot_item.setLabel("left", "Pixels")
        plot_item.enableAutoRange(axis="y")
        plot_item.setXRange(*counts_hist["range"], padding=0)

    def _draw_line(self, key, hist, xlabel, ylabel, log_enabled):
        curve = self._curves.get(key)
        plot_item = self._plots.get(key)
        if curve is None or plot_item is None:
            return
        plot_item.setLogMode(y=log_enabled)
        curve.setData(hist["bins"], hist["counts"])
        plot_item.setLabel("bottom", xlabel)
        plot_item.setLabel("left", ylabel)
        plot_item.enableAutoRange(axis="y")
        plot_item.setXRange(*hist["range"], padding=0)
