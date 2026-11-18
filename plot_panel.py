"""Reusable "histogram plots + plot options" panel.

Shared by the Diagnostics tab and the Monitored Acquisition tab so both get
identical plots, zoom/focus behavior, log-scale toggles, gamma control, and
bin/range settings without duplicating the code.
"""
import qtk as tk
from qtk import grid_into, ttk

import numpy as np

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui

import chip_layout
from collapsible_frame import CollapsibleFrame, set_background
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
# Optional (etof_correlation=True): cluster position against e-ToF time,
# every cluster paired with every e-ToF hit of its pulse.
CORRELATION_SPECS = [
    ("x_etof", "Cluster x vs e-ToF"),
    ("y_etof", "Cluster y vs e-ToF"),
]
CORRELATION_KEYS = tuple(key for key, _ in CORRELATION_SPECS)
PLOT_LABELS = {key: label for key, label in PLOT_SPECS + CORRELATION_SPECS}
PLOT_KEYS_BY_LABEL = {label: key for key, label in PLOT_SPECS + CORRELATION_SPECS}
# The hist settings key for the correlation maps (raw_conversion's
# ETOF_CORRELATION_KEY): one bin count for both axes and the e-ToF time
# range; the position range is the cluster histogram's.
ETOF_CORR_KEY = "etof_corr"

# Pixel/cluster maps span the whole detector canvas (512 px on a quad).
_DETECTOR_PX = chip_layout.get_active_layout().size

# "counts" uses max <= min to mean "scale to the data", and bins <= 0 to mean
# "one bin per integer count value, scaled to the range".
DEFAULT_HIST_SETTINGS = {
    "pixel": {"bins": _DETECTOR_PX, "min": 0.0, "max": float(_DETECTOR_PX)},
    "cluster": {"bins": _DETECTOR_PX, "min": 0.0, "max": float(_DETECTOR_PX)},
    "counts": {"bins": 0, "min": 0.0, "max": 0.0},
    "itof": {"bins": 2000, "min": 0.0, "max": 20000.0},
    "cluster_t": {"bins": 1000, "min": 0.0, "max": 1000.0},
    "etof": {"bins": 2000, "min": 0.0, "max": 1000.0},
    ETOF_CORR_KEY: {"bins": 256, "min": 0.0, "max": 1000.0},
}
# Rotation of the cluster positions before the cluster map and the x / y vs
# e-ToF maps are binned: angle (degrees, counter-clockwise) about the point
# (cx, cy) px. Always in hist_snapshot() under ROTATION_KEY. The pixel map
# stays in detector coordinates (it is what pixels are masked from).
ROTATION_KEY = "rotation"
ROTATION_DEFAULTS = {"angle": 0.0, "cx": _DETECTOR_PX / 2, "cy": _DETECTOR_PX / 2}


def read_rotation(rotation_vars, previous=None):
    """{"angle", "cx", "cy"} from the rotation fields; a field that doesn't
    parse keeps its previous value (or the default)."""
    rotation = dict(previous or ROTATION_DEFAULTS)
    for field in ROTATION_DEFAULTS:
        try:
            rotation[field] = float(rotation_vars[field].get())
        except (ValueError, KeyError):
            pass
    return rotation


def build_rotation_row(parent, rotation_vars, what):
    """A frame with the "Rotate <what> x / y by [angle] ° about x [cx] y [cy]"
    fields, for the shared rotation variables."""
    row = ttk.Frame(parent)
    ttk.Label(row, text=f"Rotate {what} x / y by").grid(row=0, column=0, sticky="w")
    ttk.Entry(row, textvariable=rotation_vars["angle"], width=7).grid(row=0, column=1, padx=(6, 2))
    ttk.Label(row, text="° about x").grid(row=0, column=2)
    ttk.Entry(row, textvariable=rotation_vars["cx"], width=7).grid(row=0, column=3, padx=(6, 2))
    ttk.Label(row, text="y").grid(row=0, column=4, padx=(4, 0))
    ttk.Entry(row, textvariable=rotation_vars["cy"], width=7).grid(row=0, column=5, padx=(6, 2))
    ttk.Label(row, text="px (counter-clockwise; same on every tab)").grid(row=0, column=6, padx=(4, 0))
    return row


class HistogramPlotPanel(ttk.Frame):
    """The 6-plot grid (pixel/cluster/counts/itof/cluster_t/etof), its zoom
    and single-plot-focus behavior, log-scale + gamma controls, and the
    bins/bounds settings table.

    The host widget owns the actual data (accumulation, processing) and
    feeds it in via `update_plots(data)`. `data` is a dict with keys
    pixel_hist, cluster_hist, itof_hist, etof_hist, cluster_t_hist, and
    optionally settings (defaults to the panel's own current settings).

    etof_correlation=True adds a "Show x / y vs e-ToF" option: two more maps
    (data keys x_etof_hist / y_etof_hist), drawn while it is on. While it is
    on, hist_snapshot() carries an ETOF_CORR_KEY entry, which is what asks
    raw_conversion.process_raw_file to compute them.
    """

    def __init__(self, parent, on_settings_changed=None, plots_first=False, shared_vars=None,
                 empty_text="No data — press Start to acquire", etof_correlation=False, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_settings_changed = on_settings_changed
        self._corr_enabled = bool(etof_correlation)
        self._plots_first = plots_first
        self._empty_text = empty_text
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
            self.log_etof_corr_var = shared_vars.get("log_etof_corr_var") or tk.BooleanVar(self, value=False)
            self.show_etof_corr_var = shared_vars.get("show_etof_corr_var") or tk.BooleanVar(self, value=False)
            self._rotation_vars = shared_vars.get("rotation_vars") or self._own_rotation_vars()
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
            self.log_etof_corr_var = tk.BooleanVar(self, value=False)
            self.show_etof_corr_var = tk.BooleanVar(self, value=False)
            self._rotation_vars = self._own_rotation_vars()

        # Base to fall back on for any field that fails to parse below, then
        # immediately replaced with whatever the (possibly shared/persisted)
        # variables actually hold.
        self._hist_settings = {key: dict(cfg) for key, cfg in DEFAULT_HIST_SETTINGS.items()}
        self._hist_settings = self._read_hist_settings()

        self.focus_choice_var = tk.StringVar(self, value=PLOT_SPECS[0][1])
        self.focus_button_var = tk.StringVar(self, value="Focus")
        self._focus_key = None

        # Populated by _build_axes(): PlotItems, and either an ImageItem
        # (pixel/cluster) or a PlotDataItem (everything else) per key, plus
        # a centered "no data yet" TextItem per plot (hidden once real data
        # comes in).
        self._plots = {}
        self._images = {}
        self._curves = {}
        self._empty_labels = {}

        self._build_ui()
        self._wire_hist_traces()
        self.update_plots(self.empty_plot_data())

    # ---- public API -----------------------------------------------------

    def hist_snapshot(self):
        return {key: dict(cfg) for key, cfg in self._hist_settings.items()}

    def _own_rotation_vars(self):
        return {key: tk.StringVar(self, value=f"{value:g}") for key, value in ROTATION_DEFAULTS.items()}

    def showing_etof_correlation(self):
        return self._corr_enabled and bool(self.show_etof_corr_var.get())

    def empty_plot_data(self):
        settings = self.hist_snapshot()
        pixel_bins = int(settings["pixel"]["bins"])
        cluster_bins = int(settings["cluster"]["bins"])
        data = {
            "pixel_hist": np.zeros((pixel_bins, pixel_bins), dtype=np.float64),
            "cluster_hist": np.zeros((cluster_bins, cluster_bins), dtype=np.float64),
            "itof_hist": make_hist_1d([], **hist_args(settings["itof"])),
            "etof_hist": make_hist_1d([], **hist_args(settings["etof"])),
            "cluster_t_hist": make_hist_1d([], **hist_args(settings["cluster_t"])),
            "settings": settings,
        }
        if ETOF_CORR_KEY in settings:
            bins = int(settings[ETOF_CORR_KEY]["bins"])
            data["x_etof_hist"] = np.zeros((bins, bins), dtype=np.float64)
            data["y_etof_hist"] = np.zeros((bins, bins), dtype=np.float64)
        return data

    def update_plots(self, data):
        self._last_data = data
        settings = data.get("settings", self.hist_snapshot())

        self._draw_map("pixel", data["pixel_hist"], settings["pixel"], self.log_pixel_var.get())
        self._draw_map("cluster", data["cluster_hist"], settings["cluster"], self.log_cluster_var.get())
        self._draw_counts(data["pixel_hist"])
        self._draw_line("itof", data["itof_hist"], "Time (ns)", "Counts", self.log_itof_var.get())
        self._draw_line("cluster_t", data["cluster_t_hist"], "Time (ns)", "Counts", self.log_cluster_t_var.get())
        self._draw_line("etof", data["etof_hist"], "Time (ns)", "Counts", self.log_etof_var.get())
        if self.showing_etof_correlation():
            corr = settings.get(ETOF_CORR_KEY, self._hist_settings.get(ETOF_CORR_KEY))
            if corr is not None:
                for key, axis in (("x_etof", "x"), ("y_etof", "y")):
                    self._draw_corr(key, axis, data.get(f"{key}_hist"), corr, settings["cluster"])

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        # Qt's default QGridLayout margins would otherwise leave the plots
        # and the "Plot options" divider rule inset from this panel's true
        # edges; the default row spacing (separate from margins) would
        # otherwise leave a small gap between the plots and that rule.
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        # Plain QWidgets don't paint a background of their own, so without
        # this, any sliver not covered by the plots themselves shows the
        # ordinary grey Qt window background instead of true white --
        # visibly off against the plots' own explicit white.
        set_background(self, QtGui.QColor(QtCore.Qt.white))

        # Log-scale/gamma/focus controls and the bins/bounds table are
        # secondary to the plots themselves, so they share one collapsible
        # footer -- collapsing it hands its space straight back to the
        # plots (Qt excludes hidden widgets from layout sizing). flush=True:
        # a plain footer reaching this panel's edges, set off from the
        # plots by a single rule instead of sitting as an inset box.
        # toggle_last, when the plots (weight=1) sit above the footer,
        # keeps the "Plot options" toggle itself pinned to the panel's far
        # edge instead of sliding up/down as the plots grow to fill
        # whatever the footer's panel isn't using.
        footer_separator = "top" if self._plots_first else "bottom"
        self._footer = CollapsibleFrame(
            self, text="Plot options", flush=True, separator=footer_separator, toggle_last=self._plots_first,
        )
        self._footer.body.columnconfigure(0, weight=1)

        # Both get their own titled/bordered card (QGroupBox, via
        # ttk.LabelFrame) -- the outer "Plot options" footer itself stays
        # flush/borderless (a deliberate earlier design choice), but its
        # contents still read as distinct grouped panels.
        self._opts_bar = ttk.LabelFrame(self._footer.body, text="Plot controls")
        self._build_opts_bar(self._opts_bar)
        self._opts_bar.grid(row=0, column=0, sticky="w", pady=(0, 8))

        # Collapsed by default: bins/bounds are a set-once-and-forget
        # thing, not something glanced at every session, so it starts
        # tucked away and only takes up space once someone opens it.
        self._hist_section = CollapsibleFrame(
            self._footer.body, text="Histogram bins / bounds", collapsed=True,
        )
        self._hist_section.body.columnconfigure(0, weight=1)
        self._build_hist_frame(self._hist_section.body)
        self._hist_section.grid(row=1, column=0, sticky="ew")

        self._plot_frame = ttk.Frame(self)
        self._plot_frame.rowconfigure(0, weight=1)
        self._plot_frame.columnconfigure(0, weight=1)
        self._glw = pg.GraphicsLayoutWidget()
        grid_into(self._glw, self._plot_frame, row=0, column=0, sticky="nsew")
        self._build_axes()

        if self._plots_first:
            self._plot_frame.grid(row=0, column=0, sticky="nsew")
            self._footer.grid(row=1, column=0, sticky="ew")
            self.rowconfigure(0, weight=1)
        else:
            self._footer.grid(row=0, column=0, sticky="ew")
            self._plot_frame.grid(row=1, column=0, sticky="nsew")
            self.rowconfigure(1, weight=1)

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
            values=[label for _, label in self._plot_specs()],
            state="readonly",
            width=18,
        )
        focus_box.grid(row=0, column=9, padx=(6, 6))
        focus_box.bind("<<ComboboxSelected>>", self._on_focus_choice)
        self._focus_box = focus_box
        ttk.Button(plot_opts, textvariable=self.focus_button_var, command=self._toggle_focus).grid(row=0, column=10)

        # A full horizontal rule sets the gamma cluster apart from the
        # log-scale/single-plot row above it, so the bar reads as three
        # distinct control groups (log-scale checkboxes | single-plot
        # selector+Focus | gamma) instead of one dense block.
        ttk.Separator(plot_opts, orient="horizontal").grid(
            row=1, column=0, columnspan=12, sticky="ew", pady=(8, 8)
        )

        ttk.Label(plot_opts, text="Gamma (pixel/cluster maps):").grid(row=2, column=0, columnspan=2, sticky="w")
        gamma_scale = ttk.Scale(
            plot_opts,
            from_=0.1,
            to=3.0,
            orient="horizontal",
            variable=self.gamma_var,
            length=160,
            command=self._on_gamma_change,
        )
        gamma_scale.grid(row=2, column=2, columnspan=3, sticky="w", padx=(6, 6))
        ttk.Label(plot_opts, textvariable=self.gamma_label_var, width=5).grid(row=2, column=5, sticky="w")
        ttk.Button(plot_opts, text="Reset", command=self._reset_gamma).grid(row=2, column=6, sticky="w", padx=(6, 0))

        ttk.Separator(plot_opts, orient="horizontal").grid(
            row=3, column=0, columnspan=12, sticky="ew", pady=(8, 8)
        )
        build_rotation_row(plot_opts, self._rotation_vars, "clusters").grid(
            row=4, column=0, columnspan=12, sticky="w"
        )

        if self._corr_enabled:
            ttk.Checkbutton(
                plot_opts, text="Show x / y vs e-ToF", variable=self.show_etof_corr_var,
                command=self._on_show_corr_toggle,
            ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
            ttk.Checkbutton(
                plot_opts, text="Log scale (x / y vs e-ToF)", variable=self.log_etof_corr_var,
                command=self._on_log_toggle,
            ).grid(row=5, column=3, columnspan=4, sticky="w", padx=(6, 0), pady=(6, 0))

    def _build_hist_frame(self, container):
        # Two separate titled/bordered tables (QGroupBox via ttk.LabelFrame)
        # instead of one combined table with two column-groups sharing
        # generic headers -- clearer at a glance which rows are spatial
        # (pixel bounds) vs. temporal (time-of-flight bounds), and each
        # table's Min/Max headers can carry the right unit for its rows.
        px_frame = ttk.LabelFrame(container, text="Pixel/Cluster histograms")
        px_frame.grid(row=0, column=0, sticky="ew")
        self._build_hist_table(px_frame, PLOT_SPECS[:3], unit="px")

        time_frame = ttk.LabelFrame(container, text="Time histograms")
        time_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        time_specs = PLOT_SPECS[3:]
        if self._corr_enabled:
            # Bins are per axis; the position range is the cluster histogram's.
            time_specs = time_specs + [(ETOF_CORR_KEY, "x / y vs e-ToF")]
        self._build_hist_table(time_frame, time_specs, unit="ns")

    def _build_hist_table(self, frame, specs, unit):
        ttk.Label(frame, text="Plot").grid(row=0, column=0, sticky="w", padx=(6, 4), pady=(4, 2))
        ttk.Label(frame, text="Bins").grid(row=0, column=1, pady=(4, 2))
        ttk.Label(frame, text=f"Min ({unit})").grid(row=0, column=2, pady=(4, 2))
        ttk.Label(frame, text=f"Max ({unit})").grid(row=0, column=3, pady=(4, 2))
        last_row = len(specs)
        for row_idx, (key, label) in enumerate(specs, start=1):
            bottom_pad = 4 if row_idx == last_row else 1
            ttk.Label(frame, text=label).grid(row=row_idx, column=0, sticky="w", padx=(6, 4), pady=(1, bottom_pad))
            for offset, field in enumerate(("bins", "min", "max"), start=1):
                ttk.Entry(frame, textvariable=self._hist_vars[key][field], width=9).grid(
                    row=row_idx, column=offset, padx=2, pady=(1, bottom_pad)
                )

    def _build_axes(self):
        self._glw.clear()  # destroys child items, so the old TextItems go with it
        self._plots = {}
        self._images = {}
        self._curves = {}
        self._empty_labels = {}

        if self._focus_key is not None:
            layout = [(self._focus_key, PLOT_LABELS[self._focus_key], 0, 0)]
        else:
            layout = [(key, label, idx // 3, idx % 3) for idx, (key, label) in enumerate(self._plot_specs())]

        axis_width, axis_height = self._axis_space()
        for key, label, row, col in layout:
            vb = ZoomFocusViewBox(on_focus=lambda k=key: self._on_plot_double_click(k))
            plot_item = self._glw.addPlot(row=row, col=col, viewBox=vb)
            plot_item.setTitle(label)
            # Fixed axis space: left to size itself, an axis grows and
            # shrinks with its tick labels (e.g. 900 -> 1000 counts), which
            # moves the plot's borders from frame to frame.
            plot_item.getAxis("left").setWidth(axis_width)
            plot_item.getAxis("bottom").setHeight(axis_height)
            self._plots[key] = plot_item
            if key in ("pixel", "cluster"):
                img = pg.ImageItem()
                plot_item.addItem(img)
                vb.setAspectLocked(True)
                self._images[key] = img
            elif key in CORRELATION_KEYS:
                img = pg.ImageItem()
                plot_item.addItem(img)
                self._images[key] = img
            else:
                self._curves[key] = plot_item.plot([], [])

            empty_label = pg.TextItem(self._empty_text, anchor=(0.5, 0.5), color=(140, 140, 140))
            empty_label.setVisible(False)
            plot_item.addItem(empty_label, ignoreBounds=True)
            self._empty_labels[key] = empty_label

        # Every column / row gets the same share of the space whatever its
        # plot contains.
        grid = self._glw.ci.layout
        for row in range(grid.rowCount()):
            grid.setRowStretchFactor(row, 1)
        for col in range(grid.columnCount()):
            grid.setColumnStretchFactor(col, 1)

    def _axis_space(self):
        """(left-axis width, bottom-axis height) in px: room for 8-character
        tick labels (e.g. -1234567, 1.5e+06) plus the axis title."""
        metrics = QtGui.QFontMetrics(self._glw.font())
        tick_text = metrics.horizontalAdvance("-8888888")
        line = metrics.height()
        # tick text + gap + axis title line + tick marks / padding
        return tick_text + line + 18, 2 * line + 14

    def _plot_specs(self):
        """The plots on show: the six standard ones, plus the x / y vs e-ToF
        maps while they are switched on."""
        return PLOT_SPECS + (CORRELATION_SPECS if self.showing_etof_correlation() else [])

    def _on_show_corr_toggle(self):
        if self._focus_key in CORRELATION_KEYS and not self.showing_etof_correlation():
            self._focus_key = None
            self.focus_button_var.set("Focus")
        self._focus_box.configure(values=[label for _, label in self._plot_specs()])
        self._build_axes()
        # The settings gain or lose ETOF_CORR_KEY, which tells the host to
        # start over (and to compute the maps from now on) and redraws.
        self._on_hist_setting_change()

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
            self.focus_button_var.set("Show All")
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

    def _setting_keys(self):
        keys = [key for key, _ in PLOT_SPECS]
        if self.showing_etof_correlation():
            keys.append(ETOF_CORR_KEY)
        return keys

    def _wire_hist_traces(self):
        keys = [key for key, _ in PLOT_SPECS] + ([ETOF_CORR_KEY] if self._corr_enabled else [])
        for key in keys:
            for field in ("bins", "min", "max"):
                self._hist_vars[key][field].trace_add("write", self._on_hist_setting_change)
        for var in self._rotation_vars.values():
            var.trace_add("write", self._on_hist_setting_change)

    def _read_hist_settings(self):
        settings = {}
        for key in self._setting_keys():
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
        settings[ROTATION_KEY] = read_rotation(self._rotation_vars, self._hist_settings.get(ROTATION_KEY))
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

    def _set_empty_placeholder(self, key, is_empty):
        text_item = self._empty_labels.get(key)
        plot_item = self._plots.get(key)
        if text_item is None or plot_item is None:
            return
        text_item.setVisible(is_empty)
        if is_empty:
            (x0, x1), (y0, y1) = plot_item.getViewBox().viewRange()
            text_item.setPos((x0 + x1) / 2, (y0 + y1) / 2)

    @staticmethod
    def _set_labels(plot_item, bottom, left):
        """Axis labels, set only when they change -- setLabel re-lays out
        the axis, which is a noticeable share of a redraw at live rates."""
        if getattr(plot_item, "_axis_labels", None) != (bottom, left):
            plot_item.setLabel("bottom", bottom)
            plot_item.setLabel("left", left)
            plot_item._axis_labels = (bottom, left)

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
        self._set_labels(plot_item, "X", "Y")
        plot_item.getViewBox().setRange(xRange=(low, high), yRange=(low, high), padding=0)
        self._set_empty_placeholder(key, not np.any(hist))

    def _draw_corr(self, key, axis, hist, corr_cfg, pos_cfg):
        """A cluster-position vs e-ToF-time map: time across, position up.
        hist is [time bin, position bin]; None (no maps in this data yet,
        e.g. accumulated before the option was switched on) draws empty."""
        img = self._images.get(key)
        plot_item = self._plots.get(key)
        if img is None or plot_item is None:
            return
        bins = max(1, int(corr_cfg["bins"]))
        if hist is None:
            hist = np.zeros((bins, bins), dtype=np.float64)
        t_lo, t_hi = float(corr_cfg["min"]), float(corr_cfg["max"])
        if t_hi <= t_lo:
            t_hi = t_lo + 1.0
        p_lo, p_hi = float(pos_cfg["min"]), float(pos_cfg["max"])
        if p_hi <= p_lo:
            p_lo, p_hi = 0.0, float(_DETECTOR_PX)
        rgba = hist_to_rgba(hist, log=self.log_etof_corr_var.get(), gamma=self.gamma_var.get())
        img.setImage(rgba, autoLevels=False)
        img.setRect(QtCore.QRectF(t_lo, p_lo, t_hi - t_lo, p_hi - p_lo))
        self._set_labels(plot_item, "e-ToF time (ns)", f"Cluster {axis} (px)")
        plot_item.getViewBox().setRange(xRange=(t_lo, t_hi), yRange=(p_lo, p_hi), padding=0)
        self._set_empty_placeholder(key, not np.any(hist))

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
        self._set_labels(plot_item, "Counts per pixel", "Pixels")
        plot_item.enableAutoRange(axis="y")
        plot_item.setXRange(*counts_hist["range"], padding=0)
        self._set_empty_placeholder("counts", not np.any(counts_hist["counts"]))

    def _draw_line(self, key, hist, xlabel, ylabel, log_enabled):
        curve = self._curves.get(key)
        plot_item = self._plots.get(key)
        if curve is None or plot_item is None:
            return
        plot_item.setLogMode(y=log_enabled)
        curve.setData(hist["bins"], hist["counts"])
        self._set_labels(plot_item, xlabel, ylabel)
        plot_item.enableAutoRange(axis="y")
        plot_item.setXRange(*hist["range"], padding=0)
        self._set_empty_placeholder(key, not np.any(hist["counts"]))
