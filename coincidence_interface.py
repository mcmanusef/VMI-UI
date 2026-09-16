"""Analysis tab: electron-ion coincidence plots of a calibrated dataset,
with click-and-drag gates on each quantity (coincidence.py)."""
import pathlib

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

import qtk as tk
from qtk import ttk, filedialog, grid_into

import app_settings
import coincidence as co
import ui_style
from calibration_interface import (
    BLUE, GREEN, GREY, BackgroundTask, PlotGrid, read_number, relax_titles, show_placeholder,
)
from qt_plots import ZoomFocusViewBox, rainforest_colormap

EMPTY_TEXT = "No data — press Load"
MODE_2D = "2D heat map"
MODE_1D = "1D plot"
MODES = (MODE_2D, MODE_1D)
# Gate plots wrap onto another row rather than getting too narrow.
PLOTS_PER_ROW = 7
# The gate plots have their own bin count, so the main plot's X and Y bins
# only redraw the main plot.
REDRAW_DELAY_MS = 150
# Quantities that are a distribution, and so are drawn as a curve; the rest
# are per-bin values drawn as points.
CURVE_QUANTITIES = (co.QUANTITY_COUNTS, co.QUANTITY_DENSITY)


def colorize(values, levels, log):
    """values[x bin, y bin] -> RGBA for an ImageItem, scaled to `levels`.
    Bins with no data (NaN), and non-positive bins on a log scale, are left
    transparent instead of being drawn as the lowest color."""
    data = np.asarray(values, dtype=np.float64).T  # row-major: row = y bin
    if log:
        with np.errstate(invalid="ignore", divide="ignore"):
            data = np.where(data > 0, np.log10(data), np.nan)
    lo, hi = levels
    span = (hi - lo) if hi > lo else 1.0
    valid = np.isfinite(data)
    norm = np.clip((np.where(valid, data, lo) - lo) / span, 0.0, 1.0)
    lut = rainforest_colormap().getLookupTable(0.0, 1.0, 256)
    rgb = lut[np.clip((norm * 255.0).astype(np.intp), 0, 255)]
    return np.dstack([rgb, np.where(valid, 255, 0).astype(np.ubyte)])


def data_levels(values, log):
    """(low, high) covering the bins that have data, in the scale drawn."""
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if log:
        data = data[data > 0]
        data = np.log10(data) if data.size else data
    if data.size == 0:
        return 0.0, 1.0
    lo, hi = float(data.min()), float(data.max())
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


class CoincidenceInterface(ttk.Frame):
    """A calibrated HDF5 dataset in; gated distributions out. The row of
    small plots gates the data (each one ignores its own gate), and the
    large plot shows the gated 1D or 2D distribution."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        def pvar(cls, key, default):
            return app_settings.persistent_var(self, cls, f"coincidence.{key}", default)

        self.path_var = pvar(tk.StringVar, "path", "")
        self.mode_var = pvar(tk.StringVar, "mode", MODE_2D)
        self.x_var = pvar(tk.StringVar, "x", "px")
        self.y_var = pvar(tk.StringVar, "y", "pz")
        self.x_bins_var = pvar(tk.StringVar, "bins", "200")
        self.y_bins_var = pvar(tk.StringVar, "y_bins", "200")
        self.gate_bins_var = pvar(tk.StringVar, "gate_bins", "200")
        self.log_var = pvar(tk.BooleanVar, "log_color", False)
        self.auto_levels_var = pvar(tk.BooleanVar, "auto_levels", True)
        self.aspect_var = pvar(tk.BooleanVar, "equal_aspect", False)
        self.quantity_var = pvar(tk.StringVar, "quantity", co.QUANTITY_DENSITY)
        # Which columns get a plot of their own, and which can be a main-plot axis.
        self.plot_columns_var = pvar(tk.StringVar, "gate_columns", ",".join(co.DEFAULT_KEYS))
        self.main_columns_var = pvar(tk.StringVar, "main_columns", ",".join(co.DEFAULT_KEYS))

        self.status_var = tk.StringVar(self, value="Idle")
        self.rows_var = tk.StringVar(self, value="--")
        self.gated_var = tk.StringVar(self, value="--")
        self.electrons_var = tk.StringVar(self, value="--")
        self.gate_info_var = tk.StringVar(self, value="No gates.")

        self._task = BackgroundTask(self)
        self._dataset = None
        self._ranges = {}         # key -> (low, high), set whether or not it is applied
        self._enabled = {}        # key -> BooleanVar: is this range applied?
        self._regions = {}        # key -> LinearRegionItem
        self._small = {}          # key -> PlotItem
        self._show_plot = {}      # key -> BooleanVar: has a plot of its own
        self._show_main = {}      # key -> BooleanVar: offered as a main-plot axis
        self._range_text = {}     # key -> (low StringVar, high StringVar)
        self._curves = {}         # key -> the gate plot's PlotDataItem, reused
        self._placeholders = {}   # plot -> its "no data" label, reused
        self._data_ranges = {}    # key -> the column's plotting range
        self._binning = {}        # key -> (low, high, bin index, inside) for the gate plots
        self._main_binning = {}   # (key, low, high, bins) -> (bin index, inside)
        self._mask_cache = {}     # key -> (range, boolean mask) for an applied gate
        self._small_drawn = set()
        self._redraw_after = None
        self._pending = "all"

        self._build_ui()
        # These only affect the main plot, so they don't recompute the gate plots.
        for var in (self.mode_var, self.x_var, self.y_var, self.x_bins_var, self.y_bins_var,
                    self.log_var, self.quantity_var, self.auto_levels_var, self.aspect_var):
            var.trace_add("write", self._schedule_main_redraw)
        self.gate_bins_var.trace_add("write", self._on_gate_bins_changed)
        self._redraw()

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=3)
        main.rowconfigure(1, weight=2)
        sidebar.rowconfigure(5, weight=1)

        ui_style.build_button_bar(sidebar, 0, [("Load", self.load_data), ("Clear Gates", self.clear_gates)])
        ui_style.StatusBlock(sidebar, 1, self.status_var)

        data = ui_style.build_section(sidebar, 2, "Dataset")
        ui_style.add_form_row(
            data, 0, "Calibrated file:", ui_style.build_path_field(data, self.path_var, self._browse),
        )
        ui_style.add_note(
            data, 1,
            "An HDF5 file from the apply calibration tab. Each electron is joined to the ions of its own "
            "pulse, so m/q is per coincidence; electrons with no ion have no m/q. p_r is the full "
            "momentum magnitude |p|.",
            pady=(4, 4),
        )

        plot = ui_style.build_section(sidebar, 3, "Main plot")
        ui_style.add_form_row(plot, 0, "Plot:", ttk.Combobox(
            plot, textvariable=self.mode_var, values=MODES, state="readonly", width=18,
        ))
        self._x_combo = ttk.Combobox(plot, textvariable=self.x_var, state="readonly", width=18)
        ui_style.add_form_row(plot, 1, "X axis:", self._x_combo)
        self._y_label = ttk.Label(plot, text="Y axis:")
        self._y_label.grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        self._y_combo = ttk.Combobox(plot, textvariable=self.y_var, state="readonly", width=18)
        self._y_combo.grid(row=2, column=1, sticky="ew", pady=4)
        ui_style.add_form_row(plot, 3, "X bins:", ttk.Entry(plot, textvariable=self.x_bins_var, width=18))
        self._y_bins_label = ttk.Label(plot, text="Y bins:")
        self._y_bins_label.grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        self._y_bins_entry = ttk.Entry(plot, textvariable=self.y_bins_var, width=18)
        self._y_bins_entry.grid(row=4, column=1, sticky="ew", pady=4)
        ui_style.add_form_row(plot, 5, "Quantity:", ttk.Combobox(
            plot, textvariable=self.quantity_var, values=co.QUANTITIES, state="readonly", width=18,
        ))
        ttk.Checkbutton(plot, text="Log color scale (2D)", variable=self.log_var).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Checkbutton(plot, text="Auto color levels", variable=self.auto_levels_var).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(0, 0)
        )
        ttk.Checkbutton(plot, text="Equal aspect ratio (2D)", variable=self.aspect_var).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(0, 0)
        )
        ui_style.add_note(
            plot, 9,
            "Probability density normalizes the counts to integrate to 1 over the plotted range. Other "
            "quantities are drawn as points, one per bin that has data; the asymmetry is "
            "(forward − backward) / total by the sign of p_z. An axis whose gate is applied is bounded "
            "by that gate. Drag the color bar's handles to set the heat map's levels, which turns auto "
            "levels off.",
            pady=(4, 4),
        )

        gates = ui_style.build_section(sidebar, 4, "Gates", pady=0)
        ui_style.add_note(
            gates, 0,
            "Drag the shaded region on a plot, or its edges, to set that column's range; the range can "
            "be edited whether or not the gate is on. “Plot” shows the column's own plot and “Gate” "
            "applies its range, independently: a gate works with no plot shown, and each plot shows the "
            "data gated by the other columns only.",
            pady=(4, 4),
        )
        ui_style.add_note(gates, 1, textvariable=self.gate_info_var)
        row = ui_style.add_stat_rows(gates, [
            ("Electron rows:", self.electrons_var),
            ("Coincidence rows:", self.rows_var),
            ("Rows after gates:", self.gated_var),
        ], start_row=2)
        ui_style.add_form_row(
            gates, row, "Gate plot bins:", ttk.Entry(gates, textvariable=self.gate_bins_var, width=18),
        )
        row += 1
        ui_style.add_note(
            gates, row,
            "Columns of the dataframe: “Plot” shows one, “Gate” applies its range, “Axis” offers it to "
            "the main plot. The low and high boxes are the range: type one and press Enter, or drag the "
            "region on its plot.",
            pady=(6, 2),
        )
        self._gates_body = gates
        self._columns_row = row + 1
        self._columns_frame = ttk.Frame(gates)
        self._columns_frame.grid(row=self._columns_row, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        self._main_area = main
        self._main_focused = False
        self._main_glw = pg.GraphicsLayoutWidget()
        self._main_plot = self._main_glw.addPlot(
            row=0, col=0, viewBox=ZoomFocusViewBox(on_focus=self._toggle_main_focus),
        )
        self._color_bar = pg.ColorBarItem(colorMap=rainforest_colormap(), interactive=True, values=(0.0, 1.0))
        self._color_bar.sigLevelsChanged.connect(self._on_levels_changed)
        self._main_glw.addItem(self._color_bar, row=0, col=1)
        relax_titles((self._main_plot,))
        grid_into(self._main_glw, main, row=0, column=0, sticky="nsew")

        self._bottom = ttk.Frame(main)
        self._bottom.grid(row=1, column=0, sticky="nsew")
        self._bottom.columnconfigure(0, weight=1)
        self._bottom.rowconfigure(0, weight=1)
        self._grid = PlotGrid(on_focus_changed=self._on_small_focus)
        grid_into(self._grid.widget, self._bottom, row=0, column=0, sticky="nsew")
        self._checks = ttk.Frame(self._bottom)
        self._checks.grid(row=1, column=0, sticky="ew")

    # ---- double-click focus ------------------------------------------------

    def _show_both(self):
        """The default split: main plot above, small plots below."""
        self._main_focused = False
        self._main_glw.setVisible(True)
        self._bottom.setVisible(True)
        self._main_area.rowconfigure(0, weight=3)
        self._main_area.rowconfigure(1, weight=2)

    def _toggle_main_focus(self):
        """Double-click on the main plot: it takes the whole tab, and again
        brings the small plots back."""
        if self._main_focused:
            self._show_both()
            return
        self._main_focused = True
        self._bottom.setVisible(False)
        self._main_area.rowconfigure(1, weight=0)

    def _on_small_focus(self, focused):
        """A focused small plot gets the whole tab too, not just the strip."""
        if focused is None:
            self._show_both()
            return
        self._main_glw.setVisible(False)
        self._main_area.rowconfigure(0, weight=0)

    def _browse(self):
        initial = self.path_var.get().strip()
        chosen = filedialog.askopenfilename(
            title="Choose calibrated dataset",
            initialdir=str(pathlib.Path(initial).parent) if initial else None,
            filetypes=[("HDF5", "*.h5")],
            parent=self,
        )
        if chosen:
            self.path_var.set(chosen)

    # ---- which columns are shown where -------------------------------------

    @staticmethod
    def _split(text):
        return [key for key in str(text).split(",") if key]

    def _build_column_table(self):
        """A row per dataframe column: show its plot, apply its gate, offer it
        as a main-plot axis, and its range as text. The gate boxes live here
        rather than with the plots, so a gate works with no plot shown."""
        self._columns_frame.grid_remove()
        self._columns_frame.deleteLater()
        self._columns_frame = ttk.Frame(self._gates_body)
        self._columns_frame.grid(row=self._columns_row, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self._show_plot, self._show_main, self._range_text, self._enabled = {}, {}, {}, {}

        plot_keys = set(self._split(self.plot_columns_var.get()))
        main_keys = set(self._split(self.main_columns_var.get()))
        for column, text in ((1, "Plot"), (2, "Gate"), (3, "Axis"), (4, "Low"), (5, "High")):
            ttk.Label(self._columns_frame, text=text, font=ui_style.SMALL_FONT).grid(row=0, column=column, padx=2)
        for row, variable in enumerate(self._dataset.variables, start=1):
            key = variable.key
            ttk.Label(self._columns_frame, text=self._dataset.label(key), font=ui_style.SMALL_FONT).grid(
                row=row, column=0, sticky="w", padx=(0, 6)
            )
            self._enabled[key] = tk.BooleanVar(self, value=False)
            self._enabled[key].trace_add("write", lambda *_, key=key: self._on_gate_toggled(key))
            for column, (store, initial, command) in enumerate((
                (self._show_plot, key in plot_keys, self._on_plot_columns_changed),
                (self._enabled, None, None),
                (self._show_main, key in main_keys, self._on_main_columns_changed),
            ), start=1):
                if initial is not None:
                    var = tk.BooleanVar(self, value=initial)
                    var.trace_add("write", lambda *_, command=command: command())
                    store[key] = var
                ttk.Checkbutton(self._columns_frame, variable=store[key]).grid(row=row, column=column, padx=2)
            self._range_text[key] = (tk.StringVar(self, value=""), tk.StringVar(self, value=""))
            for column, text_var in enumerate(self._range_text[key], start=4):
                entry = ttk.Entry(self._columns_frame, textvariable=text_var, width=7)
                entry.grid(row=row, column=column, padx=2, pady=1)
                entry.bind("<Return>", lambda _event=None, key=key: self._commit_range_text(key))
                entry.bind("<FocusOut>", lambda _event=None, key=key: self._commit_range_text(key))
            self._sync_range_text(key)
        self._columns_frame.columnconfigure(6, weight=1)

    def _sync_range_text(self, key):
        """The low/high boxes follow the region."""
        if key not in self._range_text:
            return
        value = self._ranges.get(key)
        for text_var, number in zip(self._range_text[key], value if value is not None else ("", "")):
            text_var.set(f"{number:.6g}" if value is not None else "")

    def _commit_range_text(self, key):
        """A typed range, whether or not the column has a plot."""
        if self._dataset is None or key not in self._range_text:
            return
        label = co.variable_for(key).label
        low_var, high_var = self._range_text[key]
        if not str(low_var.get()).strip() and not str(high_var.get()).strip():
            return
        try:
            low = read_number(low_var, f"{label} low")
            high = read_number(high_var, f"{label} high")
        except ValueError as exc:
            self.status_var.set(f"Invalid setting: {exc}")
            return
        if low == high:
            self.status_var.set(f"Invalid setting: {label} low and high must differ.")
            return
        self._set_range(key, (min(low, high), max(low, high)))
        if self._enabled[key].get():
            self._redraw()
        else:
            self._update_gate_info()

    def _selected(self, store):
        return [variable.key for variable in self._dataset.variables if store[variable.key].get()]

    def _on_plot_columns_changed(self):
        if self._dataset is None:
            return
        self.plot_columns_var.set(",".join(self._selected(self._show_plot)))
        self._build_small_plots()
        self._redraw()

    def _on_main_columns_changed(self):
        if self._dataset is None:
            return
        self.main_columns_var.set(",".join(self._selected(self._show_main)))
        self._refresh_axis_choices()
        self._redraw()

    def _refresh_axis_choices(self):
        keys = self._selected(self._show_main) if self._dataset is not None else []
        for combo, var, fallback in ((self._x_combo, self.x_var, 0), (self._y_combo, self.y_var, 2)):
            combo.configure(values=keys)
            if keys and var.get() not in keys:
                var.set(keys[min(fallback, len(keys) - 1)])

    def _build_small_plots(self):
        """One plot, gate checkbox and region per column with "Gate" ticked.
        The grid and the checkbox strip are replaced outright: a frame keeps
        the layout it was given, so its widgets can't simply be cleared out.
        Ranges and applied gates belong to the columns, so they survive."""
        self._bottom._own_layout().removeWidget(self._grid.widget)
        self._grid.widget.setParent(None)
        self._grid.widget.deleteLater()
        self._checks.grid_remove()
        self._checks.deleteLater()

        self._grid = PlotGrid(on_focus_changed=self._on_small_focus)
        grid_into(self._grid.widget, self._bottom, row=0, column=0, sticky="nsew")
        self._checks = ttk.Frame(self._bottom)
        self._checks.grid(row=1, column=0, sticky="ew")
        self._small, self._regions = {}, {}
        self._curves, self._placeholders, self._binning = {}, {}, {}
        self._show_both()

        keys = self._selected(self._show_plot) if self._dataset is not None else []
        for index, key in enumerate(keys):
            row, column = divmod(index, PLOTS_PER_ROW)
            plot = self._grid.add(row, column)
            label = self._dataset.label(key)
            plot.setTitle(label)
            plot.setLabel("bottom", label)
            plot.setLabel("left", "Counts")
            self._small[key] = plot
            # One curve and one placeholder per plot, updated in place: making
            # them again on every redraw is what made changes feel slow.
            curve = pg.PlotDataItem(pen=pg.mkPen(BLUE, width=1))
            plot.addItem(curve)
            self._curves[key] = curve
            self._placeholders[plot] = ui_style.add_empty_placeholder(plot, EMPTY_TEXT)

            # Always shown and draggable: the Gate box only decides whether
            # the range is applied, so a gate can be placed before it is on.
            region = pg.LinearRegionItem()
            region.setZValue(10)
            region.sigRegionChanged.connect(lambda item, key=key: self._on_region_dragging(key, item))
            region.sigRegionChangeFinished.connect(lambda item, key=key: self._on_region_changed(key, item))
            plot.addItem(region, ignoreBounds=True)
            self._regions[key] = region

            # The same variable as the row's Gate box, for reach under the plot.
            ttk.Checkbutton(
                self._checks, text=f"Gate {co.variable_for(key).label}", variable=self._enabled[key],
            ).grid(row=row, column=column, sticky="w", padx=8)
            self._checks.columnconfigure(column, weight=1)

        # Without this a title's text width becomes its plot's minimum width,
        # and the last columns get pushed out of the widget.
        relax_titles(tuple(self._small.values()))
        for key in keys:
            self._set_range(key, self._ensure_range(key))

    # ---- loading ----------------------------------------------------------------

    def load_data(self):
        if self._task.is_running():
            return
        path = self.path_var.get().strip()
        if not path:
            self.status_var.set("Calibrated file is required.")
            return
        if not pathlib.Path(path).is_file():
            self.status_var.set(f"File not found: {path}")
            return
        self.status_var.set("Loading...")
        self._task.start(
            lambda: co.load_dataset(path), self._on_loaded,
            lambda exc: self.status_var.set(f"Error: could not load the dataset: {exc}"),
        )

    def _on_loaded(self, dataset):
        self._dataset = dataset
        self._ranges = {}
        self._mask_cache = {}
        self._data_ranges = {}
        self._binning = {}
        self._main_binning = {}
        self._build_column_table()
        self._refresh_axis_choices()
        self._build_small_plots()
        self.electrons_var.set(str(dataset.n_electrons))
        self.rows_var.set(str(dataset.n_rows))
        columns = len(dataset.variables)
        self.status_var.set(
            f"Loaded {dataset.n_rows} coincidence rows and {columns} columns from "
            f"{pathlib.Path(dataset.path).name}."
        )
        self._redraw()

    # ---- gates ------------------------------------------------------------------

    @property
    def _gates(self):
        """The ranges that are switched on, i.e. the ones applied to the data."""
        return {key: value for key, value in self._ranges.items() if self._enabled[key].get()}

    def _set_range(self, key, value):
        self._ranges[key] = tuple(float(v) for v in value)
        region = self._regions.get(key)
        if region is not None:
            region.blockSignals(True)
            region.setRegion(self._ranges[key])
            region.blockSignals(False)
            self._style_region(key)
        self._sync_range_text(key)

    def _ensure_range(self, key):
        """A column's range, starting at the middle half of its own data."""
        if key not in self._ranges:
            values = self._dataset.arrays[key]
            values = values[np.isfinite(values)]
            if values.size:
                self._ranges[key] = tuple(float(v) for v in np.percentile(values, [25, 75]))
            else:
                self._ranges[key] = self._data_range(key)
        return self._ranges[key]

    def _reset_ranges(self):
        """Every range that has been set back to the middle half of its data."""
        for key in dict.fromkeys(list(self._ranges) + list(self._regions)):
            self._ranges.pop(key, None)
            self._set_range(key, self._ensure_range(key))

    def _style_region(self, key):
        """An applied gate is solid green; an unapplied one is grey, but can
        still be dragged."""
        on = self._enabled[key].get()
        region = self._regions.get(key)
        if region is None:
            return
        region.setBrush(pg.mkBrush(70, 130, 180, 60) if on else pg.mkBrush(150, 150, 150, 40))
        region.setRegion(region.getRegion())  # repaint with the new brush
        for line in region.lines:
            line.setPen(pg.mkPen(GREEN if on else GREY, width=2))

    def _update_gate_info(self):
        """The applied gates, and how many more ranges are set but off. Only
        a short line: the sidebar widens to fit a long one."""
        applied = [
            f"{co.variable_for(key).label} in [{lo:.4g}, {hi:.4g}]"
            for key, (lo, hi) in sorted(self._gates.items())
        ]
        off = len(self._ranges) - len(applied)
        text = ", ".join(applied) if applied else "No gates applied."
        self.gate_info_var.set(f"{text} {off} range(s) set but off." if off else text)

    def _on_gate_toggled(self, key):
        if self._dataset is None:
            return
        if self._enabled[key].get():
            self._set_range(key, self._ensure_range(key))
        self._style_region(key)
        self._redraw()

    def _on_region_dragging(self, key, item):
        """Cheap feedback while a region is dragged: the numbers follow the
        handle, and the plots wait until it is dropped."""
        self._ranges[key] = tuple(float(v) for v in item.getRegion())
        self._sync_range_text(key)
        self._update_gate_info()

    def _on_region_changed(self, key, item):
        self._ranges[key] = tuple(float(v) for v in item.getRegion())
        self._sync_range_text(key)
        if self._enabled[key].get():
            self._redraw()
        else:
            self._update_gate_info()

    def clear_gates(self):
        """Switches every gate off and puts the ranges back where they started."""
        for enabled in self._enabled.values():
            enabled.set(False)
        if self._dataset is not None:
            self._reset_ranges()
        self._redraw()

    # ---- drawing ----------------------------------------------------------------

    def _schedule_redraw(self, *_):
        self._schedule("all")

    def _schedule_main_redraw(self, *_):
        self._schedule("main")

    def _schedule(self, what):
        # A pending full redraw is never downgraded to a main-only one.
        self._pending = "all" if "all" in (self._pending, what) else what
        if self._redraw_after is not None:
            self.after_cancel(self._redraw_after)
        self._redraw_after = self.after(REDRAW_DELAY_MS, lambda: self._redraw(self._pending))

    def _on_levels_changed(self, *_):
        """Dragging the color bar takes over from the automatic levels."""
        if self.auto_levels_var.get():
            self.auto_levels_var.set(False)  # redraws through its trace
        else:
            self._schedule_redraw()

    def _redraw(self, what="all"):
        self._redraw_after = None
        self._pending = "main"
        is_2d = self.mode_var.get() == MODE_2D
        for widget in (self._y_label, self._y_combo, self._y_bins_label, self._y_bins_entry):
            widget.setVisible(is_2d)
        self._main_plot.clear()
        self._update_gate_info()

        if self._dataset is not None:
            try:
                if what == "all":
                    self._small_drawn = self._draw_small()
                drawn_main = self._draw_main(is_2d)
            except ValueError as exc:
                self.status_var.set(f"Invalid setting: {exc}")
                drawn_main = set()
        else:
            self._small_drawn, drawn_main = set(), set()

        self._color_bar.setVisible(is_2d and self._main_plot in drawn_main)
        self._main_plot.setAspectLocked(is_2d and self.aspect_var.get())
        self._main_plot.autoRange()
        if what == "all":
            for plot in self._small.values():
                plot.autoRange()
                ui_style.show_empty_placeholder(plot, self._placeholders[plot], plot not in self._small_drawn)
        show_placeholder(self._main_plot, EMPTY_TEXT, self._main_plot not in drawn_main)

    # ---- cached masks and binning ------------------------------------------------

    def _mask_for(self, key):
        """The rows inside one gate, remembered until its range changes."""
        gate = self._gates[key]
        cached = self._mask_cache.get(key)
        if cached is not None and cached[0] == gate:
            return cached[1]
        values = self._dataset.arrays[key]
        mask = (values >= min(gate)) & (values <= max(gate))
        self._mask_cache[key] = (gate, mask)
        return mask

    def _combined_mask(self, skip=None):
        """Rows inside every applied gate except `skip`; None means all rows."""
        masks = [self._mask_for(key) for key in self._gates if key != skip]
        if not masks:
            return None
        combined = masks[0].copy()
        for mask in masks[1:]:
            np.logical_and(combined, mask, out=combined)
        return combined

    def _main_binning_for(self, key, low, high, bins):
        """Bin indices for a main-plot axis, kept while the axis, its range
        and the bin count stay put."""
        cache_key = (key, low, high, bins)
        cached = self._main_binning.get(cache_key)
        if cached is None:
            if len(self._main_binning) > 8:
                self._main_binning.clear()
            cached = co.bin_index(self._dataset.arrays[key], low, high, bins)
            self._main_binning[cache_key] = cached
        return cached

    def _data_range(self, key):
        """A column's plotting range. Worth keeping: it is a percentile over
        the whole column, and every redraw asks for it."""
        cached = self._data_ranges.get(key)
        if cached is None:
            cached = co.default_range(self._dataset.arrays[key])
            self._data_ranges[key] = cached
        return cached

    def _binning_for(self, key, bins):
        """The gate plot's bins for one column, computed once per dataset."""
        cached = self._binning.get(key)
        if cached is None:
            values = self._dataset.arrays[key]
            low, high = self._data_range(key)
            index, inside = co.bin_index(values, low, high, bins)
            cached = (low, high, index, inside)
            self._binning[key] = cached
        return cached

    def _on_gate_bins_changed(self, *_):
        self._binning = {}
        self._schedule_redraw()

    # ---- drawing the plots --------------------------------------------------------

    def _draw_small(self):
        """Counts per gate plot, gated by every gate except the plot's own."""
        drawn = set()
        arrays = self._dataset.arrays
        pz = arrays["pz"]
        bins = read_number(self.gate_bins_var, "Gate plot bins", integer=True, positive=True)
        for key, plot in self._small.items():
            low, high, index, inside = self._binning_for(key, bins)
            mask = self._combined_mask(skip=key)
            selected = inside if mask is None else (inside & mask)
            centers, counts = co.histogram_1d(
                arrays[key], pz, bins, low, high, co.QUANTITY_COUNTS, binning=(index, selected),
            )
            self._curves[key].setData(centers, counts)
            drawn.add(plot)
        return drawn

    def _draw_main(self, is_2d):
        drawn = set()
        arrays = self._dataset.arrays
        x_bins = read_number(self.x_bins_var, "X bins", integer=True, positive=True)
        y_bins = read_number(self.y_bins_var, "Y bins", integer=True, positive=True)
        quantity = self.quantity_var.get()
        pz = arrays["pz"]

        inside = self._combined_mask()
        self.gated_var.set(str(int(inside.sum()) if inside is not None else self._dataset.n_rows))
        x_key = self.x_var.get()
        if x_key not in arrays:
            return drawn
        x_range = self._axis_range(x_key)
        x_index, x_inside = self._main_binning_for(x_key, *x_range, x_bins)
        if inside is not None:
            x_inside = x_inside & inside
        if not x_inside.any():
            return drawn

        if is_2d:
            y_key = self.y_var.get()
            if y_key not in arrays:
                return drawn
            y_range = self._axis_range(y_key)
            y_index, y_inside = self._main_binning_for(y_key, *y_range, y_bins)
            values, x_edges, y_edges = co.histogram_2d(
                None, None, pz, (x_bins, y_bins), x_range, y_range, quantity,
                binning=(x_index, y_index, x_inside & y_inside),
            )
            # Counts and density can use a log color scale; the signed
            # asymmetry can't.
            log = self.log_var.get() and quantity in CURVE_QUANTITIES
            levels = data_levels(values, log) if self.auto_levels_var.get() else self._color_bar.levels()
            self._set_bar_levels(levels)
            self._color_bar.setLabel("right", f"log10 {quantity}" if log else quantity)
            image = pg.ImageItem(colorize(values, levels, log))
            image.setRect(QtCore.QRectF(
                x_edges[0], y_edges[0], x_edges[-1] - x_edges[0], y_edges[-1] - y_edges[0],
            ))
            self._main_plot.addItem(image)
            self._main_plot.setLabel("left", self._dataset.label(y_key))
            scale = "log" if log else "linear"
            self._main_plot.setTitle(f"{quantity} of {self._dataset.label(y_key)} against "
                                     f"{self._dataset.label(x_key)} ({scale} color)")
        else:
            centers, values = co.histogram_1d(
                None, pz, x_bins, *x_range, quantity, binning=(x_index, x_inside),
            )
            if quantity in CURVE_QUANTITIES:
                self._main_plot.addItem(pg.PlotDataItem(centers, values, pen=pg.mkPen(BLUE, width=1.5)))
            else:
                # One point per bin that has data; empty bins are left out.
                has_data = np.isfinite(values)
                self._main_plot.addItem(pg.ScatterPlotItem(
                    centers[has_data], values[has_data], symbol="o", size=6,
                    pen=pg.mkPen(BLUE), brush=pg.mkBrush(BLUE),
                ))
            self._main_plot.setLabel("left", quantity)
            self._main_plot.setTitle(f"{quantity} against {self._dataset.label(x_key)}")
        self._main_plot.setLabel("bottom", self._dataset.label(x_key))
        drawn.add(self._main_plot)
        return drawn

    def _axis_range(self, key):
        """An applied gate bounds its own axis; otherwise the data's range."""
        gate = self._gates.get(key)
        return (min(gate), max(gate)) if gate is not None else self._data_range(key)

    def _set_bar_levels(self, levels):
        """Levels onto the color bar without its signal bouncing back."""
        self._color_bar.blockSignals(True)
        self._color_bar.setLevels(values=tuple(float(v) for v in levels))
        self._color_bar.blockSignals(False)
