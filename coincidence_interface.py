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
from calibration_interface import BLUE, GREEN, BackgroundTask, PlotGrid, read_number, relax_titles, show_placeholder
from qt_plots import ZoomFocusViewBox, hist_to_rgba

EMPTY_TEXT = "No data — press Load"
MODE_2D = "2D heat map"
MODE_1D = "1D plot"
MODES = (MODE_2D, MODE_1D)


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
        self.bins_var = pvar(tk.StringVar, "bins", "200")
        self.log_var = pvar(tk.BooleanVar, "log_color", False)
        self.quantity_var = pvar(tk.StringVar, "quantity", co.QUANTITY_DENSITY)

        self.status_var = tk.StringVar(self, value="Idle")
        self.rows_var = tk.StringVar(self, value="--")
        self.gated_var = tk.StringVar(self, value="--")
        self.electrons_var = tk.StringVar(self, value="--")
        self.gate_info_var = tk.StringVar(self, value="No gates.")

        self._task = BackgroundTask(self)
        self._dataset = None
        self._gates = {}          # key -> (low, high) for enabled gates
        self._enabled = {}        # key -> BooleanVar
        self._regions = {}        # key -> LinearRegionItem
        self._small = {}          # key -> PlotItem
        self._redraw_after = None

        self._build_ui()
        for var in (self.mode_var, self.x_var, self.y_var, self.bins_var, self.log_var, self.quantity_var):
            var.trace_add("write", self._schedule_redraw)
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
            "pulse, so m/q is per coincidence; electrons with no ion have no m/q.",
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
        ui_style.add_form_row(plot, 3, "Bins:", ttk.Entry(plot, textvariable=self.bins_var, width=18))
        ui_style.add_form_row(plot, 4, "Quantity:", ttk.Combobox(
            plot, textvariable=self.quantity_var, values=co.QUANTITIES, state="readonly", width=18,
        ))
        ttk.Checkbutton(plot, text="Log color scale (2D)", variable=self.log_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ui_style.add_note(
            plot, 6,
            "Probability density normalizes the counts to integrate to 1 over the plotted range. The "
            "asymmetry is (forward − backward) / total per bin, by the sign of p_z; a bin with no rows "
            "is left empty, and the log color scale applies to counts and density only.",
            pady=(4, 4),
        )

        gates = ui_style.build_section(sidebar, 4, "Gates", pady=0)
        ui_style.add_note(
            gates, 0,
            "Tick a quantity under its plot to gate on it, then drag the shaded region or its edges. "
            "Each small plot shows the data gated by the other quantities only.",
            pady=(4, 4),
        )
        ui_style.add_note(gates, 1, textvariable=self.gate_info_var)
        row = ui_style.add_stat_rows(gates, [
            ("Electron rows:", self.electrons_var),
            ("Coincidence rows:", self.rows_var),
            ("Rows after gates:", self.gated_var),
        ], start_row=2)
        ui_style.add_note(gates, row, pady=(4, 4))

        self._main_area = main
        self._main_focused = False
        self._main_glw = pg.GraphicsLayoutWidget()
        self._main_plot = self._main_glw.addPlot(
            row=0, col=0, viewBox=ZoomFocusViewBox(on_focus=self._toggle_main_focus),
        )
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

    def _build_small_plots(self):
        """One plot, gate checkbox and region per available variable. The
        grid and the checkbox strip are replaced outright: a frame keeps the
        layout it was given, so its widgets can't simply be cleared out."""
        self._bottom._own_layout().removeWidget(self._grid.widget)
        self._grid.widget.setParent(None)
        self._grid.widget.deleteLater()
        self._checks.grid_remove()
        self._checks.deleteLater()

        self._grid = PlotGrid(on_focus_changed=self._on_small_focus)
        grid_into(self._grid.widget, self._bottom, row=0, column=0, sticky="nsew")
        self._checks = ttk.Frame(self._bottom)
        self._checks.grid(row=1, column=0, sticky="ew")
        self._small, self._regions, self._enabled = {}, {}, {}
        self._show_both()

        variables = self._dataset.variables if self._dataset is not None else co.VARIABLES
        for column, variable in enumerate(variables):
            plot = self._grid.add(0, column)
            plot.setTitle(self._dataset.label(variable.key) if self._dataset else variable.axis_label)
            plot.setLabel("bottom", variable.axis_label)
            self._small[variable.key] = plot

            region = pg.LinearRegionItem(brush=pg.mkBrush(70, 130, 180, 60), pen=pg.mkPen(GREEN, width=2))
            region.setZValue(10)
            region.setVisible(False)
            region.sigRegionChangeFinished.connect(
                lambda item, key=variable.key: self._on_region_changed(key, item)
            )
            plot.addItem(region, ignoreBounds=True)
            self._regions[variable.key] = region

            enabled = tk.BooleanVar(self, value=False)
            enabled.trace_add("write", lambda *_, key=variable.key: self._on_gate_toggled(key))
            self._enabled[variable.key] = enabled
            ttk.Checkbutton(self._checks, text=f"Gate {variable.label}", variable=enabled).grid(
                row=0, column=column, sticky="w", padx=8
            )
            self._checks.columnconfigure(column, weight=1)

        # Without this a title's text width becomes its plot's minimum width,
        # and the last columns get pushed out of the widget.
        relax_titles(tuple(self._small.values()))

        keys = [variable.key for variable in variables]
        self._x_combo.configure(values=keys)
        self._y_combo.configure(values=keys)
        if self.x_var.get() not in keys and keys:
            self.x_var.set(keys[0])
        if self.y_var.get() not in keys and keys:
            self.y_var.set(keys[min(2, len(keys) - 1)])

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
        self._gates = {}
        self._build_small_plots()
        self.electrons_var.set(str(dataset.n_electrons))
        self.rows_var.set(str(dataset.n_rows))
        names = ", ".join(variable.label for variable in dataset.variables)
        self.status_var.set(f"Loaded {dataset.n_rows} coincidence rows from {pathlib.Path(dataset.path).name} ({names}).")
        self._redraw()

    # ---- gates ------------------------------------------------------------------

    def _on_gate_toggled(self, key):
        if self._dataset is None:
            return
        if self._enabled[key].get():
            values = self._dataset.arrays[key]
            inside = co.gate_mask(self._dataset.arrays, self._gates, skip=key)
            values = values[inside] if inside is not None else values
            values = values[np.isfinite(values)]
            if values.size:
                lo, hi = (float(v) for v in np.percentile(values, [25, 75]))
            else:
                lo, hi = co.default_range(self._dataset.arrays[key])
            self._gates[key] = (lo, hi)
            self._regions[key].setRegion((lo, hi))
        else:
            self._gates.pop(key, None)
        self._regions[key].setVisible(self._enabled[key].get())
        self._redraw()

    def _on_region_changed(self, key, item):
        if self._enabled.get(key) is not None and self._enabled[key].get():
            self._gates[key] = tuple(float(v) for v in item.getRegion())
            self._redraw()

    def clear_gates(self):
        for key, enabled in self._enabled.items():
            self._gates.pop(key, None)
            if enabled.get():
                enabled.set(False)  # hides the region and redraws
            self._regions[key].setVisible(False)
        self._gates = {}
        self._redraw()

    # ---- drawing ----------------------------------------------------------------

    def _schedule_redraw(self, *_):
        if self._redraw_after is not None:
            self.after_cancel(self._redraw_after)
        self._redraw_after = self.after(300, self._redraw)

    def _redraw(self):
        self._redraw_after = None
        is_2d = self.mode_var.get() == MODE_2D
        self._y_label.setVisible(is_2d)
        self._y_combo.setVisible(is_2d)
        self._main_plot.clear()
        for plot in self._small.values():
            for item in list(plot.items):
                if not isinstance(item, pg.LinearRegionItem):
                    plot.removeItem(item)

        self.gate_info_var.set(", ".join(
            f"{co.BY_KEY[key].label} in [{lo:.4g}, {hi:.4g}]" for key, (lo, hi) in sorted(self._gates.items())
        ) or "No gates.")

        drawn = set()
        if self._dataset is not None:
            try:
                drawn = self._draw_all(is_2d)
            except ValueError as exc:
                self.status_var.set(f"Invalid setting: {exc}")
        self._main_plot.autoRange()
        for plot in self._small.values():
            plot.autoRange()
            show_placeholder(plot, EMPTY_TEXT, plot not in drawn)
        show_placeholder(self._main_plot, EMPTY_TEXT, self._main_plot not in drawn)

    def _draw_all(self, is_2d):
        drawn = set()
        arrays = self._dataset.arrays
        bins = read_number(self.bins_var, "Bins", integer=True, positive=True)
        quantity = self.quantity_var.get()
        pz = arrays["pz"]

        # Small plots: counts, gated by every gate except the plot's own.
        for key, plot in self._small.items():
            inside = co.gate_mask(arrays, self._gates, skip=key)
            values = arrays[key] if inside is None else arrays[key][inside]
            lo, hi = co.default_range(arrays[key])
            centers, counts = co.histogram_1d(
                values, pz if inside is None else pz[inside], min(bins, 200), lo, hi, co.QUANTITY_COUNTS,
            )
            plot.addItem(pg.PlotDataItem(centers, counts, pen=pg.mkPen(BLUE, width=1)))
            plot.setLabel("left", "Counts")
            drawn.add(plot)

        inside = co.gate_mask(arrays, self._gates)
        self.gated_var.set(str(int(inside.sum()) if inside is not None else self._dataset.n_rows))
        x_key = self.x_var.get()
        if x_key not in arrays:
            return drawn
        x = arrays[x_key] if inside is None else arrays[x_key][inside]
        gated_pz = pz if inside is None else pz[inside]
        if x.size == 0:
            return drawn

        if is_2d:
            y_key = self.y_var.get()
            if y_key not in arrays:
                return drawn
            y = arrays[y_key] if inside is None else arrays[y_key][inside]
            values, x_edges, y_edges = co.histogram_2d(
                x, y, gated_pz, bins, co.default_range(arrays[x_key]), co.default_range(arrays[y_key]), quantity,
            )
            # Counts and density can use a log color scale; the signed
            # asymmetry can't.
            log = self.log_var.get() and quantity != co.QUANTITY_ASYMMETRY
            image = pg.ImageItem(hist_to_rgba(np.nan_to_num(values), log=log))
            image.setRect(QtCore.QRectF(
                x_edges[0], y_edges[0], x_edges[-1] - x_edges[0], y_edges[-1] - y_edges[0],
            ))
            self._main_plot.addItem(image)
            self._main_plot.setLabel("left", self._dataset.label(y_key))
            scale = "log" if log else "linear"
            self._main_plot.setTitle(f"{quantity} of {self._dataset.label(y_key)} against "
                                     f"{self._dataset.label(x_key)} ({scale} color)")
        else:
            centers, values = co.histogram_1d(x, gated_pz, bins, *co.default_range(arrays[x_key]), quantity)
            self._main_plot.addItem(pg.PlotDataItem(centers, values, pen=pg.mkPen(BLUE, width=1.5), connect="finite"))
            self._main_plot.setLabel("left", quantity)
            self._main_plot.setTitle(f"{quantity} against {self._dataset.label(x_key)}")
        self._main_plot.setLabel("bottom", self._dataset.label(x_key))
        drawn.add(self._main_plot)
        return drawn
