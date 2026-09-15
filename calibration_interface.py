"""Analysis calibration tabs.

MomentumCalibrationInterface derives a 3D electron momentum calibration
(electron_calibration.py, following electron_momentum_calibration.md) from
one circularly polarized dataset or an S + P pair of datasets.
MassCalibrationInterface fits an ion time of flight -> m/q calibration to
peaks the user picks and labels in the i-ToF spectrum.

Both tabs pick peaks the same way: Ctrl+click on a spectrum adds a peak,
and Shift+click removes the nearest one. Ring peaks stay exactly where the
user clicks, since they set the ATI ring radii for the energy scale; time
and i-ToF peaks snap to the nearest maximum.
Plain clicks and drags keep the usual rectangle zoom.
"""
import json
import pathlib
import threading
import types
from queue import Queue, Empty

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

import qtk as tk
from qtk import ttk, filedialog, messagebox, grid_into

import app_settings
import electron_calibration as ec
import mass_calibration as mc
import ui_style
from qt_plots import ZoomFocusViewBox, hist_to_rgba, mpl_color

BLUE = mpl_color("tab:blue")
ORANGE = mpl_color("tab:orange")
GREEN = mpl_color("tab:green")
RED = mpl_color("tab:red")
GREY = (150, 150, 150)

EMPTY_TEXT = "No data — press Load Data"

CLICK_ZOOM = "Zoom only"
CLICK_CENTER = "Set center"
CLICK_ANGLE = "Set angle"
CLICK_TC = "Set tc"
CLICK_MODES = (CLICK_ZOOM, CLICK_CENTER, CLICK_ANGLE, CLICK_TC)


# ---- shared helpers ---------------------------------------------------------

KIND_MOMENTUM = "momentum"
KIND_MQ = "mq"


def latest_path_key(kind):
    return f"calibration.latest_{kind}_path"


class CalibrationHub:
    """The latest calibration from each calibration tab, for the tabs that
    apply them. A tab publishes whenever its calibration changes (fit, edit,
    save, load). The path of a saved or loaded file is also persisted, so the
    latest file can be reloaded after a restart."""

    def __init__(self):
        self._latest = {}
        self._listeners = []

    def publish(self, kind, calibration, path=None):
        self._latest[kind] = (calibration, path)
        if path:
            app_settings.set(latest_path_key(kind), str(path))
        for listener in list(self._listeners):
            listener(kind, calibration, path)

    def latest(self, kind):
        """(calibration, path or None), or None if nothing was published."""
        return self._latest.get(kind)

    def subscribe(self, listener):
        self._listeners.append(listener)


def calibration_key(calibration, path):
    """Identity of a published calibration, ignoring when it was made."""
    data = {k: v for k, v in calibration.to_dict().items() if k not in ("created", "meta")}
    return json.dumps(data, sort_keys=True, default=str), str(path)


class BackgroundTask:
    """Runs a function off the UI thread and hands its result, or the
    exception it raised, to a callback on the UI thread."""

    def __init__(self, owner):
        self._owner = owner
        self._queue = Queue()
        self._thread = None
        self._poll()

    def is_running(self):
        return bool(self._thread and self._thread.is_alive())

    def start(self, fn, on_done, on_error):
        def run():
            try:
                result = fn()
            except Exception as exc:
                self._queue.put((on_error, exc))
            else:
                self._queue.put((on_done, result))

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def _poll(self):
        try:
            while True:
                callback, value = self._queue.get_nowait()
                callback(value)
        except Empty:
            pass
        self._owner.after(200, self._poll)


def connect_clicks(owner, plot, on_peak=None, on_plain=None):
    """Left clicks on a plot's data area, in data coordinates. Ctrl+click
    calls on_peak(x, remove=False) and Shift+click on_peak(x, remove=True).
    A plain click calls on_plain(x, y), but only after the double-click
    interval, so the first half of a double-click (focus) doesn't also count."""
    pending = []

    def cancel_pending():
        while pending:
            owner.after_cancel(pending.pop())

    def on_click(ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        if ev.double():
            # The layout may already have changed under the cursor, so cancel
            # regardless of which plot was hit.
            cancel_pending()
            return
        if plot.scene() is None or not plot.vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        point = plot.vb.mapSceneToView(ev.scenePos())
        mods = ev.modifiers()
        if mods & (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
            if on_peak is not None:
                ev.accept()
                on_peak(point.x(), bool(mods & QtCore.Qt.ShiftModifier))
        elif on_plain is not None:
            cancel_pending()

            def fire(x=point.x(), y=point.y()):
                pending.clear()
                on_plain(x, y)

            pending.append(owner.after(QtWidgets.QApplication.doubleClickInterval(), fire))

    plot.scene().sigMouseClicked.connect(on_click)


class PlotGrid:
    """Plots in a GraphicsLayoutWidget. Double-clicking a plot shows it alone
    across the whole widget; double-clicking it again restores the grid.
    `on_focus_changed(plot or None)` runs after each toggle, for owners that
    also want to give the widget more room."""

    def __init__(self, on_focus_changed=None):
        self.widget = pg.GraphicsLayoutWidget()
        self._cells = []
        self._focused = None
        self._on_focus_changed = on_focus_changed

    def add(self, row, col, rowspan=1, colspan=1):
        plot = self.widget.addPlot(
            row=row, col=col, rowspan=rowspan, colspan=colspan,
            viewBox=ZoomFocusViewBox(on_focus=lambda: self.toggle_focus(plot)),
        )
        self._cells.append((plot, row, col, rowspan, colspan))
        return plot

    def toggle_focus(self, plot):
        layout = self.widget.ci
        # Only the plots currently shown are in the layout.
        for shown in list(layout.items):
            layout.removeItem(shown)
        if self._focused is None:
            # Spanning every row and column keeps the grid's stretch factors
            # from reserving space for the hidden plots.
            rows = max(r + rs for _, r, _, rs, _ in self._cells)
            cols = max(c + cs for _, _, c, _, cs in self._cells)
            layout.addItem(plot, 0, 0, rows, cols)
            self._focused = plot
        else:
            for cell_plot, row, col, rowspan, colspan in self._cells:
                layout.addItem(cell_plot, row, col, rowspan, colspan)
            self._focused = None
        if self._on_focus_changed is not None:
            self._on_focus_changed(self._focused)


def read_number(var, label, integer=False, positive=False, allow_blank=False):
    text = str(var.get()).strip()
    if allow_blank and not text:
        return None
    try:
        value = int(float(text)) if integer else float(text)
    except ValueError:
        raise ValueError(f"{label} must be a number.") from None
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive.")
    return value


def set_visible(widgets, visible):
    for widget in widgets:
        getattr(widget, "_pad_container", widget).setVisible(visible)


def add_label(parent, row, text):
    label = ttk.Label(parent, text=text)
    label.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
    return label


def add_image(plot, x, y, extent_x, extent_y, bins=256, log=True):
    hist, _, _ = np.histogram2d(x, y, bins=bins, range=[extent_x, extent_y])
    img = pg.ImageItem(hist_to_rgba(hist, log=log))
    img.setRect(QtCore.QRectF(extent_x[0], extent_y[0], extent_x[1] - extent_x[0], extent_y[1] - extent_y[0]))
    plot.addItem(img)
    return hist


def add_circle(plot, xc, yc, radius, color, style=QtCore.Qt.SolidLine, width=1):
    phi = np.linspace(0, 2 * np.pi, 200)
    plot.addItem(pg.PlotDataItem(
        xc + radius * np.cos(phi), yc + radius * np.sin(phi), pen=pg.mkPen(color, width=width, style=style)
    ))


def add_marker_line(plot, x, text, color):
    plot.addItem(pg.InfiniteLine(
        pos=x, angle=90, pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine),
        label=text, labelOpts={"position": 0.92, "color": color},
    ))


def add_spectrum_curves(plot, spectrum):
    """Counts, the smoothed counts, and the clipped negative second derivative
    that peak finding uses, scaled so its maximum matches the smoothed counts."""
    plot.addItem(pg.PlotDataItem(spectrum.centers, spectrum.counts, pen=pg.mkPen(GREY, width=1), name="counts"))
    plot.addItem(pg.PlotDataItem(spectrum.centers, spectrum.smoothed, pen=pg.mkPen(BLUE, width=1.5), name="smoothed"))
    curve_max = float(spectrum.curve.max()) if spectrum.curve.size else 0.0
    if curve_max > 0:
        scale = float(spectrum.smoothed.max()) / curve_max
        plot.addItem(pg.PlotDataItem(
            spectrum.centers, spectrum.curve * scale, pen=pg.mkPen(GREEN, width=1.2),
            name=f"−d² (×{scale:.3g})",
        ))


def browse_cv4(owner, var, title):
    initial = var.get().strip()
    chosen = filedialog.askopenfilename(
        title=title,
        initialdir=str(pathlib.Path(initial).parent) if initial else None,
        filetypes=[("cv4", "*.cv4")],
        parent=owner,
    )
    if chosen:
        var.set(chosen)


def show_placeholder(plot, text, is_empty):
    ui_style.show_empty_placeholder(plot, ui_style.add_empty_placeholder(plot, text), is_empty)


def relax_titles(plots):
    """By default a title's text width becomes its plot's minimum width, which
    can push a grid of plots wider than the tab and clip the last column.
    LabelItem re-applies that minimum on every setText() and again later, so
    replace its updateMin() once per plot: keep the text height, drop the
    width. The title still spans its column (no maximum)."""
    for plot in plots:
        label = plot.titleLabel

        def update_min(label=label):
            height = label.itemRect().height()
            label._sizeHint = {
                QtCore.Qt.MinimumSize: (0, height),
                QtCore.Qt.PreferredSize: (0, height),
                QtCore.Qt.MaximumSize: (-1, -1),
                QtCore.Qt.MinimumDescent: (0, 0),
            }
            label.setMinimumWidth(0)
            label.setMinimumHeight(height)
            label.updateGeometry()

        label.updateMin = update_min
        update_min()


# ---- electron momentum calibration --------------------------------------------

class MomentumCalibrationInterface(ttk.Frame):
    """Derives a momentum calibration step by step (doc section 2): center
    and angle, ring and time peaks, the per-side z(dt) fits and the energy
    scale k, then checks the result and exports it as JSON."""

    def __init__(self, parent, hub=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._hub = hub
        self._calibration_path = None
        self._published = None

        def pvar(cls, key, default):
            return app_settings.persistent_var(self, cls, f"momentum_calibration.{key}", default)

        self.mode_var = pvar(tk.StringVar, "mode", ec.MODE_CIRCULAR)
        self.time_path_var = pvar(tk.StringVar, "time_path", "")
        self.p_path_var = pvar(tk.StringVar, "p_path", "")
        self.time_source_var = pvar(tk.StringVar, "time_source", ec.TIME_SOURCE_ETOF)
        self.single_hit_var = pvar(tk.BooleanVar, "single_hit", True)
        self.dither_var = pvar(tk.StringVar, "dither_ns", "0.26")

        self.xc_var = pvar(tk.StringVar, "xc", "128")
        self.yc_var = pvar(tk.StringVar, "yc", "128")
        self.tc_var = pvar(tk.StringVar, "tc", "500")
        self.angle_var = pvar(tk.StringVar, "angle_deg", "0")
        self.on_axis_var = pvar(tk.StringVar, "on_axis_px", "2")
        self.slice_ns_var = pvar(tk.StringVar, "slice_ns", "1")

        self.ring_width_var = pvar(tk.StringVar, "ring_width_px", "2")
        self.r_max_var = pvar(tk.StringVar, "r_max_px", "128")
        self.r_bin_var = pvar(tk.StringVar, "r_bin_px", "0.5")
        self.t_window_var = pvar(tk.StringVar, "t_window_ns", "50")
        self.t_bin_var = pvar(tk.StringVar, "t_bin_ns", "0.26")
        self.t_ignore_var = pvar(tk.StringVar, "t_ignore_ns", "1")
        # Radius smoothing keeps the keys the single shared setting used; time
        # smoothing starts from the same values.
        self.r_savgol_var = pvar(tk.StringVar, "savgol_bins", "7")
        self.r_gauss_var = pvar(tk.StringVar, "gauss_bins", "1")
        self.t_savgol_var = pvar(tk.StringVar, "t_savgol_bins", self.r_savgol_var.get())
        self.t_gauss_var = pvar(tk.StringVar, "t_gauss_bins", self.r_gauss_var.get())
        self.n_peaks_var = pvar(tk.StringVar, "peaks_per_side", "6")

        self.early_terms_var = pvar(tk.StringVar, "early_terms", "1")
        self.late_terms_var = pvar(tk.StringVar, "late_terms", "2")
        self.wavelength_var = pvar(tk.StringVar, "wavelength_nm", "800")
        self.k_var = pvar(tk.StringVar, "k", "")

        self.symmetrize_var = pvar(tk.BooleanVar, "symmetrize", False)
        self.validation_var = pvar(tk.StringVar, "validation_dataset", "S")
        self.name_var = tk.StringVar(self, value=ec.default_calibration_name())
        self.click_mode_var = tk.StringVar(self, value=CLICK_ZOOM)

        self.status_var = tk.StringVar(self, value="Idle")
        self.center_info_var = tk.StringVar(self, value="")
        self.peak_info_var = tk.StringVar(self, value="")
        self.fit_info_var = tk.StringVar(self, value="")
        self.export_info_var = tk.StringVar(self, value="")

        self._task = BackgroundTask(self)
        self._events = {}
        self._center_estimates = {}
        self._r_peaks = []
        self._t_peaks = []
        self._r_spectrum = None
        self._t_spectrum = None
        self._early_fit = None
        self._late_fit = None
        self._k_fit = None
        self._redraw_after = None

        self._build_ui()
        self._on_mode_changed(initial=True)

        for var in (self.xc_var, self.yc_var, self.angle_var):
            var.trace_add("write", self._on_center_changed)
        self.tc_var.trace_add("write", self._on_tc_changed)
        for var in (
            self.on_axis_var, self.slice_ns_var, self.ring_width_var, self.r_max_var, self.r_bin_var,
            self.t_window_var, self.t_bin_var, self.r_savgol_var, self.r_gauss_var,
            self.t_savgol_var, self.t_gauss_var, self.k_var,
            self.symmetrize_var, self.validation_var,
        ):
            var.trace_add("write", self._schedule_redraw)
        self._redraw()

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.rowconfigure(0, weight=1)
        sidebar.rowconfigure(8, weight=1)

        ui_style.build_button_bar(sidebar, 0, [
            ("Load Data", self.load_data), ("Find Peaks", self.find_peaks), ("Fit", self.fit),
        ])
        ui_style.StatusBlock(sidebar, 1, self.status_var)

        data = ui_style.build_section(sidebar, 2, "Datasets")
        modes = ttk.Frame(data)
        modes.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Radiobutton(
            modes, text="One circularly polarized dataset", variable=self.mode_var,
            value=ec.MODE_CIRCULAR, command=self._on_mode_changed,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            modes, text="S and P datasets", variable=self.mode_var,
            value=ec.MODE_S_P, command=self._on_mode_changed,
        ).grid(row=1, column=0, sticky="w")
        self._time_path_label = add_label(data, 1, "Dataset:")
        ui_style.build_path_field(data, self.time_path_var, self._browse_time).grid(row=1, column=1, sticky="ew", pady=4)
        self._p_path_label = add_label(data, 2, "P dataset:")
        self._p_path_field = ui_style.build_path_field(data, self.p_path_var, self._browse_p)
        self._p_path_field.grid(row=2, column=1, sticky="ew", pady=4)
        ui_style.add_form_row(data, 3, "Time source:", ttk.Combobox(
            data, textvariable=self.time_source_var, values=ec.TIME_SOURCES, state="readonly", width=18,
        ))
        ui_style.add_form_row(data, 4, "Time dither (ns):", ttk.Entry(data, textvariable=self.dither_var, width=18))
        ttk.Checkbutton(
            data, text="Only use pulses with a single hit", variable=self.single_hit_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._mode_note = ui_style.add_note(data, 6, pady=(4, 4))

        center = ui_style.build_section(sidebar, 3, "Center and angle")
        ui_style.add_form_row(center, 0, "Center x (px):", ttk.Entry(center, textvariable=self.xc_var, width=18))
        ui_style.add_form_row(center, 1, "Center y (px):", ttk.Entry(center, textvariable=self.yc_var, width=18))
        ui_style.add_form_row(center, 2, "Time zero tc (ns):", ttk.Entry(center, textvariable=self.tc_var, width=18))
        ui_style.add_form_row(center, 3, "Angle (deg):", ttk.Entry(center, textvariable=self.angle_var, width=18))
        ui_style.build_button_bar(center, 4, [
            ("Estimate Center", self.estimate_center),
            ("Estimate tc", self.estimate_tc),
            ("Estimate Angle", self.estimate_angle),
        ], pady=(4, 6), columnspan=2)
        ui_style.add_form_row(center, 5, "Plot clicks:", ttk.Combobox(
            center, textvariable=self.click_mode_var, values=CLICK_MODES, state="readonly", width=18,
        ))
        ui_style.add_form_row(center, 6, "On-axis radius (px):", ttk.Entry(center, textvariable=self.on_axis_var, width=18))
        ui_style.add_form_row(center, 7, "Time slice |Δt| < (ns):", ttk.Entry(center, textvariable=self.slice_ns_var, width=18))
        ui_style.add_note(
            center, 8,
            "The center estimate comes from ring symmetry; center of mass and median are shown for comparison. "
            "With Plot clicks set, click the detector image or the thin time slice to place the center, or "
            "to point the orange symmetry axis through the click. Click the arrival-time or t vs x' plot "
            "(or drag the red line on either) to set tc between the early and late peaks. "
            "A wrong center shows up as rings that aren't circular. Double-click any plot to show it alone.",
            pady=(4, 4),
        )
        ui_style.add_note(center, 9, textvariable=self.center_info_var)

        peaks = ui_style.build_section(sidebar, 4, "Peaks")
        rows = [
            ("Ring slice half-width (px):", self.ring_width_var),
            ("Max radius (px):", self.r_max_var),
            ("Radius bin (px):", self.r_bin_var),
            ("Time window ± (ns):", self.t_window_var),
            ("Time bin (ns):", self.t_bin_var),
            ("Ignore |Δt| below (ns):", self.t_ignore_var),
            ("Peaks per side:", self.n_peaks_var),
        ]
        for row, (label, var) in enumerate(rows):
            ui_style.add_form_row(peaks, row, label, ttk.Entry(peaks, textvariable=var, width=18))

        # Smoothing per axis, as a small settings table (bins of that axis).
        smoothing = ttk.Frame(peaks)
        smoothing.grid(row=len(rows), column=0, columnspan=2, sticky="ew", pady=(4, 4))
        ttk.Label(smoothing, text="Smoothing (bins)").grid(row=0, column=0, sticky="w", padx=(0, 4), pady=(4, 2))
        ttk.Label(smoothing, text="S–G window").grid(row=0, column=1, pady=(4, 2))
        ttk.Label(smoothing, text="Gaussian sigma").grid(row=0, column=2, pady=(4, 2))
        for row_idx, (label, savgol_var, gauss_var) in enumerate(
            (("Radius", self.r_savgol_var, self.r_gauss_var), ("Time", self.t_savgol_var, self.t_gauss_var)),
            start=1,
        ):
            ttk.Label(smoothing, text=label).grid(row=row_idx, column=0, sticky="w", padx=(0, 4), pady=1)
            ttk.Entry(smoothing, textvariable=savgol_var, width=9).grid(row=row_idx, column=1, padx=2, pady=1)
            ttk.Entry(smoothing, textvariable=gauss_var, width=9).grid(row=row_idx, column=2, padx=2, pady=1)
        smoothing.columnconfigure(3, weight=1)

        ui_style.add_note(
            peaks, len(rows) + 1,
            "Find Peaks keeps the most prominent peaks of the smoothed negative second derivative on each "
            "side. Ctrl+click a spectrum to add a peak, Shift+click to remove the nearest. A ring peak "
            "stays where you click, and those radii set the energy scale; time peaks snap to the nearest "
            "maximum. Rings pair with time peaks in order from the center, so both lists must start at "
            "the same ring.",
            pady=(4, 4),
        )
        ui_style.add_note(peaks, len(rows) + 2, textvariable=self.peak_info_var)

        fit = ui_style.build_section(sidebar, 5, "Fit")
        ui_style.add_form_row(fit, 0, "Early polynomial terms:", ttk.Entry(fit, textvariable=self.early_terms_var, width=18))
        ui_style.add_form_row(fit, 1, "Late polynomial terms:", ttk.Entry(fit, textvariable=self.late_terms_var, width=18))
        ui_style.add_form_row(fit, 2, "Wavelength (nm):", ttk.Entry(fit, textvariable=self.wavelength_var, width=18))
        ui_style.add_form_row(fit, 3, "k (eV/px²):", ttk.Entry(fit, textvariable=self.k_var, width=18))
        ui_style.add_note(
            fit, 4,
            "Fit sets k from the ring spacing, assuming the rings are consecutive ATI orders. "
            "Edit k afterwards to override it.",
            pady=(4, 4),
        )
        ui_style.add_note(fit, 5, textvariable=self.fit_info_var)

        check = ui_style.build_section(sidebar, 6, "Validation")
        ttk.Checkbutton(
            check, text="Symmetrize (only for inversion-symmetric physics)", variable=self.symmetrize_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._validation_label = add_label(check, 1, "Dataset:")
        self._validation_combo = ttk.Combobox(
            check, textvariable=self.validation_var, values=("S", "P"), state="readonly", width=18,
        )
        self._validation_combo.grid(row=1, column=1, sticky="ew", pady=4)

        export = ui_style.build_section(sidebar, 7, "Export", pady=0)
        ui_style.add_form_row(export, 0, "Name:", ttk.Entry(export, textvariable=self.name_var))
        ui_style.build_button_bar(
            export, 1, [("Save As...", self.save_calibration), ("Load...", self.load_calibration)],
            pady=(4, 6), columnspan=2,
        )
        ui_style.add_note(export, 2, textvariable=self.export_info_var)

        self._grid = PlotGrid()
        self._glw = self._grid.widget
        self._plot_image = self._grid.add(0, 0)
        self._plot_slice = self._grid.add(0, 1)
        self._plot_tx = self._grid.add(0, 2)
        self._plot_check = self._grid.add(0, 3)
        self._plot_rings = self._grid.add(1, 0)
        self._plot_times = self._grid.add(1, 1)
        self._plot_fit = self._grid.add(1, 2)
        self._plot_k = self._grid.add(1, 3)
        self._plot_energy = self._grid.add(2, 0, colspan=4)
        relax_titles(self._all_plots())
        for plot in (self._plot_image, self._plot_slice, self._plot_check):
            plot.setAspectLocked(True)
        # Without equal stretch the aspect-locked images take most of the
        # width and squeeze the other columns.
        for col in range(4):
            self._glw.ci.layout.setColumnStretchFactor(col, 1)
        for row in range(3):
            self._glw.ci.layout.setRowStretchFactor(row, 1)
        self._plot_image.addLegend()
        self._plot_fit.addLegend()
        self._plot_rings.addLegend(offset=(-10, 10))
        self._plot_times.addLegend(offset=(-10, 10))
        grid_into(self._glw, main, row=0, column=0, sticky="nsew")

        connect_clicks(self, self._plot_image, on_plain=self._on_image_click)
        connect_clicks(self, self._plot_slice, on_plain=self._on_image_click)
        connect_clicks(self, self._plot_tx, on_plain=self._on_tx_click)
        connect_clicks(self, self._plot_rings, on_peak=self._on_ring_click)
        connect_clicks(self, self._plot_times, on_peak=self._on_time_click, on_plain=self._on_times_plain_click)

    def _all_plots(self):
        return (
            self._plot_image, self._plot_slice, self._plot_tx, self._plot_check, self._plot_rings,
            self._plot_times, self._plot_fit, self._plot_k, self._plot_energy,
        )

    def _browse_time(self):
        browse_cv4(self, self.time_path_var, "Choose dataset")

    def _browse_p(self):
        browse_cv4(self, self.p_path_var, "Choose P dataset")

    # ---- state helpers ------------------------------------------------------

    def _is_circular(self):
        return self.mode_var.get() != ec.MODE_S_P

    def _image_events(self):
        """Dataset for the center, angle and ring radii: P, or the circular one."""
        return self._events.get("time" if self._is_circular() else "p")

    def _time_events(self):
        """Dataset for tc and the time peaks: S, or the circular one."""
        return self._events.get("time")

    def _validation_events(self):
        if not self._is_circular() and self.validation_var.get() == "P":
            return self._events.get("p")
        return self._events.get("time")

    def _on_mode_changed(self, initial=False):
        circular = self._is_circular()
        self._time_path_label.configure(text="Dataset:" if circular else "S dataset:")
        set_visible([self._p_path_label, self._p_path_field, self._validation_label, self._validation_combo], not circular)
        self._mode_note.setText(
            "Circular: the angle is the propagation axis. Ring radii come from the in-plane axis "
            "perpendicular to it, which is equivalent to the ToF axis."
            if circular else
            "S (polarization along ToF) gives tc and the time peaks. P (polarization in the detector "
            "plane) gives the center, the angle (polarization axis) and the ring radii along it."
        )
        if not initial:
            self._events = {}
            self._clear_fits()
            self.status_var.set("Mode changed. Press Load Data.")
            self._schedule_redraw()

    def _clear_fits(self):
        self._early_fit = None
        self._late_fit = None
        self._k_fit = None
        self._calibration_path = None
        self.fit_info_var.set("")

    def _publish(self, calibration):
        """Hands a changed calibration to the hub (see CalibrationHub)."""
        if self._hub is None or calibration is None:
            return
        key = calibration_key(calibration, self._calibration_path)
        if key != self._published:
            self._published = key
            self._hub.publish(KIND_MOMENTUM, calibration, self._calibration_path)

    def _on_center_changed(self, *_):
        # Ring positions are measured from the center along the axis, so
        # they no longer apply.
        if self._r_peaks:
            self._r_peaks = []
            self.peak_info_var.set("Center or angle changed, so the ring peaks were cleared. Press Find Peaks.")
        self._clear_fits()
        self._schedule_redraw()

    def _on_tc_changed(self, *_):
        self._clear_fits()
        self._schedule_redraw()

    def _schedule_redraw(self, *_):
        if self._redraw_after is not None:
            self.after_cancel(self._redraw_after)
        self._redraw_after = self.after(300, self._redraw)

    def _settings(self):
        s = types.SimpleNamespace()
        s.xc = read_number(self.xc_var, "Center x")
        s.yc = read_number(self.yc_var, "Center y")
        s.tc = read_number(self.tc_var, "Time zero")
        s.theta = np.radians(read_number(self.angle_var, "Angle"))
        s.on_axis = read_number(self.on_axis_var, "On-axis radius", positive=True)
        s.slice_ns = read_number(self.slice_ns_var, "Time slice", positive=True)
        s.ring_width = read_number(self.ring_width_var, "Ring slice half-width", positive=True)
        s.r_max = read_number(self.r_max_var, "Max radius", positive=True)
        s.r_bin = read_number(self.r_bin_var, "Radius bin", positive=True)
        s.t_window = read_number(self.t_window_var, "Time window", positive=True)
        s.t_bin = read_number(self.t_bin_var, "Time bin", positive=True)
        s.t_ignore = read_number(self.t_ignore_var, "Ignore |Δt| below")
        s.r_savgol = read_number(self.r_savgol_var, "Radius Savitzky–Golay window", integer=True)
        s.r_gauss = read_number(self.r_gauss_var, "Radius Gaussian sigma")
        s.t_savgol = read_number(self.t_savgol_var, "Time Savitzky–Golay window", integer=True)
        s.t_gauss = read_number(self.t_gauss_var, "Time Gaussian sigma")
        s.n_peaks = read_number(self.n_peaks_var, "Peaks per side", integer=True, positive=True)
        return s

    def _ring_spectrum(self, s):
        u = ec.in_plane_slice(
            self._image_events(), s.xc, s.yc, s.tc, s.theta, self._is_circular(), s.ring_width, s.slice_ns,
        )
        return ec.make_spectrum(u, -s.r_max, s.r_max, s.r_bin, s.r_savgol, s.r_gauss)

    def _time_spectrum(self, s):
        events = self._time_events()
        t = events.t[ec.on_axis_mask(events, s.xc, s.yc, s.on_axis)]
        return ec.make_spectrum(t, s.tc - s.t_window, s.tc + s.t_window, s.t_bin, s.t_savgol, s.t_gauss)

    def _ring_axis_angle(self, s):
        return s.theta + (np.pi / 2 if self._is_circular() else 0.0)

    # ---- loading ------------------------------------------------------------

    def load_data(self):
        if self._task.is_running():
            return
        circular = self._is_circular()
        paths = {"time": self.time_path_var.get().strip()}
        if not circular:
            paths["p"] = self.p_path_var.get().strip()
        for key, path in paths.items():
            name = "Dataset" if circular else ("S dataset" if key == "time" else "P dataset")
            if not path:
                self.status_var.set(f"{name} is required.")
                return
            if not pathlib.Path(path).is_file():
                self.status_var.set(f"{name} not found: {path}")
                return
        try:
            dither = read_number(self.dither_var, "Time dither")
        except ValueError as exc:
            self.status_var.set(f"Invalid setting: {exc}")
            return
        time_source = self.time_source_var.get()
        single_hit = self.single_hit_var.get()

        def work():
            return {
                key: ec.load_electron_events(path, time_source, single_hit, max(0.0, dither))
                for key, path in paths.items()
            }

        self.status_var.set("Loading...")
        self._task.start(work, self._on_loaded, lambda exc: self.status_var.set(f"Error: could not load data: {exc}"))

    def _on_loaded(self, events):
        self._events = events
        self._center_estimates = {}
        self._clear_fits()
        counts = ", ".join(f"{pathlib.Path(ev.path).name}: {ev.x.size} hits" for ev in events.values())
        self.status_var.set(f"Loaded {counts}.")
        self._schedule_redraw()

    # ---- step 1: center, tc, angle ------------------------------------------------

    def _require_events(self, events):
        if events is None:
            self.status_var.set("Load data first.")
            return False
        return True

    def estimate_center(self):
        events = self._image_events()
        if not self._require_events(events):
            return
        try:
            sym = ec.symmetry_center(events.x, events.y)
        except ValueError as exc:
            self.status_var.set(f"Error: {exc}")
            return
        self._center_estimates = {
            "ring symmetry": sym,
            "center of mass": ec.center_of_mass(events.x, events.y),
            "median": ec.median_center(events.x, events.y),
        }
        self.center_info_var.set("\n".join(
            f"{name}: ({x:.2f}, {y:.2f}) px" for name, (x, y) in self._center_estimates.items()
        ))
        self.xc_var.set(f"{sym[0]:.2f}")
        self.yc_var.set(f"{sym[1]:.2f}")
        self.status_var.set("Center set from ring symmetry.")

    def estimate_tc(self):
        events = self._time_events()
        if not self._require_events(events):
            return
        try:
            s = self._settings()
            tc = ec.estimate_time_zero(events, s.xc, s.yc, s.on_axis)
        except ValueError as exc:
            self.status_var.set(f"Error: {exc}")
            return
        self.tc_var.set(f"{tc:.3f}")
        self.status_var.set("tc set to the median on-axis arrival time. Refine it by dragging the red line.")

    def estimate_angle(self):
        events = self._image_events()
        if not self._require_events(events):
            return
        try:
            s = self._settings()
            theta = ec.principal_axis_angle(events, s.xc, s.yc, s.tc, s.slice_ns, s.r_max)
        except ValueError as exc:
            self.status_var.set(f"Error: {exc}")
            return
        if self._is_circular():
            # The thin slice is longest perpendicular to the propagation axis.
            theta = ec.wrap_axis_angle(theta + np.pi / 2)
        self.angle_var.set(f"{np.degrees(theta):.2f}")
        self.status_var.set("Angle set from the long axis of the thin time slice.")

    # ---- steps 2-3: peaks ----------------------------------------------------------

    def _on_image_click(self, x, y):
        """A plain click on the detector image or thin time slice."""
        mode = self.click_mode_var.get()
        if mode == CLICK_CENTER:
            self.xc_var.set(f"{x:.2f}")
            self.yc_var.set(f"{y:.2f}")
            self.status_var.set(f"Center set to ({x:.2f}, {y:.2f}) px.")
        elif mode == CLICK_ANGLE:
            try:
                xc = read_number(self.xc_var, "Center x")
                yc = read_number(self.yc_var, "Center y")
            except ValueError as exc:
                self.status_var.set(f"Invalid setting: {exc}")
                return
            if np.hypot(x - xc, y - yc) < 1e-6:
                return
            theta = ec.wrap_axis_angle(np.arctan2(y - yc, x - xc))
            self.angle_var.set(f"{np.degrees(theta):.2f}")
            self.status_var.set(f"Angle set to {np.degrees(theta):.2f} deg.")
        elif mode == CLICK_TC:
            self.status_var.set("Set tc by clicking the arrival-time or t vs x' plot.")

    def _on_times_plain_click(self, x, _y):
        self._set_tc_from_click(x)

    def _on_tx_click(self, _x, y):
        self._set_tc_from_click(y)

    def _set_tc_from_click(self, t):
        mode = self.click_mode_var.get()
        if mode == CLICK_TC:
            self.tc_var.set(f"{t:.3f}")
            self.status_var.set(f"tc set to {t:.3f} ns.")
        elif mode in (CLICK_CENTER, CLICK_ANGLE):
            self.status_var.set("Set the center and angle by clicking the detector image or thin time slice.")

    def find_peaks(self):
        if not self._require_events(self._image_events()) or not self._require_events(self._time_events()):
            return
        try:
            s = self._settings()
            rings = self._ring_spectrum(s)
            times = self._time_spectrum(s)
        except ValueError as exc:
            self.status_var.set(f"Invalid setting: {exc}")
            return
        self._r_peaks = (
            ec.find_spectrum_peaks(rings, s.n_peaks, -s.r_max, -s.on_axis)
            + ec.find_spectrum_peaks(rings, s.n_peaks, s.on_axis, s.r_max)
        )
        self._t_peaks = (
            ec.find_spectrum_peaks(times, s.n_peaks, s.tc - s.t_window, s.tc - s.t_ignore)
            + ec.find_spectrum_peaks(times, s.n_peaks, s.tc + s.t_ignore, s.tc + s.t_window)
        )
        self._clear_fits()
        self._update_peak_info()
        self.status_var.set("Peaks found. Check the pairing, then press Fit.")
        self._redraw()

    def _on_ring_click(self, x, remove):
        if self._r_spectrum is not None:
            # No snapping: the clicked radius is the user's ATI ring, and
            # both the z² fit and the energy scale use it as is.
            self._edit_peaks(self._r_peaks, self._r_spectrum, x, remove, snap=False)

    def _on_time_click(self, x, remove):
        if self._t_spectrum is not None:
            self._edit_peaks(self._t_peaks, self._t_spectrum, x, remove)

    def _edit_peaks(self, peaks, spectrum, x, remove, snap=True):
        """Adds a peak at x (snapped to the nearest maximum when `snap`), or
        removes the one nearest x, in place."""
        if remove:
            if not peaks:
                return
            peaks.pop(int(np.argmin([abs(p - x) for p, _ in peaks])))
        else:
            position, sigma = ec.snap_to_peak(spectrum, x)
            # The width near the click still weights the fit.
            peaks.append((position if snap else float(x), sigma))
            peaks.sort()
        self._clear_fits()
        self._update_peak_info()
        self._schedule_redraw()

    def _update_peak_info(self):
        lines = []
        r, _, half_diff = ec.ring_radii(self._r_peaks)
        if r.size:
            lines.append("Ring radii (px): " + ", ".join(f"{v:.2f}" for v in r))
            if np.any(np.isfinite(half_diff)):
                lines.append("Side offset (+ minus −)/2 (px): " + ", ".join(
                    "--" if not np.isfinite(v) else f"{v:+.2f}" for v in half_diff
                ))
        try:
            tc = read_number(self.tc_var, "Time zero")
        except ValueError:
            tc = None
        if tc is not None and self._t_peaks:
            early, late = ec.split_time_peaks(self._t_peaks, tc)
            if early:
                lines.append("Early Δt (ns): " + ", ".join(f"{dt:.2f}" for dt, _ in early))
            if late:
                lines.append("Late Δt (ns): " + ", ".join(f"{dt:.2f}" for dt, _ in late))
        self.peak_info_var.set("\n".join(lines))

    # ---- step 4 and 6: fit ---------------------------------------------------------

    def fit(self):
        try:
            tc = read_number(self.tc_var, "Time zero")
            early_terms = read_number(self.early_terms_var, "Early polynomial terms", integer=True, positive=True)
            late_terms = read_number(self.late_terms_var, "Late polynomial terms", integer=True, positive=True)
        except ValueError as exc:
            self.status_var.set(f"Invalid setting: {exc}")
            return
        r, r_sigma, _ = ec.ring_radii(self._r_peaks)
        if r.size == 0:
            self.status_var.set("Find ring peaks before fitting.")
            return
        early, late = ec.split_time_peaks(self._t_peaks, tc)
        self._early_fit = ec.fit_side(r, r_sigma, early, early_terms)
        self._late_fit = ec.fit_side(r, r_sigma, late, late_terms)
        if self._early_fit is None and self._late_fit is None:
            self.status_var.set("No time peaks to pair with the rings.")
            return

        lines = []
        self._k_fit = None
        if r.size >= 2:
            try:
                self._k_fit = ec.fit_energy_scale(r, ec.photon_energy_ev(read_number(self.wavelength_var, "Wavelength")))
            except ValueError as exc:
                lines.append(f"k not fit: {exc}")
        if self._k_fit is not None:
            self.k_var.set(f"{self._k_fit.k:.6g}")
            lines.append(
                f"k = {self._k_fit.k:.6g} eV/px² (ħω = {self._k_fit.photon_energy_ev:.4f} eV, "
                f"first ring {self._k_fit.k * r[0] ** 2:.3f} eV)."
            )
        elif r.size < 2:
            lines.append("Only one ring, so k was not fit. Enter k by hand.")

        for name, side in (("Early", self._early_fit), ("Late", self._late_fit)):
            if side is None:
                lines.append(f"{name}: no peaks, so z² = 0 on this side.")
                continue
            rms = float(np.sqrt(np.mean(side.residuals_px2 ** 2)))
            coeffs = ", ".join(f"{c:.5g}" for c in side.coeffs)
            lines.append(f"{name}: {side.n_pairs} pair(s), coefficients [{coeffs}], rms residual {rms:.3g} px².")
        if len(self._r_peaks) and (
            (self._early_fit and self._early_fit.n_pairs < r.size) or (self._late_fit and self._late_fit.n_pairs < r.size)
        ):
            lines.append("Rings beyond the last time peak were not used on that side.")
        calibration = self._current_calibration()
        if calibration is not None:
            early, late = calibration.valid_dt_range()
            lines.append(
                f"Valid for Δt from {early:.2f} to {late:+.2f} ns. Hits outside this range, "
                "where the time window ends or z² stops increasing, are not calibrated."
            )
        self.fit_info_var.set("\n".join(lines))
        self.status_var.set("Fit done. Check that the rings are circular and continuous across Δt = 0.")
        self._redraw()

    def _current_calibration(self):
        if self._early_fit is None and self._late_fit is None:
            return None
        try:
            k = read_number(self.k_var, "k", positive=True)
            dither = read_number(self.dither_var, "Time dither")
            s = self._settings()
        except ValueError:
            return None
        return ec.MomentumCalibration(
            name=self.name_var.get().strip() or ec.default_calibration_name(),
            center=(s.xc, s.yc, s.tc),
            angle=float(s.theta),
            k=k,
            early_coeffs=list(self._early_fit.coeffs) if self._early_fit else [],
            late_coeffs=list(self._late_fit.coeffs) if self._late_fit else [],
            time_source=self.time_source_var.get(),
            dither_ns=max(0.0, dither),
            dt_range=(-s.t_window, s.t_window),
        )

    # ---- step 7: export ----------------------------------------------------------

    def save_calibration(self):
        calibration = self._current_calibration()
        if calibration is None:
            self.status_var.set("Fit a calibration, and set k, before saving.")
            return
        circular = self._is_circular()
        calibration.meta = {
            "mode": self.mode_var.get(),
            "datasets": {"circular": self.time_path_var.get()} if circular else {
                "S": self.time_path_var.get(), "P": self.p_path_var.get(),
            },
            "single_hit_only": self.single_hit_var.get(),
            "wavelength_nm": self.wavelength_var.get(),
            "k_from_ring_spacing": self._k_fit is not None and abs(self._k_fit.k - calibration.k) <= 1e-6 * calibration.k,
            "early_terms": self.early_terms_var.get(),
            "late_terms": self.late_terms_var.get(),
            "ring_peaks_px": [list(p) for p in self._r_peaks],
            "time_peaks_ns": [list(p) for p in self._t_peaks],
            "settings": {
                "on_axis_px": self.on_axis_var.get(), "slice_ns": self.slice_ns_var.get(),
                "ring_width_px": self.ring_width_var.get(), "r_max_px": self.r_max_var.get(),
                "r_bin_px": self.r_bin_var.get(), "t_window_ns": self.t_window_var.get(),
                "t_bin_ns": self.t_bin_var.get(), "t_ignore_ns": self.t_ignore_var.get(),
                "r_savgol_bins": self.r_savgol_var.get(), "r_gauss_bins": self.r_gauss_var.get(),
                "t_savgol_bins": self.t_savgol_var.get(), "t_gauss_bins": self.t_gauss_var.get(),
            },
        }
        source = self.time_path_var.get().strip()
        chosen = filedialog.asksaveasfilename(
            title="Save momentum calibration",
            defaultextension=".json",
            initialfile=f"{calibration.name}.json",
            initialdir=str(pathlib.Path(source).parent) if source else None,
            filetypes=[("JSON", "*.json")],
            parent=self,
        )
        if not chosen:
            return
        try:
            calibration.save(chosen)
        except OSError as exc:
            self.status_var.set(f"Save failed: {exc}")
            return
        self._calibration_path = chosen
        self._publish(calibration)
        self.export_info_var.set(f"Saved {chosen}")
        self.status_var.set(f"Saved calibration to {chosen}.")

    def load_calibration(self):
        source = self.time_path_var.get().strip()
        chosen = filedialog.askopenfilename(
            title="Load momentum calibration",
            initialdir=str(pathlib.Path(source).parent) if source else None,
            filetypes=[("JSON", "*.json")],
            parent=self,
        )
        if not chosen:
            return
        try:
            calibration = ec.MomentumCalibration.load(chosen)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status_var.set(f"Load failed: {exc}")
            return
        meta = calibration.meta or {}
        xc, yc, tc = calibration.center
        self.xc_var.set(f"{xc:.4f}")
        self.yc_var.set(f"{yc:.4f}")
        self.tc_var.set(f"{tc:.4f}")
        self.angle_var.set(f"{np.degrees(calibration.angle):.4f}")
        self.name_var.set(calibration.name)
        if calibration.time_source in ec.TIME_SOURCES:
            self.time_source_var.set(calibration.time_source)
        self.dither_var.set(f"{calibration.dither_ns:g}")
        for key, var in (
            ("wavelength_nm", self.wavelength_var), ("early_terms", self.early_terms_var),
            ("late_terms", self.late_terms_var),
        ):
            if key in meta:
                var.set(meta[key])
        # The center/tc writes above cleared any peaks and fits.
        self._r_peaks = [tuple(p) for p in meta.get("ring_peaks_px", [])]
        self._t_peaks = [tuple(p) for p in meta.get("time_peaks_ns", [])]
        if self._r_peaks and self._t_peaks:
            self.fit()
        else:
            empty = np.array([])
            self._early_fit = ec.SideFit(list(calibration.early_coeffs), empty, empty, empty) if calibration.early_coeffs else None
            self._late_fit = ec.SideFit(list(calibration.late_coeffs), empty, empty, empty) if calibration.late_coeffs else None
        # Keep the saved k, whether it came from the ring spacing or by hand.
        self.k_var.set(f"{calibration.k:.6g}")
        self._update_peak_info()
        # The redraw below publishes the loaded calibration with this path.
        self._calibration_path = chosen
        self.export_info_var.set(f"Loaded {chosen}")
        self.status_var.set(f"Loaded calibration {calibration.name}.")
        self._schedule_redraw()

    # ---- drawing ------------------------------------------------------------

    def _redraw(self):
        self._redraw_after = None
        plots = self._all_plots()
        for plot in plots:
            plot.clear()
        self._r_spectrum = None
        self._t_spectrum = None

        try:
            s = self._settings()
        except ValueError as exc:
            self.status_var.set(f"Invalid setting: {exc}")
            s = None

        drawn = set()
        if s is not None:
            try:
                drawn = self._draw_all(s)
            except ValueError as exc:
                self.status_var.set(f"Error: {exc}")

        self._label_plots()
        for plot in plots:
            plot.autoRange()
        empty_text = {
            self._plot_check: "No calibration yet — press Fit",
            self._plot_energy: "No calibration yet — press Fit",
            self._plot_fit: "No fit yet — press Fit",
            self._plot_k: "No energy scale yet — press Fit",
        }
        for plot in plots:
            text = empty_text.get(plot, EMPTY_TEXT) if self._events or plot in empty_text else EMPTY_TEXT
            show_placeholder(plot, text, plot not in drawn)

    def _label_plots(self):
        circular = self._is_circular()
        image_name = "" if circular else " (P)"
        time_name = "" if circular else " (S)"
        self._plot_image.setTitle(f"Detector image, log scale{image_name}")
        self._plot_image.setLabel("bottom", "x (px)")
        self._plot_image.setLabel("left", "y (px)")
        self._plot_slice.setTitle(f"Thin time slice |Δt| < {self.slice_ns_var.get().strip()} ns{image_name}")
        self._plot_slice.setLabel("bottom", "x (px)")
        self._plot_slice.setLabel("left", "y (px)")
        ref = "p_y'" if circular else "p_x'"
        self._plot_check.setTitle(f"Validation: {ref} vs p_z")
        self._plot_check.setLabel("bottom", f"{ref} (a.u.)")
        self._plot_check.setLabel("left", "p_z (a.u.)")
        axis = "⊥ angle" if circular else "along angle"
        self._plot_rings.setTitle(f"Ring slice {axis}{image_name}")
        self._plot_rings.setLabel("bottom", "Position along axis (px)")
        self._plot_rings.setLabel("left", "Counts")
        self._plot_tx.setTitle(f"t vs {'y' if circular else 'x'}' (ring axis){time_name}")
        self._plot_tx.setLabel("bottom", "Position along ring axis (px)")
        self._plot_tx.setLabel("left", "t (ns)")
        self._plot_times.setTitle(f"On-axis arrival time{time_name}")
        self._plot_times.setLabel("bottom", "t (ns)")
        self._plot_times.setLabel("left", "Counts")
        self._plot_fit.setTitle("r² = P(Δt)")
        self._plot_fit.setLabel("bottom", "Δt (ns)")
        self._plot_fit.setLabel("left", "r² (px²)")
        self._plot_k.setTitle("Energy scale: r² vs ring index")
        self._plot_k.setLabel("bottom", "Ring index n")
        self._plot_k.setLabel("left", "r² (px²)")
        self._plot_energy.setTitle("Energy spectrum from |p|")
        self._plot_energy.setLabel("bottom", "Energy (eV)")
        self._plot_energy.setLabel("left", "Counts")

    def _draw_all(self, s):
        drawn = set()
        image_events = self._image_events()
        time_events = self._time_events()
        r, _, _ = ec.ring_radii(self._r_peaks)
        extent = (0.0, 256.0)
        ring_angle = self._ring_axis_angle(s)

        if image_events is not None:
            add_image(self._plot_image, image_events.x, image_events.y, extent, extent)
            styles = {"ring symmetry": ("+", RED), "center of mass": ("x", BLUE), "median": ("o", GREEN)}
            for name, (x, y) in self._center_estimates.items():
                symbol, color = styles[name]
                self._plot_image.addItem(pg.ScatterPlotItem(
                    [x], [y], symbol=symbol, size=12, pen=pg.mkPen(color, width=2), brush=None, name=name,
                ))
            self._draw_axes(self._plot_image, s, ring_angle, show_band=False)
            drawn.add(self._plot_image)

            mask = np.abs(image_events.t - s.tc) < s.slice_ns
            add_image(self._plot_slice, image_events.x[mask], image_events.y[mask], extent, extent)
            self._draw_axes(self._plot_slice, s, ring_angle, show_band=True)
            for radius in r:
                add_circle(self._plot_slice, s.xc, s.yc, radius, RED, QtCore.Qt.DotLine, width=2.5)
            drawn.add(self._plot_slice)

            self._r_spectrum = self._ring_spectrum(s)
            add_spectrum_curves(self._plot_rings, self._r_spectrum)
            pos = sorted(p for p, _ in self._r_peaks if p > 0)
            neg = sorted((p for p, _ in self._r_peaks if p < 0), reverse=True)
            for side in (pos, neg):
                for i, p in enumerate(side):
                    add_marker_line(self._plot_rings, p, str(i + 1), ORANGE)
            drawn.add(self._plot_rings)

        if time_events is not None:
            self._t_spectrum = self._time_spectrum(s)
            add_spectrum_curves(self._plot_times, self._t_spectrum)
            early, late = ec.split_time_peaks(self._t_peaks, s.tc)
            for prefix, side in (("E", early), ("L", late)):
                for i, (dt, _) in enumerate(side):
                    add_marker_line(self._plot_times, s.tc + dt, f"{prefix}{i + 1}", ORANGE)
            self._add_tc_line(self._plot_times, s.tc, angle=90)
            drawn.add(self._plot_times)

            # t against the rotated position along the ring axis, for hits
            # within the ring slice half-width of that axis.
            u, v = ec.axis_coordinates(time_events, s.xc, s.yc, s.theta, self._is_circular())
            mask = np.abs(v) < s.ring_width
            bins = (
                int(np.clip(2 * s.r_max / s.r_bin, 16, 512)),
                int(np.clip(2 * s.t_window / s.t_bin, 16, 512)),
            )
            add_image(
                self._plot_tx, u[mask], time_events.t[mask],
                (-s.r_max, s.r_max), (s.tc - s.t_window, s.tc + s.t_window), bins=bins,
            )
            self._plot_tx.addItem(pg.InfiniteLine(
                pos=0.0, angle=90, pen=pg.mkPen(ORANGE, width=1, style=QtCore.Qt.DashLine),
            ))
            self._add_tc_line(self._plot_tx, s.tc, angle=0)
            drawn.add(self._plot_tx)

        colors = {"Early": BLUE, "Late": ORANGE}
        for name, side in (("Early", self._early_fit), ("Late", self._late_fit)):
            if side is None:
                continue
            color = colors[name]
            if side.n_pairs:
                self._plot_fit.addItem(pg.ScatterPlotItem(
                    side.dt, side.r ** 2, symbol="o", size=8, pen=pg.mkPen(color), brush=pg.mkBrush(color), name=name,
                ))
                span = np.abs(side.dt).max() * 1.1
            else:
                span = s.t_window
            dt = np.linspace(-span, 0, 200) if name == "Early" else np.linspace(0, span, 200)
            self._plot_fit.addItem(pg.PlotDataItem(dt, ec.z2_polyval(side.coeffs, dt), pen=pg.mkPen(color, width=1.5)))
            drawn.add(self._plot_fit)

        if self._k_fit is not None:
            n = np.arange(self._k_fit.r.size)
            self._plot_k.addItem(pg.ScatterPlotItem(
                n, self._k_fit.r ** 2, symbol="o", size=8, pen=pg.mkPen(BLUE), brush=pg.mkBrush(BLUE),
            ))
            self._plot_k.addItem(pg.PlotDataItem(
                n, self._k_fit.slope * n + self._k_fit.intercept, pen=pg.mkPen(ORANGE, width=1.5),
            ))
            drawn.add(self._plot_k)

        calibration = self._current_calibration()
        self._publish(calibration)
        events = self._validation_events()
        if calibration is not None and events is not None:
            self._draw_validation(s, calibration, events, r)
            drawn.update((self._plot_check, self._plot_energy))
        return drawn

    def _add_tc_line(self, plot, tc, angle):
        """A draggable tc line: vertical on the arrival-time plot, horizontal on t vs x'."""
        line = pg.InfiniteLine(
            pos=tc, angle=angle, movable=True, pen=pg.mkPen(RED, width=2),
            label="tc", labelOpts={"position": 0.05, "color": RED},
        )
        line.sigPositionChangeFinished.connect(lambda item: self.tc_var.set(f"{item.value():.3f}"))
        plot.addItem(line)

    def _draw_axes(self, plot, s, ring_angle, show_band):
        plot.addItem(pg.InfiniteLine(pos=(s.xc, s.yc), angle=np.degrees(s.theta), pen=pg.mkPen(ORANGE, width=1.5)))
        plot.addItem(pg.InfiniteLine(
            pos=(s.xc, s.yc), angle=np.degrees(s.theta) + 90,
            pen=pg.mkPen(ORANGE, width=1, style=QtCore.Qt.DashLine),
        ))
        if show_band:
            normal = np.array([-np.sin(ring_angle), np.cos(ring_angle)]) * s.ring_width
            for sign in (1, -1):
                plot.addItem(pg.InfiniteLine(
                    pos=(s.xc + sign * normal[0], s.yc + sign * normal[1]), angle=np.degrees(ring_angle),
                    pen=pg.mkPen(GREEN, width=1),
                ))

    def _draw_validation(self, s, calibration, events, r):
        px, py, pz = calibration.calibrate(events.x, events.y, events.t, symmetrize=self.symmetrize_var.get())
        # Hits outside the calibration's valid dt range come back as NaN.
        valid = np.isfinite(pz)
        px, py, pz = px[valid], py[valid], pz[valid]
        scale = np.sqrt(2 * ec.EV_TO_HARTREE * calibration.k)
        ref, other = (py, px) if self._is_circular() else (px, py)
        mask = np.abs(other) < scale * s.ring_width
        p_max = scale * (r.max() * 1.25 if r.size else s.r_max)
        add_image(self._plot_check, ref[mask], pz[mask], (-p_max, p_max), (-p_max, p_max), bins=200)
        for radius in r:
            add_circle(self._plot_check, 0.0, 0.0, scale * radius, RED, QtCore.Qt.DotLine, width=2.5)

        energy = ec.energy_ev(px, py, pz)
        e_max = calibration.k * (r.max() ** 2 * 1.4 if r.size else s.r_max ** 2)
        counts, edges = np.histogram(energy, bins=400, range=(0.0, e_max))
        self._plot_energy.addItem(pg.PlotDataItem(0.5 * (edges[:-1] + edges[1:]), counts, pen=pg.mkPen(BLUE, width=1.5)))
        for i, radius in enumerate(r):
            add_marker_line(self._plot_energy, calibration.k * radius ** 2, str(i + 1), ORANGE)


# ---- ion time of flight -> m/q -----------------------------------------------

class MassCalibrationInterface(ttk.Frame):
    """i-ToF spectrum in, t = t0 + a*sqrt(m/q) out, fit to peaks the user
    picks and labels with a species or an m/q value."""

    def __init__(self, parent, hub=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._hub = hub
        self._calibration_path = None
        self._published = None

        def pvar(cls, key, default):
            return app_settings.persistent_var(self, cls, f"mq_calibration.{key}", default)

        self.path_var = pvar(tk.StringVar, "path", "")
        self.bin_var = pvar(tk.StringVar, "bin_ns", "5")
        self.t_min_var = pvar(tk.StringVar, "t_min_ns", "0")
        self.t_max_var = pvar(tk.StringVar, "t_max_ns", "")
        self.log_var = pvar(tk.BooleanVar, "log_scale", False)
        self.snap_var = pvar(tk.StringVar, "snap_ns", "50")
        self.fix_t0_var = pvar(tk.BooleanVar, "fix_t0", False)
        self.mq_bin_var = pvar(tk.StringVar, "mq_bin", "0.1")
        self.mq_max_var = pvar(tk.StringVar, "mq_max", "")

        self.status_var = tk.StringVar(self, value="Idle")
        self.fit_info_var = tk.StringVar(self, value="")
        self.export_info_var = tk.StringVar(self, value="")

        self._task = BackgroundTask(self)
        self._tof = None
        self._centers = None
        self._counts = None
        self._peaks = []
        self._calibration = None
        self._redraw_after = None

        self._build_ui()
        for var in (self.bin_var, self.t_min_var, self.t_max_var, self.log_var, self.mq_bin_var, self.mq_max_var):
            var.trace_add("write", self._schedule_redraw)
        self._redraw()

    def _build_ui(self):
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.rowconfigure(0, weight=1)
        sidebar.rowconfigure(6, weight=1)

        ui_style.build_button_bar(sidebar, 0, [("Load Data", self.load_data), ("Clear Peaks", self.clear_peaks)])
        ui_style.StatusBlock(sidebar, 1, self.status_var)

        data = ui_style.build_section(sidebar, 2, "Dataset")
        ui_style.add_form_row(data, 0, "cv4 file:", ui_style.build_path_field(data, self.path_var, self._browse))
        ui_style.add_form_row(data, 1, "Bin width (ns):", ttk.Entry(data, textvariable=self.bin_var, width=18))
        ui_style.add_form_row(data, 2, "t min (ns):", ttk.Entry(data, textvariable=self.t_min_var, width=18))
        ui_style.add_form_row(data, 3, "t max (ns):", ttk.Entry(data, textvariable=self.t_max_var, width=18))
        ttk.Checkbutton(data, text="Log scale", variable=self.log_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ui_style.add_note(data, 5, "Leave t max blank to fit the spectrum.", pady=(4, 4))

        peaks = ui_style.build_section(sidebar, 3, "Peaks")
        ui_style.add_note(
            peaks, 0,
            "Ctrl+click a peak in the ToF spectrum to label it, Shift+click to remove the nearest label. "
            "Labels are an m/q number or a formula with its charge: H2O+, N2+, Xe++ or Xe^2+. "
            "Formulas use the most abundant isotope.",
            pady=(4, 4),
        )
        ui_style.add_form_row(peaks, 1, "Snap window ± (ns):", ttk.Entry(peaks, textvariable=self.snap_var, width=18))
        self._tree = ttk.Treeview(peaks, columns=("label", "mq", "t", "residual"), show="headings", height=6)
        for col, text, width in (
            ("label", "Label", 80), ("mq", "m/q", 80), ("t", "t (ns)", 80), ("residual", "t − fit (ns)", 80),
        ):
            self._tree.heading(col, text=text)
            self._tree.column(col, width=width)
        self._tree.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
        ui_style.build_button_bar(peaks, 3, [("Remove Selected", self.remove_selected)], pady=(4, 6), columnspan=2)

        fit = ui_style.build_section(sidebar, 4, "Fit")
        ttk.Checkbutton(fit, text="Fix t0 at 0", variable=self.fix_t0_var, command=self._refit).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ui_style.add_form_row(fit, 1, "m/q bin:", ttk.Entry(fit, textvariable=self.mq_bin_var, width=18))
        ui_style.add_form_row(fit, 2, "m/q max:", ttk.Entry(fit, textvariable=self.mq_max_var, width=18))
        ui_style.add_note(fit, 3, textvariable=self.fit_info_var, pady=(4, 4))

        export = ui_style.build_section(sidebar, 5, "Export", pady=0)
        ui_style.build_button_bar(
            export, 0, [("Save As...", self.save_calibration), ("Load...", self.load_calibration)],
            pady=(4, 6), columnspan=2,
        )
        ui_style.add_note(export, 1, textvariable=self.export_info_var)

        self._grid = PlotGrid()
        self._glw = self._grid.widget
        self._plot_tof = self._grid.add(0, 0, colspan=2)
        self._plot_fit = self._grid.add(1, 0)
        self._plot_mq = self._grid.add(1, 1)
        relax_titles((self._plot_tof, self._plot_fit, self._plot_mq))
        grid_into(self._glw, main, row=0, column=0, sticky="nsew")
        connect_clicks(self, self._plot_tof, on_peak=self._on_tof_click)

    def _browse(self):
        browse_cv4(self, self.path_var, "Choose dataset")

    def _schedule_redraw(self, *_):
        if self._redraw_after is not None:
            self.after_cancel(self._redraw_after)
        self._redraw_after = self.after(300, self._redraw)

    # ---- loading --------------------------------------------------------------

    def load_data(self):
        if self._task.is_running():
            return
        path = self.path_var.get().strip()
        if not path:
            self.status_var.set("cv4 file is required.")
            return
        if not pathlib.Path(path).is_file():
            self.status_var.set(f"cv4 file not found: {path}")
            return
        self.status_var.set("Loading...")
        self._task.start(
            lambda: mc.load_ion_tof(path), self._on_loaded,
            lambda exc: self.status_var.set(f"Error: could not load data: {exc}"),
        )

    def _on_loaded(self, tof):
        self._tof = tof
        self.status_var.set(f"Loaded {tof.size} ion hits from {pathlib.Path(self.path_var.get()).name}.")
        self._redraw()

    # ---- peaks and fit ----------------------------------------------------------

    def _on_tof_click(self, x, remove):
        if remove:
            if self._peaks:
                self._peaks.pop(int(np.argmin([abs(p["t"] - x) for p in self._peaks])))
                self._peaks_changed()
            return
        if self._counts is None:
            return
        try:
            window = read_number(self.snap_var, "Snap window", positive=True)
        except ValueError as exc:
            self.status_var.set(f"Invalid setting: {exc}")
            return
        t = mc.snap_tof_peak(self._centers, self._counts, x, window)
        text = ""
        while True:
            text, ok = QtWidgets.QInputDialog.getText(
                self.window(), "Label peak",
                f"Peak at {t:.1f} ns.\nSpecies or m/q (e.g. H2O+, Xe++, Xe^2+, 18):",
                QtWidgets.QLineEdit.Normal, text,
            )
            if not ok:
                return
            try:
                mq = mc.parse_label(text)
                break
            except ValueError as exc:
                messagebox.showerror("Invalid label", str(exc), parent=self)
        self._peaks.append({"label": text.strip(), "mq": mq, "t": t})
        self._peaks_changed()

    def remove_selected(self):
        rows = sorted((self._tree.indexOfTopLevelItem(item) for item in self._tree.selectedItems()), reverse=True)
        for row in rows:
            if 0 <= row < len(self._peaks):
                self._peaks.pop(row)
        if rows:
            self._peaks_changed()

    def clear_peaks(self):
        self._peaks = []
        self._peaks_changed()

    def _peaks_changed(self):
        self._peaks.sort(key=lambda p: p["t"])
        self._refit()

    def _publish(self):
        """Hands a changed calibration to the hub (see CalibrationHub)."""
        if self._hub is None or self._calibration is None:
            return
        key = calibration_key(self._calibration, self._calibration_path)
        if key != self._published:
            self._published = key
            self._hub.publish(KIND_MQ, self._calibration, self._calibration_path)

    def _refit(self):
        self._calibration = None
        self._calibration_path = None
        if self._peaks:
            try:
                self._calibration = mc.fit_mass_calibration(self._peaks, fix_t0=self.fix_t0_var.get())
            except ValueError as exc:
                self.fit_info_var.set(str(exc))
        else:
            self.fit_info_var.set("")
        if self._calibration is not None:
            residuals = np.array(self._calibration.residuals())
            rms = float(np.sqrt(np.mean(residuals ** 2)))
            self.fit_info_var.set(
                f"t = {self._calibration.t0:.3f} ns + {self._calibration.a:.4f} ns · √(m/q) "
                f"from {len(self._peaks)} peak(s); rms residual {rms:.3g} ns."
            )
        self._refresh_table()
        self._redraw()
        self._publish()

    def _refresh_table(self):
        self._tree.delete(*self._tree.get_children())
        residuals = self._calibration.residuals() if self._calibration is not None else [None] * len(self._peaks)
        for peak, residual in zip(self._peaks, residuals):
            self._tree.insert("", "end", values=(
                peak["label"], f"{peak['mq']:.4f}", f"{peak['t']:.2f}",
                "--" if residual is None else f"{residual:+.2f}",
            ))

    # ---- export -----------------------------------------------------------------

    def save_calibration(self):
        if self._calibration is None:
            self.status_var.set("Label enough peaks to fit a calibration before saving.")
            return
        source = self.path_var.get().strip()
        self._calibration.meta = {"dataset": source}
        chosen = filedialog.asksaveasfilename(
            title="Save m/q calibration",
            defaultextension=".json",
            initialfile="mq_calibration.json",
            initialdir=str(pathlib.Path(source).parent) if source else None,
            filetypes=[("JSON", "*.json")],
            parent=self,
        )
        if not chosen:
            return
        try:
            self._calibration.save(chosen)
        except OSError as exc:
            self.status_var.set(f"Save failed: {exc}")
            return
        self._calibration_path = chosen
        self._publish()
        self.export_info_var.set(f"Saved {chosen}")
        self.status_var.set(f"Saved calibration to {chosen}.")

    def load_calibration(self):
        source = self.path_var.get().strip()
        chosen = filedialog.askopenfilename(
            title="Load m/q calibration",
            initialdir=str(pathlib.Path(source).parent) if source else None,
            filetypes=[("JSON", "*.json")],
            parent=self,
        )
        if not chosen:
            return
        try:
            calibration = mc.MassCalibration.load(chosen)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status_var.set(f"Load failed: {exc}")
            return
        self._peaks = [dict(p) for p in calibration.peaks]
        self.fix_t0_var.set(calibration.fix_t0)
        self._peaks_changed()
        if self._calibration is None:
            # A hand-written file without peaks: use its coefficients as-is.
            self._calibration = calibration
            self._refresh_table()
            self._redraw()
        self._calibration_path = chosen
        self._publish()
        self.export_info_var.set(f"Loaded {chosen}")
        self.status_var.set(f"Loaded calibration from {chosen}.")

    # ---- drawing ------------------------------------------------------------------

    def _redraw(self):
        self._redraw_after = None
        plots = (self._plot_tof, self._plot_fit, self._plot_mq)
        for plot in plots:
            plot.clear()
        self._centers = None
        self._counts = None
        log = self.log_var.get()
        self._plot_tof.setLogMode(False, log)
        self._plot_mq.setLogMode(False, log)

        drawn = set()
        try:
            drawn = self._draw_all(log)
        except ValueError as exc:
            self.status_var.set(f"Invalid setting: {exc}")

        self._plot_tof.setTitle("Ion time of flight")
        self._plot_tof.setLabel("bottom", "t (ns)")
        self._plot_tof.setLabel("left", "Counts")
        self._plot_fit.setTitle("t = t0 + a·√(m/q)")
        self._plot_fit.setLabel("bottom", "√(m/q) (√(u/e))")
        self._plot_fit.setLabel("left", "t (ns)")
        self._plot_mq.setTitle("Mass spectrum")
        self._plot_mq.setLabel("bottom", "m/q (u/e)")
        self._plot_mq.setLabel("left", "Counts")
        for plot in plots:
            plot.autoRange()
        show_placeholder(self._plot_tof, EMPTY_TEXT, self._plot_tof not in drawn)
        no_fit = "No calibration yet — Ctrl+click peaks to label them"
        show_placeholder(self._plot_fit, no_fit, self._plot_fit not in drawn)
        show_placeholder(self._plot_mq, EMPTY_TEXT if self._tof is None else no_fit, self._plot_mq not in drawn)

    def _draw_all(self, log):
        drawn = set()
        cal = self._calibration

        if self._tof is not None and self._tof.size:
            bin_ns = read_number(self.bin_var, "Bin width", positive=True)
            lo = read_number(self.t_min_var, "t min", allow_blank=True) or 0.0
            hi = read_number(self.t_max_var, "t max", allow_blank=True)
            if hi is None:
                hi = float(np.percentile(self._tof, 99.9)) * 1.05
            if hi <= lo:
                raise ValueError("t max must be greater than t min.")
            n_bins = max(8, int(np.ceil((hi - lo) / bin_ns)))
            counts, edges = np.histogram(self._tof, bins=n_bins, range=(lo, lo + n_bins * bin_ns))
            self._centers = 0.5 * (edges[:-1] + edges[1:])
            self._counts = counts
            self._plot_tof.addItem(pg.PlotDataItem(
                self._centers, self._plot_counts(counts, log), pen=pg.mkPen(BLUE, width=1), connect="finite",
            ))
            drawn.add(self._plot_tof)

        if self._centers is not None:
            for peak in self._peaks:
                add_marker_line(self._plot_tof, peak["t"], peak["label"], ORANGE)

        if cal is not None:
            s = np.sqrt([p["mq"] for p in self._peaks]) if self._peaks else np.array([])
            if s.size:
                t = np.array([p["t"] for p in self._peaks])
                self._plot_fit.addItem(pg.ScatterPlotItem(
                    s, t, symbol="o", size=8, pen=pg.mkPen(BLUE), brush=pg.mkBrush(BLUE),
                ))
                for si, ti, peak in zip(s, t, self._peaks):
                    label = pg.TextItem(peak["label"], anchor=(0, 1), color=BLUE)
                    label.setPos(si, ti)
                    self._plot_fit.addItem(label)
            s_max = (s.max() if s.size else 10.0) * 1.15
            grid = np.linspace(0, s_max, 100)
            self._plot_fit.addItem(pg.PlotDataItem(grid, cal.t0 + cal.a * grid, pen=pg.mkPen(ORANGE, width=1.5)))
            drawn.add(self._plot_fit)

            if self._tof is not None and self._tof.size:
                mq = cal.mass_to_charge(self._tof)
                mq = mq[np.isfinite(mq)]
                mq_bin = read_number(self.mq_bin_var, "m/q bin", positive=True)
                mq_max = read_number(self.mq_max_var, "m/q max", allow_blank=True)
                if mq_max is None:
                    top = max((p["mq"] for p in self._peaks), default=0.0)
                    mq_max = max(top * 1.3, float(np.percentile(mq, 99.5)) if mq.size else 1.0)
                if mq_max <= 0:
                    raise ValueError("m/q max must be positive.")
                n_bins = max(8, int(np.ceil(mq_max / mq_bin)))
                counts, edges = np.histogram(mq, bins=n_bins, range=(0.0, n_bins * mq_bin))
                self._plot_mq.addItem(pg.PlotDataItem(
                    0.5 * (edges[:-1] + edges[1:]), self._plot_counts(counts, log),
                    pen=pg.mkPen(BLUE, width=1), connect="finite",
                ))
                for peak in self._peaks:
                    add_marker_line(self._plot_mq, peak["mq"], peak["label"], ORANGE)
                drawn.add(self._plot_mq)
        return drawn

    @staticmethod
    def _plot_counts(counts, log):
        # Log mode can't show empty bins; leave gaps instead.
        return np.where(counts > 0, counts, np.nan) if log else counts
