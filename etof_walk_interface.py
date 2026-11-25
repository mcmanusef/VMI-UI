"""Analysis tab: fits the e-ToF time-walk (and optional position) correction
from single-electron shots of a cv4 file with cluster ToT, shows how well
it works on held-out shots, and saves it as JSON. See etof_walk.py for the
method. The correction goes to the other tabs through the CalibrationHub:
the momentum calibration, apply calibration and Coincidence tabs apply it
when asked, and Diagnostics' x / y vs e-ToF maps use the last saved one."""
import pathlib

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

import qtk as tk
from qtk import ttk, filedialog, grid_into

import app_settings
import data_range
import etof_walk as ew
import ui_style
from calibration_interface import (
    BLUE, GREY, KIND_ETOF_WALK, ORANGE, BackgroundTask, PlotGrid, browse_cv4, read_number, relax_titles,
    show_placeholder,
)
from qt_plots import hist_to_rgba

EMPTY_TEXT = "No data — press Load Data"
NO_FIT_TEXT = "No correction yet — press Fit"
IMAGE_BINS = 200


class EtofWalkCalibrationInterface(ttk.Frame):
    """cv4 file (with cluster ToT) in; an e-ToF walk correction out."""

    def __init__(self, parent, hub=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._hub = hub
        self._correction = None
        self._correction_path = None

        def pvar(cls, key, default):
            return app_settings.persistent_var(self, cls, f"etof_walk.{key}", default)

        self.path_var = pvar(tk.StringVar, "path", "")
        self.etof_min_var = pvar(tk.StringVar, "etof_min_ns", "")
        self.etof_max_var = pvar(tk.StringVar, "etof_max_ns", "")
        self.t_min_var = pvar(tk.StringVar, "t_min_ns", "")
        self.t_max_var = pvar(tk.StringVar, "t_max_ns", "")
        self.min_hits_var = pvar(tk.StringVar, "min_hits", str(ew.DEFAULT_MIN_HITS))
        self.n_bins_var = pvar(tk.StringVar, "n_bins", str(ew.DEFAULT_N_BINS))
        self.ref_lnq_var = pvar(tk.StringVar, "ref_lnq", f"{ew.DEFAULT_REF_LNQ:g}")
        self.position_var = pvar(tk.BooleanVar, "position_term", False)
        self.cell_px_var = pvar(tk.StringVar, "cell_px", str(ew.DEFAULT_CELL_PX))
        self.position_lnq_bins_var = pvar(tk.StringVar, "position_lnq_bins", str(ew.DEFAULT_POSITION_LNQ_BINS))
        self.seed_var = pvar(tk.StringVar, "seed", "0")
        self.log_var = pvar(tk.BooleanVar, "log_color", True)

        self.status_var = tk.StringVar(self, value="Idle")
        self.data_info_var = tk.StringVar(self, value="")
        self.result_var = tk.StringVar(self, value="")
        self.export_info_var = tk.StringVar(self, value="")

        self._task = BackgroundTask(self)
        self._shots = None          # etof_walk.Shots of the loaded file
        self._regions = None        # auto (lo, hi) of the e-ToF and cluster times
        self._report = None         # etof_walk.FitReport of the last fit

        self._build_ui()
        self.log_var.trace_add("write", lambda *_: self._redraw())
        self._redraw()

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.rowconfigure(0, weight=1)
        sidebar.rowconfigure(7, weight=1)

        ui_style.build_button_bar(sidebar, 0, [("Load Data", self.load_data), ("Fit", self.fit)])
        ui_style.StatusBlock(sidebar, 1, self.status_var)

        data = ui_style.build_section(sidebar, 2, "Dataset")
        ui_style.add_form_row(data, 0, "cv4 file:", ui_style.build_path_field(data, self.path_var, self._browse))
        ui_style.add_note(data, 1, textvariable=self.data_info_var, pady=(4, 4))
        ui_style.add_note(
            data, 2,
            "Needs a cv4 file with cluster ToT.",
            tip="The file must have been converted after cluster ToT was added. A grouped file uses "
                "every group.",
            pady=(0, 4),
        )

        sel = ui_style.build_section(sidebar, 3, "Shot selection")
        ui_style.add_form_row(sel, 0, "e-ToF min (ns):", ttk.Entry(sel, textvariable=self.etof_min_var, width=18))
        ui_style.add_form_row(sel, 1, "e-ToF max (ns):", ttk.Entry(sel, textvariable=self.etof_max_var, width=18))
        ui_style.add_form_row(sel, 2, "Cluster t min (ns):", ttk.Entry(sel, textvariable=self.t_min_var, width=18))
        ui_style.add_form_row(sel, 3, "Cluster t max (ns):", ttk.Entry(sel, textvariable=self.t_max_var, width=18))
        ui_style.add_form_row(sel, 4, "Min cluster hits:", ttk.Entry(sel, textvariable=self.min_hits_var, width=18))
        ui_style.add_note(
            sel, 5,
            "Single-electron shots only. Blank bound = the data's range.",
            tip="A shot is used when its pulse has exactly one e-ToF hit, inside the e-ToF window, "
                "and exactly one cluster, with at least the minimum hits and its time inside the "
                "cluster window.\n\n"
                "The e-ToF pulse-width cut is the e-ToF gate on Hardware > TDC mapping, applied when "
                "acquiring.",
            pady=(4, 4),
        )

        fit = ui_style.build_section(sidebar, 4, "Fit")
        ui_style.add_form_row(fit, 0, "ln Q bins:", ttk.Entry(fit, textvariable=self.n_bins_var, width=18))
        ui_style.add_form_row(fit, 1, "Reference ln Q:", ttk.Entry(fit, textvariable=self.ref_lnq_var, width=18))
        ttk.Checkbutton(fit, text="Include position term", variable=self.position_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ui_style.add_form_row(fit, 3, "Position cell (px):", ttk.Entry(fit, textvariable=self.cell_px_var, width=18))
        ui_style.add_form_row(
            fit, 4, "Position ln Q bins:", ttk.Entry(fit, textvariable=self.position_lnq_bins_var, width=18)
        )
        ui_style.add_form_row(fit, 5, "Random seed:", ttk.Entry(fit, textvariable=self.seed_var, width=18))
        ui_style.add_note(
            fit, 6,
            "The position term can remove physics.",
            tip="Q is the cluster's summed ToT (codes). The walk curve is the median e-ToF time in "
                "quantile bins of ln Q; corrected times are t − [f(ln Q) − f(reference)], so they "
                "stay on the scale of a reference-charge cluster and the curve transfers between "
                "runs.\n\n"
                "The position term is an x / y gradient, a per-cell map, and a per-cell map in each "
                "of the position ln Q bins (0 bins = no such stage), so the walk curve can vary "
                "across the detector. Use it only where position and flight time aren't linked by the "
                "momenta (high field).\n\n"
                "σ is checked on a random half of the shots held out of a separate fit; the saved "
                "correction uses all of them.",
            pady=(4, 4),
        )

        results = ui_style.build_section(sidebar, 5, "Results")
        ui_style.add_note(results, 0, textvariable=self.result_var, pady=(4, 4))
        ttk.Checkbutton(results, text="Log color scale", variable=self.log_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )

        export = ui_style.build_section(sidebar, 6, "Export", pady=0)
        ui_style.build_button_bar(
            export, 0, [("Save As...", self.save_correction), ("Load...", self.load_correction)],
            pady=(4, 6), columnspan=2,
        )
        ui_style.add_note(export, 1, textvariable=self.export_info_var)
        ui_style.add_note(
            export, 2,
            "Diagnostics uses the last saved or loaded file.",
            tip="Diagnostics' x / y vs e-ToF maps use the last saved or loaded correction file; the "
                "other tabs also follow an unsaved fit.",
        )

        self._grid = PlotGrid()
        self._glw = self._grid.widget
        self._plot_raw = self._grid.add(0, 0)
        self._plot_corr = self._grid.add(0, 1)
        self._plot_pos = self._grid.add(1, 0)
        self._plot_hist = self._grid.add(1, 1)
        relax_titles(self._plots())
        self._plot_pos.setAspectLocked(True)
        for i in range(2):
            self._glw.ci.layout.setColumnStretchFactor(i, 1)
            self._glw.ci.layout.setRowStretchFactor(i, 1)
        grid_into(self._glw, main, row=0, column=0, sticky="nsew")

    def _plots(self):
        return (self._plot_raw, self._plot_corr, self._plot_pos, self._plot_hist)

    def _browse(self):
        browse_cv4(self, self.path_var, "Choose dataset")

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
            lambda: self._load(path), self._on_loaded,
            lambda exc: self.status_var.set(f"Error: could not load data: {exc}"),
        )

    @staticmethod
    def _load(path):
        shots = ew.load_shots(path)

        def region(values):
            values = values[np.isfinite(values)]
            return data_range.dense_range(values) if values.size else (0.0, 1.0)

        return shots, {"etof": region(shots.t_etof), "t": region(shots.t)}

    def _on_loaded(self, result):
        self._shots, self._regions = result
        self._report = None
        name = pathlib.Path(self._shots.path).name
        lo_e, hi_e = self._regions["etof"]
        lo_t, hi_t = self._regions["t"]
        self.data_info_var.set(
            f"{self._shots.t.size} clusters and {self._shots.t_etof.size} e-ToF hits. e-ToF times mostly "
            f"{lo_e:.1f}–{hi_e:.1f} ns, cluster times {lo_t:.1f}–{hi_t:.1f} ns."
        )
        self.status_var.set(f"Loaded {name}. Set the windows and press Fit.")
        self._redraw()

    # ---- fitting ------------------------------------------------------------------

    def _selection(self):
        def bound(var, label, auto):
            value = read_number(var, label, allow_blank=True)
            return auto if value is None else value

        lo_e, hi_e = self._regions["etof"]
        lo_t, hi_t = self._regions["t"]
        selection = ew.Selection(
            etof_min=bound(self.etof_min_var, "e-ToF min", lo_e),
            etof_max=bound(self.etof_max_var, "e-ToF max", hi_e),
            t_min=bound(self.t_min_var, "Cluster t min", lo_t),
            t_max=bound(self.t_max_var, "Cluster t max", hi_t),
            min_hits=read_number(self.min_hits_var, "Min cluster hits", integer=True),
        )
        if selection.etof_max <= selection.etof_min or selection.t_max <= selection.t_min:
            raise ValueError("Each window's max must be greater than its min.")
        return selection

    def fit(self):
        if self._task.is_running():
            return
        if self._shots is None:
            self.status_var.set("Load a dataset first.")
            return
        try:
            selection = self._selection()
            n_bins = read_number(self.n_bins_var, "ln Q bins", integer=True, positive=True)
            ref_lnq = read_number(self.ref_lnq_var, "Reference ln Q")
            cell_px = read_number(self.cell_px_var, "Position cell", integer=True, positive=True)
            position_lnq_bins = read_number(self.position_lnq_bins_var, "Position ln Q bins", integer=True)
            if position_lnq_bins < 0:
                raise ValueError("Position ln Q bins must be 0 or more.")
            seed = read_number(self.seed_var, "Random seed", integer=True)
        except ValueError as exc:
            self.status_var.set(f"Invalid setting: {exc}")
            return
        shots = self._shots
        use_position = self.position_var.get()

        def work():
            selected = ew.select_shots(shots, selection)
            return ew.fit_and_validate(
                selected, n_bins=n_bins, ref_lnq=ref_lnq, use_position=use_position, cell_px=cell_px,
                seed=seed, selection=selection, source=shots.path, position_lnq_bins=position_lnq_bins,
            )

        self.status_var.set("Fitting...")
        self._task.start(work, self._on_fitted, lambda exc: self.status_var.set(f"Fit failed: {exc}"))

    def _on_fitted(self, result):
        self._correction, self._report = result
        self._correction_path = None
        self._show_result()
        self.export_info_var.set("Unsaved fit.")
        self.status_var.set(f"Fitted on {self._correction.n_shots} single-electron shots.")
        self._publish()
        self._redraw()

    def _show_result(self):
        corr = self._correction
        if corr is None:
            self.result_var.set("")
            return
        sigma = corr.held_out_sigma_ns or {}
        lines = [f"{corr.n_shots} single-electron shots. {corr.describe()}."]
        if sigma:
            parts = [f"raw {sigma['raw'] * 1000:.0f} ps", f"walk {sigma['walk'] * 1000:.0f} ps"]
            if "position" in sigma:
                parts.append(f"+ position {sigma['position'] * 1000:.0f} ps")
            lines.append("Held-out core σ: " + " → ".join(parts) + ".")
        lines.append(
            f"Walk over the fitted charges: {corr.t_nodes[0] - corr.walk(corr.ref_lnq):+.2f} ns at ln Q "
            f"{corr.lnq_nodes[0]:.2f}, {corr.t_nodes[-1] - corr.walk(corr.ref_lnq):+.2f} ns at ln Q "
            f"{corr.lnq_nodes[-1]:.2f} (relative to the reference)."
        )
        self.result_var.set(" ".join(lines))

    def _publish(self):
        if self._hub is not None and self._correction is not None:
            self._hub.publish(KIND_ETOF_WALK, self._correction, self._correction_path)

    # ---- export -----------------------------------------------------------------

    def save_correction(self):
        if self._correction is None:
            self.status_var.set("Fit a correction before saving.")
            return
        source = self.path_var.get().strip()
        chosen = filedialog.asksaveasfilename(
            title="Save e-ToF walk correction",
            defaultextension=".json",
            initialfile="etof_walk.json",
            initialdir=str(pathlib.Path(source).parent) if source else None,
            filetypes=[("JSON", "*.json")],
            parent=self,
        )
        if not chosen:
            return
        try:
            self._correction.save(chosen)
        except OSError as exc:
            self.status_var.set(f"Save failed: {exc}")
            return
        self._correction_path = chosen
        self._publish()
        self.export_info_var.set(f"Saved {chosen}")
        self.status_var.set(f"Saved the correction to {chosen}.")

    def load_correction(self):
        source = self.path_var.get().strip()
        chosen = filedialog.askopenfilename(
            title="Load e-ToF walk correction",
            initialdir=str(pathlib.Path(source).parent) if source else None,
            filetypes=[("JSON", "*.json")],
            parent=self,
        )
        if not chosen:
            return
        try:
            correction = ew.EtofWalkCorrection.load(chosen)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status_var.set(f"Load failed: {exc}")
            return
        self._correction = correction
        self._correction_path = chosen
        self._report = None
        self.ref_lnq_var.set(f"{correction.ref_lnq:g}")
        self._show_result()
        self._publish()
        self.export_info_var.set(f"Loaded {chosen}")
        self.status_var.set(f"Loaded the correction from {chosen}. Press Fit to refit on the loaded data.")
        self._redraw()

    # ---- drawing ------------------------------------------------------------------

    def _redraw(self):
        for plot in self._plots():
            plot.clear()
        drawn = set()
        try:
            drawn = self._draw_all()
        except ValueError as exc:
            self.status_var.set(f"Could not draw: {exc}")
        titles = (
            (self._plot_raw, "e-ToF vs cluster charge", "ln Q (Q = ToT sum, codes)", "e-ToF t (ns)"),
            (self._plot_corr, "Walk-corrected e-ToF vs charge", "ln Q", "corrected e-ToF t (ns)"),
            (self._plot_pos, "Median residual vs position (ns)", "cluster x (px)", "cluster y (px)"),
            (self._plot_hist, "Held-out shots: e-ToF − median", "t − median (ns)", "Counts"),
        )
        for plot, title, bottom, left in titles:
            plot.setTitle(title)
            plot.setLabel("bottom", bottom)
            plot.setLabel("left", left)
            plot.autoRange()
        for plot in self._plots():
            empty = EMPTY_TEXT if self._shots is None and self._correction is None else NO_FIT_TEXT
            show_placeholder(plot, empty, plot not in drawn)

    def _draw_all(self):
        drawn = set()
        log = self.log_var.get()
        report, corr = self._report, self._correction
        if report is not None and len(report.shots):
            s = report.shots
            q_range = (float(np.min(s.lnq)), float(np.max(s.lnq)) + 1e-9)
            for plot, t in ((self._plot_raw, s.t_etof), (self._plot_corr, report.corrected)):
                lo, hi = data_range.padded(*data_range.dense_range(t))
                hist, _, _ = np.histogram2d(s.lnq, t, bins=IMAGE_BINS, range=[q_range, (lo, hi)])
                img = pg.ImageItem(hist_to_rgba(hist, log=log))
                img.setRect(QtCore.QRectF(q_range[0], lo, q_range[1] - q_range[0], hi - lo))
                plot.addItem(img)
                drawn.add(plot)

            resid = report.corrected - np.median(report.corrected)
            if corr is not None and corr.position is None:
                resid = report.walk_only - np.median(report.walk_only)
            cell = corr.position["cell_px"] if corr is not None and corr.position else ew.DEFAULT_CELL_PX
            cell_map, cell_px, n_cells = ew.residual_map(s.x, s.y, resid, cell_px=cell)
            finite = cell_map[np.isfinite(cell_map)]
            if finite.size:
                level = max(float(np.percentile(np.abs(finite), 98)), 1e-3)
                img = pg.ImageItem(cell_map)
                img.setColorMap(pg.colormap.get("CET-D1"))
                img.setLevels((-level, level))
                img.setRect(QtCore.QRectF(0, 0, n_cells * cell_px, n_cells * cell_px))
                self._plot_pos.addItem(img)
                label = pg.TextItem(f"±{level * 1000:.0f} ps", anchor=(0, 0), color=GREY)
                label.setPos(0, n_cells * cell_px)
                self._plot_pos.addItem(label)
                drawn.add(self._plot_pos)

            held = report.held_out
            raw = s.t_etof[held] - np.median(s.t_etof[held])
            half = max(5.0 * ew.mad_sigma(raw), 0.5)
            bins = np.linspace(-half, half, 201)
            centers = 0.5 * (bins[:-1] + bins[1:])
            curves = [("raw", raw, GREY), ("walk", report.walk_only[held], BLUE)]
            if corr is not None and corr.position is not None:
                curves.append(("walk + position", report.corrected[held], ORANGE))
            self._plot_hist.addLegend(offset=(-10, 10))
            for name, values, color in curves:
                counts, _ = np.histogram(values - np.median(values), bins=bins)
                self._plot_hist.addItem(pg.PlotDataItem(
                    centers, counts, pen=pg.mkPen(color, width=1.5), name=name,
                ))
            drawn.add(self._plot_hist)

        if corr is not None:
            grid = np.linspace(corr.lnq_nodes[0] - 0.5, corr.lnq_nodes[-1] + 0.5, 200)
            ref_t = float(corr.walk(corr.ref_lnq))
            self._plot_raw.addItem(pg.PlotDataItem(grid, corr.walk(grid), pen=pg.mkPen(ORANGE, width=2)))
            self._plot_raw.addItem(pg.InfiniteLine(
                pos=corr.ref_lnq, angle=90, pen=pg.mkPen(ORANGE, width=1, style=QtCore.Qt.DashLine),
                label="reference", labelOpts={"position": 0.92, "color": ORANGE},
            ))
            self._plot_corr.addItem(pg.InfiniteLine(
                pos=ref_t, angle=0, pen=pg.mkPen(ORANGE, width=1, style=QtCore.Qt.DashLine),
            ))
            drawn.add(self._plot_raw)
        return drawn
