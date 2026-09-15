"""Analysis tab: applies the latest calibrations to a cv4 dataset and saves
the electrons (and ions) as pandas DataFrames in HDF5. The calibrations
follow the momentum and m/q calibration tabs through a CalibrationHub; the
tables and columns are described in apply_calibration.py."""
import pathlib
import threading

import numpy as np
import pyqtgraph as pg

import qtk as tk
from qtk import ttk, filedialog, messagebox, grid_into

import app_settings
import apply_calibration as ac
import electron_calibration as ec
import mass_calibration as mc
import ui_style
from calibration_interface import (
    BLUE, KIND_MOMENTUM, KIND_MQ, BackgroundTask, CalibrationHub, PlotGrid, add_image, browse_cv4,
    latest_path_key, relax_titles, show_placeholder,
)

EMPTY_TEXT = "No data — press Apply and Save"
LOADERS = {KIND_MOMENTUM: ec.MomentumCalibration.load, KIND_MQ: mc.MassCalibration.load}
TAB_NAMES = {KIND_MOMENTUM: "momentum calibration", KIND_MQ: "m/q calibration"}


class CalibrationApplyInterface(ttk.Frame):
    """cv4 dataset in; an HDF5 file of calibrated electrons, raw hits and
    (optionally) ions with m/q out, with previews of the result."""

    def __init__(self, parent, hub=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._hub = hub if hub is not None else CalibrationHub()

        def pvar(cls, key, default):
            return app_settings.persistent_var(self, cls, f"apply_calibration.{key}", default)

        self.path_var = pvar(tk.StringVar, "path", "")
        self.output_var = pvar(tk.StringVar, "output_path", "")
        self.electrons_only_var = pvar(tk.BooleanVar, "electrons_only", False)

        self.status_var = tk.StringVar(self, value="Idle")
        self.momentum_info_var = tk.StringVar(self, value="--")
        self.momentum_source_var = tk.StringVar(self, value="")
        self.mq_info_var = tk.StringVar(self, value="--")
        self.mq_source_var = tk.StringVar(self, value="")
        self.electrons_var = tk.StringVar(self, value="--")
        self.uncalibrated_var = tk.StringVar(self, value="--")
        self.ions_var = tk.StringVar(self, value="--")
        self.groups_var = tk.StringVar(self, value="--")
        self.output_info_var = tk.StringVar(self, value="")

        self._progress_var = tk.DoubleVar(self, value=0.0)
        self._stop_event = threading.Event()
        self._progress = None
        self._task = BackgroundTask(self)
        self._calibrations = {KIND_MOMENTUM: None, KIND_MQ: None}
        self._result = None

        self._build_ui()
        self._load_latest()
        self._hub.subscribe(self._on_published)
        self._redraw()

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.rowconfigure(0, weight=1)
        sidebar.rowconfigure(5, weight=1)

        ui_style.build_button_bar(sidebar, 0, [("Apply and Save", self.apply), ("Stop", self.stop)])
        ui_style.StatusBlock(sidebar, 1, self.status_var, progress_var=self._progress_var)

        cal = ui_style.build_section(sidebar, 2, "Calibrations")
        ttk.Label(cal, text="Momentum:").grid(row=0, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ui_style.add_note(cal, 1, textvariable=self.momentum_info_var, pady=0)
        ui_style.add_note(cal, 2, textvariable=self.momentum_source_var)
        ttk.Label(cal, text="m/q:").grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ui_style.add_note(cal, 4, textvariable=self.mq_info_var, pady=0)
        ui_style.add_note(cal, 5, textvariable=self.mq_source_var)
        ui_style.build_button_bar(cal, 6, [
            ("Load Momentum...", lambda: self._browse_calibration(KIND_MOMENTUM)),
            ("Load m/q...", lambda: self._browse_calibration(KIND_MQ)),
        ], pady=(4, 6), columnspan=2)
        ui_style.add_note(
            cal, 7,
            "These follow the calibration tabs: a fit, edit, save or load there replaces them here. "
            "At startup the last saved or loaded calibration files are used.",
            pady=(0, 4),
        )

        data = ui_style.build_section(sidebar, 3, "Dataset")
        ui_style.add_form_row(data, 0, "cv4 file:", ui_style.build_path_field(data, self.path_var, self._browse_cv4))
        ui_style.add_form_row(
            data, 1, "Output file:", ui_style.build_path_field(data, self.output_var, self._browse_output),
        )
        ui_style.add_note(
            data, 2, f"Leave the output file blank to save next to the cv4 file as <name>{ac.OUTPUT_SUFFIX}.",
        )
        ttk.Checkbutton(data, text="Only include electron data", variable=self.electrons_only_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ui_style.add_note(
            data, 4,
            "pz uses the e-ToF time, with the calibration's time dither added. Pairs outside the "
            "calibration's valid Δt range are kept with NaN momenta. A grouped file from the parameter "
            "grouping tab is processed group by group, with its parameter value in a parameter column.",
            pady=(4, 4),
        )

        output = ui_style.build_section(sidebar, 4, "Output", pady=0)
        ui_style.add_stat_rows(output, [
            ("Electron rows:", self.electrons_var),
            ("Outside valid Δt:", self.uncalibrated_var),
            ("Ion rows:", self.ions_var),
            ("Parameter groups:", self.groups_var),
        ])
        ui_style.add_note(output, 4, textvariable=self.output_info_var, pady=(4, 4))
        ui_style.add_note(
            output, 5,
            "Tables (pd.read_hdf(path, key)): electrons (raw x, y, t, t_etof, pulse, t_pulse and hit "
            "counts, with dt, px, py, pz, energy), raw/clusters, raw/etof, and ions (t_tof, pulse, "
            "t_pulse, n_tof, mq) unless only electron data is included. Grouped input adds a leading "
            "parameter column to each, and is written a group at a time; read one value with "
            "where=\"parameter == value\".",
        )

        self._grid = PlotGrid()
        self._glw = self._grid.widget
        self._plot_xy = self._grid.add(0, 0)
        self._plot_xz = self._grid.add(0, 1)
        self._plot_yz = self._grid.add(0, 2)
        self._plot_energy = self._grid.add(1, 0, colspan=2)
        self._plot_mq = self._grid.add(1, 2)
        relax_titles(self._plots())
        for plot in (self._plot_xy, self._plot_xz, self._plot_yz):
            plot.setAspectLocked(True)
        for col in range(3):
            self._glw.ci.layout.setColumnStretchFactor(col, 1)
        for row in range(2):
            self._glw.ci.layout.setRowStretchFactor(row, 1)
        grid_into(self._glw, main, row=0, column=0, sticky="nsew")

    def _plots(self):
        return (self._plot_xy, self._plot_xz, self._plot_yz, self._plot_energy, self._plot_mq)

    def _browse_cv4(self):
        browse_cv4(self, self.path_var, "Choose dataset")

    def _browse_output(self):
        initial = self.output_var.get().strip()
        source = self.path_var.get().strip()
        if not initial and source:
            initial = str(ac.default_output_path(source))
        chosen = filedialog.asksaveasfilename(
            title="Save calibrated data",
            defaultextension=".h5",
            initialfile=pathlib.Path(initial).name if initial else None,
            initialdir=str(pathlib.Path(initial).parent) if initial else None,
            filetypes=[("HDF5", "*.h5")],
            parent=self,
        )
        if chosen:
            self.output_var.set(chosen)

    # ---- calibrations -----------------------------------------------------------

    def _load_latest(self):
        """The calibrations the tabs already published, or else the last
        saved or loaded files."""
        for kind in (KIND_MOMENTUM, KIND_MQ):
            latest = self._hub.latest(kind)
            if latest is not None:
                self._on_published(kind, *latest)
                continue
            path = app_settings.get(latest_path_key(kind))
            if not path:
                self._show_calibration(kind)
                continue
            if not pathlib.Path(path).is_file():
                self._show_calibration(kind, note=f"Last calibration file not found: {path}")
                continue
            try:
                self._set_calibration(kind, LOADERS[kind](path), f"File: {path}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self._show_calibration(kind, note=f"Could not load {path}: {exc}")

    def _on_published(self, kind, calibration, path):
        source = f"File: {path}" if path else f"Unsaved, from the {TAB_NAMES[kind]} tab"
        self._set_calibration(kind, calibration, source)
        if not self._task.is_running():
            self.status_var.set(f"Using the latest {TAB_NAMES[kind]} ({source}).")

    def _browse_calibration(self, kind):
        chosen = filedialog.askopenfilename(
            title=f"Load {TAB_NAMES[kind]}", filetypes=[("JSON", "*.json")], parent=self,
        )
        if not chosen:
            return
        try:
            calibration = LOADERS[kind](chosen)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status_var.set(f"Load failed: {exc}")
            return
        self._set_calibration(kind, calibration, f"File: {chosen}")
        self.status_var.set(f"Loaded {TAB_NAMES[kind]} from {chosen}.")

    def _set_calibration(self, kind, calibration, source):
        # A copy, so later edits in the calibration tab don't change it.
        self._calibrations[kind] = type(calibration).from_dict(calibration.to_dict())
        self._show_calibration(kind, note=source)

    def _show_calibration(self, kind, note=""):
        calibration = self._calibrations[kind]
        if kind == KIND_MOMENTUM:
            if calibration is None:
                self.momentum_info_var.set("None. Fit, save or load one in the momentum calibration tab.")
            else:
                early, late = calibration.valid_dt_range()
                self.momentum_info_var.set(
                    f"{calibration.name}: k = {calibration.k:.4g} eV/px², valid for Δt from "
                    f"{early:.2f} to {late:+.2f} ns."
                )
                if calibration.time_source != ec.TIME_SOURCE_ETOF:
                    note += f" Derived from {calibration.time_source} times; applied to e-ToF times here."
            self.momentum_source_var.set(note.strip())
        else:
            if calibration is None:
                self.mq_info_var.set("None. Ions are saved without m/q.")
            else:
                self.mq_info_var.set(f"t = {calibration.t0:.3f} ns + {calibration.a:.4f} ns · √(m/q)")
            self.mq_source_var.set(note.strip())

    # ---- apply ----------------------------------------------------------------------

    def apply(self):
        if self._task.is_running():
            return
        momentum = self._calibrations[KIND_MOMENTUM]
        if momentum is None:
            self.status_var.set("A momentum calibration is required. Fit, save or load one first.")
            return
        path = self.path_var.get().strip()
        if not path:
            self.status_var.set("cv4 file is required.")
            return
        if not pathlib.Path(path).is_file():
            self.status_var.set(f"cv4 file not found: {path}")
            return
        output = pathlib.Path(self.output_var.get().strip() or ac.default_output_path(path))
        if output.resolve() == pathlib.Path(path).resolve():
            self.status_var.set("Output file must differ from the cv4 file.")
            return
        if output.exists() and not messagebox.askyesno(
            "File exists", f"{output} already exists.\n\nOverwrite it?", parent=self,
        ):
            self.status_var.set("Apply canceled.")
            return

        electrons_only = self.electrons_only_var.get()
        mq = None if electrons_only else self._calibrations[KIND_MQ]
        self._stop_event.clear()
        self._progress = None
        self._progress_var.set(0.0)
        self.status_var.set(f"Processing {pathlib.Path(path).name}...")
        self._task.start(
            lambda: ac.apply_to_file(
                path, output, momentum, mq, electrons_only, progress=self._on_progress, stop_event=self._stop_event,
            ),
            self._on_done,
            self._on_error,
        )
        self._poll_progress()

    def stop(self):
        """Stops after the part being processed; no output file is kept."""
        if not self._task.is_running():
            return
        self._stop_event.set()
        self.status_var.set("Stopping after the current part...")

    def _on_progress(self, done, total, name):
        # Worker thread: only store it; _poll_progress shows it.
        self._progress = (done, total, name)

    def _poll_progress(self):
        progress = self._progress
        if progress is not None and not self._stop_event.is_set():
            done, total, name = progress
            self._progress_var.set(100.0 * done / total if total else 100.0)
            if done < total:
                self.status_var.set(f"Processing {name} ({done + 1}/{total})...")
        if self._task.is_running():
            self.after(200, self._poll_progress)

    def _on_error(self, exc):
        self._progress_var.set(0.0)
        if isinstance(exc, ac.ApplyStopped):
            self.status_var.set("Stopped. No output file was written.")
        else:
            self.status_var.set(f"Error: could not apply the calibration: {exc}")

    def _on_done(self, result):
        self._progress_var.set(100.0)
        self._result = result
        self.electrons_var.set(str(result.n_electrons))
        self.uncalibrated_var.set(str(result.n_uncalibrated))
        self.ions_var.set("--" if result.electrons_only else str(result.n_ions))
        self.groups_var.set(f"{result.n_groups} ({result.parameter_label})" if result.n_groups else "--")
        self.output_info_var.set(f"Saved {result.output_path}")
        ions = "" if result.electrons_only else f" and {result.n_ions} ion rows"
        groups = f" from {result.n_groups} {result.parameter_label} groups" if result.n_groups else ""
        self.status_var.set(
            f"Finished. Saved {result.n_electrons} electron rows{ions}{groups} to {result.output_path}."
        )
        self._redraw()

    # ---- drawing ------------------------------------------------------------------

    def _redraw(self):
        plots = self._plots()
        for plot in plots:
            plot.clear()
        drawn = set()
        result = self._result

        if result is not None:
            valid = np.isfinite(result.pz)
            px, py, pz, energy = (a[valid] for a in (result.px, result.py, result.pz, result.energy))
            if pz.size:
                p_max = float(np.percentile(np.abs(np.concatenate([px, py, pz])), 99.5)) or 1.0
                extent = (-p_max, p_max)
                for plot, a, b in ((self._plot_xy, px, py), (self._plot_xz, px, pz), (self._plot_yz, py, pz)):
                    add_image(plot, a, b, extent, extent, bins=200)
                    drawn.add(plot)
                e_max = float(np.percentile(energy, 99.5)) or 1.0
                counts, edges = np.histogram(energy, bins=400, range=(0.0, e_max))
                self._plot_energy.addItem(pg.PlotDataItem(
                    0.5 * (edges[:-1] + edges[1:]), counts, pen=pg.mkPen(BLUE, width=1.5),
                ))
                drawn.add(self._plot_energy)
            if result.mq is not None:
                mq = result.mq[np.isfinite(result.mq)]
                if mq.size:
                    mq_max = float(np.percentile(mq, 99.9)) * 1.05 or 1.0
                    counts, edges = np.histogram(mq, bins=1000, range=(0.0, mq_max))
                    self._plot_mq.addItem(pg.PlotDataItem(
                        0.5 * (edges[:-1] + edges[1:]), counts, pen=pg.mkPen(BLUE, width=1),
                    ))
                    drawn.add(self._plot_mq)

        for plot, (a, b) in (
            (self._plot_xy, ("p_x", "p_y")), (self._plot_xz, ("p_x", "p_z")), (self._plot_yz, ("p_y", "p_z")),
        ):
            plot.setTitle(f"{a} vs {b}, log scale")
            plot.setLabel("bottom", f"{a} (a.u.)")
            plot.setLabel("left", f"{b} (a.u.)")
        self._plot_energy.setTitle("Electron energy")
        self._plot_energy.setLabel("bottom", "Energy (eV)")
        self._plot_energy.setLabel("left", "Counts")
        self._plot_mq.setTitle("Mass spectrum")
        self._plot_mq.setLabel("bottom", "m/q (u/e)")
        self._plot_mq.setLabel("left", "Counts")
        for plot in plots:
            plot.autoRange()

        mq_text = EMPTY_TEXT
        if result is not None and result.electrons_only:
            mq_text = "Ions not included — only electron data was saved"
        elif result is not None and result.mq is None:
            mq_text = "No m/q calibration — ions saved without m/q"
        for plot in plots:
            show_placeholder(plot, mq_text if plot is self._plot_mq else EMPTY_TEXT, plot not in drawn)
