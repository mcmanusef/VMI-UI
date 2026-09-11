"""Interactive momentum-calibration wizard.

Walks through, on a single shared plot:

  1. xy center + axis  -- drag the X / the small round handle along the
     line (or type numbers) to set the detector image's center of mass
     and its axis of greatest variation, then Confirm centers + rotates
     both loaded datasets onto that axis.
  2. e-ToF time zero    -- drag the X (position only) on an r-vs-e-ToF
     heatmap to set "time zero" (center_t), then Confirm offsets both
     datasets' e-ToF column by it.
  3. radial peaks       -- smoothed radial (r) intensity distribution,
     click peaks to select/deselect the ones that are real ATI orders.
  4. up/down peaks       -- smoothed e-ToF "up"/"down" branch distributions,
     same click-to-select.
  5. results             -- runs the r -> energy and up/down time -> energy
     fits, shows the calibrated px/pz histogram with ATI-order rings
     overlaid, and offers Save Fit / Save Calibrated Dataset.

If only one cv4 file is given, it's used for every step. If two are given,
the "xy calibration" file drives steps 1 and 3, the "e-ToF calibration"
file drives steps 2 and 4, and step 5 combines both. See
momentum_calibration.py's module docstring for the underlying physics.
"""
import numpy as np
import pandas as pd
import qtk as tk
from qtk import ttk, filedialog, messagebox, grid_into

import pyqtgraph as pg
from PyQt5 import QtCore

import app_settings
import momentum_calibration as mc
from qt_plots import hist_to_rgba, mpl_color, circle_curve, ZoomFocusViewBox
from scrollable_frame import ScrollableFrame

_SELECTED_COLOR = "#39d353"
_DESELECTED_COLOR = "#9aa0a6"
_HIT_PIXELS = 8.0
# Distance of the axis-rotation handle from the center, as a fraction of
# the plot's extent span -- chosen once per xy_center draw so the handle
# sits at a reasonable, fixed *data-space* radius from the center.
_AXIS_HANDLE_FRACTION = 0.4


class MomentumCalibrationInterface(ttk.Frame):
    def __init__(self, parent, coordinator=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._coordinator = coordinator
        if coordinator is not None and hasattr(coordinator, "register"):
            coordinator.register(self)

        self.xy_path_var = tk.StringVar(self, value=app_settings.get("momcal.xy_path", ""))
        self.etof_path_var = tk.StringVar(self, value=app_settings.get("momcal.etof_path", ""))
        self.wavelength_var = tk.StringVar(self, value=app_settings.get("momcal.wavelength_nm", "800"))
        # A coarse pre-filter on the raw cluster-time column `t`, applied to
        # both datasets at load time -- before any plot is built -- to drop
        # obvious background/noise clusters (e.g. reflections, other
        # pulses' leftovers) that would otherwise skew the xy heatmap, the
        # center-of-mass/axis fit, and everything downstream.
        self.t_gate_min_var = tk.StringVar(self, value=app_settings.get("momcal.t_gate_min", "0"))
        self.t_gate_max_var = tk.StringVar(self, value=app_settings.get("momcal.t_gate_max", "1000"))
        self.load_status_var = tk.StringVar(self, value="No data loaded.")
        self.stage_status_var = tk.StringVar(self, value="")

        # Pipeline state -----------------------------------------------
        self._xy_df = None
        self._etof_df = None
        self._stage = "init"

        self._cx = 0.0
        self._cy = 0.0
        self._angle = 0.0
        self._xy_extent = (0.0, 256.0)

        self._t_center_x = 0.0
        self._t_center_t = 0.0
        self._t_x_range = (-128.0, 128.0)
        self._etof_range = (0.0, 30000.0)
        self._t_x_bins = 150
        self._etof_bins = 150
        self._radial_r_max = 128.0
        self._radial_t_tol = 100.0
        self._updown_t_max = 1000.0
        self._updown_r_max = 20.0

        self._radial_fit = None
        self._up_fit = None
        self._down_fit = None
        self._calibration = None

        self._radial_centers = None
        self._radial_smoothed = None
        self._radial_peak_idx = np.array([], dtype=int)
        self._radial_selected = {}
        self._radial_scatter = None

        self._up_centers = None
        self._up_smoothed = None
        self._up_peak_idx = np.array([], dtype=int)
        self._up_selected = {}
        self._up_scatter = None

        self._down_smoothed = None
        self._down_peak_idx = np.array([], dtype=int)
        self._down_selected = {}
        self._down_scatter = None

        # xy_center / t_center native draggable overlay items (see
        # _draw_xy_center_plot / _draw_t_center_plot); rebuilt each time
        # the plot is cleared for a new stage.
        self._center_target = None
        self._axis_handle = None
        self._axis_handle_radius = 0.0
        self._axis_line = None
        self._t_target = None
        self._updating_fields = False

        self._build_ui()

    # ---- top-level layout -------------------------------------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.LabelFrame(self, text="Calibration datasets")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="XY calibration cv4:").grid(row=0, column=0, sticky="w", padx=(6, 4), pady=4)
        ttk.Entry(top, textvariable=self.xy_path_var).grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(top, text="Browse...", command=lambda: self._browse_cv4(self.xy_path_var)).grid(
            row=0, column=2, padx=(0, 6), pady=4
        )

        ttk.Label(top, text="e-ToF calibration cv4 (optional):").grid(row=1, column=0, sticky="w", padx=(6, 4), pady=4)
        ttk.Entry(top, textvariable=self.etof_path_var).grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(top, text="Browse...", command=lambda: self._browse_cv4(self.etof_path_var)).grid(
            row=1, column=2, padx=(0, 6), pady=4
        )
        ttk.Label(top, text="(leave blank to use the XY file for every step)", font=("Segoe UI", 8)).grid(
            row=2, column=1, sticky="w", padx=(0, 6)
        )

        ttk.Label(top, text="Photon wavelength (nm):").grid(row=3, column=0, sticky="w", padx=(6, 4), pady=4)
        ttk.Entry(top, textvariable=self.wavelength_var, width=12).grid(row=3, column=1, sticky="w", pady=4)

        gate_row = ttk.Frame(top)
        gate_row.grid(row=4, column=0, columnspan=2, sticky="w", padx=(6, 4), pady=4)
        ttk.Label(gate_row, text="Coarse t gate (ns):").grid(row=0, column=0, sticky="w")
        ttk.Entry(gate_row, textvariable=self.t_gate_min_var, width=10).grid(row=0, column=1, padx=(6, 2))
        ttk.Label(gate_row, text="to").grid(row=0, column=2, padx=2)
        ttk.Entry(gate_row, textvariable=self.t_gate_max_var, width=10).grid(row=0, column=3, padx=(2, 0))
        ttk.Label(gate_row, text="(applied to both datasets before anything is plotted)", font=("Segoe UI", 8)).grid(
            row=0, column=4, sticky="w", padx=(10, 0)
        )

        ttk.Button(top, text="Load / Restart", command=self.load_data).grid(row=3, column=2, rowspan=2, padx=(0, 6), pady=4)
        ttk.Label(top, textvariable=self.load_status_var, font=("Segoe UI", 8)).grid(
            row=5, column=0, columnspan=3, sticky="w", padx=(6, 6), pady=(0, 4)
        )

        plot_frame = ttk.Frame(self)
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 6))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self._plot_widget = pg.PlotWidget(viewBox=ZoomFocusViewBox())
        self._plot_item = self._plot_widget.getPlotItem()
        # Registered once; PlotItem.addItem() auto-adds any later item
        # passed a `name=` to this same legend (radial/up-down peak
        # stages), and clear() removes prior entries as their items go.
        self._plot_item.addLegend()
        grid_into(self._plot_widget, plot_frame, row=0, column=0, sticky="nsew")

        stage_container = ScrollableFrame(self, width=900)
        stage_container.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        self._stage_frame = stage_container.body
        self._stage_frame.columnconfigure(0, weight=1)

        self.stage_status_label = ttk.Label(self, textvariable=self.stage_status_var, wraplength=900, justify="left")
        self.stage_status_label.grid(row=3, column=0, sticky="w", padx=10, pady=(0, 10))

        self._plot_item.setTitle("Load a dataset to begin.")

    def default_fields(self):
        return {
            "momcal.xy_path": self.xy_path_var,
            "momcal.etof_path": self.etof_path_var,
            "momcal.wavelength_nm": self.wavelength_var,
            "momcal.t_gate_min": self.t_gate_min_var,
            "momcal.t_gate_max": self.t_gate_max_var,
        }

    def is_running(self):
        return False  # No background acquisition here; nothing for the coordinator to stop.

    def _browse_cv4(self, var):
        chosen = filedialog.askopenfilename(title="Select cv4 file", filetypes=[("cv4", "*.cv4")], parent=self)
        if chosen:
            var.set(chosen)

    # ---- data loading -------------------------------------------------

    def load_data(self):
        xy_path = self.xy_path_var.get().strip()
        etof_path = self.etof_path_var.get().strip()
        if not xy_path:
            self.load_status_var.set("Choose at least the XY calibration file.")
            return

        try:
            gate_min = float(self.t_gate_min_var.get())
            gate_max = float(self.t_gate_max_var.get())
            if gate_max <= gate_min:
                raise ValueError("gate max must be greater than gate min")
        except ValueError as exc:
            self.load_status_var.set(f"Invalid t gate: {exc}")
            return

        try:
            xy_df = mc.apply_t_gate(mc.load_cv4_events(xy_path), gate_min, gate_max)
        except Exception as exc:
            self.load_status_var.set(f"Could not load XY file: {exc}")
            return

        if etof_path:
            try:
                etof_df = mc.apply_t_gate(mc.load_cv4_events(etof_path), gate_min, gate_max)
            except Exception as exc:
                self.load_status_var.set(f"Could not load e-ToF file: {exc}")
                return
            note = (
                f"Loaded {len(xy_df)} xy-events and {len(etof_df)} e-ToF-events "
                f"(gated to t in {gate_min:g}-{gate_max:g} ns)."
            )
        else:
            etof_df = xy_df
            note = f"Loaded {len(xy_df)} events, gated to t in {gate_min:g}-{gate_max:g} ns (single dataset used for every step)."

        if len(xy_df) == 0 or len(etof_df) == 0:
            self.load_status_var.set(
                "No events left after the coarse t gate -- check the input file(s) or widen the gate."
            )
            return

        self._xy_df = xy_df
        self._etof_df = etof_df
        self._radial_fit = None
        self._up_fit = None
        self._down_fit = None
        self._calibration = None

        x = self._xy_df["x"].to_numpy()
        y = self._xy_df["y"].to_numpy()
        pad_x = 0.05 * max(1.0, x.max() - x.min())
        pad_y = 0.05 * max(1.0, y.max() - y.min())
        lo = min(x.min() - pad_x, y.min() - pad_y)
        hi = max(x.max() + pad_x, y.max() + pad_y)
        self._xy_extent = (float(lo), float(hi))
        self._cx, self._cy = mc.center_of_mass(x, y)
        self._angle = mc.principal_axis_angle(x, y)

        self.load_status_var.set(note)
        self._enter_stage("xy_center")

    def _unique_dfs(self):
        if self._xy_df is self._etof_df:
            return [self._xy_df]
        return [self._xy_df, self._etof_df]

    # ---- stage plumbing -------------------------------------------------

    def _enter_stage(self, stage):
        self._stage = stage
        self.stage_status_var.set("")
        for child in list(self._stage_frame.winfo_children()):
            child.destroy()
        getattr(self, f"_build_{stage}_controls")()
        getattr(self, f"_draw_{stage}_plot")()

    def _labeled_entry(self, parent, row, label, var, col=0, width=10):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(6, 4), pady=3)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=col + 1, sticky="w", pady=3)

    # ===================================================================
    # Stage 1: xy center + axis
    # ===================================================================

    def _build_xy_center_controls(self):
        frame = ttk.LabelFrame(self._stage_frame, text="Step 1: XY image center + axis")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.center_x_var = tk.StringVar(self, value=f"{self._cx:.3f}")
        self.center_y_var = tk.StringVar(self, value=f"{self._cy:.3f}")
        self.angle_deg_var = tk.StringVar(self, value=f"{np.degrees(self._angle):.3f}")

        self._labeled_entry(frame, 0, "Center x:", self.center_x_var)
        self._labeled_entry(frame, 0, "Center y:", self.center_y_var, col=2)
        self._labeled_entry(frame, 0, "Angle (deg):", self.angle_deg_var, col=4)

        for var in (self.center_x_var, self.center_y_var, self.angle_deg_var):
            var.trace_add("write", self._on_xy_fields_changed)

        ttk.Label(
            frame,
            text="Drag the red X to move the center, drag the white round handle to rotate the axis "
                 "-- or type numbers directly.",
            font=("Segoe UI", 8),
        ).grid(row=1, column=0, columnspan=6, sticky="w", padx=6, pady=(0, 4))

        ttk.Button(frame, text="Confirm center + axis", command=self._confirm_xy_center).grid(
            row=2, column=0, columnspan=2, padx=6, pady=(0, 6), sticky="w"
        )

    def _draw_xy_center_plot(self):
        self._plot_item.clear()
        hist, _, _ = mc.xy_heatmap(
            self._xy_df["x"].to_numpy(), self._xy_df["y"].to_numpy(), bins=200, extent=self._xy_extent
        )
        low, high = self._xy_extent
        img = pg.ImageItem(hist_to_rgba(hist, log=True))
        img.setRect(QtCore.QRectF(low, low, high - low, high - low))
        self._plot_item.addItem(img)

        self._axis_handle_radius = _AXIS_HANDLE_FRACTION * (high - low)

        self._axis_line = pg.InfiniteLine(pos=(self._cx, self._cy), angle=0, pen=pg.mkPen("w", width=1.5), movable=False)
        self._plot_item.addItem(self._axis_line)

        self._center_target = pg.TargetItem(
            pos=(self._cx, self._cy), size=16, symbol="x", pen=pg.mkPen("r", width=2.5), movable=True,
        )
        self._center_target.sigPositionChanged.connect(self._on_center_target_moved)
        self._plot_item.addItem(self._center_target)

        self._axis_handle = pg.TargetItem(
            pos=(0, 0), size=12, symbol="o", pen=pg.mkPen("w", width=2), movable=True,
        )
        self._axis_handle.sigPositionChanged.connect(self._on_axis_handle_moved)
        self._plot_item.addItem(self._axis_handle)

        self._plot_item.setTitle("XY heatmap")
        self._plot_item.setLabel("bottom", "x (px)")
        self._plot_item.setLabel("left", "y (px)")
        self._plot_item.getViewBox().setAspectLocked(True)
        self._plot_item.getViewBox().setRange(xRange=self._xy_extent, yRange=self._xy_extent, padding=0)
        self._update_xy_overlay()

    def _update_xy_overlay(self):
        handle_x = self._cx + self._axis_handle_radius * np.cos(self._angle)
        handle_y = self._cy + self._axis_handle_radius * np.sin(self._angle)

        self._center_target.blockSignals(True)
        self._center_target.setPos(self._cx, self._cy)
        self._center_target.blockSignals(False)

        self._axis_handle.blockSignals(True)
        self._axis_handle.setPos(handle_x, handle_y)
        self._axis_handle.blockSignals(False)

        self._axis_line.setAngle(np.degrees(self._angle))
        self._axis_line.setPos((self._cx, self._cy))

    def _on_center_target_moved(self, target):
        if self._stage != "xy_center":
            return
        pos = target.pos()
        self._cx, self._cy = float(pos.x()), float(pos.y())
        self._sync_xy_fields()
        self._update_xy_overlay()

    def _on_axis_handle_moved(self, target):
        if self._stage != "xy_center":
            return
        pos = target.pos()
        angle = np.arctan2(pos.y() - self._cy, pos.x() - self._cx)
        if angle < -np.pi / 2:
            angle += np.pi
        elif angle >= np.pi / 2:
            angle -= np.pi
        self._angle = float(angle)
        self._sync_xy_fields()
        self._update_xy_overlay()

    def _on_xy_fields_changed(self, *_):
        if self._updating_fields or self._stage != "xy_center":
            return
        try:
            self._cx = float(self.center_x_var.get())
            self._cy = float(self.center_y_var.get())
            self._angle = np.radians(float(self.angle_deg_var.get()))
        except ValueError:
            return
        self._update_xy_overlay()

    def _sync_xy_fields(self):
        self._updating_fields = True
        try:
            self.center_x_var.set(f"{self._cx:.3f}")
            self.center_y_var.set(f"{self._cy:.3f}")
            self.angle_deg_var.set(f"{np.degrees(self._angle):.3f}")
        finally:
            self._updating_fields = False

    def _confirm_xy_center(self):
        for df in self._unique_dfs():
            x_new, y_new = mc.center_and_rotate(df["x"], df["y"], self._cx, self._cy, self._angle)
            df["x"] = x_new
            df["y"] = y_new

        x_all = self._etof_df["x"].to_numpy()
        pad = 0.05 * max(1.0, x_all.max() - x_all.min())
        self._t_x_range = (float(x_all.min() - pad), float(x_all.max() + pad))

        etof_all = self._etof_df["etof"].to_numpy()
        # Default the etof-axis bounds to a 50 ns window centered on the
        # peak (mode) of the etof distribution, rather than its full span --
        # a much better starting guess for where t0 actually is, since the
        # bulk of the data (the prompt feature) sits near the peak. The
        # initial marker position follows the same peak, so it starts
        # visible inside that window.
        counts, edges = np.histogram(etof_all, bins=200)
        peak_idx = int(np.argmax(counts))
        peak = float(0.5 * (edges[peak_idx] + edges[peak_idx + 1]))
        self._etof_range = (peak - 25.0, peak + 25.0)
        self._t_center_x, _ = mc.center_of_mass(x_all, etof_all)
        self._t_center_t = peak

        self.stage_status_var.set(
            f"XY centered on ({self._cx:.2f}, {self._cy:.2f}), rotated {np.degrees(self._angle):.2f} deg."
        )
        self._enter_stage("t_center")

    # ===================================================================
    # Stage 2: e-ToF time zero
    # ===================================================================

    def _build_t_center_controls(self):
        frame = ttk.LabelFrame(self._stage_frame, text="Step 2: e-ToF time zero")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.t_center_x_var = tk.StringVar(self, value=f"{self._t_center_x:.3f}")
        self.t_center_t_var = tk.StringVar(self, value=f"{self._t_center_t:.3f}")
        self._labeled_entry(frame, 0, "Center x:", self.t_center_x_var)
        self._labeled_entry(frame, 0, "Center t:", self.t_center_t_var, col=2)
        for var in (self.t_center_x_var, self.t_center_t_var):
            var.trace_add("write", self._on_t_fields_changed)

        ttk.Label(
            frame,
            text="Drag the red X or type numbers -- this sets time zero (pz = 0).",
            font=("Segoe UI", 8),
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 4))

        self.t_xmin_var = tk.StringVar(self, value=f"{self._t_x_range[0]:.3f}")
        self.t_xmax_var = tk.StringVar(self, value=f"{self._t_x_range[1]:.3f}")
        self.etof_min_var = tk.StringVar(self, value=f"{self._etof_range[0]:.3f}")
        self.etof_max_var = tk.StringVar(self, value=f"{self._etof_range[1]:.3f}")
        self.t_xbins_var = tk.StringVar(self, value=str(self._t_x_bins))
        self.etof_bins_var = tk.StringVar(self, value=str(self._etof_bins))

        self._labeled_entry(frame, 2, "x min:", self.t_xmin_var)
        self._labeled_entry(frame, 2, "x max:", self.t_xmax_var, col=2)
        self._labeled_entry(frame, 2, "x bins:", self.t_xbins_var, col=4)
        self._labeled_entry(frame, 3, "etof min:", self.etof_min_var)
        self._labeled_entry(frame, 3, "etof max:", self.etof_max_var, col=2)
        self._labeled_entry(frame, 3, "etof bins:", self.etof_bins_var, col=4)
        ttk.Label(
            frame,
            text="(bounds for the plot used to find t0 -- not a data filter; etof bounds default to a "
                 "50 ns window around the peak of the data)",
            font=("Segoe UI", 8),
        ).grid(row=4, column=0, columnspan=6, sticky="w", padx=6, pady=(0, 4))
        ttk.Button(frame, text="Recompute", command=self._recompute_t_center).grid(
            row=3, column=6, padx=(12, 6), pady=3, sticky="w"
        )

        ttk.Button(frame, text="Confirm time zero", command=self._confirm_t_center).grid(
            row=5, column=0, columnspan=2, padx=6, pady=(0, 6), sticky="w"
        )

    def _read_t_center_settings(self):
        x_min = float(self.t_xmin_var.get())
        x_max = float(self.t_xmax_var.get())
        etof_min = float(self.etof_min_var.get())
        etof_max = float(self.etof_max_var.get())
        if x_max <= x_min or etof_max <= etof_min:
            raise ValueError("max must be greater than min for both x and etof.")
        return {
            "x_range": (x_min, x_max),
            "etof_range": (etof_min, etof_max),
            "x_bins": max(2, int(float(self.t_xbins_var.get()))),
            "etof_bins": max(2, int(float(self.etof_bins_var.get()))),
        }

    def _recompute_t_center(self):
        try:
            settings = self._read_t_center_settings()
        except ValueError as exc:
            self.stage_status_var.set(f"Invalid t0-plot bounds: {exc}")
            return
        self._t_x_range = settings["x_range"]
        self._etof_range = settings["etof_range"]
        self._t_x_bins = settings["x_bins"]
        self._etof_bins = settings["etof_bins"]
        self._draw_t_center_plot()

    def _draw_t_center_plot(self):
        self._plot_item.clear()
        hist, _, _ = mc.x_etof_heatmap(
            self._etof_df["x"].to_numpy(),
            self._etof_df["etof"].to_numpy(),
            x_bins=self._t_x_bins,
            x_range=self._t_x_range,
            t_bins=self._etof_bins,
            t_range=self._etof_range,
        )
        x0, x1 = self._t_x_range
        y0, y1 = self._etof_range
        img = pg.ImageItem(hist_to_rgba(hist, log=True))
        img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        self._plot_item.addItem(img)

        self._t_target = pg.TargetItem(
            pos=(self._t_center_x, self._t_center_t), size=16, symbol="x", pen=pg.mkPen("r", width=2.5), movable=True,
        )
        self._t_target.sigPositionChanged.connect(self._on_t_target_moved)
        self._plot_item.addItem(self._t_target)

        self._plot_item.setTitle("x vs e-ToF -- find time zero")
        self._plot_item.setLabel("bottom", "x (px, centered/rotated)")
        self._plot_item.setLabel("left", "e-ToF (ns)")
        self._plot_item.getViewBox().setAspectLocked(False)
        self._plot_item.getViewBox().setRange(xRange=self._t_x_range, yRange=self._etof_range, padding=0)

    def _on_t_target_moved(self, target):
        if self._stage != "t_center":
            return
        pos = target.pos()
        self._t_center_x, self._t_center_t = float(pos.x()), float(pos.y())
        self._sync_t_fields()

    def _on_t_fields_changed(self, *_):
        if self._updating_fields or self._stage != "t_center":
            return
        try:
            self._t_center_x = float(self.t_center_x_var.get())
            self._t_center_t = float(self.t_center_t_var.get())
        except ValueError:
            return
        if self._t_target is not None:
            self._t_target.blockSignals(True)
            self._t_target.setPos(self._t_center_x, self._t_center_t)
            self._t_target.blockSignals(False)

    def _sync_t_fields(self):
        self._updating_fields = True
        try:
            self.t_center_x_var.set(f"{self._t_center_x:.3f}")
            self.t_center_t_var.set(f"{self._t_center_t:.3f}")
        finally:
            self._updating_fields = False

    def _confirm_t_center(self):
        for df in self._unique_dfs():
            df["etof"] = df["etof"] - self._t_center_t

        r_xy = np.hypot(self._xy_df["x"].to_numpy(), self._xy_df["y"].to_numpy())
        self._radial_r_max = float(max(1.0, np.percentile(r_xy, 99)))
        etof_xy = self._xy_df["etof"].to_numpy()
        self._radial_t_tol = float(max(1.0, 0.1 * np.percentile(np.abs(etof_xy), 90)))

        etof_e = self._etof_df["etof"].to_numpy()
        self._updown_t_max = float(max(1.0, np.percentile(np.abs(etof_e), 99)))
        self._updown_r_max = float(max(1.0, 0.2 * self._radial_r_max))

        self.stage_status_var.set(f"e-ToF time zero set to {self._t_center_t:.3f} ns.")
        self._enter_stage("radial_peaks")

    # ===================================================================
    # Stage 3: radial peaks
    # ===================================================================

    def _build_radial_peaks_controls(self):
        frame = ttk.LabelFrame(self._stage_frame, text="Step 3: radial peaks (xy dataset)")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.r_bins_var = tk.StringVar(self, value="200")
        self.r_max_var = tk.StringVar(self, value=f"{self._radial_r_max:.3f}")
        self.r_t_tol_var = tk.StringVar(self, value=f"{self._radial_t_tol:.3f}")
        self.r_sigma_var = tk.StringVar(self, value="2.0")
        self.r_prom_var = tk.StringVar(self, value="0.05")
        self.r_distance_var = tk.StringVar(self, value="")

        self._labeled_entry(frame, 0, "r bins:", self.r_bins_var)
        self._labeled_entry(frame, 0, "r max:", self.r_max_var, col=2)
        self._labeled_entry(frame, 0, "|e-ToF| tolerance (ns):", self.r_t_tol_var, col=4, width=10)
        self._labeled_entry(frame, 1, "Smoothing sigma (bins):", self.r_sigma_var)
        self._labeled_entry(frame, 1, "Min prominence (frac):", self.r_prom_var, col=2)
        self._labeled_entry(frame, 1, "Min peak spacing (bins, opt.):", self.r_distance_var, col=4)

        ttk.Button(frame, text="Recompute", command=self._recompute_radial).grid(
            row=2, column=0, padx=6, pady=6, sticky="w"
        )
        ttk.Label(
            frame, text="Click a peak marker to select/deselect it.", font=("Segoe UI", 8)
        ).grid(row=2, column=1, columnspan=3, sticky="w")
        ttk.Button(frame, text="Confirm peak selection", command=self._confirm_radial_peaks).grid(
            row=2, column=4, padx=6, pady=6, sticky="w"
        )

    def _read_radial_settings(self):
        return {
            "r_bins": max(2, int(float(self.r_bins_var.get()))),
            "r_max": max(1e-6, float(self.r_max_var.get())),
            "t_tol": max(0.0, float(self.r_t_tol_var.get())),
            "sigma": max(1e-6, float(self.r_sigma_var.get())),
            "prominence_frac": max(1e-6, float(self.r_prom_var.get())),
            "distance": (int(float(self.r_distance_var.get())) if self.r_distance_var.get().strip() else None),
        }

    def _recompute_radial(self):
        try:
            settings = self._read_radial_settings()
        except ValueError:
            self.stage_status_var.set("Invalid radial-histogram setting.")
            return
        self._draw_radial_peaks_plot(settings)

    def _draw_radial_peaks_plot(self, settings=None):
        if settings is None:
            settings = self._read_radial_settings()
        centers, counts = mc.radial_distribution(
            self._xy_df["x"].to_numpy(),
            self._xy_df["y"].to_numpy(),
            self._xy_df["etof"].to_numpy(),
            r_bins=settings["r_bins"],
            r_max=settings["r_max"],
            t_tol=settings["t_tol"],
        )
        smoothed, peak_idx = mc.smooth_and_find_peaks(
            counts, sigma=settings["sigma"], prominence_frac=settings["prominence_frac"], distance=settings["distance"]
        )
        self._radial_centers = centers
        self._radial_smoothed = smoothed
        self._radial_peak_idx = peak_idx
        self._radial_selected = {i: True for i in range(len(peak_idx))}

        self._plot_item.clear()
        self._plot_item.addItem(pg.PlotDataItem(centers, counts, pen=pg.mkPen("lightgray", width=1.0), name="raw"))
        self._plot_item.addItem(
            pg.PlotDataItem(centers, smoothed, pen=pg.mkPen(mpl_color("tab:blue"), width=1.5), name="smoothed")
        )

        self._radial_scatter = pg.ScatterPlotItem(
            x=centers[peak_idx], y=smoothed[peak_idx], size=12,
            pen=pg.mkPen("k"), brush=pg.mkBrush(_SELECTED_COLOR),
        )
        self._radial_scatter.sigClicked.connect(self._on_radial_scatter_clicked)
        self._plot_item.addItem(self._radial_scatter)

        self._plot_item.setTitle("Radial intensity distribution")
        self._plot_item.setLabel("bottom", "r (px)")
        self._plot_item.setLabel("left", "counts")
        self._plot_item.getViewBox().setAspectLocked(False)
        self._plot_item.autoRange()

    def _update_radial_scatter_colors(self):
        colors = [
            pg.mkBrush(_SELECTED_COLOR if self._radial_selected.get(i, False) else _DESELECTED_COLOR)
            for i in range(len(self._radial_peak_idx))
        ]
        self._radial_scatter.setBrush(colors)

    def _on_radial_scatter_clicked(self, _scatter, points, _ev):
        if self._stage != "radial_peaks" or not points:
            return
        idx = points[0].index()
        self._radial_selected[idx] = not self._radial_selected.get(idx, False)
        self._update_radial_scatter_colors()

    def _confirm_radial_peaks(self):
        selected = [i for i in range(len(self._radial_peak_idx)) if self._radial_selected.get(i, False)]
        if len(selected) < 2:
            self.stage_status_var.set("Select at least 2 radial peaks (they must be spaced by one photon energy).")
            return
        r_peaks = self._radial_centers[self._radial_peak_idx[selected]]
        try:
            wavelength_nm = float(self.wavelength_var.get())
            self._radial_fit = mc.fit_radial_energy(r_peaks, wavelength_nm)
        except Exception as exc:
            self.stage_status_var.set(f"Radial fit failed: {exc}")
            return

        self.stage_status_var.set(
            f"Radial fit: a={self._radial_fit['a']:.6g} Ha/px^2, e0={self._radial_fit['e0']:.6g} Ha, "
            f"hv={self._radial_fit['hv']:.6g} Ha ({len(selected)} peaks)."
        )
        self._enter_stage("updown_peaks")

    # ===================================================================
    # Stage 4: up/down peaks
    # ===================================================================

    def _build_updown_peaks_controls(self):
        frame = ttk.LabelFrame(self._stage_frame, text="Step 4: up/down e-ToF peaks (e-ToF dataset)")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.ud_bins_var = tk.StringVar(self, value="200")
        self.ud_t_max_var = tk.StringVar(self, value=f"{self._updown_t_max:.3f}")
        self.ud_r_max_var = tk.StringVar(self, value=f"{self._updown_r_max:.3f}")
        self.ud_sigma_var = tk.StringVar(self, value="2.0")
        self.ud_prom_var = tk.StringVar(self, value="0.05")
        self.ud_distance_var = tk.StringVar(self, value="")

        self._labeled_entry(frame, 0, "t bins:", self.ud_bins_var)
        self._labeled_entry(frame, 0, "t max (ns):", self.ud_t_max_var, col=2)
        self._labeled_entry(frame, 0, "r tolerance (px):", self.ud_r_max_var, col=4)
        self._labeled_entry(frame, 1, "Smoothing sigma (bins):", self.ud_sigma_var)
        self._labeled_entry(frame, 1, "Min prominence (frac):", self.ud_prom_var, col=2)
        self._labeled_entry(frame, 1, "Min peak spacing (bins, opt.):", self.ud_distance_var, col=4)

        ttk.Button(frame, text="Recompute", command=self._recompute_updown).grid(
            row=2, column=0, padx=6, pady=6, sticky="w"
        )
        ttk.Label(
            frame,
            text=f"Up to {len(self._radial_fit['energies'])} peaks available from the radial fit, "
                 "per branch. Click a marker to select/deselect it.",
            font=("Segoe UI", 8),
        ).grid(row=2, column=1, columnspan=3, sticky="w")
        ttk.Button(frame, text="Confirm peak selection + run fits", command=self._confirm_updown_peaks).grid(
            row=2, column=4, padx=6, pady=6, sticky="w"
        )

    def _read_updown_settings(self):
        return {
            "t_bins": max(2, int(float(self.ud_bins_var.get()))),
            "t_max": max(1e-6, float(self.ud_t_max_var.get())),
            "r_max": max(1e-6, float(self.ud_r_max_var.get())),
            "sigma": max(1e-6, float(self.ud_sigma_var.get())),
            "prominence_frac": max(1e-6, float(self.ud_prom_var.get())),
            "distance": (int(float(self.ud_distance_var.get())) if self.ud_distance_var.get().strip() else None),
        }

    def _recompute_updown(self):
        try:
            settings = self._read_updown_settings()
        except ValueError:
            self.stage_status_var.set("Invalid up/down-histogram setting.")
            return
        self._draw_updown_peaks_plot(settings)

    def _draw_updown_peaks_plot(self, settings=None):
        if settings is None:
            settings = self._read_updown_settings()
        centers, up_counts, down_counts = mc.updown_distributions(
            self._etof_df["x"].to_numpy(),
            self._etof_df["y"].to_numpy(),
            self._etof_df["etof"].to_numpy(),
            t_bins=settings["t_bins"],
            t_max=settings["t_max"],
            r_max=settings["r_max"],
        )
        up_smoothed, up_peak_idx = mc.smooth_and_find_peaks(
            up_counts, sigma=settings["sigma"], prominence_frac=settings["prominence_frac"], distance=settings["distance"]
        )
        down_smoothed, down_peak_idx = mc.smooth_and_find_peaks(
            down_counts, sigma=settings["sigma"], prominence_frac=settings["prominence_frac"], distance=settings["distance"]
        )
        self._up_centers = centers
        self._up_smoothed = up_smoothed
        self._up_peak_idx = up_peak_idx
        self._up_selected = {i: True for i in range(len(up_peak_idx))}
        self._down_smoothed = down_smoothed
        self._down_peak_idx = down_peak_idx
        self._down_selected = {i: True for i in range(len(down_peak_idx))}

        self._plot_item.clear()
        self._plot_item.addItem(
            pg.PlotDataItem(centers, up_smoothed, pen=pg.mkPen(mpl_color("tab:blue"), width=1.5), name="up (t < t0)")
        )
        self._plot_item.addItem(
            pg.PlotDataItem(centers, down_smoothed, pen=pg.mkPen(mpl_color("tab:orange"), width=1.5), name="down (t > t0)")
        )

        self._up_scatter = pg.ScatterPlotItem(
            x=centers[up_peak_idx], y=up_smoothed[up_peak_idx], size=12, symbol="o",
            pen=pg.mkPen("k"), brush=pg.mkBrush(_SELECTED_COLOR),
        )
        self._up_scatter.sigClicked.connect(self._on_up_scatter_clicked)
        self._plot_item.addItem(self._up_scatter)

        self._down_scatter = pg.ScatterPlotItem(
            x=centers[down_peak_idx], y=down_smoothed[down_peak_idx], size=12, symbol="s",
            pen=pg.mkPen("k"), brush=pg.mkBrush(_SELECTED_COLOR),
        )
        self._down_scatter.sigClicked.connect(self._on_down_scatter_clicked)
        self._plot_item.addItem(self._down_scatter)

        self._plot_item.setTitle("Up / down e-ToF distributions (|t - t0|)")
        self._plot_item.setLabel("bottom", "|e-ToF - t0| (ns)")
        self._plot_item.setLabel("left", "counts")
        self._plot_item.getViewBox().setAspectLocked(False)
        self._plot_item.autoRange()

    def _update_updown_scatter_colors(self):
        up_colors = [
            pg.mkBrush(_SELECTED_COLOR if self._up_selected.get(i, False) else _DESELECTED_COLOR)
            for i in range(len(self._up_peak_idx))
        ]
        down_colors = [
            pg.mkBrush(_SELECTED_COLOR if self._down_selected.get(i, False) else _DESELECTED_COLOR)
            for i in range(len(self._down_peak_idx))
        ]
        self._up_scatter.setBrush(up_colors)
        self._down_scatter.setBrush(down_colors)

    def _on_up_scatter_clicked(self, _scatter, points, _ev):
        if self._stage != "updown_peaks" or not points:
            return
        idx = points[0].index()
        self._up_selected[idx] = not self._up_selected.get(idx, False)
        self._update_updown_scatter_colors()

    def _on_down_scatter_clicked(self, _scatter, points, _ev):
        if self._stage != "updown_peaks" or not points:
            return
        idx = points[0].index()
        self._down_selected[idx] = not self._down_selected.get(idx, False)
        self._update_updown_scatter_colors()

    def _confirm_updown_peaks(self):
        up_idx = sorted(i for i in range(len(self._up_peak_idx)) if self._up_selected.get(i, False))
        down_idx = sorted(i for i in range(len(self._down_peak_idx)) if self._down_selected.get(i, False))

        up_t = np.sort(self._up_centers[self._up_peak_idx[up_idx]]) if up_idx else np.array([])
        down_t = np.sort(self._up_centers[self._down_peak_idx[down_idx]]) if down_idx else np.array([])

        energies = np.asarray(self._radial_fit["energies"], dtype=np.float64)
        try:
            up_fit = None
            down_fit = None
            if len(up_t):
                up_energies = mc.match_energies(len(up_t), energies)
                up_fit = mc.fit_time_energy(up_t, up_energies)
            if len(down_t):
                down_energies = mc.match_energies(len(down_t), energies)
                down_fit = mc.fit_time_energy(down_t, down_energies)
        except Exception as exc:
            self.stage_status_var.set(f"Up/down fit failed: {exc}")
            return

        if up_fit is None and down_fit is None:
            self.stage_status_var.set("Select at least 1 peak in the up or down branch.")
            return

        self._up_fit = up_fit
        self._down_fit = down_fit
        self._calibration = mc.MomentumCalibration(
            cx=self._cx,
            cy=self._cy,
            angle=self._angle,
            t_center=self._t_center_t,
            wavelength_nm=float(self.wavelength_var.get()),
            radial_fit=self._radial_fit,
            up_fit=self._up_fit,
            down_fit=self._down_fit,
        )
        self.stage_status_var.set(
            f"Fits complete: {len(up_t)} up peak(s), {len(down_t)} down peak(s)."
        )
        self._enter_stage("results")

    # ===================================================================
    # Stage 5: results
    # ===================================================================

    def _build_results_controls(self):
        frame = ttk.LabelFrame(self._stage_frame, text="Step 5: calibrated momentum")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(frame, text="Save Fit...", command=self._save_fit).grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.Button(frame, text="Save Calibrated Dataset...", command=self._save_calibrated_dataset).grid(
            row=0, column=1, padx=6, pady=6, sticky="w"
        )

    def _momenta_for(self, df):
        return mc.compute_momenta(
            df["x"].to_numpy(), df["y"].to_numpy(), df["etof"].to_numpy(),
            self._radial_fit, self._up_fit, self._down_fit,
        )

    def _draw_results_plot(self):
        dfs = self._unique_dfs()
        px_parts, py_parts, pz_parts = [], [], []
        for df in dfs:
            px, py, pz = self._momenta_for(df)
            px_parts.append(px)
            py_parts.append(py)
            pz_parts.append(pz)
        px_all = np.concatenate(px_parts)
        pz_all = np.concatenate(pz_parts)

        self._plot_item.clear()
        p_max = max(1e-6, float(np.percentile(np.abs(np.concatenate([px_all, pz_all])), 99.5)))
        bins = 200
        hist, xedges, yedges = np.histogram2d(
            px_all, pz_all, bins=bins, range=[[-p_max, p_max], [-p_max, p_max]]
        )
        img = pg.ImageItem(hist_to_rgba(hist, log=True))
        img.setRect(QtCore.QRectF(-p_max, -p_max, 2 * p_max, 2 * p_max))
        self._plot_item.addItem(img)

        dash_pen = pg.mkPen("w", width=1.0, style=QtCore.Qt.DashLine)
        for radius in self._calibration.peak_momenta():
            cx_arr, cy_arr = circle_curve(radius)
            self._plot_item.addItem(pg.PlotDataItem(cx_arr, cy_arr, pen=dash_pen))

        self._plot_item.setTitle("Calibrated momentum: px vs pz (a.u.)")
        self._plot_item.setLabel("bottom", "px (a.u.)")
        self._plot_item.setLabel("left", "pz (a.u.)")
        self._plot_item.getViewBox().setAspectLocked(True)
        self._plot_item.getViewBox().setRange(xRange=(-p_max, p_max), yRange=(-p_max, p_max), padding=0)

    def _save_fit(self):
        if self._calibration is None:
            return
        chosen = filedialog.asksaveasfilename(
            title="Save momentum calibration", defaultextension=".json",
            filetypes=[("JSON", "*.json")], parent=self,
        )
        if not chosen:
            return
        try:
            self._calibration.save(chosen)
        except Exception as exc:
            self.stage_status_var.set(f"Save fit failed: {exc}")
            return
        self.stage_status_var.set(f"Saved calibration to {chosen}")

    def _save_calibrated_dataset(self):
        if self._calibration is None:
            return
        dfs = self._unique_dfs()
        labeled = []
        for label, df in zip(("xy", "etof"), dfs):
            px, py, pz = self._momenta_for(df)
            out = df.copy()
            out["px"] = px
            out["py"] = py
            out["pz"] = pz
            if len(dfs) > 1:
                out["source"] = label
            labeled.append(out)
        combined = pd.concat(labeled, ignore_index=True) if len(labeled) > 1 else labeled[0]

        chosen = filedialog.asksaveasfilename(
            title="Save calibrated dataset", defaultextension=".h5",
            filetypes=[("HDF5", "*.h5")], parent=self,
        )
        if not chosen:
            return
        try:
            mc.save_calibrated_dataset(combined, chosen)
        except Exception as exc:
            self.stage_status_var.set(f"Save dataset failed: {exc}")
            return
        self.stage_status_var.set(f"Saved calibrated dataset ({len(combined)} events) to {chosen}")
