import pathlib
import threading
import time
from queue import Queue, Empty

import numpy as np
import qtk as tk
from qtk import ttk, filedialog, messagebox, grid_into

import pyqtgraph as pg
from PyQt5 import QtCore

import app_settings
import serval_client
import ui_style
from qt_plots import hist_to_rgba, mpl_color, ZoomFocusViewBox
from timewalk import DEFAULT_CORRECTION_PATH, TimewalkCorrection, generate_correction
from tpx_processing import decode_tpx3, sort_tdcs, group_pixels_by_pulse


class TimewalkInterface(ttk.Frame):
    """Accumulates a 2D histogram of pixel (ToT, t) across frames and fits a
    time-walk correction from it: t should not depend on ToT, so any
    ToT-dependent drift of the hit-time ridge is walk to correct out. The
    correction is anchored to ~0 at the high-ToT end (those hits are least
    affected and should stay essentially unchanged), then saved to a JSON
    file that the Diagnostics / Monitored Acquisition tabs can load and
    apply to pixel times before clustering.
    """

    def __init__(self, parent, server_var=None, coordinator=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._server_var = server_var
        self._fallback_server_var = tk.StringVar(self, value="http://localhost:8080")
        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)

        # Loaded from whatever was last saved via the "Save current as
        # default" button (next to the tab bar); untouched until then.
        self.frame_rate_var = tk.StringVar(self, value=app_settings.get("timewalk.frame_rate", "10"))
        self.max_packets_var = tk.StringVar(self, value=app_settings.get("timewalk.max_packets", "10000"))
        self.use_test_file_var = tk.BooleanVar(self, value=app_settings.get("timewalk.use_test_file", False))
        self.test_file_var = tk.StringVar(
            self, value=app_settings.get("timewalk.test_file", r"C:\Temp\serval_diag\test.tpx3")
        )
        self.status_var = tk.StringVar(self, value="Idle")

        self.tot_bins_var = tk.StringVar(self, value=app_settings.get("timewalk.tot_bins", "256"))
        self.tot_min_var = tk.StringVar(self, value=app_settings.get("timewalk.tot_min", "0"))
        self.tot_max_var = tk.StringVar(self, value=app_settings.get("timewalk.tot_max", "1024"))
        self.t_bins_var = tk.StringVar(self, value=app_settings.get("timewalk.t_bins", "500"))
        self.t_min_var = tk.StringVar(self, value=app_settings.get("timewalk.t_min", "0"))
        self.t_max_var = tk.StringVar(self, value=app_settings.get("timewalk.t_max", "1000"))
        self.min_counts_var = tk.StringVar(self, value=app_settings.get("timewalk.min_counts", "20"))
        self.anchor_frac_var = tk.StringVar(self, value=app_settings.get("timewalk.anchor_frac", "10"))
        self.log_color_var = tk.BooleanVar(self, value=app_settings.get("timewalk.log_color", True))

        self.correction_path_var = tk.StringVar(
            self, value=app_settings.get("timewalk.correction_path", str(DEFAULT_CORRECTION_PATH))
        )
        self.correction_info_var = tk.StringVar(self, value="No correction generated yet.")

        self._stop_event = threading.Event()
        self._queue: Queue = Queue()
        self._worker = None
        self._temp_dir = None

        self._hist2d = None
        self._tot_edges = None
        self._t_edges = None
        self._correction = None

        self._build_ui()
        self._poll_queue()
        self._redraw()

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.rowconfigure(0, weight=1)
        # Trailing empty row keeps content packed at the top as sections
        # collapse (see Monitored Acquisition).
        sidebar.rowconfigure(5, weight=1)

        ui_style.build_button_bar(sidebar, 0, [
            ("Start", self.start),
            ("Stop", self.stop),
            ("Reset Accumulation", self.reset_accumulation),
        ])
        self._status_block = ui_style.StatusBlock(sidebar, 1, self.status_var)

        params = ui_style.build_section(sidebar, 2, "Acquisition parameters")
        ui_style.add_form_row(
            params, 0, "Frame rate (fps):", ttk.Entry(params, textvariable=self.frame_rate_var, width=18)
        )
        ui_style.add_form_row(params, 1, "Max packets:", ttk.Entry(params, textvariable=self.max_packets_var, width=18))
        ttk.Checkbutton(params, text="Use test file", variable=self.use_test_file_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ui_style.build_path_field(params, self.test_file_var, self._browse_test_file).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(2, 4)
        )

        # Set-once bins/bounds start collapsed, same as the plot panel's.
        hist = ui_style.build_section(sidebar, 3, "Histogram bins / bounds", collapsed=True, columns_stretch=None)
        ttk.Label(hist, text="Axis").grid(row=0, column=0, sticky="w", padx=(6, 4), pady=(4, 2))
        ttk.Label(hist, text="Bins").grid(row=0, column=1, pady=(4, 2))
        ttk.Label(hist, text="Min").grid(row=0, column=2, pady=(4, 2))
        ttk.Label(hist, text="Max").grid(row=0, column=3, pady=(4, 2))
        axes = [
            ("ToT", self.tot_bins_var, self.tot_min_var, self.tot_max_var),
            ("t (ns)", self.t_bins_var, self.t_min_var, self.t_max_var),
        ]
        for row_idx, (label, *field_vars) in enumerate(axes, start=1):
            bottom_pad = 4 if row_idx == len(axes) else 1
            ttk.Label(hist, text=label).grid(row=row_idx, column=0, sticky="w", padx=(6, 4), pady=(1, bottom_pad))
            for col, var in enumerate(field_vars, start=1):
                ttk.Entry(hist, textvariable=var, width=9).grid(row=row_idx, column=col, padx=2, pady=(1, bottom_pad))
        ttk.Checkbutton(hist, text="Log color scale", variable=self.log_color_var, command=self._redraw).grid(
            row=len(axes) + 1, column=0, columnspan=4, sticky="w", padx=(6, 0), pady=(4, 0)
        )

        fit = ui_style.build_section(sidebar, 4, "Correction fit", pady=0)
        ui_style.add_form_row(
            fit, 0, "Min counts per column:", ttk.Entry(fit, textvariable=self.min_counts_var, width=18)
        )
        ui_style.add_form_row(
            fit, 1, "Anchor top % of ToT:", ttk.Entry(fit, textvariable=self.anchor_frac_var, width=18)
        )
        ui_style.build_button_bar(
            fit, 2, [("Generate Correction", self.generate_correction)], pady=(4, 6), columnspan=2
        )
        ui_style.add_form_row(fit, 3, "Correction file:", ui_style.build_path_field(fit, self.correction_path_var))
        ui_style.build_button_bar(
            fit, 4, [("Save As...", self.save_correction), ("Load...", self.load_correction)],
            pady=(4, 6), columnspan=2,
        )
        ui_style.add_note(fit, 5, textvariable=self.correction_info_var)

        self._glw = pg.GraphicsLayoutWidget()
        self._plot_hist = self._glw.addPlot(row=0, col=0, viewBox=ZoomFocusViewBox())
        self._plot_hist.addLegend()
        self._plot_corr = self._glw.addPlot(row=0, col=1, viewBox=ZoomFocusViewBox())
        grid_into(self._glw, main, row=0, column=0, sticky="nsew")

    # ---- settings -----------------------------------------------------

    def _read_hist_settings(self):
        tot_bins = max(1, min(2048, int(float(self.tot_bins_var.get()))))
        tot_min = float(self.tot_min_var.get())
        tot_max = float(self.tot_max_var.get())
        t_bins = max(1, min(8192, int(float(self.t_bins_var.get()))))
        t_min = float(self.t_min_var.get())
        t_max = float(self.t_max_var.get())
        if tot_max <= tot_min or t_max <= t_min:
            raise ValueError("ToT/t max must be greater than min.")
        return tot_bins, tot_min, tot_max, t_bins, t_min, t_max

    def reset_accumulation(self):
        self._hist2d = None
        self._tot_edges = None
        self._t_edges = None
        self._correction = None
        self.correction_info_var.set("No correction generated yet.")
        self._redraw()

    def _browse_test_file(self):
        initial = self.test_file_var.get().strip()
        chosen = filedialog.askopenfilename(
            title="Choose test file",
            initialdir=str(pathlib.Path(initial).parent) if initial else None,
            filetypes=[("TPX3", "*.tpx3")],
            parent=self,
        )
        if chosen:
            self.test_file_var.set(chosen)

    # ---- server / worker (mirrors the Diagnostics tab's file-per-exposure loop) --

    def _server_url(self):
        url = ""
        if self._server_var is not None:
            url = self._server_var.get().strip()
        if not url:
            url = self._fallback_server_var.get().strip() or "http://localhost:8080"
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        return url

    def _create_temp_dir(self):
        base = pathlib.Path(r"C:\Temp")
        base.mkdir(parents=True, exist_ok=True)
        temp_dir = base / "serval_timewalk"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return str(temp_dir)

    def _poll_queue(self):
        try:
            while True:
                result = self._queue.get_nowait()
                self._apply_result(result)
        except Empty:
            pass
        self.after(200, self._poll_queue)

    def is_running(self):
        return bool(self._worker and self._worker.is_alive())

    def default_fields(self):
        """Fields the global "Save current as default" button can persist."""
        return {
            "timewalk.frame_rate": self.frame_rate_var,
            "timewalk.max_packets": self.max_packets_var,
            "timewalk.use_test_file": self.use_test_file_var,
            "timewalk.test_file": self.test_file_var,
            "timewalk.tot_bins": self.tot_bins_var,
            "timewalk.tot_min": self.tot_min_var,
            "timewalk.tot_max": self.tot_max_var,
            "timewalk.t_bins": self.t_bins_var,
            "timewalk.t_min": self.t_min_var,
            "timewalk.t_max": self.t_max_var,
            "timewalk.min_counts": self.min_counts_var,
            "timewalk.anchor_frac": self.anchor_frac_var,
            "timewalk.log_color": self.log_color_var,
            "timewalk.correction_path": self.correction_path_var,
        }

    def start(self):
        if self._coordinator is not None:
            self._coordinator.stop_others(self)

        if self.is_running():
            self.stop()

        try:
            frame_rate = float(self.frame_rate_var.get())
            max_packets = int(self.max_packets_var.get())
            hist_settings = self._read_hist_settings()
        except ValueError as exc:
            self.status_var.set(f"Invalid setting: {exc}")
            return

        self._stop_event.clear()
        self._temp_dir = self._create_temp_dir()

        test_mode = self.use_test_file_var.get()
        if not test_mode:
            try:
                server = self._server_url()
                serval_client.configure_single_trigger_destination(
                    server, 1.0 / frame_rate if frame_rate > 0 else 0.1, self._temp_dir, "twcal%Hms_"
                )
            except Exception as exc:
                self.status_var.set(f"Config failed: {exc}")
                return
            self.status_var.set(f"Running. Temp dir: {self._temp_dir}")
        else:
            self.status_var.set("Accumulating from test file.")

        self._worker = threading.Thread(
            target=self._worker_loop,
            args=(pathlib.Path(self._temp_dir), max_packets, test_mode, hist_settings),
            daemon=True,
        )
        self._worker.start()

    def stop(self):
        self._stop_event.set()
        try:
            serval_client.stop_measurement(self._server_url())
        except Exception:
            pass
        self.status_var.set("Stopping...")
        if self._worker:
            self._worker.join(timeout=2)
        self.status_var.set("Stopped.")

    def _worker_loop(self, temp_dir: pathlib.Path, max_packets, test_mode, hist_settings):
        server = self._server_url()
        last_path = None
        last_mtime = None

        while not self._stop_event.is_set():
            try:
                if test_mode:
                    newest = pathlib.Path(self.test_file_var.get().strip())
                    if not newest.exists():
                        time.sleep(0.5)
                        continue
                    newest_stat = newest.stat()
                    if newest_stat.st_size == 0:
                        time.sleep(0.5)
                        continue
                else:
                    try:
                        serval_client.start_measurement(server)
                    except Exception as exc:
                        self._queue.put({"error": f"Start failed: {exc}"})
                        time.sleep(1)
                        continue

                    newest = serval_client.wait_for_new_file(temp_dir, self._stop_event, last_path, last_mtime)
                    if newest is None:
                        continue

                    newest_stat = newest.stat()
                    if newest_stat.st_size == 0:
                        time.sleep(0.2)
                        continue

                    files = sorted(temp_dir.glob("*.tpx3"), key=lambda p: p.stat().st_mtime)
                    test_path = pathlib.Path(self.test_file_var.get().strip()) if self.test_file_var.get() else None
                    for old in files[:-1]:
                        if test_path and old.resolve() == test_path.resolve():
                            continue
                        try:
                            old.unlink()
                        except OSError:
                            pass

                result = self._process_file(newest, max_packets, hist_settings)
                if result is not None:
                    self._queue.put(result)

                last_path = newest
                last_mtime = newest_stat.st_mtime
                if test_mode:
                    time.sleep(0.5)
            except Exception as exc:
                self._queue.put({"error": str(exc)})
                time.sleep(1)

    def _process_file(self, path, max_packets, hist_settings):
        tot_bins, tot_min, tot_max, t_bins, t_min, t_max = hist_settings

        pixels, tdcs, _, _ = decode_tpx3(str(path), max_packets=max_packets)
        _, _, pulses = sort_tdcs(300.0, tdcs)
        pulses_sorted = sorted(pulses)
        pixels_by_pulse = group_pixels_by_pulse(pixels, pulses_sorted)

        tots = []
        ts = []
        for pix_list in pixels_by_pulse.values():
            for t_rel, _, _, tot in pix_list:
                tots.append(tot)
                ts.append(t_rel)

        if not tots:
            return None

        tot_edges = np.linspace(tot_min, tot_max, tot_bins + 1)
        t_edges = np.linspace(t_min, t_max, t_bins + 1)
        hist2d, _, _ = np.histogram2d(tots, ts, bins=[tot_edges, t_edges])
        return {"hist2d": hist2d, "tot_edges": tot_edges, "t_edges": t_edges}

    def _apply_result(self, result):
        if "error" in result:
            self.status_var.set(f"Processing error: {result['error']}")
            return

        hist2d = result["hist2d"]
        tot_edges = result["tot_edges"]
        t_edges = result["t_edges"]

        if (
            self._hist2d is None
            or self._hist2d.shape != hist2d.shape
            or not np.array_equal(self._tot_edges, tot_edges)
            or not np.array_equal(self._t_edges, t_edges)
        ):
            self._hist2d = hist2d.copy()
            self._tot_edges = tot_edges
            self._t_edges = t_edges
        else:
            self._hist2d = self._hist2d + hist2d

        self._redraw()

    # ---- correction generation / persistence -------------------------

    def generate_correction(self):
        if self._hist2d is None or self._hist2d.sum() == 0:
            self.status_var.set("No accumulated data to fit a correction from yet.")
            return
        try:
            min_counts = int(float(self.min_counts_var.get()))
            anchor_top_frac = max(0.001, min(1.0, float(self.anchor_frac_var.get()) / 100.0))
        except ValueError:
            self.status_var.set("Invalid min-counts / anchor-% setting.")
            return

        try:
            self._correction = generate_correction(
                self._hist2d, self._tot_edges, self._t_edges,
                min_counts=min_counts, anchor_top_frac=anchor_top_frac,
            )
        except ValueError as exc:
            self.status_var.set(str(exc))
            return

        anchor = self._correction.meta["anchor_ns"]
        lo = float(self._correction.correction_ns.min())
        hi = float(self._correction.correction_ns.max())
        self.correction_info_var.set(
            f"Correction fit from {int(self._hist2d.sum())} pixel hits. "
            f"Anchor (high-ToT) t = {anchor:.3f} ns. "
            f"Correction ranges {lo:.3f} to {hi:.3f} ns across ToT. Not yet saved."
        )
        self.status_var.set("Correction generated. Review the plot, then Save As.")
        self._redraw()

    def save_correction(self):
        if self._correction is None:
            self.status_var.set("Generate a correction first.")
            return
        initial = self.correction_path_var.get().strip() or str(DEFAULT_CORRECTION_PATH)
        chosen = filedialog.asksaveasfilename(
            title="Save time-walk correction",
            defaultextension=".json",
            initialfile=pathlib.Path(initial).name,
            initialdir=str(pathlib.Path(initial).parent),
            filetypes=[("JSON", "*.json")],
            parent=self,
        )
        if not chosen:
            return
        try:
            self._correction.save(chosen)
        except Exception as exc:
            self.status_var.set(f"Save failed: {exc}")
            return
        self.correction_path_var.set(chosen)
        self.status_var.set(f"Saved correction to {chosen}")

    def load_correction(self):
        initial = self.correction_path_var.get().strip() or str(DEFAULT_CORRECTION_PATH)
        chosen = filedialog.askopenfilename(
            title="Load time-walk correction",
            initialdir=str(pathlib.Path(initial).parent),
            filetypes=[("JSON", "*.json")],
            parent=self,
        )
        if not chosen:
            return
        try:
            self._correction = TimewalkCorrection.load(chosen)
        except Exception as exc:
            self.status_var.set(f"Load failed: {exc}")
            return
        self.correction_path_var.set(chosen)
        anchor = self._correction.meta.get("anchor_ns", float("nan"))
        self.correction_info_var.set(
            f"Loaded correction from {chosen}. Anchor (high-ToT) t = {anchor:.3f} ns."
        )
        self.status_var.set(f"Loaded {chosen}")
        self._redraw()

    # ---- drawing --------------------------------------------------------

    def _redraw(self):
        self._plot_hist.clear()
        self._plot_corr.clear()

        has_hist = self._hist2d is not None and self._hist2d.sum() > 0
        if has_hist:
            x0, x1 = self._tot_edges[0], self._tot_edges[-1]
            y0, y1 = self._t_edges[0], self._t_edges[-1]
            img = pg.ImageItem(hist_to_rgba(self._hist2d, log=self.log_color_var.get()))
            img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
            self._plot_hist.addItem(img)

        self._plot_hist.setTitle("Pixel hits: ToT vs t")
        self._plot_hist.setLabel("bottom", "ToT")
        self._plot_hist.setLabel("left", "t (ns, relative to pulse)")

        if self._correction is not None:
            tot_c = self._correction.tot_centers
            ridge = np.asarray(self._correction.meta.get("ridge_ns", []))
            if ridge.size == tot_c.size:
                self._plot_hist.addItem(
                    pg.PlotDataItem(tot_c, ridge, pen=pg.mkPen("w", width=1.2), name="fitted ridge")
                )

            self._plot_corr.addItem(
                pg.PlotDataItem(tot_c, self._correction.correction_ns, pen=pg.mkPen(mpl_color("tab:orange")))
            )
            self._plot_corr.addItem(
                pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen("gray", width=0.8, style=QtCore.Qt.DashLine))
            )
        self._plot_corr.setTitle("Fitted correction")
        self._plot_corr.setLabel("bottom", "ToT")
        self._plot_corr.setLabel("left", "Correction subtracted from t (ns)")

        self._plot_hist.autoRange()
        self._plot_corr.autoRange()

        # clear() above removes the placeholders too, so they're re-added
        # on every redraw.
        ui_style.show_empty_placeholder(
            self._plot_hist, ui_style.add_empty_placeholder(self._plot_hist), not has_hist
        )
        ui_style.show_empty_placeholder(
            self._plot_corr,
            ui_style.add_empty_placeholder(self._plot_corr, "No correction yet — press Generate Correction"),
            self._correction is None,
        )
