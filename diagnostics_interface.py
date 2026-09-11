import json
import pathlib
import threading
import time
from queue import Queue, Empty

import numpy as np
import requests
import qtk as tk
from qtk import ttk
from qtk import filedialog, messagebox

import app_settings
import serval_client
from plot_panel import HistogramPlotPanel
from timewalk import DEFAULT_CORRECTION_PATH, TimewalkCorrection, apply_timewalk_correction
from tpx_processing import (
    decode_tpx3,
    sort_tdcs,
    group_pixels_by_pulse,
    group_times_relative,
    cluster_pixels_by_pulse,
    make_pixel_hist_from_pulses,
    make_cluster_hists,
    make_hist_1d,
    hist_args,
    flatten_dict,
    copy_hist,
    add_hist,
    add_hist_2d,
    summarize_records,
    merge_stats,
)


class DiagnosticsInterface(ttk.Frame):
    def __init__(self, parent, server_var=None, plot_shared_vars=None, coordinator=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._server_var = server_var
        self._fallback_server_var = tk.StringVar(self, value="http://localhost:8080")
        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)
        self._plot_shared_vars = plot_shared_vars

        # Loaded from whatever was last saved via the "Save current as
        # default" button (next to the tab bar); untouched until then.
        self.frame_rate_var = tk.StringVar(self, value=app_settings.get("diag.frame_rate", "10"))
        self.max_packets_var = tk.StringVar(self, value=app_settings.get("diag.max_packets", "10000"))
        self.accumulate_var = tk.BooleanVar(self, value=False)
        self.use_test_file_var = tk.BooleanVar(self, value=app_settings.get("diag.use_test_file", False))
        self.test_file_var = tk.StringVar(
            self, value=app_settings.get("diag.test_file", r"C:\Temp\serval_diag\test.tpx3")
        )
        self.mask_threshold_var = tk.StringVar(self, value=app_settings.get("diag.mask_threshold", "100"))
        self.status_var = tk.StringVar(self, value="Idle")

        self.timewalk_enabled_var = tk.BooleanVar(self, value=app_settings.get("diag.timewalk_enabled", False))
        self.timewalk_path_var = tk.StringVar(
            self, value=app_settings.get("diag.timewalk_path", str(DEFAULT_CORRECTION_PATH))
        )

        self._stop_event = threading.Event()
        self._queue: Queue = Queue()
        self._worker = None
        self._temp_dir = None
        self._active_timewalk = None

        self._accumulated = None
        self._reset_accumulation = False

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        controls.columnconfigure(12, weight=1)

        ttk.Label(controls, text="Frame rate (fps):").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.frame_rate_var, width=10).grid(row=0, column=1, padx=(6, 12))

        ttk.Label(controls, text="Max packets:").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.max_packets_var, width=10).grid(row=0, column=3, padx=(6, 12))

        ttk.Checkbutton(
            controls,
            text="Accumulate",
            variable=self.accumulate_var,
            command=self._on_accumulate_toggle,
        ).grid(row=0, column=4, padx=(0, 12))

        ttk.Checkbutton(
            controls,
            text="Use test file",
            variable=self.use_test_file_var,
        ).grid(row=0, column=5, padx=(0, 6))
        ttk.Entry(controls, textvariable=self.test_file_var, width=32).grid(row=0, column=6, padx=(0, 12))

        ttk.Label(controls, text="Mask pixels >").grid(row=0, column=7, sticky="w")
        ttk.Entry(controls, textvariable=self.mask_threshold_var, width=8).grid(row=0, column=8, padx=(6, 6))
        ttk.Button(controls, text="Mask", command=self._mask_hot_pixels).grid(row=0, column=9, padx=(0, 6))
        ttk.Button(controls, text="Mask Hottest Pixel", command=self._mask_max_pixel).grid(
            row=0, column=10, padx=(0, 12)
        )

        ttk.Button(controls, text="Start", command=self.start).grid(row=0, column=11, padx=(0, 6))
        ttk.Button(controls, text="Stop", command=self.stop).grid(row=0, column=12, padx=(0, 6))

        ttk.Label(controls, textvariable=self.status_var).grid(row=0, column=13, sticky="w")

        timewalk_row = ttk.Frame(controls)
        timewalk_row.grid(row=1, column=0, columnspan=14, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            timewalk_row, text="Apply timewalk correction before clustering", variable=self.timewalk_enabled_var
        ).grid(row=0, column=0, sticky="w")
        ttk.Entry(timewalk_row, textvariable=self.timewalk_path_var, width=40).grid(
            row=0, column=1, padx=(6, 6)
        )
        ttk.Button(timewalk_row, text="Browse...", command=self._browse_timewalk).grid(row=0, column=2)

        stats = ttk.LabelFrame(self, text="Diagnostics")
        stats.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        stats.columnconfigure(1, weight=1)
        stats.columnconfigure(3, weight=1)
        stats.columnconfigure(5, weight=1)
        stats.columnconfigure(7, weight=1)

        self._stat_vars = {}
        stat_groups = [
            ("Rates", [
                ("Laser Repetition Rate", "pulses_per_second"),
                ("Analysis Speed", "analysis_speed"),
                ("Analysis Frame Time", "analysis_time_last"),
                ("Fraction Analyzed", "data_ratio"),
            ]),
            ("Per-Shot Rates", [
                ("Cluster Rate", "clusters_per_shot"),
                ("e-ToF Rate", "electrons_per_shot"),
                ("i-ToF Rate", "ions_per_shot"),
            ]),
            ("Per-Cluster Ratios", [
                ("e-ToF : Cluster Ratio", "electrons_per_cluster"),
                ("i-ToF : Cluster Ratio", "ions_per_cluster"),
                ("Average Cluster Size", "avg_cluster_size"),
            ]),
            ("Event Fractions", [
                ("Coincidence Rate", "shots_all_three"),
                ("e-ToF Multi-Hit Rate", "shots_multi_electrons"),
                ("i-ToF Multi-Hit Rate", "shots_multi_ions"),
                ("Cluster Multi-Hit Rate", "shots_multi_clusters"),
            ]),
        ]

        col_groups = [0, 2, 4, 6]
        rows_per_col = [0, 0, 0, 0]
        for group_idx, (group_label, items) in enumerate(stat_groups):
            col = col_groups[group_idx % len(col_groups)]
            row = rows_per_col[group_idx % len(col_groups)]
            ttk.Label(stats, text=group_label, font=("Segoe UI", 10, "bold")).grid(
                row=row, column=col, columnspan=2, sticky="w", padx=(6, 4), pady=(6, 2)
            )
            row += 1
            for label, key in items:
                ttk.Label(stats, text=label + ":").grid(row=row, column=col, sticky="w", padx=(6, 4), pady=2)
                var = tk.StringVar(self, value="--")
                self._stat_vars[label] = var
                ttk.Label(stats, textvariable=var).grid(row=row, column=col + 1, sticky="w", pady=2)
                row += 1
            rows_per_col[group_idx % len(col_groups)] = row

        self._plot_panel = HistogramPlotPanel(
            self,
            on_settings_changed=self._on_panel_settings_changed,
            plots_first=False,
            shared_vars=self._plot_shared_vars,
        )
        self._plot_panel.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def _on_panel_settings_changed(self):
        self._reset_accumulation = True
        self._plot_panel.update_plots(self._current_data())

    def _current_data(self):
        return self._accumulated if self._accumulated is not None else self._plot_panel.empty_plot_data()

    def _on_accumulate_toggle(self):
        if not self.accumulate_var.get():
            self._reset_accumulation = True

    def _server_url(self):
        url = ""
        if self._server_var is not None:
            url = self._server_var.get().strip()
        if not url:
            url = self._fallback_server_var.get().strip() or "http://localhost:8080"
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        return url

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
            "diag.frame_rate": self.frame_rate_var,
            "diag.max_packets": self.max_packets_var,
            "diag.use_test_file": self.use_test_file_var,
            "diag.test_file": self.test_file_var,
            "diag.mask_threshold": self.mask_threshold_var,
            "diag.timewalk_enabled": self.timewalk_enabled_var,
            "diag.timewalk_path": self.timewalk_path_var,
        }

    def start(self):
        if self._coordinator is not None:
            self._coordinator.stop_others(self)

        if self.is_running():
            self.stop()

        try:
            frame_rate = float(self.frame_rate_var.get())
            max_packets = int(self.max_packets_var.get())
        except ValueError:
            self.status_var.set("Invalid frame rate or max packets.")
            return

        if self.timewalk_enabled_var.get():
            try:
                self._active_timewalk = TimewalkCorrection.load(self.timewalk_path_var.get().strip())
            except Exception as exc:
                self.status_var.set(f"Could not load timewalk correction: {exc}")
                return
        else:
            self._active_timewalk = None

        self._reset_accumulation = True
        self._accumulated = None
        self._stop_event.clear()
        self._temp_dir = self._create_temp_dir()

        test_mode = self.use_test_file_var.get()
        if not test_mode:
            if not self._configure_server(frame_rate):
                return
            self.status_var.set(f"Running. Temp dir: {self._temp_dir}")
            worker_args = (pathlib.Path(self._temp_dir), frame_rate, max_packets, False)
        else:
            self.status_var.set("Running diagnostics on test file.")
            worker_args = (pathlib.Path(self._temp_dir), frame_rate, max_packets, True)

        self._worker = threading.Thread(
            target=self._worker_loop,
            args=worker_args,
            daemon=True,
        )
        self._worker.start()

    def stop(self):
        self._stop_event.set()
        self._stop_measurement()
        self.status_var.set("Stopping...")
        if self._worker:
            self._worker.join(timeout=2)
        self.status_var.set("Stopped.")

    def _browse_timewalk(self):
        initial = self.timewalk_path_var.get().strip() or str(DEFAULT_CORRECTION_PATH)
        chosen = filedialog.askopenfilename(
            title="Load time-walk correction",
            initialdir=str(pathlib.Path(initial).parent),
            filetypes=[("JSON", "*.json")],
            parent=self,
        )
        if chosen:
            self.timewalk_path_var.set(chosen)

    def _configure_server(self, frame_rate):
        # One frame per trigger, restarted every frame (see _worker_loop) --
        # Diagnostics wants a fresh trigger for each single frame rather
        # than the continuous, many-frames-per-command pattern used by
        # Monitored Acquisition and the Parameter Sweep tab.
        try:
            server = self._server_url()
            frame_time = 1.0 / frame_rate if frame_rate > 0 else 0.1
            serval_client.configure_single_trigger_destination(server, frame_time, self._temp_dir, "diag%Hms_")
            return True
        except Exception as exc:
            self.status_var.set(f"Config failed: {type(exc).__name__}: {exc} (server={server})")
            return False

    def _create_temp_dir(self):
        base = pathlib.Path(r"C:\Temp")
        base.mkdir(parents=True, exist_ok=True)
        temp_dir = base / "serval_diag"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return str(temp_dir)

    def _start_measurement(self):
        try:
            server = self._server_url()
            serval_client.SESSION.get(f"{server}/measurement/start", timeout=5)
        except Exception as exc:
            self.status_var.set(f"Start failed: {type(exc).__name__}: {exc}")

    def _stop_measurement(self):
        try:
            server = self._server_url()
            serval_client.SESSION.get(f"{server}/measurement/stop", timeout=5)
        except Exception as exc:
            self.status_var.set(f"Stop failed: {type(exc).__name__}: {exc}")

    def _worker_loop(self, temp_dir: pathlib.Path, frame_rate: float, max_packets: int, test_mode: bool):
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
                    # nTriggers=1 (see _configure_server) means Serval
                    # auto-stops after each single frame, so a fresh start
                    # is needed for every one.
                    self._start_measurement()
                    newest = self._wait_for_new_file(temp_dir, last_path, last_mtime)
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

                result = self._process_file(newest, frame_rate, max_packets)
                if result:
                    self._queue.put(result)

                last_path = newest
                last_mtime = newest_stat.st_mtime
                if test_mode:
                    time.sleep(0.5)
            except Exception as exc:
                self._queue.put({"error": str(exc)})
                time.sleep(1)

    def _wait_for_new_file(self, temp_dir: pathlib.Path, last_path, last_mtime):
        start = time.time()
        while not self._stop_event.is_set():
            files = sorted(temp_dir.glob("*.tpx3"), key=lambda p: p.stat().st_mtime)
            if files:
                newest = files[-1]
                newest_stat = newest.stat()
                if newest != last_path or newest_stat.st_mtime != last_mtime:
                    if newest_stat.st_size > 0 and (time.time() - newest_stat.st_mtime) > 0.2:
                        return newest
            if time.time() - start > 5:
                return None
            time.sleep(0.1)
        return None

    def _process_file(self, path: pathlib.Path, frame_rate: float, max_packets: int):
        start = time.perf_counter()
        settings = self._plot_panel.hist_snapshot()

        pixels, tdcs, processed_packets, total_packets = decode_tpx3(str(path), max_packets=max_packets)
        if self._active_timewalk is not None:
            pixels = apply_timewalk_correction(pixels, self._active_timewalk)
        etof, itof, pulses = sort_tdcs(300.0, tdcs)

        pulses_sorted = sorted(pulses)
        pixels_by_pulse = group_pixels_by_pulse(pixels, pulses_sorted)
        clusters_by_pulse, cluster_size_sum, cluster_count = cluster_pixels_by_pulse(pixels_by_pulse)
        etof_by_pulse = group_times_relative(etof, pulses_sorted)
        itof_by_pulse = group_times_relative(itof, pulses_sorted)

        pulse_records = []
        for pulse_time in pulses_sorted:
            pulse_records.append(
                (
                    pulse_time,
                    pixels_by_pulse.get(pulse_time, []),
                    clusters_by_pulse.get(pulse_time, []),
                    etof_by_pulse.get(pulse_time, []),
                    itof_by_pulse.get(pulse_time, []),
                )
            )

        pixel_hist = make_pixel_hist_from_pulses(
            pixels_by_pulse,
            bins=int(settings["pixel"]["bins"]),
            range_min=settings["pixel"]["min"],
            range_max=settings["pixel"]["max"],
        )
        cluster_hist, cluster_t_hist = make_cluster_hists(
            clusters_by_pulse,
            t_bins=int(settings["cluster_t"]["bins"]),
            t_min=settings["cluster_t"]["min"],
            t_max=settings["cluster_t"]["max"],
            xy_bins=int(settings["cluster"]["bins"]),
            xy_min=settings["cluster"]["min"],
            xy_max=settings["cluster"]["max"],
        )
        itof_hist = make_hist_1d(flatten_dict(itof_by_pulse), **hist_args(settings["itof"]))
        etof_hist = make_hist_1d(flatten_dict(etof_by_pulse), **hist_args(settings["etof"]))

        stats = summarize_records(
            pulse_records,
            cluster_size_sum,
            cluster_count,
            processed_packets,
            total_packets,
            start,
        )
        stats["analysis_time_last"] = time.perf_counter() - start

        return {
            "pixel_hist": pixel_hist,
            "cluster_hist": cluster_hist,
            "itof_hist": itof_hist,
            "etof_hist": etof_hist,
            "cluster_t_hist": cluster_t_hist,
            "stats": stats,
            "pulse_records": pulse_records,
            "settings": settings,
        }

    def _apply_result(self, result):
        if "error" in result:
            self.status_var.set(f"Processing error: {result['error']}")
            return

        settings_changed = (
            self._accumulated is not None
            and self._accumulated.get("settings") != result.get("settings")
        )

        if (
            self._reset_accumulation
            or self._accumulated is None
            or not self.accumulate_var.get()
            or settings_changed
        ):
            self._accumulated = {
                "pixel_hist": result["pixel_hist"].copy(),
                "cluster_hist": result["cluster_hist"].copy(),
                "itof_hist": copy_hist(result["itof_hist"]),
                "etof_hist": copy_hist(result["etof_hist"]),
                "cluster_t_hist": copy_hist(result["cluster_t_hist"]),
                "stats": result["stats"].copy(),
                "settings": result["settings"],
            }
            self._reset_accumulation = False
        else:
            self._accumulated["pixel_hist"] = add_hist_2d(
                self._accumulated["pixel_hist"], result["pixel_hist"]
            )
            self._accumulated["cluster_hist"] = add_hist_2d(
                self._accumulated["cluster_hist"], result["cluster_hist"]
            )
            self._accumulated["itof_hist"] = add_hist(self._accumulated["itof_hist"], result["itof_hist"])
            self._accumulated["etof_hist"] = add_hist(self._accumulated["etof_hist"], result["etof_hist"])
            self._accumulated["cluster_t_hist"] = add_hist(
                self._accumulated["cluster_t_hist"],
                result["cluster_t_hist"],
            )
            self._accumulated["stats"] = merge_stats(self._accumulated["stats"], result["stats"])

        self._plot_panel.update_plots(self._accumulated)
        self._update_stats(self._accumulated["stats"])

    def _update_stats(self, stats):
        def fmt(value):
            return f"{value:.3f}" if isinstance(value, float) else str(value)

        analysis_speed = (stats["real_time"] / stats["analysis_time"]) if stats["analysis_time"] > 0 else 0

        self._stat_vars["Laser Repetition Rate"].set(fmt(stats["pulses_per_second"]))
        self._stat_vars["Analysis Speed"].set(fmt(analysis_speed))
        self._stat_vars["Analysis Frame Time"].set(fmt(stats.get("analysis_time_last", 0)))
        self._stat_vars["Fraction Analyzed"].set(fmt(stats["data_ratio"]))

        self._stat_vars["Cluster Rate"].set(fmt(stats["clusters_per_shot"]))
        self._stat_vars["e-ToF Rate"].set(fmt(stats["electrons_per_shot"]))
        self._stat_vars["i-ToF Rate"].set(fmt(stats["ions_per_shot"]))

        self._stat_vars["e-ToF : Cluster Ratio"].set(fmt(stats["electrons_per_cluster"]))
        self._stat_vars["i-ToF : Cluster Ratio"].set(fmt(stats["ions_per_cluster"]))
        self._stat_vars["Average Cluster Size"].set(fmt(stats["avg_cluster_size"]))

        self._stat_vars["Coincidence Rate"].set(fmt(stats["shots_all_three"]))
        self._stat_vars["e-ToF Multi-Hit Rate"].set(fmt(stats["shots_multi_electrons"]))
        self._stat_vars["i-ToF Multi-Hit Rate"].set(fmt(stats["shots_multi_ions"]))
        self._stat_vars["Cluster Multi-Hit Rate"].set(fmt(stats["shots_multi_clusters"]))

    def _mask_hot_pixels(self):
        was_running = self.is_running()
        if was_running:
            self.stop()
        try:
            self._do_mask_hot_pixels()
        finally:
            if was_running:
                self.start()

    def _do_mask_hot_pixels(self):
        try:
            threshold = int(self.mask_threshold_var.get())
        except ValueError:
            self.status_var.set("Invalid mask threshold.")
            return

        if self._accumulated is None:
            self.status_var.set("No data to mask.")
            return

        pixel_hist = self._accumulated.get("pixel_hist")
        if pixel_hist is None:
            self.status_var.set("No pixel histogram to mask.")
            return

        cfg = self._accumulated.get("settings", self._plot_panel.hist_snapshot())["pixel"]
        if int(cfg["bins"]) != 256 or float(cfg["min"]) != 0.0 or float(cfg["max"]) != 256.0:
            self.status_var.set("Masking needs the pixel map at 256 bins over 0-256.")
            return

        hot_coords = np.argwhere(pixel_hist > threshold)
        if hot_coords.size == 0:
            self.status_var.set("No pixels above threshold.")
            return

        mask_count = int(hot_coords.shape[0])
        if not messagebox.askyesno(
            "Confirm Masking",
            f"Pixels to be masked: {mask_count}\n"
            f"Threshold: counts > {threshold}\n"
            "Proceed with masking on chip 0?",
            parent=self,
        ):
            self.status_var.set("Mask canceled.")
            return

        try:
            server = self._server_url()
            masked = 0
            for x, y in hot_coords:
                url = f"{server}/detector/chips/0/mask/{int(y)}/{int(x)}"
                serval_client.SESSION.put(url, data="1", timeout=2)
                masked += 1
            self.status_var.set(f"Masked {masked} pixels (>{threshold}).")
        except Exception as exc:
            self.status_var.set(f"Mask failed: {type(exc).__name__}: {exc}")

    def _mask_max_pixel(self):
        was_running = self.is_running()
        if was_running:
            self.stop()
        try:
            self._do_mask_max_pixel()
        finally:
            if was_running:
                self.start()

    def _do_mask_max_pixel(self):
        if self._accumulated is None:
            self.status_var.set("No data to mask.")
            return

        pixel_hist = self._accumulated.get("pixel_hist")
        if pixel_hist is None or pixel_hist.size == 0:
            self.status_var.set("No pixel histogram to mask.")
            return

        cfg = self._accumulated.get("settings", self._plot_panel.hist_snapshot())["pixel"]
        if int(cfg["bins"]) != 256 or float(cfg["min"]) != 0.0 or float(cfg["max"]) != 256.0:
            self.status_var.set("Masking needs the pixel map at 256 bins over 0-256.")
            return

        max_count = float(pixel_hist.max())
        if max_count <= 0:
            self.status_var.set("No pixel counts to mask.")
            return

        x, y = (int(v) for v in np.unravel_index(np.argmax(pixel_hist), pixel_hist.shape))

        try:
            server = self._server_url()
            url = f"{server}/detector/chips/0/mask/{y}/{x}"
            serval_client.SESSION.put(url, data="1", timeout=2)
            self.status_var.set(f"Masked hottest pixel ({x}, {y}) with {int(max_count)} counts.")
        except Exception as exc:
            self.status_var.set(f"Mask failed: {type(exc).__name__}: {exc}")
