import collections
import json
import os
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
import chip_layout
import platform_paths
import raw_conversion
import serval_client
import shared_state
import ui_style
from plot_panel import HistogramPlotPanel
from cluster_timing import load_correction
from timewalk import DEFAULT_CORRECTION_PATH
from tpx_processing import copy_hist, add_hist, add_hist_2d, merge_stats


DEFAULT_TEMP_DIR = str(platform_paths.SCRATCH_ROOT / "serval_diag")
# Serval names Diagnostics' files <prefix><time>..., and only those are read
# or cleaned up, so the temp folder can be any folder.
FILE_PREFIX = "diag"
# How often the temp folder is checked for a newly finished frame.
_FILE_POLL_S = 0.02
# The newest frame (no newer file after it yet) counts as finished once it
# has gone this long unmodified -- the last frame of a batch.
_SETTLE_S = 0.2
# How often Serval is asked whether a finished batch has wound down.
_BATCH_POLL_S = 0.05
# Plot refresh period (ms): results that arrive in between are merged and
# drawn once.
_REDRAW_MS = 30


def _list_frames(temp_dir):
    """(path, mtime, size) of every Diagnostics frame in temp_dir, oldest
    first (Serval's own file counter breaks mtime ties)."""
    frames = []
    with os.scandir(temp_dir) as entries:
        for entry in entries:
            if entry.name.startswith(FILE_PREFIX) and entry.name.endswith(".tpx3"):
                try:
                    st = entry.stat()
                except OSError:
                    continue
                frames.append((pathlib.Path(entry.path), st.st_mtime, st.st_size))
    frames.sort(key=lambda f: (f[1], f[0].name))
    return frames


def _newest_finished(frames):
    """Index into `frames` of the newest frame Serval is done writing, or
    None. In continuous mode Serval opens the next frame's file when a
    frame ends, so any frame with a newer one after it is finished; the
    newest one only counts once it has sat unmodified for _SETTLE_S."""
    now = time.time()
    for i in range(len(frames) - 1, -1, -1):
        _, mtime, size = frames[i]
        if size <= 0:
            continue
        if i < len(frames) - 1 or now - mtime > _SETTLE_S:
            return i
    return None


class DiagnosticsInterface(ttk.Frame):
    def __init__(self, parent, server_var=None, plot_shared_vars=None, coordinator=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._server_var = server_var
        self._fallback_server_var = tk.StringVar(self, value="http://localhost:8080")
        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)
        self._plot_shared_vars = plot_shared_vars
        self.min_cluster_size_var = shared_state.min_cluster_size_var(self, plot_shared_vars)
        self.cluster_timing_mode_var = shared_state.cluster_timing_mode_var(self, plot_shared_vars)

        # Loaded from whatever was last saved via the "Save current as
        # default" button (next to the tab bar); untouched until then.
        self.frame_rate_var = tk.StringVar(self, value=app_settings.get("diag.frame_rate", "10"))
        # Off = every packet of each frame is analysed.
        self.limit_packets_var = tk.BooleanVar(self, value=app_settings.get("diag.limit_packets", True))
        self.max_packets_var = tk.StringVar(self, value=app_settings.get("diag.max_packets", "10000"))
        self.temp_dir_var = tk.StringVar(self, value=app_settings.get("diag.temp_dir", DEFAULT_TEMP_DIR))
        self.accumulate_var = tk.BooleanVar(self, value=False)
        self.use_test_file_var = tk.BooleanVar(self, value=app_settings.get("diag.use_test_file", False))
        self.test_file_var = tk.StringVar(
            self, value=app_settings.get("diag.test_file", str(pathlib.Path(DEFAULT_TEMP_DIR) / "test.tpx3"))
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
        # When each recent frame was shown (for the displayed frame rate),
        # and how many finished frames the live loop had to skip.
        self._shown_times = collections.deque()
        self._frames_skipped = 0
        self._batch_frames = 1

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.rowconfigure(0, weight=1)
        # Trailing empty row keeps content packed at the top as sections
        # collapse (see Monitored Acquisition).
        sidebar.rowconfigure(8, weight=1)

        ui_style.build_button_bar(sidebar, 0, [("Start", self.start), ("Stop", self.stop)])
        self._status_block = ui_style.StatusBlock(sidebar, 1, self.status_var)

        params = ui_style.build_section(sidebar, 2, "Acquisition parameters")
        ui_style.add_form_row(
            params, 0, "Frame rate (fps):", ttk.Entry(params, textvariable=self.frame_rate_var, width=18)
        )
        packets = ttk.Frame(params)
        packets.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            packets, text="Limit to", variable=self.limit_packets_var, command=self._on_limit_packets_toggle,
        ).grid(row=0, column=0, sticky="w", padx=(0, 4))
        self._max_packets_entry = ttk.Entry(packets, textvariable=self.max_packets_var, width=10)
        self._max_packets_entry.grid(row=0, column=1, sticky="ew")
        ui_style.add_form_row(params, 1, "Packets per frame:", packets)
        self._on_limit_packets_toggle()
        ui_style.add_form_row(
            params, 2, "Temp folder:", ui_style.build_path_field(params, self.temp_dir_var, self._browse_temp_dir),
        )
        ui_style.add_note(
            params, 3,
            f"Serval writes each frame here as {FILE_PREFIX}*.tpx3; older {FILE_PREFIX}*.tpx3 files in it are "
            "deleted as new ones arrive. Other files are left alone.",
        )
        ttk.Checkbutton(
            params, text="Accumulate", variable=self.accumulate_var, command=self._on_accumulate_toggle,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            params, text="Use test file", variable=self.use_test_file_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ui_style.build_path_field(params, self.test_file_var, self._browse_test_file).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(2, 4)
        )
        ttk.Checkbutton(
            params, text="Apply timewalk correction before clustering", variable=self.timewalk_enabled_var
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ui_style.build_path_field(params, self.timewalk_path_var, self._browse_timewalk).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(2, 4)
        )
        ui_style.build_cluster_timing_mode_row(params, 9, self.cluster_timing_mode_var)
        ui_style.add_form_row(params, 11, "Min cluster size (hits):", ttk.Entry(params, textvariable=self.min_cluster_size_var, width=18))

        # An occasional maintenance action, not an every-session one.
        mask = ui_style.build_section(sidebar, 3, "Hot pixel masking", collapsed=True)
        ui_style.add_form_row(
            mask, 0, "Mask pixels above (counts):", ttk.Entry(mask, textvariable=self.mask_threshold_var, width=18)
        )
        ui_style.build_button_bar(mask, 1, [
            ("Mask", self._mask_hot_pixels),
            ("Mask Hottest Pixel", self._mask_max_pixel),
        ], pady=(4, 0), columnspan=2)

        self._stat_vars = {}
        stat_groups = [
            ("Rates", [
                ("Frames shown (fps):", "display_fps"),
                ("Frames skipped:", "frames_skipped"),
                ("Laser repetition rate:", "pulses_per_second"),
                ("Analysis speed:", "analysis_speed"),
                ("Analysis frame time:", "analysis_time_last"),
                ("Fraction analyzed:", "data_ratio"),
            ]),
            ("Per-shot rates", [
                ("Cluster rate:", "clusters_per_shot"),
                ("e-ToF rate:", "electrons_per_shot"),
                ("i-ToF rate:", "ions_per_shot"),
            ]),
            ("Per-cluster ratios", [
                ("e-ToF : cluster ratio:", "electrons_per_cluster"),
                ("i-ToF : cluster ratio:", "ions_per_cluster"),
                ("Average cluster size:", "avg_cluster_size"),
            ]),
            ("Event fractions", [
                ("Coincidence rate:", "shots_all_three"),
                ("e-ToF multi-hit rate:", "shots_multi_electrons"),
                ("i-ToF multi-hit rate:", "shots_multi_ions"),
                ("Cluster multi-hit rate:", "shots_multi_clusters"),
            ]),
        ]
        for idx, (title, items) in enumerate(stat_groups):
            last = idx == len(stat_groups) - 1
            body = ui_style.build_section(sidebar, 4 + idx, title, pady=0 if last else (0, 8))
            rows = []
            for label, key in items:
                var = tk.StringVar(self, value="--")
                self._stat_vars[key] = var
                rows.append((label, var))
            ui_style.add_stat_rows(body, rows)

        self._plot_panel = HistogramPlotPanel(
            main,
            on_settings_changed=self._on_panel_settings_changed,
            plots_first=True,
            shared_vars=self._plot_shared_vars,
        )
        self._plot_panel.grid(row=0, column=0, sticky="nsew")

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
        """Merge every result that arrived since the last tick, then redraw
        once -- drawing costs the same however many frames went into it."""
        got_frame = False
        try:
            while True:
                result = self._queue.get_nowait()
                if "error" in result:
                    self.status_var.set(f"Processing error: {result['error']}")
                    continue
                self._accumulate(result)
                got_frame = True
        except Empty:
            pass
        if got_frame:
            self._plot_panel.update_plots(self._accumulated)
            self._update_stats(self._accumulated["stats"])
        self.after(_REDRAW_MS, self._poll_queue)

    def is_running(self):
        return bool(self._worker and self._worker.is_alive())

    def default_fields(self):
        """Fields the global "Save current as default" button can persist."""
        return {
            "diag.frame_rate": self.frame_rate_var,
            "diag.limit_packets": self.limit_packets_var,
            "diag.max_packets": self.max_packets_var,
            "diag.temp_dir": self.temp_dir_var,
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
            max_packets = int(self.max_packets_var.get()) if self.limit_packets_var.get() else None
        except ValueError:
            self.status_var.set("Invalid frame rate or packet limit.")
            return
        if max_packets is not None and max_packets < 1:
            self.status_var.set("The packet limit must be at least 1 (or untick Limit to).")
            return

        if self.timewalk_enabled_var.get():
            try:
                self._active_timewalk = load_correction(self.timewalk_path_var.get().strip(), mode=self.cluster_timing_mode_var.get())
            except Exception as exc:
                self.status_var.set(f"Could not load timewalk correction: {exc}")
                return
        else:
            self._active_timewalk = None

        self._reset_accumulation = True
        self._accumulated = None
        self._shown_times.clear()
        self._frames_skipped = 0
        self._stop_event.clear()
        test_mode = self.use_test_file_var.get()
        try:
            self._temp_dir = self._create_temp_dir()
        except OSError as exc:
            self.status_var.set(f"Could not use the temp folder: {exc}")
            return

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

    def _on_limit_packets_toggle(self):
        self._max_packets_entry.setEnabled(bool(self.limit_packets_var.get()))

    def _browse_timewalk(self):
        initial = self.timewalk_path_var.get().strip() or str(DEFAULT_CORRECTION_PATH)
        chosen = filedialog.askopenfilename(
            title="Load time-walk correction",
            initialdir=str(pathlib.Path(initial).parent),
            filetypes=[("Time-walk / cluster timing", "*.json *.npz"), ("Legacy LUT", "*.json"), ("Cluster timing", "*.npz")],
            parent=self,
        )
        if chosen:
            self.timewalk_path_var.set(chosen)

    def _browse_temp_dir(self):
        chosen = filedialog.askdirectory(
            initialdir=self.temp_dir_var.get().strip() or DEFAULT_TEMP_DIR,
            title="Choose temp folder for diagnostic frames",
            parent=self,
        )
        if chosen:
            self.temp_dir_var.set(chosen)

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

    def _configure_server(self, frame_rate):
        """Continuous mode: one start_measurement() makes Serval write a new
        raw file every frame on its own, so the view isn't held back by a
        start/re-arm round trip per frame. Each start is a bounded batch
        (serval_client.max_frames_per_batch); _live_loop starts the next
        batch once Serval goes idle."""
        server = self._server_url()
        try:
            frame_time = 1.0 / frame_rate if frame_rate > 0 else 0.1
            self._batch_frames = serval_client.max_frames_per_batch(frame_time)
            serval_client.configure_continuous_destination(
                server, frame_time, self._temp_dir, f"{FILE_PREFIX}%Hms_", n_triggers=self._batch_frames,
            )
            return True
        except Exception as exc:
            self.status_var.set(f"Config failed: {type(exc).__name__}: {exc} (server={server})")
            return False

    def _create_temp_dir(self):
        """The temp folder setting (blank: the default), created if needed."""
        folder = pathlib.Path(self.temp_dir_var.get().strip() or DEFAULT_TEMP_DIR).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder)

    def _stop_measurement(self):
        try:
            server = self._server_url()
            serval_client.SESSION.get(f"{server}/measurement/stop", timeout=5)
        except Exception as exc:
            self.status_var.set(f"Stop failed: {type(exc).__name__}: {exc}")

    def _worker_loop(self, temp_dir: pathlib.Path, frame_rate: float, max_packets, test_mode: bool):
        frame_time = 1.0 / frame_rate if frame_rate > 0 else 0.1
        if test_mode:
            self._test_file_loop(frame_time, max_packets)
        else:
            self._live_loop(temp_dir, frame_time, max_packets)

    def _test_file_loop(self, frame_time, max_packets):
        """Re-process the test file once per frame period."""
        while not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                path = pathlib.Path(self.test_file_var.get().strip())
                if path.is_file() and path.stat().st_size > 0:
                    self._queue.put(self._process_file(path, max_packets))
            except Exception as exc:
                self._queue.put({"error": str(exc)})
            self._stop_event.wait(max(0.0, frame_time - (time.perf_counter() - started)))

    def _live_loop(self, temp_dir, frame_time, max_packets):
        """Show the newest finished frame Serval has written, as soon as it
        is finished. Serval keeps writing frames on its own (continuous
        mode), so if processing a frame takes longer than a frame period
        the frames that finished meanwhile are skipped (and counted) rather
        than queued -- the view never lags behind the detector."""
        server = self._server_url()
        test_path = self.test_file_var.get().strip()
        keep = pathlib.Path(test_path).resolve() if test_path else None
        last_shown = None
        last_new_file = time.perf_counter()
        last_idle_check = 0.0
        skipped = 0
        # Finished frames seen in the current batch: once the whole batch
        # is in, the next one is started as soon as Serval reports idle.
        batch_seen = 0
        # Fallback if frames stop arriving before a batch looks complete.
        idle_check = max(1.0, 3.0 * frame_time + 0.5)

        try:
            serval_client.start_measurement(server)
        except Exception as exc:
            self._queue.put({"error": f"Start failed: {type(exc).__name__}: {exc}"})

        while not self._stop_event.is_set():
            try:
                frames = _list_frames(temp_dir)
                index = _newest_finished(frames)
                if index is not None and frames[index][0] != last_shown:
                    path = frames[index][0]
                    # Finished frames since the last one shown never get shown.
                    names = [f[0] for f in frames]
                    start = 0
                    if last_shown is not None and last_shown in names:
                        start = names.index(last_shown) + 1
                    if last_shown is not None:
                        skipped += max(0, index - start)
                    batch_seen += index - start + 1
                    # Everything older than this frame is done with.
                    for old in names[:index]:
                        if keep is None or old.resolve() != keep:
                            try:
                                old.unlink()
                            except OSError:
                                pass
                    last_shown = path
                    last_new_file = time.perf_counter()
                    result = self._process_file(path, max_packets)
                    result["frames_skipped"] = skipped
                    self._queue.put(result)
                    continue

                now = time.perf_counter()
                batch_done = batch_seen >= self._batch_frames and now - last_idle_check > _BATCH_POLL_S
                if batch_done or now - last_new_file > idle_check:
                    last_idle_check = last_new_file = now
                    if self._restart_batch_if_idle(server, frame_time):
                        batch_seen = 0
                self._stop_event.wait(_FILE_POLL_S)
            except Exception as exc:
                self._queue.put({"error": str(exc)})
                self._stop_event.wait(0.5)

    def _restart_batch_if_idle(self, server, frame_time):
        """Serval auto-stops after each batch of frames; start the next.
        True if a new batch was started."""
        try:
            if serval_client.measurement_busy(server):
                return False
            serval_client.restart_continuous_batch(server, frame_time, self._batch_frames)
            return True
        except Exception as exc:
            if not self._stop_event.is_set():
                self._queue.put({"error": f"Batch restart failed: {type(exc).__name__}: {exc}"})
            return False

    def _process_file(self, path: pathlib.Path, max_packets):
        """max_packets: analyse only each frame's first max_packets packets
        (None = all of them)."""
        frame = raw_conversion.process_raw_file(
            path, self._plot_panel.hist_snapshot(), self._active_timewalk, max_packets=max_packets
        )
        return frame["plot_result"]

    def _accumulate(self, result):
        now = time.perf_counter()
        self._shown_times.append(now)
        while self._shown_times and now - self._shown_times[0] > 2.0:
            self._shown_times.popleft()
        self._frames_skipped = result.get("frames_skipped", self._frames_skipped)

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

        if result.get("warning"):
            self.status_var.set(f"Warning: {result['warning']}")
        elif self.status_var.get().startswith("Warning: "):
            self.status_var.set("Running. Pulse triggers are back.")

    def _update_stats(self, stats):
        def fmt(value):
            return f"{value:.3f}" if isinstance(value, float) else str(value)

        analysis_speed = (stats["real_time"] / stats["analysis_time"]) if stats["analysis_time"] > 0 else 0
        # Every other stat var is keyed by its own summarize_frame key.
        shown = self._shown_times
        display_fps = (len(shown) - 1) / (shown[-1] - shown[0]) if len(shown) > 1 and shown[-1] > shown[0] else 0.0
        derived = {
            "analysis_speed": analysis_speed,
            "analysis_time_last": stats.get("analysis_time_last", 0),
            "display_fps": display_fps,
            "frames_skipped": self._frames_skipped,
        }
        for key, var in self._stat_vars.items():
            var.set(fmt(derived[key] if key in derived else stats[key]))

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

        layout = self._mask_layout()
        if layout is None:
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
            "Proceed with masking?",
            parent=self,
        ):
            self.status_var.set("Mask canceled.")
            return

        try:
            server = self._server_url()
            masked = 0
            for x, y in hot_coords:
                local = layout.to_local(int(x), int(y))
                if local is None:
                    continue
                chip, lx, ly = local
                url = f"{server}/detector/chips/{chip}/mask/{ly}/{lx}"
                serval_client.SESSION.put(url, data="1", timeout=2)
                masked += 1
            self.status_var.set(f"Masked {masked} pixels (>{threshold}).")
        except Exception as exc:
            self.status_var.set(f"Mask failed: {type(exc).__name__}: {exc}")

    def _mask_layout(self):
        """The chip layout to map pixel-map bins back to Serval's per-chip
        pixels, or None (with a status message) if the pixel map isn't one
        bin per detector pixel."""
        layout = chip_layout.get_active_layout()
        size = layout.size
        cfg = self._accumulated.get("settings", self._plot_panel.hist_snapshot())["pixel"]
        if int(cfg["bins"]) != size or float(cfg["min"]) != 0.0 or float(cfg["max"]) != float(size):
            self.status_var.set(f"Masking needs the pixel map at {size} bins over 0-{size}.")
            return None
        return layout

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

        layout = self._mask_layout()
        if layout is None:
            return

        max_count = float(pixel_hist.max())
        if max_count <= 0:
            self.status_var.set("No pixel counts to mask.")
            return

        x, y = (int(v) for v in np.unravel_index(np.argmax(pixel_hist), pixel_hist.shape))

        local = layout.to_local(x, y)
        if local is None:
            self.status_var.set(f"Pixel ({x}, {y}) isn't on any chip.")
            return
        chip, lx, ly = local

        try:
            server = self._server_url()
            url = f"{server}/detector/chips/{chip}/mask/{ly}/{lx}"
            serval_client.SESSION.put(url, data="1", timeout=2)
            self.status_var.set(
                f"Masked hottest pixel ({x}, {y}) -- chip {chip} ({lx}, {ly}) -- with {int(max_count)} counts."
            )
        except Exception as exc:
            self.status_var.set(f"Mask failed: {type(exc).__name__}: {exc}")
