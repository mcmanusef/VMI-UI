import collections
import datetime
import os
import pathlib
import threading
import time
from queue import Queue, Empty

import qtk as tk
from qtk import ttk, messagebox, filedialog

import app_settings
import cv4_writer
import platform_paths
import raw_conversion
import serval_client
import shared_state
import tdc_mapping
import time_estimate
import tpx_processing
import ui_style
from cv4_writer import Cv4Writer
from plot_panel import HistogramPlotPanel
from cluster_timing import CLUSTER_TIMING_ATTR, ClusterTimingCalibration, load_correction
from timewalk import DEFAULT_CORRECTION_PATH


# Serval names this tab's raw files <prefix><time>...; only those are picked
# up, so stray files in the raw folder are left alone.
FILE_PREFIX = "acq"
# How often the raw folder is checked for newly finished frames.
_FILE_POLL_S = 0.05
# How often Serval is asked whether a batch that looks complete has ended.
_BATCH_POLL_S = 0.05
# Plot refresh period (ms): results that arrive in between are merged and
# drawn once (as on Diagnostics).
_REDRAW_MS = 30


def _created(st):
    """When a file was created, where the platform records it (Windows; some
    other systems via st_birthtime), else its mtime. Windows only updates a
    still-open file's listed mtime lazily, so by mtime the frame Serval is
    still writing can sort before the one it just closed."""
    birth = getattr(st, "st_birthtime", None)
    if birth is not None:
        return birth
    return st.st_ctime if os.name == "nt" else st.st_mtime


def _list_frames(raw_dir, exclude):
    """(path, size) of every raw frame of this tab in raw_dir whose name
    isn't in `exclude`, oldest first (Serval's own file counter breaks
    ties). Sizes come from the listing and can lag on Windows: read a
    finished file's real size from the file itself."""
    frames = []
    with os.scandir(raw_dir) as entries:
        for entry in entries:
            name = entry.name
            if name in exclude or not (name.startswith(FILE_PREFIX) and name.endswith(".tpx3")):
                continue
            try:
                st = entry.stat()
            except OSError:
                continue
            frames.append((pathlib.Path(entry.path), _created(st), st.st_size))
    frames.sort(key=lambda f: (f[1], f[0].name))
    return [(path, size) for path, _, size in frames]


class AcquisitionInterface(ttk.Frame):
    """Monitored acquisition: a new raw TPX3 file per frame, clustered and
    processed on the fly, appended into a cv4 (HDF5) file as it comes in.

    Combines the acquisition parameters/metadata (from the Collection tab)
    with the live plots (from the Diagnostics tab) in one screen: parameters
    in a sidebar, plots + plot options in the main section.
    """

    def __init__(self, parent, server_var=None, plot_shared_vars=None, acq_shared_vars=None, coordinator=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._server_var = server_var
        self._fallback_server_var = tk.StringVar(self, value="http://localhost:8080")
        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)
        self._plot_shared_vars = plot_shared_vars
        self._run_min_cluster_size = None
        self.min_cluster_size_var = shared_state.min_cluster_size_var(self, plot_shared_vars)
        self.cluster_timing_mode_var = shared_state.cluster_timing_mode_var(self, plot_shared_vars)

        # Shared live with the Collection parameters tab (see shared_state.py)
        # when acq_shared_vars is given; otherwise falls back to independent,
        # unshared variables.
        acq = acq_shared_vars or {}
        self.frame_time_var = acq.get("frame_time_var") or tk.StringVar(self, value="1")
        self.run_duration_var = acq.get("run_duration_var") or tk.StringVar(self, value="0")
        self.save_folder_var = acq.get("save_folder_var") or tk.StringVar(self, value=self._default_save_folder())
        # Loaded from whatever was last saved via the "Save current as
        # default" button (next to the tab bar); untouched until then.
        self.keep_raw_var = tk.BooleanVar(self, value=app_settings.get("acquisition.keep_raw", True))
        self.timewalk_enabled_var = tk.BooleanVar(self, value=app_settings.get("acquisition.timewalk_enabled", False))
        self.timewalk_path_var = tk.StringVar(
            self, value=app_settings.get("acquisition.timewalk_path", str(DEFAULT_CORRECTION_PATH))
        )

        self.target_var = acq.get("target_var") or tk.StringVar(self)
        self.target_pressure_var = acq.get("target_pressure_var") or tk.StringVar(self)
        self.background_pressure_var = acq.get("background_pressure_var") or tk.StringVar(self)
        self.power_var = acq.get("power_var") or tk.StringVar(self)
        self.spot_size_var = acq.get("spot_size_var") or tk.StringVar(self)
        self.polarization_var = acq.get("polarization_var") or tk.StringVar(self)
        self.wavelength_var = acq.get("wavelength_var") or tk.StringVar(self)

        self.status_var = tk.StringVar(self, value="Idle")
        self._progress_var = tk.DoubleVar(self, value=0.0)
        self.eta_var = tk.StringVar(self, value="--")

        self.collected_var = tk.StringVar(self, value="--")
        self.frames_var = tk.StringVar(self, value="--")
        self.pending_var = tk.StringVar(self, value="--")
        self.analysis_speed_var = tk.StringVar(self, value="--")
        self.dead_time_var = tk.StringVar(self, value="--")
        self.processing_fps_var = tk.StringVar(self, value="--")
        self.rep_rate_var = tk.StringVar(self, value="--")
        self.pulses_var = tk.StringVar(self, value="--")
        self.clusters_var = tk.StringVar(self, value="--")
        self.etof_var = tk.StringVar(self, value="--")
        self.itof_var = tk.StringVar(self, value="--")
        self.cv4_path_var = tk.StringVar(self, value="--")

        self._stop_event = threading.Event()
        # Graceful stop: don't trigger any more frames, but let whatever's
        # already queued finish processing -- unlike _stop_event, which the
        # processor also honors immediately and can abandon a backlog
        # mid-drain. See finish_and_stop().
        self._finish_event = threading.Event()
        self._collection_done = threading.Event()
        self._completed_naturally = False
        self._collected_frame_count = 0
        self._queue: Queue = Queue()
        self._pending_queue: Queue = Queue()
        self._collector = None
        self._processor = None
        self._active_timewalk = None

        self._accumulated = None
        self._reset_accumulation = False
        self._target_frames = 0
        self._batch_frames = 1
        # Raw files already in the raw folder at Start (an earlier run's):
        # never treated as this run's frames.
        self._stale_names = frozenset()
        # When each recent frame was processed (for the processing rate).
        self._processed_times = collections.deque()

        self._build_ui()
        self._poll_queue()

    def _default_save_folder(self):
        return platform_paths.dated_data_folder("acquisition")

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.rowconfigure(0, weight=1)

        self._build_sidebar(sidebar)

        self._plot_panel = HistogramPlotPanel(
            main,
            on_settings_changed=self._on_panel_settings_changed,
            plots_first=True,
            shared_vars=self._plot_shared_vars,
            etof_correlation=True,
        )
        self._plot_panel.grid(row=0, column=0, sticky="nsew")

    def _build_sidebar(self, sidebar):
        # A trailing empty row soaks up any leftover vertical space (e.g.
        # once collapsed sections have shrunk) so the real content stays
        # packed at the top and pulls upward as sections collapse, rather
        # than leftover space landing inside one of the rows above.
        sidebar.rowconfigure(8, weight=1)

        # "Force Stop" aborts in-flight frames; the normal Stop lets them
        # finish processing cleanly -- hence the danger styling/separation.
        ui_style.build_button_bar(
            sidebar, 0, [("Start", self.start), ("Stop", self.finish_and_stop)], danger=("Force Stop", self.stop),
        )
        self._status_block = ui_style.StatusBlock(
            sidebar, 1, self.status_var, progress_var=self._progress_var, eta_var=self.eta_var
        )

        params = ui_style.build_section(sidebar, 2, "Acquisition parameters")
        ui_style.add_form_row(params, 0, "Frame time (s):", ttk.Entry(params, textvariable=self.frame_time_var, width=18))
        ui_style.add_form_row(
            params, 1, "Run duration (s):", ttk.Entry(params, textvariable=self.run_duration_var, width=18)
        )
        ui_style.add_form_row(
            params, 2, "Save folder:", ui_style.build_path_field(params, self.save_folder_var, self._browse_folder)
        )
        ttk.Checkbutton(
            params, text="Keep raw TPX3 files", variable=self.keep_raw_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            params, text="Apply timewalk correction before clustering", variable=self.timewalk_enabled_var
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ui_style.build_path_field(params, self.timewalk_path_var, self._browse_timewalk).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(2, 4)
        )
        ui_style.build_cluster_timing_mode_row(params, 6, self.cluster_timing_mode_var)
        ui_style.add_form_row(params, 8, "Min cluster size (hits):", ttk.Entry(params, textvariable=self.min_cluster_size_var, width=18))

        self.notes = ui_style.build_metadata_section(sidebar, 3, self)

        live = ui_style.build_section(sidebar, 4, "Live totals")
        cv4_row = ui_style.add_stat_rows(live, [
            ("Frames collected:", self.collected_var),
            ("Frames processed:", self.frames_var),
            ("Pending (backlog):", self.pending_var),
            ("Processing rate (fps):", self.processing_fps_var),
            ("Dead time:", self.dead_time_var),
            ("Analysis speed:", self.analysis_speed_var),
            ("Laser repetition rate:", self.rep_rate_var),
            ("Pulses:", self.pulses_var),
            ("Clusters:", self.clusters_var),
            ("e-ToF events:", self.etof_var),
            ("i-ToF events:", self.itof_var),
        ])
        ttk.Label(live, text="cv4 file:", font=ui_style.SMALL_FONT).grid(row=cv4_row, column=0, sticky="nw", pady=(6, 0))
        ui_style.add_note(live, cv4_row + 1, textvariable=self.cv4_path_var, pady=0)

        # The same run-wide rates Diagnostics shows, over everything
        # processed so far.
        self._stat_vars = {}
        stat_groups = [
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
            body = ui_style.build_section(sidebar, 5 + idx, title, pady=0 if last else (0, 8))
            rows = []
            for label, key in items:
                var = tk.StringVar(self, value="--")
                self._stat_vars[key] = var
                rows.append((label, var))
            ui_style.add_stat_rows(body, rows)

    def _browse_folder(self):
        chosen = filedialog.askdirectory(
            initialdir=self.save_folder_var.get().strip() or ".",
            title="Choose save folder",
            parent=self,
        )
        if chosen:
            self.save_folder_var.set(chosen)

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

    # ---- plot panel wiring --------------------------------------------------

    def _on_panel_settings_changed(self):
        self._reset_accumulation = True
        self._plot_panel.update_plots(self._current_data())

    def _current_data(self):
        return self._accumulated if self._accumulated is not None else self._plot_panel.empty_plot_data()

    # ---- server helpers -----------------------------------------------------

    def _server_url(self):
        url = ""
        if self._server_var is not None:
            url = self._server_var.get().strip()
        if not url:
            url = self._fallback_server_var.get().strip() or "http://localhost:8080"
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        return url

    # ---- start / stop ---------------------------------------------------

    def is_running(self):
        return bool((self._collector and self._collector.is_alive()) or (self._processor and self._processor.is_alive()))

    def default_fields(self):
        """Fields the global "Save current as default" button can persist
        (the frame time / run duration / save folder / metadata fields are
        already live-shared and auto-persisted via acq_shared_vars)."""
        return {
            "acquisition.keep_raw": self.keep_raw_var,
            "acquisition.timewalk_enabled": self.timewalk_enabled_var,
            "acquisition.timewalk_path": self.timewalk_path_var,
        }

    def start(self):
        if self._coordinator is not None:
            self._coordinator.stop_others(self)

        if self.is_running():
            self.stop()

        try:
            frame_time = float(self.frame_time_var.get())
            if frame_time <= 0:
                raise ValueError("Frame time must be positive.")
        except ValueError as exc:
            self.status_var.set(f"Invalid frame time: {exc}")
            return

        try:
            run_duration = float(self.run_duration_var.get() or 0)
        except ValueError:
            self.status_var.set("Invalid run duration.")
            return

        if self.timewalk_enabled_var.get():
            try:
                self._active_timewalk = load_correction(self.timewalk_path_var.get().strip(), mode=self.cluster_timing_mode_var.get())
            except Exception as exc:
                self.status_var.set(f"Could not load timewalk correction: {exc}")
                return
        else:
            self._active_timewalk = None

        raw_dir, cv4_path = self._prepare_save_folder()
        if raw_dir is None:
            self.status_var.set("Start canceled.")
            return

        # Go by frame count, not wall time: dead time around each
        # start/stop (server round-trips, polling, etc.) means a wall-clock
        # cutoff would collect less than `run_duration` seconds of actual
        # data. Collecting a fixed number of frames instead guarantees
        # `target_frames * frame_time` == `run_duration` of real exposure,
        # however much wall time that ends up taking.
        target_frames = int(round(run_duration / frame_time)) if run_duration > 0 else 0
        self._target_frames = target_frames

        metadata = self._collect_metadata(frame_time, run_duration)
        metadata["Target Frames"] = target_frames
        if self._active_timewalk is not None:
            metadata["Timewalk Correction File"] = self.timewalk_path_var.get().strip()
            if isinstance(self._active_timewalk, ClusterTimingCalibration):
                metadata[CLUSTER_TIMING_ATTR] = self._active_timewalk.describe()
        # Fixed for the whole run, so every frame matches what's recorded.
        self._run_min_cluster_size = tpx_processing.get_min_cluster_size()
        metadata[tpx_processing.MIN_CLUSTER_SIZE_ATTR] = self._run_min_cluster_size
        metadata[tdc_mapping.PIXEL_OFFSET_ATTR] = tdc_mapping.get_active_mapping().pixel_offset_ns

        try:
            self._stale_names = frozenset(path.name for path, _ in _list_frames(raw_dir, ()))
        except OSError as exc:
            self.status_var.set(f"Could not read {raw_dir}: {exc}")
            return

        server = self._server_url()
        try:
            # Continuous mode, as on Diagnostics: one start makes Serval
            # write a new raw file every frame on its own, in bounded
            # batches (see _collector_loop), instead of a start/re-arm
            # round trip -- and its dead time -- per frame.
            self._batch_frames = serval_client.max_frames_per_batch(frame_time)
            first_batch = min(self._batch_frames, target_frames) if target_frames > 0 else self._batch_frames
            serval_client.configure_continuous_destination(
                server, frame_time, raw_dir, f"{FILE_PREFIX}%Hms_", n_triggers=first_batch,
            )
        except Exception as exc:
            self.status_var.set(f"Config failed: {type(exc).__name__}: {exc} (server={server})")
            return

        self._reset_accumulation = True
        self._accumulated = None
        self._stop_event.clear()
        self._finish_event.clear()
        self._collection_done.clear()
        self._pending_queue = Queue()
        self.cv4_path_var.set(str(cv4_path))
        self.collected_var.set("0")
        self.frames_var.set("0")
        self.pending_var.set("0")
        self.dead_time_var.set("--")
        self.analysis_speed_var.set("--")
        self.processing_fps_var.set("--")
        self.rep_rate_var.set("--")
        for var in self._stat_vars.values():
            var.set("--")
        self._processed_times.clear()
        self._progress_var.set(0.0)
        self.eta_var.set("Estimating..." if target_frames > 0 else "--")
        if target_frames > 0:
            self.status_var.set(f"Collecting {target_frames} frame(s). Writing {cv4_path.name}")
        else:
            self.status_var.set(f"Collecting. Writing {cv4_path.name}")

        # Collection and processing run on separate threads: collection keeps
        # triggering/writing new raw files as fast as the detector allows
        # (so it isn't slowed by clustering), and processing drains a queue
        # of raw files -- continuing after collection ends, until the
        # backlog is empty or Stop is pressed.
        self._collector = threading.Thread(
            target=self._collector_loop,
            args=(raw_dir, frame_time, target_frames),
            daemon=True,
        )
        self._processor = threading.Thread(
            target=self._processor_loop,
            args=(self.keep_raw_var.get(), cv4_path, metadata),
            daemon=True,
        )
        self._collector.start()
        self._processor.start()

    def stop(self):
        self._stop_event.set()
        self.eta_var.set("--")
        try:
            serval_client.stop_measurement(self._server_url())
        except Exception:
            pass
        self.status_var.set("Stopping...")
        if self._collector:
            self._collector.join(timeout=5)
        if self._processor:
            self._processor.join(timeout=5)
        leftover = self._pending_queue.qsize()
        if leftover:
            self.status_var.set(f"Stopped. {leftover} collected frame(s) were not processed.")
        else:
            self.status_var.set("Stopped.")

    def finish_and_stop(self):
        """Stop triggering new frames, but let whatever's already queued
        finish processing -- ends with every triggered raw file either fully
        processed or, if it was never triggered, not existing at all,
        instead of Stop's immediate cutoff which can abandon a mid-drain
        backlog. Doesn't block: the cv4 gets renamed off .partial once the
        processor actually drains and reports "done" (only a Force Stop
        that abandons unprocessed frames leaves it as .partial.cv4)."""
        if not self.is_running():
            return
        self._finish_event.set()
        self.status_var.set("Finishing queued frames, then stopping...")

    def _prepare_save_folder(self):
        folder_text = self.save_folder_var.get().strip()
        if not folder_text:
            self.status_var.set("Save folder is required.")
            return None, None
        folder = pathlib.Path(folder_text)
        folder.mkdir(parents=True, exist_ok=True)
        raw_dir = folder / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        cv4_path = folder / f"{folder.name}.cv4"
        if cv4_path.exists():
            if not messagebox.askyesno(
                "File exists",
                f"{cv4_path} already exists.\nOverwrite it?",
                parent=self,
            ):
                return None, None
            cv4_path.unlink()
        return raw_dir, cv4_path

    def _collect_metadata(self, frame_time, run_duration):
        return {
            "Target": self.target_var.get().strip(),
            "Target Pressure": self.target_pressure_var.get().strip(),
            "Background Pressure": self.background_pressure_var.get().strip(),
            "Power": self.power_var.get().strip(),
            "Spot Size": self.spot_size_var.get().strip(),
            "Polarization": self.polarization_var.get().strip(),
            "Wavelength": self.wavelength_var.get().strip(),
            "Notes": self.notes.get("1.0", "end").strip(),
            "Frame Time (s)": frame_time,
            "Run Duration (s)": run_duration,
            "Save Folder": self.save_folder_var.get().strip(),
            "Start Time (UTC)": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    # ---- collector thread: trigger + save raw files, as fast as the ------
    # ---- detector allows -- never blocked on clustering -------------------

    def _collector_loop(self, raw_dir, frame_time, target_frames):
        """Hand every frame Serval finishes writing to the processor, until
        `target_frames` have been collected (target_frames <= 0 means
        "until stopped"). Going by frame count rather than wall time means
        `target_frames * frame_time` is always the real exposure collected,
        however much wall time it ends up taking.

        Continuous mode, as on Diagnostics: one start_measurement() makes
        Serval write a new raw file every frame on its own, in batches of at
        most serval_client.max_frames_per_batch frames (the last batch of a
        bounded run is cut to exactly the frames still wanted, so Serval
        stops on the target frame itself). A frame is finished once Serval
        has opened a newer file after it, or -- the last frame of a batch --
        once Serval has gone idle; the next batch is started then. Unlike
        Diagnostics, no frame is ever skipped: every finished frame is
        queued, and this loop never waits on the processor.

        Stops triggering on either _stop_event (hard) or _finish_event
        (graceful: frames Serval already wrote are still queued, and the
        processor keeps draining the backlog, see _processor_loop). An
        unbounded run (target_frames <= 0) has no target to fall short of,
        so it's always "complete" whenever it ends; a bounded run is only
        complete if it actually reached target_frames under its own steam.
        """
        server = self._server_url()
        collection_start = time.time()
        seen = set(self._stale_names)
        collected = 0
        self._completed_naturally = target_frames <= 0

        def remaining():
            return target_frames - collected if target_frames > 0 else None

        def batch_size():
            left = remaining()
            return self._batch_frames if left is None else min(self._batch_frames, left)

        def take(frames):
            """Queue finished frames (oldest first) and mark them seen."""
            nonlocal collected
            for path, _ in frames:
                seen.add(path.name)
                left = remaining()
                if left is not None and left <= 0:
                    continue
                # From the file itself: on Windows a directory listing's
                # size (and os.stat's, which can read the same entry) can
                # still be 0 for a file Serval has only just closed.
                try:
                    with open(path, "rb") as handle:
                        size = handle.seek(0, os.SEEK_END)
                except OSError:
                    size = 0
                if size <= 0:
                    self._queue.put({"error": f"{path.name} is empty; skipped."})
                    continue
                collected += 1
                self._collected_frame_count = collected
                self._pending_queue.put(path)
                elapsed_wall = time.time() - collection_start
                dead_time_pct = 0.0
                if elapsed_wall > 0:
                    dead_time_pct = max(0.0, (elapsed_wall - collected * frame_time) / elapsed_wall * 100.0)
                self._queue.put({
                    "collected_index": collected,
                    "pending": self._pending_queue.qsize(),
                    "dead_time_pct": dead_time_pct,
                    "elapsed_wall": elapsed_wall,
                })

        # Frames this batch was started for, and how many of them have
        # turned up so far.
        batch_target = batch_size()
        batch_seen = 0
        last_new_file = time.perf_counter()
        last_idle_check = 0.0
        # Fallback if frames stop arriving before a batch looks complete.
        idle_check = max(1.0, 3.0 * frame_time + 0.5)

        try:
            try:
                serval_client.start_measurement(server)
            except Exception as exc:
                self._queue.put({"error": f"Start failed: {type(exc).__name__}: {exc}"})

            while not self._stop_event.is_set() and not self._finish_event.is_set():
                if target_frames > 0 and collected >= target_frames:
                    self._completed_naturally = True
                    break
                try:
                    frames = _list_frames(raw_dir, seen)
                    # Every frame with a newer file after it is finished.
                    if len(frames) > 1:
                        take(frames[:-1])
                        batch_seen += len(frames) - 1
                        last_new_file = time.perf_counter()
                        continue

                    now = time.perf_counter()
                    # The newest file is the batch's last frame: done once
                    # Serval goes idle.
                    batch_done = batch_seen + len(frames) >= batch_target and now - last_idle_check > _BATCH_POLL_S
                    if batch_done or now - last_new_file > idle_check:
                        last_idle_check = last_new_file = now
                        if not serval_client.measurement_busy(server):
                            take(_list_frames(raw_dir, seen))
                            if target_frames > 0 and collected >= target_frames:
                                continue
                            if self._stop_event.is_set() or self._finish_event.is_set():
                                break
                            batch_target = batch_size()
                            batch_seen = 0
                            serval_client.restart_continuous_batch(server, frame_time, batch_target)
                    self._stop_event.wait(_FILE_POLL_S)
                except Exception as exc:
                    if not self._stop_event.is_set():
                        self._queue.put({"error": f"{type(exc).__name__}: {exc}"})
                    self._stop_event.wait(0.5)
        except Exception as exc:
            self._queue.put({"error": str(exc)})
        finally:
            try:
                serval_client.stop_measurement(server)
                # Doesn't confirm Serval actually wound down -- and the
                # *next* run's destination reconfigure can otherwise race
                # against a still-stopping measurement.
                serval_client.wait_for_measurement_idle(server)
            except Exception:
                pass
            if not self._stop_event.is_set():
                # Finish & Stop (or the target reached): whatever Serval
                # wrote before stopping is real data, so it's kept.
                try:
                    take(_list_frames(raw_dir, seen))
                except Exception as exc:
                    self._queue.put({"error": str(exc)})
            self._collection_done.set()
            self._queue.put({"collection_done": True, "collected_index": collected})

    # ---- processor thread: cluster + write cv4, draining the backlog -----
    # ---- until it's empty (collection finished) or Stop is pressed -------

    def _processor_loop(self, keep_raw, cv4_path, metadata):
        partial_path = cv4_writer.partial_path_for(cv4_path)
        try:
            writer = Cv4Writer(partial_path, metadata=metadata)
        except Exception as exc:
            self._queue.put({"error": f"Could not open {partial_path}: {exc}"})
            return

        frame_index = 0
        # Only the hard _stop_event aborts this early. Finish & Stop
        # (_finish_event) deliberately isn't checked here -- it only tells
        # the collector to stop triggering *new* frames; this loop keeps
        # draining whatever's already queued.
        try:
            while not self._stop_event.is_set():
                try:
                    path = self._pending_queue.get(timeout=0.2)
                except Empty:
                    if self._collection_done.is_set() and self._pending_queue.empty():
                        break
                    continue

                try:
                    frame = self._process_file(path)
                    writer.append_events(frame["events"])
                    writer.flush()
                    frame_index += 1

                    if not keep_raw:
                        try:
                            path.unlink()
                        except OSError:
                            pass

                    self._queue.put(
                        {
                            "frame_index": frame_index,
                            "pending": self._pending_queue.qsize(),
                            **frame["plot_result"],
                        }
                    )
                except Exception as exc:
                    self._queue.put({"error": f"Processing {path.name} failed: {exc}"})
                finally:
                    self._pending_queue.task_done()
        except Exception as exc:
            self._queue.put({"error": str(exc)})
        finally:
            # On a Force Stop this loop can exit while the collector is still
            # winding down; wait (briefly) so the collected count is final
            # before deciding whether every collected frame got processed.
            self._collection_done.wait(timeout=10)
            # .partial only if raw files that were actually collected were
            # left unprocessed (Force Stop mid-backlog). A graceful Stop, or
            # a run stopped before its target but with everything collected
            # processed, still gets the plain .cv4 name; "Complete" separately
            # records whether the run reached its target.
            all_processed = frame_index >= self._collected_frame_count
            writer.set_attrs({
                "Frames Collected": self._collected_frame_count,
                "Frames Processed": frame_index,
                "Complete": self._completed_naturally,
                "All Collected Frames Processed": all_processed,
            })
            writer.close()
            final_path = cv4_writer.finalize_partial_path(partial_path, all_processed)
            self._queue.put({
                "done": True,
                "frame_index": frame_index,
                "frames_collected": self._collected_frame_count,
                "final_path": str(final_path),
            })

    def _process_file(self, path: pathlib.Path):
        # Shared with Analysis > conversion, so offline conversion of these
        # same raw files produces an identical cv4.
        return raw_conversion.process_raw_file(
            path, self._plot_panel.hist_snapshot(), self._active_timewalk, self._run_min_cluster_size
        )

    # ---- main-thread queue handling ---------------------------------------

    def _poll_queue(self):
        """Apply every result that arrived since the last tick, then redraw
        once -- drawing costs the same however many frames went into it, so
        a backlog draining fast doesn't stall on redraws."""
        got_frame = False
        try:
            while True:
                result = self._queue.get_nowait()
                got_frame = self._apply_result(result) or got_frame
        except Empty:
            pass
        if got_frame:
            self._redraw()
        self.after(_REDRAW_MS, self._poll_queue)

    def _apply_result(self, result):
        if "error" in result:
            self.status_var.set(f"Error: {result['error']}")
            return

        if result.get("collection_done"):
            self.collected_var.set(str(result.get("collected_index", "--")))
            self.eta_var.set("--")
            if not self._stop_event.is_set():
                pending = self._pending_queue.qsize()
                if pending:
                    self.status_var.set(
                        f"Collection finished ({result.get('collected_index', 0)} frame(s)). "
                        f"Processing {pending} remaining frame(s) in the background..."
                    )
                else:
                    self.status_var.set(f"Collection finished ({result.get('collected_index', 0)} frame(s)).")
            return

        if result.get("done"):
            processed = result.get("frame_index", 0)
            collected = result.get("frames_collected", processed)
            if "final_path" in result:
                self.cv4_path_var.set(result["final_path"])
            self.status_var.set(f"Finished. {processed}/{collected} frame(s) processed.")
            return

        if "collected_index" in result:
            collected = result["collected_index"]
            self.collected_var.set(str(collected))
            self.pending_var.set(str(result.get("pending", "--")))
            if "dead_time_pct" in result:
                self.dead_time_var.set(f"{result['dead_time_pct']:.1f}%")
            if self._target_frames > 0:
                frac = min(1.0, collected / self._target_frames)
                self._progress_var.set(frac * 100.0)
                # Self-corrects for dead time: based on the actual observed
                # collection rate so far (elapsed_wall / collected), not a
                # naive frame_time * frames_remaining that ignores it.
                elapsed_wall = result.get("elapsed_wall")
                if elapsed_wall is not None and frac > 0:
                    self.eta_var.set(time_estimate.format_eta(elapsed_wall / frac - elapsed_wall))
            return

        self.frames_var.set(str(result.get("frame_index", "--")))
        if result.get("warning"):
            self.status_var.set(f"Warning: frame {result.get('frame_index', '--')}: {result['warning']}")
        elif self.status_var.get().startswith("Warning: frame "):
            self.status_var.set("Collecting. Pulse triggers are back.")
        if "pending" in result:
            self.pending_var.set(str(result["pending"]))

        if self._reset_accumulation:
            self._accumulated = None
            self._reset_accumulation = False
        self._accumulated = raw_conversion.accumulate_plot_result(self._accumulated, result)

        now = time.perf_counter()
        self._processed_times.append(now)
        while self._processed_times and now - self._processed_times[0] > 2.0:
            self._processed_times.popleft()
        return True

    def _redraw(self):
        self._plot_panel.update_plots(self._accumulated)

        stats = self._accumulated["stats"]
        self.pulses_var.set(str(stats.get("total_pulses", "--")))
        self.clusters_var.set(str(stats.get("total_clusters", "--")))
        self.etof_var.set(str(stats.get("total_etof", "--")))
        self.itof_var.set(str(stats.get("total_itof", "--")))

        analysis_time = stats.get("analysis_time", 0)
        real_time = stats.get("real_time", 0)
        if analysis_time > 0:
            self.analysis_speed_var.set(f"{real_time / analysis_time:.3f}")
        else:
            self.analysis_speed_var.set("--")
        rep_rate = stats.get("pulses_per_second", 0)
        self.rep_rate_var.set(f"{rep_rate:.3f}" if rep_rate else "--")

        shown = self._processed_times
        if len(shown) > 1 and shown[-1] > shown[0]:
            self.processing_fps_var.set(f"{(len(shown) - 1) / (shown[-1] - shown[0]):.2f}")

        for key, var in self._stat_vars.items():
            value = stats.get(key)
            var.set(f"{value:.3f}" if isinstance(value, float) else str(value if value is not None else "--"))
