import datetime
import pathlib
import threading
import time
from queue import Queue, Empty

import qtk as tk
from qtk import ttk, messagebox, filedialog

import app_settings
import cv4_writer
import serval_client
import shared_state
import time_estimate
from collapsible_frame import CollapsibleFrame
from cv4_writer import Cv4Writer
from plot_panel import HistogramPlotPanel
from scrollable_frame import ScrollableFrame
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

        self._build_ui()
        self._poll_queue()

    def _default_save_folder(self):
        return rf"C:\DATA\{datetime.date.today().strftime('%Y%m%d')}\acquisition"

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Collapsing this hands the whole sidebar's width back to the
        # plots -- Qt excludes hidden widgets from layout sizing, so the
        # column shrinks to just the "Controls" toggle.
        sidebar_section = CollapsibleFrame(self, text="Controls")
        sidebar_section.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        sidebar_section.body.columnconfigure(0, weight=1)
        sidebar_section.body.rowconfigure(0, weight=1)

        # Wide enough for the sidebar's widest row (the button bar) plus the
        # vertical scrollbar that appears once every section is expanded --
        # the default 300 clips content with no way to reach it, since the
        # horizontal scrollbar is intentionally off.
        sidebar = ScrollableFrame(sidebar_section.body, width=400)
        sidebar.grid(row=0, column=0, sticky="nsew")

        main = ttk.Frame(self)
        main.grid(row=0, column=1, sticky="nsew", pady=10, padx=(0, 10))
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self._build_sidebar(sidebar.body)

        self._plot_panel = HistogramPlotPanel(
            main,
            on_settings_changed=self._on_panel_settings_changed,
            plots_first=True,
            shared_vars=self._plot_shared_vars,
        )
        self._plot_panel.grid(row=0, column=0, sticky="nsew")

    def _build_sidebar(self, sidebar):
        # A trailing empty row soaks up any leftover vertical space (e.g.
        # once collapsed sections have shrunk) so the real content stays
        # packed at the top and pulls upward as sections collapse, rather
        # than leftover space landing inside one of the rows above.
        sidebar.rowconfigure(5, weight=1)

        buttons = ttk.Frame(sidebar)
        buttons.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(buttons, text="Start", command=self.start).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Stop", command=self.finish_and_stop).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(buttons, text="Force Stop", command=self.stop).grid(row=0, column=2)

        status = ttk.Frame(sidebar)
        status.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(status, textvariable=self.status_var, wraplength=220, justify="left").grid(
            row=0, column=0, sticky="w"
        )
        self._progress = ttk.Progressbar(
            status, variable=self._progress_var, maximum=100.0, mode="determinate", length=220
        )
        self._progress.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(status, textvariable=self.eta_var, wraplength=220, justify="left", font=("Segoe UI", 8)).grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )

        params_section = CollapsibleFrame(sidebar, text="Acquisition parameters")
        params_section.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        params = params_section.body
        params.columnconfigure(1, weight=1)

        self._add_row(params, 0, "Frame time (s):", ttk.Entry(params, textvariable=self.frame_time_var, width=18))
        self._add_row(
            params, 1, "Run duration (s):", ttk.Entry(params, textvariable=self.run_duration_var, width=18)
        )
        ttk.Label(
            params,
            text="(0 = run until stopped; otherwise collects\nround(duration / frame time) frames, "
                 "not just\nuntil that much wall time has passed)",
            font=("Segoe UI", 8),
            justify="left",
        ).grid(row=2, column=1, sticky="w", pady=(0, 4))

        folder_row = ttk.Frame(params)
        folder_row.grid(row=3, column=1, sticky="ew", pady=4)
        folder_row.columnconfigure(0, weight=1)
        ttk.Entry(folder_row, textvariable=self.save_folder_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(folder_row, text="...", width=3, command=self._browse_folder).grid(row=0, column=1, padx=(4, 0))
        ttk.Label(params, text="Save folder:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)

        ttk.Checkbutton(
            params, text="Keep raw TPX3 files", variable=self.keep_raw_var
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(
            params,
            text="Unchecked: each raw file is deleted once it's\nclustered and appended to the .cv4 file.",
            font=("Segoe UI", 8),
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 4))

        ttk.Checkbutton(
            params, text="Apply timewalk correction before clustering", variable=self.timewalk_enabled_var
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))
        timewalk_row = ttk.Frame(params)
        timewalk_row.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(2, 4))
        timewalk_row.columnconfigure(0, weight=1)
        ttk.Entry(timewalk_row, textvariable=self.timewalk_path_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(timewalk_row, text="...", width=3, command=self._browse_timewalk).grid(
            row=0, column=1, padx=(4, 0)
        )

        meta_section = CollapsibleFrame(sidebar, text="Metadata")
        meta_section.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        meta = meta_section.body
        meta.columnconfigure(1, weight=1)

        self._add_row(meta, 0, "Target:", ttk.Entry(meta, textvariable=self.target_var))
        self._add_row(meta, 1, "Target Pressure:", ttk.Entry(meta, textvariable=self.target_pressure_var))
        self._add_row(meta, 2, "Background Pressure:", ttk.Entry(meta, textvariable=self.background_pressure_var))
        self._add_row(meta, 3, "Power:", ttk.Entry(meta, textvariable=self.power_var))
        self._add_row(meta, 4, "Spot Size:", ttk.Entry(meta, textvariable=self.spot_size_var))
        self._add_row(meta, 5, "Polarization:", ttk.Entry(meta, textvariable=self.polarization_var))
        self._add_row(meta, 6, "Wavelength:", ttk.Entry(meta, textvariable=self.wavelength_var))

        ttk.Label(meta, text="Notes:").grid(row=7, column=0, sticky="nw", padx=(0, 8), pady=4)
        self.notes = tk.Text(meta, wrap="word", height=4, width=24)
        self.notes.grid(row=7, column=1, sticky="nsew", pady=4)
        shared_state.wire_notes_widget(self.notes)

        live_section = CollapsibleFrame(sidebar, text="Live totals")
        live_section.grid(row=4, column=0, sticky="ew")
        live = live_section.body
        live.columnconfigure(1, weight=1)
        stat_rows = [
            ("Frames collected:", self.collected_var),
            ("Frames processed:", self.frames_var),
            ("Pending (backlog):", self.pending_var),
            ("Dead time:", self.dead_time_var),
            ("Analysis speed:", self.analysis_speed_var),
            ("Pulses:", self.pulses_var),
            ("Clusters:", self.clusters_var),
            ("e-ToF events:", self.etof_var),
            ("i-ToF events:", self.itof_var),
        ]
        row = 0
        for row, (label, var) in enumerate(stat_rows):
            ttk.Label(live, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=1)
            ttk.Label(live, textvariable=var).grid(row=row, column=1, sticky="w", pady=1)
        cv4_row = row + 1
        ttk.Label(live, text="cv4 file:", font=("Segoe UI", 8)).grid(row=cv4_row, column=0, sticky="nw", pady=(6, 0))
        ttk.Label(
            live, textvariable=self.cv4_path_var, font=("Segoe UI", 8), wraplength=220, justify="left"
        ).grid(row=cv4_row + 1, column=0, columnspan=2, sticky="w")

    def _add_row(self, parent, row, label, widget):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        widget.grid(row=row, column=1, sticky="ew", pady=4)

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
            filetypes=[("JSON", "*.json")],
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
                self._active_timewalk = TimewalkCorrection.load(self.timewalk_path_var.get().strip())
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

        try:
            server = self._server_url()
            # One frame per trigger, restarted every frame (see
            # _collector_loop) -- back to this tab's original design rather
            # than the continuous, many-frames-per-command pattern (still
            # available in serval_client.configure_continuous_destination
            # for future use).
            serval_client.configure_single_trigger_destination(server, frame_time, raw_dir, "acq%Hms_")
        except Exception as exc:
            self.status_var.set(f"Config failed: {exc}")
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
        backlog. Doesn't block: the cv4 gets finalized (.partial or not)
        once the processor actually drains and reports "done"."""
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
        """Keeps triggering/collecting new raw files until `target_frames`
        have been collected (target_frames <= 0 means "until stopped").
        Going by frame count rather than wall time means dead time around
        each start/stop cycle (server round-trips, polling, detector
        readout) doesn't eat into how much data actually gets collected --
        `target_frames * frame_time` is always the real exposure collected,
        however much wall time it ends up taking.

        This loop is never blocked waiting on the processor: each new file
        is handed off to the pending queue and this loop immediately moves
        on to trigger the next frame, so a slow analysis never slows down
        collection.

        One frame per trigger: the destination was configured for
        nTriggers=1 (configure_single_trigger_destination, called by
        start()), so a fresh start_measurement() is issued every single
        frame below -- this tab's original design. The continuous,
        many-frames-per-command pattern (configure_continuous_destination,
        serval_client.max_frames_per_batch, still used by Collection
        parameters and Quick Monitor) remains available in serval_client.py
        for future use; this tab just isn't using it right now.

        Stops triggering on either _stop_event (hard) or _finish_event
        (graceful -- the processor keeps draining the backlog regardless,
        see _processor_loop). An unbounded run (target_frames <= 0) has no
        target to fall short of, so it's always "complete" whenever it ends;
        a bounded run is only complete if it actually reached target_frames
        under its own steam, not because it was stopped early.
        """
        server = self._server_url()
        last_path = None
        last_mtime = None
        collection_start = time.time()
        frame_index = 0
        self._completed_naturally = target_frames <= 0

        # wait_for_new_file's default timeout (5s) is shorter than many
        # exposures; scale it to frame_time so we don't give up on (and
        # re-trigger over) an exposure that simply hasn't finished yet --
        # that was the source of "frame time isn't applied" for anything
        # much longer than ~5s.
        wait_timeout = max(5.0, frame_time * 5.0 + 2.0)

        try:
            while not self._stop_event.is_set() and not self._finish_event.is_set():
                if target_frames > 0 and frame_index >= target_frames:
                    self._completed_naturally = True
                    break

                try:
                    serval_client.start_measurement(server)
                except Exception as exc:
                    self._queue.put({"error": f"Start failed: {exc}"})
                    time.sleep(1)
                    continue

                newest = serval_client.wait_for_new_file(
                    raw_dir, self._stop_event, last_path, last_mtime, timeout=wait_timeout
                )
                if newest is None:
                    continue

                newest_stat = newest.stat()
                if newest_stat.st_size == 0:
                    time.sleep(0.2)
                    continue

                last_path = newest
                last_mtime = newest_stat.st_mtime
                frame_index += 1
                self._collected_frame_count = frame_index

                elapsed_wall = time.time() - collection_start
                measured_time = frame_index * frame_time
                dead_time_pct = 0.0
                if elapsed_wall > 0:
                    dead_time_pct = max(0.0, (elapsed_wall - measured_time) / elapsed_wall * 100.0)

                self._pending_queue.put(newest)
                self._queue.put(
                    {
                        "collected_index": frame_index,
                        "pending": self._pending_queue.qsize(),
                        "dead_time_pct": dead_time_pct,
                        "elapsed_wall": elapsed_wall,
                    }
                )
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
            self._collection_done.set()
            self._queue.put({"collection_done": True, "collected_index": frame_index})

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
                    writer.append_frame(
                        frame["pulses_sorted"],
                        frame["clusters_by_pulse"],
                        frame["etof_by_pulse"],
                        frame["itof_by_pulse"],
                    )
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
            writer.set_attrs({
                "Frames Collected": self._collected_frame_count,
                "Frames Processed": frame_index,
                "Complete": self._completed_naturally,
            })
            writer.close()
            final_path = cv4_writer.finalize_partial_path(partial_path, self._completed_naturally)
            self._queue.put({
                "done": True,
                "frame_index": frame_index,
                "frames_collected": self._collected_frame_count,
                "final_path": str(final_path),
            })

    def _process_file(self, path: pathlib.Path):
        start = time.perf_counter()
        settings = self._plot_panel.hist_snapshot()

        # No max_packets cap: monitored acquisition processes every packet.
        pixels, tdcs, processed_packets, total_packets = decode_tpx3(str(path), max_packets=None)
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
            pulse_records, cluster_size_sum, cluster_count, processed_packets, total_packets, start
        )
        stats["analysis_time_last"] = time.perf_counter() - start

        plot_result = {
            "pixel_hist": pixel_hist,
            "cluster_hist": cluster_hist,
            "itof_hist": itof_hist,
            "etof_hist": etof_hist,
            "cluster_t_hist": cluster_t_hist,
            "stats": stats,
            "settings": settings,
        }

        return {
            "pulses_sorted": pulses_sorted,
            "clusters_by_pulse": clusters_by_pulse,
            "etof_by_pulse": etof_by_pulse,
            "itof_by_pulse": itof_by_pulse,
            "plot_result": plot_result,
        }

    # ---- main-thread queue handling ---------------------------------------

    def _poll_queue(self):
        try:
            while True:
                result = self._queue.get_nowait()
                self._apply_result(result)
        except Empty:
            pass
        self.after(200, self._poll_queue)

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
        if "pending" in result:
            self.pending_var.set(str(result["pending"]))

        settings_changed = (
            self._accumulated is not None
            and self._accumulated.get("settings") != result.get("settings")
        )

        if self._reset_accumulation or self._accumulated is None or settings_changed:
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
            self._accumulated["pixel_hist"] = add_hist_2d(self._accumulated["pixel_hist"], result["pixel_hist"])
            self._accumulated["cluster_hist"] = add_hist_2d(
                self._accumulated["cluster_hist"], result["cluster_hist"]
            )
            self._accumulated["itof_hist"] = add_hist(self._accumulated["itof_hist"], result["itof_hist"])
            self._accumulated["etof_hist"] = add_hist(self._accumulated["etof_hist"], result["etof_hist"])
            self._accumulated["cluster_t_hist"] = add_hist(
                self._accumulated["cluster_t_hist"], result["cluster_t_hist"]
            )
            self._accumulated["stats"] = merge_stats(self._accumulated["stats"], result["stats"])

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
