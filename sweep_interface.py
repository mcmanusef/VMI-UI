"""Parameter Sweep: move a Newport XPS-D stage through a set of positions,
collecting + clustering data into one cv4 file per physical position (not
one per movement -- every pass that revisits a position appends to the same
file, see position_meta in _sweep_loop). Each cv4 is opened, appended to,
and closed again for every single processed raw file rather than held open
for the whole run (Cv4Writer reopens an existing file in place instead of
truncating it -- see cv4_writer.py).

At each position/pass ("visit"): move, report the measured position,
trigger frames one at a time (nTriggers=1, a fresh start_measurement() per
frame -- see _trigger_visit; the continuous, many-frames-per-command
pattern remains available in serval_client.py, just not used here) until
the dwell condition (cluster count or elapsed time) is met, then move on --
without waiting for those frames to actually be processed. A single
persistent background thread (_sweep_processor_loop) processes every raw
file strictly in the order it arrived, completely decoupled from stage
movement, appending into whichever position's cv4 that visit belongs to.
In cluster-dwell mode, the whole position list is calibrated once up front
-- _CALIBRATION_FRAME_COUNT frames taken and processed at every position --
to fix how many frames each position needs; that count is never adjusted
again during the real sweep (see _calibrate_frame_counts).

clusters_per_shot / electrons_per_shot / ions_per_shot -- already computed
by tpx_processing.summarize_records / merge_stats, same as every other tab
-- are exactly the "average cluster/e-ToF/i-ToF rate" plotted against
measured position, reported per visit once that visit's own frames finish
processing (see _finish_visit).

The stage is never homed or moved except by an explicit button here --
Connect only logs in and reads firmware/object info.
"""
import datetime
import json
import math
import pathlib
import threading
import time
from queue import Queue, Empty

import numpy as np
import qtk as tk
from qtk import ttk, filedialog, grid_into

import pyqtgraph as pg

import app_settings
import cv4_writer
import serval_client
from qt_plots import mpl_color, ZoomFocusViewBox
import time_estimate
from cv4_writer import Cv4Writer
import ui_style
from stage_interface import describe_move_failure
from tpx_processing import (
    decode_tpx3,
    sort_tdcs,
    group_pixels_by_pulse,
    group_times_relative,
    cluster_pixels_by_pulse,
    summarize_records,
    merge_stats,
)

_CALIBRATION_FRAME_COUNT = 4  # frames collected per calibration point

_RATE_KEYS = ("clusters_per_shot", "electrons_per_shot", "ions_per_shot")
_RATE_LABELS = {
    "clusters_per_shot": "Cluster rate",
    "electrons_per_shot": "e-ToF rate",
    "ions_per_shot": "i-ToF rate",
}
_RATE_COLORS = {
    "clusters_per_shot": "tab:blue",
    "electrons_per_shot": "tab:orange",
    "ions_per_shot": "tab:green",
}


class SweepInterface(ttk.Frame):
    def __init__(self, parent, server_var=None, acq_shared_vars=None, stage_ui=None, coordinator=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._server_var = server_var
        self._fallback_server_var = tk.StringVar(self, value="http://localhost:8080")
        # The stage connection itself lives on the Stage Control tab (see
        # stage_interface.StageInterface) -- sweeps are driven through
        # whatever XPSStage instance is connected there, rather than this
        # tab opening its own separate connection to the same physical
        # stage.
        self._stage_ui = stage_ui
        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)

        acq = acq_shared_vars or {}
        self.target_var = acq.get("target_var") or tk.StringVar(self)
        self.target_pressure_var = acq.get("target_pressure_var") or tk.StringVar(self)
        self.background_pressure_var = acq.get("background_pressure_var") or tk.StringVar(self)
        self.power_var = acq.get("power_var") or tk.StringVar(self)
        self.spot_size_var = acq.get("spot_size_var") or tk.StringVar(self)
        self.polarization_var = acq.get("polarization_var") or tk.StringVar(self)
        self.wavelength_var = acq.get("wavelength_var") or tk.StringVar(self)

        # Sweep-specific (not shared with Collection/Acquisition -- a
        # sweep's per-position exposure and save location are their own
        # concern), but still persisted individually.
        self.frame_time_var = app_settings.persistent_var(self, tk.StringVar, "sweep.frame_time", "1")
        self.save_folder_var = app_settings.persistent_var(
            self, tk.StringVar, "sweep.save_folder", self._default_save_folder()
        )
        self.position_mode_var = app_settings.persistent_var(self, tk.StringVar, "sweep.position_mode", "range")
        self.range_low_var = app_settings.persistent_var(self, tk.StringVar, "sweep.range_low", "0")
        self.range_high_var = app_settings.persistent_var(self, tk.StringVar, "sweep.range_high", "10")
        self.range_count_var = app_settings.persistent_var(self, tk.StringVar, "sweep.range_count", "11")
        self.position_list_var = app_settings.persistent_var(self, tk.StringVar, "sweep.position_list", "")
        self.passes_var = app_settings.persistent_var(self, tk.StringVar, "sweep.passes", "1")
        self.dwell_mode_var = app_settings.persistent_var(self, tk.StringVar, "sweep.dwell_mode", "clusters")
        self.dwell_value_var = app_settings.persistent_var(self, tk.StringVar, "sweep.dwell_value", "10000")

        self.status_var = tk.StringVar(self, value="Idle")
        self._progress_var = tk.DoubleVar(self, value=0.0)
        self.eta_var = tk.StringVar(self, value="--")

        self._sweep_thread = None
        self._stop_event = threading.Event()
        # Graceful stop: don't trigger any more frames/positions, but let
        # whatever's already queued finish processing -- unlike _stop_event,
        # which can abandon a backlog mid-drain. See finish_and_stop().
        self._finish_event = threading.Event()
        self._queue: Queue = Queue()
        self._points = []  # every completed visit, forward and backward alike

        # Estimated time to completion: the average of *measured*
        # per-position durations (move + settle + collect, whatever that
        # actually took) times how many positions are left. No pre-sweep
        # guess -- it reads "Estimating..." until the first position
        # finishes and there's a real duration to base it on.
        self._eta_after_id = None
        self._position_durations = []
        self._sweep_total_positions = 0
        self._sweep_completed = 0
        self._sweep_finish_estimate = None

        # Live collector/processor backlog reporting for whichever position
        # is currently being collected -- same idea as Monitored
        # Acquisition's "Live totals".
        self.position_collected_var = tk.StringVar(self, value="--")
        self.position_processed_var = tk.StringVar(self, value="--")
        self.position_pending_var = tk.StringVar(self, value="--")
        self.position_dead_time_var = tk.StringVar(self, value="--")
        self.position_estimate_var = tk.StringVar(self, value="--")

        self._build_ui()
        self._poll_queue()

    def _default_save_folder(self):
        return rf"C:\DATA\{datetime.date.today().strftime('%Y%m%d')}\sweep"

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.rowconfigure(1, weight=1)

        self._build_sidebar(sidebar)
        self._build_main(main)

    def _build_sidebar(self, sidebar):
        # Trailing empty row keeps content packed at the top as sections
        # collapse (see Monitored Acquisition).
        sidebar.rowconfigure(7, weight=1)

        # Same Stop / Force Stop pair as Monitored Acquisition: "Force Stop"
        # can abandon a backlog mid-drain, hence the danger styling.
        ui_style.build_button_bar(
            sidebar, 0, [("Start", self.start), ("Stop", self.finish_and_stop)], danger=("Force Stop", self.stop),
        )
        # "Position 3/10 done (...)" is mid-sweep progress, not idle.
        self._status_block = ui_style.StatusBlock(
            sidebar, 1, self.status_var, progress_var=self._progress_var, eta_var=self.eta_var,
            is_running=lambda text: "done (measured" in text or any(k in text for k in ui_style.STATUS_GREEN_KEYWORDS),
        )
        # The stage connection itself (IP/group, Connect/Initialize/Home,
        # manual jog + Set Zero) lives on the Stage Control tab -- see
        # stage_interface.StageInterface -- so a sweep just needs that
        # connection to already be up before Start.
        ui_style.add_note(
            sidebar, 2,
            '"Stop" lets the current backlog fully process before stopping (no half-done files); '
            '"Force Stop" cuts off immediately. Uses the connection from the Stage Control tab -- '
            "connect/initialize/home there first.",
            columnspan=1, pady=(0, 8),
        )

        positions = ui_style.build_section(sidebar, 3, "Positions")
        ttk.Radiobutton(
            positions, text="Range", value="range", variable=self.position_mode_var
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        range_row = ttk.Frame(positions)
        range_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 4))
        # Three label+entry pairs on one line only fit the 400px sidebar
        # without the row frame's own default margins on top of the
        # section's, and with small minimum entry widths -- the entries
        # stretch to share whatever width the row actually has.
        ui_style.zero_margins(range_row)
        for idx, (label, var) in enumerate([
            ("Low:", self.range_low_var),
            ("High:", self.range_high_var),
            ("Count:", self.range_count_var),
        ]):
            ttk.Label(range_row, text=label).grid(row=0, column=2 * idx, sticky="w", padx=(0 if idx == 0 else 8, 4))
            ttk.Entry(range_row, textvariable=var, width=5).grid(row=0, column=2 * idx + 1, sticky="ew")
            range_row.columnconfigure(2 * idx + 1, weight=1)
        ttk.Radiobutton(
            positions, text="List (comma-separated)", value="list", variable=self.position_mode_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Entry(positions, textvariable=self.position_list_var).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(2, 4)
        )

        options = ui_style.build_section(sidebar, 4, "Sweep options")
        ui_style.add_form_row(options, 0, "Passes:", ttk.Entry(options, textvariable=self.passes_var, width=18))
        ui_style.add_note(options, 1, "1 = forward only; more than 1 alternates forward/backward.")
        dwell_row = ttk.Frame(options)
        dwell_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Radiobutton(
            dwell_row, text="Dwell for clusters:", value="clusters", variable=self.dwell_mode_var
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            dwell_row, text="Dwell for seconds:", value="time", variable=self.dwell_mode_var
        ).grid(row=1, column=0, sticky="w")
        ttk.Entry(dwell_row, textvariable=self.dwell_value_var, width=10).grid(row=0, column=1, rowspan=2, padx=(8, 0))
        ui_style.add_form_row(
            options, 3, "Frame time (s):", ttk.Entry(options, textvariable=self.frame_time_var, width=18)
        )
        ui_style.add_form_row(
            options, 4, "Save folder:", ui_style.build_path_field(options, self.save_folder_var, self._browse_folder)
        )

        self.notes = ui_style.build_metadata_section(sidebar, 5, self)

        live = ui_style.build_section(sidebar, 6, "Live collection", pady=0)
        ui_style.add_stat_rows(live, [
            ("Est. frames needed (this visit):", self.position_estimate_var),
            ("Frames collected (this visit):", self.position_collected_var),
            ("Frames processed (sweep total):", self.position_processed_var),
            ("Pending (sweep backlog):", self.position_pending_var),
            ("Dead time (this visit):", self.position_dead_time_var),
        ])

    def _build_main(self, main):
        columns = ("pass", "direction", "requested", "measured", "clusters", "etof", "itof", "files", "cv4")
        headers = {
            "pass": "Pass", "direction": "Dir", "requested": "Requested", "measured": "Measured",
            "clusters": "Cluster/shot", "etof": "e-ToF/shot", "itof": "i-ToF/shot",
            "files": "Files", "cv4": "cv4 file",
        }
        self._tree = ttk.Treeview(main, columns=columns, show="headings", height=8)
        for col in columns:
            self._tree.heading(col, text=headers[col])
            self._tree.column(col, width=90 if col != "cv4" else 160, anchor="center")
        self._tree.grid(row=0, column=0, sticky="ew")

        plot_frame = ttk.Frame(main)
        plot_frame.grid(row=1, column=0, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        ui_style.zero_margins(plot_frame)

        self._plot_widget = pg.PlotWidget(viewBox=ZoomFocusViewBox())
        self._plot_item = self._plot_widget.getPlotItem()
        self._plot_item.addLegend()
        grid_into(self._plot_widget, plot_frame, row=0, column=0, sticky="nsew")
        self._redraw_plot()

    def _browse_folder(self):
        chosen = filedialog.askdirectory(
            initialdir=self.save_folder_var.get().strip() or ".",
            title="Choose save folder",
            parent=self,
        )
        if chosen:
            self.save_folder_var.set(chosen)

    # ---- server helper -----------------------------------------------------

    def _server_url(self):
        url = ""
        if self._server_var is not None:
            url = self._server_var.get().strip()
        if not url:
            url = self._fallback_server_var.get().strip() or "http://localhost:8080"
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        return url

    # ---- start / stop -------------------------------------------------

    def is_running(self):
        return bool(self._sweep_thread and self._sweep_thread.is_alive())

    def _read_positions(self):
        if self.position_mode_var.get() == "range":
            low = float(self.range_low_var.get())
            high = float(self.range_high_var.get())
            count = int(self.range_count_var.get())
            if count < 1:
                raise ValueError("Count must be at least 1.")
            return [float(v) for v in np.linspace(low, high, count)]

        text = self.position_list_var.get().strip()
        if not text:
            raise ValueError("Position list is empty.")
        return [float(v) for v in text.split(",") if v.strip()]

    def _collect_metadata(self, frame_time, dwell_mode, dwell_value, passes):
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
            "Dwell Mode": dwell_mode,
            "Dwell Value": dwell_value,
            "Passes": passes,
            "Start Time (UTC)": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def start(self):
        if self._coordinator is not None:
            self._coordinator.stop_others(self)

        if self.is_running():
            self.stop()

        stage = self._stage_ui.get_stage() if self._stage_ui is not None else None
        if not stage or not stage.connected:
            self.status_var.set("Connect to the stage first (see the Stage Control tab).")
            return
        group_name = self._stage_ui.group_var.get().strip()
        if not group_name:
            self.status_var.set("Group name is required (see the Stage Control tab).")
            return

        try:
            positions = self._read_positions()
        except ValueError as exc:
            self.status_var.set(f"Invalid positions: {exc}")
            return
        if not positions:
            self.status_var.set("No positions to sweep.")
            return

        try:
            passes = max(1, int(self.passes_var.get()))
        except ValueError:
            self.status_var.set("Invalid pass count.")
            return

        dwell_mode = self.dwell_mode_var.get()
        try:
            dwell_value = float(self.dwell_value_var.get())
            if dwell_value <= 0:
                raise ValueError
        except ValueError:
            self.status_var.set("Invalid dwell value.")
            return

        try:
            frame_time = float(self.frame_time_var.get())
            if frame_time <= 0:
                raise ValueError
        except ValueError:
            self.status_var.set("Invalid frame time.")
            return

        folder_text = self.save_folder_var.get().strip()
        if not folder_text:
            self.status_var.set("Save folder is required.")
            return
        save_folder = pathlib.Path(folder_text)
        save_folder.mkdir(parents=True, exist_ok=True)
        raw_root = save_folder / "raw"
        raw_root.mkdir(parents=True, exist_ok=True)

        metadata_base = self._collect_metadata(frame_time, dwell_mode, dwell_value, passes)
        server = self._server_url()
        zero_offset = self._stage_ui.zero_offset_var.get()

        self._stop_event.clear()
        self._finish_event.clear()
        self._tree.delete(*self._tree.get_children())
        self._points = []  # every completed visit, forward and backward alike
        self._redraw_plot()
        self._progress_var.set(0.0)
        self._reset_position_live_stats()
        self.position_processed_var.set("0")
        self.position_pending_var.set("0")

        self._position_durations = []
        self._sweep_total_positions = len(positions) * passes
        self._sweep_completed = 0
        self._sweep_finish_estimate = None
        if dwell_mode == "clusters":
            self.status_var.set("Calibrating...")
            self.eta_var.set("Calibrating...")
        else:
            self.status_var.set("Sweeping...")
            self.eta_var.set("Estimating... (need one completed position)")
        self._start_eta_ticker()

        self._sweep_thread = threading.Thread(
            target=self._sweep_loop,
            args=(
                stage, server, group_name, zero_offset, positions, passes,
                dwell_mode, dwell_value, frame_time, raw_root, save_folder, metadata_base,
            ),
            daemon=True,
        )
        self._sweep_thread.start()

    def stop(self):
        self._stop_event.set()
        self.status_var.set("Stopping...")
        # Send the stop directly, right now, instead of only relying on the
        # sweep thread's own finally blocks to get there -- that thread can
        # be blocked for up to 120s inside a stage move/settle (or any
        # Serval HTTP call), none of which check _stop_event, so the
        # join(timeout=5) below can easily give up before the thread ever
        # reaches a stop_measurement() call of its own. This way Serval
        # always gets told to stop the moment Stop is pressed, regardless of
        # what the worker thread happens to be doing.
        try:
            serval_client.stop_measurement(self._server_url())
        except Exception:
            pass
        if self._sweep_thread:
            self._sweep_thread.join(timeout=5)
        self.status_var.set("Stopped.")
        self._stop_eta_ticker()
        self.eta_var.set("--")

    def finish_and_stop(self):
        """Stop moving/triggering new frames, but let the whole sweep-wide
        backlog already queued keep processing -- so the sweep ends with
        every triggered raw file either fully processed or, if it was never
        triggered, not existing at all, instead of Stop's immediate cutoff
        which can abandon a mid-drain backlog. Doesn't block: each
        position's cv4 gets its final .partial (or completed) name once
        the whole backlog has actually finished draining (see
        _sweep_loop's tail, after the processor thread is joined)."""
        if not self.is_running():
            return
        self._finish_event.set()
        self.status_var.set("Finishing current position, then stopping...")

    # ---- estimated time to completion ---------------------------------

    def _start_eta_ticker(self):
        self._stop_eta_ticker()
        self._eta_after_id = self.after(1000, self._tick_eta)

    def _stop_eta_ticker(self):
        if self._eta_after_id is not None:
            self.after_cancel(self._eta_after_id)
            self._eta_after_id = None

    def _tick_eta(self):
        if self._sweep_finish_estimate is not None:
            remaining = (self._sweep_finish_estimate - datetime.datetime.now()).total_seconds()
            self.eta_var.set(time_estimate.format_eta(remaining))
        self._eta_after_id = self.after(1000, self._tick_eta)

    def _record_position_duration(self, elapsed_seconds):
        """Update the running average per-position duration (move + settle
        + collect, whatever that actually took) and re-derive the finish
        estimate from it and however many positions are left."""
        self._position_durations.append(elapsed_seconds)
        avg = sum(self._position_durations) / len(self._position_durations)
        remaining_positions = max(0, self._sweep_total_positions - self._sweep_completed)
        self._sweep_finish_estimate = datetime.datetime.now() + datetime.timedelta(
            seconds=avg * remaining_positions
        )

    # ---- sweep worker thread -----------------------------------------

    def _process_frame(self, path):
        """Decode + cluster one raw file. Returns (pulses_sorted,
        clusters_by_pulse, etof_by_pulse, itof_by_pulse, stats) -- the same
        per-frame pipeline every other tab uses."""
        frame_start = time.perf_counter()
        pixels, tdcs, processed_packets, total_packets = decode_tpx3(str(path), max_packets=None)
        etof, itof, pulses = sort_tdcs(300.0, tdcs)
        pulses_sorted = sorted(pulses)
        pixels_by_pulse = group_pixels_by_pulse(pixels, pulses_sorted)
        clusters_by_pulse, cluster_size_sum, cluster_count = cluster_pixels_by_pulse(pixels_by_pulse)
        etof_by_pulse = group_times_relative(etof, pulses_sorted)
        itof_by_pulse = group_times_relative(itof, pulses_sorted)
        pulse_records = [
            (pt, pixels_by_pulse.get(pt, []), clusters_by_pulse.get(pt, []),
             etof_by_pulse.get(pt, []), itof_by_pulse.get(pt, []))
            for pt in pulses_sorted
        ]
        stats = summarize_records(
            pulse_records, cluster_size_sum, cluster_count, processed_packets, total_packets, frame_start
        )
        return pulses_sorted, clusters_by_pulse, etof_by_pulse, itof_by_pulse, stats

    def _wait_for_next_frame(self, raw_dir, wait_timeout, last_path, last_mtime):
        """Block for the next raw file to land and settle. Doesn't trigger
        anything itself -- the measurement is started once per position (see
        _collect_position) in CONTINUOUS mode with a very large trigger
        count, so Serval keeps writing a new raw file every frame_time on
        its own; this just watches for the next one to show up. Returns
        (path, stat) or (None, None) if nothing showed up within the
        timeout."""
        newest = serval_client.wait_for_new_file(raw_dir, self._stop_event, last_path, last_mtime, timeout=wait_timeout)
        if newest is None:
            return None, None
        newest_stat = newest.stat()
        if newest_stat.st_size == 0:
            return None, None
        return newest, newest_stat

    def _trigger_visit(self, server, raw_dir, frame_time, dwell_mode, dwell_value, frame_count,
                        pending_queue, visit_id, visit_records):
        """Collector half only -- triggers frames one at a time
        (nTriggers=1, a fresh start_measurement() per frame -- the
        destination was already configured for this by the caller,
        _sweep_loop) for this single visit (one pass through one position)
        and hands each raw file straight to the sweep-wide pending_queue,
        tagged with visit_id, in the order it arrives.

        Does NOT wait for those frames to actually be processed: this
        returns (and the caller moves the stage to the next position) as
        soon as this visit's own dwell condition is met, regardless of how
        far behind _sweep_processor_loop's persistent, single background
        thread is. That thread drains pending_queue strictly in arrival
        order, so files always get processed in the order they came in,
        never out of order and never blocking the stage from moving on.

        Records this visit's outcome (how many frames it triggered, and
        whether it reached its dwell target naturally vs. being cut short)
        directly into visit_records[visit_id] for the processor to pick up
        once it finishes that visit's last frame. Returns (frames_collected,
        completed_naturally).
        """
        wait_timeout = max(120.0, frame_time * 5.0 + 2.0)
        last_path = None
        last_mtime = None
        frames_collected = 0
        collection_start = time.time()
        completed_naturally = False
        # Absolute safety ceiling regardless of source (a pathological
        # calibration estimate can't spin the collector forever).
        max_frames = 5000

        def stopping():
            return self._stop_event.is_set() or self._finish_event.is_set()

        try:
            while not stopping() and frames_collected < max_frames:
                if dwell_mode == "time" and (time.time() - collection_start) >= dwell_value:
                    completed_naturally = True
                    break
                if dwell_mode == "clusters" and frame_count and frames_collected >= frame_count:
                    completed_naturally = True
                    break

                try:
                    serval_client.start_measurement(server)
                except Exception as exc:
                    self._queue.put({"error": f"Start failed: {exc}"})
                    time.sleep(1)
                    continue

                try:
                    newest, newest_stat = self._wait_for_next_frame(raw_dir, wait_timeout, last_path, last_mtime)
                except Exception as exc:
                    self._queue.put({"error": str(exc)})
                    time.sleep(1)
                    continue
                if newest is None:
                    continue

                last_path, last_mtime = newest, newest_stat.st_mtime
                frames_collected += 1
                pending_queue.put((newest, visit_id))

                elapsed_wall = time.time() - collection_start
                measured_time = frames_collected * frame_time
                dead_time_pct = max(0.0, (elapsed_wall - measured_time) / elapsed_wall * 100.0) if elapsed_wall > 0 else 0.0
                self._queue.put({
                    "position_collected": frames_collected,
                    "position_dead_time_pct": dead_time_pct,
                })
        finally:
            try:
                serval_client.stop_measurement(server)
                # Doesn't confirm Serval actually wound down. Without
                # waiting for it, the *next* position's move + destination
                # reconfigure can race against a still-stopping measurement
                # and hang.
                serval_client.wait_for_measurement_idle(server)
            except Exception:
                pass

        visit_records[visit_id]["frames_expected"] = frames_collected
        visit_records[visit_id]["completed_naturally"] = completed_naturally
        return frames_collected, completed_naturally

    def _sweep_processor_loop(self, pending_queue, position_meta, visit_records,
                               mover_done_event, summary, summary_path):
        """Persistent, single background thread for the whole sweep --
        drains pending_queue strictly in the order frames arrived
        (Queue.get() is FIFO, and this is the only thread that ever reads
        from it, so arrival order is exactly processing order) completely
        decoupled from stage movement: _sweep_loop's mover never waits for
        this to catch up before moving to the next position.

        Every visit (one pass through one position) writes into the SAME
        cv4 file for that physical position, but the file isn't held open
        across frames: it's reopened (Cv4Writer appends to an existing file
        rather than truncating it, see cv4_writer.py) for each raw file,
        written to, and closed again immediately after -- even if the next
        queued item is for the same position. That keeps at most one cv4
        file handle open at a time regardless of how many positions are in
        flight, at the cost of a reopen per frame.

        Only the hard self._stop_event aborts this early (possibly leaving
        already-collected-but-unprocessed raw files behind). Finish-and-stop
        only tells the mover to stop triggering *new* frames/positions; this
        loop keeps draining whatever's already queued regardless, so every
        raw file that was actually triggered ends up either fully processed
        or (if it never got far enough to be triggered at all) not existing
        at all.
        """
        total_processed = 0
        try:
            while not self._stop_event.is_set():
                try:
                    newest, visit_id = pending_queue.get(timeout=0.2)
                except Empty:
                    if mover_done_event.is_set() and pending_queue.empty():
                        break
                    continue
                try:
                    record = visit_records[visit_id]
                    position_key = record["position_key"]
                    meta = position_meta[position_key]
                    writer = Cv4Writer(
                        cv4_writer.partial_path_for(meta["cv4_path"]), metadata=meta["base_metadata"]
                    )
                    try:
                        pulses_sorted, clusters_by_pulse, etof_by_pulse, itof_by_pulse, frame_stats = (
                            self._process_frame(newest)
                        )
                        writer.append_frame(pulses_sorted, clusters_by_pulse, etof_by_pulse, itof_by_pulse)
                        writer.flush()

                        total_processed += 1
                        record["frames_processed"] += 1
                        record["stats"] = (
                            frame_stats if record["stats"] is None else merge_stats(record["stats"], frame_stats)
                        )
                        meta["frames_processed"] += 1
                        self._queue.put({
                            "sweep_processed_total": total_processed,
                            "sweep_pending": pending_queue.qsize(),
                        })

                        if (
                            record["frames_expected"] is not None
                            and record["frames_processed"] >= record["frames_expected"]
                        ):
                            self._finish_visit(writer, meta, record, position_key, summary, summary_path)
                    finally:
                        writer.close()
                except Exception as exc:
                    self._queue.put({"error": f"Processing {newest.name} failed: {exc}"})
                finally:
                    pending_queue.task_done()
        except Exception as exc:
            self._queue.put({"error": str(exc)})
        finally:
            mover_done_event.set()  # in case we're exiting due to an error, not the mover finishing

    def _finish_visit(self, writer, meta, record, position_key, summary, summary_path):
        """Called (from the processor thread) once every frame a visit
        triggered has actually been processed -- updates that position's
        cv4 metadata (a running "Visits" list, since multiple passes now
        share one file per position) and reports the visit's own stats
        (not the position's cumulative total across all visits) to the UI
        for the treeview row / plot point, same shape as before."""
        visit_entry = {
            "Pass Number": record["pass"],
            "Pass Direction": record["direction"],
            "Measured Position": record["measured"],
            "Frames Collected": record["frames_expected"],
            "Frames Processed": record["frames_processed"],
            "Completed Naturally": record["completed_naturally"],
        }
        meta["visits"].append(visit_entry)
        writer.set_attrs({
            "Visits": json.dumps(meta["visits"]),
            "Frames Collected": meta["frames_collected"],
            "Frames Processed": meta["frames_processed"],
        })

        stats = record["stats"] or {}
        entry = {
            "pass": record["pass"],
            "direction": record["direction"],
            "requested_position": position_key,
            "measured_position": record["measured"],
            "clusters_per_shot": stats.get("clusters_per_shot", 0),
            "electrons_per_shot": stats.get("electrons_per_shot", 0),
            "ions_per_shot": stats.get("ions_per_shot", 0),
            "cv4_file": meta["cv4_path"].name,
            "raw_folder": record["raw_folder_name"],
            "elapsed_seconds": record["elapsed_seconds"],
            "frames_collected": record["frames_expected"],
            "frames_processed": record["frames_processed"],
        }
        summary.append(entry)
        try:
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except OSError:
            pass

        self._queue.put({
            "position_done": True,
            "global_index": record["global_index"],
            "total_positions": record["total_positions"],
            **entry,
        })

    def _calibrate_frame_counts(self, stage, server, group_name, zero_offset, positions, frame_time, dwell_value, raw_root):
        """Cluster-dwell mode only, run once before the real sweep starts:
        visit every distinct requested position, collect and process
        _CALIBRATION_FRAME_COUNT frames there, and use the average cluster
        count per frame over that batch to decide how many frames that
        position will need to reach dwell_value clusters. That number is
        then fixed for the whole sweep -- every pass collects exactly that
        many frames at that position, with no adjustment afterward based on
        how the real cluster total turns out (unlike the old behavior,
        which kept topping up the count if it undershot).

        Several frames (rather than a single one) smooth out shot-to-shot
        rate fluctuations, at the cost of _CALIBRATION_FRAME_COUNT frames'
        worth of time per position up front. Frames are triggered one at a
        time -- same one-frame-per-command pattern as _trigger_visit -- and
        their stats accumulated via merge_stats as they're processed.

        Calibration frames are real triggered exposures, so they're saved
        (not discarded) into their own raw_root/calibration subfolder, same
        as every other raw file this tab writes -- just not appended to any
        position's cv4.

        Returns {requested_position: frame_count}. A position where
        calibration itself fails (bad move, no frames, processing error) is
        simply left out of the dict; the caller falls back to the median of
        whatever positions did calibrate successfully.
        """
        calib_root = raw_root / "calibration"
        calib_root.mkdir(parents=True, exist_ok=True)
        wait_timeout = max(120.0, frame_time * 5.0 + 2.0)
        frame_counts = {}
        total = len(positions)

        def stopping():
            return self._stop_event.is_set() or self._finish_event.is_set()

        for idx, requested in enumerate(positions, start=1):
            if stopping():
                break
            self._queue.put({
                "calibration_progress": idx,
                "calibration_total": total,
                "calibration_position": requested,
            })
            target = zero_offset + requested

            try:
                stage.move_absolute(group_name, target)
                stage.wait_for_settle(group_name, timeout=120.0)
            except Exception as exc:
                self._queue.put({
                    "error": f"Calibration move failed at position {requested}: "
                             f"{describe_move_failure(stage, group_name, exc)}"
                })
                continue

            calib_dir = calib_root / f"calib_{idx:03d}_{requested:+.4f}"
            calib_dir.mkdir(parents=True, exist_ok=True)
            try:
                serval_client.configure_single_trigger_destination(
                    server, frame_time, calib_dir, "calib%Hms_"
                )
            except Exception as exc:
                self._queue.put({"error": f"Calibration config failed at position {requested}: {exc}"})
                continue

            # Collect _CALIBRATION_FRAME_COUNT frames, one trigger at a
            # time, accumulating stats across all of them.
            last_path = None
            last_mtime = None
            frames_collected = 0
            combined_stats = None
            try:
                while not stopping() and frames_collected < _CALIBRATION_FRAME_COUNT:
                    try:
                        serval_client.start_measurement(server)
                    except Exception as exc:
                        self._queue.put({"error": f"Calibration start failed at position {requested}: {exc}"})
                        break

                    try:
                        newest, newest_stat = self._wait_for_next_frame(calib_dir, wait_timeout, last_path, last_mtime)
                    except Exception as exc:
                        self._queue.put({"error": f"Calibration collection failed at position {requested}: {exc}"})
                        break
                    if newest is None:
                        break
                    last_path, last_mtime = newest, newest_stat.st_mtime

                    try:
                        _, _, _, _, stats = self._process_frame(newest)
                    except Exception as exc:
                        self._queue.put({"error": f"Calibration processing failed at position {requested}: {exc}"})
                        break

                    frames_collected += 1
                    combined_stats = stats if combined_stats is None else merge_stats(combined_stats, stats)
                    self._queue.put({
                        "calibration_position_collected": frames_collected,
                        "calibration_position_total": _CALIBRATION_FRAME_COUNT,
                    })
            finally:
                try:
                    serval_client.stop_measurement(server)
                    serval_client.wait_for_measurement_idle(server)
                except Exception:
                    pass

            if not frames_collected or combined_stats is None:
                self._queue.put({"error": f"Calibration got no frames at position {requested}."})
                continue

            clusters_total = combined_stats.get("total_clusters", 0)
            # A calibration window with zero clusters still needs *some*
            # positive rate to divide by -- treat the average as 1 rather
            # than dividing by zero (that position will just end up with
            # the largest frame_count of the sweep, which is the correct
            # behavior for a genuinely low-rate position).
            clusters_per_frame = max(clusters_total / frames_collected, 1.0)
            frame_count = max(1, math.ceil(dwell_value / clusters_per_frame))
            frame_counts[requested] = frame_count
            self._queue.put({
                "calibration_result": True,
                "position": requested,
                "clusters_per_frame": clusters_per_frame,
                "frames_sampled": frames_collected,
                "frame_count": frame_count,
            })

        return frame_counts

    def _sweep_loop(
        self, stage, server, group_name, zero_offset, positions, passes,
        dwell_mode, dwell_value, frame_time, raw_root, save_folder, metadata_base,
    ):
        """The mover: moves the stage through every position/pass and
        triggers each visit's frames (see _trigger_visit), but never waits
        for those frames to actually be processed before moving on --
        _sweep_processor_loop runs the whole time in its own persistent
        background thread, draining frames strictly in the order they
        arrived, decoupled from stage movement. Every physical position
        gets exactly one cv4 file, shared across every pass that visits it
        (position_meta, built up here and finalized at the end once the
        processor has been joined) -- not one file per movement.
        """
        total_positions = len(positions) * passes
        global_index = 0
        move_failures = 0
        config_failures = 0
        position_index = {p: i + 1 for i, p in enumerate(positions)}

        # Cluster-dwell mode: calibrate every position's frame count once,
        # up front, before triggering any real data collection -- see
        # _calibrate_frame_counts. Time-dwell mode doesn't need this;
        # dwell_value there is already an exact, known duration.
        frame_counts = {}
        fallback_count = None
        if dwell_mode == "clusters":
            frame_counts = self._calibrate_frame_counts(
                stage, server, group_name, zero_offset, positions, frame_time, dwell_value, raw_root
            )
            if self._stop_event.is_set() or self._finish_event.is_set():
                self._queue.put({"sweep_done": True, "reason": "Stopped during calibration."})
                return
            known_counts = sorted(frame_counts.values())
            fallback_count = known_counts[len(known_counts) // 2] if known_counts else 1
            total_frames_est = passes * sum(frame_counts.get(p, fallback_count) for p in positions)
            self._queue.put({
                "calibration_done": True,
                "estimated_seconds": total_frames_est * frame_time,
            })

        # Whole-sweep processing infrastructure -- see _sweep_processor_loop
        # and _finish_visit. position_meta is only ever written to by the
        # processor thread (frames_processed, visits); this (mover) thread
        # only ever reads/creates entries before any frame for that
        # position is ever queued, so there's no race on first use. Each
        # position's cv4 is opened/closed per frame by the processor
        # (Cv4Writer appends to an existing file rather than truncating),
        # not held open here.
        pending_queue: Queue = Queue()
        position_meta = {}
        visit_records = {}
        mover_done_event = threading.Event()
        summary = []
        summary_path = save_folder / "sweep_positions.json"
        visit_id = 0

        processor = threading.Thread(
            target=self._sweep_processor_loop,
            args=(pending_queue, position_meta, visit_records, mover_done_event, summary, summary_path),
            daemon=True,
        )
        processor.start()

        try:
            for pass_idx in range(passes):
                if self._stop_event.is_set() or self._finish_event.is_set():
                    break
                direction = "forward" if pass_idx % 2 == 0 else "backward"
                seq = positions if direction == "forward" else list(reversed(positions))

                for requested in seq:
                    if self._stop_event.is_set() or self._finish_event.is_set():
                        break
                    global_index += 1
                    target = zero_offset + requested
                    position_start = time.time()
                    visit_tag = f"pos_{global_index:03d}_{requested:+.4f}"

                    try:
                        stage.move_absolute(group_name, target)
                        measured_abs = stage.wait_for_settle(group_name, timeout=120.0)
                    except Exception as exc:
                        # One bad position (a transient controller hiccup, a
                        # requested value outside the travel limits, ...)
                        # shouldn't abort the rest of the list/pass -- report
                        # it and move on to the next requested position.
                        move_failures += 1
                        self._queue.put({"error": f"Move failed: {describe_move_failure(stage, group_name, exc)}"})
                        continue
                    measured = measured_abs - zero_offset

                    # Raw files stay organized per visit (traceable to
                    # exactly when/which pass produced them); it's only the
                    # processed cv4 output that's consolidated per position.
                    position_raw_dir = raw_root / visit_tag
                    position_raw_dir.mkdir(parents=True, exist_ok=True)

                    if requested not in position_meta:
                        base_metadata = dict(metadata_base)
                        base_metadata.update({
                            "Requested Position": requested,
                            "Zero Offset": zero_offset,
                        })
                        position_meta[requested] = {
                            "cv4_path": save_folder / f"pos_{position_index[requested]:03d}_{requested:+.4f}.cv4",
                            "base_metadata": base_metadata,
                            "visits": [],
                            "frames_collected": 0,
                            "frames_processed": 0,
                        }

                    if dwell_mode == "clusters":
                        # Known exactly, from calibration.
                        frame_count = frame_counts.get(requested, fallback_count)
                    else:
                        # Not exact (wall-clock dwell -- actual delivered
                        # frame count can land a bit either side of
                        # dwell_value / frame_time depending on
                        # timing/poll jitter) -- _trigger_visit just keeps
                        # triggering one frame at a time until the
                        # elapsed-time check fires.
                        frame_count = None

                    # One frame per trigger (see _trigger_visit) -- this
                    # tab's original design, rather than the continuous,
                    # many-frames-per-command pattern (still available via
                    # serval_client.configure_continuous_destination /
                    # max_frames_per_batch for future use).
                    try:
                        serval_client.configure_single_trigger_destination(
                            server, frame_time, position_raw_dir, "sweep%Hms_"
                        )
                    except Exception as exc:
                        config_failures += 1
                        self._queue.put({"error": f"Config failed at position {requested}: {exc}"})
                        continue

                    visit_id += 1
                    visit_records[visit_id] = {
                        "position_key": requested,
                        "pass": pass_idx + 1,
                        "direction": direction,
                        "measured": measured,
                        "frames_processed": 0,
                        "frames_expected": None,
                        "completed_naturally": False,
                        "stats": None,
                        "raw_folder_name": position_raw_dir.name,
                        "global_index": global_index,
                        "total_positions": total_positions,
                        "elapsed_seconds": None,
                    }

                    self._queue.put({"position_started": True})
                    if dwell_mode == "clusters" and frame_count:
                        self._queue.put({"position_estimate": frame_count})
                    frames_collected, _ = self._trigger_visit(
                        server, position_raw_dir, frame_time, dwell_mode, dwell_value, frame_count,
                        pending_queue, visit_id, visit_records,
                    )
                    visit_records[visit_id]["elapsed_seconds"] = time.time() - position_start
                    position_meta[requested]["frames_collected"] += frames_collected

                    if frames_collected == 0:
                        # Nothing was even triggered (e.g. stopped right as
                        # this visit began) -- no frame will ever arrive for
                        # it, so the processor will never see it finish;
                        # drop the empty record rather than leave it
                        # dangling.
                        visit_records.pop(visit_id, None)
                        continue

                    if self._finish_event.is_set():
                        break

                if self._finish_event.is_set():
                    break
        finally:
            # Tell the processor no more visits/frames are coming, then wait
            # -- with no timeout -- for it to fully drain whatever's already
            # queued. This is what makes "not ending when the stage moves"
            # safe: the mover is long done, but files keep processing in the
            # background for as long as it genuinely takes, in arrival
            # order, before any writer gets closed. A hard Stop still cuts
            # this short quickly since the processor checks _stop_event on
            # every item.
            mover_done_event.set()
            processor.join()

        sweep_completed_naturally = not (self._stop_event.is_set() or self._finish_event.is_set())
        # Each position's cv4 was opened/closed per frame throughout (see
        # _sweep_processor_loop), so there's nothing still open here --
        # just a final reopen per position (skipping any that never
        # actually got a processed frame, i.e. no file was ever created)
        # to stamp the run-level attrs and rename off .partial.
        for meta in position_meta.values():
            if meta["frames_processed"] == 0:
                continue
            partial_path = cv4_writer.partial_path_for(meta["cv4_path"])
            writer = Cv4Writer(partial_path)
            writer.set_attrs({
                "Frames Collected": meta["frames_collected"],
                "Frames Processed": meta["frames_processed"],
                "Complete": sweep_completed_naturally,
            })
            writer.close()
            cv4_writer.finalize_partial_path(partial_path, sweep_completed_naturally)

        if self._stop_event.is_set():
            cause = "Stopped by user."
        elif self._finish_event.is_set():
            cause = "Told to finish the current position and stop."
        else:
            cause = "Ran through every requested position/pass."

        succeeded = sum(len(meta["visits"]) for meta in position_meta.values())
        positions_written = sum(1 for meta in position_meta.values() if meta["frames_processed"] > 0)
        detail = f"{succeeded}/{global_index} visit(s) fully processed across {positions_written} position(s)"
        failures = []
        if move_failures:
            failures.append(f"{move_failures} move failure(s)")
        if config_failures:
            failures.append(f"{config_failures} config failure(s)")
        if failures:
            detail += " (" + ", ".join(failures) + ")"
        reason = f"{cause} {detail}."

        self._queue.put({"sweep_done": True, "reason": reason})

    # ---- main-thread queue handling / drawing -----------------------------

    def _poll_queue(self):
        try:
            while True:
                result = self._queue.get_nowait()
                self._apply_result(result)
        except Empty:
            pass
        self.after(100, self._poll_queue)  # matches the 10 Hz position/state poll

    def _apply_result(self, result):
        if "error" in result:
            self.status_var.set(f"Error: {result['error']}")
            return
        if result.get("sweep_done"):
            reason = result.get("reason", "Sweep finished.")
            print(f"[Parameter Sweep] {reason}")
            self.status_var.set(reason)
            self._stop_eta_ticker()
            self.eta_var.set("--")
            return
        if "calibration_progress" in result:
            self.status_var.set(
                f"Calibrating position {result['calibration_progress']}/{result['calibration_total']} "
                f"({result['calibration_position']:+.4f})..."
            )
            return
        if "calibration_position_collected" in result:
            self.status_var.set(
                f"Calibrating... {result['calibration_position_collected']}/"
                f"{result['calibration_position_total']} frame(s)"
            )
            return
        if result.get("calibration_result"):
            self.position_estimate_var.set(
                f"{result['clusters_per_frame']:.1f} clusters/frame (avg over {result['frames_sampled']} frame(s)) "
                f"-> {result['frame_count']} frame(s)"
            )
            return
        if result.get("calibration_done"):
            self._sweep_finish_estimate = datetime.datetime.now() + datetime.timedelta(
                seconds=result["estimated_seconds"]
            )
            self.eta_var.set(time_estimate.format_eta(result["estimated_seconds"]))
            self.status_var.set("Calibration done. Sweeping...")
            return
        if result.get("position_started"):
            self._reset_position_live_stats()
            return
        if "position_estimate" in result:
            self.position_estimate_var.set(f"~{result['position_estimate']}")
            return
        if "position_collected" in result:
            # This visit's own triggering progress (mover thread) -- how
            # many frames have been triggered so far at the current
            # position, independent of how far the persistent background
            # processor has gotten through the sweep-wide backlog.
            self.position_collected_var.set(str(result["position_collected"]))
            if "position_dead_time_pct" in result:
                self.position_dead_time_var.set(f"{result['position_dead_time_pct']:.1f}%")
            return
        if "sweep_processed_total" in result:
            # Sweep-wide totals from the persistent processor thread, not
            # tied to whichever position the mover is currently visiting.
            self.position_processed_var.set(str(result["sweep_processed_total"]))
            self.position_pending_var.set(str(result.get("sweep_pending", "--")))
            return
        if result.get("position_done"):
            self._add_position_row(result)
            self._points.append(result)
            self._redraw_plot()
            total = result.get("total_positions", 0)
            idx = result.get("global_index", 0)
            if total:
                self._progress_var.set(100.0 * idx / total)
            self._sweep_completed = idx
            self._record_position_duration(result["elapsed_seconds"])
            self.status_var.set(f"Position {idx}/{total} done (measured {result['measured_position']:.4f}).")
            return

    def _reset_position_live_stats(self):
        # Only the per-visit fields reset at the start of each new visit --
        # position_processed_var/position_pending_var are sweep-wide totals
        # from the persistent background processor now, not tied to
        # whichever position the mover just started moving to, so they keep
        # counting across visits instead of flashing back to 0.
        self.position_collected_var.set("0")
        self.position_dead_time_var.set("--")
        self.position_estimate_var.set("--")

    def _add_position_row(self, result):
        self._tree.insert(
            "", "end",
            values=(
                result["pass"], result["direction"],
                f"{result['requested_position']:.4f}", f"{result['measured_position']:.4f}",
                f"{result['clusters_per_shot']:.4g}", f"{result['electrons_per_shot']:.4g}",
                f"{result['ions_per_shot']:.4g}",
                f"{result['frames_processed']}/{result['frames_collected']}",
                result["cv4_file"],
            ),
        )
        children = self._tree.get_children()
        if children:
            self._tree.see(children[-1])

    def _redraw_plot(self):
        # Every visit to a given requested position -- regardless of which
        # pass it came from or which direction that pass was moving -- is
        # plotted as its own faint, unconnected point, but the line is
        # drawn through each position's mean rate rather than zigzagging
        # through every individual visit.
        self._plot_item.clear()
        if self._points:
            by_position = {}
            for p in self._points:
                by_position.setdefault(p["requested_position"], []).append(p)
            ordered = sorted(by_position.items(), key=lambda kv: kv[0])
            for key in _RATE_KEYS:
                color = mpl_color(_RATE_COLORS[key])
                xs_all = [p["measured_position"] for p in self._points]
                ys_all = [p[key] for p in self._points]
                faint = pg.mkColor(color)
                faint.setAlpha(89)  # ~0.35 alpha, matching the old matplotlib scatter
                self._plot_item.addItem(
                    pg.ScatterPlotItem(x=xs_all, y=ys_all, size=6, pen=None, brush=pg.mkBrush(faint))
                )
                xs_mean = [np.mean([p["measured_position"] for p in group]) for _, group in ordered]
                ys_mean = [np.mean([p[key] for p in group]) for _, group in ordered]
                self._plot_item.addItem(
                    pg.PlotDataItem(
                        xs_mean, ys_mean, pen=pg.mkPen(color, width=1.5),
                        symbol="o", symbolSize=6, symbolBrush=color, name=_RATE_LABELS[key],
                    )
                )
        self._plot_item.setTitle("Average cluster / e-ToF / i-ToF rate vs. position")
        self._plot_item.setLabel("bottom", "Position (relative to zero)")
        self._plot_item.setLabel("left", "Rate per shot")
        self._plot_item.autoRange()
        # clear() above removes the placeholder too, so it's re-added here.
        placeholder = ui_style.add_empty_placeholder(self._plot_item, "No data — press Start to sweep")
        ui_style.show_empty_placeholder(self._plot_item, placeholder, not self._points)
