"""Analysis tabs: offline work on data the acquisition tabs already wrote.

ConversionInterface converts a folder of raw TPX3 files into cv4 files with
the same per-file pipeline and .partial/finalize logic Monitored Acquisition
uses live (see raw_conversion.py). The other Analysis tabs are placeholders
for now (PlaceholderInterface).
"""
import datetime
import pathlib
import threading
import time
from queue import Queue, Empty

import qtk as tk
from qtk import ttk, messagebox, filedialog

import app_settings
import cv4_writer
import raw_conversion
import time_estimate
import ui_style
from plot_panel import HistogramPlotPanel
from timewalk import DEFAULT_CORRECTION_PATH, TimewalkCorrection


class ConversionInterface(ttk.Frame):
    """Raw data folder in, cv4 file(s) out -- one per run, or one per
    position for a parameter sweep -- with the converted data's histograms
    accumulating in the plots as it goes."""

    def __init__(self, parent, plot_shared_vars=None, acq_shared_vars=None, **kwargs):
        super().__init__(parent, **kwargs)

        # Starts out pointing at wherever the acquisition tabs last saved to.
        acq_folder_var = (acq_shared_vars or {}).get("save_folder_var")
        default_source = acq_folder_var.get() if acq_folder_var is not None else ""
        self.source_folder_var = app_settings.persistent_var(
            self, tk.StringVar, "analysis.source_folder", default_source
        )
        self.output_folder_var = app_settings.persistent_var(self, tk.StringVar, "analysis.output_folder", "")
        self.timewalk_enabled_var = app_settings.persistent_var(
            self, tk.BooleanVar, "analysis.timewalk_enabled", False
        )
        self.timewalk_path_var = app_settings.persistent_var(
            self, tk.StringVar, "analysis.timewalk_path", str(DEFAULT_CORRECTION_PATH)
        )
        self._plot_shared_vars = plot_shared_vars

        self.status_var = tk.StringVar(self, value="Idle")
        self._progress_var = tk.DoubleVar(self, value=0.0)
        self.eta_var = tk.StringVar(self, value="--")

        self.plan_cv4_var = tk.StringVar(self, value="--")
        self.plan_raw_var = tk.StringVar(self, value="--")
        self.plan_list_var = tk.StringVar(self, value="")
        self.files_var = tk.StringVar(self, value="--")
        self.failed_var = tk.StringVar(self, value="--")
        self.analysis_speed_var = tk.StringVar(self, value="--")
        self.pulses_var = tk.StringVar(self, value="--")
        self.clusters_var = tk.StringVar(self, value="--")
        self.etof_var = tk.StringVar(self, value="--")
        self.itof_var = tk.StringVar(self, value="--")
        self.cv4_path_var = tk.StringVar(self, value="--")

        self._stop_event = threading.Event()
        self._queue: Queue = Queue()
        self._worker = None
        self._accumulated = None
        self._total_files = 0
        self._job_count = 0
        self._job_index = 0
        self._job_name = ""
        self._failed_files = 0

        self._build_ui()
        self._poll_queue()

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
            empty_text="No data — press Start to convert",
        )
        self._plot_panel.grid(row=0, column=0, sticky="nsew")

    def _build_sidebar(self, sidebar):
        sidebar.rowconfigure(5, weight=1)

        ui_style.build_button_bar(
            sidebar, 0, [("Start", self.start), ("Stop", self.stop), ("Scan Folder", self.scan)],
        )
        self._status_block = ui_style.StatusBlock(
            sidebar, 1, self.status_var, progress_var=self._progress_var, eta_var=self.eta_var
        )

        params = ui_style.build_section(sidebar, 2, "Conversion parameters")
        ui_style.add_form_row(
            params, 0, "Raw data folder:",
            ui_style.build_path_field(params, self.source_folder_var, self._browse_source),
        )
        ui_style.add_form_row(
            params, 1, "Output folder:",
            ui_style.build_path_field(params, self.output_folder_var, self._browse_output),
        )
        ui_style.add_note(
            params, 2,
            "Pick a run folder (or its raw folder) from any acquisition tab. Leave the output "
            "folder blank to write the cv4 file(s) into the run folder.",
        )
        ttk.Checkbutton(
            params, text="Apply timewalk correction before clustering", variable=self.timewalk_enabled_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ui_style.build_path_field(params, self.timewalk_path_var, self._browse_timewalk).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(2, 4)
        )

        found = ui_style.build_section(sidebar, 3, "Detected files")
        list_row = ui_style.add_stat_rows(found, [
            ("cv4 files to write:", self.plan_cv4_var),
            ("Raw files:", self.plan_raw_var),
        ])
        ui_style.add_note(found, list_row, textvariable=self.plan_list_var, pady=0)

        totals = ui_style.build_section(sidebar, 4, "Totals", pady=0)
        cv4_row = ui_style.add_stat_rows(totals, [
            ("Files processed:", self.files_var),
            ("Failed files:", self.failed_var),
            ("Analysis speed:", self.analysis_speed_var),
            ("Pulses:", self.pulses_var),
            ("Clusters:", self.clusters_var),
            ("e-ToF events:", self.etof_var),
            ("i-ToF events:", self.itof_var),
        ])
        ttk.Label(totals, text="cv4 file:", font=ui_style.SMALL_FONT).grid(
            row=cv4_row, column=0, sticky="nw", pady=(6, 0)
        )
        ui_style.add_note(totals, cv4_row + 1, textvariable=self.cv4_path_var, pady=0)

    def _browse_source(self):
        chosen = filedialog.askdirectory(
            initialdir=self.source_folder_var.get().strip() or ".",
            title="Choose raw data folder",
            parent=self,
        )
        if chosen:
            self.source_folder_var.set(chosen)
            self.scan()

    def _browse_output(self):
        chosen = filedialog.askdirectory(
            initialdir=self.output_folder_var.get().strip() or self.source_folder_var.get().strip() or ".",
            title="Choose output folder",
            parent=self,
        )
        if chosen:
            self.output_folder_var.set(chosen)
            self.scan()

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
        # The next converted file starts a fresh total (see
        # raw_conversion.accumulate_plot_result); until then, redraw what's there.
        self._plot_panel.update_plots(self._current_data())

    def _current_data(self):
        return self._accumulated if self._accumulated is not None else self._plot_panel.empty_plot_data()

    # ---- scan / start / stop ------------------------------------------------

    def is_running(self):
        return bool(self._worker and self._worker.is_alive())

    def scan(self):
        """Work out (and show) what Start would convert. Returns the jobs,
        or None if the folder is missing or has no raw files."""
        folder_text = self.source_folder_var.get().strip()
        if not folder_text:
            self._show_plan([])
            self.status_var.set("Raw data folder is required.")
            return None
        folder = pathlib.Path(folder_text)
        if not folder.is_dir():
            self._show_plan([])
            self.status_var.set(f"Raw data folder not found: {folder}")
            return None
        try:
            jobs = raw_conversion.plan_conversion(folder, self.output_folder_var.get().strip() or None)
        except OSError as exc:
            self._show_plan([])
            self.status_var.set(f"Error: could not read {folder}: {exc}")
            return None
        self._show_plan(jobs)
        if not jobs:
            self.status_var.set(f"No raw .tpx3 files found in {folder}.")
            return None
        if not self.is_running():
            self.status_var.set(f"Ready to convert {len(jobs)} cv4 file(s).")
        return jobs

    def _show_plan(self, jobs):
        if not jobs:
            self.plan_cv4_var.set("--")
            self.plan_raw_var.set("--")
            self.plan_list_var.set("")
            return
        self.plan_cv4_var.set(str(len(jobs)))
        self.plan_raw_var.set(str(sum(len(job.raw_files) for job in jobs)))
        lines = [f"{job.cv4_path.name} ({len(job.raw_files)} raw)" for job in jobs[:20]]
        if len(jobs) > 20:
            lines.append(f"... and {len(jobs) - 20} more")
        self.plan_list_var.set("\n".join(lines))

    def start(self):
        if self.is_running():
            return

        if self.timewalk_enabled_var.get():
            try:
                timewalk = TimewalkCorrection.load(self.timewalk_path_var.get().strip())
            except Exception as exc:
                self.status_var.set(f"Could not load timewalk correction: {exc}")
                return
        else:
            timewalk = None

        jobs = self.scan()
        if not jobs:
            return

        # A leftover .partial.cv4 has to go too: Cv4Writer appends to an
        # existing file rather than truncating it.
        existing = [
            job for job in jobs
            if job.cv4_path.exists() or cv4_writer.partial_path_for(job.cv4_path).exists()
        ]
        if existing:
            names = "\n".join(job.cv4_path.name for job in existing[:10])
            if len(existing) > 10:
                names += f"\n... and {len(existing) - 10} more"
            if not messagebox.askyesno(
                "Files exist",
                f"{len(existing)} cv4 file(s) already exist in {existing[0].cv4_path.parent}:\n{names}\n\n"
                "Overwrite them?",
                parent=self,
            ):
                self.status_var.set("Start canceled.")
                return
            try:
                for job in existing:
                    for path in (job.cv4_path, cv4_writer.partial_path_for(job.cv4_path)):
                        if path.exists():
                            path.unlink()
            except OSError as exc:
                self.status_var.set(f"Error: could not remove an existing cv4 file: {exc}")
                return

        try:
            jobs[0].cv4_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.status_var.set(f"Error: could not create output folder: {exc}")
            return

        converted_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for job in jobs:
            job.metadata["Converted (UTC)"] = converted_at
            if timewalk is not None:
                job.metadata["Timewalk Correction File"] = self.timewalk_path_var.get().strip()

        self._stop_event.clear()
        self._accumulated = None
        self._plot_panel.update_plots(self._current_data())
        self._total_files = sum(len(job.raw_files) for job in jobs)
        self._job_count = len(jobs)
        self._failed_files = 0
        self.files_var.set(f"0/{self._total_files}")
        self.failed_var.set("0")
        for var in (self.analysis_speed_var, self.pulses_var, self.clusters_var, self.etof_var, self.itof_var):
            var.set("--")
        self._progress_var.set(0.0)
        self.eta_var.set("Estimating...")
        self.status_var.set(f"Processing {self._total_files} raw file(s) into {len(jobs)} cv4 file(s)...")

        self._worker = threading.Thread(target=self._worker_loop, args=(jobs, timewalk), daemon=True)
        self._worker.start()

    def stop(self):
        """Stops after the file currently being processed; any cv4 that
        didn't get all of its raw files keeps the .partial.cv4 name."""
        if not self.is_running():
            return
        self._stop_event.set()
        self.status_var.set("Stopping after the current file...")

    # ---- worker thread ------------------------------------------------------

    def _worker_loop(self, jobs, timewalk):
        start = time.time()
        files_done = 0
        results = []

        def on_file(path, frame, error):
            nonlocal files_done
            files_done += 1
            message = {"files_done": files_done, "elapsed": time.time() - start}
            if error is not None:
                message["error"] = f"Processing {path.name} failed: {error}"
            else:
                message["plot_result"] = frame["plot_result"]
            self._queue.put(message)

        try:
            for job_index, job in enumerate(jobs, start=1):
                if self._stop_event.is_set():
                    break
                self._queue.put({"job_started": job_index, "cv4_path": str(job.cv4_path)})
                final_path, processed = raw_conversion.convert_job(
                    job, self._plot_panel.hist_snapshot, timewalk, self._stop_event, on_file
                )
                results.append((final_path, processed, len(job.raw_files)))
                self._queue.put({"job_done": str(final_path)})
        except Exception as exc:
            self._queue.put({"fatal": str(exc)})
        finally:
            self._queue.put({
                "done": True,
                "results": results,
                "files_done": files_done,
                "stopped": self._stop_event.is_set(),
            })

    # ---- main-thread queue handling -----------------------------------------

    def _poll_queue(self):
        try:
            while True:
                self._apply_result(self._queue.get_nowait())
        except Empty:
            pass
        self.after(200, self._poll_queue)

    def _apply_result(self, result):
        if result.get("done"):
            self._on_done(result)
            return

        if "fatal" in result:
            self.status_var.set(f"Error: {result['fatal']}")
            return

        if "job_started" in result:
            self._job_index = result["job_started"]
            self._job_name = pathlib.Path(result["cv4_path"]).name
            self.cv4_path_var.set(result["cv4_path"])
            return

        if "job_done" in result:
            self.cv4_path_var.set(result["job_done"])
            return

        files_done = result["files_done"]
        self.files_var.set(f"{files_done}/{self._total_files}")
        frac = min(1.0, files_done / self._total_files) if self._total_files else 1.0
        self._progress_var.set(frac * 100.0)
        elapsed = result.get("elapsed")
        if elapsed is not None and 0 < frac < 1:
            self.eta_var.set(time_estimate.format_eta(elapsed / frac - elapsed))

        if "error" in result:
            self._failed_files += 1
            self.failed_var.set(str(self._failed_files))
            self.status_var.set(f"Error: {result['error']}")
            return

        if not self._stop_event.is_set():
            self.status_var.set(
                f"Processing {self._job_name} ({self._job_index}/{self._job_count}), "
                f"file {files_done}/{self._total_files}..."
            )

        self._accumulated = raw_conversion.accumulate_plot_result(self._accumulated, result["plot_result"])
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

    def _on_done(self, result):
        self.eta_var.set("--")
        results = result["results"]
        partial = sum(1 for _, processed, total in results if processed < total)
        converted = f"{result['files_done']}/{self._total_files} raw file(s) processed"
        if result["stopped"]:
            self.status_var.set(
                f"Stopped. {converted}; {partial} unfinished cv4 file(s) keep the .partial.cv4 name."
            )
        elif self._failed_files:
            self.status_var.set(
                f"Finished with {self._failed_files} failed file(s). {converted}; "
                f"{partial} cv4 file(s) keep the .partial.cv4 name."
            )
        elif len(results) < self._job_count:
            # A fatal error already set the status; leave it showing.
            return
        else:
            self._progress_var.set(100.0)
            self.status_var.set(f"Finished. Wrote {len(results)} cv4 file(s) from {self._total_files} raw file(s).")


class PlaceholderInterface(ttk.Frame):
    """Stand-in for an Analysis tab that hasn't been built yet."""

    def __init__(self, parent, name, **kwargs):
        super().__init__(parent, **kwargs)
        page = ui_style.build_page_layout(self)
        ui_style.make_wrapping_label(page, text=f"{name} isn't implemented yet.").grid(
            row=0, column=0, sticky="ew", padx=12, pady=12
        )
