"""Analysis tab: converts a folder of raw TPX3 files straight into HDF5
files, one per raw file, with none of Analysis > conversion's clustering,
pulse-grouping or timewalk correction -- just each raw file's pixel hits
and TDC markers (PicoTDC and FPGA TDC alike). See raw_to_h5.py.
"""
import pathlib
import threading
import time
from queue import Queue, Empty

import qtk as tk
from qtk import ttk, messagebox, filedialog

import app_settings
import raw_to_h5
import time_estimate
import ui_style


class RawToH5Interface(ttk.Frame):
    """Raw data folder in, one .h5 file per raw .tpx3 file out, with just
    the raw pixel hits and TDC markers."""

    def __init__(self, parent, acq_shared_vars=None, **kwargs):
        super().__init__(parent, **kwargs)

        # Starts out pointing at wherever the acquisition tabs last saved to.
        acq_folder_var = (acq_shared_vars or {}).get("save_folder_var")
        default_source = acq_folder_var.get() if acq_folder_var is not None else ""
        self.source_folder_var = app_settings.persistent_var(
            self, tk.StringVar, "raw_to_h5.source_folder", default_source
        )
        self.output_folder_var = app_settings.persistent_var(self, tk.StringVar, "raw_to_h5.output_folder", "")

        self.status_var = tk.StringVar(self, value="Idle")
        self._progress_var = tk.DoubleVar(self, value=0.0)
        self.eta_var = tk.StringVar(self, value="--")

        self.plan_files_var = tk.StringVar(self, value="--")
        self.files_var = tk.StringVar(self, value="--")
        self.failed_var = tk.StringVar(self, value="--")
        self.speed_var = tk.StringVar(self, value="--")
        self.pixels_var = tk.StringVar(self, value="--")
        self.tdcs_var = tk.StringVar(self, value="--")

        self._stop_event = threading.Event()
        self._queue: Queue = Queue()
        self._worker = None
        self._jobs = []
        self._total_pixels = 0
        self._total_tdcs = 0
        self._failed_files = 0

        self._build_ui()
        self._poll_queue()

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.rowconfigure(0, weight=1)

        self._build_sidebar(sidebar)

        card = ui_style.build_card(main, 0, "Converted files")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(0, weight=1)
        self._tree = ttk.Treeview(card, columns=("raw", "h5", "pixels", "tdcs"), show="headings")
        for col, text, width in (
            ("raw", "Raw file", 200), ("h5", "h5 file", 200),
            ("pixels", "Pixels", 90), ("tdcs", "TDCs", 90),
        ):
            self._tree.heading(col, text=text)
            self._tree.column(col, width=width)
        self._tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _build_sidebar(self, sidebar):
        sidebar.rowconfigure(4, weight=1)

        ui_style.build_button_bar(
            sidebar, 0, [("Start", self.start), ("Stop", self.stop), ("Scan Folder", self.scan)],
        )
        ui_style.StatusBlock(
            sidebar, 1, self.status_var, progress_var=self._progress_var, eta_var=self.eta_var
        )

        params = ui_style.build_section(sidebar, 2, "Conversion parameters")
        ui_style.add_form_row(params, 0, "Raw data (file or folder):", self._build_source_field(params))
        ui_style.add_form_row(
            params, 1, "Output folder:",
            ui_style.build_path_field(params, self.output_folder_var, self._browse_output),
        )
        ui_style.add_note(
            params, 2,
            "Pick a single raw .tpx3 file, or a run folder (or its raw folder) from any acquisition "
            "tab. Leave the output folder blank to write each raw file's .h5 next to it. Each h5 "
            "holds pixel hits (x, y, toa, tot, pixel_chip) and TDC hits (tdc_time, tdc_type, "
            "tdc_channel, tdc_edge, tdc_chip) -- PicoTDC and FPGA TDC alike -- with no clustering, "
            "pulse-grouping, or timewalk correction.",
        )

        totals = ui_style.build_section(sidebar, 3, "Totals", pady=0)
        ui_style.add_stat_rows(totals, [
            ("Raw files found:", self.plan_files_var),
            ("Files processed:", self.files_var),
            ("Failed files:", self.failed_var),
            ("Conversion speed:", self.speed_var),
            ("Pixels written:", self.pixels_var),
            ("TDCs written:", self.tdcs_var),
        ])

    def _build_source_field(self, parent):
        """Like ui_style.build_path_field, but with two browse buttons --
        the source can be a single raw file or a whole folder."""
        row = ttk.Frame(parent)
        row.columnconfigure(0, weight=1)
        entry = ttk.Entry(row, textvariable=self.source_folder_var)
        entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(row, text="File...", width=6, command=self._browse_source_file).grid(
            row=0, column=1, padx=(4, 0)
        )
        ttk.Button(row, text="Folder...", width=8, command=self._browse_source_folder).grid(
            row=0, column=2, padx=(4, 0)
        )
        ui_style.wire_path_entry(entry, self.source_folder_var)
        return row

    def _browse_source_file(self):
        chosen = filedialog.askopenfilename(
            initialdir=self.source_folder_var.get().strip() or ".",
            title="Choose a raw .tpx3 file",
            filetypes=[("TPX3", "*.tpx3")],
            parent=self,
        )
        if chosen:
            self.source_folder_var.set(chosen)
            self.scan()

    def _browse_source_folder(self):
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

    # ---- scan / start / stop ------------------------------------------------

    def is_running(self):
        return bool(self._worker and self._worker.is_alive())

    def scan(self):
        """Work out (and show) what Start would convert. Returns the jobs,
        or None if the source is missing or has no raw files."""
        source_text = self.source_folder_var.get().strip()
        if not source_text:
            self._show_plan([])
            self.status_var.set("Raw data file or folder is required.")
            return None
        source = pathlib.Path(source_text)
        if not source.is_file() and not source.is_dir():
            self._show_plan([])
            self.status_var.set(f"Raw data file or folder not found: {source}")
            return None
        try:
            jobs = raw_to_h5.plan_conversion(source, self.output_folder_var.get().strip() or None)
        except OSError as exc:
            self._show_plan([])
            self.status_var.set(f"Error: could not read {source}: {exc}")
            return None
        self._show_plan(jobs)
        if not jobs:
            self.status_var.set(f"No raw .tpx3 files found in {source}.")
            return None
        if not self.is_running():
            self.status_var.set(f"Ready to convert {len(jobs)} raw file(s) to h5.")
        return jobs

    def _show_plan(self, jobs):
        self._jobs = jobs
        self.plan_files_var.set(str(len(jobs)) if jobs else "--")

    def start(self):
        if self.is_running():
            return

        jobs = self.scan()
        if not jobs:
            return

        existing = [job for job in jobs if job.h5_path.exists()]
        if existing:
            names = "\n".join(job.h5_path.name for job in existing[:10])
            if len(existing) > 10:
                names += f"\n... and {len(existing) - 10} more"
            if not messagebox.askyesno(
                "Files exist",
                f"{len(existing)} h5 file(s) already exist:\n{names}\n\nOverwrite them?",
                parent=self,
            ):
                self.status_var.set("Start canceled.")
                return

        self._tree.delete(*self._tree.get_children())
        self._stop_event.clear()
        self._total_pixels = 0
        self._total_tdcs = 0
        self._failed_files = 0
        self.files_var.set(f"0/{len(jobs)}")
        self.failed_var.set("0")
        self.pixels_var.set("0")
        self.tdcs_var.set("0")
        self.speed_var.set("--")
        self._progress_var.set(0.0)
        self.eta_var.set("Estimating...")
        self.status_var.set(f"Converting {len(jobs)} raw file(s) to h5...")

        self._worker = threading.Thread(target=self._worker_loop, args=(jobs,), daemon=True)
        self._worker.start()

    def stop(self):
        """Stops after the file currently being converted."""
        if not self.is_running():
            return
        self._stop_event.set()
        self.status_var.set("Stopping after the current file...")

    # ---- worker thread ------------------------------------------------------

    def _worker_loop(self, jobs):
        start = time.time()
        files_done = 0
        for job in jobs:
            if self._stop_event.is_set():
                break
            try:
                counts = raw_to_h5.convert_file(job.raw_path, job.h5_path)
            except Exception as exc:
                files_done += 1
                self._queue.put({
                    "files_done": files_done, "total": len(jobs), "elapsed": time.time() - start,
                    "error": f"Converting {job.raw_path.name} failed: {exc}",
                })
                continue
            files_done += 1
            self._queue.put({
                "files_done": files_done, "total": len(jobs), "elapsed": time.time() - start,
                "job": job, "counts": counts,
            })
        self._queue.put({
            "done": True, "files_done": files_done, "total": len(jobs),
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

        files_done = result["files_done"]
        total = result["total"]
        self.files_var.set(f"{files_done}/{total}")
        frac = min(1.0, files_done / total) if total else 1.0
        self._progress_var.set(frac * 100.0)
        elapsed = result.get("elapsed")
        if elapsed:
            self.speed_var.set(f"{files_done / elapsed:.2f} files/s")
        if elapsed is not None and 0 < frac < 1:
            self.eta_var.set(time_estimate.format_eta(elapsed / frac - elapsed))

        if "error" in result:
            self._failed_files += 1
            self.failed_var.set(str(self._failed_files))
            self.status_var.set(f"Error: {result['error']}")
            return

        job = result["job"]
        counts = result["counts"]
        self._total_pixels += counts["pixels"]
        self._total_tdcs += counts["tdcs"]
        self.pixels_var.set(str(self._total_pixels))
        self.tdcs_var.set(str(self._total_tdcs))
        self._tree.insert("", "end", values=(
            job.raw_path.name, job.h5_path.name, counts["pixels"], counts["tdcs"],
        ))

        if not self._stop_event.is_set():
            self.status_var.set(f"Converting file {files_done}/{total}...")

    def _on_done(self, result):
        self.eta_var.set("--")
        if result["stopped"]:
            self.status_var.set(f"Stopped. {result['files_done']}/{result['total']} raw file(s) converted.")
        elif self._failed_files:
            self.status_var.set(
                f"Finished with {self._failed_files} failed file(s). "
                f"{result['files_done']}/{result['total']} raw file(s) converted."
            )
        else:
            self._progress_var.set(100.0)
            self.status_var.set(f"Finished. Converted {result['files_done']} raw file(s) to h5.")
