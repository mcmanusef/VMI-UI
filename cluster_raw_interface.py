"""Offline reprocessing: cluster raw .tpx3 file(s) (optionally applying a
time-walk correction first) and write the result to a single cv4 file.

Same decode -> [timewalk] -> cluster -> Cv4Writer.append_frame pipeline used
live by the Monitored Acquisition tab in the main VMI UI (acquisition_
interface.py), just driven from files already on disk instead of from a
running Serval measurement.
"""
import pathlib
import threading
import time
from queue import Queue, Empty

import qtk as tk
from qtk import ttk, filedialog, messagebox

import app_settings
import cv4_writer
from cv4_writer import Cv4Writer
from timewalk import DEFAULT_CORRECTION_PATH, TimewalkCorrection, apply_timewalk_correction
from tpx_processing import (
    decode_tpx3,
    sort_tdcs,
    group_pixels_by_pulse,
    group_times_relative,
    cluster_pixels_by_pulse,
)


class ClusterRawInterface(ttk.Frame):
    """Pick individual .tpx3 files or whole folders, cluster them (with an
    optional time-walk correction), and append the result to one output
    cv4 file."""

    def __init__(self, parent, coordinator=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)

        self._files: list[pathlib.Path] = []

        self.output_path_var = tk.StringVar(self, value=app_settings.get("cluster_raw.output_path", ""))
        self.cutoff_ns_var = tk.StringVar(self, value=app_settings.get("cluster_raw.cutoff_ns", "300"))
        self.max_packets_var = tk.StringVar(self, value=app_settings.get("cluster_raw.max_packets", ""))
        self.keep_going_var = tk.BooleanVar(self, value=app_settings.get("cluster_raw.keep_going", True))

        self.timewalk_enabled_var = tk.BooleanVar(self, value=app_settings.get("cluster_raw.timewalk_enabled", False))
        self.timewalk_path_var = tk.StringVar(
            self, value=app_settings.get("cluster_raw.timewalk_path", str(DEFAULT_CORRECTION_PATH))
        )

        self.file_count_var = tk.StringVar(self, value="No files selected.")
        self.status_var = tk.StringVar(self, value="Idle")
        self._progress_var = tk.DoubleVar(self, value=0.0)

        self._stop_event = threading.Event()
        self._queue: Queue = Queue()
        self._worker = None
        self._active_timewalk = None

        self._build_ui()
        self._poll_queue()

    # ---- UI construction ---------------------------------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=1)

        header = ttk.Label(self, text="Cluster raw files", font=("Segoe UI", 12, "bold"))
        header.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        picker = ttk.LabelFrame(self, text="Input .tpx3 files")
        picker.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        picker.columnconfigure(3, weight=1)

        ttk.Button(picker, text="Add Files...", command=self._add_files).grid(row=0, column=0, padx=(6, 6), pady=6)
        ttk.Button(picker, text="Add Folder...", command=self._add_folder).grid(row=0, column=1, padx=(0, 6), pady=6)
        ttk.Button(picker, text="Clear", command=self._clear_files).grid(row=0, column=2, padx=(0, 6), pady=6)
        ttk.Label(picker, textvariable=self.file_count_var).grid(row=0, column=3, sticky="w", padx=(6, 6))

        self._file_list = tk.Listbox(picker, height=6)
        self._file_list.grid(row=1, column=0, columnspan=4, sticky="ew", padx=6, pady=(0, 6))

        settings = ttk.LabelFrame(self, text="Processing settings")
        settings.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))
        settings.columnconfigure(5, weight=1)

        ttk.Label(settings, text="Output cv4 file:").grid(row=0, column=0, sticky="w", padx=(6, 4), pady=6)
        ttk.Entry(settings, textvariable=self.output_path_var, width=48).grid(
            row=0, column=1, columnspan=4, sticky="ew", padx=(0, 6), pady=6
        )
        ttk.Button(settings, text="Browse...", command=self._browse_output).grid(row=0, column=5, sticky="w", padx=(0, 6))

        ttk.Label(settings, text="Pulse cutoff (ns):").grid(row=1, column=0, sticky="w", padx=(6, 4), pady=4)
        ttk.Entry(settings, textvariable=self.cutoff_ns_var, width=10).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(settings, text="Max packets/file (blank = all):").grid(row=1, column=2, sticky="w", padx=(12, 4), pady=4)
        ttk.Entry(settings, textvariable=self.max_packets_var, width=10).grid(row=1, column=3, sticky="w", pady=4)

        ttk.Checkbutton(
            settings, text="Keep going after a file error", variable=self.keep_going_var
        ).grid(row=1, column=4, columnspan=2, sticky="w", padx=(12, 6), pady=4)

        ttk.Checkbutton(
            settings, text="Apply timewalk correction before clustering", variable=self.timewalk_enabled_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=(6, 4), pady=4)
        ttk.Entry(settings, textvariable=self.timewalk_path_var, width=40).grid(
            row=2, column=2, columnspan=3, sticky="ew", padx=(0, 6), pady=4
        )
        ttk.Button(settings, text="Browse...", command=self._browse_timewalk).grid(row=2, column=5, sticky="w", padx=(0, 6))

        run = ttk.Frame(self)
        run.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 6))
        ttk.Button(run, text="Start", command=self.start).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(run, text="Stop", command=self.stop).grid(row=0, column=1, padx=(0, 12))
        self._progress = ttk.Progressbar(run, variable=self._progress_var, maximum=100.0, length=240)
        self._progress.grid(row=0, column=2, padx=(0, 12))
        ttk.Label(run, textvariable=self.status_var).grid(row=0, column=3, sticky="w")

    # ---- file selection -------------------------------------------------

    def _add_files(self):
        chosen = filedialog.askopenfilenames(
            title="Add .tpx3 files", filetypes=[("Timepix3 raw", "*.tpx3"), ("All files", "*.*")], parent=self
        )
        if not chosen:
            return
        self._files.extend(pathlib.Path(p) for p in chosen)
        self._refresh_file_list()

    def _add_folder(self):
        chosen = filedialog.askdirectory(title="Add folder of .tpx3 files", parent=self)
        if not chosen:
            return
        found = sorted(pathlib.Path(chosen).rglob("*.tpx3"))
        if not found:
            messagebox.showinfo("No files found", f"No .tpx3 files under {chosen}", parent=self)
            return
        self._files.extend(found)
        self._refresh_file_list()

    def _clear_files(self):
        self._files = []
        self._refresh_file_list()

    def _refresh_file_list(self):
        # De-duplicate, keep a stable (sorted) order.
        seen = {}
        for p in self._files:
            seen[str(p.resolve())] = p
        self._files = sorted(seen.values(), key=lambda p: str(p))

        self._file_list.delete(0, "end")
        for p in self._files:
            self._file_list.insert("end", str(p))
        self.file_count_var.set(f"{len(self._files)} file(s) selected.")

    def _browse_output(self):
        initial = self.output_path_var.get().strip() or "output.cv4"
        chosen = filedialog.asksaveasfilename(
            title="Output cv4 file",
            defaultextension=".cv4",
            initialfile=pathlib.Path(initial).name,
            filetypes=[("cv4", "*.cv4")],
            parent=self,
        )
        if chosen:
            self.output_path_var.set(chosen)

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

    # ---- persistence / coordinator hooks --------------------------------

    def default_fields(self):
        return {
            "cluster_raw.output_path": self.output_path_var,
            "cluster_raw.cutoff_ns": self.cutoff_ns_var,
            "cluster_raw.max_packets": self.max_packets_var,
            "cluster_raw.keep_going": self.keep_going_var,
            "cluster_raw.timewalk_enabled": self.timewalk_enabled_var,
            "cluster_raw.timewalk_path": self.timewalk_path_var,
        }

    def is_running(self):
        return bool(self._worker and self._worker.is_alive())

    # ---- run / stop -------------------------------------------------------

    def start(self):
        if self._coordinator is not None:
            self._coordinator.stop_others(self)
        if self.is_running():
            self.stop()

        if not self._files:
            self.status_var.set("No input files selected.")
            return
        output_path = self.output_path_var.get().strip()
        if not output_path:
            self.status_var.set("Choose an output cv4 file.")
            return

        try:
            cutoff_ns = float(self.cutoff_ns_var.get())
        except ValueError:
            self.status_var.set("Invalid pulse cutoff.")
            return
        max_packets_text = self.max_packets_var.get().strip()
        try:
            max_packets = int(max_packets_text) if max_packets_text else None
        except ValueError:
            self.status_var.set("Invalid max packets.")
            return

        if self.timewalk_enabled_var.get():
            try:
                self._active_timewalk = TimewalkCorrection.load(self.timewalk_path_var.get().strip())
            except Exception as exc:
                self.status_var.set(f"Could not load timewalk correction: {exc}")
                return
        else:
            self._active_timewalk = None

        output = pathlib.Path(output_path)
        if output.exists() or cv4_writer.partial_path_for(output).exists():
            if not messagebox.askyesno(
                "Output exists",
                f"{output.name} (or a .partial version of it) already exists and new frames will "
                "be appended to it. Continue?",
                parent=self,
            ):
                self.status_var.set("Cancelled.")
                return

        self._stop_event.clear()
        self._progress_var.set(0.0)
        self.status_var.set(f"Processing 0/{len(self._files)}...")
        self._worker = threading.Thread(
            target=self._worker_loop,
            args=(list(self._files), output, cutoff_ns, max_packets, self.keep_going_var.get()),
            daemon=True,
        )
        self._worker.start()

    def stop(self):
        self._stop_event.set()
        self.status_var.set("Stopping...")
        if self._worker:
            self._worker.join(timeout=5)

    # ---- worker -----------------------------------------------------------

    def _worker_loop(self, files, output, cutoff_ns, max_packets, keep_going):
        partial_path = cv4_writer.partial_path_for(output)
        total = len(files)
        completed_naturally = False
        try:
            writer = Cv4Writer(partial_path)
        except Exception as exc:
            self._queue.put({"error": f"Could not open {partial_path}: {exc}", "done": True})
            return

        processed = 0
        errors = 0
        try:
            for idx, path in enumerate(files):
                if self._stop_event.is_set():
                    break
                try:
                    pixels, tdcs, _, _ = decode_tpx3(str(path), max_packets=max_packets)
                    if self._active_timewalk is not None:
                        pixels = apply_timewalk_correction(pixels, self._active_timewalk)
                    etof, itof, pulses = sort_tdcs(cutoff_ns, tdcs)

                    pulses_sorted = sorted(pulses)
                    pixels_by_pulse = group_pixels_by_pulse(pixels, pulses_sorted)
                    clusters_by_pulse, _, _ = cluster_pixels_by_pulse(pixels_by_pulse)
                    etof_by_pulse = group_times_relative(etof, pulses_sorted)
                    itof_by_pulse = group_times_relative(itof, pulses_sorted)

                    writer.append_frame(pulses_sorted, clusters_by_pulse, etof_by_pulse, itof_by_pulse)
                    writer.flush()
                    processed += 1
                except Exception as exc:
                    errors += 1
                    self._queue.put({"file_error": f"{path.name}: {exc}"})
                    if not keep_going:
                        break

                self._queue.put({
                    "progress": (idx + 1) / total if total else 1.0,
                    "processed": processed,
                    "errors": errors,
                    "total": total,
                })

            completed_naturally = not self._stop_event.is_set()
        except Exception as exc:
            self._queue.put({"error": str(exc)})
        finally:
            writer.set_attrs({
                "Files Selected": total,
                "Files Processed": processed,
                "Files Errored": errors,
                "Complete": completed_naturally,
                "Source": "cluster_raw_interface",
            })
            writer.close()
            final_path = cv4_writer.finalize_partial_path(partial_path, completed_naturally)
            self._queue.put({
                "done": True,
                "processed": processed,
                "errors": errors,
                "total": total,
                "final_path": str(final_path),
            })

    # ---- queue polling ------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                result = self._queue.get_nowait()
                self._apply_result(result)
        except Empty:
            pass
        self.after(200, self._poll_queue)

    def _apply_result(self, result):
        if "file_error" in result:
            self.status_var.set(f"Error: {result['file_error']}")
            return
        if "error" in result:
            self.status_var.set(f"Failed: {result['error']}")
            return
        if "progress" in result:
            self._progress_var.set(result["progress"] * 100.0)
            self.status_var.set(
                f"Processing {result['processed']}/{result['total']} "
                f"({result['errors']} error(s))..."
            )
        if result.get("done"):
            self._progress_var.set(100.0 if result.get("total") else 0.0)
            self.status_var.set(
                f"Done. Wrote {result['processed']}/{result['total']} file(s) "
                f"({result['errors']} error(s)) to {result.get('final_path', '')}"
            )
