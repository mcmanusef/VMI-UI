import json
import pathlib
import shutil
import time
from datetime import date
import tkinter as tk
from tkinter import ttk

import requests

import serval_client
import shared_state
import time_estimate


class CollectionInterface(ttk.Frame):
    def __init__(self, parent, server_var=None, acq_shared_vars=None, coordinator=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._server_var = server_var
        self._fallback_server_var = tk.StringVar(self, value="http://localhost:8080")
        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)

        # Shared live with the Monitored Acquisition tab (see shared_state.py)
        # when acq_shared_vars is given; otherwise falls back to independent,
        # unshared variables.
        acq = acq_shared_vars or {}
        self.frame_time_var = acq.get("frame_time_var") or tk.StringVar(self, value="1")
        self.run_duration_var = acq.get("run_duration_var") or tk.StringVar(self, value="10")
        self.save_folder_var = acq.get("save_folder_var") or tk.StringVar(self, value=self._default_save_folder())

        self.target_var = acq.get("target_var") or tk.StringVar(self)
        self.target_pressure_var = acq.get("target_pressure_var") or tk.StringVar(self)
        self.background_pressure_var = acq.get("background_pressure_var") or tk.StringVar(self)
        self.power_var = acq.get("power_var") or tk.StringVar(self)
        self.spot_size_var = acq.get("spot_size_var") or tk.StringVar(self)
        self.polarization_var = acq.get("polarization_var") or tk.StringVar(self)
        self.wavelength_var = acq.get("wavelength_var") or tk.StringVar(self)

        self.status_var = tk.StringVar(self, value="Idle")
        self.eta_var = tk.StringVar(self, value="--")
        self._stop_after_id = None
        self._batch_after_id = None
        self._batch_state = None
        self._progress_after_id = None
        self._progress_start_time = None
        self._progress_duration = None
        self._progress_var = tk.DoubleVar(self, value=0.0)

        self._build_ui()

    def _default_save_folder(self):
        return rf"C:\DATA\{date.today().strftime('%Y%m%d')}\test"

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Label(self, text="Collection parameters", font=("Segoe UI", 12, "bold"))
        header.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        form = ttk.Frame(self)
        form.grid(row=1, column=0, sticky="ew", padx=10)
        form.columnconfigure(1, weight=1)

        self._add_row(form, 0, "Frame time (s):", ttk.Entry(form, textvariable=self.frame_time_var))
        self._add_row(form, 1, "Run duration (s):", ttk.Entry(form, textvariable=self.run_duration_var))
        self._add_row(form, 2, "Save folder:", ttk.Entry(form, textvariable=self.save_folder_var))

        meta = ttk.LabelFrame(self, text="Metadata")
        meta.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        meta.columnconfigure(1, weight=1)

        self._add_row(meta, 0, "Target:", ttk.Entry(meta, textvariable=self.target_var))
        self._add_row(meta, 1, "Target Pressure:", ttk.Entry(meta, textvariable=self.target_pressure_var))
        self._add_row(meta, 2, "Background Pressure:", ttk.Entry(meta, textvariable=self.background_pressure_var))
        self._add_row(meta, 3, "Power:", ttk.Entry(meta, textvariable=self.power_var))
        self._add_row(meta, 4, "Spot Size:", ttk.Entry(meta, textvariable=self.spot_size_var))
        self._add_row(meta, 5, "Polarization:", ttk.Entry(meta, textvariable=self.polarization_var))
        self._add_row(meta, 6, "Wavelength:", ttk.Entry(meta, textvariable=self.wavelength_var))

        ttk.Label(meta, text="Notes:").grid(row=7, column=0, sticky="nw", padx=(0, 8), pady=4)
        self.notes = tk.Text(meta, wrap="word", height=5)
        self.notes.grid(row=7, column=1, sticky="nsew", pady=4)
        meta.rowconfigure(7, weight=1)
        shared_state.wire_notes_widget(self.notes)

        buttons = ttk.Frame(self)
        buttons.grid(row=3, column=0, sticky="w", padx=10, pady=(0, 10))
        ttk.Button(buttons, text="Start Run", command=self.start_run).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Stop Run", command=self.stop_run).grid(row=0, column=1)

        status = ttk.Frame(self)
        status.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
        status.columnconfigure(1, weight=1)
        ttk.Label(status, text="Status:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=1, sticky="w")
        self._progress = ttk.Progressbar(
            status,
            variable=self._progress_var,
            maximum=100.0,
            mode="determinate",
            length=240,
        )
        self._progress.grid(row=0, column=2, sticky="e", padx=(12, 0))
        ttk.Label(status, textvariable=self.eta_var, font=("Segoe UI", 8)).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )

    def _add_row(self, parent, row, label, widget):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        widget.grid(row=row, column=1, sticky="ew", pady=4)

    def _server_url(self):
        url = ""
        if self._server_var is not None:
            url = self._server_var.get().strip()
        if not url:
            url = self._fallback_server_var.get().strip() or "http://localhost:8080"
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        return url

    def _ensure_folder(self, folder):
        path = pathlib.Path(folder)
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"Save path exists and is not a folder: {path}")
            choice = self._prompt_existing_folder(path)
            if choice != "continue" and choice != "overwrite":
                return None
            if choice == "overwrite":
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _prompt_existing_folder(self, path):
        result = {"choice": None}

        dialog = tk.Toplevel(self)
        dialog.title("Folder exists")
        dialog.resizable(False, False)
        parent = self.winfo_toplevel()
        dialog.transient(parent)
        dialog.grab_set()

        message = (
            "The save folder already exists:\n"
            f"{path}\n\n"
            "Continue: keep the folder and overwrite metadata.\n"
            "Overwrite: delete the folder and recreate it.\n"
            "Stop: cancel the run."
        )
        ttk.Label(dialog, text=message, justify="left").grid(
            row=0, column=0, columnspan=3, padx=12, pady=(12, 8), sticky="w"
        )

        def choose(option):
            result["choice"] = option
            dialog.destroy()

        ttk.Button(dialog, text="Continue", command=lambda: choose("continue")).grid(
            row=1, column=0, padx=8, pady=(0, 12)
        )
        ttk.Button(dialog, text="Overwrite", command=lambda: choose("overwrite")).grid(
            row=1, column=1, padx=8, pady=(0, 12)
        )
        ttk.Button(dialog, text="Stop", command=lambda: choose("stop")).grid(
            row=1, column=2, padx=8, pady=(0, 12)
        )

        dialog.protocol("WM_DELETE_WINDOW", lambda: choose("stop"))
        dialog.update_idletasks()
        parent.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - dialog.winfo_width()) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - dialog.winfo_height()) // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.wait_window()
        return result["choice"]

    def _write_metadata(self, folder):
        metadata = {
            "Target": self.target_var.get().strip(),
            "Target Pressure": self.target_pressure_var.get().strip(),
            "Background Pressure": self.background_pressure_var.get().strip(),
            "Power": self.power_var.get().strip(),
            "Spot Size": self.spot_size_var.get().strip(),
            "Polarization": self.polarization_var.get().strip(),
            "Wavelength": self.wavelength_var.get().strip(),
            "Notes": self.notes.get("1.0", "end").strip(),
            "Frame Time (s)": self.frame_time_var.get().strip(),
            "Run Duration (s)": self.run_duration_var.get().strip(),
            "Save Folder": str(folder),
        }
        path = folder / "run_metadata.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _configure_detector(self, server, frame_time, n_triggers):
        config = json.loads(serval_client.SESSION.get(f"{server}/detector/config").text)
        config["TriggerMode"] = "CONTINUOUS"
        config["ExposureTime"] = frame_time
        config["TriggerPeriod"] = frame_time
        config["nTriggers"] = n_triggers
        serval_client.SESSION.put(f"{server}/detector/config", data=json.dumps(config))

    def _configure_destination(self, server, folder):
        destination = {
            "Raw": [
                {
                    "Base": folder.resolve().as_uri(),
                    "FilePattern": "raw%Hms_",
                    # Undocumented field; "SINGLE_FILE" (the default) would
                    # bundle every frame from this run into one growing
                    # file since nTriggers can be > 1 here. "FRAME" splits
                    # into one file per frame, matching every other tab.
                    "SplitStrategy": "FRAME",
                }
            ]
        }
        serval_client.SESSION.put(f"{server}/server/destination", data=json.dumps(destination))

    def is_running(self):
        return self._stop_after_id is not None

    def start_run(self):
        if self._coordinator is not None:
            self._coordinator.stop_others(self)

        try:
            server = self._server_url()
            frame_time = float(self.frame_time_var.get())
            run_duration = float(self.run_duration_var.get())
            if frame_time <= 0 or run_duration <= 0:
                raise ValueError("Frame time and run duration must be positive.")

            folder = self._ensure_folder(self.save_folder_var.get().strip())
            if folder is None:
                self.status_var.set("Run cancelled.")
                return

            # Total frame count for the whole run, same as before. No
            # single start_measurement() call is left running for the whole
            # (potentially long) run_duration though: each batch is capped
            # to at most max_frames_per_batch(frame_time) frames (a
            # minute/frame_time worth); once a batch's frames are in,
            # _next_batch reconfigures nTriggers and restarts for the next
            # one, on a timer sized to that batch's own duration. This
            # timer only ever configures/restarts -- the overall
            # self._stop_after_id below is still what actually ends the run
            # at run_duration, same as before.
            n_triggers = max(1, int(run_duration / frame_time))
            batch_cap = serval_client.max_frames_per_batch(frame_time)
            first_batch = min(batch_cap, n_triggers)
            self._configure_detector(server, frame_time, first_batch)
            self._configure_destination(server, folder)
            self._write_metadata(folder)

            resp = serval_client.SESSION.get(f"{server}/measurement/start")
            self.status_var.set(resp.text.strip() or "Run started.")

            self._batch_state = {
                "server": server,
                "frame_time": frame_time,
                "batch_cap": batch_cap,
                "remaining": n_triggers - first_batch,
            }
            if self._batch_after_id is not None:
                self.after_cancel(self._batch_after_id)
                self._batch_after_id = None
            if self._batch_state["remaining"] > 0:
                self._batch_after_id = self.after(int(first_batch * frame_time * 1000), self._next_batch)

            if self._stop_after_id is not None:
                self.after_cancel(self._stop_after_id)
            self._stop_after_id = self.after(int(run_duration * 1000), self.stop_run)
            self._start_progress(run_duration)
        except Exception as exc:
            self.status_var.set(f"Start failed: {type(exc).__name__}: {exc}")

    def _next_batch(self):
        """Fires when the current batch's frames are expected to be used
        up -- reconfigures nTriggers for the next batch and restarts, so no
        single start_measurement() call runs for more than a minute. Purely
        a Serval-side detail; the run's actual stop time is still governed
        by self._stop_after_id / run_duration, unaffected by this."""
        self._batch_after_id = None
        state = self._batch_state
        if state is None or state["remaining"] <= 0:
            return
        try:
            batch = min(state["batch_cap"], state["remaining"])
            server = state["server"]
            serval_client.wait_for_measurement_idle(server, timeout=10.0)
            self._configure_detector(server, state["frame_time"], batch)
            serval_client.SESSION.get(f"{server}/measurement/start")
        except Exception as exc:
            self.status_var.set(f"Batch restart failed: {type(exc).__name__}: {exc}")
            return
        state["remaining"] -= batch
        if state["remaining"] > 0:
            self._batch_after_id = self.after(int(batch * state["frame_time"] * 1000), self._next_batch)

    def stop(self):
        """Alias so the TabCoordinator can stop any tab uniformly."""
        self.stop_run()

    def stop_run(self):
        if self._stop_after_id is not None:
            self.after_cancel(self._stop_after_id)
            self._stop_after_id = None
        if self._batch_after_id is not None:
            self.after_cancel(self._batch_after_id)
            self._batch_after_id = None
        self._batch_state = None
        self._stop_progress()
        try:
            resp = serval_client.SESSION.get(f"{self._server_url()}/measurement/stop")
            self.status_var.set(resp.text.strip() or "Run stopped.")
        except Exception as exc:
            self.status_var.set(f"Stop failed: {type(exc).__name__}: {exc}")

    def _start_progress(self, run_duration):
        self._progress_duration = max(0.001, float(run_duration))
        self._progress_start_time = time.perf_counter()
        self._progress_var.set(0.0)
        if self._progress_after_id is not None:
            self.after_cancel(self._progress_after_id)
        self._progress_after_id = self.after(100, self._update_progress)

    def _stop_progress(self):
        if self._progress_after_id is not None:
            self.after_cancel(self._progress_after_id)
            self._progress_after_id = None
        self._progress_var.set(0.0)
        self._progress_start_time = None
        self._progress_duration = None
        self.eta_var.set("--")

    def _update_progress(self):
        if self._progress_start_time is None or self._progress_duration is None:
            return
        elapsed = time.perf_counter() - self._progress_start_time
        frac = min(1.0, max(0.0, elapsed / self._progress_duration))
        self._progress_var.set(frac * 100.0)
        self.eta_var.set(time_estimate.format_eta(self._progress_duration - elapsed))
        if frac >= 1.0:
            self._progress_after_id = None
            self.eta_var.set("--")
            return
        self._progress_after_id = self.after(200, self._update_progress)
