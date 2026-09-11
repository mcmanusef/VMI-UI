import json
import pathlib
import tempfile
import time
import qtk as tk
from qtk import ttk

import requests

import serval_client


class ServalInterface(ttk.Frame):
    def __init__(self, parent, coordinator=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)

        defaults = self._load_defaults()

        self.server_var = tk.StringVar(self, value=defaults["server"])
        self.bpc_var = tk.StringVar(self, value=defaults["bpc_file"])
        self.dacs_var = tk.StringVar(self, value=defaults["dacs_file"])
        self.bias_voltage_var = tk.StringVar(self, value=str(defaults["bias_voltage"]))
        self.dest_var = tk.StringVar(self, value=defaults["destination"])
        self.file_pattern_var = tk.StringVar(self, value=defaults["file_pattern"])
        self.status_var = tk.StringVar(self, value="Idle")

        self._build_ui()
        self._schedule_dashboard_refresh()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        layout = ttk.Frame(self)
        layout.grid(row=0, column=0, sticky="nsew")
        layout.columnconfigure(0, weight=3)
        layout.columnconfigure(1, weight=2)
        layout.rowconfigure(1, weight=1)

        left = ttk.Frame(layout)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        header = ttk.Label(left, text="Serval configuration", font=("Segoe UI", 12, "bold"))
        header.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        form = ttk.Frame(left)
        form.grid(row=1, column=0, sticky="ew", padx=10)
        form.columnconfigure(1, weight=1)

        self._add_row(form, 0, "Server URL:", ttk.Entry(form, textvariable=self.server_var))
        self._add_row(form, 1, "BPC file:", ttk.Entry(form, textvariable=self.bpc_var))
        self._add_row(form, 2, "DACS file:", ttk.Entry(form, textvariable=self.dacs_var))
        self._add_row(form, 3, "Bias voltage:", ttk.Entry(form, textvariable=self.bias_voltage_var))

        buttons = ttk.Frame(form)
        buttons.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="Ping", command=self.ping).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Initialize", command=self.initialize).grid(row=0, column=1, padx=(0, 8))

        status = ttk.Frame(left)
        status.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        status.columnconfigure(1, weight=1)
        status.rowconfigure(1, weight=1)

        ttk.Label(status, text="Status:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=1, sticky="w")

        self.log = tk.Text(status, wrap="word", height=8)
        self.log.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

        right = ttk.Frame(layout)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(0, 10), pady=10)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="Dashboard", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        self.dashboard = tk.Text(right, wrap="word", height=12)
        self.dashboard.grid(row=1, column=0, sticky="nsew")
        self.dashboard.configure(state="disabled")

    def _add_row(self, parent, row, label, widget):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        widget.grid(row=row, column=1, sticky="ew", pady=4)

    def _load_defaults(self):
        defaults = {
            "server": "http://localhost:8080",
            "bpc_file": r"C:\SoPhy\pixelconfig_20240514.bpc",
            "dacs_file": r"C:\SoPhy\pixelconfig_20240514.bpc.dacs",
            "bias_voltage": 100,
            "destination": r"C:\serval_test",
            "file_pattern": "raw%Hms_",
        }

        path = pathlib.Path(__file__).with_name("serval_config.json")
        if not path.exists():
            return defaults

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return defaults

        for key in defaults:
            if key in data and data[key] is not None:
                defaults[key] = data[key]
        return defaults

    def _server_url(self):
        url = self.server_var.get().strip()
        if not url:
            raise ValueError("Server URL is required.")
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        return url

    def _log(self, message):
        self.log.insert("end", message + "\n")
        self.log.see("end")

    def _set_status(self, text):
        self.status_var.set(text)
        self._log(text)

    def _schedule_dashboard_refresh(self):
        self._refresh_dashboard()
        # Runs unconditionally for the app's entire lifetime regardless of
        # which tab is active. Now goes through serval_client.SESSION (a
        # shared, reused requests.Session) rather than a bare requests.get()
        # that opened/tore down a fresh TCP connection every call -- kept at
        # 2000ms since a live JSON dashboard dump doesn't need sub-second
        # refresh anyway, though this alone turned out not to explain the
        # destination-config timeouts (see serval_client.py's SESSION
        # comment for the current leading theory: proxy env vars).
        self.after(2000, self._schedule_dashboard_refresh)

    def _refresh_dashboard(self):
        try:
            resp = serval_client.SESSION.get(f"{self._server_url()}/dashboard", timeout=1)
            data = json.loads(resp.text)
            display = json.dumps(data, indent=2, sort_keys=True)
        except Exception as exc:
            display = f"Dashboard unavailable: {exc}"

        self.dashboard.configure(state="normal")
        self.dashboard.delete("1.0", "end")
        self.dashboard.insert("1.0", display)
        self.dashboard.configure(state="disabled")

    def ping(self):
        try:
            resp = serval_client.SESSION.get(self._server_url(), timeout=5)
            self._set_status(f"Ping ok ({resp.status_code})")
        except Exception as exc:
            self._set_status(f"Ping failed: {exc}")

    def is_running(self):
        # No background worker here -- start_measurement/initialize are
        # one-shot button actions, not a tracked long-running collection.
        return False

    def stop(self):
        self.stop_measurement()

    def initialize(self):
        if self._coordinator is not None:
            self._coordinator.stop_others(self)
        try:
            server = self._server_url()
            self._load_configs(server)
            self._configure_detector(server)
            self._configure_destination(server)
            self._initial_data_run(server)
            self._set_status("Initialization complete.")
        except Exception as exc:
            self._set_status(f"Initialization failed: {type(exc).__name__}: {exc}")

    def _load_configs(self, server):
        bpc_file = self.bpc_var.get().strip()
        dacs_file = self.dacs_var.get().strip()
        if bpc_file:
            resp = serval_client.SESSION.get(f"{server}/config/load?format=pixelconfig&file={bpc_file}")
            self._log(resp.text.strip())
        if dacs_file:
            resp = serval_client.SESSION.get(f"{server}/config/load?format=dacs&file={dacs_file}")
            self._log(resp.text.strip())

    def _configure_detector(self, server):
        bias_voltage = float(self.bias_voltage_var.get())

        config = json.loads(serval_client.SESSION.get(f"{server}/detector/config").text)
        config["BiasVoltage"] = bias_voltage
        config["BiasEnabled"] = True
        resp = serval_client.SESSION.put(f"{server}/detector/config", data=json.dumps(config))
        self._log(resp.text.strip())

    def _configure_destination(self, server):
        dest = self.dest_var.get().strip()
        if not dest:
            raise ValueError("Destination folder is required.")
        self._configure_destination_with(server, dest, self.file_pattern_var.get().strip())

    def _configure_destination_with(self, server, dest, pattern):
        destination = {
            "Raw": [
                {
                    "Base": pathlib.Path(dest).as_uri(),
                    "FilePattern": pattern or "raw%Hms_",
                }
            ],
        }
        resp = serval_client.SESSION.put(f"{server}/server/destination", data=json.dumps(destination))
        self._log(resp.text.strip())

    def _initial_data_run(self, server):
        temp_dir = tempfile.mkdtemp(prefix="serval_init_")
        self._log(f"Temporary run folder: {temp_dir}")
        self._configure_destination_with(server, temp_dir, "init%Hms_")
        resp = serval_client.SESSION.get(f"{server}/measurement/start")
        self._log(resp.text.strip() or "Measurement started for init run.")
        time.sleep(1)
        resp = serval_client.SESSION.get(f"{server}/measurement/stop")
        self._log(resp.text.strip() or "Measurement stopped for init run.")
        self._configure_destination(server)

    def start_measurement(self):
        if self._coordinator is not None:
            self._coordinator.stop_others(self)
        try:
            resp = serval_client.SESSION.get(f"{self._server_url()}/measurement/start")
            self._set_status(resp.text.strip() or "Measurement started.")
        except Exception as exc:
            self._set_status(f"Start failed: {type(exc).__name__}: {exc}")

    def stop_measurement(self):
        try:
            resp = serval_client.SESSION.get(f"{self._server_url()}/measurement/stop")
            self._set_status(resp.text.strip() or "Measurement stopped.")
        except Exception as exc:
            self._set_status(f"Stop failed: {type(exc).__name__}: {exc}")
