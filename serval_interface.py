import json
import pathlib
import tempfile
import time
import qtk as tk
from qtk import ttk, filedialog

import requests

import serval_client
import ui_style


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
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.rowconfigure(0, weight=2)
        main.rowconfigure(1, weight=1)
        # Trailing empty row keeps content packed at the top.
        sidebar.rowconfigure(3, weight=1)

        ui_style.build_button_bar(sidebar, 0, [("Ping", self.ping), ("Initialize", self.initialize)])
        self._status_block = ui_style.StatusBlock(sidebar, 1, self.status_var)

        server = ui_style.build_section(sidebar, 2, "Detector setup", pady=0)
        ui_style.add_form_row(server, 0, "Server URL:", ttk.Entry(server, textvariable=self.server_var))
        ui_style.add_form_row(
            server, 1, "BPC file:", ui_style.build_path_field(server, self.bpc_var, self._browse_bpc)
        )
        ui_style.add_form_row(
            server, 2, "DACS file:", ui_style.build_path_field(server, self.dacs_var, self._browse_dacs)
        )
        ui_style.add_form_row(
            server, 3, "Bias voltage (V):", ttk.Entry(server, textvariable=self.bias_voltage_var, width=18)
        )

        dashboard = ui_style.build_card(main, 0, "Dashboard")
        dashboard.columnconfigure(0, weight=1)
        dashboard.rowconfigure(0, weight=1)
        self.dashboard = tk.Text(dashboard, wrap="word", height=12)
        self.dashboard.grid(row=0, column=0, sticky="nsew")
        self.dashboard.configure(state="disabled")

        log = ui_style.build_card(main, 1, "Log", pady=8)
        log.columnconfigure(0, weight=1)
        log.rowconfigure(0, weight=1)
        self.log = tk.Text(log, wrap="word", height=8)
        self.log.grid(row=0, column=0, sticky="nsew")

    def _browse_bpc(self):
        self._browse_config_file(self.bpc_var, "Choose pixel config (BPC) file", [("BPC", "*.bpc")])

    def _browse_dacs(self):
        self._browse_config_file(self.dacs_var, "Choose DACS file", [("DACS", "*.dacs")])

    def _browse_config_file(self, var, title, filetypes):
        initial = var.get().strip()
        chosen = filedialog.askopenfilename(
            title=title,
            initialdir=str(pathlib.Path(initial).parent) if initial else None,
            filetypes=filetypes + [("All files", "*.*")],
            parent=self,
        )
        if chosen:
            var.set(chosen)

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
            self._set_status(f"Ping OK ({resp.status_code}).")
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
