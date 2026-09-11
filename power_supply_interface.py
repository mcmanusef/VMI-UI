"""Power Supply: control the CAEN SY127 HV crate from the VMI UI.

This is the SY127 channel-control panel (originally a standalone app under
CAEN_PS_Control/gui.py) folded into the Experiment Control notebook as one
more tab. It is self-contained -- the serial wrapper, the menu-walking
worker thread, and the screen parser all live here -- so the VMI UI has no
build-time dependency on the CAEN_PS_Control package. pyserial is imported
lazily: if it is missing the tab still loads and just refuses to connect.

How it works under the hood:
  * A worker thread owns the serial port and walks the SY127's menu UI.
  * Every ~1 s the worker pulls DISPLAY STATUS for GROUP ALL, parses the
    fixed-format table, and pushes a dict onto a queue.
  * The Tk main loop drains that queue (via .after) and updates the widgets.
  * Apply / ON / OFF put a write request on a second queue; the worker
    drains writes between status reads, navigating Main -> B -> A ->
    SINGLE CHANNEL -> set, then returns to DISPLAY STATUS.

Assumptions (same as the original panel):
  * Channels still have their default names (CH00, CH01, ...). If they were
    renamed via the FORMAT menu, fix the "Channels" field in the connection
    bar or rename them back.
  * GROUP ALL contains every channel (factory default per the manual).
"""

import queue
import re
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

try:
    import serial
except ImportError:  # pyserial not installed -- tab loads but can't connect
    serial = None

import app_settings


# Default channels to show; overridable from the connection bar.
DEFAULT_CHANNELS = "CH00, CH01, CH02, CH03, CH08, CH09"

# How often to ask the device to refresh DISPLAY STATUS
POLL_SECONDS = 1.0

# Menu key mnemonics from the SY127 manual
KEY_TOP = "1"       # back to MAIN MENU
KEY_BACK = "2"      # back one menu


# --- Serial wrapper ----------------------------------------------------------

class SY127:
    """Wrapper around a serial port talking to an SY127 Communication Controller.

    Factory shipping defaults: 9600 baud, parity disabled, 1 stop bit,
    XON/XOFF on. Pin 20 (DSR) on the DB-25 must be asserted externally if the
    USB-serial cable doesn't drive it, or the SY127 may stall.
    """

    def __init__(self, port="COM3", baudrate=9600,
                 parity=None, bytesize=None, stopbits=None,
                 xonxoff=True, timeout=0.1):
        if serial is None:
            raise RuntimeError(
                "pyserial is not installed. Run 'pip install pyserial' to use "
                "the Power Supply tab."
            )
        self.ser = serial.Serial(
            port=port, baudrate=baudrate,
            bytesize=bytesize if bytesize is not None else serial.EIGHTBITS,
            parity=parity if parity is not None else serial.PARITY_NONE,
            stopbits=stopbits if stopbits is not None else serial.STOPBITS_ONE,
            xonxoff=xonxoff, timeout=timeout,
        )

    def close(self):
        self.ser.close()

    def write(self, data):
        if isinstance(data, str):
            data = data.encode("ascii", errors="replace")
        self.ser.write(data)
        self.ser.flush()

    def send_key(self, key):
        self.write(key)

    def read_screen(self, settle=0.4, max_wait=3.0):
        """Read until the line goes quiet for `settle` s, or `max_wait` total."""
        buf = bytearray()
        deadline = time.monotonic() + max_wait
        last_data = time.monotonic()
        while time.monotonic() < deadline:
            chunk = self.ser.read(256)
            if chunk:
                buf.extend(chunk)
                last_data = time.monotonic()
            elif buf and (time.monotonic() - last_data) >= settle:
                break
        return buf.decode("ascii", errors="replace")

    def drain(self):
        self.ser.reset_input_buffer()


# --- Screen parsing --------------------------------------------------------

def strip_vt52(text):
    """Strip VT52 escape sequences for clean text parsing."""
    text = re.sub(r"\x1bY..", "", text, flags=re.DOTALL)
    text = re.sub(r"\x1b.", "", text)
    return text


# Field order in the DISPLAY STATUS table (Tab. 2 in the manual)
_STATUS_FIELDS = ["vmon", "imon", "v0", "v1", "i0", "i1",
                  "rup", "rdw", "trip", "status", "ramp"]


def parse_status_screen(text):
    """Parse a DISPLAY STATUS screen.

    Returns (header_dict, channels_dict). header_dict has 'hv_enable' (ON/OFF),
    'active_v' (V0/V1), 'group'. channels_dict maps 'CH00' -> {field: str}.
    """
    text = strip_vt52(text)
    header = {}
    channels = {}

    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue

        if "HV-ENABLE" in line:
            header["hv_enable"] = "ON" if "HV-ENABLE ON" in line else "OFF"
            m = re.search(r"ACTIVE\s+(V[01])", line)
            if m:
                header["active_v"] = m.group(1)
            m = re.search(r"GROUP\s+(\S+)", line)
            if m:
                header["group"] = m.group(1)

        m = re.match(r"^\s*(CH\d+)\s+(.+)$", line)
        if not m:
            continue
        ch_name = m.group(1)
        fields = m.group(2).split()
        if len(fields) < 10:
            continue
        data = {}
        for i, field_name in enumerate(_STATUS_FIELDS):
            data[field_name] = fields[i] if i < len(fields) else ""
        channels[ch_name] = data

    return header, channels


# --- Worker thread -------------------------------------------------------

class SY127Worker:
    """Background thread that owns the SY127 serial connection and walks the
    menu UI on behalf of the GUI."""

    # Per-parameter menu letters in the SINGLE CHANNEL screen (Tab. 4).
    # STATUS (N) is intentionally not here -- see _toggle_status.
    PARAM_LETTERS = {
        "v0": "C", "i0": "F",
        "ramp_up": "I", "ramp_down": "J",
    }

    def __init__(self, sy, status_q, log_q):
        self.sy = sy
        self.status_q = status_q
        self.log_q = log_q
        self.cmd_q = queue.Queue()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop_now(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def queue_write(self, channel, params):
        """Schedule a parameter change. params: dict like {'v0': '1500'}."""
        self.cmd_q.put(("write", (channel, params)))

    def queue_status_toggle(self, channel):
        """Schedule an ON/OFF toggle. The SY127 flips STATUS on a single 'N'
        keystroke -- no value prompt -- so the worker just navigates and
        presses N."""
        self.cmd_q.put(("toggle", channel))

    # -- thread body --

    def _run(self):
        try:
            self._navigate_to_status()
        except Exception as e:
            self.log_q.put(f"Initialization failed: {e}")
            return

        next_read = time.monotonic()

        while not self.stop.is_set():
            wrote = False
            try:
                while True:
                    cmd, data = self.cmd_q.get_nowait()
                    if cmd == "write":
                        ch, params = data
                        try:
                            self._write_param(ch, params)
                            self.log_q.put(f"{ch}: applied {params}")
                            wrote = True
                        except Exception as e:
                            self.log_q.put(f"{ch}: write failed: {e}")
                    elif cmd == "toggle":
                        ch = data
                        try:
                            self._toggle_status(ch)
                            self.log_q.put(f"{ch}: status toggled")
                            wrote = True
                        except Exception as e:
                            self.log_q.put(f"{ch}: toggle failed: {e}")
            except queue.Empty:
                pass

            if wrote:
                try:
                    self._navigate_to_status()
                except Exception as e:
                    self.log_q.put(f"Navigation failed: {e}")
                next_read = time.monotonic()  # immediate readback after write

            if time.monotonic() >= next_read:
                try:
                    self._read_status()
                except Exception as e:
                    self.log_q.put(f"Read failed: {e}")
                next_read = time.monotonic() + POLL_SECONDS

            time.sleep(0.05)

    # -- menu navigation primitives --

    def _navigate_to_status(self):
        """Get to DISPLAY STATUS / GROUP ALL from any starting state."""
        self.sy.drain()
        self.sy.send_key(KEY_TOP)                             # MAIN MENU
        self.sy.read_screen(settle=0.3, max_wait=2.0)
        self.sy.send_key("A")                                 # DISPLAY STATUS
        screen = self.sy.read_screen(settle=0.3, max_wait=2.0)

        # 7 named groups + ALL, so 8 cycles is the worst case.
        for _ in range(8):
            if re.search(r"GROUP\s+ALL", strip_vt52(screen)):
                break
            self.sy.send_key("Q")                             # NEXT GROUP
            screen = self.sy.read_screen(settle=0.3, max_wait=2.0)

    def _read_status(self):
        self.sy.send_key("O")                                 # REDISPLAY
        screen = self.sy.read_screen(settle=0.3, max_wait=2.0)
        header, channels = parse_status_screen(screen)
        if channels:
            self.status_q.put((header, channels))

    def _write_param(self, channel, params):
        """Walk MAIN MENU -> B -> A -> select channel -> set fields."""
        self.sy.drain()
        self.sy.send_key(KEY_TOP); self.sy.read_screen(settle=0.2, max_wait=2.0)
        self.sy.send_key("B"); self.sy.read_screen(settle=0.2, max_wait=2.0)
        self.sy.send_key("A"); self.sy.read_screen(settle=0.2, max_wait=2.0)

        self.sy.send_key("A")                                 # pick channel by name
        time.sleep(0.1)
        self.sy.write(f"{channel}\r")
        self.sy.read_screen(settle=0.3, max_wait=2.0)

        for name, value in params.items():
            letter = self.PARAM_LETTERS.get(name)
            if letter is None:
                continue
            self.sy.send_key(letter)
            time.sleep(0.1)
            self.sy.write(f"{value}\r")
            self.sy.read_screen(settle=0.3, max_wait=2.0)

    def _toggle_status(self, channel):
        """Walk MAIN MENU -> B -> A -> select channel -> press N (no value)."""
        self.sy.drain()
        self.sy.send_key(KEY_TOP); self.sy.read_screen(settle=0.2, max_wait=2.0)
        self.sy.send_key("B"); self.sy.read_screen(settle=0.2, max_wait=2.0)
        self.sy.send_key("A"); self.sy.read_screen(settle=0.2, max_wait=2.0)

        self.sy.send_key("A")
        time.sleep(0.1)
        self.sy.write(f"{channel}\r")
        self.sy.read_screen(settle=0.3, max_wait=2.0)

        # The SY127 flips STATUS on the keystroke itself -- no value, no CR.
        self.sy.send_key("N")
        self.sy.read_screen(settle=0.3, max_wait=2.0)


# --- GUI: a single channel row ---------------------------------------------

class ChannelRow:
    """One row in the channel table."""

    STATUS_COLORS = {
        "ON": "#0a7", "OFF": "#888", "OVC": "#c80",
        "OVV": "#c00", "UVV": "#c00", "UNV": "#c00", "TRIP": "#c00",
    }

    # Channel is effectively powered up (toggle turns it off) vs off
    # (toggle turns it on).
    _ON_LIKE = {"ON", "OVC", "OVV", "UVV", "UNV"}
    _OFF_LIKE = {"OFF", "TRIP"}

    def __init__(self, parent, row_index, channel, on_apply_cb, on_status_cb):
        self.channel = channel
        self.on_apply_cb = on_apply_cb
        self.on_status_cb = on_status_cb

        # When True, the next status update overwrites the setpoint entries
        # (used at connect time and after each Apply).
        self._refresh_setpoints_pending = True

        self.vmon_var = tk.StringVar(value="—")
        self.imon_var = tk.StringVar(value="—")
        self.v0_var = tk.StringVar()
        self.i0_var = tk.StringVar()
        self.rup_var = tk.StringVar()
        self.rdw_var = tk.StringVar()
        self.status_var = tk.StringVar(value="—")

        col = 0
        ttk.Label(parent, text=channel, font=("TkDefaultFont", 10, "bold"),
                  width=6, anchor="w").grid(row=row_index, column=col, padx=4, pady=3, sticky="w")
        col += 1

        ttk.Label(parent, textvariable=self.vmon_var, width=8, anchor="e",
                  foreground="#0a7", font=("TkFixedFont", 10, "bold")
                  ).grid(row=row_index, column=col, padx=4, sticky="e")
        col += 1

        self.v0_entry = ttk.Entry(parent, textvariable=self.v0_var, width=8, justify="right")
        self.v0_entry.grid(row=row_index, column=col, padx=4)
        self.v0_entry.bind("<Return>", lambda e: self.apply())
        col += 1

        ttk.Label(parent, textvariable=self.imon_var, width=8, anchor="e",
                  foreground="#0a7", font=("TkFixedFont", 10, "bold")
                  ).grid(row=row_index, column=col, padx=4, sticky="e")
        col += 1

        self.i0_entry = ttk.Entry(parent, textvariable=self.i0_var, width=8, justify="right")
        self.i0_entry.grid(row=row_index, column=col, padx=4)
        self.i0_entry.bind("<Return>", lambda e: self.apply())
        col += 1

        self.rup_entry = ttk.Entry(parent, textvariable=self.rup_var, width=8, justify="right")
        self.rup_entry.grid(row=row_index, column=col, padx=4)
        self.rup_entry.bind("<Return>", lambda e: self.apply())
        col += 1

        self.rdw_entry = ttk.Entry(parent, textvariable=self.rdw_var, width=8, justify="right")
        self.rdw_entry.grid(row=row_index, column=col, padx=4)
        self.rdw_entry.bind("<Return>", lambda e: self.apply())
        col += 1

        self.status_label = ttk.Label(parent, textvariable=self.status_var, width=8,
                                      anchor="center", font=("TkDefaultFont", 10, "bold"))
        self.status_label.grid(row=row_index, column=col, padx=4)
        col += 1

        ttk.Button(parent, text="ON", width=4,
                   command=lambda: self._request_status_change("ON")
                   ).grid(row=row_index, column=col, padx=1)
        col += 1

        ttk.Button(parent, text="OFF", width=4,
                   command=lambda: self._request_status_change("OFF")
                   ).grid(row=row_index, column=col, padx=1)
        col += 1

        ttk.Button(parent, text="Apply", width=6, command=self.apply
                   ).grid(row=row_index, column=col, padx=8)

    def _request_status_change(self, desired):
        """The SY127 only knows how to toggle, so check the last-known status
        and only fire a toggle if it would move the channel toward the desired
        state -- avoids clicking 'ON' on an already-ON channel toggling it OFF."""
        current = self.status_var.get().split("/")[0]
        if current in ("—", ""):
            return  # no status seen yet; don't act blindly
        will_toggle = (
            (desired == "ON" and current in self._OFF_LIKE) or
            (desired == "OFF" and current in self._ON_LIKE)
        )
        if will_toggle:
            self.on_status_cb(self.channel)

    def update(self, data):
        """Apply a fresh status snapshot to this row."""
        self.vmon_var.set(data.get("vmon", "—"))
        self.imon_var.set(data.get("imon", "—"))

        st = data.get("status", "—")
        ramp = data.get("ramp", "")
        display_status = f"{st}/{ramp}" if ramp in ("RUP", "RDW") else st
        self.status_var.set(display_status)
        self.status_label.config(foreground=self.STATUS_COLORS.get(st, "#000"))

        # Only overwrite setpoint entries when explicitly requested, else the
        # user's in-progress edits get wiped every tick.
        if self._refresh_setpoints_pending:
            self.v0_var.set(data.get("v0", ""))
            self.i0_var.set(data.get("i0", ""))
            self.rup_var.set(data.get("rup", ""))
            self.rdw_var.set(data.get("rdw", ""))
            self._refresh_setpoints_pending = False

    def request_setpoint_refresh(self):
        self._refresh_setpoints_pending = True

    def apply(self):
        params = {
            "v0": self.v0_var.get().strip(),
            "i0": self.i0_var.get().strip(),
            "ramp_up": self.rup_var.get().strip(),
            "ramp_down": self.rdw_var.get().strip(),
        }
        params = {k: v for k, v in params.items() if v}
        if not params:
            return
        self.on_apply_cb(self.channel, params)
        # After the write the device's value is truth again.
        self._refresh_setpoints_pending = True


# --- The tab ------------------------------------------------------------

class PowerSupplyInterface(ttk.Frame):
    """Live VMON / IMON / status for the SY127 HV channels, with editable
    V0 / I0 / ramp and ON / OFF -- without touching the menu terminal."""

    POLL_MS = 100  # how often the GUI drains the worker's queues

    def __init__(self, parent, coordinator=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)

        self.sy = None
        self.worker = None
        self.status_q = queue.Queue()
        self.log_q = queue.Queue()
        self.rows = {}

        # Connection settings, persisted via the global "Save current as
        # default" button (see default_fields).
        self.port_var = tk.StringVar(self, value=app_settings.get("power_supply.port", "COM3"))
        self.baud_var = tk.StringVar(self, value=app_settings.get("power_supply.baud", "9600"))
        self.parity_var = tk.StringVar(self, value=app_settings.get("power_supply.parity", "N"))
        self.bits_var = tk.StringVar(self, value=app_settings.get("power_supply.bits", "8"))
        self.stop_var = tk.StringVar(self, value=app_settings.get("power_supply.stop", "1"))
        self.channels_var = tk.StringVar(
            self, value=app_settings.get("power_supply.channels", DEFAULT_CHANNELS)
        )

        self.hv_enable_var = tk.StringVar(self, value="HV-ENABLE: ?")
        self.active_var = tk.StringVar(self, value="")
        self.status_var = tk.StringVar(self, value="Disconnected")

        self._build_ui()
        self.after(self.POLL_MS, self._poll_queues)

    # -- coordinator contract --
    #
    # The power supply tab is not a Serval measurement, so it never fights
    # over the trigger/destination config. is_running() stays False so other
    # tabs' start() never tears down the HV connection.

    def is_running(self):
        return False

    def stop(self):
        pass

    def default_fields(self):
        return {
            "power_supply.port": self.port_var,
            "power_supply.baud": self.baud_var,
            "power_supply.parity": self.parity_var,
            "power_supply.bits": self.bits_var,
            "power_supply.stop": self.stop_var,
            "power_supply.channels": self.channels_var,
        }

    # -- layout --

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        self._build_connection_bar()
        self._build_channel_grid()
        self._build_log()
        self._build_status_bar()

        if serial is None:
            self._log("pyserial is not installed -- 'pip install pyserial' to enable connecting.")
            self.status_var.set("pyserial not installed")

    def _build_connection_bar(self):
        bar = ttk.LabelFrame(self, text="Connection")
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        for label, var, width in [
            ("Port", self.port_var, 8),
            ("Baud", self.baud_var, 6),
            ("Parity", self.parity_var, 3),
            ("Bits", self.bits_var, 3),
            ("Stop", self.stop_var, 3),
        ]:
            ttk.Label(bar, text=f"{label}:").pack(side="left", padx=(8, 2))
            ttk.Entry(bar, textvariable=var, width=width).pack(side="left")

        ttk.Label(bar, text="Channels:").pack(side="left", padx=(8, 2))
        ttk.Entry(bar, textvariable=self.channels_var, width=32).pack(side="left")

        self.connect_btn = ttk.Button(bar, text="Connect", command=self._toggle_connect)
        self.connect_btn.pack(side="left", padx=12)

        self.hv_enable_label = ttk.Label(bar, textvariable=self.hv_enable_var,
                                         font=("TkDefaultFont", 10, "bold"))
        self.hv_enable_label.pack(side="left", padx=12)
        ttk.Label(bar, textvariable=self.active_var, foreground="#666").pack(side="left", padx=4)

    def _parse_channels(self):
        return [c.strip().upper() for c in self.channels_var.get().replace(";", ",").split(",") if c.strip()]

    def _build_channel_grid(self):
        self._grid = ttk.LabelFrame(self, text="Channels")
        self._grid.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        self._populate_channel_grid()

        self._hint = ttk.Label(
            self,
            text=("Edit V0 / I0 / Ramp values then click Apply (or press Enter "
                  "in any field). ON / OFF buttons act immediately."),
            foreground="#666", font=("TkDefaultFont", 9, "italic"))
        self._hint.grid(row=1, column=0, sticky="sw", padx=8, pady=(0, 2))
        self._hint.lift()

    def _populate_channel_grid(self):
        for child in self._grid.winfo_children():
            child.destroy()
        self.rows = {}

        headers = ["Channel", "V mon", "V0 set", "I mon", "I0 set",
                   "Ramp Up", "Ramp Dn", "Status", "", "", ""]
        for i, h in enumerate(headers):
            ttk.Label(self._grid, text=h, font=("TkDefaultFont", 9, "bold"),
                      anchor="center").grid(row=0, column=i, padx=4, pady=4, sticky="ew")
        ttk.Separator(self._grid, orient="horizontal").grid(
            row=1, column=0, columnspan=len(headers), sticky="ew", pady=2)

        for i, ch in enumerate(self._parse_channels()):
            self.rows[ch] = ChannelRow(
                self._grid, i + 2, ch,
                on_apply_cb=self._on_apply_params,
                on_status_cb=self._on_toggle_status,
            )

    def _build_log(self):
        log_frame = ttk.LabelFrame(self, text="Activity")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=6, wrap="word", font=("TkFixedFont", 9))
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.config(yscrollcommand=scroll.set)

    def _build_status_bar(self):
        ttk.Label(self, textvariable=self.status_var, anchor="w",
                  relief="sunken", padding=(6, 2)).grid(row=3, column=0, sticky="ew")

    # -- connection management --

    def _toggle_connect(self):
        if self.sy is None:
            self._connect()
        else:
            self._disconnect()

    def _connect(self):
        if serial is None:
            messagebox.showerror("pyserial missing",
                                 "pyserial is not installed. Run 'pip install pyserial'.")
            return

        parity_map = {"N": serial.PARITY_NONE, "E": serial.PARITY_EVEN, "O": serial.PARITY_ODD}
        bits_map = {"7": serial.SEVENBITS, "8": serial.EIGHTBITS}
        stop_map = {"1": serial.STOPBITS_ONE, "2": serial.STOPBITS_TWO}

        # Rebuild the row set in case the channel list was edited.
        self._populate_channel_grid()
        self._hint.lift()

        try:
            self.sy = SY127(
                port=self.port_var.get().strip(),
                baudrate=int(self.baud_var.get()),
                parity=parity_map[self.parity_var.get().strip().upper()],
                bytesize=bits_map[self.bits_var.get().strip()],
                stopbits=stop_map[self.stop_var.get().strip()],
            )
        except Exception as e:
            messagebox.showerror("Connection failed", str(e))
            self.sy = None
            return

        for row in self.rows.values():
            row.request_setpoint_refresh()

        self.worker = SY127Worker(self.sy, self.status_q, self.log_q)
        self.worker.start()

        self.connect_btn.config(text="Disconnect")
        self.status_var.set(f"Connected to {self.port_var.get()} — initializing…")
        self._log(f"Connected to {self.port_var.get()}")

    def _disconnect(self):
        if self.worker is not None:
            self.worker.stop_now()
            self.worker = None
        if self.sy is not None:
            try:
                self.sy.close()
            except Exception:
                pass
            self.sy = None
        self.connect_btn.config(text="Connect")
        self.status_var.set("Disconnected")
        self.hv_enable_var.set("HV-ENABLE: ?")
        self.hv_enable_label.config(foreground="#000")
        self.active_var.set("")
        self._log("Disconnected")

    # -- callbacks from rows --

    def _on_apply_params(self, channel, params):
        if self.worker is None:
            self._log("Not connected -- connect first.")
            return
        self.worker.queue_write(channel, params)
        self._log(f"{channel}: queued {params}")

    def _on_toggle_status(self, channel):
        if self.worker is None:
            self._log("Not connected -- connect first.")
            return
        self.worker.queue_status_toggle(channel)
        self._log(f"{channel}: queued status toggle")

    # -- queue draining (Tk main thread) --

    def _poll_queues(self):
        try:
            while True:
                header, data = self.status_q.get_nowait()
                self._apply_status(header, data)
        except queue.Empty:
            pass

        try:
            while True:
                self._log(self.log_q.get_nowait())
        except queue.Empty:
            pass

        self.after(self.POLL_MS, self._poll_queues)

    def _apply_status(self, header, data):
        if "hv_enable" in header:
            state = header["hv_enable"]
            self.hv_enable_var.set(f"HV-ENABLE: {state}")
            self.hv_enable_label.config(foreground="#0a7" if state == "ON" else "#888")

        bits = []
        if "active_v" in header:
            bits.append(f"Active: {header['active_v']}")
        if "group" in header:
            bits.append(f"Group: {header['group']}")
        self.active_var.set(" | ".join(bits))

        for ch, row in self.rows.items():
            if ch in data:
                row.update(data[ch])

        self.status_var.set(f"Last update: {time.strftime('%H:%M:%S')}")

    # -- log helpers --

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")
        line_count = int(self.log.index("end-1c").split(".")[0])
        if line_count > 200:
            self.log.delete("1.0", f"{line_count - 150}.0")

    def destroy(self):
        self._disconnect()
        super().destroy()
