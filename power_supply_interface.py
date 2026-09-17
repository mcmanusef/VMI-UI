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
import qtk as tk
from qtk import messagebox, ttk

try:
    import serial
except ImportError:  # pyserial not installed -- tab loads but can't connect
    serial = None

import app_settings
import ui_style


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


def _format_setpoint(value):
    """A V0 value as typed into the SY127: at most one decimal place, no
    trailing zeros (1500.0 -> "1500", 742.25 -> "742.2")."""
    return f"{round(value, 1):g}"


class PowerSupplyUnavailable(RuntimeError):
    """A channel group can't be driven right now -- not connected, no such
    group, or its V0 ratios were never captured."""


class GroupVoltageControl:
    """Drives one channel group's V0 from another thread (the parameter
    sweep lowering the voltages while the stage moves), using the same lead
    V0 + captured ratios/differences scheme as the Groups card's Apply
    button.

    It never touches a widget: the writes go through the SY127 worker's
    command queue, log lines through the log queue, and the readback comes
    from the status snapshot the GUI thread keeps up to date (see
    PowerSupplyInterface._apply_status). Build it on the GUI thread, with
    PowerSupplyInterface.group_control(), and use it from anywhere.
    """

    # How far VMON may sit from its target and still count as "the ramp has
    # finished" -- the SY127 reports whole volts and hovers a little either
    # side of V0 rather than landing on it exactly.
    SETTLE_TOLERANCE_V = 5.0
    SETTLE_TOLERANCE_FRAC = 0.02

    def __init__(self, ui, group, members, mode, factors):
        self._ui = ui
        self.group = group
        self.members = list(members)
        self.mode = mode
        self.factors = dict(factors)

    def targets(self, lead_v0):
        """{channel: V0} for a given lead voltage -- rounded exactly the way
        the values actually written to the device are, so the readback
        check below compares against what was really asked for. In "ratio"
        mode each member is lead_v0 times its captured ratio; in
        "difference" mode each member is lead_v0 plus its captured offset."""
        if self.mode == "difference":
            return {ch: float(_format_setpoint(lead_v0 + self.factors[ch])) for ch in self.members}
        return {ch: float(_format_setpoint(lead_v0 * self.factors[ch])) for ch in self.members}

    def set_lead_v0(self, lead_v0):
        """Queue the V0 write for every member. Returns (targets,
        requested_at) for wait_until_settled()."""
        worker = self._ui.worker
        if worker is None:
            raise PowerSupplyUnavailable("Power Supply tab isn't connected to the crate.")
        targets = self.targets(lead_v0)
        requested_at = time.monotonic()
        for ch in self.members:
            worker.queue_write(ch, {"v0": _format_setpoint(targets[ch])})
            row = self._ui.rows.get(ch)
            if row is not None:
                # After the write the device's value is truth again. This
                # only sets a flag -- no widget is touched -- so it is safe
                # off the GUI thread; the entry itself is refreshed by the
                # next status update.
                row.request_setpoint_refresh()
        self._ui.log_q.put(f"Group {self.group}: moving to lead V0 {_format_setpoint(lead_v0)}")
        return targets, requested_at

    def wait_until_settled(self, targets, requested_at, timeout=120.0, stop_event=None, poll=0.25):
        """Block until every member's VMON has reached its target, judged
        only on status snapshots read after requested_at -- an older one
        still shows the voltage the group is ramping away from. Returns
        True once it's there, False if the timeout ran out or stop_event
        was set first."""
        deadline = time.monotonic() + timeout
        while True:
            if stop_event is not None and stop_event.is_set():
                return False
            stamp, channels = self._ui.status_snapshot()
            if stamp > requested_at and self._all_reached(channels, targets):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll)

    def _all_reached(self, channels, targets):
        for ch, target in targets.items():
            data = channels.get(ch)
            if not data:
                return False
            if data.get("ramp") in ("RUP", "RDW"):
                return False
            try:
                vmon = float(data.get("vmon", ""))
            except ValueError:
                return False
            if abs(vmon - target) > max(self.SETTLE_TOLERANCE_V, abs(target) * self.SETTLE_TOLERANCE_FRAC):
                return False
        return True


# --- GUI: a single channel row ---------------------------------------------

class ChannelRow:
    """One row in the channel table."""

    STATUS_COLORS = {
        "ON": ui_style.RUNNING, "OFF": ui_style.IDLE, "OVC": ui_style.WARNING,
        "OVV": ui_style.DANGER, "UVV": ui_style.DANGER, "UNV": ui_style.DANGER, "TRIP": ui_style.DANGER,
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

        self.vmon_var = tk.StringVar(value="--")
        self.imon_var = tk.StringVar(value="--")
        self.v0_var = tk.StringVar()
        self.i0_var = tk.StringVar()
        self.rup_var = tk.StringVar()
        self.rdw_var = tk.StringVar()
        self.status_var = tk.StringVar(value="--")

        # Operator-facing name and move-together group, saved per channel ID
        # so they survive reconnects, channel-list edits and restarts.
        self.name_var = app_settings.persistent_var(None, tk.StringVar, f"power_supply.name.{channel}", "")
        self.group_name_var = app_settings.persistent_var(
            None, tk.StringVar, f"power_supply.group.{channel}", ""
        )

        col = 0
        ttk.Label(parent, text=channel, width=6, anchor="w").grid(
            row=row_index, column=col, padx=(6, 4), pady=1, sticky="w"
        )
        col += 1

        ttk.Entry(parent, textvariable=self.name_var, width=10).grid(row=row_index, column=col, padx=2, pady=1)
        col += 1

        ttk.Entry(parent, textvariable=self.group_name_var, width=5).grid(row=row_index, column=col, padx=2, pady=1)
        col += 1

        # Monitor readouts in a fixed-width font so digits line up.
        ttk.Label(parent, textvariable=self.vmon_var, width=6, anchor="e",
                  font=("TkFixedFont", 10)).grid(row=row_index, column=col, padx=2, pady=1, sticky="e")
        col += 1

        self.v0_entry = ttk.Entry(parent, textvariable=self.v0_var, width=6, justify="right")
        self.v0_entry.grid(row=row_index, column=col, padx=2, pady=1)
        self.v0_entry.bind("<Return>", lambda e: self.apply())
        col += 1

        ttk.Label(parent, textvariable=self.imon_var, width=6, anchor="e",
                  font=("TkFixedFont", 10)).grid(row=row_index, column=col, padx=2, pady=1, sticky="e")
        col += 1

        self.i0_entry = ttk.Entry(parent, textvariable=self.i0_var, width=6, justify="right")
        self.i0_entry.grid(row=row_index, column=col, padx=2, pady=1)
        self.i0_entry.bind("<Return>", lambda e: self.apply())
        col += 1

        self.rup_entry = ttk.Entry(parent, textvariable=self.rup_var, width=6, justify="right")
        self.rup_entry.grid(row=row_index, column=col, padx=2, pady=1)
        self.rup_entry.bind("<Return>", lambda e: self.apply())
        col += 1

        self.rdw_entry = ttk.Entry(parent, textvariable=self.rdw_var, width=6, justify="right")
        self.rdw_entry.grid(row=row_index, column=col, padx=2, pady=1)
        self.rdw_entry.bind("<Return>", lambda e: self.apply())
        col += 1

        self.status_label = ttk.Label(parent, textvariable=self.status_var, width=6, anchor="center")
        self.status_label.grid(row=row_index, column=col, padx=2, pady=1)
        col += 1

        ttk.Button(parent, text="ON", width=4,
                   command=lambda: self._request_status_change("ON")
                   ).grid(row=row_index, column=col, padx=2, pady=1)
        col += 1

        ttk.Button(parent, text="OFF", width=4,
                   command=lambda: self._request_status_change("OFF")
                   ).grid(row=row_index, column=col, padx=2, pady=1)
        col += 1

        ttk.Button(parent, text="Apply", width=6, command=self.apply
                   ).grid(row=row_index, column=col, padx=(8, 6), pady=1)

    @property
    def label(self):
        """Channel ID plus its name, if it has one -- for logs and the
        Groups card."""
        name = self.name_var.get().strip()
        return f"{self.channel} ({name})" if name else self.channel

    def _request_status_change(self, desired):
        """The SY127 only knows how to toggle, so check the last-known status
        and only fire a toggle if it would move the channel toward the desired
        state -- avoids clicking 'ON' on an already-ON channel toggling it OFF."""
        current = self.status_var.get().split("/")[0]
        if current in ("--", ""):
            return  # no status seen yet; don't act blindly
        will_toggle = (
            (desired == "ON" and current in self._OFF_LIKE) or
            (desired == "OFF" and current in self._ON_LIKE)
        )
        if will_toggle:
            self.on_status_cb(self.channel)

    def update(self, data):
        """Apply a fresh status snapshot to this row."""
        self.vmon_var.set(data.get("vmon", "--"))
        self.imon_var.set(data.get("imon", "--"))

        st = data.get("status", "--")
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
        # Last DISPLAY STATUS read, as (monotonic timestamp, {channel:
        # fields}). Replaced wholesale, never mutated, so other threads can
        # read it through status_snapshot() -- see GroupVoltageControl.
        self._status_snapshot = (float("-inf"), {})

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

        self.hv_enable_var = tk.StringVar(self, value="--")
        self.active_var = tk.StringVar(self, value="--")
        self.group_var = tk.StringVar(self, value="--")
        self.status_var = tk.StringVar(self, value="Disconnected.")

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
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.rowconfigure(2, weight=1)
        # Trailing empty row keeps content packed at the top.
        sidebar.rowconfigure(4, weight=1)

        self.connect_btn = ui_style.build_button_bar(sidebar, 0, [("Connect", self._toggle_connect)])[0]
        # Green for as long as the serial connection is open, not just while
        # a status word happens to be showing.
        self._status_block = ui_style.StatusBlock(
            sidebar, 1, self.status_var, is_running=lambda _text: self.sy is not None
        )

        connection = ui_style.build_section(sidebar, 2, "Connection")
        for row, (label, var) in enumerate([
            ("Port:", self.port_var),
            ("Baud:", self.baud_var),
            ("Parity (N/E/O):", self.parity_var),
            ("Data bits:", self.bits_var),
            ("Stop bits:", self.stop_var),
        ]):
            ui_style.add_form_row(connection, row, label, ttk.Entry(connection, textvariable=var, width=18))
        ui_style.add_form_row(connection, 5, "Channels:", ttk.Entry(connection, textvariable=self.channels_var))

        crate = ui_style.build_section(sidebar, 3, "Crate", pady=0)
        ttk.Label(crate, text="HV enable:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=1)
        self.hv_enable_label = ttk.Label(crate, textvariable=self.hv_enable_var)
        self.hv_enable_label.grid(row=0, column=1, sticky="w", pady=1)
        ui_style.add_stat_rows(crate, [
            ("Active setpoint:", self.active_var),
            ("Group:", self.group_var),
        ], start_row=1)

        self._grid_card = ui_style.build_card(main, 0, "Channels", sticky="new")
        self._group_card = ui_style.build_card(main, 1, "Groups", sticky="new")
        self._group_card.columnconfigure(0, weight=1)
        self._grid = None
        self._group_grid = None
        self._group_lead_vars = {}
        self._group_ref_vars = {}
        self._group_mode_vars = {}
        self._populate_channel_grid()
        ui_style.build_button_bar(
            self._grid_card, 1, [("Toggle All", self._toggle_all_channels)], pady=(0, 8)
        )

        log_card = ui_style.build_card(main, 2, "Activity", pady=8)
        log_card.rowconfigure(0, weight=1)
        log_card.columnconfigure(0, weight=1)
        self.log = tk.Text(log_card, height=6, wrap="word", font=("TkFixedFont", 9))
        self.log.grid(row=0, column=0, sticky="nsew")

        if serial is None:
            self._log("pyserial is not installed -- 'pip install pyserial' to enable connecting.")
            self.status_var.set("pyserial not installed.")

    def _parse_channels(self):
        return [c.strip().upper() for c in self.channels_var.get().replace(";", ",").split(",") if c.strip()]

    def _populate_channel_grid(self):
        # Rebuilt (e.g. after the channel list is edited) by swapping in a
        # fresh inner frame, rather than destroying the old rows' widgets
        # (and their padding containers) one by one.
        if self._grid is not None:
            self._grid.destroy()
        self._grid = ttk.Frame(self._grid_card)
        self._grid.grid(row=0, column=0, sticky="ew")
        # 13 columns barely fit next to the sidebar; the card already
        # insets the table, so the frame's own default margins would only
        # push it past the main area's width.
        ui_style.zero_margins(self._grid)
        self.rows = {}

        headers = ["Channel", "Name", "Group", "V mon", "V0 set", "I mon", "I0 set",
                   "Ramp up", "Ramp down", "Status", "", "", ""]
        for i, h in enumerate(headers):
            ttk.Label(self._grid, text=h, anchor="center").grid(
                row=0, column=i, padx=(6, 4) if i == 0 else 2, pady=(4, 2), sticky="ew"
            )
        ttk.Separator(self._grid, orient="horizontal").grid(
            row=1, column=0, columnspan=len(headers), sticky="ew", pady=2)

        channels = self._parse_channels()
        for i, ch in enumerate(channels):
            row = ChannelRow(
                self._grid, i + 2, ch,
                on_apply_cb=self._on_apply_params,
                on_status_cb=self._on_toggle_status,
            )
            row.name_var.trace_add("write", self._on_group_config_changed)
            row.group_name_var.trace_add("write", self._on_group_config_changed)
            self.rows[ch] = row

        ui_style.add_note(
            self._grid, len(channels) + 2,
            "Edit V0 / I0 / ramp values, then click Apply (or press Enter in any field). "
            "ON / OFF act immediately. Name and Group are saved per channel. Toggle All below flips "
            "every listed channel together (all off if they're not all in the same state).",
            columnspan=len(headers), pady=(6, 4),
        )
        self._populate_group_grid()

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
        self.status_var.set(f"Connected to {self.port_var.get()}. Initializing...")
        self._log(f"Connected to {self.port_var.get()}")

    def _disconnect(self):
        # Drop the last reading so a reconnect can't settle a group against
        # whatever the crate showed before the port was closed.
        self._status_snapshot = (float("-inf"), {})
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
        self.status_var.set("Disconnected.")
        self.hv_enable_var.set("--")
        self.hv_enable_label.config(foreground="#000")
        self.active_var.set("--")
        self.group_var.set("--")
        self._log("Disconnected")

    # -- callbacks from rows --

    def _channel_label(self, channel):
        row = self.rows.get(channel)
        return row.label if row is not None else channel

    def _on_apply_params(self, channel, params):
        if self.worker is None:
            self._log("Not connected -- connect first.")
            return
        self.worker.queue_write(channel, params)
        self._log(f"{self._channel_label(channel)}: queued {params}")

    def _on_toggle_status(self, channel):
        if self.worker is None:
            self._log("Not connected -- connect first.")
            return
        self.worker.queue_status_toggle(channel)
        self._log(f"{self._channel_label(channel)}: queued status toggle")

    def _member_state(self, channel):
        """"ON"/"OFF" for a channel's last-known status, or None if it
        hasn't been read yet (or isn't a channel we know about)."""
        row = self.rows.get(channel)
        if row is None:
            return None
        current = row.status_var.get().split("/")[0]
        if current in ChannelRow._ON_LIKE:
            return "ON"
        if current in ChannelRow._OFF_LIKE:
            return "OFF"
        return None

    def _toggle_members(self, channels, label):
        """Flip a set of channels together, like one switch: all-ON goes to
        OFF and all-OFF goes to ON, but if they disagree (or some haven't
        reported a status yet) the safe move is to turn everything off
        rather than guess which way each one should go."""
        if self.worker is None:
            self._log("Not connected -- connect first.")
            return
        known = {ch: self._member_state(ch) for ch in channels}
        known = {ch: s for ch, s in known.items() if s is not None}
        if not known:
            self._log(f"{label}: no status read yet.")
            return
        if len(known) == len(channels) and len(set(known.values())) == 1:
            desired = "OFF" if next(iter(known.values())) == "ON" else "ON"
        else:
            desired = "OFF"

        queued = 0
        for ch, current in known.items():
            if current != desired:
                self.worker.queue_status_toggle(ch)
                self._log(f"{self._channel_label(ch)}: queued status toggle")
                queued += 1
        skipped = [ch for ch in channels if ch not in known]
        if skipped:
            self._log(f"{label}: skipping {', '.join(skipped)} (no status yet).")
        elif queued == 0:
            self._log(f"{label}: already {desired}.")

    def _toggle_all_channels(self):
        channels = list(self.rows)
        if not channels:
            return
        self._toggle_members(channels, "All channels")

    def _toggle_group(self, group):
        members = self._groups().get(group)
        if not members:
            return
        self._toggle_members(members, f"Group {group}")

    # -- channel groups --
    #
    # Channels sharing a Group name move together: "Capture" records each
    # member's V0 relative to the group's reference channel (its first
    # member in table order by default, or whichever channel is picked in
    # the Reference column), either as a ratio or as a fixed difference
    # (the Mode column). Apply then sets the reference channel to a new V0
    # and every other member to that value scaled by its ratio or offset by
    # its difference. The captured factors are stored rather than
    # re-derived from the current setpoints on every Apply, so repeated
    # moves (and one-decimal rounding) never drift the proportions/gaps.

    GROUP_RATIOS_KEY = "power_supply.group_ratios"
    GROUP_REFERENCE_KEY = "power_supply.group_reference"
    GROUP_MODE_KEY = "power_supply.group_mode"

    MODE_LABELS = {"ratio": "Ratio", "difference": "Difference"}
    MODE_VALUES = {label: mode for mode, label in MODE_LABELS.items()}

    def _groups(self):
        """Group name -> member channel IDs, in table order (first = lead)."""
        groups = {}
        for ch, row in self.rows.items():
            name = row.group_name_var.get().strip()
            if name:
                groups.setdefault(name, []).append(ch)
        return groups

    def _group_reference(self, group, members):
        """The channel `group`'s ratios/differences are measured against:
        whichever channel was picked in the Reference column, or the first
        member in table order if none was picked (or the picked one is no
        longer in the group)."""
        stored = (app_settings.get(self.GROUP_REFERENCE_KEY) or {}).get(group)
        return stored if stored in members else members[0]

    def _set_group_reference(self, group, channel):
        stored = dict(app_settings.get(self.GROUP_REFERENCE_KEY) or {})
        stored[group] = channel
        app_settings.set(self.GROUP_REFERENCE_KEY, stored)
        self._log(f"Group {group}: reference channel set to {self._channel_label(channel)}.")
        self._populate_group_grid()

    def _group_mode(self, group):
        """"ratio" (default) or "difference" -- whether `group`'s other
        members are locked to the reference channel by a fixed V0 ratio or
        a fixed V0 offset."""
        stored = (app_settings.get(self.GROUP_MODE_KEY) or {}).get(group)
        return stored if stored in self.MODE_LABELS else "ratio"

    def _set_group_mode(self, group, mode):
        stored = dict(app_settings.get(self.GROUP_MODE_KEY) or {})
        stored[group] = mode
        app_settings.set(self.GROUP_MODE_KEY, stored)
        self._log(f"Group {group}: mode set to {self.MODE_LABELS[mode]}.")
        self._populate_group_grid()

    def _stored_ratios(self, group, members):
        """The captured ratios/differences for `group` as {channel: value},
        or None if none were captured or they were captured for a
        different reference channel, mode, or set of channels than the
        group has now."""
        reference = self._group_reference(group, members)
        mode = self._group_mode(group)
        stored = (app_settings.get(self.GROUP_RATIOS_KEY) or {}).get(group)
        if (not stored or stored.get("lead") != reference or stored.get("mode", "ratio") != mode
                or set(stored.get("ratios", {})) != set(members)):
            return None
        return stored["ratios"]

    def _on_group_config_changed(self, *_args):
        self._populate_group_grid()

    def _populate_group_grid(self):
        # Rebuilt on every Name/Group edit; typed Lead V0 values carry over.
        lead_text = {group: var.get() for group, var in self._group_lead_vars.items()}
        if self._group_grid is not None:
            self._group_grid.destroy()
        grid = self._group_grid = ttk.Frame(self._group_card)
        grid.grid(row=0, column=0, sticky="ew")
        grid.columnconfigure(1, weight=1)
        self._group_lead_vars = {}
        self._group_ref_vars = {}
        self._group_mode_vars = {}

        groups = self._groups()
        if not groups:
            ui_style.add_note(
                grid, 0,
                "No groups yet — type the same name into the Group column of two or more channels.",
                columnspan=6, pady=(4, 4),
            )
            return

        headers = ["Group", "Channels", "Reference", "Mode", "Ratio / diff", "Lead V0", "", "", ""]
        for i, header in enumerate(headers):
            ttk.Label(grid, text=header).grid(
                row=0, column=i, sticky="w", padx=(6, 4) if i == 0 else 2, pady=(4, 2)
            )
        ttk.Separator(grid, orient="horizontal").grid(
            row=1, column=0, columnspan=len(headers), sticky="ew", pady=2)

        for r, (group, members) in enumerate(groups.items(), start=2):
            ttk.Label(grid, text=group).grid(row=r, column=0, sticky="w", padx=(6, 4), pady=1)
            ui_style.make_wrapping_label(grid, ", ".join(self.rows[ch].label for ch in members)).grid(
                row=r, column=1, sticky="ew", padx=2, pady=1
            )

            reference = self._group_reference(group, members)
            ref_var = tk.StringVar(self, value=reference)
            self._group_ref_vars[group] = ref_var
            ref_combo = ttk.Combobox(
                grid, textvariable=ref_var, values=members, width=6, state="readonly"
            )
            ref_combo.grid(row=r, column=2, sticky="w", padx=2, pady=1)
            ref_combo.bind(
                "<<ComboboxSelected>>",
                lambda _e, g=group: self._set_group_reference(g, self._group_ref_vars[g].get()),
            )

            mode = self._group_mode(group)
            mode_var = tk.StringVar(self, value=self.MODE_LABELS[mode])
            self._group_mode_vars[group] = mode_var
            mode_combo = ttk.Combobox(
                grid, textvariable=mode_var, values=list(self.MODE_LABELS.values()),
                width=9, state="readonly",
            )
            mode_combo.grid(row=r, column=3, sticky="w", padx=2, pady=1)
            mode_combo.bind(
                "<<ComboboxSelected>>",
                lambda _e, g=group: self._set_group_mode(g, self.MODE_VALUES[self._group_mode_vars[g].get()]),
            )

            ratios = self._stored_ratios(group, members)
            if ratios is not None:
                fmt = "{:+.4g}".format if mode == "difference" else "{:.4g}".format
                ratio_label = ttk.Label(grid, text=" : ".join(fmt(ratios[ch]) for ch in members))
            elif (app_settings.get(self.GROUP_RATIOS_KEY) or {}).get(group):
                ratio_label = ttk.Label(grid, text="Out of date — capture again", foreground=ui_style.WARNING)
            else:
                ratio_label = ttk.Label(grid, text="Not captured", foreground=ui_style.WARNING)
            ratio_label.grid(row=r, column=4, sticky="w", padx=2, pady=1)

            lead_var = tk.StringVar(self, value=lead_text.get(group, self.rows[reference].v0_var.get()))
            self._group_lead_vars[group] = lead_var
            lead_entry = ttk.Entry(grid, textvariable=lead_var, width=6, justify="right")
            lead_entry.grid(row=r, column=5, padx=2, pady=1)
            lead_entry.bind("<Return>", lambda _e, g=group: self._apply_group(g))

            ttk.Button(grid, text="Toggle", width=6, command=lambda g=group: self._toggle_group(g)).grid(
                row=r, column=6, padx=2, pady=1
            )
            ttk.Button(grid, text="Capture", command=lambda g=group: self._capture_group_ratios(g)).grid(
                row=r, column=7, padx=2, pady=1
            )
            ttk.Button(grid, text="Apply", width=6, command=lambda g=group: self._apply_group(g)).grid(
                row=r, column=8, padx=(8, 6), pady=1
            )

        ui_style.add_note(
            grid, len(groups) + 2,
            "Toggle turns every member ON or OFF together: if they're all already in the same state it "
            "flips them the other way, otherwise (mixed, or status not read yet) it turns them all off. "
            "Reference picks which channel the others are measured against (defaults to the first "
            "channel). Mode picks how: Ratio keeps each member a fixed multiple of the reference V0, "
            "Difference keeps each member a fixed number of volts above or below it. Capture records "
            "each member's current V0 relative to the reference under that mode. Apply sets the "
            "reference channel's V0 to Lead V0 and scales/offsets the others to match. Each channel "
            "still ramps at its own rate, so the relationship holds exactly once ramping finishes.",
            columnspan=len(headers), pady=(6, 4),
        )

    def _capture_group_ratios(self, group):
        members = self._groups().get(group)
        if not members:
            return
        try:
            values = {ch: float(self.rows[ch].v0_var.get()) for ch in members}
        except ValueError:
            self._log(f"Group {group}: every channel needs a numeric V0 set value to capture.")
            return
        lead = self._group_reference(group, members)
        mode = self._group_mode(group)

        if mode == "difference":
            factors = {ch: values[ch] - values[lead] for ch in members}
            noun, fmt = "differences", "{:+.4g}".format
        else:
            if values[lead] <= 0:
                self._log(
                    f"Group {group}: reference channel {self.rows[lead].label} needs a V0 above 0 "
                    "to capture ratios."
                )
                return
            factors = {ch: values[ch] / values[lead] for ch in members}
            noun, fmt = "ratios", "{:.4g}".format

        stored = dict(app_settings.get(self.GROUP_RATIOS_KEY) or {})
        stored[group] = {"lead": lead, "mode": mode, "ratios": factors}
        app_settings.set(self.GROUP_RATIOS_KEY, stored)
        self._group_lead_vars[group].set(_format_setpoint(values[lead]))
        self._log(
            f"Group {group}: captured V0 {noun} (reference {self.rows[lead].label}) "
            + ", ".join(f"{self.rows[ch].label} {fmt(factors[ch])}" for ch in members)
        )
        self._populate_group_grid()

    def _apply_group(self, group):
        try:
            control = self.group_control(group)
        except PowerSupplyUnavailable as exc:
            self._log(str(exc))
            return
        try:
            lead_v0 = float(self._group_lead_vars[group].get())
        except ValueError:
            self._log(f"Group {group}: invalid Lead V0.")
            return
        if lead_v0 < 0:
            self._log(f"Group {group}: Lead V0 can't be negative.")
            return

        targets, _ = control.set_lead_v0(lead_v0)
        for ch, value in targets.items():
            self.rows[ch].v0_var.set(_format_setpoint(value))

    # -- driving a group from another thread --

    def group_names(self):
        """Every configured group name, in table order."""
        return list(self._groups())

    def group_control(self, group):
        """A GroupVoltageControl for `group`: the handle other tabs use to
        set its voltages off the GUI thread (the parameter sweep lowering
        them while the stage moves). Raises PowerSupplyUnavailable if this
        tab isn't connected, there is no such group, or its ratios/
        differences haven't been captured -- the same three things Apply
        refuses on."""
        groups = self._groups()
        members = groups.get(group)
        if not members:
            known = ", ".join(groups) or "none configured"
            raise PowerSupplyUnavailable(
                f"No channel group named {group!r} on the Power Supply tab (groups: {known})."
            )
        if self.worker is None:
            raise PowerSupplyUnavailable("Power Supply tab isn't connected to the crate.")
        mode = self._group_mode(group)
        factors = self._stored_ratios(group, members)
        if factors is None:
            noun = "differences" if mode == "difference" else "ratios"
            raise PowerSupplyUnavailable(
                f"Group {group}: capture {noun} first (none captured for its current channels/mode)."
            )
        return GroupVoltageControl(self, group, members, mode, factors)

    def group_lead_v0(self, group):
        """`group`'s reference channel V0 set value, as the text shown in
        the channel table (the device's own value, refreshed on every
        status read), or None if there's no such group or no value read
        yet."""
        members = self._groups().get(group)
        if not members:
            return None
        reference = self._group_reference(group, members)
        return self.rows[reference].v0_var.get().strip() or None

    def status_snapshot(self):
        """(monotonic timestamp, {channel: status fields}) of the last
        DISPLAY STATUS read -- safe to call from any thread; the tuple is
        replaced on every read rather than mutated in place."""
        return self._status_snapshot

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
        self._status_snapshot = (time.monotonic(), data)
        if "hv_enable" in header:
            state = header["hv_enable"]
            self.hv_enable_var.set(state)
            self.hv_enable_label.config(foreground=ui_style.RUNNING if state == "ON" else ui_style.IDLE)
        if "active_v" in header:
            self.active_var.set(header["active_v"])
        if "group" in header:
            self.group_var.set(header["group"])

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
