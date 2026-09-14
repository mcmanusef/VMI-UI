"""Manual stage control: connect to a Newport XPS-D controller, initialize
and home it, and jog it to arbitrary positions by hand.

This is the single owner of the stage connection -- Parameter Sweep drives
its sweeps through whatever XPSStage instance is connected here (see
get_stage()) rather than opening a second, independent connection to the
same physical stage.
"""
import threading
import time
from queue import Queue, Empty

import qtk as tk
from qtk import ttk

import app_settings
import ui_style
import xps_client


def describe_move_failure(stage, group_name, exc):
    """XPS motion errors (e.g. -22 "Not allowed action") usually mean the
    group isn't in a state that accepts motion commands -- most often it
    just hasn't been homed yet. Rather than guess at status codes (they
    vary by firmware), ask the controller for its own current status text
    and fold that into the message so the operator can see why. Shared with
    sweep_interface.py, which drives moves through this tab's connection
    (see StageInterface.get_stage())."""
    try:
        _, status_text = stage.status(group_name)
        return f"{exc} -- current group status: {status_text}. (Try Home if it hasn't been homed yet.)"
    except Exception:
        return str(exc)


class StageInterface(ttk.Frame):
    def __init__(self, parent, coordinator=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)

        # Persisted since the IP/group don't change often.
        self.stage_ip_var = app_settings.persistent_var(self, tk.StringVar, "xps.ip", "192.168.93.51")
        self.group_var = app_settings.persistent_var(self, tk.StringVar, "xps.group", "")
        # Plain-Python mirror of group_var, kept in sync below -- Tkinter/Tcl
        # isn't thread-safe, and _position_poll_loop (a background thread
        # that runs for the app's whole lifetime) needs the group name on
        # every tick without ever calling .get() on the Tk variable itself
        # from off the main thread.
        self._group_name_cache = self.group_var.get().strip()
        self.group_var.trace_add("write", lambda *_: setattr(self, "_group_name_cache", self.group_var.get().strip()))
        self.zero_offset_var = app_settings.persistent_var(self, tk.DoubleVar, "xps.zero_position", 0.0)
        self.zero_display_var = tk.StringVar(self, value=f"{self.zero_offset_var.get():.4f}")
        self.zero_offset_var.trace_add(
            "write", lambda *_: self.zero_display_var.set(f"{self.zero_offset_var.get():.4f}")
        )
        self.current_position_var = tk.StringVar(self, value="--")
        self.group_state_var = tk.StringVar(self, value="--")
        self.move_target_var = tk.StringVar(self, value="0")
        self.stage_status_var = tk.StringVar(self, value="Not connected.")
        self.objects_var = tk.StringVar(self, value="--")

        self._stage = None
        self._queue: Queue = Queue()

        # Continuous 10 Hz position/group-state poll, running for as long as
        # the widget exists (no-ops until connected). Runs concurrently with
        # whatever a move/sweep is doing on the same stage connection --
        # XPSStage serializes access to the socket itself, so this is safe.
        self._position_poll_thread = threading.Thread(target=self._position_poll_loop, daemon=True)

        self._build_ui()
        self._poll_queue()
        self._position_poll_thread.start()

    # ---- public API for other tabs (e.g. Parameter Sweep) -----------------

    def get_stage(self):
        """The connected xps_client.XPSStage, or None if not connected yet.
        Safe to drive from another tab's background thread concurrently
        with the 10 Hz position poll above -- XPSStage serializes access to
        the socket itself."""
        return self._stage

    def is_running(self):
        return False  # Nothing here for the coordinator to stop.

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        # No data view on this tab, so its controls get the whole tab as a
        # page instead of a sidebar next to an empty main area.
        page = ui_style.build_page_layout(self)

        ui_style.build_button_bar(page, 0, [
            ("Connect", self._connect_stage),
            ("Initialize", self._initialize_stage),
            ("Home", self._home_stage),
        ])
        # Green for as long as a connection is up, not just while a status
        # word like "Moving..." happens to be showing.
        self._status_block = ui_style.StatusBlock(
            page, 1, self.stage_status_var,
            is_running=lambda text: "connecting" in text or bool(self._stage and self._stage.connected),
        )

        stage = ui_style.build_section(page, 2, "Stage (Newport XPS-D)")
        ui_style.add_form_row(stage, 0, "IP:", ttk.Entry(stage, textvariable=self.stage_ip_var, width=18))
        ui_style.add_form_row(stage, 1, "Group name:", ttk.Entry(stage, textvariable=self.group_var, width=18))
        objects_row = ui_style.add_stat_rows(stage, [("Group state:", self.group_state_var)], start_row=2)
        ttk.Label(stage, text="Objects:", font=ui_style.SMALL_FONT).grid(
            row=objects_row, column=0, sticky="nw", pady=(6, 0)
        )
        ui_style.add_note(stage, objects_row + 1, textvariable=self.objects_var, pady=0)

        manual = ui_style.build_section(page, 3, "Manual control", pady=0)
        ui_style.add_stat_rows(manual, [("Position (rel. zero):", self.current_position_var)])
        zero_entry = ttk.Entry(manual, textvariable=self.zero_display_var, width=12)
        zero_entry.bind("<Return>", self._commit_zero_edit)
        zero_entry.bind("<FocusOut>", self._commit_zero_edit)
        ui_style.add_form_row(manual, 1, "Saved zero (raw):", zero_entry)
        ui_style.add_form_row(manual, 2, "Move to:", ttk.Entry(manual, textvariable=self.move_target_var, width=12))
        ui_style.build_button_bar(manual, 3, [
            ("Move", self._move_stage),
            ("Refresh", self._refresh_position),
            ("Set Current as Zero", self._set_zero),
        ], pady=(4, 6), columnspan=2)
        ui_style.add_note(
            manual, 4,
            'A Move failing with "Not allowed action" usually means the stage '
            "hasn't been initialized/homed yet -- try Initialize, then Home. This "
            "connection is shared with Parameter Sweep.",
        )

    # ---- stage connection / manual control (background threads) ----------

    def _require_stage_group(self):
        if not self._stage or not self._stage.connected:
            self.stage_status_var.set("Connect to the stage first.")
            return False
        if not self.group_var.get().strip():
            self.stage_status_var.set("Group name is required.")
            return False
        return True

    def _connect_stage(self):
        ip = self.stage_ip_var.get().strip()
        if not ip:
            self.stage_status_var.set("IP is required.")
            return
        self.stage_status_var.set("Connecting...")
        threading.Thread(target=self._connect_worker, args=(ip,), daemon=True).start()

    def _connect_worker(self, ip):
        try:
            stage = xps_client.XPSStage(ip)
            stage.connect()
            objects = stage.objects_list()
        except Exception as exc:
            self._queue.put({"stage_error": str(exc)})
            return
        self._stage = stage
        # ObjectsListGet returns every object (groups AND their individual
        # positioners, e.g. "Group2;Group2.Pos;..."); positioner entries
        # have a '.' in the name, group names don't -- keep only the
        # group-level entries, since that's what Group* commands take.
        groups = sorted({
            entry.strip() for entry in objects.split(";") if entry.strip() and "." not in entry.strip()
        })
        self._queue.put({"stage_connected": True, "objects": ", ".join(groups)})

    def _initialize_stage(self):
        if not self._require_stage_group():
            return
        self.stage_status_var.set("Initializing...")
        threading.Thread(
            target=self._initialize_worker, args=(self.group_var.get().strip(),), daemon=True
        ).start()

    def _initialize_worker(self, group_name):
        try:
            self._stage.initialize(group_name)
        except Exception as exc:
            self._queue.put({"stage_error": str(exc)})
            return
        self._queue.put({"stage_initialized": True})

    def _home_stage(self):
        if not self._require_stage_group():
            return
        self.stage_status_var.set("Homing...")
        threading.Thread(target=self._home_worker, args=(self.group_var.get().strip(),), daemon=True).start()

    def _home_worker(self, group_name):
        try:
            self._stage.home(group_name)
            pos = self._stage.wait_for_settle(group_name, timeout=120.0)
        except Exception as exc:
            self._queue.put({"stage_error": str(exc)})
            return
        self._queue.put({"stage_homed": True, "position": pos})

    def _move_stage(self):
        if not self._require_stage_group():
            return
        try:
            target = float(self.move_target_var.get())
        except ValueError:
            self.stage_status_var.set("Invalid target position.")
            return
        self.stage_status_var.set("Moving...")
        threading.Thread(
            target=self._move_worker,
            args=(self.group_var.get().strip(), self.zero_offset_var.get() + target),
            daemon=True,
        ).start()

    def _move_worker(self, group_name, absolute_target):
        try:
            self._stage.move_absolute(group_name, absolute_target)
            pos = self._stage.wait_for_settle(group_name, timeout=120.0)
        except Exception as exc:
            self._queue.put({"stage_error": describe_move_failure(self._stage, group_name, exc)})
            return
        self._queue.put({"stage_moved": True, "position": pos})

    def _refresh_position(self):
        if not self._require_stage_group():
            return
        threading.Thread(target=self._refresh_worker, args=(self.group_var.get().strip(),), daemon=True).start()

    def _refresh_worker(self, group_name):
        try:
            pos = self._stage.position(group_name)
        except Exception as exc:
            self._queue.put({"stage_error": str(exc)})
            return
        self._queue.put({"stage_position": pos})

    def _set_zero(self):
        if not self._require_stage_group():
            return
        threading.Thread(target=self._set_zero_worker, args=(self.group_var.get().strip(),), daemon=True).start()

    def _set_zero_worker(self, group_name):
        try:
            pos = self._stage.position(group_name)
        except Exception as exc:
            self._queue.put({"stage_error": str(exc)})
            return
        self._queue.put({"stage_zero_set": True, "position": pos})

    def _update_position_display(self, raw_position):
        relative = raw_position - self.zero_offset_var.get()
        self.current_position_var.set(f"{relative:.4f}")

    def _commit_zero_edit(self, _event=None):
        """The "Saved zero" field is directly editable (not just settable
        via "Set Current as Zero") -- commits on Enter or losing focus.
        Invalid text just reverts to the last valid value rather than
        raising, same as every other numeric field in this app."""
        try:
            value = float(self.zero_display_var.get())
        except ValueError:
            self.zero_display_var.set(f"{self.zero_offset_var.get():.4f}")
            return
        self.zero_offset_var.set(value)  # persists automatically; re-formats this field too

    def _position_poll_loop(self):
        """Runs for the lifetime of the widget, refreshing the displayed
        position and group state at 10 Hz whenever a stage is connected and
        a group name is set. Silently skips a tick on error rather than
        spamming the status line 10x/second -- a real problem still shows
        up clearly the moment a manual action (Move, Home, ...) is tried.
        Keeps ticking even while Parameter Sweep is driving the same
        connection from its own background thread -- XPSStage serializes
        access to the socket itself.
        """
        while True:
            stage = self._stage
            group_name = self._group_name_cache
            if stage is not None and stage.connected and group_name:
                try:
                    pos = stage.position(group_name)
                    _, status_text = stage.status(group_name)
                    self._queue.put({"stage_position": pos, "stage_group_state": status_text})
                except Exception:
                    pass
            time.sleep(0.1)  # 10 Hz

    # ---- main-thread queue handling -----------------------------------

    def _poll_queue(self):
        try:
            while True:
                result = self._queue.get_nowait()
                self._apply_result(result)
        except Empty:
            pass
        self.after(100, self._poll_queue)

    def _apply_result(self, result):
        if "stage_error" in result:
            self.stage_status_var.set(f"Error: {result['stage_error']}")
            return
        if result.get("stage_connected"):
            self.stage_status_var.set("Connected.")
            self.objects_var.set(result.get("objects", ""))
            return
        if result.get("stage_initialized"):
            self.stage_status_var.set("Initialized.")
            return
        if result.get("stage_homed"):
            self._update_position_display(result["position"])
            self.stage_status_var.set("Homed.")
            return
        if result.get("stage_moved"):
            self._update_position_display(result["position"])
            self.stage_status_var.set("Move complete.")
            return
        if "stage_position" in result:
            self._update_position_display(result["stage_position"])
            if "stage_group_state" in result:
                self.group_state_var.set(result["stage_group_state"])
            return
        if result.get("stage_zero_set"):
            self.zero_offset_var.set(result["position"])
            self._update_position_display(result["position"])
            self.stage_status_var.set(f"Zero set at raw position {result['position']:.4f}.")
            return
