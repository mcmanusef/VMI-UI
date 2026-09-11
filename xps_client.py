"""Newport XPS-D socket client: connectivity + motion.

Extends the read-only protocol already verified in test_xps_connection.py
(same "Command(args)" -> "<errorCode>,<value>,...,EndOfAPI" framing, per the
XPS Unified Programmer's Manual) with the motion commands needed to drive a
stage, using the exact command strings confirmed against Newport's own
reference driver (pyepics/newportxps, XPS_C8_drivers.py):

    GroupHomeSearch(group)
    GroupMoveAbsolute(group,position)
    GroupPositionCurrentGet(group,double *)
    GroupKill(group) / GroupInitialize(group)

This module never homes or moves anything on its own -- every motion call
here is explicit; the caller (sweep_interface.py) decides when to use them.
"""
import socket
import threading
import time

XPS_PORT = 5001  # Newport XPS command interface's standard port.
XPS_TIMEOUT = 5.0  # seconds

_END_OF_API = ",EndOfAPI"


class XPSError(Exception):
    """Raised when the controller returns a non-zero error code."""


def connect(ip, port=XPS_PORT, timeout=XPS_TIMEOUT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((ip, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def send_and_receive(sock, command):
    """Send one XPS command string and return (error_code, value_string)."""
    sock.send(command.encode("ascii"))
    reply = sock.recv(1024).decode("ascii", errors="replace")
    while _END_OF_API not in reply:
        reply += sock.recv(1024).decode("ascii", errors="replace")
    body = reply[: -len(_END_OF_API)]
    error_str, _, value = body.partition(",")
    return int(error_str), value


def call(sock, command, *, check=True):
    error, value = send_and_receive(sock, command)
    if check and error != 0:
        try:
            _, message = send_and_receive(sock, f"ErrorStringGet({error}, char *)")
        except Exception:
            message = "<could not retrieve error string>"
        raise XPSError(f"{command} -> error {error}: {message}")
    return value


def login(sock, username, password):
    call(sock, f"Login({username},{password})")


def firmware_version_get(sock):
    return call(sock, "FirmwareVersionGet(char *)")


def objects_list_get(sock):
    """Semicolon-separated list of configured groups/positioners."""
    return call(sock, "ObjectsListGet(char *)")


def group_status_get(sock, group_name):
    return int(call(sock, f"GroupStatusGet({group_name},int *)"))


def group_status_string_get(sock, status_code):
    return call(sock, f"GroupStatusStringGet({status_code}, char*)")


# ---- motion -----------------------------------------------------------

def group_home_search(sock, group_name):
    call(sock, f"GroupHomeSearch({group_name})")


def group_move_absolute(sock, group_name, position):
    call(sock, f"GroupMoveAbsolute({group_name},{position})")


def group_position_current_get(sock, group_name):
    return float(call(sock, f"GroupPositionCurrentGet({group_name},double *)"))


def group_kill(sock, group_name):
    call(sock, f"GroupKill({group_name})")


def group_initialize(sock, group_name):
    call(sock, f"GroupInitialize({group_name})")


class XPSStage:
    """Thin, explicit wrapper: connect/login once, then home/move/read as
    separate calls the UI drives directly. Never homes or moves on its own.

    The XPS command protocol is strictly one-command-then-its-response on a
    single persistent socket -- it has no way to tell two interleaved
    commands' responses apart. Since a background poller (continuous
    position/status refresh) now runs concurrently with whatever a move or
    a sweep is doing on the same socket, every public method that touches
    the socket holds `_lock` for just that one command/response, so callers
    never desync each other's replies. Held per-command rather than for an
    entire wait_for_settle() call, so a multi-minute settle-wait doesn't
    starve the poller the whole time.
    """

    def __init__(self, ip, port=XPS_PORT, username="Administrator", password="Administrator", timeout=XPS_TIMEOUT):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout
        self._sock = None
        self.firmware_version = None
        self._lock = threading.Lock()

    @property
    def connected(self):
        return self._sock is not None

    def connect(self):
        with self._lock:
            self.close()
            self._sock = connect(self.ip, self.port, self.timeout)
            login(self._sock, self.username, self.password)
            self.firmware_version = firmware_version_get(self._sock)

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def objects_list(self):
        with self._lock:
            return objects_list_get(self._sock)

    def initialize(self, group_name):
        with self._lock:
            group_initialize(self._sock, group_name)

    def home(self, group_name):
        with self._lock:
            group_home_search(self._sock, group_name)

    def move_absolute(self, group_name, position, retry_timeout=120.0, retry_interval=1.0):
        """Send GroupMoveAbsolute, retrying once per `retry_interval`
        seconds if it fails (a transient communication hiccup, the group
        momentarily reporting busy, ...) instead of giving up on the first
        attempt. Only raises the (last) error once `retry_timeout` seconds
        have passed with every attempt still failing."""
        start = time.time()
        last_exc = None
        while time.time() - start < retry_timeout:
            try:
                with self._lock:
                    group_move_absolute(self._sock, group_name, position)
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(retry_interval)
        raise last_exc

    def position(self, group_name):
        with self._lock:
            return group_position_current_get(self._sock, group_name)

    def status(self, group_name):
        with self._lock:
            code = group_status_get(self._sock, group_name)
            text = group_status_string_get(self._sock, code)
            return code, text

    def wait_for_settle(self, group_name, timeout=120.0, poll=0.1, settle_reads=3):
        """Poll the measured position until it stops changing for
        `settle_reads` consecutive reads AND the controller no longer
        reports itself as moving, or it goes `timeout` seconds without the
        position changing at all. Returns the final measured position.

        `timeout` is a stall timeout, not an overall move-duration cap: it
        only starts counting from the last time the position actually
        changed, and resets every time it does. A long but genuinely
        progressing move (a big travel distance, a slow stage, ...) is
        never cut short partway through just for taking a while -- only a
        move that's stopped making any progress trips it.

        Position stability alone isn't quite enough: the controller can
        still be finishing a move (e.g. a settling/backlash phase) for a
        moment after the reported position has stopped ticking, and issuing
        the next GroupMoveAbsolute while it's still "MOVING" gets rejected
        with error -22 "Not allowed action". We don't hardcode the numeric
        status codes (they vary by firmware) -- just check the status text
        for "moving" as an extra confirmation, falling back to position-only
        if the status text is ever unrecognizable so this never blocks
        forever on a wording mismatch.
        """
        last = self.position(group_name)
        stable = 0
        last_change_time = time.time()
        while time.time() - last_change_time < timeout:
            time.sleep(poll)
            try:
                current = self.position(group_name)
            except Exception:
                # A transient read hiccup -- e.g. colliding with the
                # continuous 10 Hz position/state poll on the same
                # connection -- isn't a real move failure, and it isn't
                # "stopped changing" either. Back off to a 1-second retry
                # cadence (rather than hammering at the normal `poll` rate)
                # and keep trying; the `timeout` stall clock above is what
                # actually gives up, and only after it goes the full 2
                # minutes with nothing but failures.
                time.sleep(1.0)
                continue
            if abs(current - last) < 1e-6:
                stable += 1
                if stable >= settle_reads and not self._is_moving(group_name):
                    return current
            else:
                stable = 0
                last_change_time = time.time()
            last = current
        return last

    def _is_moving(self, group_name):
        try:
            _, text = self.status(group_name)
        except Exception:
            # Can't tell -- assume it's still moving rather than falsely
            # declaring settled (which would let the caller send the next
            # move command too early and get rejected).
            return True
        return "mov" in text.lower()
