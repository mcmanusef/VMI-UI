"""Small Serval HTTP helpers for the "one raw file per frame, poll for it"
acquisition patterns shared by the acquisition tabs: a fresh start per
frame (configure_single_trigger_destination) or one start per batch of
frames (configure_continuous_destination, restart_continuous_batch).
"""
import json
import math
import pathlib
import threading
import time

import requests

import platform_paths

# Shared, reused connection to Serval. Two reasons to use this instead of
# bare requests.get/put() (which each open a brand-new TCP connection and
# throw it away -- requests.api.request() literally does
# `with sessions.Session() as session: ...` under the hood for every single
# bare call):
#   1. trust_env=False turns off requests' default behaviour of reading
#      HTTP_PROXY/HTTPS_PROXY/NO_PROXY (and the Windows system proxy
#      settings) from the process environment. Serval is always local/LAN;
#      a proxy should never legitimately sit between us and it, and a
#      misconfigured/unreachable one would explain calls failing fast from
#      inside the long-running UI process while a freshly-launched
#      standalone script (different/inherited env) reaches the exact same
#      server fine at the exact same time.
#   2. Connection reuse (keep-alive) instead of a fresh TCP handshake (and
#      TIME_WAIT socket) per call.
# Every module in this app that talks to Serval should import and use this
# SESSION rather than calling the bare `requests` module directly.
SESSION = requests.Session()
SESSION.trust_env = False

class DashboardPoller:
    """Fetches /dashboard on a background thread so the GUI thread never
    waits on Serval: with Serval down or unreachable, each request blocks
    for its full connect timeout (twice over for "localhost", which tries
    IPv6 then IPv4), which froze the whole window for ~2 s out of every 4
    when this ran from a UI timer.

    The GUI side calls set_server() with the current URL and reads
    `latest` -- (data, error), one of which is None -- from its own timer;
    both are plain attribute swaps, safe across threads.
    """

    def __init__(self, interval=2.0, timeout=1.0):
        self.latest = (None, None)
        self._server = None
        self._interval = interval
        self._timeout = timeout
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_server(self, server):
        self._server = server

    def _run(self):
        while True:
            server = self._server
            if server:
                try:
                    resp = SESSION.get(f"{server}/dashboard", timeout=self._timeout)
                    self.latest = (json.loads(resp.text), None)
                except Exception as exc:
                    self.latest = (None, exc)
            time.sleep(self._interval)


# NOT "comfortably under MaxPulseCount" (2147483647) -- that was the
# original idea (an effectively-unbounded placeholder for "run until we
# say stop"), but per the trigger-mode manual, nTriggers in CONTINUOUS mode
# is literally "equal to the number of raw data files produced". Combined
# with SplitStrategy: FRAME (one file per trigger, see _configure_destination
# below), asking Serval to prepare for two billion individual files is what
# was hanging the destination PUT -- not SplitStrategy itself. The dwell/
# target-frame logic in each tab is what actually decides when to stop, so
# this just needs to comfortably outlast any realistic single run, not
# approach the int32 ceiling.
LONG_RUN_TRIGGERS = 50_000

# No single CONTINUOUS-mode measurement (one configure + one
# start_measurement() call) should run longer than this before the caller
# reconfigures/restarts it for the next batch of frames -- keeps every
# individual Serval call short-lived and bounded instead of trusting one
# start command to faithfully keep going for an entire (potentially very
# long) run.
MAX_BATCH_SECONDS = 60.0


def max_frames_per_batch(frame_time):
    """The most frames a single CONTINUOUS-mode start_measurement() should
    be configured for, so that batch can't run longer than
    MAX_BATCH_SECONDS before the caller has to reconfigure + issue a fresh
    start for the next one. Always at least 1 -- a frame_time longer than
    MAX_BATCH_SECONDS on its own can't be split any further, so that single
    frame is its own (over-length) batch."""
    if frame_time <= 0:
        return 1
    return max(1, math.floor(MAX_BATCH_SECONDS / frame_time))


def configure_single_trigger_destination(server, frame_time, dest_dir, file_pattern, timeout=30):
    """Continuous mode, one trigger per raw file, writing into dest_dir.
    Caller must call start_measurement() again for every frame (nTriggers=1
    means Serval auto-stops after each one)."""
    _configure_destination(server, frame_time, dest_dir, file_pattern, n_triggers=1, timeout=timeout)


def configure_continuous_destination(server, frame_time, dest_dir, file_pattern, n_triggers=None, timeout=30):
    """Continuous mode, writing into dest_dir. One start_measurement() call
    then captures many frames in a row -- Serval keeps writing a new raw
    file every frame_time on its own -- so the caller only needs to
    (re)issue start/stop or reconfigure this when something actually
    changes (a new destination folder, a different frame_time, or genuinely
    stopping), not once per frame.

    Pass `n_triggers` whenever the caller already knows exactly how many
    frames it wants (a fixed target-frame count, a calibrated cluster-dwell
    frame count, a single calibration frame, ...) -- nTriggers in
    CONTINUOUS mode is literally "the number of raw data files produced",
    so this tells Serval the real number instead of the LONG_RUN_TRIGGERS
    placeholder, letting it auto-stop at exactly the right frame instead of
    relying solely on this app's own dwell/target-frame check. Omit it (or
    pass None) for a genuinely open-ended run (until the user stops it),
    which is what LONG_RUN_TRIGGERS is for."""
    _configure_destination(
        server, frame_time, dest_dir, file_pattern,
        n_triggers=n_triggers if n_triggers else LONG_RUN_TRIGGERS,
        timeout=timeout,
    )


def _configure_destination(server, frame_time, dest_dir, file_pattern, n_triggers, timeout):
    # A config / destination change while a measurement is still running
    # (e.g. one another tab just asked to stop) is rejected or hangs.
    ensure_idle(server)
    # timeout defaults to 30s, not the 5s used for the simpler start/stop
    # GETs below -- applying a destination change (especially switching
    # SplitStrategy) appears to genuinely take Serval longer than a plain
    # config read, and this gets called once per sweep position, so a too-
    # short timeout here was failing under real use even though the exact
    # same PUT succeeds fine standalone with more room to breathe.
    config = json.loads(SESSION.get(f"{server}/detector/config", timeout=timeout).text)
    config["TriggerMode"] = "CONTINUOUS"
    config["ExposureTime"] = frame_time
    config["TriggerPeriod"] = frame_time
    config["nTriggers"] = n_triggers
    SESSION.put(f"{server}/detector/config", data=json.dumps(config), timeout=timeout)

    destination = {
        "Raw": [
            {
                "Base": platform_paths.file_uri(dest_dir),
                "FilePattern": file_pattern,
                # Undocumented field (found by hand -- not in the manual or
                # searchable anywhere); "SINGLE_FILE" is the default and
                # bundles every frame from one run into one growing file.
                # "FRAME" splits into one file per frame, which is what the
                # whole raw-file-per-trigger collector pattern here (and
                # especially configure_continuous_destination's one-start,
                # many-frames design) depends on.
                "SplitStrategy": "FRAME",
            }
        ]
    }
    SESSION.put(f"{server}/server/destination", data=json.dumps(destination), timeout=timeout)


def restart_continuous_batch(server, frame_time, n_triggers, timeout=5):
    """Start the next CONTINUOUS-mode batch of `n_triggers` frames once the
    previous one has auto-stopped. Only /detector/config is reset; the
    destination set by configure_continuous_destination stays as it is."""
    ensure_idle(server)
    config = json.loads(SESSION.get(f"{server}/detector/config", timeout=timeout).text)
    config["TriggerMode"] = "CONTINUOUS"
    config["ExposureTime"] = frame_time
    config["TriggerPeriod"] = frame_time
    config["nTriggers"] = n_triggers
    SESSION.put(f"{server}/detector/config", data=json.dumps(config), timeout=timeout)
    start_measurement(server, timeout=timeout)


def start_measurement(server, timeout=5):
    # Raise on a rejected start (e.g. Serval still busy finishing the
    # previous measurement) rather than letting the caller wait out its
    # whole new-file timeout for a frame that was never started.
    SESSION.get(f"{server}/measurement/start", timeout=timeout).raise_for_status()


def stop_measurement(server, timeout=5):
    SESSION.get(f"{server}/measurement/stop", timeout=timeout)


# Substrings of Measurement.Status values that mean Serval is still busy --
# "STOPPING" specifically, not the bare "STOP" a final "STOPPED" contains.
_BUSY_MARKERS = ("RECORD", "PREPAR", "STOPPING", "STARTING")


def measurement_status(server, timeout=5):
    """Measurement.Status from /dashboard, upper-cased. "" when there is no
    measurement at all (Serval reports "Measurement": null until the first
    one, which used to raise here and read as "busy" until a timeout)."""
    resp = SESSION.get(f"{server}/dashboard", timeout=timeout)
    measurement = json.loads(resp.text).get("Measurement") or {}
    return str(measurement.get("Status") or "").upper()


def _is_busy(status):
    return any(marker in status for marker in _BUSY_MARKERS)


def measurement_busy(server, timeout=2):
    """True while /dashboard reports a measurement recording, preparing,
    starting or stopping."""
    return _is_busy(measurement_status(server, timeout=timeout))


def wait_for_measurement_idle(server, timeout=30.0, poll=0.2):
    """Poll /dashboard until Measurement.Status no longer looks like an
    active recording/preparing state, or `timeout` elapses. Returns the
    idle status ("IDLE" when there is no measurement), or None on timeout.

    stop_measurement() is fire-and-forget -- it doesn't confirm Serval has
    actually wound down, and a start or destination/config PUT issued while
    it is still stopping is rejected or can hang. Best-effort: if
    /dashboard itself is unreachable, returns None after `timeout` rather
    than blocking forever.
    """
    start = time.time()
    while True:
        try:
            status = measurement_status(server)
            if not _is_busy(status):
                return status or "IDLE"
        except Exception:
            pass
        if time.time() - start >= timeout:
            return None
        time.sleep(poll)


class ServalBusy(RuntimeError):
    """Serval is still running a measurement that won't stop."""


def ensure_idle(server, timeout=30.0):
    """Make sure no measurement is running before a start or a
    destination/config change: wait for the current one to end on its own,
    and if it doesn't within `timeout`, stop it and wait again. Raises
    ServalBusy if Serval still reports a measurement after that, and the
    request's own error straight away if Serval can't be reached at all."""
    if not _is_busy(measurement_status(server)):
        return
    if wait_for_measurement_idle(server, timeout=timeout) is not None:
        return
    try:
        stop_measurement(server)
    except Exception:
        pass
    if wait_for_measurement_idle(server, timeout=15.0) is None:
        raise ServalBusy("Serval is still running a measurement and didn't stop when asked.")


def capture_single_frame(server, directory, frame_time, stop_event, pattern="*.tpx3", settle=0.3,
                         poll=0.1, start_timeout=10.0):
    """Take one frame with a destination set up by
    configure_single_trigger_destination (nTriggers=1) and return the raw
    file(s) it wrote into `directory`, oldest first -- or [] if stop_event
    was set (the measurement is then stopped) or nothing was written.

    Frame completion is taken from Serval itself: the measurement started
    here has to come and go (busy, then idle) before the new file is
    picked up. Waiting for the file's mtime to go quiet instead returns a
    file that is still being written whenever the data rate pauses
    mid-exposure, and the next start then lands while that measurement is
    still running and is rejected.

    Raises ServalBusy if Serval can't be brought idle first, and
    TimeoutError if the frame doesn't finish within frame_time + 30 s."""
    directory = pathlib.Path(directory)
    ensure_idle(server, timeout=start_timeout)
    before = {path: _stat_key(path) for path in directory.glob(pattern)}
    start_measurement(server)
    started = time.time()
    seen_busy = False
    deadline = started + frame_time + 30.0
    while True:
        if stop_event is not None and stop_event.is_set():
            try:
                stop_measurement(server)
                wait_for_measurement_idle(server, timeout=10.0)
            except Exception:
                pass
            return []
        try:
            busy = _is_busy(measurement_status(server))
        except Exception:
            busy = None
        if busy:
            seen_busy = True
        # Done once Serval is idle again -- after having been seen busy, or
        # (in case the whole frame fell between two polls) after at least
        # one exposure's worth of time.
        elif busy is False and (seen_busy or time.time() - started >= frame_time + 0.5):
            break
        if time.time() > deadline:
            try:
                stop_measurement(server)
            except Exception:
                pass
            raise TimeoutError(f"Serval didn't finish a {frame_time:g} s frame within {frame_time + 30:g} s.")
        time.sleep(poll)

    # The file is closed once the measurement ends; give the filesystem a
    # moment for its size to settle (network shares can lag).
    settle_deadline = time.time() + max(5.0, 10 * settle)
    new_files, sizes = [], None
    while time.time() < settle_deadline:
        new_files = sorted(
            (path for path in directory.glob(pattern) if before.get(path) != _stat_key(path)),
            key=lambda path: (path.stat().st_mtime, path.name),
        )
        current = [path.stat().st_size for path in new_files]
        if new_files and current == sizes and all(current):
            return new_files
        sizes = current
        time.sleep(settle)
    return [path for path in new_files if path.stat().st_size > 0]


def _stat_key(path):
    try:
        st = path.stat()
        return st.st_size, st.st_mtime
    except OSError:
        return None


def wait_for_new_file(directory: pathlib.Path, stop_event, last_path, last_mtime, timeout=5.0, settle=0.2, poll=0.1,
                      pattern="*.tpx3"):
    """Poll `directory` for a file matching `pattern` that is newer than
    (last_path, last_mtime) and appears to have finished being written
    (settle seconds since its mtime). Returns None if stopped or nothing
    shows up in time.
    """
    start = time.time()
    while not stop_event.is_set():
        files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
        if files:
            newest = files[-1]
            newest_stat = newest.stat()
            if newest != last_path or newest_stat.st_mtime != last_mtime:
                if newest_stat.st_size > 0 and (time.time() - newest_stat.st_mtime) > settle:
                    return newest
        if time.time() - start > timeout:
            return None
        time.sleep(poll)
    return None
