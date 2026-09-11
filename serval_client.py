"""Small Serval HTTP helpers for the "one raw file per trigger, poll for it"
acquisition pattern shared by the Diagnostics and Monitored Acquisition tabs.
"""
import json
import math
import pathlib
import time

import requests

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
                "Base": pathlib.Path(dest_dir).as_uri(),
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


def start_measurement(server, timeout=5):
    SESSION.get(f"{server}/measurement/start", timeout=timeout)


def stop_measurement(server, timeout=5):
    SESSION.get(f"{server}/measurement/stop", timeout=timeout)


def wait_for_measurement_idle(server, timeout=30.0, poll=0.2):
    """Poll /dashboard until Measurement.Status no longer looks like an
    active recording/preparing state, or `timeout` elapses.

    stop_measurement() is fire-and-forget -- it doesn't confirm Serval has
    actually wound down. That's usually fine, but under CONTINUOUS mode
    with a very large nTriggers (configure_continuous_destination), Serval
    can take a real moment to actually stop, and a destination/config PUT
    issued for the *next* run while it's still stopping the previous one
    can hang -- this closes that race by confirming idle first. Best-effort:
    if /dashboard itself is unreachable or the status text doesn't look
    like anything recognizable, just returns after `timeout` rather than
    blocking forever.
    """
    # Substrings of known/likely "still busy" states -- "STOPPING" (in
    # progress) specifically, not the bare "STOP" that a legitimate final
    # "STOPPED"/"IDLE" status could also be mistaken to contain.
    busy_markers = ("RECORD", "PREPAR", "STOPPING", "STARTING")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = SESSION.get(f"{server}/dashboard", timeout=5)
            status = str(json.loads(resp.text).get("Measurement", {}).get("Status", "")).upper()
            if status and not any(marker in status for marker in busy_markers):
                return status
        except Exception:
            pass
        time.sleep(poll)
    return None


def wait_for_new_file(directory: pathlib.Path, stop_event, last_path, last_mtime, timeout=5.0, settle=0.2, poll=0.1):
    """Poll `directory` for a *.tpx3 file that is newer than (last_path,
    last_mtime) and appears to have finished being written (settle seconds
    since its mtime). Returns None if stopped or nothing shows up in time.
    """
    start = time.time()
    while not stop_event.is_set():
        files = sorted(directory.glob("*.tpx3"), key=lambda p: p.stat().st_mtime)
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
