"""Standalone diagnostic: find a working value for Serval's undocumented
"SplitStrategy" destination field (a field on the "Raw" OutputChannel
object, alongside Base/FilePattern -- confirmed by hand, not documented
anywhere: checked the manual and a general web search, nothing).

Originally this tried to elicit Serval's Jackson enum-deserialization error
(which normally lists every valid constant, e.g. "not one of the values
accepted for Enum class: [...]") by sending an obviously-bogus string. That
didn't work here: PUTting a *valid* config back unchanged succeeds fast,
but adding SplitStrategy with a bogus value hangs rather than failing fast
-- so unlike Format/Mode/IntegrationMode, this field doesn't fail cleanly
on a bad value.

So instead: try a short list of plausible *real* values directly, one at a
time, restoring the known-good config between each attempt. A valid value
should apply cleanly and quickly, the same way the current "SINGLE_FILE"
already does -- so whichever candidate returns 200 OK promptly is very
likely the one you want (a per-frame/independent-file strategy), and
whichever ones hang or get rejected are not.

This matters for the app's Sweep/Monitored Acquisition tabs: they now issue
one CONTINUOUS-mode start per position/run (see
serval_client.configure_continuous_destination) instead of a fresh start
per frame, and depend on Serval writing a separate raw file per frame under
that single run. If SplitStrategy defaults to SINGLE_FILE, that's almost
certainly why frames would land in one growing file instead of separate
ones under that design.

Safety: reads the current /server/destination first and restores it after
every attempt (not just at the end), so a hung/rejected candidate doesn't
leave a later one (or the final state) worse off. Still a live PUT against
whatever server you point it at.

Run directly:
    python test_serval_split_strategy.py [server_url]
    (defaults to http://localhost:8080)
"""
import json
import sys

import requests

DEFAULT_SERVER = "http://localhost:8080"
TEST_FOLDER = "file:/C:/DATA/split_strategy_test"

# Plausible real values for "split output into a separate file per frame",
# given the confirmed existing constant is named SINGLE_FILE. Ordered
# roughly by how likely the naming convention seems.
CANDIDATE_VALUES = [
    "MULTI_FILE",
    "PER_FRAME",
    "PER_TRIGGER",
    "SPLIT_FILE",
    "MULTI_FRAME",
    "FRAME",
    "INDEPENDENT_FILE",
    "PER_MEASUREMENT",
]

ATTEMPT_TIMEOUT = 1  # seconds -- long enough for a real apply, short enough not to waste time on a hang


def server_url():
    url = sys.argv[1].strip() if len(sys.argv) > 1 else DEFAULT_SERVER
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url


def show_dashboard_status(server):
    try:
        resp = requests.get(f"{server}/dashboard", timeout=15)
        data = json.loads(resp.text)
        measurement = data.get("Measurement", {})
        print(f"Dashboard Measurement.Status: {measurement.get('Status')}\n")
    except Exception as exc:
        print(f"  Could not read /dashboard: {exc}\n")


def stop_any_active_measurement(server):
    print("GET /measurement/stop (in case something's still running)...")
    try:
        resp = requests.get(f"{server}/measurement/stop", timeout=30)
        print(f"  {resp.status_code}: {resp.text}\n")
    except Exception as exc:
        print(f"  Stop request failed/timed out: {exc}\n")


def get_destination(server):
    resp = requests.get(f"{server}/server/destination", timeout=15)
    print(f"GET /server/destination -> {resp.status_code}: {resp.text}\n")
    return resp.text


def put_destination(server, description, destination, timeout=ATTEMPT_TIMEOUT):
    print(f"--- {description} ---")
    print("PUT payload:", json.dumps(destination))
    try:
        resp = requests.put(f"{server}/server/destination", data=json.dumps(destination), timeout=timeout)
        print(f"  {resp.status_code}: {resp.text}\n")
        return resp.status_code == 200
    except Exception as exc:
        print(f"  Request failed/timed out: {exc}\n")
        return False


def restore(server, original_text):
    try:
        resp = requests.put(f"{server}/server/destination", data=original_text, timeout=ATTEMPT_TIMEOUT)
        ok = resp.status_code == 200
        print(f"  Restored known-good config -> {resp.status_code}: {resp.text}\n")
        return ok
    except Exception as exc:
        print(f"  Could not restore known-good config: {exc}\n")
        return False


def main():
    server = server_url()
    print(f"Using Serval at {server}\n")

    show_dashboard_status(server)
    stop_any_active_measurement(server)

    try:
        original_text = get_destination(server)
        original = json.loads(original_text)
    except Exception as exc:
        print(f"  Could not read/parse current destination: {exc}\n")
        return

    raw_channels = original.get("Raw") or [{"Base": TEST_FOLDER, "FilePattern": "test%Hms_"}]

    working_values = []
    for value in CANDIDATE_VALUES:
        modified = dict(original)
        modified["Raw"] = [dict(raw_channels[0], SplitStrategy=value)]
        ok = put_destination(server, f'Trying SplitStrategy = "{value}"', modified)
        if ok:
            working_values.append(value)
        # Always restore the known-good config before the next attempt --
        # including after a hang, so one bad candidate can't cloud the next.
        restore(server, original_text)

    if working_values:
        print(f"Candidate(s) that were accepted (200 OK): {working_values}")
        print(
            "Accepted doesn't necessarily mean CORRECT -- Serval may accept\n"
            "any string here and only fail later when it actually tries to\n"
            "split files, or may silently fall back to a default. Run a short\n"
            "real measurement with each accepted candidate configured and\n"
            "check whether the raw folder actually gets one file per frame."
        )
    else:
        print(
            "None of the candidates were accepted cleanly. Either the real\n"
            "value isn't in this guess list (check GET /* for the current\n"
            "SplitStrategy value's exact casing/spelling for a hint at the\n"
            "naming convention), or SplitStrategy hangs on any value it\n"
            "doesn't already have applied -- in which case decompiling the\n"
            "Serval .jar for com.amscins.api.server's SplitStrategy enum, or\n"
            "asking AMSCins support directly, may be the only reliable path."
        )


if __name__ == "__main__":
    main()
