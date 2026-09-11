"""Reproduce (or rule out) the Sweep tab's destination-config hang, using the
*actual* serval_client.configure_continuous_destination() call -- not a
reimplementation -- so there's no risk of this test drifting from what the
app really does.

Times each of the three underlying requests separately (GET /detector/config,
PUT /detector/config, PUT /server/destination) so we can see exactly which
one is slow/hanging, instead of just knowing "configure_continuous_destination
timed out" the way the app's own error message does.

If this script succeeds quickly: the current code (SplitStrategy: FRAME,
LONG_RUN_TRIGGERS reduced to 50_000) is fine in isolation, and the problem is
something specific to the Sweep tab's own calling context (its background
stage-position poller thread, a stale/reused folder, thread contention on the
Serval connection, etc.) -- not serval_client.py itself.

If this script ALSO hangs: the nTriggers reduction didn't fix it (or wasn't
the actual cause), and we need to look elsewhere in the payload.

Run directly:
    python test_configure_continuous_destination.py [server_url] [dest_folder] [frame_time]
    (defaults: http://localhost:8080, C:\\DATA\\split_strategy_repro, 1.0)
"""
import json
import pathlib
import sys
import time

import requests

import serval_client

DEFAULT_SERVER = "http://localhost:8080"
# DEFAULT_FOLDER = r"C:\DATA\split_strategy_repro"


def server_url():
    return sys.argv[1].strip() if len(sys.argv) > 1 else DEFAULT_SERVER


def dest_folder():
    return sys.argv[2].strip() if len(sys.argv) > 2 else DEFAULT_FOLDER


def frame_time():
    return float(sys.argv[3]) if len(sys.argv) > 3 else 1.0


def timed(label, fn):
    print(f"{label} ...", flush=True)
    start = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - start
        print(f"  done in {elapsed:.2f}s\n")
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"  FAILED after {elapsed:.2f}s: {exc}\n")
        return None


def main():
    server = server_url()
    folder = pathlib.Path(dest_folder())
    folder.mkdir(parents=True, exist_ok=True)
    ft = frame_time()

    print(f"Server: {server}")
    print(f"Folder: {folder}  (Base URI: {folder.as_uri()})")
    print(f"frame_time: {ft}")
    print(f"LONG_RUN_TRIGGERS currently: {serval_client.LONG_RUN_TRIGGERS}\n")

    timed("GET /dashboard", lambda: requests.get(f"{server}/dashboard", timeout=15).text)

    # Break configure_continuous_destination's own steps out individually so
    # we know exactly which request is slow, not just the whole thing.
    config = timed(
        "GET /detector/config",
        lambda: json.loads(requests.get(f"{server}/detector/config", timeout=30).text),
    )
    if config is None:
        print("Can't continue without the current detector config.")
        return

    config["TriggerMode"] = "CONTINUOUS"
    config["ExposureTime"] = ft
    config["TriggerPeriod"] = ft
    config["nTriggers"] = serval_client.LONG_RUN_TRIGGERS
    timed(
        "PUT /detector/config (CONTINUOUS, nTriggers=LONG_RUN_TRIGGERS)",
        lambda: requests.put(f"{server}/detector/config", data=json.dumps(config), timeout=30).text,
    )

    destination = {
        "Raw": [
            {
                "Base": folder.as_uri(),
                "FilePattern": "repro%Hms_",
                "SplitStrategy": "FRAME",
            }
        ]
    }
    timed(
        "PUT /server/destination (SplitStrategy: FRAME)",
        lambda: requests.put(f"{server}/server/destination", data=json.dumps(destination), timeout=30).text,
    )

    # Now the real thing, exactly as sweep_interface.py calls it.
    timed(
        "serval_client.configure_continuous_destination() -- the actual app code",
        lambda: serval_client.configure_continuous_destination(server, ft, folder, "repro2%Hms_"),
    )

    print(
        "If everything above finished quickly: the current serval_client.py\n"
        "code is fine in isolation, so the Sweep tab's hang is coming from\n"
        "something in its own calling context, not this config/payload.\n"
        "If a specific line above was slow/failed: that pinpoints exactly\n"
        "which request is the actual problem."
    )


if __name__ == "__main__":
    main()
