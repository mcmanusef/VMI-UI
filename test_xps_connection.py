"""Connectivity test for a Newport XPS-D motion controller.

Standalone script (no pytest, no vendor driver needed) that talks the XPS
"Command Interface" protocol directly over a raw TCP socket, per the
XPS Unified Programmer's Manual: a plain-text command like
    "FirmwareVersionGet(char *)"
is sent as-is, and the controller replies with
    "<errorCode>,<value1>,<value2>,...,EndOfAPI"
This is the same protocol Newport's own XPS_C8_drivers.py wraps (see
https://github.com/pyepics/newportxps) -- reimplemented here in a few dozen
lines so this test has no dependency on that vendor file being installed.

This only exercises READ-ONLY queries (login, firmware version, object
list, group status) -- it deliberately does NOT home or move anything, so
it's safe to run without knowing the physical stage setup.

Run directly:
    python test_xps_connection.py
"""
import socket

XPS_IP = "192.168.93.51"
XPS_PORT = 5001  # Newport XPS command interface's standard port.
XPS_TIMEOUT = 5.0  # seconds

# Newport ships XPS controllers with this account by default. Change these
# if the controller has been configured with different credentials.
XPS_USERNAME = "Administrator"
XPS_PASSWORD = "Administrator"

_END_OF_API = ",EndOfAPI"


class XPSError(Exception):
    """Raised when the controller returns a non-zero error code."""


def connect(ip=XPS_IP, port=XPS_PORT, timeout=XPS_TIMEOUT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((ip, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def send_and_receive(sock, command):
    """Send one XPS command string and return (error_code, value_string).

    Mirrors XPS_C8_drivers.py's __sendAndReceive: response accumulates
    across recv() calls until the ",EndOfAPI" suffix shows up, then
    "<code>,<rest>" is split on the *first* comma only (the rest can
    itself contain commas for multi-value responses).
    """
    sock.send(command.encode("ascii"))
    reply = sock.recv(1024).decode("ascii", errors="replace")
    while _END_OF_API not in reply:
        reply += sock.recv(1024).decode("ascii", errors="replace")
    body = reply[: -len(_END_OF_API)]
    error_str, _, value = body.partition(",")
    return int(error_str), value


def call(sock, command, *, check=True):
    """send_and_receive, optionally raising XPSError on a non-zero code."""
    error, value = send_and_receive(sock, command)
    if check and error != 0:
        # Best-effort human-readable message; don't let a failed
        # ErrorStringGet call mask the original error.
        try:
            _, message = send_and_receive(sock, f"ErrorStringGet({error}, char *)")
        except Exception:
            message = "<could not retrieve error string>"
        raise XPSError(f"{command} -> error {error}: {message}")
    return value


def login(sock, username=XPS_USERNAME, password=XPS_PASSWORD):
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


def main():
    print(f"Connecting to XPS-D at {XPS_IP}:{XPS_PORT} ...")
    sock = connect()
    try:
        print("  Connected. Logging in...")
        login(sock)
        print(f"  Login OK ({XPS_USERNAME}).")

        version = firmware_version_get(sock)
        print(f"  Firmware version: {version}")

        objects = objects_list_get(sock)
        print(f"  Configured objects: {objects or '<none>'}")

        # Best-effort: pull group names out of the object list (entries
        # look like "GroupName.PositionerName;...") and report each
        # group's status. Not fatal if this doesn't match on a given
        # config -- it's a bonus, not the point of the connectivity test.
        groups = sorted({entry.split(".", 1)[0] for entry in objects.split(";") if "." in entry})
        for group in groups:
            try:
                status_code = group_status_get(sock, group)
                status_text = group_status_string_get(sock, status_code)
                print(f"  Group {group!r}: status {status_code} ({status_text})")
            except XPSError as exc:
                print(f"  Group {group!r}: status query failed: {exc}")

        print("Connectivity test PASSED.")
    except XPSError as exc:
        print(f"Connectivity test FAILED: {exc}")
    except OSError as exc:
        print(f"Connectivity test FAILED: could not reach {XPS_IP}:{XPS_PORT} ({exc}).")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
