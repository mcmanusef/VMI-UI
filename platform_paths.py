"""Per-platform defaults, so the app runs the same on Windows and Linux.

Windows keeps the lab's historical defaults (C:\\DATA, C:\\Temp, COM3);
elsewhere they live under the user's home / the system temp folder and a
typical USB-serial device name. Everything here is only a default: any
folder or port the user picks is used as-is.
"""
import datetime
import pathlib
import re
import sys
import tempfile

IS_WINDOWS = sys.platform == "win32"

DATA_ROOT = pathlib.Path(r"C:\DATA") if IS_WINDOWS else pathlib.Path.home() / "DATA"
SCRATCH_ROOT = pathlib.Path(r"C:\Temp") if IS_WINDOWS else pathlib.Path(tempfile.gettempdir())
DEFAULT_SERIAL_PORT = "COM3" if IS_WINDOWS else "/dev/ttyUSB0"

_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def dated_data_folder(leaf):
    """<data root>/<today as YYYYMMDD>/<leaf>, the default save folder."""
    return str(DATA_ROOT / datetime.date.today().strftime("%Y%m%d") / leaf)


def file_uri(path):
    """A local folder as the file:// URI Serval's destination wants. Made
    absolute first: Path.as_uri() refuses relative paths."""
    return pathlib.Path(path).expanduser().resolve().as_uri()


def is_foreign_path(value):
    """True for a Windows drive path (e.g. a saved "C:\\..." setting) when
    not running on Windows -- meaningless here, so callers fall back to
    their default instead."""
    return not IS_WINDOWS and isinstance(value, str) and bool(_DRIVE_PATH_RE.match(value.strip()))
