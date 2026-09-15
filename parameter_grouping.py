"""Collects the cv4 files of a parameter scan into one grouped file, with
each file's group labelled by its parameter value. Headless (no Qt).

Grouped file layout (HDF5, .cv4 extension):
  root attrs  Layout = "grouped cv4", Parameter Label, Parameter Mode,
              Parameter Settings (JSON), Source Folder, Created
  /<value>    One group per scan file, named by its parameter value
              ("+0.5", "-3335.64", with " (2)" etc. for repeats). It holds
              that file's datasets (x, y, t, cluster_corr, t_etof, ...)
              and attributes, plus Parameter Value, Stage Position and
              Source File.

Groups are written one file at a time with HDF5's dataset copy, which
streams the stored data, so memory use doesn't grow with the size of the
scan (about 40 MB for an 800 MB file).

A scan file's stage position comes from its "Requested Position" attribute
(written by the parameter sweep), or else its pos_<index>_<position> name.
"""
import datetime
import json
import pathlib
import re
import dataclasses

import h5py
import numpy as np

LAYOUT = "grouped cv4"
GROUPED_SUFFIX = "_grouped.cv4"

MODE_DIRECT = "Direct"
MODE_QWP = "QWP angle → ellipticity"
MODE_DELAY = "Position → delay (fs)"
MODES = (MODE_DIRECT, MODE_QWP, MODE_DELAY)
LABELS = {MODE_QWP: "Ellipticity", MODE_DELAY: "Delay (fs)"}

C_MM_PER_FS = 2.99792458e-4

_POSITION_RE = re.compile(r"^pos_\d+_([+-]?\d+(?:\.\d+)?)$")


class CollectStopped(Exception):
    """Collecting was stopped before the grouped file was complete."""


@dataclasses.dataclass
class ScanFile:
    path: pathlib.Path
    position: float


def default_output_path(folder):
    folder = pathlib.Path(folder)
    return folder / f"{folder.name}{GROUPED_SUFFIX}"


def find_scan_files(folder):
    """(scan files sorted by position, names of skipped .cv4 files). Skips
    unfinished .partial.cv4 files, grouped files and files with no position."""
    files, skipped = [], []
    for path in sorted(pathlib.Path(folder).glob("*.cv4")):
        if path.name.endswith(".partial.cv4"):
            skipped.append(path.name)
            continue
        position = None
        try:
            with h5py.File(str(path), "r") as f:
                if f.attrs.get("Layout") == LAYOUT:
                    skipped.append(path.name)
                    continue
                if "Requested Position" in f.attrs:
                    position = float(f.attrs["Requested Position"])
        except OSError:
            skipped.append(path.name)
            continue
        if position is None:
            match = _POSITION_RE.match(path.stem)
            if match:
                position = float(match.group(1))
        if position is None:
            skipped.append(path.name)
        else:
            files.append(ScanFile(path, position))
    files.sort(key=lambda scan: scan.position)
    return files, skipped


def qwp_ellipticity(angle_deg, zero_deg=0.0):
    """Ellipticity tan(chi) of linearly polarized light after a quarter-wave
    plate at angle_deg, where zero_deg is the angle that leaves it linear:
    sin(2 chi) = sin(2 (angle - zero)). The sign gives the handedness; +-45
    degrees from zero is circular."""
    two_theta = np.radians(2.0 * (np.asarray(angle_deg, dtype=np.float64) - zero_deg))
    return np.tan(0.5 * np.arcsin(np.sin(two_theta)))


def position_to_delay_fs(position_mm, zero_mm=0.0, passes=2):
    """Optical delay of a delay stage: passes * (position - zero) / c."""
    return passes * (np.asarray(position_mm, dtype=np.float64) - zero_mm) / C_MM_PER_FS


def parameter_values(positions, mode, qwp_zero_deg=0.0, delay_zero_mm=0.0, delay_passes=2):
    positions = np.asarray(positions, dtype=np.float64)
    if mode == MODE_DIRECT:
        values = positions.copy()
    elif mode == MODE_QWP:
        values = qwp_ellipticity(positions, qwp_zero_deg)
    elif mode == MODE_DELAY:
        values = position_to_delay_fs(positions, delay_zero_mm, delay_passes)
    else:
        raise ValueError(f"Unknown parameter mode: {mode}")
    # Rounding drops float noise such as tan(45 deg) = 0.9999999999999999.
    return np.round(values, 9) + 0.0


def group_names(values):
    """A group name per value; repeated values get " (2)", " (3)", ..."""
    names, seen = [], {}
    for value in values:
        base = f"{float(value) + 0.0:+.6g}"
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base} ({seen[base]})")
    return names


def write_grouped(output, files, values, label, mode, settings, source_folder, progress=None, stop_event=None):
    """Writes the grouped file (under a .partial name, renamed when complete).
    progress(done, total, name) is called before and after each file.
    Raises CollectStopped, leaving no file, if stop_event is set."""
    output = pathlib.Path(output)
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    names = group_names(values)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(str(partial), "w") as out:
            out.attrs["Layout"] = LAYOUT
            out.attrs["Parameter Label"] = label
            out.attrs["Parameter Mode"] = mode
            out.attrs["Parameter Settings"] = json.dumps(settings)
            out.attrs["Source Folder"] = str(source_folder)
            out.attrs["Created"] = datetime.datetime.now().isoformat(timespec="seconds")
            for index, (scan, value, name) in enumerate(zip(files, values, names)):
                if progress is not None:
                    progress(index, len(files), scan.path.name)
                with h5py.File(str(scan.path), "r") as src:
                    group = out.create_group(name)
                    for key in src.keys():
                        if stop_event is not None and stop_event.is_set():
                            raise CollectStopped()
                        # HDF5's object copy streams the stored chunks, so it
                        # doesn't load the dataset into memory.
                        src.copy(src[key], group, name=key)
                    for key, attr in src.attrs.items():
                        group.attrs[key] = attr
                    group.attrs["Parameter Value"] = float(value)
                    group.attrs["Stage Position"] = float(scan.position)
                    group.attrs["Source File"] = scan.path.name
                if progress is not None:
                    progress(index + 1, len(files), scan.path.name)
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return output


def is_grouped(path):
    with h5py.File(str(path), "r") as f:
        return f.attrs.get("Layout") == LAYOUT


def read_group_info(path):
    """(label, mode, [(group name, value, source file)] sorted by value)."""
    with h5py.File(str(path), "r") as f:
        groups = [
            (name, float(f[name].attrs["Parameter Value"]), str(f[name].attrs.get("Source File", "")))
            for name in f.keys() if isinstance(f[name], h5py.Group)
        ]
        label = str(f.attrs.get("Parameter Label", ""))
        mode = str(f.attrs.get("Parameter Mode", ""))
    return label, mode, sorted(groups, key=lambda g: g[1])
