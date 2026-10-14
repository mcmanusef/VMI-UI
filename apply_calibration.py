"""Applies a momentum calibration, and optionally an m/q calibration, to a
cv4 dataset and saves the result as pandas DataFrames in an HDF5 file.
Headless (no Qt), like electron_calibration.py.

The input is a cv4 file, or a grouped file from the parameter grouping tab
(parameter_grouping.py). For a grouped file every group is processed and
every table gets a leading "parameter" column with that group's value; the
parameter label is saved in the attributes.

Tables (read back with pd.read_hdf(path, key)):
  electrons     One row per detector cluster / e-ToF hit pair from the same
                pulse. Raw columns: pulse, t_pulse, x, y, t (camera time),
                t_etof, and n_clusters / n_etof (hits in that pulse, to
                filter ambiguous pairings). Calibrated columns: dt (dithered
                t_etof - tc), px, py, pz (a.u.) and energy (eV). pz always
                comes from the e-ToF time. Pairs outside the calibration's
                valid dt range have NaN momenta and energy.
  raw/clusters  Every detector cluster, paired or not: x, y, t, pulse, t_pulse.
  raw/etof      Every e-ToF hit: t_etof, pulse, t_pulse.
  ions          Every i-ToF hit: t_tof, pulse, t_pulse, n_tof, plus mq (u/e)
                when an m/q calibration is given. Left out with electrons_only.
Pulse indices are per cv4 file, so in grouped output (parameter, pulse)
identifies a pulse. Pulse 0 means no matching pulse, which includes hits
more than a pulse period after the last trigger.

The output is still one file, but it is written a part (cv4 file or group)
at a time: each part is calibrated and appended before the next is read, so
memory use is about one group's worth. Tables use pandas' "table" format,
with parameter as a data column, so one value can be read on its own:
pd.read_hdf(path, "electrons", where="parameter == 0.5").

Every table's attributes hold the source path, both calibrations as JSON,
the cv4 run metadata as JSON, and for grouped input the parameter label,
mode and groups.
"""
import dataclasses
import datetime
import json
import pathlib

import h5py
import numpy as np
import pandas as pd

import electron_calibration as ec
import parameter_grouping as pgr

OUTPUT_SUFFIX = "_calibrated.h5"
ELECTRON_FIELDS = ("x", "y", "t", "cluster_corr", "t_etof", "etof_corr", "t_pulse")
ION_FIELDS = ("t_tof", "tof_corr")


def default_output_path(cv4_path):
    path = pathlib.Path(cv4_path)
    return path.with_name(path.stem + OUTPUT_SUFFIX)


def _plain(value):
    """A cv4 attribute as something json.dumps accepts."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


def _read_node(node, electrons_only, name):
    """(arrays by field name, attributes) from a cv4 file or group."""
    fields = ELECTRON_FIELDS if electrons_only else ELECTRON_FIELDS + ION_FIELDS
    missing = [field for field in fields if field not in node]
    if missing:
        raise ValueError(f"{name} has no {', '.join(missing)} dataset(s).")
    raw = {field: np.asarray(node[field]) for field in fields}
    return raw, {key: _plain(value) for key, value in node.attrs.items()}


def read_cv4(path, electrons_only=False):
    """(arrays by field name, run metadata) from a plain cv4 file."""
    with h5py.File(str(path), "r") as f:
        return _read_node(f, electrons_only, pathlib.Path(path).name)


def _pulse_columns(corr, t_pulse):
    """pulse (int32) and t_pulse for events that reference pulses by index."""
    pulse = np.asarray(corr, dtype=np.int64)
    return pulse.astype(np.int32), t_pulse[pulse]


def _matched_pulses(corr, t, t_pulse):
    """corr with hits that belong to no pulse moved to pulse 0, cv4's "no
    matching pulse" index. Hits after the last trigger are all given the
    last pulse, however late they are; those more than a pulse period
    (the median trigger spacing) after it are moved."""
    corr = np.array(corr, dtype=np.int64)
    triggers = t_pulse[1:]
    if triggers.size >= 2:
        period = float(np.median(np.diff(triggers)))
        corr[(corr == t_pulse.size - 1) & (np.asarray(t, dtype=np.float64) >= period)] = 0
    return corr


def build_tables(raw, momentum, mq=None, electrons_only=False, rng=None, parameter=None):
    """The DataFrames described in the module docstring. With `parameter`,
    each table starts with a "parameter" column holding that value.

    With momentum None the electrons are left uncalibrated: one row per
    cluster/e-ToF pair as usual, plus a row with a NaN t_etof for each
    cluster whose pulse has no e-ToF hit, and no dt, px, py, pz or energy."""
    t_pulse = np.asarray(raw["t_pulse"], dtype=np.float64)
    cluster_corr = np.asarray(raw["cluster_corr"], dtype=np.int64)
    etof_corr = np.asarray(raw["etof_corr"], dtype=np.int64)
    for name, corr in (("cluster_corr", cluster_corr), ("etof_corr", etof_corr)):
        if corr.size and (corr.min() < 0 or corr.max() >= t_pulse.size):
            raise ValueError(f"{name} points outside t_pulse.")
    cluster_corr = _matched_pulses(cluster_corr, raw["t"], t_pulse)
    etof_corr = _matched_pulses(etof_corr, raw["t_etof"], t_pulse)

    # Pulse 0 is cv4's "no matching pulse" sentinel, not a real pulse. It's
    # left out before pairing: every unmatched cluster times every unmatched
    # e-ToF hit can be billions of pairs.
    c_valid = np.flatnonzero(cluster_corr != 0)
    e_valid = np.flatnonzero(etof_corr != 0)
    c_rows, e_rows = ec.pair_by_pulse(cluster_corr[c_valid], etof_corr[e_valid], single_hit_only=False)
    c_rows, e_rows = c_valid[c_rows], e_valid[e_rows]
    clusters_per_pulse = np.bincount(cluster_corr, minlength=t_pulse.size)
    etof_per_pulse = np.bincount(etof_corr, minlength=t_pulse.size)
    if momentum is None:
        # Keep the clusters with no e-ToF hit to pair with, in cluster order.
        lone = c_valid[etof_per_pulse[cluster_corr[c_valid]] == 0]
        c_rows = np.concatenate([c_rows, lone])
        e_rows = np.concatenate([e_rows, np.full(lone.size, -1, dtype=e_rows.dtype)])
        order = np.argsort(c_rows, kind="stable")
        c_rows, e_rows = c_rows[order], e_rows[order]

    x = np.asarray(raw["x"], dtype=np.float64)[c_rows]
    y = np.asarray(raw["y"], dtype=np.float64)[c_rows]
    t_etof = np.asarray(raw["t_etof"], dtype=np.float64)[e_rows]
    if momentum is None:
        t_etof[e_rows < 0] = np.nan
    pulse = cluster_corr[c_rows]
    electrons = {
        "pulse": pulse.astype(np.int32),
        "t_pulse": t_pulse[pulse],
        "x": x,
        "y": y,
        "t": np.asarray(raw["t"], dtype=np.float64)[c_rows],
        "t_etof": t_etof,
        "n_clusters": clusters_per_pulse[pulse].astype(np.int32),
        "n_etof": etof_per_pulse[pulse].astype(np.int32),
    }
    if momentum is not None:
        t_used = t_etof
        if momentum.dither_ns > 0 and t_etof.size:
            rng = rng if rng is not None else np.random.default_rng()
            t_used = t_etof + rng.uniform(0.0, momentum.dither_ns, size=t_etof.size)
        px, py, pz = momentum.calibrate(x, y, t_used)
        electrons.update({
            "dt": t_used - momentum.center[2],
            "px": px,
            "py": py,
            "pz": pz,
            "energy": ec.energy_ev(px, py, pz),
        })
    tables = {"electrons": pd.DataFrame(electrons)}

    pulse, pulse_time = _pulse_columns(cluster_corr, t_pulse)
    tables["raw/clusters"] = pd.DataFrame({
        "x": raw["x"], "y": raw["y"], "t": raw["t"], "pulse": pulse, "t_pulse": pulse_time,
    })
    pulse, pulse_time = _pulse_columns(etof_corr, t_pulse)
    tables["raw/etof"] = pd.DataFrame({"t_etof": raw["t_etof"], "pulse": pulse, "t_pulse": pulse_time})

    if not electrons_only:
        tof_corr = np.asarray(raw["tof_corr"], dtype=np.int64)
        if tof_corr.size and (tof_corr.min() < 0 or tof_corr.max() >= t_pulse.size):
            raise ValueError("tof_corr points outside t_pulse.")
        tof_corr = _matched_pulses(tof_corr, raw["t_tof"], t_pulse)
        pulse, pulse_time = _pulse_columns(tof_corr, t_pulse)
        ions = {
            "t_tof": np.asarray(raw["t_tof"], dtype=np.float64),
            "pulse": pulse,
            "t_pulse": pulse_time,
            "n_tof": np.bincount(tof_corr, minlength=t_pulse.size)[tof_corr].astype(np.int32),
        }
        if mq is not None:
            ions["mq"] = mq.mass_to_charge(ions["t_tof"])
        tables["ions"] = pd.DataFrame(ions)

    if parameter is not None:
        for frame in tables.values():
            frame.insert(0, "parameter", np.full(len(frame), float(parameter)))
    return tables


class ApplyStopped(Exception):
    """Applying was stopped before the output file was complete."""


def has_ion_fields(node):
    return all(field in node for field in ION_FIELDS)


def calibrate_in_memory(cv4_path, momentum=None, mq=None, rng=None, progress=None, stop_event=None):
    """The electrons and ions tables apply_to_file would write, kept in
    memory instead: (electrons, ions, parameter label). Either calibration
    can be None, which leaves that part uncalibrated (see build_tables).
    ions is an empty DataFrame when the file has no i-ToF data. The raw/
    tables are left out. progress and stop_event work as in apply_to_file."""
    source_name = pathlib.Path(cv4_path).name
    if pgr.is_grouped(cv4_path):
        label, _, groups = pgr.read_group_info(cv4_path)
        parts = [(name, value) for name, value, _ in groups]
    else:
        label, parts = "", [(None, None)]

    electrons, ions = [], []
    with h5py.File(str(cv4_path), "r") as source:
        for index, (name, value) in enumerate(parts):
            if stop_event is not None and stop_event.is_set():
                raise ApplyStopped()
            if progress is not None:
                progress(index, len(parts), source_name if name is None else f"group {name}")
            node = source if name is None else source[name]
            electrons_only = not has_ion_fields(node)
            raw, _ = _read_node(node, electrons_only, source_name if name is None else f"Group {name}")
            tables = build_tables(raw, momentum, None if electrons_only else mq, electrons_only, rng, value)
            del raw
            electrons.append(tables["electrons"])
            if "ions" in tables:
                ions.append(tables["ions"])
            del tables
    if progress is not None:
        progress(len(parts), len(parts), "")
    return (
        pd.concat(electrons, ignore_index=True),
        pd.concat(ions, ignore_index=True) if ions else pd.DataFrame(),
        label,
    )


@dataclasses.dataclass
class ApplyResult:
    output_path: str
    n_electrons: int
    n_uncalibrated: int
    n_ions: int
    electrons_only: bool
    px: np.ndarray
    py: np.ndarray
    pz: np.ndarray
    energy: np.ndarray
    mq: np.ndarray = None
    parameter_label: str = ""
    n_groups: int = 0


def apply_to_file(cv4_path, output_path, momentum, mq=None, electrons_only=False, rng=None,
                  progress=None, stop_event=None):
    """Calibrates a cv4 or grouped file into one output file, a part at a
    time (see the module docstring). progress(done, total, name) is called
    before each part and at the end. Raises ApplyStopped, leaving no file,
    if stop_event is set between parts."""
    mq = None if electrons_only else mq
    source_name = pathlib.Path(cv4_path).name
    attrs = {
        "source_cv4": str(cv4_path),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "pz_time_source": ec.TIME_SOURCE_ETOF,
        "momentum_calibration": json.dumps(momentum.to_dict()),
        "mq_calibration": json.dumps(mq.to_dict()) if mq is not None else "",
        "parameter_label": "",
        "parameter_mode": "",
        "groups": "",
    }
    if pgr.is_grouped(cv4_path):
        label, mode, groups = pgr.read_group_info(cv4_path)
        parts = [(name, value) for name, value, _ in groups]
        attrs.update({
            "parameter_label": label,
            "parameter_mode": mode,
            "groups": json.dumps([{"group": n, "parameter": v, "source_file": s} for n, v, s in groups]),
        })
    else:
        groups = []
        parts = [(None, None)]

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    partial.unlink(missing_ok=True)
    # Only what the previews plot is kept in memory across parts.
    kept = {name: [] for name in ("px", "py", "pz", "energy", "mq")}
    n_electrons = n_uncalibrated = n_ions = 0
    metadata = {}
    try:
        with h5py.File(str(cv4_path), "r") as source, \
                pd.HDFStore(str(partial), mode="w", complevel=5, complib="blosc:lz4") as store:
            for index, (name, value) in enumerate(parts):
                if stop_event is not None and stop_event.is_set():
                    raise ApplyStopped()
                if progress is not None:
                    progress(index, len(parts), source_name if name is None else f"group {name}")
                if name is None:
                    raw, metadata = _read_node(source, electrons_only, source_name)
                else:
                    raw, metadata[name] = _read_node(source[name], electrons_only, f"Group {name}")
                tables = build_tables(raw, momentum, mq, electrons_only, rng, parameter=value)
                del raw
                for key, frame in tables.items():
                    store.append(
                        key, frame, format="table", index=False,
                        data_columns=["parameter"] if value is not None else None,
                    )
                electrons = tables["electrons"]
                n_electrons += len(electrons)
                n_uncalibrated += int(electrons["pz"].isna().sum())
                for column in ("px", "py", "pz", "energy"):
                    kept[column].append(electrons[column].to_numpy())
                ions = tables.get("ions")
                if ions is not None:
                    n_ions += len(ions)
                    if "mq" in ions:
                        kept["mq"].append(ions["mq"].to_numpy())
                del tables, electrons, ions

            attrs["run_metadata"] = json.dumps(metadata)
            for key in store.keys():
                storer_attrs = store.get_storer(key).attrs
                for attr_name, attr_value in attrs.items():
                    setattr(storer_attrs, attr_name, attr_value)
        partial.replace(output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    if progress is not None:
        progress(len(parts), len(parts), "")

    def joined(column):
        return np.concatenate(kept[column]) if kept[column] else np.array([])

    return ApplyResult(
        output_path=str(output_path),
        n_electrons=n_electrons,
        n_uncalibrated=n_uncalibrated,
        n_ions=n_ions,
        electrons_only=electrons_only,
        px=joined("px"),
        py=joined("py"),
        pz=joined("pz"),
        energy=joined("energy"),
        mq=joined("mq") if kept["mq"] else None,
        parameter_label=attrs["parameter_label"],
        n_groups=len(groups),
    )
