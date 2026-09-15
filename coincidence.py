"""Electron-ion coincidence analysis of a calibrated dataset. Headless (no Qt).

Input is an HDF5 file written by the apply calibration tab
(apply_calibration.py). Each electron is joined to the ions of its own
pulse, so m/q is a per-row coincidence quantity: an electron in a pulse
with two ions gives two rows, and one with no ion keeps a NaN m/q (gating
on m/q then drops it). Rows whose momenta are NaN, i.e. outside the
calibration's valid dt range, are dropped on load.

Gates are (low, high) ranges per variable. A plot of one variable shows the
data gated by every other enabled gate, so a gate never cuts into its own
plot; the main plot uses all of them.
"""
import dataclasses
import pathlib

import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class Variable:
    key: str
    label: str
    unit: str = ""

    @property
    def axis_label(self):
        return f"{self.label} ({self.unit})" if self.unit else self.label


VARIABLES = (
    Variable("px", "p_x", "a.u."),
    Variable("py", "p_y", "a.u."),
    Variable("pz", "p_z", "a.u."),
    Variable("pr", "p_r", "a.u."),
    Variable("t", "Cluster t", "ns"),
    Variable("mq", "m/q", "u/e"),
    Variable("parameter", "Parameter"),
)
BY_KEY = {variable.key: variable for variable in VARIABLES}

QUANTITY_COUNTS = "Counts"
QUANTITY_DENSITY = "Probability density"
QUANTITY_ASYMMETRY = "Forward–backward asymmetry"
QUANTITIES = (QUANTITY_COUNTS, QUANTITY_DENSITY, QUANTITY_ASYMMETRY)

ELECTRON_COLUMNS = ("parameter", "pulse", "t", "px", "py", "pz")
ION_COLUMNS = ("parameter", "pulse", "mq")


@dataclasses.dataclass
class Dataset:
    """Coincidence rows as plain arrays, one entry per available variable."""
    arrays: dict
    n_electrons: int
    parameter_label: str = ""
    path: str = ""

    @property
    def n_rows(self):
        return int(next(iter(self.arrays.values())).size) if self.arrays else 0

    @property
    def variables(self):
        return tuple(variable for variable in VARIABLES if variable.key in self.arrays)

    def label(self, key):
        variable = BY_KEY[key]
        if key == "parameter" and self.parameter_label:
            return f"{self.parameter_label}"
        return variable.axis_label


def _read_table(path, key, columns):
    try:
        frame = pd.read_hdf(path, key, columns=list(columns))
    except (TypeError, ValueError):
        # Older "fixed" format files don't support reading a column subset.
        frame = pd.read_hdf(path, key)
    return frame[[column for column in columns if column in frame.columns]]


def load_dataset(path):
    """Electron rows joined to their pulse's ions (see the module docstring)."""
    path = str(path)
    with pd.HDFStore(path, "r") as store:
        keys = {key.strip("/") for key in store.keys()}
        if "electrons" not in keys:
            raise ValueError(f"{pathlib.Path(path).name} has no electrons table.")
        parameter_label = str(getattr(store.get_storer("electrons").attrs, "parameter_label", "") or "")
    electrons = _read_table(path, "electrons", ELECTRON_COLUMNS)
    ions = _read_table(path, "ions", ION_COLUMNS) if "ions" in keys else pd.DataFrame()

    rows = electrons
    if "mq" in ions.columns and "pulse" in electrons.columns:
        on = ["pulse"] + (["parameter"] if "parameter" in electrons and "parameter" in ions else [])
        rows = electrons.merge(ions, on=on, how="left")

    px = rows["px"].to_numpy(dtype=np.float64)
    py = rows["py"].to_numpy(dtype=np.float64)
    pz = rows["pz"].to_numpy(dtype=np.float64)
    keep = np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
    arrays = {"px": px[keep], "py": py[keep], "pz": pz[keep], "pr": np.hypot(px[keep], py[keep])}
    for key in ("t", "mq", "parameter"):
        if key in rows.columns:
            arrays[key] = rows[key].to_numpy(dtype=np.float64)[keep]
    return Dataset(arrays=arrays, n_electrons=len(electrons), parameter_label=parameter_label, path=path)


def default_range(values, pad=0.02):
    """A plotting range that ignores the extreme tails."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(values, [0.1, 99.9]))
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return lo - 0.5, lo + 0.5
    span = hi - lo
    return lo - pad * span, hi + pad * span


def gate_mask(arrays, gates, skip=None):
    """Rows inside every gate except `skip`; None means every row."""
    mask = None
    for key, (lo, hi) in gates.items():
        if key == skip or key not in arrays:
            continue
        values = arrays[key]
        inside = (values >= min(lo, hi)) & (values <= max(lo, hi))
        mask = inside if mask is None else (mask & inside)
    return mask


def _finite(*arrays):
    mask = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _density(counts, cell):
    """Counts scaled so they integrate to 1 over the plotted range."""
    total = float(counts.sum()) * cell
    return counts / total if total > 0 else counts


def _asymmetry(forward, backward):
    total = (forward + backward).astype(np.float64)
    return np.divide(forward - backward, total, out=np.full(total.shape, np.nan), where=total > 0)


def histogram_1d(x, pz, bins, lo, hi, quantity=QUANTITY_DENSITY):
    """(bin centers, values) of `quantity` against x."""
    edges = np.linspace(lo, hi, int(bins) + 1)
    keep = _finite(x, pz)
    x, pz = x[keep], pz[keep]
    centers = 0.5 * (edges[:-1] + edges[1:])
    if quantity == QUANTITY_ASYMMETRY:
        forward, _ = np.histogram(x[pz > 0], bins=edges)
        backward, _ = np.histogram(x[pz < 0], bins=edges)
        return centers, _asymmetry(forward, backward)
    counts, _ = np.histogram(x, bins=edges)
    counts = counts.astype(np.float64)
    if quantity == QUANTITY_DENSITY:
        return centers, _density(counts, float(edges[1] - edges[0]))
    if quantity == QUANTITY_COUNTS:
        return centers, counts
    raise ValueError(f"Unknown quantity: {quantity}")


def histogram_2d(x, y, pz, bins, x_range, y_range, quantity=QUANTITY_DENSITY):
    """(values[x bin, y bin], x edges, y edges) of `quantity` in 2D."""
    x_edges = np.linspace(x_range[0], x_range[1], int(bins) + 1)
    y_edges = np.linspace(y_range[0], y_range[1], int(bins) + 1)
    keep = _finite(x, y, pz)
    x, y, pz = x[keep], y[keep], pz[keep]
    if quantity == QUANTITY_ASYMMETRY:
        forward, _, _ = np.histogram2d(x[pz > 0], y[pz > 0], bins=[x_edges, y_edges])
        backward, _, _ = np.histogram2d(x[pz < 0], y[pz < 0], bins=[x_edges, y_edges])
        return _asymmetry(forward, backward), x_edges, y_edges
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    if quantity == QUANTITY_DENSITY:
        cell = float((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))
        return _density(counts, cell), x_edges, y_edges
    if quantity == QUANTITY_COUNTS:
        return counts, x_edges, y_edges
    raise ValueError(f"Unknown quantity: {quantity}")
