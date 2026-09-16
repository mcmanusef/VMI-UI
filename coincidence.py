"""Electron-ion coincidence analysis of a calibrated dataset. Headless (no Qt).

Input is an HDF5 file written by the apply calibration tab
(apply_calibration.py). Each electron is joined to the ions of its own
pulse, so m/q is a per-row coincidence quantity: an electron in a pulse
with two ions gives two rows, and one with no ion keeps a NaN m/q (gating
on m/q then drops it). Rows whose momenta are NaN, i.e. outside the
calibration's valid dt range, are dropped on load.

p_r is the full momentum magnitude, sqrt(px^2 + py^2 + pz^2).

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


# Label and unit for the columns the apply calibration tab writes; anything
# else keeps its column name.
KNOWN = {
    "px": ("p_x", "a.u."), "py": ("p_y", "a.u."), "pz": ("p_z", "a.u."), "pr": ("p_r", "a.u."),
    "energy": ("Energy", "eV"), "t": ("Cluster t", "ns"), "dt": ("Δt", "ns"),
    "t_etof": ("e-ToF t", "ns"), "t_tof": ("Ion ToF", "ns"), "t_pulse": ("Pulse t", "ns"),
    "mq": ("m/q", "u/e"), "x": ("x", "px"), "y": ("y", "px"), "pulse": ("Pulse", ""),
    "n_clusters": ("Clusters in pulse", ""), "n_etof": ("e-ToF hits in pulse", ""),
    "n_tof": ("Ions in pulse", ""), "parameter": ("Parameter", ""),
}
# Shown in the gate plots and offered as main-plot axes unless the user says
# otherwise.
DEFAULT_KEYS = ("px", "py", "pz", "pr", "t", "mq", "parameter")
PREFERRED_ORDER = (
    "px", "py", "pz", "pr", "energy", "t", "dt", "t_etof", "mq", "t_tof", "parameter",
    "x", "y", "pulse", "t_pulse", "n_clusters", "n_etof", "n_tof",
)


def variable_for(key):
    label, unit = KNOWN.get(key, (key, ""))
    return Variable(key, label, unit)


def order_keys(keys):
    """The known columns in a sensible order, then anything else by name."""
    keys = list(keys)
    return [key for key in PREFERRED_ORDER if key in keys] + sorted(
        key for key in keys if key not in PREFERRED_ORDER
    )

QUANTITY_COUNTS = "Counts"
QUANTITY_DENSITY = "Probability density"
QUANTITY_ASYMMETRY = "Forward–backward asymmetry"
QUANTITIES = (QUANTITY_COUNTS, QUANTITY_DENSITY, QUANTITY_ASYMMETRY)

MOMENTUM_COLUMNS = ("px", "py", "pz")


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
        """Every column in the file, plus p_r, in display order."""
        return tuple(variable_for(key) for key in order_keys(self.arrays))

    def label(self, key):
        if key == "parameter" and self.parameter_label:
            return self.parameter_label
        return variable_for(key).axis_label


def load_dataset(path):
    """Electron rows joined to their pulse's ions (see the module docstring).
    Every numeric column of both tables is kept, so any of them can be
    plotted or gated on."""
    path = str(path)
    with pd.HDFStore(path, "r") as store:
        keys = {key.strip("/") for key in store.keys()}
        if "electrons" not in keys:
            raise ValueError(f"{pathlib.Path(path).name} has no electrons table.")
        parameter_label = str(getattr(store.get_storer("electrons").attrs, "parameter_label", "") or "")
    electrons = pd.read_hdf(path, "electrons")
    missing = [column for column in MOMENTUM_COLUMNS if column not in electrons.columns]
    if missing:
        raise ValueError(f"The electrons table has no {', '.join(missing)} column(s).")
    ions = pd.read_hdf(path, "ions") if "ions" in keys else pd.DataFrame()

    rows = electrons
    if not ions.empty and "pulse" in electrons.columns and "pulse" in ions.columns:
        on = ["pulse"] + (["parameter"] if "parameter" in electrons and "parameter" in ions else [])
        # Columns the electrons already carry would only collide on the join.
        extra = [column for column in ions.columns if column in electrons.columns and column not in on]
        rows = electrons.merge(ions.drop(columns=extra), on=on, how="left")

    px, py, pz = (rows[column].to_numpy(dtype=np.float64) for column in MOMENTUM_COLUMNS)
    keep = np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
    px, py, pz = px[keep], py[keep], pz[keep]
    arrays = {"px": px, "py": py, "pz": pz, "pr": np.sqrt(px ** 2 + py ** 2 + pz ** 2)}
    for column in rows.columns:
        if column not in arrays and pd.api.types.is_numeric_dtype(rows[column]):
            arrays[column] = rows[column].to_numpy(dtype=np.float64)[keep]
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


def bin_index(values, lo, hi, bins):
    """(bin of each row, rows that land in the range) for uniform bins.
    Rows on the upper edge go in the last bin; NaN lands nowhere. Binning
    this way and counting with bincount avoids re-sorting the whole column
    on every redraw, which is what np.histogram does."""
    values = np.asarray(values, dtype=np.float64)
    bins = int(bins)
    scale = bins / (hi - lo) if hi > lo else 0.0
    with np.errstate(invalid="ignore"):
        scaled = (values - lo) * scale
        index = np.floor(np.where(np.isfinite(scaled), scaled, -1.0)).astype(np.int64)
        index[values == hi] = bins - 1
        inside = (index >= 0) & (index < bins)
    return index, inside


def _density(counts, cell):
    """Counts scaled so they integrate to 1 over the plotted range."""
    total = float(counts.sum()) * cell
    return counts / total if total > 0 else counts


def _asymmetry(forward, backward):
    total = (forward + backward).astype(np.float64)
    return np.divide(forward - backward, total, out=np.full(total.shape, np.nan), where=total > 0)


def counts_1d(index, inside, bins):
    return np.bincount(index[inside], minlength=int(bins)).astype(np.float64)


def histogram_1d(x, pz, bins, lo, hi, quantity=QUANTITY_DENSITY, binning=None):
    """(bin centers, values) of `quantity` against x. `binning` is a cached
    (index, inside) from bin_index() for the same x, range and bin count."""
    bins = int(bins)
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    index, inside = bin_index(x, lo, hi, bins) if binning is None else binning
    inside = inside & np.isfinite(pz)
    if quantity == QUANTITY_ASYMMETRY:
        forward = counts_1d(index, inside & (pz > 0), bins)
        backward = counts_1d(index, inside & (pz < 0), bins)
        return centers, _asymmetry(forward, backward)
    counts = counts_1d(index, inside, bins)
    if quantity == QUANTITY_DENSITY:
        return centers, _density(counts, float(edges[1] - edges[0]))
    if quantity == QUANTITY_COUNTS:
        return centers, counts
    raise ValueError(f"Unknown quantity: {quantity}")


def histogram_2d(x, y, pz, bins, x_range, y_range, quantity=QUANTITY_DENSITY, binning=None):
    """(values[x bin, y bin], x edges, y edges) of `quantity` in 2D. `bins`
    is one count for both axes, or (x bins, y bins). `binning` is a cached
    (x index, y index, inside) for the same columns, ranges and bin counts,
    in which case x and y are not read."""
    x_bins, y_bins = (int(bins), int(bins)) if np.isscalar(bins) else (int(bins[0]), int(bins[1]))
    x_edges = np.linspace(x_range[0], x_range[1], x_bins + 1)
    y_edges = np.linspace(y_range[0], y_range[1], y_bins + 1)
    if binning is None:
        x_index, x_inside = bin_index(x, x_range[0], x_range[1], x_bins)
        y_index, y_inside = bin_index(y, y_range[0], y_range[1], y_bins)
        inside = x_inside & y_inside
    else:
        x_index, y_index, inside = binning
    inside = inside & np.isfinite(pz)
    flat = x_index * y_bins + y_index

    def counts_2d(selected):
        counts = np.bincount(flat[selected], minlength=x_bins * y_bins)
        return counts.reshape(x_bins, y_bins).astype(np.float64)

    if quantity == QUANTITY_ASYMMETRY:
        forward = counts_2d(inside & (pz > 0))
        backward = counts_2d(inside & (pz < 0))
        return _asymmetry(forward, backward), x_edges, y_edges
    counts = counts_2d(inside)
    if quantity == QUANTITY_DENSITY:
        cell = float((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))
        return _density(counts, cell), x_edges, y_edges
    if quantity == QUANTITY_COUNTS:
        return counts, x_edges, y_edges
    raise ValueError(f"Unknown quantity: {quantity}")
