"""e-ToF time-walk (and optional position) correction. Headless (no Qt).

The e-ToF time comes from a leading-edge discriminator on the MCP pickoff,
so a small MCP pulse crosses threshold later than a large one: time walk.
The same electron also makes a Timepix cluster whose summed ToT ("charge",
Q) tracks the MCP pulse amplitude, so each electron's walk delay is
predicted from its own cluster and subtracted:

    t_corr = t_etof - ([f(ln Q) - f(ref_lnq)] + pos(x, y))

f is the median e-ToF time in quantile bins of ln Q, interpolated (flat
beyond the fitted range); subtracting f(ref_lnq) keeps the corrected times
on the scale of a ref_lnq cluster, so a correction trained on one run
transfers to another as a shape. pos (optional) is a linear x / y gradient
plus a per-cell map of what is left, both fitted on the walk-corrected
residuals; it removes a detector-wide timing gradient but can also remove
physics wherever position and flight time are linked by the momenta (low
field), so it is off unless asked for.

Fitting uses single-electron shots only (select_shots): exactly one e-ToF
hit in the pulse, inside the e-ToF window, and exactly one cluster, with at
least min_hits pixels, a positive ToT sum and its time inside the cluster
window -- so the cluster / e-ToF pairing is one-to-one. Pairing is by pulse
alone. The spec's e-ToF pulse-width cut can't be made from a cv4 file; set
it as the e-ToF gate on the Hardware > TDC mapping page when acquiring.

Data come from cv4 files written with cluster ToT (cv4_writer's
cluster_tot_sum, in ToT codes); older files have to be reconverted from
raw. The correction is saved as JSON (EtofWalkCorrection.save / load).
"""
import dataclasses
import datetime
import json
import pathlib
import threading

import h5py
import numpy as np

import chip_layout
import parameter_grouping as pgr

FILE_FORMAT = "etof_walk_v1"
KIND = "etof_walk"
# Same key calibration_interface.latest_path_key(KIND) gives: where the
# CalibrationHub remembers the last saved or loaded correction file.
LATEST_PATH_KEY = f"calibration.latest_{KIND}_path"

DEFAULT_REF_LNQ = 7.2
DEFAULT_N_BINS = 60
DEFAULT_MIN_HITS = 8
DEFAULT_CELL_PX = 16
DEFAULT_MIN_CELL_COUNT = 20
MAD_TO_SIGMA = 1.4826

NO_TOT_MESSAGE = (
    "{name} has no cluster ToT (it was written before cv4 files recorded it). Reconvert it from "
    "raw (Analysis > conversion) to use the e-ToF walk correction."
)


def mad_sigma(values):
    """Core width: 1.4826 x the median absolute deviation (NaN if empty)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


# ---- data ---------------------------------------------------------------------

@dataclasses.dataclass
class Shots:
    """Every cluster and e-ToF hit of a cv4 file (all groups of a grouped
    one). pulse ids are unique across groups; 0-based ids < 0 never occur,
    and cv4's "no matching pulse" index 0 maps to -1."""
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    tot_sum: np.ndarray
    n_hits: np.ndarray
    cluster_pulse: np.ndarray
    t_etof: np.ndarray
    etof_pulse: np.ndarray
    path: str = ""


def has_cluster_tot(node):
    return "cluster_tot_sum" in node


def load_shots(path):
    """Shots from a plain or grouped cv4 file. Raises ValueError if it has
    no cluster ToT."""
    path = str(path)
    name = pathlib.Path(path).name
    if pgr.is_grouped(path):
        _, _, groups = pgr.read_group_info(path)
        parts = [group for group, _, _ in groups]
    else:
        parts = [None]
    pieces = {key: [] for key in ("x", "y", "t", "tot_sum", "n_hits", "cluster_pulse", "t_etof", "etof_pulse")}
    base = 0
    with h5py.File(path, "r") as f:
        for part in parts:
            node = f if part is None else f[part]
            if not has_cluster_tot(node):
                raise ValueError(NO_TOT_MESSAGE.format(name=name if part is None else f"{name} (group {part})"))
            n_pulses = node["t_pulse"].shape[0]

            def pulse_ids(corr):
                corr = np.asarray(corr, dtype=np.int64)
                return np.where(corr > 0, corr + base, -1)

            pieces["x"].append(np.asarray(node["x"], dtype=np.float64))
            pieces["y"].append(np.asarray(node["y"], dtype=np.float64))
            pieces["t"].append(np.asarray(node["t"], dtype=np.float64))
            pieces["tot_sum"].append(np.asarray(node["cluster_tot_sum"], dtype=np.float64))
            pieces["n_hits"].append(np.asarray(node["cluster_n"], dtype=np.int64))
            pieces["cluster_pulse"].append(pulse_ids(node["cluster_corr"]))
            pieces["t_etof"].append(np.asarray(node["t_etof"], dtype=np.float64))
            pieces["etof_pulse"].append(pulse_ids(node["etof_corr"]))
            base += n_pulses
    return Shots(**{key: np.concatenate(value) for key, value in pieces.items()}, path=path)


@dataclasses.dataclass
class Selection:
    """The shot selection's settings (ns windows, as on the tab)."""
    etof_min: float
    etof_max: float
    t_min: float
    t_max: float
    min_hits: int = DEFAULT_MIN_HITS

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SelectedShots:
    """One row per single-electron shot: its cluster and its e-ToF time."""
    x: np.ndarray
    y: np.ndarray
    lnq: np.ndarray
    tot_sum: np.ndarray
    t_etof: np.ndarray

    def __len__(self):
        return int(self.t_etof.size)


def select_shots(shots, selection):
    """The single-electron shots (see the module docstring)."""
    s = selection
    n_ids = int(max(shots.cluster_pulse.max(initial=-1), shots.etof_pulse.max(initial=-1))) + 1
    e_ok = shots.etof_pulse >= 0
    c_ok = shots.cluster_pulse >= 0
    etof_count = np.bincount(shots.etof_pulse[e_ok], minlength=n_ids)
    cluster_count = np.bincount(shots.cluster_pulse[c_ok], minlength=n_ids)

    e_rows = np.flatnonzero(e_ok)
    e_rows = e_rows[etof_count[shots.etof_pulse[e_rows]] == 1]
    te = shots.t_etof[e_rows]
    e_rows = e_rows[(te >= s.etof_min) & (te <= s.etof_max)]

    c_rows = np.flatnonzero(c_ok)
    c_rows = c_rows[cluster_count[shots.cluster_pulse[c_rows]] == 1]
    tot = shots.tot_sum[c_rows]
    tc = shots.t[c_rows]
    keep = (
        (shots.n_hits[c_rows] >= s.min_hits) & np.isfinite(tot) & (tot > 0)
        & (tc >= s.t_min) & (tc <= s.t_max)
    )
    c_rows = c_rows[keep]

    # Both lists hold at most one row per pulse: match them up by pulse.
    etof_row_of = np.full(n_ids, -1, dtype=np.int64)
    etof_row_of[shots.etof_pulse[e_rows]] = e_rows
    paired = etof_row_of[shots.cluster_pulse[c_rows]]
    c_rows, e_rows = c_rows[paired >= 0], paired[paired >= 0]
    tot = shots.tot_sum[c_rows]
    return SelectedShots(
        x=shots.x[c_rows], y=shots.y[c_rows], lnq=np.log(tot), tot_sum=tot, t_etof=shots.t_etof[e_rows],
    )


# ---- fitting ------------------------------------------------------------------

def fit_walk(lnq, t, n_bins=DEFAULT_N_BINS):
    """(ln Q nodes, t nodes): median ln Q and median t in n_bins quantile
    bins of ln Q (empty bins dropped)."""
    lnq = np.asarray(lnq, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if lnq.size < 2:
        raise ValueError("Too few shots to fit the walk curve.")
    n_bins = max(2, min(int(n_bins), lnq.size // 5 or 1))
    edges = np.unique(np.quantile(lnq, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return np.array([float(np.median(lnq))]), np.array([float(np.median(t))])
    index = np.clip(np.searchsorted(edges, lnq, side="right") - 1, 0, edges.size - 2)
    nodes_q, nodes_t = [], []
    for b in range(edges.size - 1):
        in_bin = index == b
        if in_bin.any():
            nodes_q.append(float(np.median(lnq[in_bin])))
            nodes_t.append(float(np.median(t[in_bin])))
    order = np.argsort(nodes_q)
    return np.asarray(nodes_q)[order], np.asarray(nodes_t)[order]


def fit_position(x, y, resid, cell_px=DEFAULT_CELL_PX, min_count=DEFAULT_MIN_CELL_COUNT, size=None):
    """The position term for walk-corrected residuals: a linear gradient
    about the median centroid (least squares after a 4-sigma MAD clip), then
    the median of what is left in each cell_px x cell_px cell (cells with
    fewer than min_count shots get 0). Returns its dict (see PositionTerm)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.asarray(resid, dtype=np.float64)
    if size is None:
        size = chip_layout.get_active_layout().size
    x0, y0 = float(np.median(x)), float(np.median(y))
    sigma = mad_sigma(r)
    keep = np.isfinite(r)
    if np.isfinite(sigma) and sigma > 0:
        keep &= np.abs(r - np.median(r[keep])) < 4.0 * sigma
    design = np.column_stack([np.ones(keep.sum()), x[keep] - x0, y[keep] - y0])
    (a, bx, by), *_ = np.linalg.lstsq(design, r[keep], rcond=None)
    left = r - (a + bx * (x - x0) + by * (y - y0))

    cell_px = max(1, int(cell_px))
    n_cells = int(np.ceil(size / cell_px))
    ix = np.clip((x // cell_px).astype(np.int64), 0, n_cells - 1)
    iy = np.clip((y // cell_px).astype(np.int64), 0, n_cells - 1)
    cell = ix * n_cells + iy
    cell_map = np.zeros(n_cells * n_cells)
    order = np.argsort(cell[keep], kind="stable")
    cells_sorted = cell[keep][order]
    left_sorted = left[keep][order]
    starts = np.flatnonzero(np.r_[True, np.diff(cells_sorted) != 0])
    ends = np.r_[starts[1:], cells_sorted.size]
    for start, end in zip(starts, ends):
        if end - start >= min_count:
            cell_map[cells_sorted[start]] = float(np.median(left_sorted[start:end]))
    return {
        "x0": x0, "y0": y0, "bx": float(bx), "by": float(by), "cell_px": cell_px, "n_cells": n_cells,
        "map": cell_map.reshape(n_cells, n_cells).tolist(),
    }


def _position_delay(position, x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = position["bx"] * (x - position["x0"]) + position["by"] * (y - position["y0"])
    cell_map = np.asarray(position["map"], dtype=np.float64)
    n = int(position["n_cells"])
    cell_px = float(position["cell_px"])
    with np.errstate(invalid="ignore"):
        ix = np.floor(x / cell_px)
        iy = np.floor(y / cell_px)
    inside = np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    out = np.where(np.isfinite(out), out, 0.0)
    out[inside] += cell_map[ix[inside].astype(np.int64), iy[inside].astype(np.int64)]
    return out


# ---- the correction -------------------------------------------------------------

@dataclasses.dataclass
class EtofWalkCorrection:
    lnq_nodes: list
    t_nodes: list
    ref_lnq: float = DEFAULT_REF_LNQ
    position: dict = None
    n_shots: int = 0
    held_out_sigma_ns: dict = dataclasses.field(default_factory=dict)
    selection: dict = dataclasses.field(default_factory=dict)
    source: str = ""
    created: str = ""

    def walk(self, lnq):
        """f(ln Q), the median e-ToF time of clusters of that charge."""
        return np.interp(lnq, self.lnq_nodes, self.t_nodes)

    def delay(self, tot_sum, x=None, y=None):
        """The delay to subtract from each e-ToF time (ns), from its
        cluster's ToT sum (codes) and, with a position term, its detector
        x / y. 0 where the ToT sum is missing or not positive."""
        tot = np.asarray(tot_sum, dtype=np.float64)
        out = np.zeros(tot.shape, dtype=np.float64)
        ok = np.isfinite(tot) & (tot > 0)
        if not ok.any():
            return out
        out[ok] = self.walk(np.log(tot[ok])) - float(self.walk(self.ref_lnq))
        if self.position is not None and x is not None and y is not None:
            out[ok] += _position_delay(self.position, np.asarray(x)[ok], np.asarray(y)[ok])
        return out

    def apply(self, t_etof, tot_sum, x=None, y=None):
        """Walk-corrected e-ToF times."""
        return np.asarray(t_etof, dtype=np.float64) - self.delay(tot_sum, x, y)

    def describe(self):
        position = "with position term" if self.position is not None else "walk only"
        sigma = self.held_out_sigma_ns or {}
        widths = " → ".join(
            f"{sigma[key] * 1000:.0f} ps" for key in ("raw", "walk", "position") if key in sigma
        )
        text = f"{len(self.lnq_nodes)} nodes, ref ln Q {self.ref_lnq:g}, {position}"
        return f"{text}; held-out σ {widths}" if widths else text

    def to_dict(self):
        data = dataclasses.asdict(self)
        data["format"] = FILE_FORMAT
        return data

    @classmethod
    def from_dict(cls, data):
        if data.get("format") != FILE_FORMAT:
            raise ValueError("Not an e-ToF walk correction file.")
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in fields})

    def save(self, path):
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


_cache = {}
_cache_lock = threading.Lock()


def load_cached(path):
    """EtofWalkCorrection.load, remembered per (path, modification time) so
    per-frame processing doesn't re-read the file."""
    path = str(path)
    key = (path, pathlib.Path(path).stat().st_mtime)
    with _cache_lock:
        if key not in _cache:
            _cache.clear()
            _cache[key] = EtofWalkCorrection.load(path)
        return _cache[key]


@dataclasses.dataclass
class FitReport:
    """What the tab plots: every selected shot, with its corrected time
    from the final correction, and which were held out of the check fit."""
    shots: SelectedShots
    corrected: np.ndarray
    walk_only: np.ndarray
    held_out: np.ndarray
    sigma_ns: dict


def _build(shots, rows, n_bins, ref_lnq, use_position, cell_px, min_count):
    lnq_nodes, t_nodes = fit_walk(shots.lnq[rows], shots.t_etof[rows], n_bins)
    correction = EtofWalkCorrection(lnq_nodes=lnq_nodes.tolist(), t_nodes=t_nodes.tolist(), ref_lnq=float(ref_lnq))
    if use_position:
        resid = shots.t_etof[rows] - correction.walk(shots.lnq[rows])
        correction.position = fit_position(shots.x[rows], shots.y[rows], resid, cell_px, min_count)
    return correction


def fit_and_validate(shots, n_bins=DEFAULT_N_BINS, ref_lnq=DEFAULT_REF_LNQ, use_position=False,
                     cell_px=DEFAULT_CELL_PX, min_count=DEFAULT_MIN_CELL_COUNT, seed=0,
                     selection=None, source=""):
    """(EtofWalkCorrection, FitReport). A check fit on a random half of the
    shots is scored on the other half (MAD sigma of the e-ToF time raw,
    walk-corrected, and walk + position corrected); the returned correction
    is then fitted on all of them."""
    n = len(shots)
    if n < 20:
        raise ValueError(f"Only {n} single-electron shots were selected; widen the windows or use more data.")
    rng = np.random.default_rng(seed)
    held_out = rng.random(n) < 0.5
    train = np.flatnonzero(~held_out)
    test = np.flatnonzero(held_out)

    check = _build(shots, train, n_bins, ref_lnq, False, cell_px, min_count)
    t_test = shots.t_etof[test]
    walk_test = t_test - check.walk(shots.lnq[test])
    sigma = {"raw": mad_sigma(t_test), "walk": mad_sigma(walk_test)}
    if use_position:
        check.position = fit_position(
            shots.x[train], shots.y[train],
            shots.t_etof[train] - check.walk(shots.lnq[train]), cell_px, min_count,
        )
        sigma["position"] = mad_sigma(walk_test - _position_delay(check.position, shots.x[test], shots.y[test]))

    correction = _build(shots, np.arange(n), n_bins, ref_lnq, use_position, cell_px, min_count)
    correction.n_shots = n
    correction.held_out_sigma_ns = sigma
    correction.selection = selection.to_dict() if selection is not None else {}
    correction.source = str(source)
    correction.created = datetime.datetime.now().isoformat(timespec="seconds")
    walk_only = dataclasses.replace(correction, position=None)
    report = FitReport(
        shots=shots,
        corrected=correction.apply(shots.t_etof, shots.tot_sum, shots.x, shots.y),
        walk_only=walk_only.apply(shots.t_etof, shots.tot_sum),
        held_out=held_out,
        sigma_ns=sigma,
    )
    return correction, report


def residual_map(x, y, resid, cell_px=DEFAULT_CELL_PX, min_count=5, size=None):
    """Median residual per cell, NaN where a cell has fewer than min_count
    shots: (map[x cell, y cell], cell_px, n_cells), for plotting."""
    if size is None:
        size = chip_layout.get_active_layout().size
    n_cells = int(np.ceil(size / cell_px))
    ix = np.clip((np.asarray(x) // cell_px).astype(np.int64), 0, n_cells - 1)
    iy = np.clip((np.asarray(y) // cell_px).astype(np.int64), 0, n_cells - 1)
    cell = ix * n_cells + iy
    out = np.full(n_cells * n_cells, np.nan)
    order = np.argsort(cell, kind="stable")
    cells_sorted = cell[order]
    values = np.asarray(resid, dtype=np.float64)[order]
    starts = np.flatnonzero(np.r_[True, np.diff(cells_sorted) != 0]) if cells_sorted.size else np.zeros(0, int)
    ends = np.r_[starts[1:], cells_sorted.size]
    for start, end in zip(starts, ends):
        if end - start >= min_count:
            out[cells_sorted[start]] = float(np.median(values[start:end]))
    return out.reshape(n_cells, n_cells), cell_px, n_cells
