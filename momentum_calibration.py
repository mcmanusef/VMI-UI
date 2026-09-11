"""Pure-logic helpers for momentum calibration: cv4 -> per-event DataFrame,
center/axis finding, radial + up/down time-of-flight spectra, peak fitting,
and the final position/time -> momentum conversion.

No Tkinter/matplotlib dependencies here (mirrors tpx_processing.py /
timewalk.py) so the math stays unit-testable and reusable headless.

--- Physical picture ------------------------------------------------------
A cv4 file holds, per laser pulse: zero or more detector clusters (x, y, t)
and zero or more e-ToF hits (t_etof) from the same pulse, tied together by a
shared pulse index (cluster_corr / etof_corr). Pairing every cluster with
every e-ToF hit from the same pulse (an inner join on that index) gives one
row per (cluster, e-ToF) coincidence -- the natural per-event table for a
VMI + e-ToF spectrometer, where genuine single-hit pulses dominate and any
extra combinations from multi-hit pulses just add uncorrelated background
that the later histogram/centroid steps average over.

The detector image (x, y) encodes the in-plane momentum (px, py); the e-ToF
timing encodes the out-of-plane momentum (pz), referenced to a "time zero"
(center_t) at which pz == 0. Both are calibrated against a photoelectron ATI
comb: peaks equally spaced by one photon energy, seen both as rings in the
radial (r = sqrt(x^2+y^2)) image and as peaks in the e-ToF spectrum.

- In-plane: energy ~ r^2 (a single scale factor `a`), so px = sqrt(2a)*x,
  py = sqrt(2a)*y once x, y are centered on the image center and rotated so
  the detector's symmetry axis lies along x (atomic units, m_e = 1).
- Out-of-plane: energy vs. |t - center_t| is not a simple power law, so it's
  fit with a polynomial in that time offset (no constant or linear term --
  E must vanish at pz == 0, and to leading order E ~ pz^2 near threshold),
  separately for the "up" branch (t < center_t: electrons that flew
  straight to the detector) and "down" branch (t > center_t: electrons
  initially ejected away, turned around by the extraction field).
"""
import dataclasses
import json
import pathlib

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

HARTREE_EV = 27.211386245988
HC_EV_NM = 1239.8419843320025  # h*c, in eV * nm


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cv4_events(path) -> pd.DataFrame:
    """Read a cv4 file into a per-event DataFrame with columns
    [pulse, x, y, t, etof] -- one row per (cluster, e-ToF hit) pair sharing a
    pulse. See the module docstring for why an inner join on the pulse index
    is the right pairing here."""
    with h5py.File(str(path), "r") as f:
        clusters = pd.DataFrame({
            "pulse": np.asarray(f["cluster_corr"]),
            "x": np.asarray(f["x"], dtype=np.float64),
            "y": np.asarray(f["y"], dtype=np.float64),
            "t": np.asarray(f["t"], dtype=np.float64),
        })
        etof = pd.DataFrame({
            "pulse": np.asarray(f["etof_corr"]),
            "etof": np.asarray(f["t_etof"], dtype=np.float64),
        })
    return clusters.merge(etof, on="pulse", how="inner").reset_index(drop=True)


def apply_t_gate(df: pd.DataFrame, t_min: float, t_max: float) -> pd.DataFrame:
    """Coarse pre-filter on the raw cluster-time column `t` (detector hit
    time relative to its pulse): keep only rows with t_min <= t <= t_max.
    Meant to be applied to both datasets right after loading, before any
    plot or fit sees the data -- drops obvious background/noise clusters
    (reflections, other pulses' leftovers, etc.) that would otherwise skew
    the xy heatmap, the center-of-mass/axis fit, and everything
    downstream."""
    mask = (df["t"] >= t_min) & (df["t"] <= t_max)
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Center of mass / principal axis
# ---------------------------------------------------------------------------

def center_of_mass(x, y, weights=None):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0
    if weights is None:
        return float(x.mean()), float(y.mean())
    w = np.asarray(weights, dtype=np.float64)
    wsum = w.sum()
    if wsum <= 0:
        return float(x.mean()), float(y.mean())
    return float((x * w).sum() / wsum), float((y * w).sum() / wsum)


def principal_axis_angle(x, y, weights=None):
    """Angle (radians, in [-pi/2, pi/2)) of the axis of greatest variation
    of the (x, y) distribution, from the eigenvector of the covariance
    matrix with the largest eigenvalue."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return 0.0
    cx, cy = center_of_mass(x, y, weights)
    dx = x - cx
    dy = y - cy
    if weights is None:
        cov = np.cov(np.vstack([dx, dy]))
    else:
        w = np.asarray(weights, dtype=np.float64)
        wsum = w.sum()
        if wsum <= 0:
            return 0.0
        cov = np.array([
            [(w * dx * dx).sum(), (w * dx * dy).sum()],
            [(w * dx * dy).sum(), (w * dy * dy).sum()],
        ]) / wsum
    evals, evecs = np.linalg.eigh(cov)
    principal = evecs[:, int(np.argmax(evals))]
    angle = float(np.arctan2(principal[1], principal[0]))
    # A line has no direction, so fold into [-pi/2, pi/2).
    if angle < -np.pi / 2:
        angle += np.pi
    elif angle >= np.pi / 2:
        angle -= np.pi
    return angle


def center_and_rotate(x, y, cx, cy, angle):
    """Translate by (-cx, -cy) then rotate by -angle, so the axis that was
    at `angle` from +x lands along +x and (cx, cy) becomes the origin."""
    x = np.asarray(x, dtype=np.float64) - cx
    y = np.asarray(y, dtype=np.float64) - cy
    c, s = np.cos(-angle), np.sin(-angle)
    x_rot = c * x - s * y
    y_rot = s * x + c * y
    return x_rot, y_rot


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

def xy_heatmap(x, y, bins=256, extent=(0.0, 256.0)):
    hist, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[[extent[0], extent[1]], [extent[0], extent[1]]]
    )
    return hist, xedges, yedges


def x_etof_heatmap(x, etof, x_bins=128, x_range=(-128.0, 128.0), t_bins=128, t_range=(0.0, 30000.0)):
    """2D histogram of (signed) x (already centered/rotated) vs. e-ToF, used
    to locate time-zero (center_t). Deliberately signed x rather than the
    one-sided r = sqrt(x^2+y^2) -- a center of mass along a strictly
    non-negative axis is biased toward larger r (more phase space out
    there), while x is symmetric about 0 and gives an unbiased centroid."""
    x = np.asarray(x, dtype=np.float64)
    hist, x_edges, t_edges = np.histogram2d(
        x, etof, bins=[x_bins, t_bins], range=[[x_range[0], x_range[1]], [t_range[0], t_range[1]]]
    )
    return hist, x_edges, t_edges


def radial_distribution(x, y, etof, r_bins=200, r_max=128.0, t_tol=200.0):
    """1D histogram of r (x, y already centered/rotated), restricted to
    |etof| < t_tol (etof already offset by center_t) so only events with
    ~zero out-of-plane momentum contribute -- otherwise pz smears the
    apparent radius for a given in-plane energy."""
    r = np.hypot(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    etof = np.asarray(etof, dtype=np.float64)
    mask = np.abs(etof) < t_tol
    counts, edges = np.histogram(r[mask], bins=int(r_bins), range=(0.0, float(r_max)))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def updown_distributions(x, y, etof, t_bins=200, t_max=20000.0, r_max=20.0):
    """Two 1D histograms of |etof| (etof already offset by center_t; x, y
    already centered/rotated), restricted to r < r_max so only events with
    ~zero in-plane momentum contribute. "up" is the etof < 0 branch
    (electrons that flew straight to the detector); "down" is etof > 0
    (electrons initially ejected away, turned around by the extraction
    field)."""
    r = np.hypot(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    etof = np.asarray(etof, dtype=np.float64)
    mask = r < r_max
    up_dt = -etof[mask & (etof < 0)]
    down_dt = etof[mask & (etof > 0)]
    edges = np.linspace(0.0, float(t_max), int(t_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    up_counts, _ = np.histogram(up_dt, bins=edges)
    down_counts, _ = np.histogram(down_dt, bins=edges)
    return centers, up_counts, down_counts


def smooth_and_find_peaks(counts, sigma=2.0, prominence_frac=0.05, distance=None):
    """Gaussian-smooth `counts` and find local maxima. `prominence_frac` is
    a fraction of the smoothed peak height, so it scales with the data
    instead of needing an absolute count threshold."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        return counts, np.array([], dtype=int)
    smoothed = gaussian_filter1d(counts, sigma=max(1e-6, float(sigma)))
    peak = float(smoothed.max())
    prominence = max(1e-9, peak * float(prominence_frac))
    kwargs = {"prominence": prominence}
    if distance:
        kwargs["distance"] = max(1, int(distance))
    peak_idx, _ = find_peaks(smoothed, **kwargs)
    return smoothed, peak_idx


# ---------------------------------------------------------------------------
# Photon energy / fits
# ---------------------------------------------------------------------------

def photon_energy_hartree(wavelength_nm: float) -> float:
    """Photon energy for `wavelength_nm` (vacuum wavelength, nm), in
    Hartree (atomic units)."""
    wavelength_nm = float(wavelength_nm)
    if wavelength_nm <= 0:
        raise ValueError("Wavelength must be positive.")
    return (HC_EV_NM / wavelength_nm) / HARTREE_EV


def fit_radial_energy(r_peaks, wavelength_nm):
    """Fit the r -> energy scale factor `a` (E = a*r^2) from selected
    radial-comb peaks, using only that they're spaced by one photon energy
    (their *absolute* ATI order doesn't matter -- shifting every order by a
    constant just shifts the fitted `e0`, which is a free parameter).

    Returns a dict with `a`, `e0`, `hv`, the sorted `r_peaks`, and the
    idealized comb `energies` (e0 + n*hv) at each peak -- it's these
    idealized energies, not the noisier a*r_i^2, that later steps reuse."""
    r_peaks = np.sort(np.asarray(r_peaks, dtype=np.float64))
    if len(r_peaks) < 2:
        raise ValueError("Select at least 2 radial peaks to fit the r -> energy calibration.")
    hv = photon_energy_hartree(wavelength_nm)
    n = np.arange(len(r_peaks), dtype=np.float64)
    x = r_peaks ** 2
    y = n * hv
    # y = a*x - e0
    a, neg_e0 = np.polyfit(x, y, 1)
    e0 = -neg_e0
    energies = e0 + n * hv
    return {
        "a": float(a),
        "e0": float(e0),
        "hv": float(hv),
        "r_peaks": r_peaks.tolist(),
        "energies": energies.tolist(),
    }


def match_energies(n_peaks, radial_energies):
    """The lowest `n_peaks` idealized comb energies from the radial fit,
    ascending -- matched by rank to a set of up/down peaks sorted ascending
    in |t - center_t| (larger time offset == larger |pz| == higher order)."""
    radial_energies = np.sort(np.asarray(radial_energies, dtype=np.float64))
    if n_peaks > len(radial_energies):
        raise ValueError(
            f"Selected {n_peaks} peaks here but only {len(radial_energies)} energies "
            "are available from the radial fit -- select fewer peaks here, or more in "
            "the radial spectrum."
        )
    return radial_energies[:n_peaks]


def fit_time_energy(t_peaks, energies):
    """Fit E(t) = c2*t^2 + c3*t^3 + ... + c_{k+1}*t^{k+1} (no constant or
    linear term) to the k selected (|t - center_t|, energy) peak pairs --
    exactly as many free coefficients as peaks, so the system solves
    exactly (least squares for k > available conditioning safety)."""
    t_peaks = np.asarray(t_peaks, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    k = len(t_peaks)
    if k == 0:
        raise ValueError("Select at least 1 peak to fit.")
    if k != len(energies):
        raise ValueError("Number of time peaks and matched energies must agree.")
    powers = np.arange(2, 2 + k)
    a_mat = t_peaks[:, None] ** powers[None, :]
    coeffs, *_ = np.linalg.lstsq(a_mat, energies, rcond=None)
    return {"powers": powers.tolist(), "coeffs": coeffs.tolist(), "t_peaks": t_peaks.tolist(), "energies": energies.tolist()}


def eval_time_energy(t, fit):
    """Evaluate a fit_time_energy() polynomial at |t| (the fit is defined
    on the positive time-offset domain it was built from)."""
    t = np.abs(np.asarray(t, dtype=np.float64))
    powers = np.asarray(fit["powers"], dtype=np.float64)
    coeffs = np.asarray(fit["coeffs"], dtype=np.float64)
    return np.sum(coeffs[None, :] * (t[..., None] ** powers[None, :]), axis=-1)


def energy_to_momentum(energy_hartree):
    """p = sqrt(2*E) (atomic units, m_e = 1). Negative energies (fit
    artifacts just past the calibrated range) are clipped to zero."""
    energy = np.clip(np.asarray(energy_hartree, dtype=np.float64), 0.0, None)
    return np.sqrt(2.0 * energy)


def compute_momenta(x, y, etof, radial_fit, up_fit=None, down_fit=None):
    """x, y already centered/rotated; etof already offset by center_t.
    Returns (px, py, pz), atomic units."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    etof = np.asarray(etof, dtype=np.float64)

    a = radial_fit["a"]
    scale = np.sqrt(2.0 * a) if a > 0 else 0.0
    px = scale * x
    py = scale * y

    pz = np.zeros_like(etof)
    down_mask = etof > 0
    up_mask = etof < 0
    if down_fit is not None and np.any(down_mask):
        pz[down_mask] = energy_to_momentum(eval_time_energy(etof[down_mask], down_fit))
    if up_fit is not None and np.any(up_mask):
        pz[up_mask] = -energy_to_momentum(eval_time_energy(etof[up_mask], up_fit))
    return px, py, pz


# ---------------------------------------------------------------------------
# Persisted calibration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MomentumCalibration:
    cx: float
    cy: float
    angle: float
    t_center: float
    wavelength_nm: float
    radial_fit: dict
    up_fit: dict = None
    down_fit: dict = None
    meta: dict = dataclasses.field(default_factory=dict)

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in dataclasses.fields(cls)}})

    def save(self, path):
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def peak_momenta(self):
        """sqrt(2*E) for every energy in the radial comb -- the radii of
        the ATI rings, for overlaying on a px/pz (or px/py) plot."""
        return energy_to_momentum(np.asarray(self.radial_fit["energies"], dtype=np.float64))

    def apply(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Apply this calibration to a *raw* (uncentered) events DataFrame,
        as returned by load_cv4_events -- i.e. from a fresh acquisition, not
        one already transformed by the calibration wizard. Returns a new
        DataFrame with x, y, etof centered/rotated/offset in place, plus
        px, py, pz columns."""
        out = raw_df.copy()
        out["x"], out["y"] = center_and_rotate(out["x"], out["y"], self.cx, self.cy, self.angle)
        out["etof"] = out["etof"] - self.t_center
        px, py, pz = compute_momenta(out["x"], out["y"], out["etof"], self.radial_fit, self.up_fit, self.down_fit)
        out["px"] = px
        out["py"] = py
        out["pz"] = pz
        return out


def save_calibrated_dataset(df: pd.DataFrame, path):
    """Save a calibrated (or intermediate) per-event DataFrame to a plain
    HDF5 file -- one flat dataset per column. Distinct from the pulse-
    indexed cv4 layout (cv4_writer.py) since this is already the
    flattened, paired per-event table."""
    path = pathlib.Path(path)
    with h5py.File(str(path), "w") as f:
        for col in df.columns:
            f.create_dataset(col, data=df[col].to_numpy(dtype=np.float64))
