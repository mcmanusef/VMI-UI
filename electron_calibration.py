"""3D electron momentum calibration, as described in
electron_momentum_calibration.md. Headless (no Qt), like timewalk.py.

Model (doc section 1): shift hits to the zero-momentum point (xc, yc, tc),
use E = k * r^2 in the detector plane, and map the time offset to an
equivalent radius with z(dt)^2 = P_early(dt) for dt < 0 and P_late(dt) for
dt > 0. Each P has no constant or linear term. Then p = sign * sqrt(2 E),
in atomic units, followed by a rotation by the angle theta.

Derivation (doc section 2): find ring radii along an in-plane reference
axis and on-axis arrival-time peaks, pair them by rank on each time side,
fit r_i^2 = P(dt_i), and take k from the ring spacing (r_n^2 against n has
slope hbar*omega / k).

Two ways to get rings on both axes are supported:
  - "circular": one circularly polarized run. The distribution is symmetric
    about the propagation axis, which lies in the detector plane, so the ToF
    axis is equivalent to the in-plane axis perpendicular to propagation.
    theta is the propagation axis; ring radii come from the perpendicular
    axis. Rings and time peaks both come from that one run.
  - "s_p": one run with polarization along the ToF axis (S) for the time
    peaks, and one with polarization in the detector plane (P) for the
    center, angle and ring radii along the polarization axis (theta).
"""
import dataclasses
import datetime
import json
import pathlib

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import fftconvolve, find_peaks, peak_widths, savgol_filter

EV_TO_HARTREE = 0.0367493
HC_EV_NM = 1239.841984
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

TIME_SOURCE_ETOF = "e-ToF"
TIME_SOURCE_CLUSTER = "Cluster t"
TIME_SOURCES = (TIME_SOURCE_ETOF, TIME_SOURCE_CLUSTER)

MODE_CIRCULAR = "circular"
MODE_S_P = "s_p"


def photon_energy_ev(wavelength_nm):
    wavelength_nm = float(wavelength_nm)
    if wavelength_nm <= 0:
        raise ValueError("Wavelength must be positive.")
    return HC_EV_NM / wavelength_nm


def rotate(x, y, theta):
    """The doc's rotation: the axis at angle theta (from +x toward +y) lands on +x'."""
    return x * np.cos(theta) + y * np.sin(theta), y * np.cos(theta) - x * np.sin(theta)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ElectronEvents:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    path: str
    time_source: str


def pair_by_pulse(cluster_corr, etof_corr, single_hit_only):
    """Row indices (cluster_rows, etof_rows) of every cluster/e-ToF pair
    that shares a pulse. With single_hit_only, only pulses that have
    exactly one of each are kept, which avoids ambiguous pairings."""
    if single_hit_only:
        c_pulses, c_counts = np.unique(cluster_corr, return_counts=True)
        e_pulses, e_counts = np.unique(etof_corr, return_counts=True)
        common = np.intersect1d(c_pulses[c_counts == 1], e_pulses[e_counts == 1], assume_unique=True)
        c_rows = np.flatnonzero(np.isin(cluster_corr, common))
        e_rows = np.flatnonzero(np.isin(etof_corr, common))
        c_rows = c_rows[np.argsort(cluster_corr[c_rows], kind="stable")]
        e_rows = e_rows[np.argsort(etof_corr[e_rows], kind="stable")]
        return c_rows, e_rows

    order = np.argsort(etof_corr, kind="stable")
    sorted_corr = etof_corr[order]
    lo = np.searchsorted(sorted_corr, cluster_corr, side="left")
    hi = np.searchsorted(sorted_corr, cluster_corr, side="right")
    counts = hi - lo
    c_rows = np.repeat(np.arange(cluster_corr.size), counts)
    starts = np.repeat(lo, counts)
    offsets = np.arange(c_rows.size) - np.repeat(np.cumsum(counts) - counts, counts)
    return c_rows, order[starts + offsets]


def load_electron_events(path, time_source=TIME_SOURCE_ETOF, single_hit_only=True, dither_ns=0.0, rng=None):
    """Electron hits (x, y, t) from a cv4 file. With the e-ToF time source
    each detector cluster is paired with the e-ToF hit(s) from its pulse.
    With the cluster time source, t is the camera time instead. dither_ns
    adds uniform(0, dither_ns) to t, which removes stripes from digitized
    times (doc: timing dither)."""
    with h5py.File(str(path), "r") as f:
        x = np.asarray(f["x"], dtype=np.float64)
        y = np.asarray(f["y"], dtype=np.float64)
        cluster_corr = np.asarray(f["cluster_corr"])
        if time_source == TIME_SOURCE_ETOF:
            t_etof = np.asarray(f["t_etof"], dtype=np.float64)
            etof_corr = np.asarray(f["etof_corr"])
        else:
            t_cluster = np.asarray(f["t"], dtype=np.float64)

    if time_source == TIME_SOURCE_ETOF:
        c_rows, e_rows = pair_by_pulse(cluster_corr, etof_corr, single_hit_only)
        x, y, t = x[c_rows], y[c_rows], t_etof[e_rows]
    else:
        if single_hit_only:
            pulses, counts = np.unique(cluster_corr, return_counts=True)
            keep = np.isin(cluster_corr, pulses[counts == 1])
            x, y, t_cluster = x[keep], y[keep], t_cluster[keep]
        t = t_cluster

    if dither_ns > 0 and t.size:
        rng = rng if rng is not None else np.random.default_rng()
        t = t + rng.uniform(0.0, dither_ns, size=t.size)
    return ElectronEvents(x=x, y=y, t=t, path=str(path), time_source=time_source)


# ---------------------------------------------------------------------------
# Step 1: center and angle
# ---------------------------------------------------------------------------

def symmetry_center(x, y, extent=(0.0, 256.0), bin_px=1.0, background_sigma_px=6.0):
    """Center of the (x, y) image from point symmetry, not from the center of
    mass. The log image minus a blurred copy keeps the ring structure and
    drops smooth backgrounds. The autoconvolution C(s) = sum_i I(i) I(s - i)
    peaks at s = 2c when the image is symmetric about c."""
    bins = max(8, int(round((extent[1] - extent[0]) / bin_px)))
    hist, edges, _ = np.histogram2d(x, y, bins=bins, range=[extent, extent])
    if not hist.any():
        raise ValueError("No hits in the detector range.")
    image = np.log1p(hist)
    image = image - gaussian_filter(image, background_sigma_px / bin_px)
    conv = fftconvolve(image, image, mode="full")
    i, j = np.unravel_index(int(np.argmax(conv)), conv.shape)

    def refine(values, idx):
        if 0 < idx < values.size - 1:
            a, b, c = values[idx - 1], values[idx], values[idx + 1]
            denom = a - 2 * b + c
            if denom != 0:
                return idx + 0.5 * (a - c) / denom
        return float(idx)

    ci = refine(conv[:, j], i) / 2.0
    cj = refine(conv[i, :], j) / 2.0
    width = edges[1] - edges[0]
    return edges[0] + (ci + 0.5) * width, edges[0] + (cj + 0.5) * width


def center_of_mass(x, y):
    return float(np.mean(x)), float(np.mean(y))


def median_center(x, y):
    return float(np.median(x)), float(np.median(y))


def on_axis_mask(events, xc, yc, radius_px):
    return np.hypot(events.x - xc, events.y - yc) < radius_px


def estimate_time_zero(events, xc, yc, radius_px, t_window=None):
    """Starting guess for tc: the median on-axis arrival time. The user
    refines it so that it sits between the early and late peak series."""
    t = events.t[on_axis_mask(events, xc, yc, radius_px)]
    if t_window is not None:
        t = t[(t >= t_window[0]) & (t <= t_window[1])]
    if t.size == 0:
        raise ValueError("No on-axis hits; check the center and on-axis radius.")
    return float(np.median(t))


def principal_axis_angle(events, xc, yc, tc, slice_ns, r_max_px):
    """Angle (radians, in [-pi/2, pi/2)) of the long axis of a thin time
    slice |dt| < slice_ns, from second moments about (xc, yc). For linear
    in-plane polarization this is the polarization axis. For circular
    polarization it is perpendicular to the propagation axis."""
    dx, dy = events.x - xc, events.y - yc
    mask = (np.abs(events.t - tc) < slice_ns) & (np.hypot(dx, dy) < r_max_px)
    dx, dy = dx[mask], dy[mask]
    if dx.size < 3:
        raise ValueError("Too few hits in the thin time slice to estimate an angle.")
    cov = np.array([[np.mean(dx * dx), np.mean(dx * dy)], [np.mean(dx * dy), np.mean(dy * dy)]])
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    return wrap_axis_angle(float(np.arctan2(major[1], major[0])))


def wrap_axis_angle(theta):
    """An axis has no direction, so fold into [-pi/2, pi/2)."""
    return (theta + np.pi / 2) % np.pi - np.pi / 2


# ---------------------------------------------------------------------------
# Steps 2 and 3: slices and peak finding
# ---------------------------------------------------------------------------

def axis_coordinates(events, xc, yc, theta, perpendicular):
    """(u, v): signed position along the reference axis and across it. The
    reference axis is x' (at theta), or y' when `perpendicular`."""
    u, v = rotate(events.x - xc, events.y - yc, theta)
    return (v, u) if perpendicular else (u, v)


def in_plane_slice(events, xc, yc, tc, theta, perpendicular, half_width_px, half_width_ns):
    """Signed position along the reference axis for hits within
    half_width_px of that axis and |dt| < half_width_ns."""
    u, v = axis_coordinates(events, xc, yc, theta, perpendicular)
    mask = (np.abs(v) < half_width_px) & (np.abs(events.t - tc) < half_width_ns)
    return u[mask]


def smooth_counts(counts, savgol_window=7, gauss_sigma=1.0):
    """Savitzky-Golay, then a light Gaussian (doc step 2)."""
    smoothed = np.asarray(counts, dtype=np.float64)
    window = int(savgol_window)
    if window >= 3:
        window |= 1
        if window > smoothed.size:
            window = smoothed.size if smoothed.size % 2 else smoothed.size - 1
        if window >= 3:
            smoothed = savgol_filter(smoothed, window, 2)
    if gauss_sigma > 0:
        smoothed = gaussian_filter1d(smoothed, float(gauss_sigma))
    return smoothed


def peak_curve(smoothed):
    """Negative second derivative, clipped at 0. It separates shoulders on a
    steeply falling background better than the raw histogram does."""
    return np.clip(-np.gradient(np.gradient(smoothed)), 0.0, None)


@dataclasses.dataclass
class Spectrum:
    """A 1D histogram with its smoothed and peak-finding curves."""
    centers: np.ndarray
    counts: np.ndarray
    smoothed: np.ndarray
    curve: np.ndarray

    @property
    def bin_width(self):
        return float(self.centers[1] - self.centers[0]) if self.centers.size > 1 else 1.0


def make_spectrum(values, lo, hi, bin_width, savgol_window=7, gauss_sigma=1.0):
    if hi <= lo or bin_width <= 0:
        raise ValueError("Histogram max must exceed min, and bin width must be positive.")
    n_bins = max(8, int(np.ceil((hi - lo) / bin_width)))
    counts, edges = np.histogram(values, bins=n_bins, range=(lo, lo + n_bins * bin_width))
    smoothed = smooth_counts(counts, savgol_window, gauss_sigma)
    return Spectrum(0.5 * (edges[:-1] + edges[1:]), counts, smoothed, peak_curve(smoothed))


def _peak_sigma(spectrum, index):
    try:
        fwhm_bins = peak_widths(spectrum.curve, [index], rel_height=0.5)[0][0]
    except ValueError:
        fwhm_bins = 0.0
    return max(float(fwhm_bins), 1.0) * spectrum.bin_width * FWHM_TO_SIGMA


def find_spectrum_peaks(spectrum, n_peaks, lo, hi):
    """The n_peaks most prominent peaks of the peak curve with lo < position
    < hi, as [(position, sigma)] sorted by position."""
    indices, props = find_peaks(spectrum.curve, prominence=0)
    inside = (spectrum.centers[indices] > lo) & (spectrum.centers[indices] < hi)
    indices, prominences = indices[inside], props["prominences"][inside]
    keep = indices[np.argsort(prominences)[::-1][: max(0, int(n_peaks))]]
    return sorted((float(spectrum.centers[i]), _peak_sigma(spectrum, i)) for i in keep)


def snap_to_peak(spectrum, position, window_bins=4):
    """A manually clicked peak: the maximum of the peak curve within
    window_bins of the click, or the click itself if the curve is flat there."""
    idx = int(np.clip(np.searchsorted(spectrum.centers, position), 0, spectrum.centers.size - 1))
    lo, hi = max(0, idx - window_bins), min(spectrum.centers.size, idx + window_bins + 1)
    best = lo + int(np.argmax(spectrum.curve[lo:hi]))
    if spectrum.curve[best] <= 0:
        return float(position), spectrum.bin_width * 2 * FWHM_TO_SIGMA
    return float(spectrum.centers[best]), _peak_sigma(spectrum, best)


# ---------------------------------------------------------------------------
# Step 4: match and fit; step 6: energy scale
# ---------------------------------------------------------------------------

def ring_radii(signed_peaks):
    """Ring radii from peaks on both sides of the reference axis, each side
    sorted by distance from center and paired by rank. Where both sides
    have ring i, the radius is their mean. Returns (r, sigma, half_diff);
    half_diff is NaN where only one side has the ring. A consistent nonzero
    half_diff means the center is off along that axis."""
    pos = sorted((p, s) for p, s in signed_peaks if p > 0)
    neg = sorted(((-p, s) for p, s in signed_peaks if p < 0))
    r, sigma, half_diff = [], [], []
    for i in range(max(len(pos), len(neg))):
        sides = [side[i] for side in (pos, neg) if i < len(side)]
        r.append(np.mean([p for p, _ in sides]))
        sigma.append(np.sqrt(np.mean([s * s for _, s in sides])))
        half_diff.append((pos[i][0] - neg[i][0]) / 2 if len(sides) == 2 else np.nan)
    return np.array(r), np.array(sigma), np.array(half_diff)


def split_time_peaks(time_peaks, tc):
    """(early, late) lists of (dt, sigma), each sorted by |dt|."""
    early = sorted(((t - tc, s) for t, s in time_peaks if t < tc), key=lambda p: -p[0])
    late = sorted((t - tc, s) for t, s in time_peaks if t > tc)
    return early, late


def z2_polyval(coeffs, dt):
    """z^2 = P(dt), with coeffs highest power first and without the trailing
    [0, 0] (doc convention: [a] -> a*t^2, [b, a] -> b*t^3 + a*t^2)."""
    if not len(coeffs):
        return np.zeros_like(np.asarray(dt, dtype=np.float64))
    return np.polyval(list(coeffs) + [0.0, 0.0], dt)


def monotonic_dt_limit(coeffs, sign, limit):
    """How far from dt = 0 (toward sign, up to |limit|) |z|^2 = P(dt) keeps
    increasing and stays positive. Beyond that point the fitted polynomial
    folds back, so hits there would get a smaller |pz| or be clipped to 0.
    Returns a signed dt. A side without coefficients (z^2 = 0) keeps the
    full limit."""
    limit = abs(float(limit))
    if not len(coeffs):
        return sign * limit
    # P(dt) = dt^2 Q(dt) and P'(dt) = dt R(dt), so the limit is the nearest
    # root of Q or R on this side, unless P already fails just past 0.
    full = np.array(list(coeffs) + [0.0, 0.0], dtype=np.float64)
    q = np.array(coeffs, dtype=np.float64)
    r = np.polyder(full)[:-1]
    eps = sign * min(limit, 1.0) * 1e-9
    if np.polyval(q, eps) <= 0 or sign * np.polyval(r, eps) * eps <= 0:
        return 0.0
    nearest = limit
    for poly in (q, r):
        if poly.size > 1:
            for root in np.roots(poly):
                if abs(root.imag) < 1e-9 and sign * root.real > 0:
                    nearest = min(nearest, sign * root.real)
    return float(sign * nearest)


@dataclasses.dataclass
class SideFit:
    coeffs: list
    dt: np.ndarray
    r: np.ndarray
    residuals_px2: np.ndarray

    @property
    def n_pairs(self):
        return int(self.dt.size)


def fit_side(r, r_sigma, time_peaks, n_terms):
    """Fit r_i^2 = P(dt_i) for one side (doc step 4). Pairs the i-th ring
    radius with the i-th time peak. Uses a linear least squares on the
    coefficients, weighted by the propagated uncertainty of each pair,
    sigma = hypot(2 r sigma_r, P'(dt) sigma_t), where P' comes from an
    unweighted first pass. Returns None if there are no pairs."""
    n = min(len(r), len(time_peaks))
    if n == 0:
        return None
    n_terms = max(1, min(int(n_terms), n))
    r = np.asarray(r[:n], dtype=np.float64)
    r_sigma = np.asarray(r_sigma[:n], dtype=np.float64)
    dt = np.array([p[0] for p in time_peaks[:n]])
    dt_sigma = np.array([p[1] for p in time_peaks[:n]])
    powers = np.arange(n_terms + 1, 1, -1)
    design = dt[:, None] ** powers[None, :]
    target = r ** 2

    coeffs = np.linalg.lstsq(design, target, rcond=None)[0]
    slope = np.sum(coeffs[None, :] * powers[None, :] * dt[:, None] ** (powers[None, :] - 1), axis=1)
    sigma = np.hypot(2 * r * r_sigma, slope * dt_sigma)
    if np.all(sigma > 0):
        coeffs = np.linalg.lstsq(design / sigma[:, None], target / sigma, rcond=None)[0]

    residuals = target - design @ coeffs
    return SideFit(coeffs=[float(c) for c in coeffs], dt=dt, r=r, residuals_px2=residuals)


@dataclasses.dataclass
class EnergyScaleFit:
    k: float
    slope: float
    intercept: float
    photon_energy_ev: float
    r: np.ndarray


def fit_energy_scale(r, photon_energy):
    """k from the ring spacing (doc step 6): r_n^2 against ring index n has
    slope hbar*omega / k. Assumes the rings are consecutive ATI orders."""
    r = np.asarray(r, dtype=np.float64)
    if r.size < 2:
        raise ValueError("Need at least 2 ring radii to fit the energy scale.")
    slope, intercept = np.polyfit(np.arange(r.size), r ** 2, 1)
    if slope <= 0:
        raise ValueError("Ring radii must increase with index.")
    return EnergyScaleFit(
        k=float(photon_energy / slope), slope=float(slope), intercept=float(intercept),
        photon_energy_ev=float(photon_energy), r=r,
    )


# ---------------------------------------------------------------------------
# Applying a calibration (doc section 1) and export (step 7)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MomentumCalibration:
    name: str
    center: tuple
    angle: float
    k: float
    early_coeffs: list
    late_coeffs: list
    time_source: str = TIME_SOURCE_ETOF
    dither_ns: float = 0.0
    created: str = ""
    meta: dict = dataclasses.field(default_factory=dict)
    # (early, late) signed dt limits in ns; None means no stored limit.
    dt_range: tuple = None

    def z2(self, dt):
        dt = np.asarray(dt, dtype=np.float64)
        return np.where(dt < 0, z2_polyval(self.early_coeffs, dt), z2_polyval(self.late_coeffs, dt))

    def valid_dt_range(self):
        """(early, late) dt limits where the calibration applies: the stored
        dt_range, shortened on each side to where z^2 stops increasing."""
        early, late = self.dt_range if self.dt_range is not None else (-np.inf, np.inf)
        search = 1e4  # ns; far past any electron arrival for an unlimited side
        early = max(early, monotonic_dt_limit(self.early_coeffs, -1, min(-early, search)))
        late = min(late, monotonic_dt_limit(self.late_coeffs, 1, min(late, search)))
        return early, late

    def calibrate(self, x, y, t, symmetrize=False, dither=False, rng=None):
        """(px, py, pz) in atomic units. pz > 0 for late electrons (dt > 0).
        Hits outside valid_dt_range() get NaN for all three components, since
        the fitted polynomials don't describe them. dither=True adds
        uniform(0, dither_ns) to t first, for raw data that wasn't already
        dithered on load."""
        xc, yc, tc = self.center
        t = np.asarray(t, dtype=np.float64)
        if dither and self.dither_ns > 0:
            rng = rng if rng is not None else np.random.default_rng()
            t = t + rng.uniform(0.0, self.dither_ns, size=t.shape)
        dx = np.asarray(x, dtype=np.float64) - xc
        dy = np.asarray(y, dtype=np.float64) - yc
        dt = t - tc
        px = np.sign(dx) * np.sqrt(2 * EV_TO_HARTREE * self.k * dx ** 2)
        py = np.sign(dy) * np.sqrt(2 * EV_TO_HARTREE * self.k * dy ** 2)
        pz = np.sign(dt) * np.sqrt(2 * EV_TO_HARTREE * self.k * np.clip(self.z2(dt), 0, None))
        px, py = rotate(px, py, self.angle)
        early, late = self.valid_dt_range()
        outside = (dt < early) | (dt > late)
        px, py, pz = (np.where(outside, np.nan, a) for a in (px, py, pz))
        if symmetrize:
            px, py, pz = (np.concatenate([a, -a]) for a in (px, py, pz))
        return px, py, pz

    def to_dict(self):
        data = dataclasses.asdict(self)
        data["center"] = [float(c) for c in self.center]
        data["angle_deg"] = float(np.degrees(self.angle))
        return data

    @classmethod
    def from_dict(cls, data):
        fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in fields}
        kwargs["center"] = tuple(float(c) for c in kwargs["center"])
        if kwargs.get("dt_range") is not None:
            kwargs["dt_range"] = tuple(float(v) for v in kwargs["dt_range"])
        return cls(**kwargs)

    def save(self, path):
        if not self.created:
            self.created = datetime.datetime.now().isoformat(timespec="seconds")
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


def energy_ev(px, py, pz):
    return (px ** 2 + py ** 2 + pz ** 2) / (2 * EV_TO_HARTREE)


def default_calibration_name(date=None):
    return f"calibration_{(date or datetime.date.today()).strftime('%Y%m%d')}"
