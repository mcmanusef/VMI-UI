"""Ion time of flight -> mass-to-charge calibration. Headless (no Qt).

The user labels peaks in the i-ToF spectrum with a species ("H2O+", "Xe++")
or a plain m/q value. For ions starting at rest in a static extraction
field, t = t0 + a * sqrt(m/q), which is fit to the labelled peaks.
"""
import dataclasses
import datetime
import json
import pathlib
import re

import h5py
import numpy as np

ELECTRON_MASS_U = 0.000548579909

# Mass of the most abundant isotope (u). Resolved isotope peaks of heavier
# elements need a plain m/q number instead.
ISOTOPE_MASSES = {
    "H": 1.007825, "D": 2.014102, "He": 4.002603, "Li": 7.016003, "Be": 9.012183,
    "B": 11.009305, "C": 12.0, "N": 14.003074, "O": 15.994915, "F": 18.998403,
    "Ne": 19.992440, "Na": 22.989770, "Mg": 23.985042, "Al": 26.981538, "Si": 27.976927,
    "P": 30.973762, "S": 31.972071, "Cl": 34.968853, "Ar": 39.962383, "K": 38.963707,
    "Ca": 39.962591, "Fe": 55.934942, "Cu": 62.929598, "Zn": 63.929142, "Br": 78.918338,
    "Kr": 83.911507, "I": 126.904473, "Xe": 131.904154, "Cs": 132.905452,
}

_FORMULA_RE = re.compile(r"^((?:[A-Z][a-z]?\d*)+)(?:\^(\d+)\+?|(\++))?$")
_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_label(text):
    """m/q (u/e) for a peak label. Accepts a number ("18", "65.95") or a
    formula with an optional charge: "H2O+", "N2+" (singly charged N2),
    "Xe++" or "Xe^2+" (doubly charged). A formula with no charge is taken
    as singly charged."""
    text = text.strip().replace(" ", "")
    if not text:
        raise ValueError("Label is empty.")
    try:
        value = float(text)
    except ValueError:
        pass
    else:
        if value <= 0:
            raise ValueError("m/q must be positive.")
        return value

    match = _FORMULA_RE.match(text)
    if not match:
        raise ValueError(f"Could not read '{text}'. Use a number, or a formula like H2O+, Xe++ or Xe^2+.")
    formula, charge_digits, pluses = match.groups()
    charge = int(charge_digits) if charge_digits else (len(pluses) if pluses else 1)
    if charge <= 0:
        raise ValueError("Charge must be positive.")
    mass = 0.0
    for element, count in _ELEMENT_RE.findall(formula):
        if element not in ISOTOPE_MASSES:
            raise ValueError(f"Unknown element '{element}'. Enter the m/q value as a number instead.")
        mass += ISOTOPE_MASSES[element] * (int(count) if count else 1)
    return (mass - charge * ELECTRON_MASS_U) / charge


def load_ion_tof(path):
    with h5py.File(str(path), "r") as f:
        return np.asarray(f["t_tof"], dtype=np.float64)


def snap_tof_peak(centers, counts, t_click, window):
    """Peak position near a click: the highest bin within +/- window, then
    the count-weighted centroid of the contiguous bins above half its height."""
    centers = np.asarray(centers)
    counts = np.asarray(counts, dtype=np.float64)
    inside = np.flatnonzero(np.abs(centers - t_click) <= window)
    if inside.size == 0 or counts[inside].max() <= 0:
        return float(t_click)
    peak = inside[int(np.argmax(counts[inside]))]
    half = counts[peak] / 2
    lo = peak
    while lo > inside[0] and counts[lo - 1] > half:
        lo -= 1
    hi = peak
    while hi < inside[-1] and counts[hi + 1] > half:
        hi += 1
    weights = counts[lo:hi + 1]
    return float(np.sum(centers[lo:hi + 1] * weights) / np.sum(weights))


@dataclasses.dataclass
class MassCalibration:
    t0: float
    a: float
    fix_t0: bool = False
    peaks: list = dataclasses.field(default_factory=list)
    created: str = ""
    meta: dict = dataclasses.field(default_factory=dict)

    def tof(self, mq):
        return self.t0 + self.a * np.sqrt(np.asarray(mq, dtype=np.float64))

    def mass_to_charge(self, t):
        """m/q for each time; NaN for times before t0."""
        t = np.asarray(t, dtype=np.float64)
        return np.where(t > self.t0, ((t - self.t0) / self.a) ** 2, np.nan)

    def residuals(self):
        return [p["t"] - float(self.tof(p["mq"])) for p in self.peaks]

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data):
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def save(self, path):
        if not self.created:
            self.created = datetime.datetime.now().isoformat(timespec="seconds")
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


def fit_mass_calibration(peaks, fix_t0=False):
    """Least-squares t = t0 + a*sqrt(m/q) over peaks [{"label", "mq", "t"}].
    Needs 2 peaks, or 1 with t0 fixed at 0."""
    s = np.sqrt([p["mq"] for p in peaks])
    t = np.array([p["t"] for p in peaks], dtype=np.float64)
    if fix_t0:
        if s.size < 1:
            raise ValueError("Label at least 1 peak.")
        a, t0 = float(np.sum(s * t) / np.sum(s * s)), 0.0
    else:
        if np.unique(s).size < 2:
            raise ValueError("Label at least 2 peaks with different m/q, or fix t0 at 0.")
        a, t0 = (float(v) for v in np.polyfit(s, t, 1))
    if a <= 0:
        raise ValueError("Fit gives a non-positive time scale; check the peak labels.")
    return MassCalibration(t0=t0, a=a, fix_t0=fix_t0, peaks=[dict(p) for p in peaks])
