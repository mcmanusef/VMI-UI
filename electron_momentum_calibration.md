# 3D Electron Momentum Calibration for a Time-Resolved VMI

How to turn raw electron hits `(x, y, t)` from a velocity-map imaging (VMI) spectrometer with a
time-resolving detector (e.g. Timepix + electron ToF) into 3D momenta `(px, py, pz)` in atomic units.

The approach has two parts:

1. **Apply** a calibration: a small function with a center, a rotation angle, a detector-plane energy
   scale, and a polynomial mapping time offset → equivalent radius.
2. **Derive** the calibration from data by matching features (ATI rings) that appear both in the
   detector plane and along the time axis.

---

## 1. Model

### Coordinates

| Quantity | Meaning |
|---|---|
| `x, y` | hit position on detector (pixels) |
| `t` | electron arrival time (ns), from the best-resolution time source (e.g. a dedicated e-ToF pickoff) |
| `(xc, yc, tc)` | position and arrival time of zero-momentum electrons |
| `θ` | rotation that aligns detector axes with the laser polarization / lab frame |

Shift everything to the center first: `Δx = x − xc`, `Δy = y − yc`, `Δt = t − tc`.

### Detector plane

In a VMI, kinetic energy scales with radius squared:

```
E_xy(r) = k · r²          [eV],  k in eV / pixel²
```

### Time axis

Near `tc`, `Δt` is roughly linear in `pz`. Farther out it is not, and it behaves differently for
early (`Δt < 0`, toward detector) and late (`Δt > 0`, away from detector) electrons. Instead of
modeling the spectrometer field, map `Δt` to an **equivalent radius** `z` in pixels, so the time axis
shares the detector-plane energy scale:

```
z(Δt)² = P_early(Δt)   for Δt < 0
z(Δt)² = P_late(Δt)    for Δt > 0
E_t(Δt) = k · z(Δt)²
```

`P` is a polynomial with **no constant or linear term** (`a·t² + b·t³ + …`), which forces
`z(0) = 0` and `z ∝ |Δt|` near the center. Use as few terms as the data supports: often one term
(`a·t²`) on the side with few peaks, two on the other.

### Energy → momentum

```
p_i = sign(Δi) · sqrt(2 · E_i[eV] · 0.0367493)      # atomic units (m_e = 1); 0.0367493 Ha/eV
```

Take the sign from `Δx`, `Δy`, `Δt` directly. Squaring inside `E` discards it. Then rotate
`(px, py)` by `θ`:

```
px' =  px·cosθ + py·sinθ
py' = −px·sinθ + py·cosθ
```

Choose and document the `pz` sign convention: here `Δt > 0` (late) gives `pz > 0`.

### Optional post-processing

- **Symmetrize:** append `−p` for every event. This is valid only if the physics has inversion symmetry
  (linear polarization, achiral targets). Don't use it for PECD or ω/2ω experiments.
- **Hemisphere cut:** keep only `pz > 0` (or `< 0`) if one side of the time axis is poorly calibrated.

### Reference implementation

```python
import numpy as np

EV_TO_HARTREE = 0.0367493

def rotate(x, y, theta):
    return x * np.cos(theta) + y * np.sin(theta), y * np.cos(theta) - x * np.sin(theta)

def make_calibration(center, angle, k, early_coeffs, late_coeffs):
    """early/late_coeffs: highest power first, WITHOUT the trailing [0, 0]
    (i.e. [a] -> a·t², [b, a] -> b·t³ + a·t²)."""
    xc, yc, tc = center
    E_xy = lambda r: k * r**2

    def z2(dt):
        return np.where(dt < 0,
                        np.polyval(list(early_coeffs) + [0, 0], dt),
                        np.polyval(list(late_coeffs) + [0, 0], dt))

    def calibrate(x, y, t, symmetrize=False):
        dx, dy, dt = x - xc, y - yc, t - tc
        px = np.sign(dx) * np.sqrt(2 * EV_TO_HARTREE * E_xy(dx))
        py = np.sign(dy) * np.sqrt(2 * EV_TO_HARTREE * E_xy(dy))
        pz = np.sign(dt) * np.sqrt(2 * EV_TO_HARTREE * k * np.clip(z2(dt), 0, None))
        px, py = rotate(px, py, angle)
        if symmetrize:
            px, py, pz = (np.concatenate([a, -a]) for a in (px, py, pz))
        return px, py, pz

    return calibrate
```

Store each calibration as a named, dated function or config (`calibration_YYYYMMDD`) that holds its
center, angle, `k`, and coefficients. Re-derive it whenever voltages, detector position, or timing
cables change.

**Timing dither:** if arrival times are digitized in bins (e.g. ~0.26 ns TDC steps), add
`uniform(0, bin_width)` to `t` before calibrating. Otherwise quantization shows up as stripes in `pz`.

---

## 2. Deriving the calibration

### Data needed

Use a target with sharp, well-separated energy features. Strong-field ATI of a rare gas (Xe, Ar)
gives rings spaced by the photon energy. You need the same rings visible:

- **in the detector plane** along some axis, and
- **along the time axis**.

Two common ways to get this:

- **Two runs:** one with polarization **in the detector plane** (call it *P*), so the rings are
  strongest in `x`/`y`. One with polarization **along the ToF axis** (*S*), so the rings are strongest in `t`.
- **One run, using symmetry:** with linear polarization in the detector plane, the distribution is
  cylindrically symmetric about the polarization axis. The ToF axis is then equivalent to the in-plane
  axis perpendicular to polarization, so you can match `t` peaks to `y` peaks from the same file.
  Peaks off the polarization axis are weaker, so expect more noise.

### Step 1: center and angle

1. Histogram `(x, y)` on a log scale. Estimate `(xc, yc)` from ring symmetry, not from the center
   of mass or median, which are biased by asymmetric backgrounds or detector edges. Overplot all
   three as a check.
2. Histogram `t` for hits near `(xc, yc)` (on-axis, `r < ~2 px`). `tc` is the center of the
   distribution between the early and late ATI peak series. The zero-energy peak, if visible, sits here.
3. Select a thin time slice `|Δt| < ~1 ns`, histogram `(x, y)`, and overlay a line at angle `θ`
   through the center. Adjust `θ` until it lies along the polarization axis.
4. Iterate. A wrong center shows up later as non-circular rings.

### Step 2: in-plane ring radii

Take a 1D slice along the reference axis (e.g. `px` with `|py| < rw` and `|Δt| < tw`), separately
for positive and negative `x`:

- Histogram and smooth it (Savitzky–Golay, then light Gaussian).
- Find peaks. Peak-finding on the **negative second derivative** (clipped at 0) separates
  shoulders on a steeply falling background better than peak-finding on the raw histogram.
- Keep the N most prominent peaks and sort them by distance from center.

### Step 3: on-axis time peaks

Histogram `Δt` for on-axis hits, separately for `Δt > 0` and `Δt < 0`, using the same smoothing
and peak finding. Sort by `|Δt|`.

### Step 4: match and fit, per side

Pair the i-th radius peak `r_i` with the i-th time peak `t_i` (same ring = same energy). Then fit

```
r_i² = P(t_i)          # P with no constant/linear term
```

with `scipy.optimize.curve_fit`, weighting by the peak widths (e.g.
`sigma = hypot(width_r, width_t)`).

- Fit the late side and the early side independently. They generally need different polynomials.
- If one side has only 1–2 matched peaks, use a single `a·t²` term.
- Make sure the pairing starts at the same ring on both axes. Missing the first ring on one axis
  shifts every pair and still gives a smooth, plausible-looking, wrong fit. Check it visually.

### Step 5: validate

- Plot `(x, ±z(Δt))` (or `(px, pz)`) as a 2D histogram with equal aspect ratio. Overlay circles at the
  fitted `r_i`. **The rings must be circular and continuous across `Δt = 0`.**
- Check both datasets (S and P) if you have both. Features along the polarization axis in one should
  match the other.
- Check the energy spectrum computed from full 3D `|p|`. ATI peak spacing should be constant.

### Step 6: absolute energy scale `k`

`k` (eV/pixel²) sets the absolute scale for all three axes. Get it from the ATI peak spacing:

```
E_n = n·ħω − (Ip + Up)       →      r_n² = E_n / k
```

Plot `r_n²` against ring index `n`. The slope is `ħω / k`, so `k = ħω / slope`. This needs only the
photon energy and doesn't depend on `Up`. Re-derive `k` whenever the repeller/extractor voltages change.

### Step 7: export

Write the center, angle, `k`, and both coefficient sets into a named calibration (code or config),
along with the date and the dataset used. Keep the derivation notebook next to it.

---

## Pitfalls checklist

- [ ] Use the high-resolution electron time, not the camera ToA, for the time axis.
- [ ] Dither digitized times before calibrating.
- [ ] Center from ring symmetry, not the center of mass.
- [ ] Fit polynomials have no constant or linear term (`z(0) = 0`).
- [ ] Clip `z²` at 0 before `sqrt`. Polynomial fits can go negative outside the fitted range.
- [ ] Take signs from the raw offsets, not from the squared quantities.
- [ ] Ring pairing starts at the same ring on both axes.
- [ ] Don't extrapolate the time polynomial beyond the outermost fitted peak without checking.
- [ ] Re-derive `k` when the spectrometer voltages change.
- [ ] Symmetrize only when the physics allows it.
