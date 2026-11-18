"""Cluster-level timing calibration: an alternative to the per-ToT
time-walk LUT in timewalk.py.

The legacy correction (timewalk.TimewalkCorrection) subtracts one offset per
ToT value from every hit before clustering, and the cluster time is then the
ToT-weighted mean of the hit times plus a 1.56 ns random dither. This module
instead works on whole clusters, in the order below. Every step was measured
against a per-electron reference (the FPGA e-ToF TDC); see
fit_cluster_timing.py for how a calibration file is made.

  1. Coarse-clock column fix: whole columns whose ToA is off by an exact
     number of 25 ns clock ticks (found on chip 0, local columns 194-203, of
     the 2026-08-20 data) are shifted back.
  2. Time-walk on two axes: the hit's ToT and its share of the cluster's
     peak ToT (dim hits next to a bright pixel walk differently from dim
     hits on their own).
  3. Per-pixel walk amplitude and offset (pixel-to-pixel threshold and gain
     spread), fitted against each cluster's own time.
  4. A cluster time from the corrected hits, with each hit weighted by
     1/sigma^2 of its walk class and 3-sigma clipping, in one of three
     MODES:
        "all"    -- weighted mean of every hit,
        "bright" -- weighted mean of hits with ToT >= 1 us only (the dim
                    halo hits share a several-ns error per cluster, so they
                    add bias rather than information),
        "core"   -- a fitted mix of the ToT-class means plus a small
                    additive table in peak ToT, ring-minus-core time,
                    centroid x/y and hit count.
  5. Cluster-level terms: a smooth curve in log10(sum ToT) and a 16-pixel
     cell map of the detector, both fitted separately for each mode.

Step 1 is applied to the hits before clustering
(timewalk.apply_timewalk_correction); steps 2-5 replace each cluster's time
inside tpx_processing.cluster_hits(timing=...).

A calibration is only valid for the detector settings (threshold, bias, MCP
gain) and the chip layout it was fitted with; refit when those change.

No Tkinter or Qt here.
"""
import json
import pathlib
import time

import numpy as np

import app_settings

TOT_NS_PER_CODE = 25.0
FRAC_EDGES = np.array([0.0, 0.01, 0.02, 0.035, 0.05, 0.1, 0.2, 0.5, 1.0001])
N_FRAC = len(FRAC_EDGES)              # key stride (only N_FRAC - 1 bins are used)
CODE_CAP = 200                        # ToT codes above this share the 2D-walk row of code 200
CLASS_EDGES_NS = np.array([25.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 30000.0])
N_CLASSES = len(CLASS_EDGES_NS) - 1   # 25-100, 100-250, 250-500, 0.5-1k, 1-2k, >2k ns
CORE_CLASS = N_CLASSES - 1            # >2 us
MIX_CLASSES = (2, 3, 4)               # mixed into the core class mean in "core" mode
RING_CLASS = 4                        # 1-2 us, for the ring-minus-core feature
BRIGHT_TOT_NS = 1000.0
CELL_PX = 16
MODES = ("all", "bright", "core")
DEFAULT_MODE = "core"
MODE_SETTINGS_KEY = "cluster_timing.mode"
CLIP_NSIGMA = 3.0
CLIP_ITERATIONS = 2
CORE_TABLE_FEATURES = ("maxtot", "ring12_minus_core", "cx", "cy", "nh")
FILE_FORMAT = "cluster_timing_v1"

# The cv4 root attribute recording which cluster-timing calibration a file used.
CLUSTER_TIMING_ATTR = "Cluster Timing Calibration"


def is_calibration_path(path):
    """True for a cluster-timing calibration file (.npz), False for a legacy
    time-walk LUT (.json) -- the time-walk file selectors on each tab accept
    either and this decides which one is loaded."""
    return str(path).strip().lower().endswith(".npz")


# ---- small numeric helpers ---------------------------------------------------

def _bin_mean(ids, values, weights, n):
    s = np.bincount(ids, weights * values, minlength=n)
    sw = np.bincount(ids, weights, minlength=n)
    return np.where(sw > 0, s / np.where(sw > 0, sw, 1.0), np.nan), sw


def clipped_weighted_mean(cid, n_clusters, t, w, nsig=CLIP_NSIGMA, iterations=CLIP_ITERATIONS):
    """Per-cluster weighted mean of `t` with iterative nsig clipping (residual
    scaled by sqrt(w)). Returns (mean, keep-mask). A cluster whose every hit
    gets clipped falls back to its unclipped weighted mean."""
    # The first pass keeps every hit, so its mean is also the unclipped
    # fallback mean.
    m_all, _ = _bin_mean(cid, t, w, n_clusters)
    m = m_all
    keep = np.ones(len(t), dtype=bool)
    sqrt_w = np.sqrt(w)
    for i in range(iterations):
        if i:
            m, _ = _bin_mean(cid, t, w * keep, n_clusters)
        res = (t - np.nan_to_num(m)[cid]) * sqrt_w
        keep = np.abs(res) < nsig
    m, sw = _bin_mean(cid, t, w * keep, n_clusters)
    m = np.where(sw > 0, m, m_all)
    return m, keep


def _count_edges(v, edges):
    """np.digitize(v, edges) for finite v and increasing edges: how many
    edges are <= each value (a few compares instead of a search per value)."""
    out = np.zeros(len(v), dtype=np.int64 if len(edges) > 127 else np.int8)
    for e in edges:
        out += v >= e
    return out


def _piecewise(x, nodes, values):
    return np.interp(x, nodes, values)


def _table_bins(edges, v):
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)


# ---- per-cluster quantities shared by fitting and applying -----------------

def cluster_geometry(cid, n_clusters, x, y, tot_code):
    """Per-hit share of the cluster's peak ToT, and per-cluster ToT-weighted
    centroid, sum ToT (ns), peak ToT (ns) and hit count. `tot_code` is the raw
    10-bit ToT (25 ns units), as in tpx_processing.PixelHits.tot."""
    tot_ns = tot_code.astype(np.float64) * TOT_NS_PER_CODE
    maxtot = np.zeros(n_clusters)
    np.maximum.at(maxtot, cid, tot_ns)
    hit_max = maxtot[cid]
    frac = tot_ns / np.where(hit_max > 0, hit_max, 1.0)
    del hit_max
    wt = np.where(tot_ns > 0, tot_ns, 1.0)
    # cx and cy share their weight sum.
    sw = np.bincount(cid, wt, minlength=n_clusters)
    has_w = sw > 0
    sw_safe = np.where(has_w, sw, 1.0)
    cx = np.where(has_w, np.bincount(cid, wt * x.astype(np.float64), minlength=n_clusters) / sw_safe, np.nan)
    cy = np.where(has_w, np.bincount(cid, wt * y.astype(np.float64), minlength=n_clusters) / sw_safe, np.nan)
    sumtot = np.bincount(cid, tot_ns, minlength=n_clusters)
    nh = np.bincount(cid, minlength=n_clusters).astype(np.float64)
    return dict(tot_ns=tot_ns, frac=frac, cx=cx, cy=cy, sumtot=sumtot,
                log_sumtot=np.log10(np.maximum(sumtot, 1.0)), maxtot=maxtot, nh=nh)


def frac_bins(frac):
    """Peak-ToT-share bin (0 .. N_FRAC - 2) of each hit."""
    if np.isnan(frac).any():
        bins = np.digitize(frac, FRAC_EDGES)
    else:
        bins = _count_edges(frac, FRAC_EDGES)
    return np.clip(bins - 1, 0, N_FRAC - 2)


def walk_keys(tot_code, frac, n_codes):
    code = np.clip(tot_code.astype(np.int64), 0, n_codes - 1)
    key = np.minimum(code, CODE_CAP) * N_FRAC + frac_bins(frac)
    return code, key


def tot_classes(tot_ns, tot_code=None):
    """ToT class (0 .. N_CLASSES - 1) of each hit. With the raw codes (whole
    numbers >= 0, tot_ns = code * 25 ns) this is a lookup per code rather
    than a search per hit."""
    if tot_code is not None and len(tot_code) and tot_code.min() >= 0:
        codes = np.arange(int(tot_code.max()) + 1)
        table = np.clip(np.digitize(codes.astype(np.float64) * TOT_NS_PER_CODE, CLASS_EDGES_NS) - 1,
                        0, N_CLASSES - 1).astype(np.int8)
        return table[tot_code]
    return np.clip(np.digitize(tot_ns, CLASS_EDGES_NS) - 1, 0, N_CLASSES - 1)


def class_means(cid, n_clusters, t, w, keep, tot_ns, cls=None):
    """Weighted mean hit time per ToT class (columns) per cluster (rows);
    NaN where a cluster has no kept hit in that class. `cls`: each hit's
    tot_classes(), if already known.

    One weighted bincount over (cluster, class) pairs: each pair's sum gets
    the same terms, in the same hit order, as a separate pass per class."""
    if cls is None:
        cls = tot_classes(tot_ns)
    valid = keep & (tot_ns >= CLASS_EDGES_NS[0])
    idx = cid[valid] * N_CLASSES + cls[valid]
    wv = w[valid]
    size = n_clusters * N_CLASSES
    s = np.bincount(idx, wv * t[valid], minlength=size)
    sw = np.bincount(idx, wv, minlength=size)
    has_w = sw > 0
    out = np.where(has_w, s / np.where(has_w, sw, 1.0), np.nan)
    return out.reshape(n_clusters, N_CLASSES)


class ClusterTimingCalibration:
    """Everything the cluster-timing correction needs, loaded from / saved to
    one .npz file. `mode` picks the estimator (see MODES); it defaults to the
    app setting "cluster_timing.mode" if set, otherwise the file's own
    default."""

    def __init__(self, arrays, meta, mode=None):
        self.a = arrays     # shared, not copied: fit_calibration adds to it stage by stage
        self.meta = meta
        chosen = mode or app_settings.get(MODE_SETTINGS_KEY, None) or self.meta.get("default_mode", DEFAULT_MODE)
        if chosen not in MODES:
            raise ValueError(f"Unknown cluster timing mode {chosen!r}; expected one of {MODES}")
        self.mode = chosen
        # Cluster-time dither is off in this mode: the 1.56 ns uniform dither
        # (tpx_processing.CLUSTER_T_JITTER_NS) would add 0.45 ns rms to a
        # cluster time that is good to ~0.7 ns.
        self.jitter_ns = float(self.meta.get("jitter_ns", 0.0))
        self.path = None
        self._column_fix_map = None
        self._column_fix_layout = None

    # ---- file I/O ----

    def save(self, path):
        path = pathlib.Path(path)
        np.savez_compressed(path, meta_json=np.array(json.dumps(self.meta)), **self.a)
        self.path = str(path)

    @classmethod
    def load(cls, path, mode=None):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"]))
            if meta.get("format") != FILE_FORMAT:
                raise ValueError(f"{path} is not a {FILE_FORMAT} cluster timing calibration")
            arrays = {k: data[k] for k in data.files if k != "meta_json"}
        obj = cls(arrays, meta, mode=mode)
        obj.path = str(path)
        return obj

    def describe(self):
        return f"{self.path or '(unsaved)'} [mode={self.mode}]"

    # ---- step 1: whole-column coarse-clock offsets ----

    def column_fix_map(self, layout):
        """Canvas-sized (width x height) array of ns to subtract per pixel for
        the 25 ns column fix, built from chip-local columns so it follows the
        active chip layout."""
        key = json.dumps(layout.to_dict(), sort_keys=True) if hasattr(layout, "to_dict") else id(layout)
        if self._column_fix_map is not None and self._column_fix_layout == key:
            return self._column_fix_map
        fix = np.zeros((layout.size, layout.size))
        cols = self.a.get("column_fix", np.zeros((0, 3)))
        ys = np.arange(256)
        for chip, col, ns in np.asarray(cols).reshape(-1, 3):
            gx, gy = layout.to_global(int(chip), np.full(256, int(col)), ys)
            fix[np.asarray(gx, dtype=int), np.asarray(gy, dtype=int)] += ns
        self._column_fix_map = fix
        self._column_fix_layout = key
        return fix

    def check_layout(self, layout):
        """A warning string if the active layout differs from the one this
        calibration's per-pixel and cell maps were fitted with, else None."""
        fitted = self.meta.get("layout")
        if fitted is None or not hasattr(layout, "to_dict"):
            return None
        if json.dumps(fitted, sort_keys=True) != json.dumps(layout.to_dict(), sort_keys=True):
            return ("Chip layout differs from the one this cluster timing calibration was "
                    "fitted with; its per-pixel and cell maps will be misplaced.")
        return None

    # ---- steps 2-5 ----

    def hit_times(self, cid, n_clusters, x, y, t, tot_code, geom=None):
        """Walk- and pixel-corrected hit times and their weights (inverse
        variance). Hit coordinates are canvas coordinates; `t` already has the
        column fix applied."""
        g = geom if geom is not None else cluster_geometry(cid, n_clusters, x, y, tot_code)
        walk_table, weight_table = self._walk_tables()
        n_codes = walk_table.shape[0]
        # One lookup per hit for walk1d[code] + walk2d[key] and for
        # 1 / sigma[key]^2 (the same sums and quotients, done once per
        # (code, frac bin) instead of once per hit).
        idx = np.clip(tot_code.astype(np.int64), 0, n_codes - 1) * (N_FRAC - 1) + frac_bins(g["frac"])
        w_total = walk_table.take(idx)
        weight = weight_table.take(idx)
        del idx
        off = self.a["pixel_offset_ns"]
        slope = self.a["pixel_walk_slope"]
        pix = (np.clip(x, 0, off.shape[0] - 1).astype(np.int64) * off.shape[1]
               + np.clip(y, 0, off.shape[1] - 1).astype(np.int64))
        t_corr = t - w_total - (off.ravel().take(pix) + slope.ravel().take(pix) * w_total)
        return t_corr, weight, g

    def _walk_tables(self):
        """(walk1d[code] + walk2d[key], 1 / sigma2d[key]^2) for every
        (ToT code, frac bin), flattened as code * (N_FRAC - 1) + frac bin.
        Rebuilt if the walk arrays are replaced (fit_calibration fills them
        in stages)."""
        stamp = tuple(id(self.a[k]) for k in ("walk1d_ns", "walk2d_ns", "walk2d_sigma_ns"))
        walk1d = self.a["walk1d_ns"]
        walk2d = self.a["walk2d_ns"].ravel()
        sigma2d = self.a["walk2d_sigma_ns"].ravel()
        cached = getattr(self, "_walk_cache", None)
        if cached is not None and cached[0] == stamp:
            return cached[1], cached[2]
        code = np.arange(len(walk1d))[:, None]
        key = np.minimum(code, CODE_CAP) * N_FRAC + np.arange(N_FRAC - 1)[None, :]
        walk_table = (walk1d[code] + walk2d[key]).ravel()
        weight_table = (1.0 / sigma2d[key] ** 2).ravel()
        self._walk_cache = (stamp, walk_table, weight_table)
        return walk_table, weight_table

    def _smooth(self, prefix, log_sumtot, cx, cy):
        nodes = self.a["charge_nodes"]
        charge = _piecewise(log_sumtot, nodes, self.a[f"{prefix}_charge_ns"])
        cell = cell_ids(cx, cy, int(self.meta.get("cell_stride", 32)))
        ids = self.a[f"{prefix}_cell_ids"]
        vals = self.a[f"{prefix}_cell_ns"]
        cellv = np.zeros(len(cell))
        if len(ids):
            pos = np.clip(np.searchsorted(ids, cell), 0, len(ids) - 1)
            cellv = np.where(ids[pos] == cell, vals[pos], 0.0)
        return charge + cellv

    def cluster_times(self, cid, n_clusters, x, y, t, tot_code, mode=None):
        """Corrected time for every cluster. cid: per-hit cluster index
        (0..n_clusters-1); x, y: canvas coordinates; t: hit time (ns, after
        the column fix); tot_code: raw ToT. Returns an array of n_clusters."""
        mode = mode or self.mode
        tt, w, g = self.hit_times(cid, n_clusters, x, y, t, tot_code)
        j_all, keep = clipped_weighted_mean(cid, n_clusters, tt, w)
        if mode == "all":
            return j_all - self._smooth("all", g["log_sumtot"], g["cx"], g["cy"])
        if mode == "bright":
            t1, sw = _bin_mean(cid, tt, w * (keep & (g["tot_ns"] >= BRIGHT_TOT_NS)), n_clusters)
            t1 = np.where(sw > 0, t1, j_all)
            return t1 - self._smooth("bright", g["log_sumtot"], g["cx"], g["cy"])
        cm = class_means(cid, n_clusters, tt, w, keep, g["tot_ns"], cls=tot_classes(g["tot_ns"], tot_code))
        tmix = mix_core(cm, self.a["core_mix_weights"], j_all)
        feats = core_features(g, cm, float(self.meta.get("ring_fill_ns", 0.0)))
        table = np.zeros(n_clusters)
        for name in CORE_TABLE_FEATURES:
            edges = self.a[f"core_edges_{name}"]
            table += self.a[f"core_table_{name}"][_table_bins(edges, feats[name])]
        return tmix - self._smooth("core", g["log_sumtot"], g["cx"], g["cy"]) - table


def cell_ids(cx, cy, stride):
    return (np.floor(np.nan_to_num(cx) / CELL_PX).astype(np.int64) * stride
            + np.floor(np.nan_to_num(cy) / CELL_PX).astype(np.int64))


def mix_core(cm, weights, fallback):
    """Core-class mean plus fitted weights times (class mean - core mean) for
    the MIX_CLASSES; core mean alone if any of those is missing; `fallback`
    (the all-hit mean) if the cluster has no core hit."""
    core = cm[:, CORE_CLASS]
    diffs = np.stack([cm[:, k] - core for k in MIX_CLASSES], axis=1)
    have_all = np.all(np.isfinite(diffs), axis=1)
    mixed = np.where(have_all, core + np.nan_to_num(diffs) @ np.asarray(weights), core)
    return np.where(np.isfinite(core), mixed, fallback)


def core_features(g, cm, ring_fill):
    ring = cm[:, RING_CLASS] - cm[:, CORE_CLASS]
    return {"maxtot": g["maxtot"], "ring12_minus_core": np.where(np.isfinite(ring), ring, ring_fill),
            "cx": np.nan_to_num(g["cx"]), "cy": np.nan_to_num(g["cy"]), "nh": g["nh"]}


def load_correction(path, mode=None):
    """Load whatever a tab's time-walk file selector points at: a legacy
    timewalk.TimewalkCorrection (.json) or a ClusterTimingCalibration (.npz,
    using estimator `mode` -- see MODES -- or its usual default if None)."""
    if is_calibration_path(path):
        return ClusterTimingCalibration.load(path, mode=mode)
    from timewalk import TimewalkCorrection
    return TimewalkCorrection.load(path)


# =============================================================================
# Fitting (used by fit_cluster_timing.py). Inputs are flat numpy arrays of
# hits from clusters matched one-to-one to a reference time (e-ToF TDC).
# =============================================================================

def _robust_center_width(v):
    q = np.percentile(v, [16, 50, 84])
    return q[1], 0.5 * (q[2] - q[0])


def _grouped(keys, values, n_groups, min_count, fn):
    """fn(values of each key group) for groups with >= min_count entries."""
    order = np.argsort(keys, kind="stable")
    ks, vs = keys[order], values[order]
    bounds = np.searchsorted(ks, np.arange(n_groups + 1))
    out = [None] * n_groups
    for k in range(n_groups):
        a, b = bounds[k], bounds[k + 1]
        if b - a >= min_count:
            out[k] = fn(vs[a:b])
    return out


def fit_walk(d, tot_code, frac, cal_hit, n_codes=1024):
    """1D walk per ToT code and 2D walk per (code, frac bin), from per-hit
    d = hit time - reference time. Returns walk1d (relative to the high-ToT
    end), sigma1d, walk2d, sigma2d."""
    code, key = walk_keys(tot_code, frac, n_codes)
    res = _grouped(code[cal_hit], d[cal_hit], n_codes, 200, _robust_center_width)
    med = np.array([r[0] if r else np.nan for r in res])
    sig = np.array([r[1] if r else np.nan for r in res])
    valid = np.isfinite(med)
    if valid.sum() < 2:
        raise ValueError("Not enough hits per ToT code to fit the walk")
    kk = np.arange(n_codes)
    med_f = np.interp(kk, kk[valid], med[valid])
    sig_f = np.interp(kk, kk[valid], sig[valid])
    top = np.nonzero(valid)[0]
    anchor = float(np.median(med[top[-max(1, len(top) // 10):]]))
    r1 = d - med_f[code]
    nk = (CODE_CAP + 1) * N_FRAC
    res2 = _grouped(key[cal_hit], r1[cal_hit], nk, 100, _robust_center_width)
    walk2d = np.array([r[0] if r else 0.0 for r in res2])
    sig2 = np.array([r[1] if r else np.nan for r in res2])
    sig2 = np.where(np.isfinite(sig2), sig2, np.repeat(sig_f[:CODE_CAP + 1], N_FRAC))
    sig2 = np.maximum(sig2, 0.05)
    return med_f - anchor, sig_f, walk2d.reshape(CODE_CAP + 1, N_FRAC), sig2.reshape(CODE_CAP + 1, N_FRAC), anchor


def detect_column_ticks(resid, chip, local_col, use, tick_ns=25.0, tol_ns=4.0, min_hits=200, half_window=16):
    """Chip-local columns whose median residual sits within tol_ns of a
    non-zero multiple of the 25 ns coarse clock, relative to the median of
    the surrounding columns (+-half_window, so slow drifts across the chip,
    e.g. near the image rim, are not mistaken for ticks). Returns an (n, 3)
    array of (chip, column, ns-to-subtract)."""
    out = []
    for c in np.unique(chip[use]):
        sel = use & (chip == c)
        meds = _grouped(local_col[sel], resid[sel], 256, min_hits, np.median)
        med = np.array([np.nan if m is None else m for m in meds])
        for col in np.nonzero(np.isfinite(med))[0]:
            lo, hi = max(0, col - half_window), min(256, col + half_window + 1)
            ref = np.nanmedian(med[lo:hi])
            k = round((med[col] - ref) / tick_ns)
            if k != 0 and abs((med[col] - ref) - k * tick_ns) < tol_ns:
                out.append((int(c), int(col), k * tick_ns))
    return np.array(out, dtype=np.float64).reshape(-1, 3)


def fit_pixel_terms(cid, n_clusters, t2, w, w_total, pix, cal_hit, n_pix, n_iter=3, ridge=1e3, min_hits=50):
    """Per-pixel offset and walk-amplitude slope, fitted against each
    cluster's own (clipped weighted-mean) time, iterated n_iter times."""
    A = np.zeros(n_pix)
    B = np.zeros(n_pix)
    for _ in range(n_iter):
        t = t2 - (A[pix] + B[pix] * w_total)
        j, keep = clipped_weighted_mean(cid, n_clusters, t, w)
        ri = t - j[cid]
        m = cal_hit & keep & (np.abs(ri) < 40)
        g, xv, yv, wv = pix[m], w_total[m], ri[m], w[m]
        S = np.bincount(g, wv, minlength=n_pix)
        Sx = np.bincount(g, wv * xv, minlength=n_pix)
        Sy = np.bincount(g, wv * yv, minlength=n_pix)
        Sxx = np.bincount(g, wv * xv * xv, minlength=n_pix)
        Sxy = np.bincount(g, wv * xv * yv, minlength=n_pix)
        n = np.bincount(g, minlength=n_pix)
        xm = Sx / np.maximum(S, 1e-30)
        ym = Sy / np.maximum(S, 1e-30)
        b = (Sxy - S * xm * ym) / ((Sxx - S * xm * xm) + ridge)
        a = ym - b * xm
        ok = n >= min_hits
        A += np.where(ok, a, 0.0)
        B += np.where(ok, b, 0.0)
    return A, B


def fit_smooth(T, ref, cal, log_sumtot, cells, nodes, damp=0.3, ycut=20.0, irls=3):
    """Charge curve (piecewise linear in log10 sum ToT, first node fixed at 0)
    plus one offset per detector cell, fitted with Huber reweighting.
    Returns (charge values at nodes, sorted cell ids, cell offsets)."""
    from scipy.sparse import csr_matrix, hstack
    from scipy.sparse.linalg import lsqr
    y = T - ref
    y = y - np.median(y[cal])
    fit = cal & np.isfinite(y) & (np.abs(y) < ycut)
    n = int(fit.sum())
    ls = log_sumtot[fit]
    j = np.clip(np.searchsorted(nodes, ls) - 1, 0, len(nodes) - 2)
    f = np.clip((ls - nodes[j]) / (nodes[j + 1] - nodes[j]), 0, 1)
    rows = np.r_[np.arange(n), np.arange(n)]
    Ac = csr_matrix((np.r_[1 - f, f], (rows, np.r_[j, j + 1])), shape=(n, len(nodes)))[:, 1:]
    uc, inv = np.unique(cells[fit], return_inverse=True)
    Ap = csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, len(uc)))
    A = hstack([Ac, Ap]).tocsr()
    yf = y[fit]
    wt = np.ones(n)
    for _ in range(irls):
        sq = np.sqrt(wt)
        sol = lsqr(csr_matrix(A.multiply(sq[:, None])), yf * sq, damp=damp, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
        r = yf - A @ sol
        s = 1.4826 * np.median(np.abs(r))
        wt = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-9))
    nc = len(nodes) - 1
    return np.r_[0.0, sol[:nc]], uc, sol[nc:]


def fit_mix_weights(cm, ref, smooth_all, train, irls=5):
    """Weights of (class mean - core mean) for MIX_CLASSES that best predict
    the core-mean residual against the reference (Huber reweighted)."""
    core = cm[:, CORE_CLASS]
    Y = core - ref - smooth_all
    X = np.stack([cm[:, k] - core for k in MIX_CLASSES], axis=1)
    ok = train & np.all(np.isfinite(X), axis=1) & np.isfinite(Y)
    Y = Y - np.median(Y[ok])
    ok &= np.abs(Y) < 10
    Xo, Yo = X[ok], Y[ok]
    wt = np.ones(len(Yo))
    a = np.zeros(len(MIX_CLASSES))
    for _ in range(irls):
        sq = np.sqrt(wt)
        a, *_ = np.linalg.lstsq(Xo * sq[:, None], -Yo * sq, rcond=None)
        r = Yo + Xo @ a
        s = 1.4826 * np.median(np.abs(r))
        wt = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-9))
    return a


def fit_core_table(y, feats, train, n_bins=12, sweeps=6, ycut=8.0):
    """Additive per-feature tables (quantile bins, median backfitting) for
    the residual `y` of the mixed core estimator."""
    tr = train & np.isfinite(y) & (np.abs(y) < ycut)
    edges, bidx, tabs = {}, {}, {}
    for name in CORE_TABLE_FEATURES:
        v = feats[name]
        e = np.unique(np.quantile(v[tr], np.linspace(0, 1, n_bins + 1)))
        edges[name] = e
        bidx[name] = _table_bins(e, v)
        tabs[name] = np.zeros(len(e) - 1)
    pred = np.zeros(len(y))
    for _ in range(sweeps):
        for name in CORE_TABLE_FEATURES:
            part = y - (pred - tabs[name][bidx[name]])
            new = np.zeros(len(tabs[name]))
            for b in range(len(new)):
                sel = tr & (bidx[name] == b)
                if np.any(sel):
                    new[b] = np.median(part[sel])
            new -= np.median(new)
            pred += new[bidx[name]] - tabs[name][bidx[name]]
            tabs[name] = new
    return edges, tabs


def fit_calibration(hits, clusters, layout_dict, canvas_size, min_cluster_size, log=print):
    """Fit a full calibration.

    hits: dict of per-hit arrays cid, x, y, chip, local_col, t (ns, relative
        to the pulse), tot_code.
    clusters: dict of per-cluster arrays ref (reference time, same time base
        as t), cal (bool: use for fitting), electron (bool).
    Returns a ClusterTimingCalibration (mode = DEFAULT_MODE)."""
    t0 = time.time()
    cid = hits["cid"]
    n = len(clusters["ref"])
    x, y, tot_code = hits["x"], hits["y"], hits["tot_code"]
    ref, cal, ele = clusters["ref"], clusters["cal"], clusters["electron"]
    cal_hit = cal[cid]
    g = cluster_geometry(cid, n, x, y, tot_code)
    d = hits["t"] - ref[cid]

    # Pass 1 walk, then look for whole-column 25 ns offsets and refit.
    w1, s1, w2, s2, _ = fit_walk(d, tot_code, g["frac"], cal_hit)
    code, key = walk_keys(tot_code, g["frac"], len(w1))
    r1 = d - w1[code] - w2.ravel()[key]
    column_fix = detect_column_ticks(r1, hits["chip"], hits["local_col"], cal_hit & (g["tot_ns"] >= 150))
    t_fixed = hits["t"].copy()
    for chip, col, ns in column_fix:
        t_fixed[(hits["chip"] == chip) & (hits["local_col"] == col)] -= ns
    if len(column_fix):
        log(f"column fix: {len(column_fix)} columns, e.g. chip {int(column_fix[0,0])} col {int(column_fix[0,1])} "
            f"{column_fix[0,2]:+.0f} ns")
        d = t_fixed - ref[cid]
        w1, s1, w2, s2, _ = fit_walk(d, tot_code, g["frac"], cal_hit)
        code, key = walk_keys(tot_code, g["frac"], len(w1))
    log(f"walk fitted ({time.time() - t0:.0f} s)")

    w_total = w1[code] + w2.ravel()[key]
    t2 = t_fixed - w_total
    weight = 1.0 / s2.ravel()[key] ** 2
    pix = x.astype(np.int64) * canvas_size + y.astype(np.int64)
    A, B = fit_pixel_terms(cid, n, t2, weight, w_total, pix, cal_hit, canvas_size * canvas_size)
    log(f"per-pixel terms fitted ({time.time() - t0:.0f} s)")

    arrays = dict(column_fix=column_fix, walk1d_ns=w1, walk1d_sigma_ns=s1, walk2d_ns=w2, walk2d_sigma_ns=s2,
                  pixel_offset_ns=A.reshape(canvas_size, canvas_size), pixel_walk_slope=B.reshape(canvas_size, canvas_size))
    stride = int(np.ceil(canvas_size / CELL_PX))
    meta = dict(format=FILE_FORMAT, created=time.strftime("%Y-%m-%dT%H:%M:%S"), layout=layout_dict,
                canvas_size=canvas_size, cell_stride=stride, min_cluster_size=min_cluster_size,
                default_mode=DEFAULT_MODE, jitter_ns=0.0)
    calib = ClusterTimingCalibration(arrays, meta, mode="all")

    tt, w, _ = calib.hit_times(cid, n, x, y, t_fixed, tot_code, geom=g)
    j_all, keep = clipped_weighted_mean(cid, n, tt, w)
    sb = np.percentile(g["log_sumtot"][cal], np.linspace(0, 100, 21))
    nodes = 0.5 * (sb[1:] + sb[:-1])
    arrays["charge_nodes"] = nodes
    cells = cell_ids(g["cx"], g["cy"], stride)

    def smooth(prefix, T):
        ch, ids, vals = fit_smooth(T, ref, cal, g["log_sumtot"], cells, nodes)
        arrays[f"{prefix}_charge_ns"], arrays[f"{prefix}_cell_ids"], arrays[f"{prefix}_cell_ns"] = ch, ids, vals

    smooth("all", j_all)
    t1, sw = _bin_mean(cid, tt, w * (keep & (g["tot_ns"] >= BRIGHT_TOT_NS)), n)
    smooth("bright", np.where(sw > 0, t1, j_all))
    log(f"smooth terms fitted ({time.time() - t0:.0f} s)")

    cm = class_means(cid, n, tt, w, keep, g["tot_ns"])
    smooth_all = calib._smooth("all", g["log_sumtot"], g["cx"], g["cy"])
    a = fit_mix_weights(cm, ref, smooth_all, cal & ele)
    arrays["core_mix_weights"] = a
    tmix = mix_core(cm, a, j_all)
    smooth("core", tmix)
    ring = cm[:, RING_CLASS] - cm[:, CORE_CLASS]
    ring_fill = float(np.nanmedian(ring[cal & ele]))
    meta["ring_fill_ns"] = ring_fill
    yres = tmix - calib._smooth("core", g["log_sumtot"], g["cx"], g["cy"]) - ref
    yres = yres - np.median(yres[cal])
    edges, tabs = fit_core_table(yres, core_features(g, cm, ring_fill), cal & ele)
    for name in CORE_TABLE_FEATURES:
        arrays[f"core_edges_{name}"] = edges[name]
        arrays[f"core_table_{name}"] = tabs[name]
    log(f"core mixing {dict(zip([int(k) for k in MIX_CLASSES], np.round(a, 3)))}, tables fitted ({time.time() - t0:.0f} s)")
    return ClusterTimingCalibration(arrays, meta, mode=DEFAULT_MODE)
