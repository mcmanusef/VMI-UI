"""Fit (and check) a cluster-timing calibration from raw .tpx3 data that has
a per-particle reference time on a TDC line -- normally the e-ToF (MCP
pickoff) on FPGA TDC channel 2, as set on the "tdc mapping" Hardware tab.

    python fit_cluster_timing.py "C:/DATA/20260820/PO (-) e0.5/raw" --out cluster_timing_20260820.npz

Every raw file below the given folder(s) is a candidate; --max-files spreads
the chosen ones evenly over the run. Alternate chosen files are used for
fitting and for validation, so the validation numbers are out-of-sample.

Validation runs the production path itself (decode_tpx3 ->
apply_timewalk_correction -> build_frame_events(timing=...)) and reports,
for electrons whose cluster is matched one-to-one to a reference edge, the
sigma68 / FWHM of (cluster time - reference) for the legacy ToT-weighted
mean and for each cluster-timing mode. The report is also written next to
the .npz as .json.

A calibration only holds for the detector settings and chip layout it was
fitted with. Load the .npz in any tab's time-walk field to use it.
"""
import argparse
import json
import pathlib
import sys
import time

# Import this app's modules from this file's folder whatever the working directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

import chip_layout  # noqa: E402
import cluster_timing  # noqa: E402
import tdc_mapping  # noqa: E402
from tpx_processing import (  # noqa: E402
    build_frame_events,
    classify_tdcs,
    decode_tpx3,
    get_min_cluster_size,
)
from timewalk import apply_timewalk_correction  # noqa: E402

ELECTRON_MAX_NS = 2000.0      # reference - trigger below this: electron; above: ion
MATCH_HALF_WINDOW_NS = 35.0
ISOLATION_NS = 500.0


def find_raw_files(folders):
    files = []
    for folder in folders:
        files.extend(p for p in pathlib.Path(folder).rglob("*.tpx3") if p.is_file())
    return sorted(files, key=lambda p: (p.stat().st_mtime, str(p)))


def pick_files(files, max_files):
    if max_files is None or len(files) <= max_files:
        return files
    idx = np.unique(np.linspace(0, len(files) - 1, max_files).round().astype(int))
    return [files[i] for i in idx]


def reference_times(tdcs, mapping, reference):
    """(reference edge times, trigger times), both sorted, with no dither."""
    etof, itof, pulses = classify_tdcs(tdcs, mapping, jitter_ns=0.0)
    ref = {"etof": etof, "itof": itof}[reference]
    return np.sort(np.asarray(ref, dtype=np.float64)), np.sort(np.asarray(pulses, dtype=np.float64))


def local_lookup(layout):
    """Canvas (x, y) -> chip and chip-local column, for the column fix."""
    size = layout.size
    chip = np.full((size, size), -1, dtype=np.int64)
    col = np.full((size, size), -1, dtype=np.int64)
    for gx in range(size):
        for gy in range(size):
            loc = layout.to_local(gx, gy)
            if loc is not None:
                chip[gx, gy], col[gx, gy] = loc[0], loc[1]
    return chip, col


def cluster_like_production(t, x, y, pulses, size, min_size):
    """Same rule as tpx_processing.cluster_hits: within one trigger period,
    hits on edge-touching pixels (and repeat hits on one pixel) form a
    cluster; clusters with fewer than min_size hits are dropped. Returns
    (hit index kept, cluster id per kept hit, pulse index per cluster, t
    relative to the pulse per kept hit)."""
    pidx = np.searchsorted(pulses, t, side="right") - 1
    ok = pidx >= 0
    hit = np.nonzero(ok)[0]
    pidx = pidx[ok]
    keys = (pidx.astype(np.int64) * size + x[ok]) * size + y[ok]
    ukeys, node = np.unique(keys, return_inverse=True)
    yy = ukeys % size
    xx = (ukeys // size) % size
    edges = []
    for step, valid in ((size, xx + 1 < size), (1, yy + 1 < size)):
        nb = ukeys + step
        pos = np.clip(np.searchsorted(ukeys, nb), 0, len(ukeys) - 1)
        hitmask = valid & (ukeys[pos] == nb)
        edges.append(np.c_[np.nonzero(hitmask)[0], pos[hitmask]])
    e = np.vstack(edges) if edges else np.zeros((0, 2), int)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(ukeys), len(ukeys)))
    _, label = connected_components(g, directed=False)
    hl = label[node]
    counts = np.bincount(hl)
    keep = counts[hl] >= min_size
    _, cid = np.unique(hl[keep], return_inverse=True)
    hit, pidx = hit[keep], pidx[keep]
    cl_pulse = np.zeros(cid.max() + 1 if len(cid) else 0, dtype=np.int64)
    cl_pulse[cid] = pidx
    return hit, cid, cl_pulse, t[hit] - pulses[pidx]


def match_reference(cluster_abs, ref, delay=None):
    """One-to-one match of cluster times to isolated reference edges.
    Returns (matched mask, reference index, delay used)."""
    n = len(cluster_abs)
    if n == 0 or len(ref) < 2:
        return np.zeros(n, bool), np.zeros(n, int), delay
    i = np.clip(np.searchsorted(ref, cluster_abs), 1, len(ref) - 1)
    near = np.where(np.abs(cluster_abs - ref[i - 1]) < np.abs(cluster_abs - ref[i]), i - 1, i)
    if delay is None:
        dd = cluster_abs - ref[near]
        dd = dd[np.abs(dd) < 1000]
        h, edges = np.histogram(dd, bins=np.arange(-1000, 1001, 1.0))
        delay = float(edges[np.argmax(h)] + 0.5)
    i = np.clip(np.searchsorted(ref, cluster_abs - delay), 1, len(ref) - 1)
    j = np.where(np.abs(cluster_abs - delay - ref[i - 1]) < np.abs(cluster_abs - delay - ref[i]), i - 1, i)
    d = cluster_abs - ref[j] - delay
    left = np.where(j > 0, ref[j] - ref[np.maximum(j - 1, 0)], np.inf)
    right = np.where(j < len(ref) - 1, ref[np.minimum(j + 1, len(ref) - 1)] - ref[j], np.inf)
    ok = (np.abs(d) < MATCH_HALF_WINDOW_NS) & (left > ISOLATION_NS) & (right > ISOLATION_NS)
    used = np.bincount(j[ok], minlength=len(ref))
    ok &= used[j] == 1
    return ok, j, delay


def load_fit_file(path, layout, chip_map, col_map, mapping, reference, min_size, delay):
    pixels, tdcs, _, _ = decode_tpx3(str(path), max_packets=None, layout=layout)
    ref, pulses = reference_times(tdcs, mapping, reference)
    if len(pixels) == 0 or len(pulses) == 0 or len(ref) < 2:
        return None, delay
    t = pixels.time_ns
    x = pixels.x.astype(np.int64)
    y = pixels.y.astype(np.int64)
    tot = pixels.tot.astype(np.int64)
    del pixels
    hit, cid, cl_pulse, t_rel = cluster_like_production(t, x, y, pulses, layout.size, min_size)
    first = np.full(cl_pulse.shape[0], np.inf)
    np.minimum.at(first, cid, t_rel)
    ok, j, delay = match_reference(pulses[cl_pulse] + first, ref, delay)
    keep_hit = ok[cid]
    new_id = np.cumsum(ok) - 1
    xs, ys = x[hit][keep_hit], y[hit][keep_hit]
    hits = dict(cid=new_id[cid[keep_hit]], x=xs, y=ys, chip=chip_map[xs, ys], local_col=col_map[xs, ys],
                t=t_rel[keep_hit], tot_code=tot[hit][keep_hit])
    ref_rel = ref[j[ok]] - pulses[cl_pulse[ok]]
    return (hits, ref_rel), delay


def sigma68(v):
    q = np.percentile(v - np.median(v), [16, 84])
    return 0.5 * (q[1] - q[0])


def fwhm(v, bin_ns=0.26042):
    v = v - np.median(v)
    h, e = np.histogram(v, bins=np.arange(-30, 30, bin_ns))
    c = 0.5 * (e[1:] + e[:-1])
    above = c[h >= h.max() / 2]
    return float(above.max() - above.min() + bin_ns)


def validate(files, calib, layout, mapping, reference, min_size, delay, log=print, stop_event=None):
    """Run the production clustering path with `calib` and compare every
    mode's cluster times (and the legacy ToT-weighted mean) to the reference."""
    captured = []
    original = calib.cluster_times

    def recording(cid, n, x, y, t, tot, mode=None):
        captured.append((cid, n, x, y, t, tot))
        return original(cid, n, x, y, t, tot, mode=mode)

    calib.cluster_times = recording
    fix = calib.column_fix_map(layout)
    res = {m: [] for m in ("legacy",) + cluster_timing.MODES}
    try:
        for k, path in enumerate(files, 1):
            _check_cancel(stop_event)
            pixels, tdcs, _, _ = decode_tpx3(str(path), max_packets=None, layout=layout)
            ref, pulses = reference_times(tdcs, mapping, reference)
            if len(pixels) == 0 or len(pulses) == 0:
                continue
            pixels = apply_timewalk_correction(pixels, calib)
            captured.clear()
            events, _, _ = build_frame_events(pixels, tdcs, mapping, min_size=min_size, timing=calib)
            if not captured or len(events.cluster_t) == 0:
                continue
            # cluster_times is called with clusters in the same order as
            # events' cluster rows.
            cid, n, x, y, t, tot = captured[0]
            pulse_t = events.pulses[events.cluster_pulse]
            times = {m: original(cid, n, x, y, t, tot, mode=m) for m in cluster_timing.MODES}
            t_raw = t + fix[x.astype(int), y.astype(int)]
            w = np.where(tot > 0, tot, 1.0)
            times["legacy"] = np.bincount(cid, w * t_raw, minlength=n) / np.bincount(cid, w, minlength=n)
            ok, j, _ = match_reference(pulse_t + times["bright"], ref)
            ele = ok & (ref[j] - pulse_t < ELECTRON_MAX_NS)
            for m in res:
                res[m].append(pulse_t[ele] + times[m][ele] - ref[j[ele]])
            log(f"  validated {k}/{len(files)}: {path.name}, {int(ele.sum())} matched electrons")
    finally:
        calib.cluster_times = original
    report = {}
    for m, parts in res.items():
        if parts:
            v = np.concatenate(parts)
            report[m] = dict(n=int(len(v)), sigma68_ns=round(float(sigma68(v)), 3), fwhm_ns=round(fwhm(v), 2),
                             frac_beyond_10ns=round(float(np.mean(np.abs(v - np.median(v)) > 10)), 4))
    return report


class FitError(Exception):
    """The fit can't run on this data (no files, nothing matched, ...)."""


class FitCancelled(Exception):
    """The fit was stopped through its stop_event."""


def _check_cancel(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise FitCancelled("Fit cancelled.")


TDC_MAPPING_CHOICES = ("active", "default")


def resolve_mapping(name):
    """"active" (the Hardware tab's current mapping), "default" (the built-in
    FPGA-TDC scheme), or a path to a saved TdcMapping .json."""
    if name == "active":
        return tdc_mapping.get_active_mapping()
    if name == "default":
        return tdc_mapping.default_mapping()
    return tdc_mapping.TdcMapping.load(name)


def run_fit(folders, out, max_files=24, reference="etof", mapping="active", do_validate=True,
            log=print, stop_event=None):
    """Fit a calibration from the raw files under `folders`, save it to
    `out` (.npz) and, unless do_validate is False, validate it on the
    alternate files and save the report next to it (.json). Returns
    (calibration, validation report or None). Raises FitError if the data
    can't be fitted and FitCancelled if `stop_event` gets set (checked
    between files and between fit stages)."""
    t0 = time.time()
    layout = chip_layout.get_active_layout()
    mapping = resolve_mapping(mapping) if isinstance(mapping, str) else mapping
    min_size = get_min_cluster_size()
    files = pick_files(find_raw_files(folders), max_files)
    if not files:
        raise FitError("No .tpx3 files found.")
    fit_files = files if not do_validate else files[0::2]
    val_files = files[1::2] if do_validate else []
    log(f"{len(files)} files: {len(fit_files)} to fit, {len(val_files)} to validate; "
        f"layout {layout.width}x{layout.height}")

    chip_map, col_map = local_lookup(layout)
    parts, refs, delay, offset = [], [], None, 0
    for k, path in enumerate(fit_files, 1):
        _check_cancel(stop_event)
        loaded, delay = load_fit_file(path, layout, chip_map, col_map, mapping, reference, min_size, delay)
        if loaded is None:
            log(f"  skipped {path.name}: no pixels, triggers or reference edges")
            continue
        hits, ref_rel = loaded
        hits["cid"] = hits["cid"] + offset
        offset += len(ref_rel)
        parts.append(hits)
        refs.append(ref_rel)
        log(f"  loaded {k}/{len(fit_files)}: {path.name}, {len(ref_rel)} matched clusters "
            f"(reference delay {delay:+.1f} ns, {time.time() - t0:.0f} s)")
    if not parts:
        raise FitError("No clusters matched to the reference; check the tdc mapping "
                       "(older FPGA-TDC data needs the default mapping).")
    hits = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    ref_rel = np.concatenate(refs)
    clusters = dict(ref=ref_rel, cal=np.ones(len(ref_rel), bool), electron=ref_rel < ELECTRON_MAX_NS)
    log(f"fitting on {len(ref_rel)} clusters ({int(clusters['electron'].sum())} electrons), {len(hits['t'])} hits")
    _check_cancel(stop_event)
    calib = cluster_timing.fit_calibration(hits, clusters, layout.to_dict(), layout.size, min_size, log=log)
    del hits
    calib.meta.update(dict(reference=reference, reference_delay_ns=delay, tdc_mapping=mapping.to_dict(),
                           fit_files=[str(p) for p in fit_files], n_fit_clusters=int(len(ref_rel))))
    _check_cancel(stop_event)
    out = pathlib.Path(out)
    calib.save(out)
    log(f"saved {out} ({time.time() - t0:.0f} s)")

    report = None
    if val_files:
        report = validate(val_files, calib, layout, mapping, reference, min_size, delay, log=log,
                          stop_event=stop_event)
        calib.meta["validation"] = report
        calib.save(out)
        out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        log("validation (electrons, cluster time - reference):")
        for line in format_report(report):
            log("  " + line)
    return calib, report


def format_report(report):
    """One line per estimator of a validation report."""
    return [f"{m:7s} n={r['n']:7d}  sigma68 {r['sigma68_ns']:.3f} ns  FWHM {r['fwhm_ns']:.2f} ns  "
            f">10 ns {r['frac_beyond_10ns']:.4f}" for m, r in report.items()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="raw data folder(s); searched recursively for .tpx3")
    ap.add_argument("--out", required=True, help="output calibration (.npz)")
    ap.add_argument("--max-files", type=int, default=24, help="files to use, spread over the run (default 24)")
    ap.add_argument("--reference", choices=("etof", "itof"), default="etof",
                    help="TDC signal (per the active tdc mapping) used as the per-particle reference")
    ap.add_argument("--tdc-mapping", default="active",
                    help='"active" (the Hardware tab\'s current mapping, default), "default" (the built-in '
                         'FPGA-TDC scheme: trigger = ch1 > 300 ns, i-ToF = ch1 < 300 ns, e-ToF = ch2), '
                         'or a saved TdcMapping .json -- older FPGA-TDC data needs "default"')
    ap.add_argument("--no-validate", action="store_true", help="fit on all chosen files and skip validation")
    args = ap.parse_args(argv)
    try:
        run_fit(args.folders, args.out, max_files=args.max_files, reference=args.reference,
                mapping=args.tdc_mapping, do_validate=not args.no_validate)
    except FitError as exc:
        sys.exit(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
