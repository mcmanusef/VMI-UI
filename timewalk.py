"""Time-walk correction: a per-ToT time offset subtracted from pixel hit
times before clustering.

Time-walk is a hardware effect where a pixel's reported arrival time (t)
shifts systematically with its ToT (time-over-threshold, i.e. signal size).
The correction is generated from a 2D histogram of pixel (ToT, t) pairs: for
a real, narrow-in-time feature (an electron/ion arrival peak), t should not
depend on ToT, so any ToT-dependent drift of that ridge is the walk to
correct out. The curve is anchored to ~0 at the high-ToT end, since high-ToT
hits are the least affected by walk and should stay essentially unchanged.
"""
import json
import pathlib
import time

import numpy as np


class TimewalkCorrection:
    """A per-ToT correction curve. `correct(tot)` returns the time offset
    (ns) to *subtract* from a hit's raw t for the given tot value(s)."""

    def __init__(self, tot_centers, correction_ns, meta=None):
        self.tot_centers = np.asarray(tot_centers, dtype=np.float64)
        self.correction_ns = np.asarray(correction_ns, dtype=np.float64)
        self.meta = dict(meta or {})

    def correct(self, tot):
        tot = np.asarray(tot, dtype=np.float64)
        if self.tot_centers.size == 0:
            return np.zeros_like(tot)
        return np.interp(
            tot, self.tot_centers, self.correction_ns,
            left=self.correction_ns[0], right=self.correction_ns[-1],
        )

    def to_dict(self):
        return {
            "tot_centers": self.tot_centers.tolist(),
            "correction_ns": self.correction_ns.tolist(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["tot_centers"], data["correction_ns"], data.get("meta"))

    def save(self, path):
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


DEFAULT_CORRECTION_PATH = pathlib.Path(__file__).with_name("timewalk_correction.json")


def apply_timewalk_correction(pixels, correction):
    """Subtract correction.correct(tot) from every pixel's time_ns. Returns a
    new list; `pixels` (list of tpx_processing.PixelData) is left untouched."""
    if correction is None or not pixels:
        return pixels
    tots = np.fromiter((p.tot for p in pixels), dtype=np.float64, count=len(pixels))
    offsets = correction.correct(tots)
    return [
        type(pix)(time_ns=pix.time_ns - off, x=pix.x, y=pix.y, tot=pix.tot)
        for pix, off in zip(pixels, offsets)
    ]


def generate_correction(hist2d, tot_edges, t_edges, min_counts=20, anchor_top_frac=0.1):
    """Fit a time-walk correction from an accumulated (tot, t) 2D histogram.

    hist2d: array shaped (n_tot_bins, n_t_bins) of pixel-hit counts.
    tot_edges / t_edges: the histogram's bin edges along each axis.
    min_counts: ToT columns with fewer total hits than this are skipped (too
        noisy to trust) and their ridge value is filled in by interpolation.
    anchor_top_frac: fraction of the populated ToT columns, taken from the
        high-ToT end, whose ridge value is averaged to become the "zero
        correction" anchor -- this is what keeps high-ToT events unchanged.

    Returns a TimewalkCorrection, plus (for display) the raw per-column
    ridge values before anchoring, as `meta["ridge_ns"]`.
    """
    hist2d = np.asarray(hist2d, dtype=np.float64)
    tot_edges = np.asarray(tot_edges, dtype=np.float64)
    t_edges = np.asarray(t_edges, dtype=np.float64)
    n_tot_bins = hist2d.shape[0]
    tot_centers = 0.5 * (tot_edges[:-1] + tot_edges[1:])
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

    ridge = np.full(n_tot_bins, np.nan)
    for i in range(n_tot_bins):
        col = hist2d[i]
        total = col.sum()
        if total < min_counts:
            continue
        # Ridge = the peak (mode) of this ToT column. More robust than a
        # weighted mean when there's a diffuse background away from the
        # real, narrow-in-time feature we're aligning on.
        ridge[i] = float(t_centers[int(np.argmax(col))])

    valid = ~np.isnan(ridge)
    if valid.sum() < 2:
        raise ValueError(
            "Not enough populated ToT columns (need >= 2 with at least "
            f"{min_counts} counts) to fit a time-walk correction. Collect more data "
            "or lower the min-counts threshold."
        )

    # Fill unpopulated columns by interpolating across the valid ones so the
    # correction is defined for every ToT bin.
    ridge_filled = np.interp(tot_centers, tot_centers[valid], ridge[valid])

    valid_indices = np.nonzero(valid)[0]
    n_anchor = max(1, int(round(len(valid_indices) * anchor_top_frac)))
    anchor_indices = valid_indices[-n_anchor:]
    anchor = float(np.mean(ridge[anchor_indices]))

    correction_ns = ridge_filled - anchor
    meta = {
        "generated_at": time.time(),
        "anchor_ns": anchor,
        "min_counts": min_counts,
        "anchor_top_frac": anchor_top_frac,
        "ridge_ns": ridge_filled.tolist(),
    }
    return TimewalkCorrection(tot_centers, correction_ns, meta)
