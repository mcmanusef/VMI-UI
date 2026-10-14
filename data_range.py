"""Where a set of values (usually times) actually lies, found from the data
instead of a fixed range. Headless (no Qt).

Raw times have stray values far from the real distribution: wrapped or
unpaired hits at -1e6 ns or 1e10 ns, for example. A percentile or min/max
range is pulled out by those, so dense_range() grows outward from the
densest part of the data and stops where the density falls away.
"""
import numpy as np


def _shortest_interval(sorted_values, fraction):
    """(lo, hi) of the shortest interval holding `fraction` of the values."""
    n = sorted_values.size
    k = min(n, max(2, int(np.ceil(fraction * n))))
    widths = sorted_values[k - 1:] - sorted_values[:n - k + 1]
    i = int(np.argmin(widths))
    return float(sorted_values[i]), float(sorted_values[i + k - 1])


def dense_range(values, threshold=1e-3, core_fraction=0.5, max_gap_blocks=2):
    """(lo, hi) of the populated part of `values`.

    Starts from the shortest interval holding core_fraction of the values
    and steps outward in blocks of that width. A block counts as populated
    when its density is at least `threshold` times the core's density. The
    range ends at the last populated block before max_gap_blocks empty ones
    in a row, so gaps between peaks are crossed and sparse outliers are not.
    Raises ValueError when there are no finite values."""
    v = np.asarray(values, dtype=np.float64)
    v = np.sort(v[np.isfinite(v)])
    if v.size == 0:
        raise ValueError("No finite values.")
    if v[0] == v[-1]:
        return float(v[0]) - 0.5, float(v[-1]) + 0.5

    lo, hi = _shortest_interval(v, core_fraction)
    width = hi - lo
    if width <= 0:
        # Many identical values: use the typical spacing of the rest instead.
        width = (v[-1] - v[0]) / v.size
    min_count = threshold * core_fraction * v.size

    def count(a, b):
        return np.searchsorted(v, b, side="right") - np.searchsorted(v, a, side="left")

    def grow(edge, step, limit):
        gap = 0
        probe = edge
        while gap < max_gap_blocks and (probe > limit if step < 0 else probe < limit):
            a, b = sorted((probe, probe + step))
            probe += step
            if count(a, b) >= max(min_count, 1):
                edge, gap = probe, 0
            else:
                gap += 1
        return edge

    lo = max(grow(lo, -width, v[0]), float(v[0]))
    hi = min(grow(hi, width, v[-1]), float(v[-1]))
    return lo, hi


def padded(lo, hi, margin=0.05):
    """(lo, hi) widened by `margin` of its width on each side."""
    pad = (hi - lo) * margin
    return lo - pad, hi + pad


# More bins than any plot needs; a range this wide is a typo or outliers.
MAX_BINS = 2_000_000


def bin_count(lo, hi, bin_width, minimum=8):
    """Bins of bin_width covering [lo, hi], or ValueError if that's
    unreasonably many."""
    if hi <= lo or bin_width <= 0:
        raise ValueError("Histogram max must exceed min, and bin width must be positive.")
    n_bins = max(minimum, int(np.ceil((hi - lo) / bin_width)))
    if n_bins > MAX_BINS:
        raise ValueError(
            f"Range {lo:.6g} to {hi:.6g} at bin width {bin_width:.6g} needs {n_bins:,} bins. "
            "Narrow the range or widen the bins."
        )
    return n_bins
