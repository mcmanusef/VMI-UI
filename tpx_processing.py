"""Pure TPX3 decoding / clustering / histogram-building logic.

No Tkinter or matplotlib dependencies here so this module can be shared by
any UI (diagnostics, monitored acquisition, offline reprocessing, ...).
"""
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np

import app_settings
import chip_layout
from tdc_mapping import (
    FAMILY_PICOTDC,
    FPGA_TDC_NS_PER_COUNT,
    FPGA_TDC_SUBTYPES,
    PICOTDC_NS_PER_COUNT,
)


PIXEL_RES_NS = 25 / 16
# Pixel ToA (16-bit coarse + 14-bit mid, in 1.5625 ns steps) wraps every
# 2^34 * 1.5625 ns = 2^30 * 25 ns (~26.8 s); TDC times are folded onto the
# same period so pulses and pixels line up.
TIME_ROLLOVER_NS = (1 << 34) * PIXEL_RES_NS

# Dither widths used to smear out the quantization of the raw timestamps.
TDC_JITTER_NS = 0.26          # applied to every e-ToF and i-ToF time
CLUSTER_T_JITTER_NS = 1.56    # applied to every cluster time

# Clusters with fewer hits than this are dropped. User-editable (the
# "Min cluster size" field, shared across tabs) and saved under this key.
DEFAULT_MIN_CLUSTER_SIZE = 5
MIN_CLUSTER_SIZE_KEY = "clustering.min_size"
# The cv4 root attribute recording which minimum a file's clusters used.
MIN_CLUSTER_SIZE_ATTR = "Min Cluster Size (hits)"


# Insensitive pixels between neighbouring chips of a multi-chip detector.
# Clustering treats two hits in the same row (or column) on facing chip
# edges as touching when at most this many pixels separate them, so a
# cluster that falls across a chip boundary stays one cluster -- whether
# the layout leaves that gap empty on the canvas or abuts the tiles.
CHIP_GAP_PX = 2


def get_min_cluster_size() -> int:
    """The saved minimum cluster size (hits), or the default if it's unset
    or not a whole number >= 1."""
    try:
        value = int(str(app_settings.get(MIN_CLUSTER_SIZE_KEY, DEFAULT_MIN_CLUSTER_SIZE)).strip())
    except ValueError:
        return DEFAULT_MIN_CLUSTER_SIZE
    return value if value >= 1 else DEFAULT_MIN_CLUSTER_SIZE


# ---- decoding ---------------------------------------------------------------
#
# Everything below works on whole-file numpy arrays: a 10 s frame can hold
# ~10M pixel hits at ~60k pulses/s, far too many to touch one at a time in
# Python and still keep up with acquisition.


@dataclass
class PixelHits:
    """One raw file's pixel hits, in file order: absolute ToA (ns, folded on
    TIME_ROLLOVER_NS), detector-canvas x/y and ToT."""
    time_ns: np.ndarray
    x: np.ndarray
    y: np.ndarray
    tot: np.ndarray

    def __len__(self):
        return len(self.time_ns)

    def with_times(self, time_ns):
        return PixelHits(time_ns=time_ns, x=self.x, y=self.y, tot=self.tot)


@dataclass
class TdcHits:
    """One raw file's TDC hits, in file order. `is_pico` marks PicoTDC hits
    (sub-nibble 0xC); the rest are FPGA TDC hits. edge: 1 = rising,
    0 = falling."""
    is_pico: np.ndarray
    channel: np.ndarray
    edge: np.ndarray
    time_ns: np.ndarray

    def __len__(self):
        return len(self.time_ns)


def read_packets(path: str, max_packets: int = None):
    """(chips, packets, chunk_ends, total) for one raw file: every packet as
    one uint64 array, the chip stream each came from (the chunk header's
    byte 4, 0-3 on a quad), the cumulative packet count at the end of each
    chunk, and the file's total packet count. With max_packets, only the
    first max_packets packets are returned (total still counts them all)."""
    with open(path, "rb") as f:
        raw = f.read()
    size = len(raw)
    offsets = []
    counts = []
    chips = []
    off = 0
    while off + 8 <= size:
        payload_len = int.from_bytes(raw[off + 6:off + 8], "little")
        chip = raw[off + 4]
        off += 8
        if payload_len <= 0:
            continue
        if off + payload_len > size:
            break
        offsets.append(off)
        counts.append(payload_len // 8)
        chips.append(chip)
        off += payload_len

    if not counts:
        empty = np.zeros(0, dtype=np.uint64)
        return np.zeros(0, dtype=np.uint8), empty, np.zeros(0, dtype=np.int64), 0
    chunk_ends = np.cumsum(counts)
    total = int(chunk_ends[-1])
    # Only unpack the chunks the first max_packets packets come from.
    n_chunks = len(counts)
    if max_packets is not None and max_packets < total:
        n_chunks = int(np.searchsorted(chunk_ends, max_packets)) + 1
    words = np.concatenate([
        np.frombuffer(raw, dtype="<u8", count=n, offset=o)
        for o, n in zip(offsets[:n_chunks], counts[:n_chunks])
    ])
    packets = words - np.uint64(1 << 62)  # wraps mod 2^64, as the hardware intends
    chunk_chips = np.repeat(np.asarray(chips[:n_chunks], dtype=np.uint8), counts[:n_chunks])
    if max_packets is not None and max_packets < total:
        packets = packets[:max_packets]
        chunk_chips = chunk_chips[:max_packets]
    return chunk_chips, packets, chunk_ends, total


def detect_headers(packets: np.ndarray, chunk_ends: np.ndarray, sample_packets: int = 4096) -> Tuple[int, int]:
    """(pixel_header, tdc_header) nibbles, guessed from the first whole
    chunks holding at least `sample_packets` packets."""
    if len(chunk_ends):
        k = int(np.searchsorted(chunk_ends, sample_packets))
        sample = packets[: int(chunk_ends[min(k, len(chunk_ends) - 1)])]
    else:
        sample = packets
    counts = np.bincount((sample >> np.uint64(60)).astype(np.int64), minlength=16)

    headers_present = {idx: int(cnt) for idx, cnt in enumerate(counts) if cnt}
    if 0x7 in headers_present and 0x2 in headers_present:
        return 0x7, 0x2
    if 0x2 in headers_present and 0x1 in headers_present:
        return 0x2, 0x1
    if headers_present:
        ordered = sorted(headers_present.items(), key=lambda kv: kv[1], reverse=True)
        if len(ordered) >= 2:
            return ordered[0][0], ordered[1][0]
        return ordered[0][0], 0x2
    return 0x7, 0x2


def decode_tpx3(path: str, max_packets: int = None, layout=None):
    """(PixelHits, TdcHits, processed_packets, total_packets) for one raw
    file. Pixel x/y are detector-canvas coordinates: each chip's local x/y
    placed and oriented per `layout` (default chip_layout.get_active_layout())."""
    if layout is None:
        layout = chip_layout.get_active_layout()
    chips, packets, chunk_ends, total = read_packets(path, max_packets)
    pixel_header, tdc_header = detect_headers(packets, chunk_ends)

    headers = (packets >> np.uint64(60)).astype(np.uint8)
    pixel_mask = headers == pixel_header

    # Pixel word fields, read from its two 32-bit halves (half the memory
    # traffic of 64-bit ops): c_time [0:16], f_time [16:20], tot [20:30],
    # m_time [30:44], pix_addr [44:60] (bit 60 belongs to the header).
    words = packets[pixel_mask]
    lo = words.astype(np.uint32)
    hi = (words >> np.uint64(32)).astype(np.uint32)
    del words
    c_time = (lo & 0xFFFF).astype(np.int64)
    f_time = ((lo >> 16) & 0xF).astype(np.int64)
    tot = ((lo >> 20) & 0x3FF).astype(np.int32)
    m_time = ((lo >> 30) | ((hi & 0xFFF) << 2)).astype(np.int64)
    pix_addr = ((hi >> 12) & 0xFFFF).astype(np.int32)
    del lo, hi

    toa_ns = (c_time * (1 << 18) + m_time * (1 << 4) - f_time) * PIXEL_RES_NS
    del c_time, f_time, m_time
    pix = pix_addr & 0x0007
    xs = ((pix_addr & 0xFE00) >> 8) + (pix >> 2)
    ys = ((pix_addr & 0x01F8) >> 1) + (pix & 0x3)
    del pix_addr, pix
    pixel_chips = chips[pixel_mask]
    present = np.nonzero(np.bincount(pixel_chips, minlength=256))[0]
    if len(present) == 1:
        xs, ys = layout.to_global(int(present[0]), xs, ys)
    elif len(present) > 1:
        xs, ys = chip_layout.apply_layout(layout, pixel_chips, xs, ys)
    pixels = PixelHits(time_ns=toa_ns, x=np.asarray(xs, dtype=np.int32), y=np.asarray(ys, dtype=np.int32), tot=tot)

    tdc_words = packets[headers == tdc_header] & np.uint64(0x0FFF_FFFF_FFFF_FFFF)
    tdcs = _decode_tdc_words(_drop_repeated_tdc_words(tdc_words))
    return pixels, tdcs, len(packets), total


def _drop_repeated_tdc_words(words: np.ndarray) -> np.ndarray:
    """Every chip stream in a multi-chip file carries its own bit-identical
    copy of each TDC packet (4x on a quad), so keep only the first
    occurrence of each word, still in file order. A genuine hit can't
    repeat a word exactly: its timestamp (plus, for the FPGA TDC, its
    trigger counter) differs from every other hit's, so this only drops
    the per-chip copies -- and works however many chips carry them."""
    if len(words) == 0:
        return words
    _, first = np.unique(words, return_index=True)
    return words[np.sort(first)]


def _decode_tdc_words(words: np.ndarray) -> TdcHits:
    """TdcHits for each valid TDC word, in the order given. Each
    family/subtype is decoded separately, then the hits are put back in
    file order: classify_tdcs pairs each rising edge with the next falling
    edge on its line, which needs them interleaved as they arrived."""
    sub = ((words >> np.uint64(56)) & np.uint64(0xF)).astype(np.int64)
    order = np.arange(len(words))
    parts = []  # (order, is_pico, channel, edge, time_ns)

    pico_sel = sub == 0xC
    pico = words[pico_sel]
    pico_order = order[pico_sel]
    valid = ((pico >> np.uint64(52)) & np.uint64(1)) == 0
    pico = pico[valid]
    pico_order = pico_order[valid]
    if len(pico):
        channel = ((((pico >> np.uint64(53)) & np.uint64(0x3)) << np.uint64(4))
                   | ((pico >> np.uint64(48)) & np.uint64(0xF))).astype(np.int64)
        edge = ((pico >> np.uint64(47)) & np.uint64(1)).astype(np.int64)
        ts_lower = (pico >> np.uint64(21)) & np.uint64(0x3FFFFFF)
        spidr = pico & np.uint64(0x1FFFFF)
        ts = (spidr << np.uint64(26)) | ts_lower
        time_ns = np.mod(ts.astype(np.float64) * PICOTDC_NS_PER_COUNT, TIME_ROLLOVER_NS)
        parts.append((pico_order, np.ones(len(pico), dtype=bool), channel, edge, time_ns))

    for sub_value, (channel_id, edge_value) in FPGA_TDC_SUBTYPES.items():
        sel = sub == sub_value
        fw = words[sel]
        fw_order = order[sel]
        fine = ((fw >> np.uint64(5)) & np.uint64(0xF)).astype(np.int64)
        ok = (fine > 0) & (fine <= 12)
        fw = fw[ok]
        fine = fine[ok]
        fw_order = fw_order[ok]
        if len(fw) == 0:
            continue
        # Keep only the low 33 bits of the 35-bit coarse counter
        # (3.125 ns steps) so it wraps with the pixel ToA.
        coarse = ((fw >> np.uint64(9)) & np.uint64(0x1FFFFFFFF)).astype(np.int64)
        combined = coarse * 12 + (fine - 1)
        time_ns = combined.astype(np.float64) * FPGA_TDC_NS_PER_COUNT
        n = len(fw)
        parts.append((
            fw_order, np.zeros(n, dtype=bool), np.full(n, channel_id, dtype=np.int64),
            np.full(n, edge_value, dtype=np.int64), time_ns,
        ))

    if not parts:
        return TdcHits(np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64),
                       np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))
    cols = [np.concatenate(c) for c in zip(*parts)]
    file_order = np.argsort(cols[0], kind="stable")
    return TdcHits(*(c[file_order] for c in cols[1:]))


def apply_jitter(values, width):
    """Add a uniform random offset in [0, width) to every value."""
    arr = np.asarray(values, dtype=np.float64)
    if width <= 0 or len(arr) == 0:
        return arr
    return arr + np.random.random(arr.shape) * width


def _ringing_mask(edge, t, holdoff_ns):
    """True for each edge on one TDC line (in file order) that comes less
    than `holdoff_ns` after the line's previous falling edge -- i.e. the
    signal re-crossing threshold as the pulse before it rings down. A
    chain of rings is caught link by link, each against the ring before.
    A line that only records rising edges is checked against the previous
    rising edge instead. No holdoff -> nothing flagged."""
    flagged = np.zeros(len(t), dtype=bool)
    if not holdoff_ns or holdoff_ns <= 0 or len(t) < 2:
        return flagged
    ref_edge = 0 if np.any(edge == 0) else 1
    # Index of the latest reference edge strictly before each edge (-1 = none).
    ref_idx = np.where(edge == ref_edge, np.arange(len(t)), -1)
    prev = np.maximum.accumulate(ref_idx)[:-1]
    has_prev = prev >= 0
    gap = t[1:] - t[np.maximum(prev, 0)]
    # gap < 0 only across a timestamp rollover; never call that ringing.
    flagged[1:] = has_prev & (gap >= 0) & (gap < holdoff_ns)
    return flagged


def classify_tdcs(tdcs: TdcHits, mapping, jitter_ns: float = TDC_JITTER_NS):
    """Splits TdcHits into (etof, itof, pulses) time arrays per `mapping`
    (a tdc_mapping.TdcMapping); pulses come back sorted.

    Each signal's physical TDC line (family + channel) is matched; a signal
    with no pulse-width gate takes every hit on its line (matching its
    time_edge) directly, while a gated signal pairs each rising edge with
    the next falling edge on that same line and keeps the pairing's start
    (or end) time only when the resulting width falls in
    [gate_min_ns, gate_max_ns] -- so multiple signals can share one
    physical line and be told apart purely by pulse length (e.g. a long
    trigger pulse vs a short i-ToF pulse on the same FPGA TDC channel).

    A signal with a ring_holdoff_ns drops every pulse (or, ungated, every
    edge) that starts within that long after the previous pulse on its
    line ended -- see _ringing_mask. This runs before the width gate."""

    def resolve(signal):
        if not signal.enabled:
            return np.zeros(0, dtype=np.float64)
        on_line = (tdcs.is_pico == (signal.source.family == FAMILY_PICOTDC)) & (
            tdcs.channel == signal.source.channel
        )
        edge = tdcs.edge[on_line]
        t = tdcs.time_ns[on_line]
        ringing = _ringing_mask(edge, t, signal.ring_holdoff_ns)
        if signal.gate_min_ns is None and signal.gate_max_ns is None:
            want_edge = 1 if signal.time_edge == "rising" else 0
            return t[(edge == want_edge) & ~ringing]
        # A falling edge closes a pulse exactly when the hit before it on
        # the line is a rising edge (a later rising edge restarts the pulse;
        # a falling edge with no open pulse is ignored).
        starts = np.nonzero((edge[:-1] == 1) & (edge[1:] == 0))[0]
        start_t = t[starts]
        end_t = t[starts + 1]
        width = end_t - start_t
        lo = signal.gate_min_ns if signal.gate_min_ns is not None else float("-inf")
        hi = signal.gate_max_ns if signal.gate_max_ns is not None else float("inf")
        keep = (width >= lo) & (width <= hi) & ~ringing[starts]
        return start_t[keep] if signal.time_edge == "rising" else end_t[keep]

    pulses = np.sort(resolve(mapping.pulse_trigger))
    etof = apply_jitter(resolve(mapping.etof), jitter_ns)
    itof = apply_jitter(resolve(mapping.itof), jitter_ns)
    return etof, itof, pulses


def assign_to_pulses(times, pulses_sorted):
    """Index of the latest pulse at or before each time (-1 if none)."""
    return np.searchsorted(pulses_sorted, times, side="right") - 1


# ---- per-frame events ---------------------------------------------------------


@dataclass
class FrameEvents:
    """One raw file's events, everything indexed by `pulses` (sorted
    absolute trigger times). Cluster/e-ToF/i-ToF rows are ordered by pulse
    index; *_t are times relative to their pulse. pixel_* are the hits that
    fell after some pulse (what the clusters were built from)."""
    pulses: np.ndarray
    cluster_pulse: np.ndarray
    cluster_x: np.ndarray
    cluster_y: np.ndarray
    cluster_t: np.ndarray
    cluster_tot: np.ndarray
    cluster_n: np.ndarray
    etof_pulse: np.ndarray
    etof_t: np.ndarray
    itof_pulse: np.ndarray
    itof_t: np.ndarray
    pixel_pulse: np.ndarray
    pixel_x: np.ndarray
    pixel_y: np.ndarray
    pixel_t: np.ndarray
    pixel_tot: np.ndarray


def _relative_to_pulses(times, pulses_sorted):
    """(pulse index, time - pulse time) for every time after some pulse,
    stably sorted by pulse index."""
    idx = assign_to_pulses(times, pulses_sorted)
    ok = idx >= 0
    idx = idx[ok]
    order = np.argsort(idx, kind="stable")
    idx = idx[order]
    return idx, np.asarray(times)[ok][order] - pulses_sorted[idx]


def build_frame_events(pixels: PixelHits, tdcs: TdcHits, mapping, min_size: int = None,
                       t_jitter_ns: float = None, timing=None):
    """Sorts one file's decoded hits into pulses and clusters its pixels
    (cluster times per `timing` / `t_jitter_ns`, see cluster_hits).
    Returns (FrameEvents, cluster_size_sum, cluster_count)."""
    etof, itof, pulses = classify_tdcs(tdcs, mapping)
    pidx = assign_to_pulses(pixels.time_ns, pulses)
    ok = pidx >= 0
    pidx = pidx[ok]
    px = pixels.x[ok]
    py = pixels.y[ok]
    pt = pixels.time_ns[ok] - pulses[pidx]
    ptot = pixels.tot[ok]

    clusters = cluster_hits(pidx, px, py, pt, ptot, min_size=min_size, t_jitter_ns=t_jitter_ns, timing=timing)
    etof_pulse, etof_t = _relative_to_pulses(etof, pulses)
    itof_pulse, itof_t = _relative_to_pulses(itof, pulses)
    events = FrameEvents(
        pulses=pulses,
        cluster_pulse=clusters["pulse"],
        cluster_x=clusters["x"],
        cluster_y=clusters["y"],
        cluster_t=clusters["t"],
        cluster_tot=clusters["tot"],
        cluster_n=clusters["n"],
        etof_pulse=etof_pulse,
        etof_t=etof_t,
        itof_pulse=itof_pulse,
        itof_t=itof_t,
        pixel_pulse=pidx,
        pixel_x=px,
        pixel_y=py,
        pixel_t=pt,
        pixel_tot=ptot,
    )
    return events, int(clusters["n"].sum()), len(clusters["n"])


def seam_pairs(nodes, span, lo_x, lo_y, layout=None, gap=CHIP_GAP_PX):
    """Extra cluster-graph edges across the gaps between chips.

    `nodes` are sorted unique keys (pulse * span + x - lo_x) * span + y - lo_y
    of canvas pixels. Returns (src, dst) node indices linking every pixel to
    the pixel 2..gap+1 steps along +x or +y from it when the two sit on
    different chips -- i.e. facing chip edges, bridged across the gap. (A
    1-step neighbour is an ordinary edge the caller already links.)"""
    empty = np.zeros(0, dtype=np.int64)
    if layout is None:
        layout = chip_layout.get_active_layout()
    if gap < 1 or len(layout.chips) < 2 or len(nodes) == 0:
        return empty, empty
    chip_map = chip_layout.chip_index_map(layout)
    w, h = chip_map.shape
    reach = gap + 1
    # Only pixels within `reach` of a chip edge can link across a gap.
    near = np.zeros((w, h), dtype=bool)
    for d in range(2, reach + 1):
        near[:-d, :] |= (chip_map[:-d, :] >= 0) & (chip_map[d:, :] >= 0) & (chip_map[:-d, :] != chip_map[d:, :])
        near[:, :-d] |= (chip_map[:, :-d] >= 0) & (chip_map[:, d:] >= 0) & (chip_map[:, :-d] != chip_map[:, d:])
    ny = nodes % span + lo_y
    nx = (nodes // span) % span + lo_x
    inside = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
    cand = np.nonzero(inside)[0]
    cand = cand[near[nx[cand], ny[cand]]]
    if len(cand) == 0:
        return empty, empty
    cx, cy, ckey = nx[cand], ny[cand], nodes[cand]
    c_chip = chip_map[cx, cy]
    srcs, dsts = [], []
    for d in range(2, reach + 1):
        for step, tx, ty in ((d * span, cx + d, cy), (d, cx, cy + d)):
            # Stay inside the canvas and the key's x/y field, so a target
            # key can't alias a pixel of another row or pulse.
            ok = (tx < w) & (ty < h) & (tx - lo_x < span) & (ty - lo_y < span)
            ok[ok] &= (chip_map[tx[ok], ty[ok]] >= 0) & (chip_map[tx[ok], ty[ok]] != c_chip[ok])
            sel = np.nonzero(ok)[0]
            if len(sel) == 0:
                continue
            target = ckey[sel] + step
            pos = np.minimum(np.searchsorted(nodes, target), len(nodes) - 1)
            match = nodes[pos] == target
            srcs.append(cand[sel[match]])
            dsts.append(pos[match])
    if not srcs:
        return empty, empty
    return np.concatenate(srcs), np.concatenate(dsts)


def cluster_hits(pulse_idx, x, y, t_rel, tot, min_size: int = None, t_jitter_ns: float = None, timing=None,
                 layout=None):
    """Groups each pulse's pixel hits into clusters of touching pixels
    (4-neighbour connected within the same pulse; repeat hits on one pixel
    all count; pixels on facing edges of neighbouring chips touch across
    the CHIP_GAP_PX gap between them, see seam_pairs, for `layout`, default
    the active one) and keeps those with at least `min_size` hits (default
    get_min_cluster_size()). Positions and ToT are ToT-weighted means
    (plain means if a cluster's ToT sums to zero).

    Cluster time: by default the ToT-weighted mean hit time plus a uniform
    [0, t_jitter_ns) dither (default CLUSTER_T_JITTER_NS). With `timing` (a
    cluster_timing.ClusterTimingCalibration) it is that calibration's
    corrected cluster time instead, dithered by t_jitter_ns if given,
    otherwise by timing.jitter_ns (0 by default); a cluster the
    calibration gives no finite time keeps its undithered weighted mean.

    Returns a dict of per-cluster arrays (pulse, x, y, t, tot, n), ordered
    by pulse."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if min_size is None:
        min_size = get_min_cluster_size()
    if t_jitter_ns is None:
        t_jitter_ns = timing.jitter_ns if timing is not None else CLUSTER_T_JITTER_NS
    empty_f = np.zeros(0, dtype=np.float64)
    empty_i = np.zeros(0, dtype=np.int64)
    if len(pulse_idx) == 0:
        return {"pulse": empty_i, "x": empty_f, "y": empty_f, "t": empty_f, "tot": empty_f, "n": empty_i}

    x = np.asarray(x)
    y = np.asarray(y)
    pulse_idx = np.asarray(pulse_idx, dtype=np.int64)
    # One integer key per (pulse, x, y); `span` leaves room for x+1 / y+1
    # (and the seam_pairs reach across a chip gap) without spilling into
    # the next row or pulse.
    lo_x, lo_y = int(x.min()), int(y.min())
    span = int(max(int(x.max()) - lo_x, int(y.max()) - lo_y)) + CHIP_GAP_PX + 2
    key = (pulse_idx * span + (x - lo_x)) * span + (y - lo_y)

    # Unique (pulse, pixel) nodes, and each hit's node. A stable (timsort)
    # argsort, since the keys are already partly ordered by pulse.
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    del key
    is_first = np.empty(len(sorted_key), dtype=bool)
    is_first[0] = True
    np.not_equal(sorted_key[1:], sorted_key[:-1], out=is_first[1:])
    nodes = sorted_key[is_first]
    del sorted_key
    hit_node = np.empty(len(order), dtype=np.int64)
    hit_node[order] = np.cumsum(is_first) - 1
    del order, is_first
    n_nodes = len(nodes)

    # y+1 neighbours are simply the next node in key order.
    y_src = np.nonzero(nodes[1:] == nodes[:-1] + 1)[0]
    # x+1 neighbours: merge the sorted keys with the sorted keys + span (a
    # stable sort of two sorted runs is a linear merge); an equal pair is
    # node j (first) next to node i's x+1 key.
    merged = np.concatenate([nodes, nodes + span])
    merge_order = np.argsort(merged, kind="stable")
    merged = merged[merge_order]
    pair = np.nonzero(merged[1:] == merged[:-1])[0]
    del merged
    seam_src, seam_dst = seam_pairs(nodes, span, lo_x, lo_y, layout)
    src = np.concatenate([y_src, merge_order[pair + 1] - n_nodes, seam_src])
    dst = np.concatenate([y_src + 1, merge_order[pair], seam_dst])
    del merge_order, pair, y_src, seam_src, seam_dst
    graph = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n_nodes, n_nodes)).tocsr()
    n_comp, node_label = connected_components(graph, directed=False)
    del graph, src, dst

    label = node_label[hit_node]
    n = np.bincount(label, minlength=n_comp)
    tot_f = np.asarray(tot, dtype=np.float64)
    wsum = np.bincount(label, weights=tot_f, minlength=n_comp)
    sum_x = np.bincount(label, weights=tot_f * x, minlength=n_comp)
    sum_y = np.bincount(label, weights=tot_f * y, minlength=n_comp)
    sum_t = np.bincount(label, weights=tot_f * t_rel, minlength=n_comp)
    sum_tot = np.bincount(label, weights=tot_f * tot_f, minlength=n_comp)
    # Clusters whose ToT sums to zero fall back to equal weights (their
    # weighted sums, ToT's included, are all zero so far).
    zero = wsum == 0
    if zero.any():
        z = zero[label]
        lz = label[z]
        wsum += np.bincount(lz, minlength=n_comp)
        sum_x += np.bincount(lz, weights=x[z], minlength=n_comp)
        sum_y += np.bincount(lz, weights=y[z], minlength=n_comp)
        sum_t += np.bincount(lz, weights=np.asarray(t_rel)[z], minlength=n_comp)

    comp_pulse = np.empty(n_comp, dtype=np.int64)
    comp_pulse[node_label] = nodes // (span * span)

    keep = np.nonzero(n >= min_size)[0]
    keep = keep[np.argsort(comp_pulse[keep], kind="stable")]
    ws = wsum[keep]
    t_avg = sum_t[keep] / ws
    if timing is not None:
        t_avg = _apply_cluster_timing(t_avg, keep, n_comp, label, x, y, t_rel, tot, timing, t_jitter_ns)
    elif t_jitter_ns > 0:
        t_avg = t_avg + np.random.random(len(keep)) * t_jitter_ns
    return {
        "pulse": comp_pulse[keep],
        "x": sum_x[keep] / ws,
        "y": sum_y[keep] / ws,
        "t": t_avg,
        "tot": sum_tot[keep] / ws,
        "n": n[keep],
    }


def _apply_cluster_timing(t_avg, keep, n_comp, label, x, y, t_rel, tot, timing, t_jitter_ns):
    """`timing`'s corrected time for each kept cluster (in `keep` order),
    computed for all of them at once from their hits; clusters it gives no
    finite time keep t_avg."""
    if len(keep) == 0:
        return t_avg
    rank = np.full(n_comp, -1, dtype=np.int64)
    rank[keep] = np.arange(len(keep))
    hit_cluster = rank[label]
    sel = hit_cluster >= 0
    times = timing.cluster_times(
        hit_cluster[sel], len(keep), np.asarray(x)[sel], np.asarray(y)[sel],
        np.asarray(t_rel, dtype=np.float64)[sel], np.asarray(tot)[sel],
    )
    good = np.isfinite(times)
    if t_jitter_ns > 0:
        times = times + np.random.random(len(times)) * t_jitter_ns
    return np.where(good, times, t_avg)


# ---- histograms -------------------------------------------------------------


def _uniform_bin_index(values, bins, lo, hi):
    """(bin index, in-range mask) for equal-width bins on [lo, hi] -- the
    same binning np.histogram uses (the last bin includes hi), but a single
    pass instead of a per-value search."""
    v = np.asarray(values, dtype=np.float64)
    ok = (v >= lo) & (v <= hi)
    idx = ((v - lo) * (bins / (hi - lo))).astype(np.int64, copy=False)
    np.clip(idx, 0, bins - 1, out=idx)
    return idx, ok


def hist2d_uniform(xs, ys, bins, lo, hi):
    """Equivalent of np.histogram2d(xs, ys, bins, range=[[lo, hi], [lo, hi]])[0]."""
    if len(xs) == 0:
        return np.zeros((bins, bins), dtype=np.float64)
    ix, okx = _uniform_bin_index(xs, bins, lo, hi)
    iy, oky = _uniform_bin_index(ys, bins, lo, hi)
    ok = okx & oky
    flat = ix[ok] * bins + iy[ok]
    return np.bincount(flat, minlength=bins * bins).reshape(bins, bins).astype(np.float64)


def make_pixel_hist(xs, ys, bins: int = 256, range_min: float = 0.0, range_max: float = 256.0):
    bins = max(1, int(bins))
    if range_max <= range_min:
        range_min, range_max = 0.0, float(chip_layout.get_active_layout().size)
    return hist2d_uniform(xs, ys, bins, range_min, range_max)


def make_cluster_hists(
    xs,
    ys,
    ts,
    t_bins: int = 1000,
    t_min: float = 0.0,
    t_max: float = 1000.0,
    xy_bins: int = 256,
    xy_min: float = 0.0,
    xy_max: float = 256.0,
):
    xy_bins = max(1, int(xy_bins))
    if xy_max <= xy_min:
        xy_min, xy_max = 0.0, float(chip_layout.get_active_layout().size)
    ts = np.asarray(ts, dtype=np.float64)
    in_t = (ts >= t_min) & (ts <= t_max)
    hist = hist2d_uniform(np.asarray(xs)[in_t], np.asarray(ys)[in_t], xy_bins, xy_min, xy_max)
    return hist, make_hist_1d(ts[in_t], t_bins, t_min, t_max)

def make_counts_per_pixel_hist(pixel_hist, bins: int = 0, range_min: float = 0.0, range_max: float = 0.0):
    bins = int(bins)
    if pixel_hist is None or pixel_hist.size == 0:
        auto_range_max = max(range_max, range_min + 1.0)
        return make_hist_1d([], _auto_counts_bins(bins, range_min, auto_range_max), range_min, auto_range_max)
    values = pixel_hist.ravel()
    if range_max <= range_min:
        data_max = float(values.max()) if values.size else 0.0
        range_max = max(range_min + 1.0, data_max)
    return make_hist_1d(values, _auto_counts_bins(bins, range_min, range_max), range_min, range_max)


def _auto_counts_bins(bins: int, range_min: float, range_max: float) -> int:
    """Resolve the "counts / pixel" bin count. bins <= 0 means "one bin per
    integer count value, scaled to the current range"."""
    if bins > 0:
        return bins
    span = range_max - range_min
    return max(1, min(8192, int(round(span))))


def hist_args(cfg):
    return {
        "bins": max(1, int(cfg["bins"])),
        "range_min": float(cfg["min"]),
        "range_max": float(cfg["max"]),
    }


def make_hist_1d(values, bins, range_min, range_max):
    bins = max(1, int(bins))
    if range_max <= range_min:
        range_max = range_min + 1.0
    counts, edges = np.histogram(values, bins=bins, range=(range_min, range_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"counts": counts, "bins": centers, "range": (range_min, range_max)}


def copy_hist(hist):
    return {
        "counts": hist["counts"].copy(),
        "bins": hist["bins"],
        "range": hist["range"],
    }


def add_hist(base, new):
    if base["range"] != new["range"] or len(base["counts"]) != len(new["counts"]):
        return copy_hist(new)
    base["counts"] = base["counts"] + new["counts"]
    return base


def add_hist_2d(base, new):
    if base is None or base.shape != new.shape:
        return new.copy()
    return base + new


def auto_hist_ylim(counts):
    max_val = counts.max() if len(counts) else 0
    if max_val <= 0:
        max_val = 1
    return 0, 1.1 * max_val


def summarize_frame(events: FrameEvents, cluster_size_sum, cluster_count, processed_packets, total_packets, start_time):
    """Per-shot rates and counts for one frame's FrameEvents (`start_time`
    is the frame's time.perf_counter() start, for analysis_time)."""
    total_pulses = len(events.pulses)
    clusters = np.bincount(events.cluster_pulse, minlength=total_pulses)
    etofs = np.bincount(events.etof_pulse, minlength=total_pulses)
    itofs = np.bincount(events.itof_pulse, minlength=total_pulses)
    has_etof = etofs > 0
    has_itof = itofs > 0
    has_cluster = clusters > 0

    any_etof = int(has_etof.sum())
    any_itof = int(has_itof.sum())
    any_cluster = int(has_cluster.sum())
    multi_etof = int((etofs > 1).sum())
    multi_itof = int((itofs > 1).sum())
    multi_cluster = int((clusters > 1).sum())
    all_three = int((has_etof & has_itof & has_cluster).sum())
    total_etof = len(events.etof_t)
    total_itof = len(events.itof_t)
    total_clusters = len(events.cluster_t)

    real_time = 0.0
    if total_pulses > 1:
        real_time = float(events.pulses[-1] - events.pulses[0]) * 1e-9
    analysis_time = time.perf_counter() - start_time

    clusters_per_shot = (total_clusters / total_pulses) if total_pulses else 0
    electrons_per_shot = (total_etof / total_pulses) if total_pulses else 0
    ions_per_shot = (total_itof / total_pulses) if total_pulses else 0
    electrons_per_cluster = (total_etof / total_clusters) if total_clusters else 0
    ions_per_cluster = (total_itof / total_clusters) if total_clusters else 0
    avg_cluster_size = (cluster_size_sum / cluster_count) if cluster_count else 0
    pulses_per_second = (total_pulses / real_time) if real_time > 0 else 0
    analysis_over_real = (analysis_time / real_time) if real_time > 0 else 0
    data_ratio = (processed_packets / total_packets) if total_packets else 0

    return {
        "clusters_per_shot": clusters_per_shot,
        "electrons_per_shot": electrons_per_shot,
        "ions_per_shot": ions_per_shot,
        "shots_all_three": (all_three / total_pulses) if total_pulses else 0,
        "shots_multi_clusters": (multi_cluster / total_pulses) if total_pulses else 0,
        "shots_multi_electrons": (multi_etof / total_pulses) if total_pulses else 0,
        "shots_multi_ions": (multi_itof / total_pulses) if total_pulses else 0,
        "electrons_per_cluster": electrons_per_cluster,
        "ions_per_cluster": ions_per_cluster,
        "avg_cluster_size": avg_cluster_size,
        "pulses_per_second": pulses_per_second,
        "analysis_over_real": analysis_over_real,
        "data_ratio": data_ratio,
        "total_pulses": total_pulses,
        "total_etof": total_etof,
        "total_itof": total_itof,
        "total_clusters": total_clusters,
        "any_etof": any_etof,
        "any_itof": any_itof,
        "any_cluster": any_cluster,
        "multi_etof": multi_etof,
        "multi_itof": multi_itof,
        "multi_cluster": multi_cluster,
        "all_three": all_three,
        "cluster_size_sum": cluster_size_sum,
        "cluster_count": cluster_count,
        "processed_packets": processed_packets,
        "total_packets": total_packets,
        "analysis_time": analysis_time,
        "real_time": real_time,
    }

def merge_stats(base, new):
    merged = base.copy()
    merged["total_pulses"] += new["total_pulses"]
    merged["total_etof"] += new["total_etof"]
    merged["total_itof"] += new["total_itof"]
    merged["total_clusters"] += new["total_clusters"]
    merged["any_etof"] += new["any_etof"]
    merged["any_itof"] += new["any_itof"]
    merged["any_cluster"] += new["any_cluster"]
    merged["multi_etof"] += new["multi_etof"]
    merged["multi_itof"] += new["multi_itof"]
    merged["multi_cluster"] += new["multi_cluster"]
    merged["all_three"] += new["all_three"]
    merged["cluster_size_sum"] += new["cluster_size_sum"]
    merged["cluster_count"] += new["cluster_count"]
    merged["processed_packets"] += new["processed_packets"]
    merged["total_packets"] += new["total_packets"]
    merged["analysis_time"] += new["analysis_time"]
    merged["real_time"] += new["real_time"]
    merged["analysis_time_last"] = new.get("analysis_time_last", merged.get("analysis_time_last", 0))

    total_pulses = merged["total_pulses"]
    total_clusters = merged["total_clusters"]
    total_etof = merged["total_etof"]
    total_itof = merged["total_itof"]
    real_time = merged["real_time"]

    merged["clusters_per_shot"] = (total_clusters / total_pulses) if total_pulses else 0
    merged["electrons_per_shot"] = (total_etof / total_pulses) if total_pulses else 0
    merged["ions_per_shot"] = (total_itof / total_pulses) if total_pulses else 0
    merged["shots_all_three"] = (merged["all_three"] / total_pulses) if total_pulses else 0
    merged["shots_multi_clusters"] = (merged["multi_cluster"] / total_pulses) if total_pulses else 0
    merged["shots_multi_electrons"] = (merged["multi_etof"] / total_pulses) if total_pulses else 0
    merged["shots_multi_ions"] = (merged["multi_itof"] / total_pulses) if total_pulses else 0
    merged["electrons_per_cluster"] = (total_etof / total_clusters) if total_clusters else 0
    merged["ions_per_cluster"] = (total_itof / total_clusters) if total_clusters else 0
    merged["avg_cluster_size"] = (
        merged["cluster_size_sum"] / merged["cluster_count"]
        if merged["cluster_count"]
        else 0
    )
    merged["pulses_per_second"] = (total_pulses / real_time) if real_time > 0 else 0
    merged["analysis_over_real"] = (merged["analysis_time"] / real_time) if real_time > 0 else 0
    merged["data_ratio"] = (
        merged["processed_packets"] / merged["total_packets"]
        if merged["total_packets"]
        else 0
    )
    return merged
