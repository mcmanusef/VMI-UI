"""Pure TPX3 decoding / clustering / histogram-building logic.

No Tkinter or matplotlib dependencies here so this module can be shared by
any UI (diagnostics, monitored acquisition, offline reprocessing, ...).
"""
import bisect
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from tdc_mapping import (
    FAMILY_FPGA_TDC,
    FAMILY_PICOTDC,
    FPGA_TDC_NS_PER_COUNT,
    FPGA_TDC_SUBTYPES,
    PICOTDC_NS_PER_COUNT,
)


@dataclass
class PixelData:
    time_ns: float
    x: int
    y: int
    tot: int


PIXEL_RES_NS = 25 / 16

# Dither widths used to smear out the quantization of the raw timestamps.
TDC_JITTER_NS = 0.26          # applied to every e-ToF and i-ToF time
CLUSTER_T_JITTER_NS = 1.56    # applied to every cluster time


def iter_packet_arrays(path: str) -> Iterable[np.ndarray]:
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if not header or len(header) < 8:
                break
            payload_len = int.from_bytes(header[6:8], "little")
            if payload_len <= 0:
                continue
            payload = f.read(payload_len)
            if len(payload) < payload_len:
                break
            words = np.frombuffer(payload, dtype="<u8")
            packets = (words - (1 << 62)) & ((1 << 64) - 1)
            yield packets


def detect_headers(path: str, sample_packets: int = 4096) -> Tuple[int, int]:
    counts = np.zeros(16, dtype=np.int64)
    total = 0
    for packets in iter_packet_arrays(path):
        headers = (packets >> 60).astype(np.int64)
        counts += np.bincount(headers, minlength=16)
        total += len(headers)
        if total >= sample_packets:
            break

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


def decode_tpx3(path: str, max_packets: int = None):
    pixels: List[PixelData] = []
    tdcs: List[Tuple[int, int, int, float]] = []
    processed = 0
    total = 0
    pixel_header, tdc_header = detect_headers(path)

    for packets in iter_packet_arrays(path):
        total += len(packets)
        if max_packets is not None and processed >= max_packets:
            continue

        if max_packets is not None:
            remaining = max_packets - processed
            if remaining <= 0:
                continue
            packets = packets[:remaining]

        processed += len(packets)
        headers = packets >> 60
        reduced = packets & 0x0FFF_FFFF_FFFF_FFFF

        pixel_mask = headers == pixel_header
        if np.any(pixel_mask):
            reduced_pixels = reduced[pixel_mask]
            c_time = reduced_pixels & 0xFFFF
            f_time = (reduced_pixels >> 16) & 0xF
            tot = (reduced_pixels >> 20) & 0x3FF
            m_time = (reduced_pixels >> 30) & 0x3FFF
            pix_addr = (reduced_pixels >> 44) & 0x1FFFF

            toa_raw = c_time * (1 << 18) + m_time * (1 << 4) - f_time
            toa_ns = toa_raw * PIXEL_RES_NS

            dcol = (pix_addr & 0xFE00) >> 8
            spix = (pix_addr & 0x01F8) >> 1
            pix = pix_addr & 0x0007
            xs = dcol + (pix // 4)
            ys = spix + (pix & 0x3)

            pixels.extend(
                PixelData(time_ns=float(t), x=int(x), y=int(y), tot=int(tt))
                for t, x, y, tt in zip(toa_ns, xs, ys, tot)
            )

        tdc_mask = headers == tdc_header
        if np.any(tdc_mask):
            reduced_tdcs = reduced[tdc_mask]
            sub = (reduced_tdcs >> 56) & 0xF

            pico = reduced_tdcs[sub == 0xC]
            if len(pico):
                valid = (pico >> 52) & 1
                pico = pico[valid == 0]
            if len(pico):
                channel = (((pico >> 53) & 0x3) << 4) | ((pico >> 48) & 0xF)
                edge = (pico >> 47) & 1
                ts_lower = (pico >> 21) & 0x3FFFFFF
                spidr = pico & 0x1FFFFF
                ts = (spidr << 26) | ts_lower
                time_ns = ts.astype(np.float64) * PICOTDC_NS_PER_COUNT
                tdcs.extend(
                    (0xC, int(c), int(e), float(t))
                    for c, e, t in zip(channel, edge, time_ns)
                )

            for sub_value, (channel_id, edge_value) in FPGA_TDC_SUBTYPES.items():
                fw = reduced_tdcs[sub == sub_value]
                if len(fw) == 0:
                    continue
                fine = (fw >> 5) & 0xF
                ok = (fine > 0) & (fine <= 12)
                fw = fw[ok]
                fine = fine[ok]
                if len(fw) == 0:
                    continue
                coarse = (fw >> 9) & ((1 << 35) - 1)
                combined = coarse.astype(np.int64) * 12 + (fine.astype(np.int64) - 1)
                time_ns = combined.astype(np.float64) * FPGA_TDC_NS_PER_COUNT
                tdcs.extend((sub_value, channel_id, edge_value, float(t)) for t in time_ns)

        if max_packets is not None and processed >= max_packets:
            continue

    return pixels, tdcs, processed, total


def apply_jitter(values, width):
    """Add a uniform random offset in [0, width) to every value."""
    if width <= 0 or len(values) == 0:
        return values
    arr = np.asarray(values, dtype=np.float64)
    arr = arr + np.random.random(arr.shape) * width
    return arr.tolist()


def classify_tdcs(tdcs: List[Tuple[int, int, int, float]], mapping, jitter_ns: float = TDC_JITTER_NS):
    """Splits raw TDC hits -- (sub_nibble, channel, edge, time_ns) tuples
    from decode_tpx3's TDC branch, where sub_nibble 0xC is a PicoTDC hit and
    0xf/0xa/0xe/0xb is an FPGA TDC hit (edge: 1 = rising, 0 = falling) -- into
    pulse/etof/itof time lists per `mapping` (a tdc_mapping.TdcMapping).

    Each signal's physical TDC line (family + channel) is matched; a signal
    with no pulse-width gate takes every hit on its line (matching its
    time_edge) directly, while a gated signal pairs each rising edge with
    the next falling edge on that same line and keeps the pairing's start
    (or end) time only when the resulting width falls in
    [gate_min_ns, gate_max_ns] -- so multiple signals can share one
    physical line and be told apart purely by pulse length (e.g. a long
    trigger pulse vs a short i-ToF pulse on the same FPGA TDC channel)."""
    hits_by_line = defaultdict(list)
    for sub, channel, edge, t in tdcs:
        family = FAMILY_PICOTDC if sub == 0xC else FAMILY_FPGA_TDC
        hits_by_line[(family, channel)].append((edge, t))

    def paired_pulses(key):
        start = None
        pulses_ = []
        for edge, t in hits_by_line.get(key, ()):
            if edge == 1:
                start = t
            elif start is not None:
                pulses_.append((start, t - start))
                start = None
        return pulses_

    def resolve(signal):
        key = (signal.source.family, signal.source.channel)
        if signal.gate_min_ns is None and signal.gate_max_ns is None:
            want_edge = 1 if signal.time_edge == "rising" else 0
            return [t for edge, t in hits_by_line.get(key, ()) if edge == want_edge]
        lo = signal.gate_min_ns if signal.gate_min_ns is not None else float("-inf")
        hi = signal.gate_max_ns if signal.gate_max_ns is not None else float("inf")
        out = []
        for start, width in paired_pulses(key):
            if lo <= width <= hi:
                out.append(start if signal.time_edge == "rising" else start + width)
        return out

    pulses = resolve(mapping.pulse_trigger)
    etof = apply_jitter(resolve(mapping.etof), jitter_ns)
    itof = apply_jitter(resolve(mapping.itof), jitter_ns)
    return etof, itof, pulses


def group_times_relative(times, pulses_sorted):
    grouped = {p: [] for p in pulses_sorted}
    for t in times:
        idx = bisect.bisect_right(pulses_sorted, t) - 1
        if idx < 0:
            continue
        pulse_time = pulses_sorted[idx]
        grouped[pulse_time].append(t - pulse_time)
    return grouped


def group_pixels_by_pulse(pixels: List[PixelData], pulses_sorted):
    grouped = {p: [] for p in pulses_sorted}
    for pix in pixels:
        idx = bisect.bisect_right(pulses_sorted, pix.time_ns) - 1
        if idx < 0:
            continue
        pulse_time = pulses_sorted[idx]
        grouped[pulse_time].append((pix.time_ns - pulse_time, pix.x, pix.y, pix.tot))
    return grouped


def cluster_pixels_by_pulse(pixels_by_pulse, t_jitter_ns: float = CLUSTER_T_JITTER_NS):
    clusters_by_pulse = {}
    cluster_size_sum = 0
    cluster_count = 0

    for pulse_time, pix_list in pixels_by_pulse.items():
        if not pix_list:
            continue
        coord_map = {}
        for t_rel, x, y, tot in pix_list:
            coord_map.setdefault((x, y), []).append((t_rel, tot))

        coords = set(coord_map.keys())
        clusters = []
        while coords:
            start = coords.pop()
            stack = [start]
            cluster_coords = [start]
            while stack:
                x, y = stack.pop()
                neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                for n in neighbors:
                    if n in coords:
                        coords.remove(n)
                        stack.append(n)
                        cluster_coords.append(n)
            clusters.append(cluster_coords)

        cluster_records = []
        for cluster_id, cluster_coords in enumerate(clusters):
            cluster_pixels = []
            for coord in cluster_coords:
                cluster_pixels.extend([(coord[0], coord[1], t_rel, tot) for t_rel, tot in coord_map[coord]])
            if not cluster_pixels:
                continue
            weights = [tot for _, _, _, tot in cluster_pixels]
            weight_sum = sum(weights)
            if weight_sum == 0:
                weights = [1] * len(cluster_pixels)
                weight_sum = len(cluster_pixels)
            avg_x = sum(x * w for x, _, _, w in cluster_pixels) / weight_sum
            avg_y = sum(y * w for _, y, _, w in cluster_pixels) / weight_sum
            avg_t = sum(t * w for _, _, t, w in cluster_pixels) / weight_sum
            avg_tot = sum(tot * w for _, _, _, w in cluster_pixels) / weight_sum
            n_pixels = len(cluster_pixels)
            if n_pixels < 5:
                continue

            if t_jitter_ns > 0:
                avg_t += float(np.random.random()) * t_jitter_ns

            cluster_records.append(
                {
                    "pulse_time_ns": pulse_time,
                    "cluster_id": cluster_id,
                    "n_pixels": n_pixels,
                    "weight_sum": weight_sum,
                    "avg_x": avg_x,
                    "avg_y": avg_y,
                    "avg_t_rel_ns": avg_t,
                    "avg_tot": avg_tot,
                }
            )

            cluster_size_sum += n_pixels
            cluster_count += 1

        clusters_by_pulse[pulse_time] = cluster_records

    return clusters_by_pulse, cluster_size_sum, cluster_count


def make_pixel_hist_from_pulses(pixels_by_pulse, bins: int = 256, range_min: float = 0.0, range_max: float = 256.0):
    bins = max(1, int(bins))
    if range_max <= range_min:
        range_min, range_max = 0.0, 256.0
    points = []
    for pix_list in pixels_by_pulse.values():
        for _, x, y, _ in pix_list:
            points.append((x, y))
    if not points:
        return np.zeros((bins, bins), dtype=np.float64)
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    hist, _, _ = np.histogram2d(
        xs, ys, bins=bins, range=[[range_min, range_max], [range_min, range_max]]
    )
    return hist


def make_cluster_hists(
    clusters_by_pulse,
    t_bins: int = 1000,
    t_min: float = 0.0,
    t_max: float = 1000.0,
    xy_bins: int = 256,
    xy_min: float = 0.0,
    xy_max: float = 256.0,
):
    xy_bins = max(1, int(xy_bins))
    if xy_max <= xy_min:
        xy_min, xy_max = 0.0, 256.0

    cluster_points = []
    cluster_times = []
    for clusters in clusters_by_pulse.values():
        for cluster in clusters:
            t_rel = cluster["avg_t_rel_ns"]
            if t_rel < t_min or t_rel > t_max:
                continue
            cluster_points.append((cluster["avg_x"], cluster["avg_y"]))
            cluster_times.append(t_rel)

    if cluster_points:
        xs = np.array([c[0] for c in cluster_points])
        ys = np.array([c[1] for c in cluster_points])
        hist, _, _ = np.histogram2d(
            xs, ys, bins=xy_bins, range=[[xy_min, xy_max], [xy_min, xy_max]]
        )
    else:
        hist = np.zeros((xy_bins, xy_bins), dtype=np.float64)

    cluster_t_hist = make_hist_1d(cluster_times, t_bins, t_min, t_max)
    return hist, cluster_t_hist


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


def flatten_dict(mapping):
    out = []
    for items in mapping.values():
        out.extend(items)
    return out


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


def summarize_records(pulse_records, cluster_size_sum, cluster_count, processed_packets, total_packets, start_time):
    total_pulses = len(pulse_records)
    any_etof = 0
    any_itof = 0
    any_cluster = 0
    multi_etof = 0
    multi_itof = 0
    multi_cluster = 0
    all_three = 0
    total_etof = 0
    total_itof = 0
    total_clusters = 0

    pulse_times = []
    for pulse_time, _, clusters, etofs, itofs in pulse_records:
        pulse_times.append(pulse_time)
        has_etof = len(etofs) > 0
        has_itof = len(itofs) > 0
        has_cluster = len(clusters) > 0

        any_etof += 1 if has_etof else 0
        any_itof += 1 if has_itof else 0
        any_cluster += 1 if has_cluster else 0

        multi_etof += 1 if len(etofs) > 1 else 0
        multi_itof += 1 if len(itofs) > 1 else 0
        multi_cluster += 1 if len(clusters) > 1 else 0

        all_three += 1 if (has_etof and has_itof and has_cluster) else 0

        total_etof += len(etofs)
        total_itof += len(itofs)
        total_clusters += len(clusters)

    real_time = 0.0
    if len(pulse_times) > 1:
        real_time = (max(pulse_times) - min(pulse_times)) * 1e-9
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
