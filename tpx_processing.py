"""Pure TPX3 decoding / clustering / histogram-building logic.

No Tkinter or matplotlib dependencies here so this module can be shared by
any UI (diagnostics, monitored acquisition, offline reprocessing, ...).
"""
import bisect
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


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
    tdcs: List[Tuple[int, int, int, int]] = []
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
            f_time = (reduced_tdcs >> 5) & 0xF
            c_time = (reduced_tdcs >> 9) & ((1 << 35) - 1)
            tdc_type = (reduced_tdcs >> 56) & 0xF
            c_time = c_time & 0x1FFFFFFFF

            tdcs.extend(
                (int(t), int(c), int(f) - 1, 0)
                for t, c, f in zip(tdc_type, c_time, f_time)
            )

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


def sort_tdcs(cutoff_ns: float, tdcs: List[Tuple[int, int, int, int]], jitter_ns: float = TDC_JITTER_NS):
    start_time = 0.0
    pulses = []
    etof = []
    itof = []
    for tdc_type, c_time, ftime, _ in tdcs:
        tdc_time = 3.125 * c_time + 0.260 * ftime
        if tdc_type == 15:
            start_time = tdc_time
        elif tdc_type == 14:
            etof.append(tdc_time)
        elif tdc_type == 10:
            pulse_len = tdc_time - start_time
            if pulse_len > cutoff_ns:
                if start_time > 0:
                    pulses.append(start_time)
            else:
                if start_time > 0:
                    itof.append(start_time)

    etof = apply_jitter(etof, jitter_ns)
    itof = apply_jitter(itof, jitter_ns)
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
