"""Raw TPX3 -> cv4 conversion, shared by Monitored Acquisition (live, one raw
file at a time as frames arrive) and Analysis > conversion (offline, over a
folder of raw files an acquisition tab already wrote), so both produce
identical cv4 files.

No Qt here: the per-file pipeline, histogram accumulation for the plots,
working out which raw files go into which cv4 for each acquisition tab's
folder layout, and writing one cv4 with the .partial/finalize convention.
"""
import json
import pathlib
import re
import time
from dataclasses import dataclass, field

import cv4_writer
from cv4_writer import Cv4Writer
from timewalk import apply_timewalk_correction
from tpx_processing import (
    decode_tpx3,
    sort_tdcs,
    group_pixels_by_pulse,
    group_times_relative,
    cluster_pixels_by_pulse,
    make_pixel_hist_from_pulses,
    make_cluster_hists,
    make_hist_1d,
    hist_args,
    flatten_dict,
    copy_hist,
    add_hist,
    add_hist_2d,
    summarize_records,
    merge_stats,
)


# ---- per-file pipeline ----------------------------------------------------

def process_raw_file(path, settings, timewalk=None):
    """Decode one raw file, optionally timewalk-correct its pixels, cluster
    it, and bin its plot histograms with `settings` (a
    HistogramPlotPanel.hist_snapshot()). Returns pulses_sorted /
    clusters_by_pulse / etof_by_pulse / itof_by_pulse (what
    Cv4Writer.append_frame takes) plus plot_result."""
    start = time.perf_counter()

    # No max_packets cap: every packet is processed.
    pixels, tdcs, processed_packets, total_packets = decode_tpx3(str(path), max_packets=None)
    if timewalk is not None:
        pixels = apply_timewalk_correction(pixels, timewalk)
    etof, itof, pulses = sort_tdcs(300.0, tdcs)

    pulses_sorted = sorted(pulses)
    pixels_by_pulse = group_pixels_by_pulse(pixels, pulses_sorted)
    clusters_by_pulse, cluster_size_sum, cluster_count = cluster_pixels_by_pulse(pixels_by_pulse)
    etof_by_pulse = group_times_relative(etof, pulses_sorted)
    itof_by_pulse = group_times_relative(itof, pulses_sorted)

    pulse_records = []
    for pulse_time in pulses_sorted:
        pulse_records.append(
            (
                pulse_time,
                pixels_by_pulse.get(pulse_time, []),
                clusters_by_pulse.get(pulse_time, []),
                etof_by_pulse.get(pulse_time, []),
                itof_by_pulse.get(pulse_time, []),
            )
        )

    pixel_hist = make_pixel_hist_from_pulses(
        pixels_by_pulse,
        bins=int(settings["pixel"]["bins"]),
        range_min=settings["pixel"]["min"],
        range_max=settings["pixel"]["max"],
    )
    cluster_hist, cluster_t_hist = make_cluster_hists(
        clusters_by_pulse,
        t_bins=int(settings["cluster_t"]["bins"]),
        t_min=settings["cluster_t"]["min"],
        t_max=settings["cluster_t"]["max"],
        xy_bins=int(settings["cluster"]["bins"]),
        xy_min=settings["cluster"]["min"],
        xy_max=settings["cluster"]["max"],
    )
    itof_hist = make_hist_1d(flatten_dict(itof_by_pulse), **hist_args(settings["itof"]))
    etof_hist = make_hist_1d(flatten_dict(etof_by_pulse), **hist_args(settings["etof"]))

    stats = summarize_records(
        pulse_records, cluster_size_sum, cluster_count, processed_packets, total_packets, start
    )
    stats["analysis_time_last"] = time.perf_counter() - start

    plot_result = {
        "pixel_hist": pixel_hist,
        "cluster_hist": cluster_hist,
        "itof_hist": itof_hist,
        "etof_hist": etof_hist,
        "cluster_t_hist": cluster_t_hist,
        "stats": stats,
        "settings": settings,
    }

    return {
        "pulses_sorted": pulses_sorted,
        "clusters_by_pulse": clusters_by_pulse,
        "etof_by_pulse": etof_by_pulse,
        "itof_by_pulse": itof_by_pulse,
        "plot_result": plot_result,
    }


def accumulate_plot_result(accumulated, result):
    """Add one file's plot_result into a running total and return it. Starts
    over from a copy of `result` when there's no total yet or the histogram
    settings changed, since differently binned histograms can't be summed."""
    if accumulated is None or accumulated.get("settings") != result.get("settings"):
        return {
            "pixel_hist": result["pixel_hist"].copy(),
            "cluster_hist": result["cluster_hist"].copy(),
            "itof_hist": copy_hist(result["itof_hist"]),
            "etof_hist": copy_hist(result["etof_hist"]),
            "cluster_t_hist": copy_hist(result["cluster_t_hist"]),
            "stats": result["stats"].copy(),
            "settings": result["settings"],
        }
    accumulated["pixel_hist"] = add_hist_2d(accumulated["pixel_hist"], result["pixel_hist"])
    accumulated["cluster_hist"] = add_hist_2d(accumulated["cluster_hist"], result["cluster_hist"])
    accumulated["itof_hist"] = add_hist(accumulated["itof_hist"], result["itof_hist"])
    accumulated["etof_hist"] = add_hist(accumulated["etof_hist"], result["etof_hist"])
    accumulated["cluster_t_hist"] = add_hist(accumulated["cluster_t_hist"], result["cluster_t_hist"])
    accumulated["stats"] = merge_stats(accumulated["stats"], result["stats"])
    return accumulated


# ---- offline conversion of a raw data folder --------------------------------

# Parameter Sweep's per-visit raw folder: pos_<visit number>_<requested position>
_SWEEP_VISIT_RE = re.compile(r"^pos_(\d+)_([+-]\d+(?:\.\d+)?)$")


@dataclass
class ConversionJob:
    """One cv4 file to write, and the raw files (in order) that go into it."""
    cv4_path: pathlib.Path
    raw_files: list
    metadata: dict = field(default_factory=dict)


def list_raw_files(directory):
    """A folder's raw files in acquisition order -- by modification time,
    the same order the acquisition tabs pick up new frames in."""
    files = [p for p in pathlib.Path(directory).glob("*.tpx3") if p.is_file()]
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))


def _read_run_metadata(folder):
    """Collection parameters' run_metadata.json, if the run has one."""
    path = folder / "run_metadata.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def plan_conversion(source_folder, output_folder=None):
    """Work out the cv4 file(s) a folder of raw data converts into, split
    and named the way the acquisition tab that wrote it names its own:

    - Monitored Acquisition: <run>/raw/*.tpx3 -> <run>/<run>.cv4
    - Collection parameters: <run>/*.tpx3 -> <run>/<run>.cv4
    - Parameter Sweep: <run>/raw/pos_<visit>_<position>/*.tpx3 -> one
      <run>/pos_<index>_<position>.cv4 per position, with every pass that
      visited it appended in visit order

    `source_folder` may be the run folder or its raw/ folder. cv4 files go
    in `output_folder` if given, otherwise in the run folder. Returns a
    list of ConversionJob (empty if there are no raw files)."""
    source = pathlib.Path(source_folder)
    if source.name.lower() == "raw" and not (source / "raw").is_dir():
        run_folder, raw_dir = source.parent, source
    else:
        run_folder = source
        raw_dir = source / "raw" if (source / "raw").is_dir() else source
    out = pathlib.Path(output_folder) if output_folder else run_folder

    base_metadata = _read_run_metadata(run_folder)
    base_metadata["Source Raw Folder"] = str(raw_dir)

    jobs = []
    direct = list_raw_files(raw_dir)
    if direct:
        jobs.append(ConversionJob(out / f"{run_folder.name}.cv4", direct, dict(base_metadata)))

    visits = []
    for sub in raw_dir.iterdir():
        match = _SWEEP_VISIT_RE.match(sub.name) if sub.is_dir() else None
        if match:
            visits.append((int(match.group(1)), float(match.group(2)), sub))

    # A sweep numbers each position's cv4 by its place in the position
    # list, which is the order positions are first visited in.
    folders_by_position = {}
    for _, requested, sub in sorted(visits, key=lambda v: v[0]):
        folders_by_position.setdefault(requested, []).append(sub)
    for index, (requested, folders) in enumerate(folders_by_position.items(), start=1):
        files = [path for folder in folders for path in list_raw_files(folder)]
        if not files:
            continue
        metadata = dict(base_metadata)
        metadata["Requested Position"] = requested
        metadata["Source Raw Folder"] = ", ".join(str(folder) for folder in folders)
        jobs.append(ConversionJob(out / f"pos_{index:03d}_{requested:+.4f}.cv4", files, metadata))

    return jobs


def convert_job(job, hist_settings, timewalk=None, stop_event=None, on_file=None):
    """Write one job's raw files into its cv4 with Monitored Acquisition's
    file logic: written as <name>.partial.cv4 and renamed to <name>.cv4
    only once every raw file was processed, so a stop or a file that failed
    to process leaves the .partial name.

    `hist_settings()` returns the histogram settings each file's plot
    result is binned with. `on_file(path, frame, error)` is called after
    every raw file, with `frame` None and `error` set if it failed.
    Returns (final_path, files_processed)."""
    partial_path = cv4_writer.partial_path_for(job.cv4_path)
    writer = Cv4Writer(partial_path, metadata=job.metadata)
    processed = 0
    try:
        for path in job.raw_files:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                frame = process_raw_file(path, hist_settings(), timewalk)
                writer.append_frame(
                    frame["pulses_sorted"],
                    frame["clusters_by_pulse"],
                    frame["etof_by_pulse"],
                    frame["itof_by_pulse"],
                )
                writer.flush()
            except Exception as exc:
                if on_file is not None:
                    on_file(path, None, exc)
                continue
            processed += 1
            if on_file is not None:
                on_file(path, frame, None)
    finally:
        all_processed = processed >= len(job.raw_files)
        writer.set_attrs({
            "Frames Collected": len(job.raw_files),
            "Frames Processed": processed,
            "All Collected Frames Processed": all_processed,
        })
        writer.close()
    return cv4_writer.finalize_partial_path(partial_path, all_processed), processed
