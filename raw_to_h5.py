"""Direct TPX3 -> HDF5 conversion, with none of raw_conversion.py's
clustering, pulse-grouping or timewalk correction: each raw file's pixel
hits (x, y, toa, tot) and TDC markers (PicoTDC and FPGA TDC alike) go
straight into their own h5 file. For a run folder that is a Parameter
Sweep's raw/ folder (one subfolder per visited position, see
raw_conversion._SWEEP_VISIT_RE), each raw file inside those subfolders
gets its own h5 too.

Packets are read by their literal header nibble rather than guessed by
frequency (as tpx_processing.decode_tpx3's detect_headers does): this
hardware can mix Timepix3 pixel hits, PicoTDC hits and FPGA TDC hits (and,
on some files, other packet types this module doesn't understand) in one
raw file, across up to 4 chip streams, and detect_headers' "pick the two
most common headers" heuristic has been observed to mislabel pixel/TDC
packets on files where a third packet type dominates. See
tpx_to_h5_diagnostics.ipynb for the packet-format analysis this is based
on.

Chunk framing (Table 6.1): an 8-byte header -- bytes[0:4] the 'TPX3'
magic, byte[4] the chip-stream index, bytes[6:8] the little-endian
payload length -- followed by that many bytes of 64-bit packets.

Packet types by header nibble (bits [63:60]):
  0xb = Timepix3 pixel hit
  0x6 = TDC packet, sub-typed by the next nibble (bits [59:56]):
    0xC                    -> PicoTDC
    0xF / 0xA / 0xE / 0xB  -> FPGA TDC (two channels, rising + falling)
Any other header nibble (including an as-yet unidentified packet family
seen on some files) is skipped.
"""
import dataclasses
import json
import pathlib
import time

import h5py
import numpy as np

import chip_layout
from raw_conversion import list_raw_files, _SWEEP_VISIT_RE
from tdc_mapping import FPGA_TDC_NS_PER_COUNT, FPGA_TDC_SUBTYPES, PICOTDC_NS_PER_COUNT
from tpx_processing import PIXEL_RES_NS


@dataclasses.dataclass
class H5Job:
    """One raw file to convert, and the h5 file it becomes."""
    raw_path: pathlib.Path
    h5_path: pathlib.Path


def plan_conversion(source, output_folder=None):
    """Every raw .tpx3 file under `source` (its raw/ folder if it has one,
    including any Parameter Sweep per-position subfolders), paired with the
    h5 file it converts to. h5 files go in `output_folder` -- mirroring any
    per-position subfolder -- or next to their raw file if `output_folder`
    is blank. `source` may be a single raw .tpx3 file, or a run folder or
    its raw/ folder. Returns a list of H5Job (empty if there are no raw
    files)."""
    source = pathlib.Path(source)
    out = pathlib.Path(output_folder) if output_folder else None

    if source.is_file():
        dest_dir = out if out is not None else source.parent
        return [H5Job(source, dest_dir / (source.stem + ".h5"))]

    if source.name.lower() == "raw" and not (source / "raw").is_dir():
        raw_dir = source
    else:
        raw_dir = source / "raw" if (source / "raw").is_dir() else source

    raw_files = list_raw_files(raw_dir)
    for sub in sorted(p for p in raw_dir.iterdir() if p.is_dir() and _SWEEP_VISIT_RE.match(p.name)):
        raw_files.extend(list_raw_files(sub))

    jobs = []
    for raw_path in raw_files:
        if out is None:
            dest_dir = raw_path.parent
        else:
            rel_parent = raw_path.relative_to(raw_dir).parent
            dest_dir = out if str(rel_parent) == "." else out / rel_parent
        jobs.append(H5Job(raw_path, dest_dir / (raw_path.stem + ".h5")))
    return jobs


# ---- raw decode -------------------------------------------------------------

def iter_chip_tagged_chunks(path, chunk_bytes=64 * 1024 * 1024):
    """Yields (chip_index, words) for every chunk in one raw .tpx3 file:
    `words` is that chunk's packets as a uint64 array, `chip_index` the
    chip stream (0-3) its header named. `chunk_bytes` caps how much of one
    oversized chunk's payload is read at once; chunks are otherwise read
    whole, one header at a time."""
    chunk_bytes = (chunk_bytes // 8) * 8
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if not header or len(header) < 8:
                break
            payload_len = int.from_bytes(header[6:8], "little")
            if payload_len <= 0:
                continue
            chip = header[4]
            remaining = payload_len
            while remaining > 0:
                to_read = min(remaining, chunk_bytes)
                payload = f.read(to_read)
                if len(payload) < to_read:
                    return
                remaining -= to_read
                n_words = len(payload) // 8
                if n_words:
                    yield chip, np.frombuffer(payload, dtype="<u8", count=n_words)


def decode_raw_file(path, layout=None):
    """Decode one raw file by its packets' literal header nibbles -- no
    clustering, pulse-grouping, or timewalk correction, and no guessing at
    which nibble means what (see module docstring).

    Pixel coordinates are on the whole detector's canvas: each chip's
    local x/y placed and oriented per `layout` (default
    chip_layout.get_active_layout()), e.g. 0-511 on a quad.

    Returns a dict of 1D arrays:
      pixel_x, pixel_y       : uint16, detector-canvas pixel
      pixel_toa_ns, pixel_tot: float64 (ns), float64 (ns)
      pixel_chip             : uint8, chip stream (0-3) each hit came from
      tdc_time_ns            : float64, each TDC hit's time (ns) in its own
                                detector's native tick -- PicoTDC and FPGA
                                TDC don't share a clock, so times are only
                                comparable within the same tdc_type family
      tdc_type                : uint8, the packet's own sub-nibble --
                                0xC = PicoTDC, 0xF/0xA/0xE/0xB = FPGA TDC
      tdc_channel             : int16, PicoTDC channel (0-63), or -1 for
                                FPGA TDC (its channel is implied by tdc_type)
      tdc_edge                : uint8, 1 = rising, 0 = falling
      tdc_chip                : uint8, chip stream (0-3) each hit came from
    """
    px_x, px_y, px_toa, px_tot, px_chip = [], [], [], [], []
    tdc_time, tdc_type, tdc_channel, tdc_edge, tdc_chip = [], [], [], [], []
    if layout is None:
        layout = chip_layout.get_active_layout()

    for chip, words in iter_chip_tagged_chunks(path):
        nib = (words >> np.uint64(60)) & np.uint64(0xF)

        pix_mask = nib == np.uint64(0xB)
        if np.any(pix_mask):
            pw = words[pix_mask]
            # PixAddr (bits 59-44): 7-bit double column, 6-bit super
            # pixel, 3-bit pixel -- same split as tpx_processing.decode_tpx3.
            addr = (pw >> np.uint64(44)) & np.uint64(0xFFFF)
            dcol = (addr & np.uint64(0xFE00)) >> np.uint64(8)
            spix = (addr & np.uint64(0x01F8)) >> np.uint64(1)
            pix = addr & np.uint64(0x7)
            xs = (dcol + pix // np.uint64(4)).astype(np.int64)
            ys = (spix + (pix & np.uint64(3))).astype(np.int64)
            xs, ys = layout.to_global(chip, xs, ys)

            ftoa = (pw >> np.uint64(16)) & np.uint64(0xF)
            toa = (pw >> np.uint64(30)) & np.uint64(0x3FFF)
            coarse = pw & np.uint64(0xFFFF)
            toa_count = (((coarse << np.uint64(14)) + toa) << np.uint64(4)).astype(np.int64) \
                - ftoa.astype(np.int64)
            tot_counts = (pw >> np.uint64(20)) & np.uint64(0x3FF)

            px_x.append(xs.astype(np.uint16))
            px_y.append(ys.astype(np.uint16))
            px_toa.append(toa_count.astype(np.float64) * PIXEL_RES_NS)
            px_tot.append(tot_counts.astype(np.float64) * 25.0)
            px_chip.append(np.full(len(pw), chip, dtype=np.uint8))

        tdc_mask = nib == np.uint64(0x6)
        if np.any(tdc_mask):
            tw = words[tdc_mask]
            sub = (tw >> np.uint64(56)) & np.uint64(0xF)

            pico_words = tw[sub == np.uint64(0xC)]
            if len(pico_words):
                valid = (pico_words >> np.uint64(52)) & np.uint64(1)
                pico_words = pico_words[valid == np.uint64(0)]
            if len(pico_words):
                channel = (((pico_words >> np.uint64(53)) & np.uint64(3)) << np.uint64(4)) \
                    | ((pico_words >> np.uint64(48)) & np.uint64(0xF))
                edge = (pico_words >> np.uint64(47)) & np.uint64(1)
                ts_lower = (pico_words >> np.uint64(21)) & np.uint64(0x3FFFFFF)
                spidr = pico_words & np.uint64(0x1FFFFF)
                ts = (spidr << np.uint64(26)) | ts_lower

                tdc_time.append(ts.astype(np.float64) * PICOTDC_NS_PER_COUNT)
                tdc_type.append(np.full(len(pico_words), 0xC, dtype=np.uint8))
                tdc_channel.append(channel.astype(np.int16))
                tdc_edge.append(edge.astype(np.uint8))
                tdc_chip.append(np.full(len(pico_words), chip, dtype=np.uint8))

            for sub_value, (channel_id, edge_value) in FPGA_TDC_SUBTYPES.items():
                fw = tw[sub == np.uint64(sub_value)]
                if len(fw) == 0:
                    continue
                fine = (fw >> np.uint64(5)) & np.uint64(0xF)
                ok = (fine > np.uint64(0)) & (fine <= np.uint64(12))
                fw, fine = fw[ok], fine[ok]
                if len(fw) == 0:
                    continue
                coarse = (fw >> np.uint64(9)) & np.uint64((1 << 35) - 1)
                combined = coarse * np.uint64(12) + (fine - np.uint64(1))

                tdc_time.append(combined.astype(np.float64) * FPGA_TDC_NS_PER_COUNT)
                tdc_type.append(np.full(len(fw), sub_value, dtype=np.uint8))
                tdc_channel.append(np.full(len(fw), -1, dtype=np.int16))
                tdc_edge.append(np.full(len(fw), edge_value, dtype=np.uint8))
                tdc_chip.append(np.full(len(fw), chip, dtype=np.uint8))

    def _cat(parts, dtype):
        return np.concatenate(parts) if parts else np.array([], dtype=dtype)

    return {
        "pixel_x": _cat(px_x, np.uint16),
        "pixel_y": _cat(px_y, np.uint16),
        "pixel_toa_ns": _cat(px_toa, np.float64),
        "pixel_tot": _cat(px_tot, np.float64),
        "pixel_chip": _cat(px_chip, np.uint8),
        "tdc_time_ns": _cat(tdc_time, np.float64),
        "tdc_type": _cat(tdc_type, np.uint8),
        "tdc_channel": _cat(tdc_channel, np.int16),
        "tdc_edge": _cat(tdc_edge, np.uint8),
        "tdc_chip": _cat(tdc_chip, np.uint8),
    }


def convert_file(raw_path, h5_path, layout=None):
    """Decode one raw file and write its pixel hits and TDC markers
    straight to `h5_path` (see decode_raw_file for what's kept and the
    dataset names). Returns counts for display."""
    start = time.perf_counter()
    if layout is None:
        layout = chip_layout.get_active_layout()
    data = decode_raw_file(str(raw_path), layout)

    h5_path = pathlib.Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("x", data=data["pixel_x"])
        f.create_dataset("y", data=data["pixel_y"])
        f.create_dataset("toa", data=data["pixel_toa_ns"])
        f.create_dataset("tot", data=data["pixel_tot"])
        f.create_dataset("pixel_chip", data=data["pixel_chip"])
        f.create_dataset("tdc_time", data=data["tdc_time_ns"])
        f.create_dataset("tdc_type", data=data["tdc_type"])
        f.create_dataset("tdc_channel", data=data["tdc_channel"])
        f.create_dataset("tdc_edge", data=data["tdc_edge"])
        f.create_dataset("tdc_chip", data=data["tdc_chip"])
        f.attrs["Source Raw File"] = str(raw_path)
        f.attrs["Chip Layout"] = json.dumps(layout.to_dict())
        f.attrs["Conversion Time (s)"] = time.perf_counter() - start

    return {
        "pixels": len(data["pixel_x"]),
        "tdcs": len(data["tdc_time_ns"]),
    }
