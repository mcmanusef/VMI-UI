# TPX3 File Structure and Decoding (UConn Pipelines)

This document describes how `.tpx3` files are laid out and how the
current pipelines decode them. The description is derived from:

- Examples/uconn_vmi/uconn_pipelines.py
- Examples/uconn_vmi/uconn_processes.py
- vmi_analysis/processing/processes/source_processes.py
- vmi_analysis/processing/processes/analysis_processes.py
- vmi_analysis/processing/tpx_conversion.py

## File Layout (Chunked Stream)

A `.tpx3` file is treated as a stream of repeated chunks. Each chunk has:

1. Chunk header (8 bytes)
2. Payload (num_bytes bytes), interpreted as an array of 64-bit packets

### Chunk Header (8 bytes)

The reader extracts only a few fields from the 8-byte header:

- bytes[0:4]: ignored by the pipelines (reserved/unused)
- byte[4]: chip number (read but unused)
- byte[5]: mode (read but unused)
- bytes[6:8]: payload length in bytes (little-endian)

The payload length is expected to be a multiple of 8. Each 8-byte word is
decoded as a packet.

### Payload Interpretation

The payload is interpreted as little-endian `int64` values, then shifted:

- `packet = int64_word - 2**62`

This shift is required by the current decoder to recover the correct
packet headers. If you implement a new decoder, apply the same shift to
match the pipeline behavior.

## Packet Decoding

Each packet is a 64-bit word with a 4-bit type header:

- `header = packet >> 60`
- `reduced = packet & 0x0FFF_FFFF_FFFF_FFFF` (lower 60 bits)

Only two header values are used:

- `0x7`: pixel packet
- `0x2`: TDC packet

All other headers are ignored.

## Pixel Packet (header 0x7)

The lower 60 bits are split as follows (bit 0 is the LSB of `reduced`):

- bits 0-15: `c_time` (coarse time)
- bits 16-19: `f_time` (fine time)
- bits 20-29: `tot` (time-over-threshold)
- bits 30-43: `m_time` (medium time)
- bits 44-60: `pix_addr` (pixel address)

The time-of-arrival (ToA) is computed as:

- `toa_raw = c_time * 2**18 + m_time * 2**4 - f_time`
- `toa_ns = toa_raw * (25 / 16)`

Notes:

- `25/16 ns` is the pixel time resolution (`PIXEL_RES`).
- `tot` is kept in raw units, which correspond to 25 ns ticks.

Pixel address is mapped to coordinates using:

- `dcol = (pix_addr & 0xFE00) >> 8`
- `spix = (pix_addr & 0x01F8) >> 1`
- `pix = pix_addr & 0x0007`
- `x = dcol + (pix // 4)`
- `y = spix + (pix & 0x3)`

This yields `x, y` pixel indices (0-255 range in practice).

## TDC Packet (header 0x2)

The lower 60 bits are split as follows (bit 0 is the LSB of `reduced`):

- bits 0-4: unused by this decoder
- bits 5-8: `f_time`
- bits 9-43: `c_time`
- bits 44-55: unused by this decoder
- bits 56-59: `tdc_type`

The decoder also masks coarse time:

- `c_time = c_time & 0x1FFFFFFFF` (remove two MSBs to align wrap behavior)

Then it adjusts the fine time and computes a nanosecond timestamp:

- `f_time = f_time - 1`
- `tdc_time_ns = 3.125 * c_time + 0.260 * f_time`

### TDC Type Mapping

Raw `tdc_type` values are mapped in two different ways:

1. Generic TPX conversion (`TPXConverter` pipeline):

   - 10 -> 1 (TDC1 rising)
   - 15 -> 2 (TDC1 falling)
   - 14 -> 3 (TDC2 rising)
   - 11 -> 4 (TDC2 falling)

2. UConn VMI conversion (`VMIConverter`):

   - 15 is treated as the start of a TDC1 pulse
   - 10 is treated as the end of that pulse
   - 14 is treated as an electron ToF marker

The UConn pipeline does not remap the types to 1-4; it uses the raw codes.

## UConn VMI Decode Logic

The UConn pipelines decode a chunk into pixels and TDCs, then apply
experiment-specific logic:

1. `process_chunk`:
   - Calls `process_packet` on each 64-bit word.
   - Emits pixel tuples `(toa_ns, x, y, tot)` and raw TDC tuples.

2. Optional pixel corrections:
   - `apply_timewalk`: subtracts a ToA offset based on ToT.
   - `toa_correction`: subtracts a fixed ToA offset for a specific x-range.

3. TDC sorting (`sort_tdcs`):
   - Track TDC1F (type 15) as `start_time`.
   - When a TDC1R (type 10) arrives, compute:
     `pulse_len = tdc_time - start_time`.
   - If `pulse_len > cutoff` (default 300 ns), classify as a laser pulse.
   - Else, classify as an ion ToF marker.
   - TDC2R (type 14) is treated as an electron ToF marker.

This produces three streams:

- `pulses`: laser pulse timestamps
- `etof`: electron time-of-flight timestamps
- `itof`: ion time-of-flight timestamps

## Time Wrapping and Monotonic Queues

Pixel and TDC timestamps can wrap around in hardware. The pipelines use
`StructuredDataQueue` and `MonotonicQueue` with:

- `PERIOD = 25 * 2**30` (ns)

When `force_monotone=True`, the queue unwraps times by adding `PERIOD`
whenever a new time would go backwards by more than `max_back`.

## Minimal Decoding Recipe

The following steps mirror the pipeline behavior:

1. Read 8 bytes for a chunk header.
2. Parse payload length from bytes[6:8] (little-endian).
3. Read `num_bytes` of payload.
4. Interpret payload as `int64` words and subtract `2**62`.
5. For each word:
   - `header = word >> 60`
   - `header == 0x7`: decode pixel packet
   - `header == 0x2`: decode TDC packet
6. Convert raw fields to:
   - `PixelData(time=toa_ns, x, y, tot)`
   - `TDCData(time=tdc_time_ns, type=tdc_type)` or use VMI sorting

If you need the UConn VMI outputs, apply the `sort_tdcs` logic to split
TDCs into `pulses`, `etof`, and `itof`, and optionally apply the
timewalk/ToA corrections described above.
