"""Which physical TDC line feeds each of the pulse-trigger, e-ToF and i-ToF
logical signals used throughout collection and analysis, and how multiple
signals sharing one physical line are told apart by pulse width.

This hardware exposes two TDC families (see raw_to_h5.py /
tpx_to_h5_diagnostics.ipynb for how this was reverse-engineered from the raw
packet format):
  PicoTDC  -- up to 64 channels (header 0x6, sub-nibble 0xC)
  FPGA TDC -- 2 channels, each with a rising and falling edge
              (header 0x6, sub-nibble 0xf/0xa/0xe/0xb)

A SignalMapping names one TdcSource (family + channel) and, optionally, a
pulse-width gate: with no gate, every hit on that line (matching time_edge)
is taken directly as an event; with a gate, rising edges are paired with
the next falling edge on the same line and kept only if the resulting
width falls in [gate_min_ns, gate_max_ns] -- this is how e.g. a long
trigger pulse and a short i-ToF pulse can share one physical FPGA TDC
channel and still be told apart. See tpx_processing.classify_tdcs.

A SignalMapping can also set a ringing holdoff: a pulse (or, ungated, an
edge) that starts less than ring_holdoff_ns after the previous pulse on the
same line ended is taken to be that pulse ringing back across threshold and
is dropped -- before the width gate, so ringing after a long trigger pulse
can't pass for a short i-ToF pulse sharing its line.

Each signal can also be dithered: a uniform random offset in
[0, dither_ns) added to every time it yields, to smear out the TDC's
quantization (the FPGA TDC's step is ~0.260 ns, the PicoTDC's ~3 ps). On
by default for e-ToF and i-ToF, off for the pulse trigger; on a PicoTDC
line the width is capped at one PicoTDC step.

The mapping also holds the pixel clock offset: the Timepix pixel clock runs
ahead of the TDC, so a pixel hit is time-stamped ~180 ns before the e-ToF
edge of the same electron. Electrons arriving shortly before the next
trigger (as they do on this setup) would otherwise be put in the previous
pulse, at the very end of it, with the walked low-ToT hits of the same
electrons spilling past the trigger into the next one -- breaking the
time-walk fits and the cluster / e-ToF pairing. pixel_offset_ns is added to
every pixel hit time before it is assigned to a pulse (see
tpx_processing.build_frame_events).
"""
import dataclasses
import json
import pathlib
from typing import Optional

import app_settings

FAMILY_PICOTDC = "picotdc"
FAMILY_FPGA_TDC = "fpga_tdc"
FAMILIES = (FAMILY_PICOTDC, FAMILY_FPGA_TDC)

PICOTDC_CHANNELS = tuple(range(64))
FPGA_TDC_CHANNELS = (1, 2)

# PicoTDC: 1 count = 25 ns / 8192 (~3.0518 ps). FPGA TDC's combined
# (coarse * 12 + fine - 1) counter: 1 count = 3.125 ns / 12 (~260.42 ps).
PICOTDC_NS_PER_COUNT = 25.0 / 8192
FPGA_TDC_NS_PER_COUNT = 3.125 / 12

# FPGA TDC sub-nibble (bits [59:56]) -> (channel, edge). edge: 1 = rising, 0 = falling.
FPGA_TDC_SUBTYPES = {
    0xF: (1, 1),
    0xA: (1, 0),
    0xE: (2, 1),
    0xB: (2, 0),
}

# Default dither width: about one FPGA TDC step.
DEFAULT_DITHER_NS = 0.26

# Added to every pixel hit time before it is put in a pulse (see the module
# docstring). Measured on the 2026-11 data: pixel hits lead the e-ToF by
# ~180 ns, so 200 ns lines cluster times up with the e-ToF times.
DEFAULT_PIXEL_OFFSET_NS = 200.0
PIXEL_OFFSET_KEY = "tdc_mapping.pixel_offset_ns"
# The cv4 attribute recording the offset a file was converted with.
PIXEL_OFFSET_ATTR = "Pixel Clock Offset (ns)"

_SIGNAL_NAMES = ("pulse_trigger", "etof", "itof")
# Signals the user may switch off; the pulse trigger is always required.
OPTIONAL_SIGNALS = ("etof", "itof")


@dataclasses.dataclass
class TdcSource:
    family: str
    channel: int

    def to_dict(self):
        return {"family": self.family, "channel": self.channel}

    @classmethod
    def from_dict(cls, d):
        return cls(family=d["family"], channel=int(d["channel"]))


@dataclasses.dataclass
class SignalMapping:
    source: TdcSource
    time_edge: str = "rising"
    gate_min_ns: Optional[float] = None
    gate_max_ns: Optional[float] = None
    # False = this signal isn't recorded at all (classify_tdcs returns no
    # hits for it). Only e-ToF / i-ToF can be disabled -- see OPTIONAL_SIGNALS.
    enabled: bool = True
    # Drop pulses starting less than this long after the previous pulse on
    # the line ended (ringing). None = keep everything.
    ring_holdoff_ns: Optional[float] = None
    # Add a uniform random offset in [0, dither_ns) to every time.
    dither: bool = True
    dither_ns: float = DEFAULT_DITHER_NS

    @property
    def effective_dither_ns(self):
        """The dither width actually applied (0 when dithering is off).
        A PicoTDC line is dithered by at most its own step (~3 ps): the
        default width is one FPGA TDC step (0.26 ns), which on a PicoTDC
        line adds ~75 ps rms of noise and removes no quantization."""
        if not (self.dither and self.dither_ns > 0):
            return 0.0
        if self.source.family == FAMILY_PICOTDC:
            return min(self.dither_ns, PICOTDC_NS_PER_COUNT)
        return self.dither_ns

    def to_dict(self):
        return {
            "source": self.source.to_dict(),
            "time_edge": self.time_edge,
            "gate_min_ns": self.gate_min_ns,
            "gate_max_ns": self.gate_max_ns,
            "enabled": self.enabled,
            "ring_holdoff_ns": self.ring_holdoff_ns,
            "dither": self.dither,
            "dither_ns": self.dither_ns,
        }

    @classmethod
    def from_dict(cls, d, default=None):
        """`default` (a SignalMapping) fills fields older presets lack."""
        default = default or cls(source=TdcSource.from_dict(d["source"]))
        return cls(
            source=TdcSource.from_dict(d["source"]),
            time_edge=d.get("time_edge", "rising"),
            gate_min_ns=d.get("gate_min_ns"),
            gate_max_ns=d.get("gate_max_ns"),
            enabled=bool(d.get("enabled", True)),
            ring_holdoff_ns=d.get("ring_holdoff_ns"),
            dither=bool(d.get("dither", default.dither)),
            dither_ns=parse_dither_ns(d.get("dither_ns"), default.dither_ns),
        )


@dataclasses.dataclass
class TdcMapping:
    pulse_trigger: SignalMapping
    etof: SignalMapping
    itof: SignalMapping
    # Added to pixel hit times before pulse assignment (module docstring).
    pixel_offset_ns: float = DEFAULT_PIXEL_OFFSET_NS

    def to_dict(self):
        data = {name: getattr(self, name).to_dict() for name in _SIGNAL_NAMES}
        data["pixel_offset_ns"] = self.pixel_offset_ns
        return data

    @classmethod
    def from_dict(cls, d):
        defaults = default_mapping()
        signals = {name: SignalMapping.from_dict(d[name], getattr(defaults, name)) for name in _SIGNAL_NAMES}
        return cls(**signals, pixel_offset_ns=parse_pixel_offset_ns(d.get("pixel_offset_ns")))

    def save(self, path):
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


def default_mapping():
    """The historical, hardcoded FPGA-TDC-only scheme tpx_processing.sort_tdcs
    used to apply unconditionally (tdc_type 15/14/10, 300 ns cutoff), as a
    TdcMapping -- the out-of-the-box default so nothing changes until a user
    edits the "tdc mapping" Hardware tab."""
    return TdcMapping(
        pulse_trigger=SignalMapping(TdcSource(FAMILY_FPGA_TDC, 1), "rising", 300.0, None, dither=False),
        etof=SignalMapping(TdcSource(FAMILY_FPGA_TDC, 2), "rising", None, None),
        itof=SignalMapping(TdcSource(FAMILY_FPGA_TDC, 1), "rising", 0.0, 300.0),
    )


def parse_optional_float(value):
    """A gate bound as stored by the Hardware tab's entries (text, "" =
    no bound) or as a number/None -> float or None."""
    if value is None or isinstance(value, (int, float)):
        return None if value is None else float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_dither_ns(value, default=DEFAULT_DITHER_NS):
    """A dither width as stored by the Hardware tab's entry (text) or as a
    number -> float >= 0, or `default` if blank/invalid/negative."""
    width = parse_optional_float(value)
    return default if width is None or width < 0 else width


def parse_pixel_offset_ns(value, default=DEFAULT_PIXEL_OFFSET_NS):
    """A pixel clock offset as stored by the Hardware tab's entry (text) or
    as a number -> float (may be negative), or `default` if blank/invalid."""
    offset = parse_optional_float(value)
    return default if offset is None else offset


def _settings_key(signal_name, field):
    return f"tdc_mapping.{signal_name}.{field}"


# ---- pulse pairing ------------------------------------------------------------
#
# Which pulse clusters, e-ToF hits and i-ToF hits are taken from when they
# are paired with each other: -1 the previous pulse, 0 their own, +1 the
# next. Anything but 0 pairs them across pulses, which gives only random
# coincidences -- the false-coincidence background. Applied to what is shown
# and analysed (the live x / y vs e-ToF maps, the Coincidence tab), never to
# saved files, which always keep every hit's own pulse.

PULSE_OFFSET_SPECIES = (("clusters", "Clusters"), ("etof", "e-ToF"), ("tof", "i-ToF"))
PULSE_OFFSET_LABELS = {-1: "Previous pulse", 0: "Same pulse", 1: "Next pulse"}
_PULSE_OFFSET_BY_LABEL = {label: offset for offset, label in PULSE_OFFSET_LABELS.items()}


def pulse_offset_key(species):
    return f"tdc_mapping.pulse_offset.{species}"


def get_pulse_offsets():
    """{"clusters", "etof", "tof"} -> -1, 0 or +1, as set on the Hardware
    tab's TDC mapping page (stored as its labels)."""
    return {
        species: _PULSE_OFFSET_BY_LABEL.get(app_settings.get(pulse_offset_key(species), PULSE_OFFSET_LABELS[0]), 0)
        for species, _ in PULSE_OFFSET_SPECIES
    }


def describe_pulse_offsets(offsets):
    """e.g. "i-ToF from the next pulse", or "" when every offset is 0."""
    names = dict(PULSE_OFFSET_SPECIES)
    return ", ".join(
        f"{names[species]} from the {PULSE_OFFSET_LABELS[offset].lower()}"
        for species, offset in (offsets or {}).items() if offset
    )


def get_active_mapping():
    """The mapping every collection/analysis call site should use right
    now: whatever the "tdc mapping" Hardware tab's widgets last wrote to
    app_settings (which doubles as "the current default"), falling back to
    default_mapping() for any field never touched."""
    defaults = default_mapping()
    signals = {}
    for name in _SIGNAL_NAMES:
        default_signal = getattr(defaults, name)
        signals[name] = SignalMapping(
            source=TdcSource(
                family=app_settings.get(_settings_key(name, "family"), default_signal.source.family),
                channel=int(app_settings.get(_settings_key(name, "channel"), default_signal.source.channel)),
            ),
            time_edge=app_settings.get(_settings_key(name, "time_edge"), default_signal.time_edge),
            gate_min_ns=parse_optional_float(
                app_settings.get(_settings_key(name, "gate_min_ns"), default_signal.gate_min_ns)
            ),
            gate_max_ns=parse_optional_float(
                app_settings.get(_settings_key(name, "gate_max_ns"), default_signal.gate_max_ns)
            ),
            enabled=name not in OPTIONAL_SIGNALS
            or bool(app_settings.get(_settings_key(name, "enabled"), default_signal.enabled)),
            ring_holdoff_ns=parse_optional_float(
                app_settings.get(_settings_key(name, "ring_holdoff_ns"), default_signal.ring_holdoff_ns)
            ),
            dither=bool(app_settings.get(_settings_key(name, "dither"), default_signal.dither)),
            dither_ns=parse_dither_ns(
                app_settings.get(_settings_key(name, "dither_ns"), default_signal.dither_ns), default_signal.dither_ns
            ),
        )
    return TdcMapping(**signals, pixel_offset_ns=parse_pixel_offset_ns(app_settings.get(PIXEL_OFFSET_KEY)))
