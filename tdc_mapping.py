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

_SIGNAL_NAMES = ("pulse_trigger", "etof", "itof")


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

    def to_dict(self):
        return {
            "source": self.source.to_dict(),
            "time_edge": self.time_edge,
            "gate_min_ns": self.gate_min_ns,
            "gate_max_ns": self.gate_max_ns,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            source=TdcSource.from_dict(d["source"]),
            time_edge=d.get("time_edge", "rising"),
            gate_min_ns=d.get("gate_min_ns"),
            gate_max_ns=d.get("gate_max_ns"),
        )


@dataclasses.dataclass
class TdcMapping:
    pulse_trigger: SignalMapping
    etof: SignalMapping
    itof: SignalMapping

    def to_dict(self):
        return {name: getattr(self, name).to_dict() for name in _SIGNAL_NAMES}

    @classmethod
    def from_dict(cls, d):
        return cls(**{name: SignalMapping.from_dict(d[name]) for name in _SIGNAL_NAMES})

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
        pulse_trigger=SignalMapping(TdcSource(FAMILY_FPGA_TDC, 1), "rising", 300.0, None),
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


def _settings_key(signal_name, field):
    return f"tdc_mapping.{signal_name}.{field}"


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
        )
    return TdcMapping(**signals)
