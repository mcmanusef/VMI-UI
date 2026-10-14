"""Where each Timepix3 chip sits on the detector canvas, and which way it
faces, so hits from a multi-chip (e.g. quad) detector land in one image.

A raw .tpx3 chunk header names the chip stream its pixel hits came from
(byte 4, 0-3 on a quad); each hit's own address is that chip's local
(x, y), 0-255. Serval reports how those chips tile its image via GET
/detector/layout (ASIServer manual 4.5.6): per chip, the canvas offset of
its tile (X, Y) and an Orientation string naming which way the chip's own
x axis, then its y axis, run across the canvas -- LtR/RtL (left to right /
right to left), BtT/TtB (bottom to top / top to bottom). E.g. "LtRBtT" is
the chip as-is and "RtLTtB" is it rotated 180 degrees; the transposed
forms ("BtTLtR", ...) put the chip's x axis vertical.

The canvas follows the chips' own convention: x to the right and y upwards
from (0, 0), so a single chip at (0, 0) facing "LtRBtT" leaves coordinates
untouched. Checked against f107_000000.tpx3 (a quad): reading the layout
this way joins the four chips' hits into one continuous round detector
image, while any other reading leaves it split at the chip edges.
"""
import dataclasses
from typing import List

import numpy as np

import app_settings

CHIP_SIZE = 256

ORIENTATIONS = ("LtRBtT", "RtLBtT", "LtRTtB", "RtLTtB", "BtTLtR", "TtBLtR", "BtTRtL", "TtBRtL")

_SETTINGS_KEY = "chip_layout"


@dataclasses.dataclass
class ChipPlacement:
    chip: int
    x: int
    y: int
    orientation: str = "LtRBtT"

    def to_dict(self):
        return {"Chip": self.chip, "X": self.x, "Y": self.y, "Orientation": self.orientation}

    @classmethod
    def from_dict(cls, d):
        orientation = d.get("Orientation", "LtRBtT")
        if orientation not in ORIENTATIONS:
            raise ValueError(f"Unknown chip orientation {orientation!r}")
        return cls(chip=int(d["Chip"]), x=int(d["X"]), y=int(d["Y"]), orientation=orientation)

    def local_to_tile(self, x, y):
        """This chip's own (x, y) -> position inside its 256x256 tile."""
        last = CHIP_SIZE - 1
        x_axis, y_axis = self.orientation[:3], self.orientation[3:]
        # Each axis' token says which canvas direction it runs in.
        along = {"LtR": lambda v: ("x", v), "RtL": lambda v: ("x", last - v),
                 "BtT": lambda v: ("y", v), "TtB": lambda v: ("y", last - v)}
        out = dict([along[x_axis](x), along[y_axis](y)])
        return out["x"], out["y"]

    def tile_to_local(self, tx, ty):
        """Inverse of local_to_tile."""
        last = CHIP_SIZE - 1
        x_axis, y_axis = self.orientation[:3], self.orientation[3:]

        def read(token):
            if token == "LtR":
                return tx
            if token == "RtL":
                return last - tx
            if token == "BtT":
                return ty
            return last - ty

        return read(x_axis), read(y_axis)


@dataclasses.dataclass
class ChipLayout:
    width: int
    height: int
    chips: List[ChipPlacement]

    @property
    def size(self):
        """Side of the square image every pixel/cluster plot should span."""
        return max(self.width, self.height)

    def placement(self, chip):
        for p in self.chips:
            if p.chip == chip:
                return p
        return None

    def to_global(self, chip, x, y):
        """Chip-local pixel coordinates (scalars or arrays, all from the one
        `chip`) -> canvas coordinates. A chip missing from the layout is
        left where it is, so a single-chip file still decodes on a layout
        that doesn't list it."""
        p = self.placement(chip)
        if p is None:
            return x, y
        tx, ty = p.local_to_tile(x, y)
        return tx + p.x, ty + p.y

    def to_local(self, gx, gy):
        """Canvas pixel -> (chip, local x, local y), or None if no chip
        covers it -- for addressing Serval's per-chip pixel masks."""
        for p in self.chips:
            tx, ty = gx - p.x, gy - p.y
            if 0 <= tx < CHIP_SIZE and 0 <= ty < CHIP_SIZE:
                lx, ly = p.tile_to_local(tx, ty)
                return p.chip, int(lx), int(ly)
        return None

    def to_dict(self):
        return {"Width": self.width, "Height": self.height, "Chips": [p.to_dict() for p in self.chips]}

    @classmethod
    def from_dict(cls, d):
        return cls(
            width=int(d["Width"]), height=int(d["Height"]),
            chips=[ChipPlacement.from_dict(c) for c in d["Chips"]],
        )

    @classmethod
    def from_serval(cls, layout_json):
        """A GET /detector/layout response. Uses the "Rotated" layout --
        the one Serval itself draws images with, after any
        DetectorOrientation rotation/flip set in Serval."""
        return cls.from_dict(layout_json.get("Rotated") or layout_json["Original"])

    def describe(self):
        lines = [f"{self.width} x {self.height} px, {len(self.chips)} chip(s)"]
        for p in sorted(self.chips, key=lambda p: p.chip):
            lines.append(f"  chip {p.chip}: tile at ({p.x}, {p.y}), {p.orientation}")
        return "\n".join(lines)


def single_chip_layout():
    return ChipLayout(CHIP_SIZE, CHIP_SIZE, [ChipPlacement(0, 0, 0, "LtRBtT")])


def default_layout():
    """This lab's quad as its Serval reports it (GET /detector/layout,
    2026-10-14) -- the out-of-the-box default until the Hardware tab
    fetches a fresh one."""
    return ChipLayout(512, 512, [
        ChipPlacement(0, 256, 0, "LtRBtT"),
        ChipPlacement(1, 256, 256, "RtLTtB"),
        ChipPlacement(2, 0, 256, "RtLTtB"),
        ChipPlacement(3, 0, 0, "LtRBtT"),
    ])


def get_active_layout():
    """The layout every decode should use right now: whatever the "chip
    layout" Hardware tab last saved, else default_layout()."""
    saved = app_settings.get(_SETTINGS_KEY)
    if saved:
        try:
            return ChipLayout.from_dict(saved)
        except (KeyError, TypeError, ValueError):
            pass
    return default_layout()


def set_active_layout(layout):
    app_settings.set(_SETTINGS_KEY, layout.to_dict())


def fetch_from_serval(server, timeout=5):
    import serval_client

    response = serval_client.SESSION.get(f"{server}/detector/layout", timeout=timeout)
    response.raise_for_status()
    return ChipLayout.from_serval(response.json())


def apply_layout(layout, chips, x, y):
    """Vectorized to_global over arrays of mixed chips."""
    chips = np.asarray(chips)
    gx = np.array(x, dtype=np.int64, copy=True)
    gy = np.array(y, dtype=np.int64, copy=True)
    for chip in np.unique(chips):
        sel = chips == chip
        gx[sel], gy[sel] = layout.to_global(int(chip), gx[sel], gy[sel])
    return gx, gy

