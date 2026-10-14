"""Hardware tab: where each Timepix3 chip sits on the detector image and
which way it faces (see chip_layout.py). Every decode -- Monitored
Acquisition, Analysis > conversion, "tpx to h5", Diagnostics, Timewalk
Calibration, Parameter Sweep -- places pixel hits with the layout saved
here, via chip_layout.get_active_layout().
"""
import qtk as tk
from qtk import ttk, messagebox

import chip_layout
import ui_style


class ChipLayoutInterface(ttk.Frame):
    """Pure configuration -- the layout is saved the moment it changes."""

    def __init__(self, parent, server_var=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._server_var = server_var
        self._summary_var = tk.StringVar(self)
        self._build_ui()
        self._show(chip_layout.get_active_layout())

    def _build_ui(self):
        page = ui_style.build_page_layout(self)

        ui_style.add_note(
            page, 0,
            "How the chips of a multi-chip (e.g. quad) detector tile one image: each chip's tile "
            "position and orientation, as Serval reports them. Fetch it from Serval whenever the "
            "detector or Serval's rotate/flip setting changes. Pixel and cluster maps span the "
            "whole image, so set their range to 0 to the image size. Files already converted "
            "keep the layout they were converted with.",
        )

        section = ui_style.build_section(page, 1, "Current Layout")
        ttk.Label(section, textvariable=self._summary_var, justify="left", font=("TkFixedFont", 9)).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=4
        )
        ui_style.build_button_bar(section, 1, [
            ("Fetch from Serval", self._fetch),
            ("Single Chip", lambda: self._apply(chip_layout.single_chip_layout())),
            ("Restore Built-in Default", lambda: self._apply(chip_layout.default_layout())),
        ], columnspan=2)

    def _server_url(self):
        url = self._server_var.get().strip() if self._server_var is not None else ""
        url = url or "http://localhost:8080"
        return url if url.startswith(("http://", "https://")) else f"http://{url}"

    def _fetch(self):
        try:
            layout = chip_layout.fetch_from_serval(self._server_url())
        except Exception as exc:
            messagebox.showerror("Fetch failed", f"Could not read the layout from Serval: {exc}", parent=self)
            return
        self._apply(layout)

    def _apply(self, layout):
        chip_layout.set_active_layout(layout)
        self._show(layout)

    def _show(self, layout):
        self._summary_var.set(layout.describe())
