"""Hardware tab: lets the user say which physical TDC line feeds each of
the pulse-trigger, e-ToF and i-ToF logical signals, and how to tell apart
multiple signals sharing one line by pulse width. See tdc_mapping.py for
the underlying schema and tpx_processing.classify_tdcs for how it's
applied -- every place that classifies raw TDC hits (Monitored Acquisition,
Analysis > conversion, Diagnostics, Timewalk Calibration, Parameter Sweep)
reads the same mapping, live, via tdc_mapping.get_active_mapping(). The
"tpx to h5" raw dump is unaffected -- it already keeps every hit's channel
and edge without classifying them.
"""
import qtk as tk
from qtk import ttk, filedialog, messagebox

import app_settings
import tdc_mapping
import ui_style

_SIGNALS = (
    ("pulse_trigger", "Pulse Trigger"),
    ("etof", "e-ToF"),
    ("itof", "i-ToF"),
)

_FAMILY_LABELS = {
    tdc_mapping.FAMILY_PICOTDC: "PicoTDC",
    tdc_mapping.FAMILY_FPGA_TDC: "FPGA TDC",
}
_FAMILY_BY_LABEL = {label: family for family, label in _FAMILY_LABELS.items()}

_EDGE_LABELS = {"rising": "Rising", "falling": "Falling"}
_EDGE_BY_LABEL = {label: edge for edge, label in _EDGE_LABELS.items()}


def _channels_for(family):
    channels = tdc_mapping.PICOTDC_CHANNELS if family == tdc_mapping.FAMILY_PICOTDC else tdc_mapping.FPGA_TDC_CHANNELS
    return [str(c) for c in channels]


class TdcMappingInterface(ttk.Frame):
    """Pure configuration -- no worker thread, every edit is already "the
    default" (persisted immediately, same as every other persistent_var-
    backed field in this app)."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._vars = {}
        self._channel_boxes = {}
        self._building = True
        self._build_ui()
        self._building = False

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        page = ui_style.build_page_layout(self)

        ui_style.add_note(
            page, 0,
            "Which physical TDC line (PicoTDC channel 0-63, or FPGA TDC channel 1/2) feeds each "
            "logical signal, and how multiple signals sharing one line are told apart by pulse "
            "width. Uncheck e-ToF or i-ToF if that signal isn't connected, so no hits are recorded "
            "for it. Leave a gate blank to take every hit on that line directly. With a gate set, "
            "each rising edge is paired with the next falling edge on the same line and kept only "
            "if that pulse's width falls in range -- e.g. a long trigger pulse and a short i-ToF "
            "pulse can share one FPGA TDC channel this way. Set a ringing holdoff to drop pulses "
            "that start within that many ns after the previous pulse on the same line ended -- a "
            "signal ringing back across threshold -- before the gate is applied; leave it blank "
            "to keep every pulse. Dither adds a random offset, uniform between 0 and the width "
            "given, to every time on that signal to smooth out the TDC's time steps (FPGA TDC: "
            "~0.260 ns, PicoTDC: ~0.003 ns); it's on for e-ToF and i-ToF by default and off for "
            "the pulse trigger. These settings apply everywhere "
            "(Monitored Acquisition, Analysis > conversion, Diagnostics, Timewalk Calibration, "
            "Parameter Sweep) and are saved as you edit them -- the \"tpx to h5\" raw dump is "
            "unaffected, since it already keeps every hit's own channel and edge.",
        )

        row = 1
        for name, title in _SIGNALS:
            section = ui_style.build_section(page, row, title)
            self._build_signal_section(section, name)
            row += 1

        pixel_timing = ui_style.build_section(page, row, "Pixel timing")
        self._build_pixel_timing_section(pixel_timing)
        row += 1

        pairing = ui_style.build_section(page, row, "Pulse pairing")
        self._build_pairing_section(pairing)
        row += 1

        presets = ui_style.build_section(page, row, "Presets")
        ui_style.build_button_bar(presets, 0, [
            ("Save Preset...", self._save_preset),
            ("Load Preset...", self._load_preset),
            ("Restore Built-in Default", self._restore_default),
        ])

    def _build_signal_section(self, section, name):
        default = getattr(tdc_mapping.default_mapping(), name)

        family_var = app_settings.persistent_var(
            self, tk.StringVar, f"tdc_mapping.{name}.family", default.source.family
        )
        channel_var = app_settings.persistent_var(
            self, tk.StringVar, f"tdc_mapping.{name}.channel", str(default.source.channel)
        )
        edge_var = app_settings.persistent_var(
            self, tk.StringVar, f"tdc_mapping.{name}.time_edge", default.time_edge
        )
        gate_min_var = app_settings.persistent_var(
            self, tk.StringVar, f"tdc_mapping.{name}.gate_min_ns",
            "" if default.gate_min_ns is None else str(default.gate_min_ns),
        )
        gate_max_var = app_settings.persistent_var(
            self, tk.StringVar, f"tdc_mapping.{name}.gate_max_ns",
            "" if default.gate_max_ns is None else str(default.gate_max_ns),
        )
        ring_holdoff_var = app_settings.persistent_var(
            self, tk.StringVar, f"tdc_mapping.{name}.ring_holdoff_ns",
            "" if default.ring_holdoff_ns is None else str(default.ring_holdoff_ns),
        )
        dither_var = app_settings.persistent_var(
            self, tk.BooleanVar, f"tdc_mapping.{name}.dither", default.dither
        )
        dither_ns_var = app_settings.persistent_var(
            self, tk.StringVar, f"tdc_mapping.{name}.dither_ns", f"{default.dither_ns:g}"
        )
        self._vars[name] = {
            "family": family_var, "channel": channel_var, "time_edge": edge_var,
            "gate_min_ns": gate_min_var, "gate_max_ns": gate_max_var,
            "ring_holdoff_ns": ring_holdoff_var,
            "dither": dither_var, "dither_ns": dither_ns_var,
        }

        row = 0
        if name in tdc_mapping.OPTIONAL_SIGNALS:
            enabled_var = app_settings.persistent_var(
                self, tk.BooleanVar, f"tdc_mapping.{name}.enabled", default.enabled
            )
            self._vars[name]["enabled"] = enabled_var
            ttk.Checkbutton(
                section, text="Record this signal (uncheck if it isn't connected)", variable=enabled_var,
            ).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
            row += 1

        family_display_var = tk.StringVar(self, value=_FAMILY_LABELS[family_var.get()])
        family_box = ttk.Combobox(
            section, textvariable=family_display_var, state="readonly",
            values=list(_FAMILY_LABELS.values()),
        )
        ui_style.add_form_row(section, row, "TDC family:", family_box)

        channel_box = ttk.Combobox(
            section, textvariable=channel_var, state="readonly", values=_channels_for(family_var.get()),
        )
        ui_style.add_form_row(section, row + 1, "Channel:", channel_box)
        self._channel_boxes[name] = channel_box

        def _on_family_change(*_args, family_var=family_var, channel_var=channel_var, channel_box=channel_box):
            family = _FAMILY_BY_LABEL[family_display_var.get()]
            family_var.set(family)
            values = _channels_for(family)
            channel_box.configure(values=values)
            if channel_var.get() not in values:
                channel_var.set(values[0])

        family_display_var.trace_add("write", _on_family_change)

        edge_display_var = tk.StringVar(self, value=_EDGE_LABELS[edge_var.get()])
        edge_box = ttk.Combobox(
            section, textvariable=edge_display_var, state="readonly",
            values=list(_EDGE_LABELS.values()),
        )
        ui_style.add_form_row(section, row + 2, "Event edge:", edge_box)
        edge_display_var.trace_add(
            "write", lambda *_a, v=edge_display_var, ev=edge_var: ev.set(_EDGE_BY_LABEL[v.get()])
        )

        gate_row = ttk.Frame(section)
        gate_row.columnconfigure(0, weight=1)
        gate_row.columnconfigure(2, weight=1)
        ttk.Entry(gate_row, textvariable=gate_min_var).grid(row=0, column=0, sticky="ew")
        ttk.Label(gate_row, text=" to ").grid(row=0, column=1)
        ttk.Entry(gate_row, textvariable=gate_max_var).grid(row=0, column=2, sticky="ew")
        ui_style.add_form_row(section, row + 3, "Pulse-width gate (ns):", gate_row)

        ui_style.add_form_row(
            section, row + 4, "Ringing holdoff (ns):", ttk.Entry(section, textvariable=ring_holdoff_var),
        )

        dither_row = ttk.Frame(section)
        dither_row.columnconfigure(1, weight=1)
        ttk.Checkbutton(dither_row, text="On, width:", variable=dither_var).grid(row=0, column=0, sticky="w")
        ttk.Entry(dither_row, textvariable=dither_ns_var).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ui_style.add_form_row(section, row + 5, "Dither (ns):", dither_row)

    def _build_pixel_timing_section(self, section):
        self._pixel_offset_var = app_settings.persistent_var(
            self, tk.StringVar, tdc_mapping.PIXEL_OFFSET_KEY, f"{tdc_mapping.DEFAULT_PIXEL_OFFSET_NS:g}"
        )
        ui_style.add_form_row(
            section, 0, "Pixel clock offset (ns):", ttk.Entry(section, textvariable=self._pixel_offset_var),
        )
        ui_style.add_note(
            section, 1,
            "Added to every pixel hit time before it is put in a pulse. The Timepix pixel clock runs "
            "~180 ns ahead of the TDC, so without it electrons arriving just before a trigger land at the "
            "end of the previous pulse (t near one trigger period), split from their own e-ToF hit, with "
            "their low-ToT, walked hits spilling into the next pulse -- which breaks the time-walk fits "
            "and the cluster / e-ToF pairing. Cluster times shift by the same amount. Files converted with "
            "a different value keep the offset they were converted with.",
            pady=(4, 4),
        )

    def _build_pairing_section(self, section):
        labels = list(tdc_mapping.PULSE_OFFSET_LABELS.values())
        for row, (species, title) in enumerate(tdc_mapping.PULSE_OFFSET_SPECIES):
            var = app_settings.persistent_var(
                self, tk.StringVar, tdc_mapping.pulse_offset_key(species), tdc_mapping.PULSE_OFFSET_LABELS[0]
            )
            ui_style.add_form_row(section, row, f"{title} from:", ttk.Combobox(
                section, textvariable=var, values=labels, state="readonly", width=18,
            ))
        ui_style.add_note(
            section, len(tdc_mapping.PULSE_OFFSET_SPECIES),
            "Pair each kind of hit with the others from the previous or next pulse instead of its own. "
            "Pairing across pulses gives only random coincidences: the false-coincidence background. "
            "A neighbour more than half a trigger period away (first / last pulse, frame gaps) is left "
            "unpaired. Applies to what's shown and analysed -- Diagnostics' x / y vs e-ToF maps and the "
            "Coincidence tab (on its next Load) -- never to saved cv4 / calibrated files or calibration "
            "fits, which always keep every hit's own pulse.",
            pady=(4, 4),
        )

    # ---- presets ------------------------------------------------------------

    def _current_mapping(self):
        signals = {}
        for name, _title in _SIGNALS:
            v = self._vars[name]
            signals[name] = tdc_mapping.SignalMapping(
                source=tdc_mapping.TdcSource(family=v["family"].get(), channel=int(v["channel"].get())),
                time_edge=v["time_edge"].get(),
                gate_min_ns=tdc_mapping.parse_optional_float(v["gate_min_ns"].get()),
                gate_max_ns=tdc_mapping.parse_optional_float(v["gate_max_ns"].get()),
                enabled=v["enabled"].get() if "enabled" in v else True,
                ring_holdoff_ns=tdc_mapping.parse_optional_float(v["ring_holdoff_ns"].get()),
                dither=bool(v["dither"].get()),
                dither_ns=tdc_mapping.parse_dither_ns(v["dither_ns"].get()),
            )
        return tdc_mapping.TdcMapping(
            **signals, pixel_offset_ns=tdc_mapping.parse_pixel_offset_ns(self._pixel_offset_var.get()),
        )

    def _apply_mapping(self, mapping):
        for name, _title in _SIGNALS:
            signal = getattr(mapping, name)
            v = self._vars[name]
            v["family"].set(signal.source.family)
            self._channel_boxes[name].configure(values=_channels_for(signal.source.family))
            v["channel"].set(str(signal.source.channel))
            v["time_edge"].set(signal.time_edge)
            v["gate_min_ns"].set("" if signal.gate_min_ns is None else str(signal.gate_min_ns))
            v["gate_max_ns"].set("" if signal.gate_max_ns is None else str(signal.gate_max_ns))
            v["ring_holdoff_ns"].set("" if signal.ring_holdoff_ns is None else str(signal.ring_holdoff_ns))
            v["dither"].set(signal.dither)
            v["dither_ns"].set(f"{signal.dither_ns:g}")
            if "enabled" in v:
                v["enabled"].set(signal.enabled)
        self._pixel_offset_var.set(f"{mapping.pixel_offset_ns:g}")

    def _save_preset(self):
        chosen = filedialog.asksaveasfilename(
            title="Save TDC mapping preset",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            parent=self,
        )
        if not chosen:
            return
        try:
            self._current_mapping().save(chosen)
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not save preset: {exc}", parent=self)

    def _load_preset(self):
        chosen = filedialog.askopenfilename(
            title="Load TDC mapping preset",
            filetypes=[("JSON", "*.json")],
            parent=self,
        )
        if not chosen:
            return
        try:
            mapping = tdc_mapping.TdcMapping.load(chosen)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Load failed", f"Could not load preset: {exc}", parent=self)
            return
        self._apply_mapping(mapping)

    def _restore_default(self):
        self._apply_mapping(tdc_mapping.default_mapping())

