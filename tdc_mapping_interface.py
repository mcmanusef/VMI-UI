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
            "pulse can share one FPGA TDC channel this way. These settings apply everywhere "
            "(Monitored Acquisition, Analysis > conversion, Diagnostics, Timewalk Calibration, "
            "Parameter Sweep) and are saved as you edit them -- the \"tpx to h5\" raw dump is "
            "unaffected, since it already keeps every hit's own channel and edge.",
        )

        row = 1
        for name, title in _SIGNALS:
            section = ui_style.build_section(page, row, title)
            self._build_signal_section(section, name)
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
        self._vars[name] = {
            "family": family_var, "channel": channel_var, "time_edge": edge_var,
            "gate_min_ns": gate_min_var, "gate_max_ns": gate_max_var,
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
            )
        return tdc_mapping.TdcMapping(**signals)

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
            if "enabled" in v:
                v["enabled"].set(signal.enabled)

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

