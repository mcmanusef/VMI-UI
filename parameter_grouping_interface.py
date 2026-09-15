"""Analysis tab: collects a parameter scan's cv4 files into one grouped file,
each group labelled by its parameter value (parameter_grouping.py)."""
import pathlib
import threading

from PyQt5 import QtWidgets

import qtk as tk
from qtk import ttk, filedialog, messagebox

import app_settings
import parameter_grouping as pgr
import ui_style
from calibration_interface import BackgroundTask, add_label, read_number, set_visible

MODE_NOTES = {
    pgr.MODE_DIRECT: "Group values are the stage positions. Collect asks for the parameter label.",
    pgr.MODE_QWP: (
        "Ellipticity = tan χ, with sin 2χ = sin 2(θ − zero angle), for linearly polarized light into a "
        "quarter-wave plate at stage angle θ. The sign gives the handedness; ±45° from the zero angle "
        "is circular."
    ),
    pgr.MODE_DELAY: (
        "Delay = passes × (position − zero) / c, with positions in mm. Use 2 passes for a retroreflector."
    ),
}


class ParameterGroupingInterface(ttk.Frame):
    """Scan folder in, one grouped cv4 file out, with the groups previewed
    in a table."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        def pvar(cls, key, default):
            return app_settings.persistent_var(self, cls, f"parameter_grouping.{key}", default)

        self.folder_var = pvar(tk.StringVar, "folder", "")
        self.output_var = pvar(tk.StringVar, "output_path", "")
        self.mode_var = pvar(tk.StringVar, "mode", pgr.MODE_DIRECT)
        self.direct_label_var = pvar(tk.StringVar, "direct_label", "Position")
        self.qwp_zero_var = pvar(tk.StringVar, "qwp_zero_deg", "0")
        self.delay_zero_var = pvar(tk.StringVar, "delay_zero_mm", "0")
        self.delay_passes_var = pvar(tk.StringVar, "delay_passes", "2")

        self.status_var = tk.StringVar(self, value="Idle")
        self._progress_var = tk.DoubleVar(self, value=0.0)
        self.files_var = tk.StringVar(self, value="--")
        self.skipped_var = tk.StringVar(self, value="--")
        self.skipped_list_var = tk.StringVar(self, value="")
        self.mode_note_var = tk.StringVar(self, value="")
        self.output_info_var = tk.StringVar(self, value="")

        self._task = BackgroundTask(self)
        self._stop_event = threading.Event()
        self._progress = None
        self._files = []

        self._build_ui()
        self._on_mode_changed()
        self.mode_var.trace_add("write", self._on_mode_changed)
        for var in (self.qwp_zero_var, self.delay_zero_var, self.delay_passes_var):
            var.trace_add("write", self._refresh_table)
        if self.folder_var.get().strip():
            self.scan()

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        sidebar, main = ui_style.build_sidebar_layout(self)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        sidebar.rowconfigure(5, weight=1)

        ui_style.build_button_bar(
            sidebar, 0, [("Collect", self.collect), ("Stop", self.stop), ("Scan Folder", self.scan)],
        )
        ui_style.StatusBlock(sidebar, 1, self.status_var, progress_var=self._progress_var)

        scan = ui_style.build_section(sidebar, 2, "Parameter scan")
        ui_style.add_form_row(
            scan, 0, "Scan folder:", ui_style.build_path_field(scan, self.folder_var, self._browse_folder),
        )
        ui_style.add_form_row(
            scan, 1, "Output file:", ui_style.build_path_field(scan, self.output_var, self._browse_output),
        )
        ui_style.add_note(
            scan, 2,
            f"Pick a parameter sweep's run folder. Leave the output file blank to write "
            f"<folder name>{pgr.GROUPED_SUFFIX} into it.",
        )

        param = ui_style.build_section(sidebar, 3, "Parameter")
        ui_style.add_form_row(param, 0, "Parameter:", ttk.Combobox(
            param, textvariable=self.mode_var, values=pgr.MODES, state="readonly", width=18,
        ))
        self._qwp_label = add_label(param, 1, "QWP zero angle (deg):")
        self._qwp_entry = ttk.Entry(param, textvariable=self.qwp_zero_var, width=18)
        self._qwp_entry.grid(row=1, column=1, sticky="ew", pady=4)
        self._zero_label = add_label(param, 2, "Zero position (mm):")
        self._zero_entry = ttk.Entry(param, textvariable=self.delay_zero_var, width=18)
        self._zero_entry.grid(row=2, column=1, sticky="ew", pady=4)
        self._passes_label = add_label(param, 3, "Passes:")
        self._passes_entry = ttk.Entry(param, textvariable=self.delay_passes_var, width=18)
        self._passes_entry.grid(row=3, column=1, sticky="ew", pady=4)
        ui_style.add_note(param, 4, textvariable=self.mode_note_var, pady=(4, 4))

        found = ui_style.build_section(sidebar, 4, "Detected files", pady=0)
        list_row = ui_style.add_stat_rows(found, [("Scan files:", self.files_var), ("Skipped:", self.skipped_var)])
        ui_style.add_note(found, list_row, textvariable=self.skipped_list_var, pady=(4, 4))
        ui_style.add_note(found, list_row + 1, textvariable=self.output_info_var)

        card = ui_style.build_card(main, 0, "Groups")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(0, weight=1)
        self._tree = ttk.Treeview(card, columns=("group", "file", "position", "value"), show="headings")
        for col, text, width in (
            ("group", "Group", 120), ("file", "Scan file", 260), ("position", "Stage position", 120),
            ("value", "Parameter value", 140),
        ):
            self._tree.heading(col, text=text)
            self._tree.column(col, width=width)
        self._tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _browse_folder(self):
        chosen = filedialog.askdirectory(
            initialdir=self.folder_var.get().strip() or ".", title="Choose parameter scan folder", parent=self,
        )
        if chosen:
            self.folder_var.set(chosen)
            self.scan()

    def _browse_output(self):
        folder = self.folder_var.get().strip()
        initial = self.output_var.get().strip() or (str(pgr.default_output_path(folder)) if folder else "")
        chosen = filedialog.asksaveasfilename(
            title="Save grouped file",
            defaultextension=".cv4",
            initialfile=pathlib.Path(initial).name if initial else None,
            initialdir=str(pathlib.Path(initial).parent) if initial else None,
            filetypes=[("cv4", "*.cv4")],
            parent=self,
        )
        if chosen:
            self.output_var.set(chosen)

    def _on_mode_changed(self, *_):
        mode = self.mode_var.get()
        set_visible((self._qwp_label, self._qwp_entry), mode == pgr.MODE_QWP)
        set_visible((self._zero_label, self._zero_entry, self._passes_label, self._passes_entry), mode == pgr.MODE_DELAY)
        self.mode_note_var.set(MODE_NOTES.get(mode, ""))
        self._refresh_table()

    # ---- scan -----------------------------------------------------------------

    def scan(self):
        if self._task.is_running():
            return
        folder = self.folder_var.get().strip()
        if not folder:
            self.status_var.set("Scan folder is required.")
            return
        if not pathlib.Path(folder).is_dir():
            self.status_var.set(f"Scan folder not found: {folder}")
            return
        self.status_var.set("Scanning folder...")
        self._task.start(
            lambda: pgr.find_scan_files(folder), self._on_scanned,
            lambda exc: self.status_var.set(f"Error: could not read {folder}: {exc}"),
        )

    def _on_scanned(self, found):
        self._files, skipped = found
        self.files_var.set(str(len(self._files)))
        self.skipped_var.set(str(len(skipped)))
        self.skipped_list_var.set("Skipped: " + ", ".join(skipped) if skipped else "")
        self._refresh_table()
        if self._files:
            self.status_var.set(f"Ready to collect {len(self._files)} scan file(s).")
        else:
            self.status_var.set("No scan cv4 files with a position found in the folder.")

    def _settings(self):
        """(parameter values, settings saved in the file) for the current mode."""
        mode = self.mode_var.get()
        settings = {}
        if mode == pgr.MODE_QWP:
            settings["QWP zero angle (deg)"] = read_number(self.qwp_zero_var, "QWP zero angle")
        elif mode == pgr.MODE_DELAY:
            settings["Zero position (mm)"] = read_number(self.delay_zero_var, "Zero position")
            settings["Passes"] = read_number(self.delay_passes_var, "Passes", integer=True, positive=True)
        values = pgr.parameter_values(
            [scan.position for scan in self._files], mode,
            qwp_zero_deg=settings.get("QWP zero angle (deg)", 0.0),
            delay_zero_mm=settings.get("Zero position (mm)", 0.0),
            delay_passes=settings.get("Passes", 2),
        )
        return values, settings

    def _refresh_table(self, *_):
        self._tree.delete(*self._tree.get_children())
        if not self._files:
            return
        try:
            values, _ = self._settings()
        except ValueError as exc:
            self.status_var.set(f"Invalid setting: {exc}")
            return
        for scan, value, name in zip(self._files, values, pgr.group_names(values)):
            self._tree.insert("", "end", values=(name, scan.path.name, f"{scan.position:+.4f}", f"{value:.6g}"))

    # ---- collect ----------------------------------------------------------------

    def collect(self):
        if self._task.is_running():
            return
        if not self._files:
            self.status_var.set("Scan a folder with scan cv4 files first.")
            return
        try:
            values, settings = self._settings()
        except ValueError as exc:
            self.status_var.set(f"Invalid setting: {exc}")
            return
        mode = self.mode_var.get()
        if mode == pgr.MODE_DIRECT:
            label, ok = QtWidgets.QInputDialog.getText(
                self.window(), "Parameter label", "Label for the scanned parameter (saved in the file):",
                QtWidgets.QLineEdit.Normal, self.direct_label_var.get(),
            )
            if not ok:
                self.status_var.set("Collect canceled.")
                return
            label = label.strip()
            if not label:
                self.status_var.set("Parameter label is required.")
                return
            self.direct_label_var.set(label)
        else:
            label = pgr.LABELS[mode]

        folder = self.folder_var.get().strip()
        output = pathlib.Path(self.output_var.get().strip() or pgr.default_output_path(folder))
        if any(output.resolve() == scan.path.resolve() for scan in self._files):
            self.status_var.set("Output file must not be one of the scan files.")
            return
        if output.exists() and not messagebox.askyesno(
            "File exists", f"{output} already exists.\n\nOverwrite it?", parent=self,
        ):
            self.status_var.set("Collect canceled.")
            return

        files = list(self._files)
        self._stop_event.clear()
        self._progress = None
        self._progress_var.set(0.0)
        self.status_var.set(f"Collecting {len(files)} scan file(s)...")
        self._task.start(
            lambda: pgr.write_grouped(
                output, files, values, label, mode, settings, folder,
                progress=self._on_progress, stop_event=self._stop_event,
            ),
            lambda path: self._on_collected(path, len(files), label),
            self._on_collect_error,
        )
        self._poll_progress()

    def stop(self):
        if not self._task.is_running():
            return
        self._stop_event.set()
        self.status_var.set("Stopping...")

    def _on_progress(self, done, total, name):
        # Worker thread: only store it; _poll_progress shows it.
        self._progress = (done, total, name)

    def _poll_progress(self):
        progress = self._progress
        if progress is not None and not self._stop_event.is_set():
            done, total, name = progress
            self._progress_var.set(100.0 * done / total if total else 100.0)
            if done < total:
                self.status_var.set(f"Collecting {name} ({done + 1}/{total})...")
        if self._task.is_running():
            self.after(200, self._poll_progress)

    def _on_collected(self, path, count, label):
        self._progress_var.set(100.0)
        self.output_info_var.set(f"Saved {path}")
        self.status_var.set(f"Finished. Collected {count} group(s) labelled by {label} into {path}.")

    def _on_collect_error(self, exc):
        self._progress_var.set(0.0)
        if isinstance(exc, pgr.CollectStopped):
            self.status_var.set("Stopped. No grouped file was written.")
        else:
            self.status_var.set(f"Error: could not collect the scan: {exc}")
