"""Standalone offline-analysis UI: cluster raw .tpx3 files into cv4, and run
the interactive momentum calibration wizard against the result. A separate
program from UI.py (the live acquisition/diagnostics UI) -- run this one
directly (`python analysis_ui.py`) whenever there's no Serval server, just
files to reprocess.
"""
import qtk as tk
from qtk import ttk

import app_settings
from cluster_raw_interface import ClusterRawInterface
from momentum_calibration_interface import MomentumCalibrationInterface
from tab_coordinator import TabCoordinator


class AnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VMI Analysis")
        self.minsize(1000, 750)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._coordinator = TabCoordinator()

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self._save_defaults_status_var = tk.StringVar(self, value="")
        ttk.Button(toolbar, text="Save current as default", command=self._save_current_as_default).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(toolbar, textvariable=self._save_defaults_status_var, font=("Segoe UI", 8)).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )

        notebook = ttk.Notebook(self)
        notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        cluster_tab = ttk.Frame(notebook)
        momcal_tab = ttk.Frame(notebook)
        notebook.add(cluster_tab, text="cluster raw files")
        notebook.add(momcal_tab, text="momentum calibration")

        self._build_cluster_tab(cluster_tab)
        self._build_momcal_tab(momcal_tab)

    def _save_current_as_default(self):
        saved = 0
        for tab in self._coordinator._tabs:
            fields = getattr(tab, "default_fields", None)
            if fields is None:
                continue
            app_settings.update({key: var.get() for key, var in fields().items()})
            saved += 1
        self._save_defaults_status_var.set(f"Saved current values as default ({saved} tab(s)).")

    def _build_cluster_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        ui = ClusterRawInterface(parent, coordinator=self._coordinator)
        ui.grid(row=0, column=0, sticky="nsew")

    def _build_momcal_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        ui = MomentumCalibrationInterface(parent, coordinator=self._coordinator)
        ui.grid(row=0, column=0, sticky="nsew")


if __name__ == "__main__":
    app = AnalysisApp()
    app.mainloop()
