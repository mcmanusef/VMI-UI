# simple_tabs_ui.py
# pip install PyQt5

import qtk as tk
from qtk import ttk

import app_settings
from acquisition_interface import AcquisitionInterface
from collection_interface import CollectionInterface
from diagnostics_interface import DiagnosticsInterface
from power_supply_interface import PowerSupplyInterface
from quick_monitor_interface import QuickMonitorInterface
from serval_interface import ServalInterface
from shared_state import make_plot_shared_vars, make_acquisition_shared_vars
from sweep_interface import SweepInterface
from tab_coordinator import TabCoordinator
from timewalk_interface import TimewalkInterface

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Experiment Control")
        self.minsize(900, 600)

        # Top-level layout
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Shared, persisted state: same live Tk variables handed to every
        # tab that needs them, so editing e.g. frame time or histogram bins
        # in one tab is reflected in the other immediately, and both keep
        # their value across restarts (see shared_state.py / app_settings.py).
        self._coordinator = TabCoordinator()
        self._plot_shared_vars = make_plot_shared_vars(self)
        self._acq_shared_vars = make_acquisition_shared_vars(self)

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

        # Tabs
        serval_tab = ttk.Frame(notebook)
        collection_tab = ttk.Frame(notebook)
        diagnostics_tab = ttk.Frame(notebook)
        acquisition_tab = ttk.Frame(notebook)
        timewalk_tab = ttk.Frame(notebook)
        quick_monitor_tab = ttk.Frame(notebook)
        sweep_tab = ttk.Frame(notebook)
        power_supply_tab = ttk.Frame(notebook)

        notebook.add(serval_tab, text="serval config")
        notebook.add(collection_tab, text="collection parameters")
        notebook.add(diagnostics_tab, text="diagnostics")
        notebook.add(acquisition_tab, text="monitored acquisition")
        notebook.add(timewalk_tab, text="timewalk calibration")
        notebook.add(quick_monitor_tab, text="quick monitor")
        notebook.add(sweep_tab, text="parameter sweep")
        notebook.add(power_supply_tab, text="power supply")

        # Fill each tab with a simple example layout
        self._build_serval_config(serval_tab)
        self._build_collection_params(collection_tab)
        self._build_diagnostics(diagnostics_tab)
        self._build_acquisition(acquisition_tab)
        self._build_timewalk(timewalk_tab)
        self._build_quick_monitor(quick_monitor_tab)
        self._build_sweep(sweep_tab)
        self._build_power_supply(power_supply_tab)

    def _save_current_as_default(self):
        """Snapshot every tab's user-editable fields into app_settings.json
        as the new defaults. Fields already wired to a shared/persisted Tk
        variable (see shared_state.py) auto-save on every change already --
        this is for the rest of each tab's fields, which otherwise reset to
        their hardcoded defaults every restart. Each tab that has any such
        fields exposes them via a default_fields() -> {key: var} method;
        tabs with nothing extra to save (already fully shared/persisted)
        just don't define one.
        """
        saved = 0
        for tab in self._coordinator._tabs:
            fields = getattr(tab, "default_fields", None)
            if fields is None:
                continue
            app_settings.update({key: var.get() for key, var in fields().items()})
            saved += 1
        self._save_defaults_status_var.set(f"Saved current values as default ({saved} tab(s)).")

    def _build_serval_config(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.serval_ui = ServalInterface(parent, coordinator=self._coordinator)
        self.serval_ui.grid(row=0, column=0, sticky="nsew")

    def _build_collection_params(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        server_var = getattr(self, "serval_ui", None)
        collection_ui = CollectionInterface(
            parent,
            server_var=server_var.server_var if server_var else None,
            acq_shared_vars=self._acq_shared_vars,
            coordinator=self._coordinator,
        )
        collection_ui.grid(row=0, column=0, sticky="nsew")

    def _build_diagnostics(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        server_var = getattr(self, "serval_ui", None)
        diagnostics_ui = DiagnosticsInterface(
            parent,
            server_var=server_var.server_var if server_var else None,
            plot_shared_vars=self._plot_shared_vars,
            coordinator=self._coordinator,
        )
        diagnostics_ui.grid(row=0, column=0, sticky="nsew")

    def _build_acquisition(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        server_var = getattr(self, "serval_ui", None)
        acquisition_ui = AcquisitionInterface(
            parent,
            server_var=server_var.server_var if server_var else None,
            plot_shared_vars=self._plot_shared_vars,
            acq_shared_vars=self._acq_shared_vars,
            coordinator=self._coordinator,
        )
        acquisition_ui.grid(row=0, column=0, sticky="nsew")

    def _build_timewalk(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        server_var = getattr(self, "serval_ui", None)
        timewalk_ui = TimewalkInterface(
            parent,
            server_var=server_var.server_var if server_var else None,
            coordinator=self._coordinator,
        )
        timewalk_ui.grid(row=0, column=0, sticky="nsew")

    def _build_quick_monitor(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        server_var = getattr(self, "serval_ui", None)
        quick_monitor_ui = QuickMonitorInterface(
            parent,
            server_var=server_var.server_var if server_var else None,
            coordinator=self._coordinator,
        )
        quick_monitor_ui.grid(row=0, column=0, sticky="nsew")

    def _build_sweep(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        server_var = getattr(self, "serval_ui", None)
        sweep_ui = SweepInterface(
            parent,
            server_var=server_var.server_var if server_var else None,
            acq_shared_vars=self._acq_shared_vars,
            coordinator=self._coordinator,
        )
        sweep_ui.grid(row=0, column=0, sticky="nsew")

    def _build_power_supply(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        power_supply_ui = PowerSupplyInterface(parent, coordinator=self._coordinator)
        power_supply_ui.grid(row=0, column=0, sticky="nsew")


if __name__ == "__main__":
    app = App()
    app.mainloop()
