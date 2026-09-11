# simple_tabs_ui.py
# pip install PyQt5

import qtk as tk
from qtk import ttk

from acquisition_interface import AcquisitionInterface
from collection_interface import CollectionInterface
from diagnostics_interface import DiagnosticsInterface
from power_supply_interface import PowerSupplyInterface
from serval_interface import ServalInterface
from shared_state import make_plot_shared_vars, make_acquisition_shared_vars
from stage_interface import StageInterface
from sweep_interface import SweepInterface
from tab_coordinator import TabCoordinator
from timewalk_interface import TimewalkInterface

def _prepare_tab(parent: ttk.Frame):
    """Every tab page is just a single-cell grid holding one widget that
    fills it -- factored out since qtk's GridMixin creates that grid with
    Qt's default (non-zero) margins the first time anything's gridded into
    it, which otherwise stacks with each interface's own padx/pady and
    keeps its content (and, for Monitored Acquisition/Diagnostics, the
    "Controls"/"Plot options" divider rules) from ever reaching the tab's
    real edges."""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)
    parent.layout().setContentsMargins(0, 0, 0, 0)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Experiment Control")
        self.minsize(900, 600)

        # Top-level layout
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.layout().setContentsMargins(0, 0, 0, 0)

        # Shared, persisted state: same live Tk variables handed to every
        # tab that needs them, so editing e.g. frame time or histogram bins
        # in one tab is reflected in the other immediately, and both keep
        # their value across restarts (see shared_state.py / app_settings.py).
        self._coordinator = TabCoordinator()
        self._plot_shared_vars = make_plot_shared_vars(self)
        self._acq_shared_vars = make_acquisition_shared_vars(self)

        # Two top-level groups -- Hardware (things you connect to/operate)
        # and Acquisition (things that collect or analyze data) -- each its
        # own inner tab bar.
        groups = ttk.Notebook(self)
        groups.grid(row=0, column=0, sticky="nsew")

        hardware_group = ttk.Frame(groups)
        acquisition_group = ttk.Frame(groups)
        groups.add(hardware_group, text="Hardware")
        groups.add(acquisition_group, text="Acquisition")

        hardware = self._build_group_notebook(hardware_group)
        acquisition = self._build_group_notebook(acquisition_group)

        # Hardware tabs
        serval_tab = ttk.Frame(hardware)
        stage_tab = ttk.Frame(hardware)
        power_supply_tab = ttk.Frame(hardware)
        hardware.add(serval_tab, text="serval config")
        hardware.add(stage_tab, text="stage control")
        hardware.add(power_supply_tab, text="power supply")

        # Acquisition tabs
        collection_tab = ttk.Frame(acquisition)
        diagnostics_tab = ttk.Frame(acquisition)
        acquisition_tab = ttk.Frame(acquisition)
        timewalk_tab = ttk.Frame(acquisition)
        sweep_tab = ttk.Frame(acquisition)
        acquisition.add(collection_tab, text="collection parameters")
        acquisition.add(diagnostics_tab, text="diagnostics")
        acquisition.add(acquisition_tab, text="monitored acquisition")
        acquisition.add(timewalk_tab, text="timewalk calibration")
        acquisition.add(sweep_tab, text="parameter sweep")

        # Fill each tab with its interface. Stage must be built before
        # Sweep -- Sweep drives its moves through the stage connection
        # StageInterface owns (see stage_interface.py / sweep_interface.py).
        # Quick Monitor (quick_monitor_interface.py) isn't wired in as a tab
        # right now, but the module is left intact for later.
        self._build_serval_config(serval_tab)
        self._build_stage(stage_tab)
        self._build_power_supply(power_supply_tab)
        self._build_collection_params(collection_tab)
        self._build_diagnostics(diagnostics_tab)
        self._build_acquisition(acquisition_tab)
        self._build_timewalk(timewalk_tab)
        self._build_sweep(sweep_tab)

    def _build_group_notebook(self, parent: ttk.Frame):
        _prepare_tab(parent)
        notebook = ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky="nsew")
        return notebook

    def _build_serval_config(self, parent: ttk.Frame):
        _prepare_tab(parent)

        self.serval_ui = ServalInterface(parent, coordinator=self._coordinator)
        self.serval_ui.grid(row=0, column=0, sticky="nsew")

    def _build_stage(self, parent: ttk.Frame):
        _prepare_tab(parent)

        self.stage_ui = StageInterface(parent, coordinator=self._coordinator)
        self.stage_ui.grid(row=0, column=0, sticky="nsew")

    def _build_collection_params(self, parent: ttk.Frame):
        _prepare_tab(parent)

        server_var = getattr(self, "serval_ui", None)
        collection_ui = CollectionInterface(
            parent,
            server_var=server_var.server_var if server_var else None,
            acq_shared_vars=self._acq_shared_vars,
            coordinator=self._coordinator,
        )
        collection_ui.grid(row=0, column=0, sticky="nsew")

    def _build_diagnostics(self, parent: ttk.Frame):
        _prepare_tab(parent)

        server_var = getattr(self, "serval_ui", None)
        diagnostics_ui = DiagnosticsInterface(
            parent,
            server_var=server_var.server_var if server_var else None,
            plot_shared_vars=self._plot_shared_vars,
            coordinator=self._coordinator,
        )
        diagnostics_ui.grid(row=0, column=0, sticky="nsew")

    def _build_acquisition(self, parent: ttk.Frame):
        _prepare_tab(parent)

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
        _prepare_tab(parent)

        server_var = getattr(self, "serval_ui", None)
        timewalk_ui = TimewalkInterface(
            parent,
            server_var=server_var.server_var if server_var else None,
            coordinator=self._coordinator,
        )
        timewalk_ui.grid(row=0, column=0, sticky="nsew")

    def _build_sweep(self, parent: ttk.Frame):
        _prepare_tab(parent)

        server_var = getattr(self, "serval_ui", None)
        sweep_ui = SweepInterface(
            parent,
            server_var=server_var.server_var if server_var else None,
            acq_shared_vars=self._acq_shared_vars,
            stage_ui=getattr(self, "stage_ui", None),
            coordinator=self._coordinator,
        )
        sweep_ui.grid(row=0, column=0, sticky="nsew")

    def _build_power_supply(self, parent: ttk.Frame):
        _prepare_tab(parent)

        power_supply_ui = PowerSupplyInterface(parent, coordinator=self._coordinator)
        power_supply_ui.grid(row=0, column=0, sticky="nsew")


if __name__ == "__main__":
    app = App()
    app.mainloop()
