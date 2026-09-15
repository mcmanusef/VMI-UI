# simple_tabs_ui.py
# pip install PyQt5

import qtk as tk
from qtk import ttk

from acquisition_interface import AcquisitionInterface
from analysis_interface import ConversionInterface
from coincidence_interface import CoincidenceInterface
from calibration_apply_interface import CalibrationApplyInterface
from calibration_interface import CalibrationHub, MassCalibrationInterface, MomentumCalibrationInterface
from collection_interface import CollectionInterface
from diagnostics_interface import DiagnosticsInterface
from parameter_grouping_interface import ParameterGroupingInterface
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
        # The calibration tabs publish their latest calibration here, and
        # the apply calibration tab picks it up.
        self._calibration_hub = CalibrationHub()

        # Three top-level groups -- Hardware (things you connect to/operate),
        # Acquisition (things that collect data) and Analysis (offline work
        # on data already collected) -- each its own inner tab bar.
        groups = ttk.Notebook(self)
        groups.grid(row=0, column=0, sticky="nsew")

        hardware_group = ttk.Frame(groups)
        acquisition_group = ttk.Frame(groups)
        analysis_group = ttk.Frame(groups)
        groups.add(hardware_group, text="Hardware")
        groups.add(acquisition_group, text="Acquisition")
        groups.add(analysis_group, text="Analysis")

        hardware = self._build_group_notebook(hardware_group)
        acquisition = self._build_group_notebook(acquisition_group)
        analysis = self._build_group_notebook(analysis_group)

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

        # Analysis tabs
        conversion_tab = ttk.Frame(analysis)
        momentum_calibration_tab = ttk.Frame(analysis)
        mq_calibration_tab = ttk.Frame(analysis)
        apply_calibration_tab = ttk.Frame(analysis)
        grouping_tab = ttk.Frame(analysis)
        coincidence_tab = ttk.Frame(analysis)
        analysis.add(conversion_tab, text="conversion")
        analysis.add(momentum_calibration_tab, text="momentum calibration")
        analysis.add(mq_calibration_tab, text="m/q calibration")
        analysis.add(apply_calibration_tab, text="apply calibration")
        analysis.add(grouping_tab, text="parameter grouping")
        analysis.add(coincidence_tab, text="coincidence")

        # Fill each tab with its interface. Stage and Power Supply must be
        # built before Sweep -- Sweep drives its moves through the stage
        # connection StageInterface owns, and (optionally) ramps a Power
        # Supply channel group down around each move (see
        # stage_interface.py / power_supply_interface.py /
        # sweep_interface.py).
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
        self._build_conversion(conversion_tab)
        self._build_momentum_calibration(momentum_calibration_tab)
        self._build_mq_calibration(mq_calibration_tab)
        self._build_apply_calibration(apply_calibration_tab)
        self._build_parameter_grouping(grouping_tab)
        self._build_coincidence(coincidence_tab)

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
            power_supply_ui=getattr(self, "power_supply_ui", None),
            coordinator=self._coordinator,
        )
        sweep_ui.grid(row=0, column=0, sticky="nsew")

    def _build_conversion(self, parent: ttk.Frame):
        _prepare_tab(parent)

        conversion_ui = ConversionInterface(
            parent,
            plot_shared_vars=self._plot_shared_vars,
            acq_shared_vars=self._acq_shared_vars,
        )
        conversion_ui.grid(row=0, column=0, sticky="nsew")

    def _build_momentum_calibration(self, parent: ttk.Frame):
        _prepare_tab(parent)

        momentum_ui = MomentumCalibrationInterface(parent, hub=self._calibration_hub)
        momentum_ui.grid(row=0, column=0, sticky="nsew")

    def _build_mq_calibration(self, parent: ttk.Frame):
        _prepare_tab(parent)

        mq_ui = MassCalibrationInterface(parent, hub=self._calibration_hub)
        mq_ui.grid(row=0, column=0, sticky="nsew")

    def _build_apply_calibration(self, parent: ttk.Frame):
        _prepare_tab(parent)

        apply_ui = CalibrationApplyInterface(parent, hub=self._calibration_hub)
        apply_ui.grid(row=0, column=0, sticky="nsew")

    def _build_parameter_grouping(self, parent: ttk.Frame):
        _prepare_tab(parent)

        grouping_ui = ParameterGroupingInterface(parent)
        grouping_ui.grid(row=0, column=0, sticky="nsew")

    def _build_coincidence(self, parent: ttk.Frame):
        _prepare_tab(parent)

        coincidence_ui = CoincidenceInterface(parent)
        coincidence_ui.grid(row=0, column=0, sticky="nsew")

    def _build_power_supply(self, parent: ttk.Frame):
        _prepare_tab(parent)

        self.power_supply_ui = PowerSupplyInterface(parent, coordinator=self._coordinator)
        self.power_supply_ui.grid(row=0, column=0, sticky="nsew")


if __name__ == "__main__":
    app = App()
    app.mainloop()
