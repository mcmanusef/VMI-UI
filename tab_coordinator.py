"""Cross-tab "only one measurement at a time" coordination.

Serval only has one active trigger/destination configuration at a time, so
two tabs driving it simultaneously would just fight over that config. Every
tab that can start a measurement registers itself here; each tab's start()
asks the coordinator to stop every other *currently running* tab first.
"""


class TabCoordinator:
    def __init__(self):
        self._tabs = []

    def register(self, tab):
        self._tabs.append(tab)

    def stop_others(self, active_tab):
        for tab in self._tabs:
            if tab is active_tab:
                continue
            try:
                if tab.is_running():
                    tab.stop()
            except Exception:
                pass
