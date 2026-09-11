"""A collapsible section: a clickable header (bold title + a small arrow)
that shows/hides a body frame below/inside it -- for tucking away a block
of controls that's large, secondary, or just not needed all the time,
without removing it from the tab.

Usage: build widgets into `.body` (a plain qtk.ttk.Frame) instead of
directly into a CollapsibleFrame -- same convention as ScrollableFrame's
`.body`. Works for both a small block (e.g. one LabelFrame's worth of
fields) and a whole sidebar (nest a ScrollableFrame inside `.body`): when
collapsed, the body is hidden entirely and the widget's footprint shrinks
to just the header, so whatever it shares a row/column with (a plot area,
typically) reclaims that space -- Qt's layouts exclude hidden widgets from
size calculations, no manual resizing needed.
"""
from PyQt5 import QtCore, QtWidgets

from qtk import GridMixin, TkCompatMixin, ttk


class CollapsibleFrame(TkCompatMixin, GridMixin, QtWidgets.QWidget):
    def __init__(self, parent=None, text="", collapsed=False, **_kwargs):
        super().__init__(parent)
        self._title = text

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self._toggle = QtWidgets.QToolButton(self)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self._toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._toggle.clicked.connect(self._on_toggle)
        outer.addWidget(self._toggle, 0, QtCore.Qt.AlignLeft)

        self._panel = QtWidgets.QFrame(self)
        self._panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        panel_layout = QtWidgets.QVBoxLayout(self._panel)
        panel_layout.setContentsMargins(6, 6, 6, 6)
        self.body = ttk.Frame(self._panel)
        panel_layout.addWidget(self.body)
        outer.addWidget(self._panel)

        self._panel.setVisible(not collapsed)
        self._update_toggle_text()

    def _update_toggle_text(self):
        arrow = "▾" if self._toggle.isChecked() else "▸"  # small down/right triangle
        self._toggle.setText(f"{arrow} {self._title}")

    def _on_toggle(self, checked):
        self._panel.setVisible(checked)
        self._update_toggle_text()

    def is_collapsed(self):
        return not self._toggle.isChecked()

    def set_collapsed(self, collapsed):
        self._toggle.setChecked(not collapsed)
        self._on_toggle(not collapsed)
