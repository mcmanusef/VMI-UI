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

Two looks, picked with `flush`:
  - boxed (default) -- a bordered box, for a block that's one of several
    peers stacked inside some other container (e.g. "Metadata" among
    other sections in a sidebar).
  - flush (`flush=True`) -- no box, no inset margin, meant to span its
    whole side of whatever it's next to (a plots area, typically) instead
    of floating as an inset panel; visually separated from that neighbor
    by a single rule on the given `separator` edge instead.
"""
from PyQt5 import QtCore, QtWidgets

from qtk import GridMixin, TkCompatMixin, ttk


class CollapsibleFrame(TkCompatMixin, GridMixin, QtWidgets.QWidget):
    def __init__(self, parent=None, text="", collapsed=False, flush=False, separator="top", **_kwargs):
        super().__init__(parent)
        self._title = text

        content = QtWidgets.QWidget(self)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(2)

        self._toggle = QtWidgets.QToolButton(content)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self._toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._toggle.clicked.connect(self._on_toggle)
        content_layout.addWidget(self._toggle, 0, QtCore.Qt.AlignLeft)

        self._panel = QtWidgets.QFrame(content)
        panel_layout = QtWidgets.QVBoxLayout(self._panel)
        if flush:
            self._panel.setFrameShape(QtWidgets.QFrame.NoFrame)
            panel_layout.setContentsMargins(0, 6, 0, 0)
        else:
            self._panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
            panel_layout.setContentsMargins(6, 6, 6, 6)
        self.body = ttk.Frame(self._panel)
        panel_layout.addWidget(self.body)
        content_layout.addWidget(self._panel)
        self._panel.setVisible(not collapsed)

        if flush:
            # A single rule on the edge that borders the neighbor (a plots
            # area, typically) instead of a box all the way around --
            # content otherwise runs edge-to-edge with this widget itself.
            rule = QtWidgets.QFrame(self)
            vertical = separator in ("left", "right")
            rule.setFrameShape(QtWidgets.QFrame.VLine if vertical else QtWidgets.QFrame.HLine)
            rule.setFrameShadow(QtWidgets.QFrame.Sunken)
            outer = QtWidgets.QHBoxLayout(self) if vertical else QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(6)
            if separator in ("left", "top"):
                outer.addWidget(rule)
                outer.addWidget(content)
            else:
                outer.addWidget(content)
                outer.addWidget(rule)
        else:
            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.addWidget(content)

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
