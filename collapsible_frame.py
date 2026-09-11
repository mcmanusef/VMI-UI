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

`vertical_label=True` (only sensible for a flush, left/right-separated
sidebar) draws the toggle's text rotated top-to-bottom, as a narrow
labeled strip beside the panel instead of a horizontal bar above it.

`toggle_last=True` (only sensible for a flush, top/bottom-separated
footer) puts the toggle *after* the panel instead of before it, so it
reads as a fixed-position bar pinned to the far edge (typically the
bottom of the tab) with the panel opening/closing above it, rather than
a header that itself moves every time something above it (typically a
plots area sized to fill whatever's left) grows or shrinks to
compensate.
"""
from PyQt5 import QtCore, QtGui, QtWidgets

from qtk import GridMixin, TkCompatMixin, ttk

# One consistent color tier for every CollapsibleFrame in the app: flush
# (sidebar/footer) containers get a light grey, and the boxed sections
# nested inside them get a step darker, so a sidebar full of collapsible
# blocks reads as one uniform hierarchy instead of a patchwork of
# whatever each tab happened to inherit. Actual input fields (Entry/Text)
# paint their own white "Base" background regardless, matching the white
# plots next to all this. Deliberately a bit darker than Qt's own default
# window grey (#f0f0f0) for clearer contrast against that white.
FLUSH_BACKGROUND = QtGui.QColor("#e9e9e9")
BOXED_BACKGROUND = QtGui.QColor("#d8d8d8")


def set_background(widget, color):
    """Force a solid background fill. Plain QWidgets (and QScrollArea /
    its viewport) don't paint one of their own by default -- they just
    show whatever's behind them -- so without this, backgrounds don't
    layer the way nested boxes need them to."""
    widget.setAutoFillBackground(True)
    palette = widget.palette()
    palette.setColor(QtGui.QPalette.Window, color)
    palette.setColor(QtGui.QPalette.Base, color)
    widget.setPalette(palette)


_TOGGLE_STYLE = """
QToolButton { border: none; font-weight: bold; padding: 4px; }
QToolButton:hover { background-color: rgba(0, 0, 0, 25); }
QToolButton:pressed { background-color: rgba(0, 0, 0, 45); }
"""


class _VerticalToolButton(QtWidgets.QToolButton):
    """A QToolButton whose label is drawn rotated 90 degrees (reads
    top-to-bottom), for a toggle that sits in a narrow vertical strip."""

    def sizeHint(self):
        size = super().sizeHint()
        # A few extra px on the strip's width (the rotated button's own
        # *height*) than the bare text needs -- a wider, easier-to-hit
        # target for what's otherwise a thin sliver along the tab's edge.
        return QtCore.QSize(size.height() + 10, size.width())

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.rotate(90)
        painter.translate(0, -self.width())
        painter.setFont(self.font())
        painter.setPen(self.palette().buttonText().color())
        rect = QtCore.QRect(0, 0, self.height(), self.width())
        painter.drawText(rect, QtCore.Qt.AlignCenter, self.text())


class CollapsibleFrame(TkCompatMixin, GridMixin, QtWidgets.QWidget):
    def __init__(
        self, parent=None, text="", collapsed=False, flush=False, separator="top",
        vertical_label=False, toggle_last=False, **_kwargs,
    ):
        super().__init__(parent)
        self._title = text

        toggle_cls = _VerticalToolButton if vertical_label else QtWidgets.QToolButton
        self._toggle = toggle_cls(self)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self._toggle.setStyleSheet(_TOGGLE_STYLE)
        self._toggle.clicked.connect(self._on_toggle)

        self._panel = QtWidgets.QFrame(self)
        panel_layout = QtWidgets.QVBoxLayout(self._panel)
        if flush:
            self._panel.setFrameShape(QtWidgets.QFrame.NoFrame)
            top_margin = 0 if (vertical_label or toggle_last) else 6
            panel_layout.setContentsMargins(0, top_margin, 0, 0)
        else:
            self._panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
            panel_layout.setContentsMargins(6, 6, 6, 6)
        self.body = ttk.Frame(self._panel)
        panel_layout.addWidget(self.body)
        self._panel.setVisible(not collapsed)

        content = QtWidgets.QWidget(self)
        if vertical_label:
            # Toggle is a narrow labeled strip beside the panel, not a
            # horizontal bar above it.
            content_layout = QtWidgets.QHBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(4)
            content_layout.addWidget(self._toggle, 0, QtCore.Qt.AlignTop)
            content_layout.addWidget(self._panel)
        else:
            content_layout = QtWidgets.QVBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(2)
            if toggle_last:
                content_layout.addWidget(self._panel)
                content_layout.addWidget(self._toggle, 0, QtCore.Qt.AlignLeft)
            else:
                content_layout.addWidget(self._toggle, 0, QtCore.Qt.AlignLeft)
                content_layout.addWidget(self._panel)

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

        if flush:
            # The whole thing -- title included -- is the sidebar/footer
            # itself, so it's all one shade.
            for w in (self, content, self._panel):
                set_background(w, FLUSH_BACKGROUND)
        else:
            # Only the box gets the darker shade; the title stays
            # untouched (transparent), showing whatever it's sitting
            # on -- typically the lighter flush background around it --
            # rather than reading as part of the darker box below it.
            set_background(self._panel, BOXED_BACKGROUND)

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
