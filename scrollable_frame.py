"""A vertically-scrollable container for tall sidebars.

Usage: build widgets into `.body` (a plain qtk.ttk.Frame) instead of
directly into a ScrollableFrame -- the scrollbar only actually does
anything once `.body`'s natural height exceeds the visible area, so this
is a drop-in replacement for a plain ttk.Frame sidebar with no behavior
change when everything already fits.

Native PyQt5 (QScrollArea already handles wheel scrolling, resize-to-width
and the scrollbar itself, so this doesn't need qtk's tkinter-flavored
Canvas/Scrollbar dance the original tkinter version used).
"""
from PyQt5 import QtCore, QtWidgets

from qtk import GridMixin, TkCompatMixin, ttk


class ScrollableFrame(TkCompatMixin, GridMixin, QtWidgets.QScrollArea):
    def __init__(self, parent=None, width=300, **_kwargs):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setMinimumWidth(width)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.body = ttk.Frame(self)
        self.setWidget(self.body)
