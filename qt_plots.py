"""Shared pyqtgraph plotting utilities used by every ported figure in the
app: the cmasher-backed colormap, 2D-histogram -> RGBA color mapping (the
pyqtgraph equivalent of plot_panel.build_log_norm/build_power_norm +
ax.imshow), a matplotlib "tab:*" color-name shim, and a ViewBox subclass
that gives every plot the same rectangle-zoom / double-click-focus /
right-click-reset mouse behavior matplotlib's RectangleSelector + manual
button handling gave plot_panel.py's diagnostics grid.

Orientation note (see hist_to_rgba): every 2D-histogram plot in the app
used to call `ax.imshow(hist.T, origin="lower", ...)` on a
numpy.histogram2d()-shaped array (hist[i, j] = x-bin i, y-bin j).
Empirically (imageAxisOrder="row-major"), feeding `hist.T` straight into
ImageItem.setImage() reproduces that exact orientation with no extra
flip needed -- so hist_to_rgba takes `hist` (not `hist.T`) and transposes
internally, keeping call sites a drop-in replacement.
"""
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

try:
    import cmasher as cmr
except Exception:
    cmr = None

# Light background/black foreground to match the rest of the (Fusion
# light-themed) UI -- the same look matplotlib's default axes gave before.
pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k", antialias=True)

_MPL_COLORS = {
    "tab:blue": "#1f77b4",
    "tab:orange": "#ff7f0e",
    "tab:green": "#2ca02c",
    "tab:red": "#d62728",
}


def mpl_color(name):
    """Translate the matplotlib-only color names ("tab:blue", ...) used
    throughout the ported code into something pg.mkPen/mkBrush (i.e.
    QColor) understands; anything else passes through unchanged."""
    return _MPL_COLORS.get(name, name)


_rainforest_cmap = None


def rainforest_colormap():
    """cmasher's 'rainforest' colormap as a pg.ColorMap (256-entry LUT),
    falling back to pyqtgraph's builtin 'viridis' if cmasher isn't
    installed -- the same colormap + fallback every 2D histogram in the
    app used via matplotlib (`cmr.rainforest if cmr is not None else
    "viridis"`)."""
    global _rainforest_cmap
    if _rainforest_cmap is None:
        if cmr is not None:
            samples = cmr.rainforest(np.linspace(0.0, 1.0, 256))  # Nx4 float [0, 1]
            colors = (samples * 255).astype(np.ubyte)
            _rainforest_cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=colors)
        else:
            _rainforest_cmap = pg.colormap.get("viridis")
    return _rainforest_cmap


def hist_to_rgba(hist, *, log=False, gamma=1.0):
    """2D histogram -> (rows, cols, 4) uint8 RGBA array, ready for
    `ImageItem.setImage(rgba, autoLevels=False)`. See the module
    docstring for the orientation this expects.

    log=True reproduces matplotlib's LogNorm default of masking
    non-positive bins as fully transparent (log-scaled color otherwise);
    the linear/gamma path leaves zero mapped to the colormap's lowest
    color, same as plain imshow / PowerNorm.
    """
    data = np.asarray(hist, dtype=np.float64).T
    lut = rainforest_colormap().getLookupTable(0.0, 1.0, 256)

    if log:
        positive = data[data > 0]
        if positive.size == 0:
            return np.zeros((*data.shape, 4), dtype=np.ubyte)
        vmin = max(float(positive.min()), 1e-3)
        vmax = max(float(positive.max()), vmin * 1.1)
        norm = (np.log(np.clip(data, vmin, None)) - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
        alpha = np.where(data > 0, 255, 0).astype(np.ubyte)
    else:
        vmax = float(data.max())
        if vmax <= 0:
            return np.zeros((*data.shape, 4), dtype=np.ubyte)
        norm = np.clip(data, 0.0, None) / vmax
        if abs(gamma - 1.0) > 1e-6:
            norm = norm ** gamma
        alpha = np.full(data.shape, 255, dtype=np.ubyte)

    idx = np.clip((np.nan_to_num(norm) * 255.0).astype(np.intp), 0, 255)
    rgb = lut[idx]
    return np.dstack([rgb, alpha])


class ZoomFocusViewBox(pg.ViewBox):
    """A ViewBox where left-drag draws a zoom rectangle (pyqtgraph's
    native equivalent of matplotlib's RectangleSelector), right-click
    resets the view instead of opening the context menu, and left
    double-click toggles single-plot focus -- matching plot_panel.py's
    original matplotlib mouse handling. `on_reset`/`on_focus` are plain
    no-arg callbacks; either may be left as None where that behavior
    isn't wanted (only plot_panel.py wires up `on_focus`; a `None`
    `on_reset` just falls back to autoRange())."""

    def __init__(self, *args, on_reset=None, on_focus=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_reset = on_reset
        self._on_focus = on_focus
        self.setMouseMode(pg.ViewBox.RectMode)
        self.setMenuEnabled(False)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.RightButton:
            ev.ignore()
            return
        super().mouseDragEvent(ev, axis=axis)

    def mouseClickEvent(self, ev):
        if ev.double() and ev.button() == QtCore.Qt.LeftButton and self._on_focus is not None:
            ev.accept()
            self._on_focus()
            return
        if ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            if self._on_reset is not None:
                self._on_reset()
            else:
                self.autoRange()
            return
        super().mouseClickEvent(ev)


def circle_curve(radius, n=128):
    """(x, y) arrays tracing a circle of the given radius, centered at
    the origin -- for drawing dashed order-ring overlays via
    PlotDataItem (replaces matplotlib's `Circle` patch)."""
    theta = np.linspace(0.0, 2 * np.pi, n)
    return radius * np.cos(theta), radius * np.sin(theta)
