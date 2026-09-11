"""A small tkinter/ttk-API-compatible layer backed by real PyQt5 widgets.

This is NOT a general-purpose tkinter emulator -- it covers exactly the
subset of the tkinter/ttk surface this project's UI code used (grid
geometry, the Tk variable classes + trace_add, the handful of widgets and
dialogs, a couple of event bindings), so that most of each interface file's
widget-construction and variable-wiring code ports over with only an import
change (`import tkinter as tk` -> `import qtk as tk`,
`from tkinter import ttk` -> `from qtk import ttk`), while every widget on
screen is a genuine QWidget rendered by Qt5. Anything not covered here
(matplotlib canvas, the scrollable sidebar) is written directly in native
PyQt5 in its own module rather than shimmed.

Layout: `.grid(row=, column=, rowspan=, columnspan=, sticky=, padx=, pady=)`
on any widget adds it to a QGridLayout owned by its Qt parent (created
lazily on first use). `sticky` is translated to a per-axis Qt alignment:
an axis with both of its edge letters (e.g. "ew") means "stretch to fill
that axis" (Qt: leave that axis's alignment unset, which fills it), a
single edge letter anchors to that edge, and no letters for an axis means
center -- this matches tkinter's own sticky semantics.
"""
import pathlib
import re
import sys

from PyQt5 import QtCore, QtGui, QtWidgets


# ---------------------------------------------------------------------------
# Tk variables
# ---------------------------------------------------------------------------

class Variable:
    """Stand-in for tkinter.Variable: plain Python object (no QObject/
    QApplication ordering concerns), get()/set()/trace_add("write", cb) --
    trace callbacks fire on every set(), same as Tcl's write trace."""

    def __init__(self, master=None, value=None, name=None):
        del master, name
        self._value = self._coerce(value) if value is not None else self._default()
        self._traces = []

    def _default(self):
        return None

    def _coerce(self, value):
        return value

    def get(self):
        return self._value

    def set(self, value):
        self._value = self._coerce(value)
        for cb in list(self._traces):
            cb("", "", "write")

    def trace_add(self, mode, callback):
        assert mode == "write", f"unsupported trace mode: {mode}"
        self._traces.append(callback)
        return callback

    def trace_remove(self, mode, cbname):
        del mode
        try:
            self._traces.remove(cbname)
        except ValueError:
            pass


class StringVar(Variable):
    def _default(self):
        return ""

    def _coerce(self, value):
        return "" if value is None else str(value)


class BooleanVar(Variable):
    def _default(self):
        return False

    def _coerce(self, value):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)


class IntVar(Variable):
    def _default(self):
        return 0

    def _coerce(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(float(value))


class DoubleVar(Variable):
    def _default(self):
        return 0.0

    def _coerce(self, value):
        return float(value)


class TclError(Exception):
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sticky_to_alignment(sticky):
    s = (sticky or "").lower()
    has_w, has_e = "w" in s, "e" in s
    has_n, has_s = "n" in s, "s" in s
    flags = QtCore.Qt.Alignment()
    if has_w and not has_e:
        flags |= QtCore.Qt.AlignLeft
    elif has_e and not has_w:
        flags |= QtCore.Qt.AlignRight
    elif not has_w and not has_e:
        flags |= QtCore.Qt.AlignHCenter
    if has_n and not has_s:
        flags |= QtCore.Qt.AlignTop
    elif has_s and not has_n:
        flags |= QtCore.Qt.AlignBottom
    elif not has_n and not has_s:
        flags |= QtCore.Qt.AlignVCenter
    return flags


_FIXED_FONT_ALIASES = {"tkfixedfont", "courier", "courier new"}
_DEFAULT_FONT_ALIASES = {"tkdefaultfont"}


def _apply_font(widget, font_spec):
    if not font_spec:
        return
    if isinstance(font_spec, str):
        widget.setFont(QtGui.QFont(font_spec))
        return
    family = font_spec[0]
    size = font_spec[1] if len(font_spec) > 1 else None
    style = font_spec[2] if len(font_spec) > 2 else ""
    family_key = str(family).strip().lower()
    if family_key in _FIXED_FONT_ALIASES:
        f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
    elif family_key in _DEFAULT_FONT_ALIASES:
        f = QtWidgets.QApplication.font()
    else:
        f = QtGui.QFont(family)
    if size:
        f.setPointSize(abs(int(size)))
    if style and "bold" in style:
        f.setBold(True)
    if style and "italic" in style:
        f.setItalic(True)
    widget.setFont(f)


_ANCHOR_MAP = {
    "n": QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
    "s": QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom,
    "e": QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
    "w": QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
    "ne": QtCore.Qt.AlignRight | QtCore.Qt.AlignTop,
    "nw": QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
    "se": QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom,
    "sw": QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom,
    "center": QtCore.Qt.AlignCenter,
}
_JUSTIFY_MAP = {
    "left": QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
    "right": QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
    "center": QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
}


def _apply_width_chars(widget, chars):
    if not chars:
        return
    fm = widget.fontMetrics()
    widget.setMinimumWidth(fm.horizontalAdvance("0") * int(chars) + 14)


# ---------------------------------------------------------------------------
# grid geometry manager + after()/bind()/winfo_* mixins
# ---------------------------------------------------------------------------

class GridMixin:
    """`.grid()` places this widget in a QGridLayout owned by its Qt
    parent (created lazily); `.columnconfigure()`/`.rowconfigure()`
    configure the QGridLayout this widget owns for placing ITS children."""

    def _own_layout(self):
        layout = self.layout()
        if layout is None:
            layout = QtWidgets.QGridLayout(self)
            self.setLayout(layout)
        if not isinstance(layout, QtWidgets.QGridLayout):
            raise TclError(f"{self!r} already has a non-grid layout")
        return layout

    def grid(self, row=0, column=0, rowspan=1, columnspan=1, sticky="", padx=0, pady=0, **_kw):
        parent = self.parentWidget()
        if parent is None:
            raise TclError("grid() called on a widget with no parent")
        layout = parent._own_layout() if isinstance(parent, GridMixin) else _grid_layout_for(parent)

        target = self
        px = tuple(padx) if isinstance(padx, (tuple, list)) else (padx, padx)
        py = tuple(pady) if isinstance(pady, (tuple, list)) else (pady, pady)
        if any(px) or any(py):
            container = getattr(self, "_pad_container", None)
            if container is None:
                container = QtWidgets.QWidget(parent)
                inner = QtWidgets.QVBoxLayout(container)
                inner.setSpacing(0)
                inner.addWidget(self)
                self._pad_container = container
            container.layout().setContentsMargins(px[0], py[0], px[1], py[1])
            target = container

        layout.addWidget(target, row, column, rowspan, columnspan, _sticky_to_alignment(sticky))
        return self

    def grid_remove(self):
        parent = self.parentWidget()
        target = getattr(self, "_pad_container", None) or self
        if parent is not None and parent.layout() is not None:
            parent.layout().removeWidget(target)
        target.setParent(None)

    grid_forget = grid_remove

    def columnconfigure(self, index, weight=None, minsize=None, **_kw):
        layout = self._own_layout()
        if weight is not None:
            layout.setColumnStretch(index, weight)
        if minsize is not None:
            layout.setColumnMinimumWidth(index, minsize)

    def rowconfigure(self, index, weight=None, minsize=None, **_kw):
        layout = self._own_layout()
        if weight is not None:
            layout.setRowStretch(index, weight)
        if minsize is not None:
            layout.setRowMinimumHeight(index, minsize)

    def pack(self, side="top", padx=0, pady=0, **_kw):
        """Only `side` (top/bottom/left/right) + padx/pady are supported --
        the only pack() usage in this codebase is a simple left-to-right
        toolbar row."""
        parent = self.parentWidget()
        if parent is None:
            raise TclError("pack() called on a widget with no parent")
        layout = parent.layout()
        horizontal = side in ("left", "right")
        if layout is None:
            layout = QtWidgets.QHBoxLayout(parent) if horizontal else QtWidgets.QVBoxLayout(parent)
            parent.setLayout(layout)
        if not isinstance(layout, (QtWidgets.QHBoxLayout, QtWidgets.QVBoxLayout)):
            raise TclError(f"{parent!r} already has a non-pack layout")

        target = self
        px = tuple(padx) if isinstance(padx, (tuple, list)) else (padx, padx)
        py = tuple(pady) if isinstance(pady, (tuple, list)) else (pady, pady)
        if any(px) or any(py):
            container = QtWidgets.QWidget(parent)
            inner = QtWidgets.QVBoxLayout(container)
            inner.setSpacing(0)
            inner.setContentsMargins(px[0], py[0], px[1], py[1])
            inner.addWidget(self)
            target = container

        if side in ("left", "top"):
            layout.addWidget(target)
        else:
            layout.insertWidget(0, target)


def _grid_layout_for(widget):
    layout = widget.layout()
    if layout is None:
        layout = QtWidgets.QGridLayout(widget)
        widget.setLayout(layout)
    if not isinstance(layout, QtWidgets.QGridLayout):
        raise TclError(f"{widget!r} already has a non-grid layout")
    return layout


def grid_into(widget, parent, row=0, column=0, rowspan=1, columnspan=1, sticky="", padx=0, pady=0):
    """Grid an arbitrary QWidget that isn't a GridMixin -- namely a
    matplotlib FigureCanvasQTAgg, which is a real QWidget in its own
    right -- into `parent`'s grid layout."""
    layout = parent._own_layout() if isinstance(parent, GridMixin) else _grid_layout_for(parent)

    target = widget
    px = tuple(padx) if isinstance(padx, (tuple, list)) else (padx, padx)
    py = tuple(pady) if isinstance(pady, (tuple, list)) else (pady, pady)
    if any(px) or any(py):
        container = QtWidgets.QWidget(parent)
        inner = QtWidgets.QVBoxLayout(container)
        inner.setSpacing(0)
        inner.setContentsMargins(px[0], py[0], px[1], py[1])
        inner.addWidget(widget)
        target = container

    layout.addWidget(target, row, column, rowspan, columnspan, _sticky_to_alignment(sticky))


class _FakeEvent:
    def __init__(self, widget):
        self.widget = widget


class TkCompatMixin:
    """`.after()`/`.after_cancel()` (QTimer-backed), a small `.bind()` for
    the handful of event sequences this project uses, and the `winfo_*`
    geometry queries used by the one hand-rolled dialog (collection_interface's
    "folder exists" prompt)."""

    def after(self, ms, callback=None, *args):
        if callback is None:
            return None
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)

        def _fire():
            getattr(self, "_after_timers", {}).pop(after_id, None)
            callback(*args)

        timer.timeout.connect(_fire)
        if not hasattr(self, "_after_timers"):
            self._after_timers = {}
        after_id = id(timer)
        self._after_timers[after_id] = timer
        timer.start(max(0, int(ms)))
        return after_id

    def after_cancel(self, after_id):
        timers = getattr(self, "_after_timers", None)
        if not timers:
            return
        timer = timers.pop(after_id, None)
        if timer is not None:
            timer.stop()

    def bind(self, sequence, func, add=None):
        del add
        if not hasattr(self, "_tk_bindings"):
            self._tk_bindings = {}
        self._tk_bindings.setdefault(sequence, []).append(func)
        installed = getattr(self, "_installed_bindings", None)
        if installed is None:
            installed = self._installed_bindings = set()
        if sequence not in installed:
            installed.add(sequence)
            self._install_binding(sequence)

    def _install_binding(self, sequence):
        pass

    def _fire_binding(self, sequence, event=None):
        for func in getattr(self, "_tk_bindings", {}).get(sequence, []):
            func(event if event is not None else _FakeEvent(self))

    def winfo_toplevel(self):
        return self.window()

    def winfo_rootx(self):
        return self.mapToGlobal(QtCore.QPoint(0, 0)).x()

    def winfo_rooty(self):
        return self.mapToGlobal(QtCore.QPoint(0, 0)).y()

    def winfo_width(self):
        return self.width()

    def winfo_height(self):
        return self.height()

    def update_idletasks(self):
        QtWidgets.QApplication.processEvents()

    def winfo_children(self):
        return [c for c in self.children() if isinstance(c, QtWidgets.QWidget)]

    def lift(self):
        self.raise_()

    def destroy(self):
        parent = self.parentWidget()
        target = getattr(self, "_pad_container", None) or self
        if parent is not None and parent.layout() is not None:
            parent.layout().removeWidget(target)
        target.setParent(None)
        target.deleteLater()


# ---------------------------------------------------------------------------
# root window
# ---------------------------------------------------------------------------

_qapp = None


def _ensure_app():
    global _qapp
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle("Fusion")
    _qapp = app
    return app


class Tk(TkCompatMixin, GridMixin, QtWidgets.QWidget):
    def __init__(self):
        _ensure_app()
        super().__init__(None)

    def title(self, text):
        self.setWindowTitle(text)

    def minsize(self, w, h):
        self.setMinimumSize(w, h)

    def mainloop(self):
        w = max(self.minimumWidth(), self.sizeHint().width())
        h = max(self.minimumHeight(), self.sizeHint().height())
        self.resize(w, h)
        self.show()
        _ensure_app().exec_()


# ---------------------------------------------------------------------------
# Toplevel (used by collection_interface's "folder exists" dialog)
# ---------------------------------------------------------------------------

class Toplevel(TkCompatMixin, GridMixin, QtWidgets.QDialog):
    def __init__(self, parent=None, **_kw):
        super().__init__(parent.window() if parent is not None else None)
        self._close_callback = None

    def title(self, text):
        self.setWindowTitle(text)

    def resizable(self, x, y):
        del x, y

    def transient(self, parent):
        del parent

    def grab_set(self):
        self.setModal(True)

    def protocol(self, name, callback):
        if name == "WM_DELETE_WINDOW":
            self._close_callback = callback

    def closeEvent(self, event):
        if self._close_callback is not None:
            self._close_callback()
            event.accept()
        else:
            super().closeEvent(event)

    def geometry(self, spec=None):
        if spec is None:
            g = self.frameGeometry()
            return f"{g.width()}x{g.height()}+{g.x()}+{g.y()}"
        m = re.match(r"\+(-?\d+)\+(-?\d+)", spec)
        if m:
            self.move(int(m.group(1)), int(m.group(2)))

    def update_idletasks(self):
        # Not shown yet at this point in the usual build-then-center-then-show
        # dialog flow, so force the layout to compute a real sizeHint-based
        # size -- otherwise winfo_width()/winfo_height() right after this
        # would report Qt's pre-layout default size instead.
        self.adjustSize()
        super().update_idletasks()

    def wait_window(self):
        self.exec_()

    def destroy(self):
        self.accept()


# ---------------------------------------------------------------------------
# ttk-equivalent widgets
# ---------------------------------------------------------------------------

class Frame(TkCompatMixin, GridMixin, QtWidgets.QWidget):
    def __init__(self, parent=None, **_kw):
        super().__init__(parent)


class LabelFrame(TkCompatMixin, GridMixin, QtWidgets.QGroupBox):
    def __init__(self, parent=None, text="", **_kw):
        super().__init__(parent)
        self.setTitle(text)


class Label(TkCompatMixin, GridMixin, QtWidgets.QLabel):
    def __init__(self, parent=None, text="", textvariable=None, font=None, wraplength=None,
                 justify=None, anchor=None, width=None, relief=None, padding=None,
                 foreground=None, **_kw):
        super().__init__(parent)
        self._var = textvariable
        if textvariable is not None:
            self.setText(str(textvariable.get()))
            textvariable.trace_add("write", self._on_var_write)
        else:
            self.setText(str(text))
        if font:
            _apply_font(self, font)
        if wraplength:
            self.setWordWrap(True)
            self.setMaximumWidth(wraplength)
        align = _ANCHOR_MAP.get(anchor) if anchor else (_JUSTIFY_MAP.get(justify) if justify else None)
        if align:
            self.setAlignment(align)
        if width:
            _apply_width_chars(self, width)
        if relief == "sunken":
            self.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        if padding:
            px, py = padding if isinstance(padding, (tuple, list)) and len(padding) == 2 else (padding, padding)
            self.setContentsMargins(px, py, px, py)
        if foreground:
            self.setStyleSheet(f"color: {foreground};")

    def _on_var_write(self, *_):
        self.setText(str(self._var.get()))

    def configure(self, **kw):
        if "text" in kw:
            self.setText(str(kw["text"]))
        if "foreground" in kw:
            self.setStyleSheet(f"color: {kw['foreground']};")

    config = configure


class Button(TkCompatMixin, GridMixin, QtWidgets.QPushButton):
    def __init__(self, parent=None, text="", command=None, width=None, textvariable=None, **_kw):
        super().__init__(parent)
        self._var = textvariable
        if textvariable is not None:
            self.setText(str(textvariable.get()))
            textvariable.trace_add("write", self._on_var_write)
        else:
            self.setText(str(text))
        if command is not None:
            self.clicked.connect(lambda _checked=False: command())
        if width:
            _apply_width_chars(self, width)

    def _on_var_write(self, *_):
        self.setText(str(self._var.get()))

    def configure(self, **kw):
        if "text" in kw:
            self.setText(str(kw["text"]))

    config = configure


class Checkbutton(TkCompatMixin, GridMixin, QtWidgets.QCheckBox):
    def __init__(self, parent=None, text="", variable=None, command=None, **_kw):
        super().__init__(text, parent)
        self._var = variable
        self._command = command
        self._updating = False
        if variable is not None:
            self.setChecked(bool(variable.get()))
            variable.trace_add("write", self._on_var_write)
        self.stateChanged.connect(self._on_state_changed)

    def _on_var_write(self, *_):
        self._updating = True
        try:
            self.setChecked(bool(self._var.get()))
        finally:
            self._updating = False

    def _on_state_changed(self, state):
        if self._updating:
            return
        if self._var is not None:
            self._var.set(state == QtCore.Qt.Checked)
        if self._command is not None:
            self._command()


class Radiobutton(TkCompatMixin, GridMixin, QtWidgets.QRadioButton):
    def __init__(self, parent=None, text="", variable=None, value=None, command=None, **_kw):
        super().__init__(text, parent)
        self._var = variable
        self._value = value
        self._command = command
        self._updating = False
        if variable is not None:
            self.setChecked(variable.get() == value)
            variable.trace_add("write", self._on_var_write)
        self.toggled.connect(self._on_toggled)

    def _on_var_write(self, *_):
        self._updating = True
        try:
            self.setChecked(self._var.get() == self._value)
        finally:
            self._updating = False

    def _on_toggled(self, checked):
        if self._updating or not checked:
            return
        if self._var is not None:
            self._var.set(self._value)
        if self._command is not None:
            self._command()


class Entry(TkCompatMixin, GridMixin, QtWidgets.QLineEdit):
    def __init__(self, parent=None, textvariable=None, width=None, show=None, justify=None, **_kw):
        super().__init__(parent)
        self._var = textvariable
        self._updating = False
        if textvariable is not None:
            self.setText(str(textvariable.get()))
            textvariable.trace_add("write", self._on_var_write)
        self.textEdited.connect(self._on_text_edited)
        if width:
            _apply_width_chars(self, width)
        if show:
            self.setEchoMode(QtWidgets.QLineEdit.Password)
        if justify:
            self.setAlignment(_JUSTIFY_MAP.get(justify, QtCore.Qt.AlignLeft))

    def _on_var_write(self, *_):
        value = str(self._var.get())
        if self.text() != value:
            self._updating = True
            try:
                self.setText(value)
            finally:
                self._updating = False

    def _on_text_edited(self, text):
        if self._updating:
            return
        if self._var is not None:
            self._var.set(text)

    def _install_binding(self, sequence):
        if sequence == "<Return>":
            self.returnPressed.connect(lambda: self._fire_binding("<Return>"))

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self._fire_binding("<FocusOut>")


class Combobox(TkCompatMixin, GridMixin, QtWidgets.QComboBox):
    def __init__(self, parent=None, textvariable=None, values=None, state=None, width=None, **_kw):
        super().__init__(parent)
        self._var = textvariable
        self._updating = False
        if values:
            self.addItems([str(v) for v in values])
        self.setEditable(state != "readonly")
        if textvariable is not None:
            self.setCurrentText(str(textvariable.get()))
            textvariable.trace_add("write", self._on_var_write)
        self.currentTextChanged.connect(self._on_text_changed)
        if width:
            _apply_width_chars(self, width)

    def _on_var_write(self, *_):
        value = str(self._var.get())
        if self.currentText() != value:
            self._updating = True
            try:
                self.setCurrentText(value)
            finally:
                self._updating = False

    def _on_text_changed(self, text):
        if self._updating:
            return
        if self._var is not None:
            self._var.set(text)

    def configure(self, **kw):
        if "values" in kw:
            current = self.currentText()
            self.blockSignals(True)
            self.clear()
            self.addItems([str(v) for v in kw["values"]])
            self.setCurrentText(current)
            self.blockSignals(False)
        if "state" in kw:
            self.setEditable(kw["state"] != "readonly")

    config = configure

    def _install_binding(self, sequence):
        if sequence == "<<ComboboxSelected>>":
            self.activated.connect(lambda *_a: self._fire_binding("<<ComboboxSelected>>"))


class Scale(TkCompatMixin, GridMixin, QtWidgets.QSlider):
    """Float-valued slider (ttk.Scale is continuous; QSlider is integer),
    mapped onto an internal 0..STEPS integer range."""

    _STEPS = 1000

    def __init__(self, parent=None, from_=0.0, to=1.0, orient="horizontal", variable=None,
                 length=None, command=None, **_kw):
        qt_orient = QtCore.Qt.Horizontal if orient == "horizontal" else QtCore.Qt.Vertical
        super().__init__(qt_orient, parent)
        self._from, self._to = float(from_), float(to)
        self.setMinimum(0)
        self.setMaximum(self._STEPS)
        self._var = variable
        self._command = command
        self._updating = False
        if length:
            (self.setMinimumWidth if qt_orient == QtCore.Qt.Horizontal else self.setMinimumHeight)(length)
        if variable is not None:
            self.setValue(self._to_step(variable.get()))
            variable.trace_add("write", self._on_var_write)
        self.valueChanged.connect(self._on_value_changed)

    def _to_step(self, value):
        value = min(max(value, self._from), self._to)
        span = self._to - self._from
        frac = (value - self._from) / span if span else 0
        return int(round(frac * self._STEPS))

    def _from_step(self, step):
        return self._from + (step / self._STEPS) * (self._to - self._from)

    def _on_var_write(self, *_):
        self._updating = True
        try:
            self.setValue(self._to_step(self._var.get()))
        finally:
            self._updating = False

    def _on_value_changed(self, step):
        if self._updating:
            return
        value = self._from_step(step)
        if self._var is not None:
            self._var.set(value)
        if self._command is not None:
            self._command(value)


class Progressbar(TkCompatMixin, GridMixin, QtWidgets.QProgressBar):
    _STEPS = 1000

    def __init__(self, parent=None, variable=None, maximum=100.0, mode="determinate", length=None, **_kw):
        del mode
        super().__init__(parent)
        self.setRange(0, self._STEPS)
        self.setTextVisible(False)
        self._maximum = float(maximum) if maximum else 100.0
        self._var = variable
        if length:
            self.setMinimumWidth(length)
        if variable is not None:
            self._on_var_write()
            variable.trace_add("write", self._on_var_write)

    def _on_var_write(self, *_):
        value = float(self._var.get()) if self._var is not None else 0.0
        frac = 0.0 if self._maximum <= 0 else max(0.0, min(1.0, value / self._maximum))
        self.setValue(int(round(frac * self._STEPS)))


class Separator(GridMixin, QtWidgets.QFrame):
    def __init__(self, parent=None, orient="horizontal", **_kw):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.HLine if orient == "horizontal" else QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class Notebook(TkCompatMixin, GridMixin, QtWidgets.QTabWidget):
    def __init__(self, parent=None, **_kw):
        super().__init__(parent)

    def add(self, child, text=""):
        self.addTab(child, text)


class Treeview(TkCompatMixin, GridMixin, QtWidgets.QTreeWidget):
    def __init__(self, parent=None, columns=(), show="headings", height=None, **_kw):
        del show
        super().__init__(parent)
        self._columns = list(columns)
        self.setColumnCount(len(self._columns))
        self.setHeaderLabels([str(c) for c in self._columns])
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        if height:
            fm = self.fontMetrics()
            self.setMinimumHeight(fm.lineSpacing() * height + 34)

    def heading(self, col, text=""):
        self.headerItem().setText(self._columns.index(col), text)

    def column(self, col, width=None, anchor=None, **_kw):
        del anchor
        if width is not None:
            self.setColumnWidth(self._columns.index(col), width)

    def configure(self, **_kw):
        pass  # yscrollcommand etc: QTreeWidget already has its own scrollbar

    config = configure

    def insert(self, parent_id, index, values=()):
        del parent_id, index
        item = QtWidgets.QTreeWidgetItem([str(v) for v in values])
        self.addTopLevelItem(item)
        return item

    def get_children(self, item=""):
        del item
        return [self.topLevelItem(i) for i in range(self.topLevelItemCount())]

    def delete(self, *items):
        for it in items:
            idx = self.indexOfTopLevelItem(it)
            if idx >= 0:
                self.takeTopLevelItem(idx)

    def see(self, item):
        self.scrollToItem(item)


class Text(TkCompatMixin, GridMixin, QtWidgets.QTextEdit):
    def __init__(self, parent=None, wrap="word", height=None, width=None, font=None, **_kw):
        super().__init__(parent)
        self.setLineWrapMode(
            QtWidgets.QTextEdit.WidgetWidth if wrap == "word" else QtWidgets.QTextEdit.NoWrap
        )
        if font:
            _apply_font(self, font)
        if height:
            fm = self.fontMetrics()
            self.setMinimumHeight(fm.lineSpacing() * height + 12)
        if width:
            _apply_width_chars(self, width)

    def insert(self, index, text):
        cursor = self.textCursor()
        cursor.movePosition(
            QtGui.QTextCursor.End if index == "end" else QtGui.QTextCursor.Start
        )
        cursor.insertText(text)
        self.setTextCursor(cursor)

    def _line_start_cursor(self, line_number):
        cursor = QtGui.QTextCursor(self.document())
        cursor.movePosition(QtGui.QTextCursor.Start)
        if line_number > 1:
            cursor.movePosition(QtGui.QTextCursor.Down, QtGui.QTextCursor.MoveAnchor, line_number - 1)
        return cursor

    def delete(self, start, end):
        if start == "1.0" and end == "end":
            self.clear()
            return
        # Only other pattern used in this codebase: "1.0" .. "<line>.0",
        # trimming a rolling log down to its last N lines.
        start_line = 1 if start == "1.0" else int(str(start).split(".")[0])
        end_line = int(str(end).split(".")[0])
        cursor = self._line_start_cursor(start_line)
        cursor.setPosition(self._line_start_cursor(end_line).position(), QtGui.QTextCursor.KeepAnchor)
        cursor.removeSelectedText()

    def get(self, start, end):
        del start, end  # "end" and "end-1c" both resolve to the full text
        return self.toPlainText()

    def index(self, spec):
        # Only the line number (the part before ".") is ever consumed by
        # callers in this codebase.
        del spec
        return f"{self.document().blockCount()}.0"

    def see(self, index):
        del index  # only "end" is used here
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def configure(self, **kw):
        if "state" in kw:
            self.setReadOnly(kw["state"] == "disabled")

    config = configure

    def edit_modified(self, value=None):
        if value is None:
            return self.document().isModified()
        self.document().setModified(value)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self._fire_binding("<FocusOut>")


class Style:
    """No-op: the "clam" ttk theme is replaced by QApplication's Fusion
    style, set once in Tk.__init__."""

    def __init__(self, master=None):
        del master

    def theme_use(self, name=None):
        del name


# ---------------------------------------------------------------------------
# messagebox / filedialog
# ---------------------------------------------------------------------------

class messagebox:
    @staticmethod
    def _parent_window(parent):
        return parent.window() if parent is not None else None

    @staticmethod
    def askyesno(title, message, parent=None, **_kw):
        result = QtWidgets.QMessageBox.question(
            messagebox._parent_window(parent), title, message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return result == QtWidgets.QMessageBox.Yes

    @staticmethod
    def showinfo(title, message, parent=None, **_kw):
        QtWidgets.QMessageBox.information(messagebox._parent_window(parent), title, message)

    @staticmethod
    def showerror(title, message, parent=None, **_kw):
        QtWidgets.QMessageBox.critical(messagebox._parent_window(parent), title, message)

    @staticmethod
    def showwarning(title, message, parent=None, **_kw):
        QtWidgets.QMessageBox.warning(messagebox._parent_window(parent), title, message)


class filedialog:
    @staticmethod
    def _parent_window(parent):
        return parent.window() if parent is not None else None

    @staticmethod
    def _filter_string(filetypes):
        if not filetypes:
            return ""
        parts = []
        for label, pattern in filetypes:
            patterns = pattern if isinstance(pattern, (list, tuple)) else (pattern,)
            parts.append(f"{label} ({' '.join(patterns)})")
        return ";;".join(parts)

    @staticmethod
    def _start_path(initialdir=None, initialfile=None):
        base = pathlib.Path(initialdir) if initialdir else None
        if base is not None and initialfile:
            return str(base / initialfile)
        if base is not None:
            return str(base)
        return initialfile or ""

    @staticmethod
    def askdirectory(title=None, initialdir=None, parent=None, **_kw):
        return QtWidgets.QFileDialog.getExistingDirectory(
            filedialog._parent_window(parent), title or "", initialdir or "",
        )

    @staticmethod
    def askopenfilename(title=None, filetypes=None, initialdir=None, initialfile=None, parent=None, **_kw):
        path, _sel = QtWidgets.QFileDialog.getOpenFileName(
            filedialog._parent_window(parent), title or "",
            filedialog._start_path(initialdir, initialfile),
            filedialog._filter_string(filetypes),
        )
        return path

    @staticmethod
    def askopenfilenames(title=None, filetypes=None, initialdir=None, parent=None, **_kw):
        paths, _sel = QtWidgets.QFileDialog.getOpenFileNames(
            filedialog._parent_window(parent), title or "", initialdir or "",
            filedialog._filter_string(filetypes),
        )
        return tuple(paths)

    @staticmethod
    def asksaveasfilename(title=None, filetypes=None, initialdir=None, initialfile=None,
                           defaultextension=None, parent=None, **_kw):
        path, _sel = QtWidgets.QFileDialog.getSaveFileName(
            filedialog._parent_window(parent), title or "",
            filedialog._start_path(initialdir, initialfile),
            filedialog._filter_string(filetypes),
        )
        if path and defaultextension and not pathlib.Path(path).suffix:
            path += defaultextension
        return path


# ---------------------------------------------------------------------------
# `from qtk import ttk` namespace
# ---------------------------------------------------------------------------

class ttk:
    Frame = Frame
    LabelFrame = LabelFrame
    Label = Label
    Button = Button
    Checkbutton = Checkbutton
    Radiobutton = Radiobutton
    Entry = Entry
    Combobox = Combobox
    Scale = Scale
    Progressbar = Progressbar
    Separator = Separator
    Notebook = Notebook
    Treeview = Treeview
    Style = Style
