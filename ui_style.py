"""Shared building blocks for the app's look, as set by the Monitored
Acquisition tab -- see STYLE_GUIDE.md for the rules these implement.

Tabs build their layout out of these instead of re-deriving margins,
colors and spacing on their own, so every tab stays on the same
grey/grey/white surface hierarchy, spacing scale and component set.
"""
import html

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

import qtk as tk
from qtk import ttk

import shared_state
from cluster_timing import MODES as CLUSTER_TIMING_MODES
from collapsible_frame import CollapsibleFrame, FLUSH_BACKGROUND, set_background
from scrollable_frame import ScrollableFrame

# ---- tokens -------------------------------------------------------------

SMALL_FONT = ("TkDefaultFont", 8)  # the platform's UI font, at 8 pt
SIDEBAR_WIDTH = 400
PAGE_MAX_WIDTH = 640

WHITE = QtGui.QColor(QtCore.Qt.white)
DANGER = "#c62828"
DANGER_HOVER = "#d32f2f"
DANGER_PRESSED = "#8e0000"
RUNNING = "#2e7d32"
WARNING = "#ef6c00"
IDLE = "#9e9e9e"
PLACEHOLDER_RGB = (140, 140, 140)

# Marks a note that has more to say on hover (see add_note / set_tip).
TIP_MARK = " ⓘ"
TIP_WIDTH = 380

EMPTY_PLOT_TEXT = "No data — press Start to acquire"

DANGER_BUTTON_STYLE = (
    "QPushButton {"
    f"  background-color: {DANGER}; color: white; font-weight: bold;"
    f"  border: 1px solid {DANGER_PRESSED}; border-radius: 3px; padding: 4px 10px;"
    "}"
    f"QPushButton:hover {{ background-color: {DANGER_HOVER}; }}"
    f"QPushButton:pressed {{ background-color: {DANGER_PRESSED}; }}"
)


# ---- page layouts -------------------------------------------------------

def zero_margins(widget, spacing=False):
    """Qt's default QGridLayout margins (and, separately, its default
    spacing) leave structural containers inset from their parent's real
    edges -- visible as a gap next to a divider rule or between a sidebar
    and the plots. Breathing room comes from explicit padx/pady instead."""
    layout = widget._own_layout() if hasattr(widget, "_own_layout") else widget.layout()
    layout.setContentsMargins(0, 0, 0, 0)
    if spacing:
        layout.setSpacing(0)


def _flush_scroll_area(parent, width):
    scroll = ScrollableFrame(parent, width=width)
    # QScrollArea and its viewport don't inherit a background on their own.
    set_background(scroll, FLUSH_BACKGROUND)
    set_background(scroll.viewport(), FLUSH_BACKGROUND)
    scroll.body.columnconfigure(0, weight=1)
    return scroll


def build_sidebar_layout(owner, title="Controls", width=SIDEBAR_WIDTH):
    """The standard tab layout: a collapsible, scrollable, flush grey
    "Controls" sidebar on the left (a narrow vertical toggle strip, set off
    from the rest by a single rule), and a white main area filling the
    rest. Collapsing the sidebar hands its whole width back to the main
    area. The sidebar's width leaves room for its widest row plus the
    vertical scrollbar once every section is expanded (horizontal
    scrolling is intentionally off).

    Returns (sidebar_body, main); build controls into sidebar_body (one
    column, stacked rows) and the data view (plots, tables, logs) into
    main. Call main.rowconfigure() for whichever row should fill."""
    owner.columnconfigure(1, weight=1)
    owner.rowconfigure(0, weight=1)
    zero_margins(owner, spacing=True)

    section = CollapsibleFrame(owner, text=title, flush=True, separator="right", vertical_label=True)
    section.grid(row=0, column=0, sticky="nsew")
    section.body.columnconfigure(0, weight=1)
    section.body.rowconfigure(0, weight=1)
    scroll = _flush_scroll_area(section.body, width)
    scroll.grid(row=0, column=0, sticky="nsew")

    main = ttk.Frame(owner)
    main.grid(row=0, column=1, sticky="nsew")
    main.columnconfigure(0, weight=1)
    zero_margins(main, spacing=True)
    set_background(main, WHITE)
    return scroll.body, main


def build_page_layout(owner, max_width=PAGE_MAX_WIDTH):
    """For a tab with no data view to give the space to (just forms and
    actions): the sidebar's content column as a whole, scrollable, flush
    grey page, capped at `max_width` and left-aligned so fields don't
    stretch across a wide window. Returns the column to build into."""
    owner.columnconfigure(0, weight=1)
    owner.rowconfigure(0, weight=1)
    zero_margins(owner, spacing=True)

    scroll = _flush_scroll_area(owner, 300)
    scroll.grid(row=0, column=0, sticky="nsew")
    outer = scroll.body
    # Column 0 is capped by the page's maximum width; column 1 soaks up
    # whatever's left, which keeps the page left-aligned.
    outer.columnconfigure(1, weight=1)
    outer.rowconfigure(1, weight=1)

    page = ttk.Frame(outer)
    page.setMaximumWidth(max_width)
    page.grid(row=0, column=0, sticky="new")
    page.columnconfigure(0, weight=1)
    return page


# ---- sidebar components -------------------------------------------------

def style_danger_button(button):
    button.setStyleSheet(DANGER_BUTTON_STYLE)


class FlowLayout(QtWidgets.QLayout):
    """Lays its items out left to right at their size hints, wrapping onto
    another row whenever the next one wouldn't fit the width it's given --
    so a row of buttons never runs out of a fixed-width sidebar (which
    doesn't scroll sideways) however many there are or however long their
    labels. Its minimum width is just its widest single item."""

    def __init__(self, parent=None, hspacing=8, vspacing=6):
        super().__init__(parent)
        self._items = []
        self._hspacing = hspacing
        self._vspacing = vspacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self):
        # Everything on one row.
        hints = [item.sizeHint() for item in self._items if not item.isEmpty()]
        width = sum(h.width() for h in hints) + self._hspacing * max(0, len(hints) - 1)
        return QtCore.QSize(width, max((h.height() for h in hints), default=0))

    def minimumSize(self):
        hints = [item.sizeHint() for item in self._items if not item.isEmpty()]
        return QtCore.QSize(
            max((h.width() for h in hints), default=0), max((h.height() for h in hints), default=0)
        )

    def _arrange(self, rect, apply):
        x, y, row_height = rect.x(), rect.y(), 0
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            if x > rect.x() and x + hint.width() > rect.right() + 1:
                x = rect.x()
                y += row_height + self._vspacing
                row_height = 0
            if apply:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self._hspacing
            row_height = max(row_height, hint.height())
        return y + row_height - rect.y()


class FlowFrame(tk.TkCompatMixin, tk.GridMixin, QtWidgets.QWidget):
    """A frame whose children (added with .add()) wrap like text -- see
    FlowLayout. Grid it sticky="ew" so it's told the width to wrap to."""

    def __init__(self, parent=None, hspacing=8, vspacing=6):
        super().__init__(parent)
        self._flow = FlowLayout(self, hspacing, vspacing)
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def add(self, widget):
        self._flow.addWidget(widget)
        return widget


def build_button_bar(parent, row, actions, danger=None, pady=(0, 6), columnspan=1):
    """A row of plain action buttons (`actions`: [(text, command), ...]),
    optionally followed by one danger button (`danger`: (text, command)) --
    set apart by a separator and extra padding, so a misaimed click on the
    last normal button doesn't land on it. Buttons that don't fit the
    width wrap onto another row (see FlowLayout). Returns the buttons in
    order. Pass columnspan=2 when placing it inside a two-column form."""
    bar = FlowFrame(parent)
    bar.grid(row=row, column=0, columnspan=columnspan, sticky="ew", pady=pady)
    buttons = [bar.add(ttk.Button(bar, text=text, command=command)) for text, command in actions]
    if danger is not None:
        # The separator and the danger button wrap as one unit.
        group = QtWidgets.QWidget(bar)
        box = QtWidgets.QHBoxLayout(group)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(10)
        if actions:
            line = QtWidgets.QFrame(group)
            line.setFrameShape(QtWidgets.QFrame.VLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            box.addWidget(line)
        button = ttk.Button(group, text=danger[0], command=danger[1])
        box.addWidget(button)
        style_danger_button(button)
        bar.add(group)
        buttons.append(button)
    return buttons


# Status text is free-form (set from many call sites) rather than a
# structured state enum, so the badge color is inferred from keywords in
# the current text. Red is checked first, so e.g. "Start failed" is red.
STATUS_RED_KEYWORDS = (
    "error", "failed", "invalid", "required", "could not", "stopped", "canceled", "cancelled",
    "not installed", "unavailable",
)
STATUS_GREEN_KEYWORDS = (
    "collecting", "finishing", "processing", "stopping", "running", "accumulating",
    "sweeping", "calibrating", "started", "connecting", "initializing", "homing", "moving",
)


class StatusBlock:
    """Colored status badge + status text, with an optional progress bar
    and "ETA:" line below. `is_running(text)` overrides the green keyword
    match -- e.g. "connected" for a hardware tab -- while red keywords
    still take priority."""

    def __init__(self, parent, row, status_var, progress_var=None, eta_var=None,
                 is_running=None, pady=(0, 8)):
        self._var = status_var
        self._is_running = is_running

        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=pady)
        frame.columnconfigure(0, weight=1)
        head = ttk.Frame(frame)
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(1, weight=1)
        self.badge = QtWidgets.QLabel(head)
        self.badge.setFixedSize(10, 10)
        head._own_layout().addWidget(self.badge, 0, 0, QtCore.Qt.AlignVCenter)
        make_wrapping_label(head, textvariable=status_var).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.progress = None
        if progress_var is not None:
            self.progress = ttk.Progressbar(
                frame, variable=progress_var, maximum=100.0, mode="determinate", length=220
            )
            self.progress.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        if eta_var is not None:
            eta_row = ttk.Frame(frame)
            eta_row.grid(row=2, column=0, sticky="ew", pady=(4, 0))
            eta_row.columnconfigure(1, weight=1)
            ttk.Label(eta_row, text="ETA:", font=SMALL_FONT).grid(row=0, column=0, sticky="nw")
            make_wrapping_label(eta_row, textvariable=eta_var, font=SMALL_FONT).grid(
                row=0, column=1, sticky="ew", padx=(4, 0)
            )

        status_var.trace_add("write", self.refresh)
        self.refresh()

    def refresh(self, *_args):
        text = str(self._var.get()).lower()
        if any(k in text for k in STATUS_RED_KEYWORDS):
            color = DANGER
        elif self._is_running(text) if self._is_running else any(k in text for k in STATUS_GREEN_KEYWORDS):
            color = RUNNING
        else:
            color = IDLE
        self.badge.setStyleSheet(
            f"background-color: {color}; border-radius: 5px; border: 1px solid rgba(0, 0, 0, 40);"
        )


def build_section(parent, row, title, collapsed=False, pady=(0, 8), columns_stretch=1):
    """A boxed collapsible section stacked in a sidebar/page; returns its
    body, set up as a two-column form (labels | stretching fields)."""
    section = CollapsibleFrame(parent, text=title, collapsed=collapsed)
    section.grid(row=row, column=0, sticky="ew", pady=pady)
    body = section.body
    if columns_stretch is not None:
        body.columnconfigure(columns_stretch, weight=1)
    return body


def add_form_row(parent, row, label, widget):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
    widget.grid(row=row, column=1, sticky="ew", pady=4)
    return widget


def add_stat_rows(parent, rows, start_row=0):
    """Dense read-only "label: value" rows; `rows` is [(label, var), ...].
    Returns the next free row."""
    row = start_row
    for label, var in rows:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=1)
        ttk.Label(parent, textvariable=var).grid(row=row, column=1, sticky="w", pady=1)
        row += 1
    return row


def make_wrapping_label(parent, text="", textvariable=None, font=None):
    """A word-wrapping label for text of unknown length (status messages,
    file paths, notes). Grid it with sticky="ew": it wraps to whatever
    width the layout gives it. A fixed wraplength (a maximum width) on a
    word-wrapped QLabel instead makes the layout guess its height for the
    wrong width and clip the text."""
    label = ttk.Label(parent, text=text, textvariable=textvariable, font=font, justify="left")
    label.setWordWrap(True)
    # A word-wrapped QLabel's own minimum size hint is roughly its whole
    # text on a few long lines -- wide enough to push a sidebar's scroll
    # body out past the viewport. It can always wrap narrower instead.
    label.setMinimumWidth(1)
    return label


def set_tip(widget, text):
    """A hover text box for `widget`. Rich text, so Qt word-wraps it (a
    plain-text tooltip is one long line) -- at TIP_WIDTH px. Blank lines
    in `text` separate paragraphs."""
    body = "<br><br>".join(html.escape(part.strip()) for part in text.split("\n\n"))
    widget.setToolTip(f"<qt><table width='{TIP_WIDTH}'><tr><td>{body}</td></tr></table></qt>")
    return widget


def add_note(parent, row, text="", textvariable=None, column=0, columnspan=2, pady=(0, 4), tip=None):
    """Small-print (8 pt) wrapping text: helper notes, file paths. Keep
    guide text to one short line and put the rest in `tip`: the note then
    ends in TIP_MARK and shows `tip` in a hover box (see set_tip)."""
    label = make_wrapping_label(parent, text, textvariable, SMALL_FONT)
    label.grid(row=row, column=column, columnspan=columnspan, sticky="ew", pady=pady)
    if tip:
        set_note(label, text, tip)
    return label


def set_note(label, text, tip=None):
    """Change a note's text and hover tip (for notes that follow a mode)."""
    label.setText(text + TIP_MARK if tip else text)
    if tip:
        set_tip(label, tip)
    else:
        label.setToolTip("")


class PathEntryFilter(QtCore.QObject):
    """Shows the full path while its entry has focus (for editing/reading
    exactly), and a middle-elided version -- plus the full path as a
    tooltip -- once focus leaves. Middle-eliding rather than the naive
    start/end truncation a plain QLineEdit does keeps both the
    informative filename/leaf tail and the drive/share head visible."""

    def __init__(self, entry, var):
        super().__init__(entry)
        self._entry = entry
        self._var = var
        var.trace_add("write", self._refresh)
        entry.installEventFilter(self)
        self._refresh()

    def eventFilter(self, _obj, event):
        et = event.type()
        if et == QtCore.QEvent.FocusIn:
            self._entry.setText(str(self._var.get()))
        elif et in (QtCore.QEvent.FocusOut, QtCore.QEvent.Resize):
            self._refresh()
        return False

    def _refresh(self, *_args):
        entry, var = self._entry, self._var
        full = str(var.get())
        entry.setToolTip(full)
        if entry.hasFocus():
            return
        metrics = QtGui.QFontMetrics(entry.font())
        width = max(entry.width() - 8, 20)
        elided = metrics.elidedText(full, QtCore.Qt.ElideMiddle, width)
        if entry.text() != elided:
            entry.setText(elided)


def wire_path_entry(entry, var):
    entry._path_filter = PathEntryFilter(entry, var)


def build_cluster_timing_mode_row(parent, row, var):
    """The "Cluster timing mode" dropdown under a tab's time-walk file field
    (only used when that file is a cluster timing calibration, .npz)."""
    add_form_row(parent, row, "Cluster timing mode:", ttk.Combobox(
        parent, textvariable=var, values=CLUSTER_TIMING_MODES, state="readonly", width=16,
    ))
    add_note(
        parent, row + 1, "Only for a .npz cluster timing file.",
        tip="core = best timing, bright = hits with ToT >= 1 us only, all = every hit. "
            "Ignored for a .json time-walk curve.",
    )


def build_path_field(parent, var, browse_command=None):
    """A stretching, middle-eliding path entry with a trailing "..." browse
    button. Returns the (ungridded) row frame."""
    row = ttk.Frame(parent)
    row.columnconfigure(0, weight=1)
    entry = ttk.Entry(row, textvariable=var)
    entry.grid(row=0, column=0, sticky="ew")
    if browse_command is not None:
        ttk.Button(row, text="...", width=3, command=browse_command).grid(row=0, column=1, padx=(4, 0))
    wire_path_entry(entry, var)
    return row


def build_metadata_section(parent, row, owner, pady=(0, 8)):
    """The run metadata block shared by Collection, Monitored Acquisition
    and Parameter Sweep. Reads the owner's shared *_var attributes; field
    labels mirror the metadata keys written to disk (Title Case), which is
    why they don't follow the usual sentence-case label rule. Returns the
    notes Text widget."""
    meta = build_section(parent, row, "Metadata", pady=pady)
    add_form_row(meta, 0, "Target:", ttk.Entry(meta, textvariable=owner.target_var))
    add_form_row(meta, 1, "Target Pressure:", ttk.Entry(meta, textvariable=owner.target_pressure_var))
    add_form_row(meta, 2, "Background Pressure:", ttk.Entry(meta, textvariable=owner.background_pressure_var))
    add_form_row(meta, 3, "Power:", ttk.Entry(meta, textvariable=owner.power_var))
    add_form_row(meta, 4, "Spot Size:", ttk.Entry(meta, textvariable=owner.spot_size_var))
    add_form_row(meta, 5, "Polarization:", ttk.Entry(meta, textvariable=owner.polarization_var))
    add_form_row(meta, 6, "Wavelength:", ttk.Entry(meta, textvariable=owner.wavelength_var))

    ttk.Label(meta, text="Notes:").grid(row=7, column=0, sticky="nw", padx=(0, 8), pady=4)
    # No fixed width: it stretches with the column, and a width in
    # characters becomes a minimum that pushes a 400px sidebar past its
    # scroll area.
    notes = tk.Text(meta, wrap="word", height=4)
    notes.grid(row=7, column=1, sticky="nsew", pady=4)
    shared_state.wire_notes_widget(notes)
    return notes


# ---- main-area components -----------------------------------------------

def build_card(parent, row, title, column=0, sticky="nsew", padx=8, pady=(8, 0)):
    """A titled, bordered group (QGroupBox) for a data view in the white
    main area -- a table or log that needs a heading."""
    card = ttk.LabelFrame(parent, text=title)
    card.grid(row=row, column=column, sticky=sticky, padx=padx, pady=pady)
    return card


def add_empty_placeholder(plot_item, text=EMPTY_PLOT_TEXT):
    """A centered grey "no data" label for a pyqtgraph PlotItem, hidden
    until show_empty_placeholder(..., True). Re-add it after any
    plot_item.clear(), which removes it along with everything else."""
    item = pg.TextItem(text, anchor=(0.5, 0.5), color=PLACEHOLDER_RGB)
    item.setVisible(False)
    plot_item.addItem(item, ignoreBounds=True)
    return item


def show_empty_placeholder(plot_item, item, is_empty):
    item.setVisible(is_empty)
    if is_empty:
        (x0, x1), (y0, y1) = plot_item.getViewBox().viewRange()
        item.setPos((x0 + x1) / 2, (y0 + y1) / 2)
