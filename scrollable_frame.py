"""A vertically-scrollable container for tall sidebars.

Usage: build widgets into `.body` (a plain ttk.Frame) instead of directly
into a ScrollableFrame -- the scrollbar only actually does anything once
`.body`'s natural height exceeds the visible area, so this is a drop-in
replacement for a plain ttk.Frame sidebar with no behavior change when
everything already fits.
"""
import tkinter as tk
from tkinter import ttk


_CONTENT_MARGIN = 14  # gap left between the body content and the scrollbar


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, width=300, **kwargs):
        super().__init__(parent, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Unlike a plain Frame, a Canvas can't measure its child and size
        # itself to fit -- give it an explicit starting width (a sidebar's
        # content is a fairly predictable width) so it doesn't default down
        # to Tk's generic 200x200 canvas size and clip everything.
        self._canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, width=width)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        self._canvas.configure(yscrollcommand=scrollbar.set)

        self.body = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>", self._on_body_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<Enter>", self._bind_mousewheel)
        self._canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_body_configure(self, _event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # Keep the body a bit narrower than the canvas so content wraps
        # instead of needing a horizontal scrollbar too, and so there's a
        # visible gap between the panels' edge and the scrollbar rather than
        # the scrollbar butting right up against them.
        self._canvas.itemconfigure(self._window, width=max(0, event.width - _CONTENT_MARGIN))

    def _bind_mousewheel(self, _event=None):
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None):
        self._canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
