# Experiment Control UI — Style Sheet

Reference tab: **Acquisition › monitored acquisition**. Every tab in `UI.py`
now follows it.

The implementation is **`ui_style.py`**. Build tabs from its helpers instead
of setting margins, colors and spacing by hand. The primitives it wraps live
in `collapsible_frame.py`, `scrollable_frame.py`, `plot_panel.py`, `qtk.py`
and `qt_plots.py`. When this sheet and the code disagree, the code wins;
update this sheet.

---

## 1. Foundations

| Token | Value | Where set |
|---|---|---|
| Qt style | `Fusion` | `qtk._ensure_app` |
| Base font | system default **+1 pt** (app-wide) | `qtk._ensure_app` |
| Small print | `ui_style.SMALL_FONT` = `("Segoe UI", 8)`: ETA, file paths, notes | `add_note`, per-label `font=` |
| Fixed-width readouts | `("TkFixedFont", 10)`: live monitor values that need their digits aligned | Power Supply channel table |
| Emphasis | bold only for section toggles and the danger button | |
| Plot config | `background="w"`, `foreground="k"`, `antialias=True`, `imageAxisOrder="row-major"` | `qt_plots.py` |
| 2D colormap | cmasher `rainforest` (falls back to `viridis`) | `qt_plots.rainforest_colormap` |

## 2. Color tokens

### Surfaces (the grey / grey / white hierarchy)

| Token | Hex | Used for |
|---|---|---|
| `FLUSH_BACKGROUND` | `#e9e9e9` | Edge-to-edge containers: sidebar, footer, full-tab page, and their scroll areas |
| `BOXED_BACKGROUND` | `#d8d8d8` | Boxed sections nested inside a flush container |
| `ui_style.WHITE` | `#ffffff` | The main area: plots, tables, logs. Input fields paint their own white |

The two greys are deliberately darker than Qt's default `#f0f0f0` so they
contrast clearly with the white main area. The layout helpers apply them.
If you apply them by hand, use `collapsible_frame.set_background`. Plain
`QWidget`, `QScrollArea`, and its `viewport()` **do not inherit** a fill.

### Semantic (`ui_style` constants)

| Constant | Hex | Used for |
|---|---|---|
| `DANGER` | `#c62828` | Danger button fill; error/stopped status badge; tripped/fault channel status |
| `DANGER_HOVER` | `#d32f2f` | Danger button `:hover` |
| `DANGER_PRESSED` | `#8e0000` | Danger button `:pressed` and 1px border |
| `RUNNING` | `#2e7d32` | Running/connected status badge; channel ON; HV enabled |
| `WARNING` | `#ef6c00` | Degraded-but-on state (SY127 `OVC` over-current) |
| `IDLE` | `#9e9e9e` | Idle status badge; channel OFF; HV disabled |
| `PLACEHOLDER_RGB` | `rgb(140,140,140)` | Empty-plot placeholder text |

### Interaction overlays (translucent black, so they work on any grey)

| State | Value |
|---|---|
| Hover | `rgba(0,0,0,25)` |
| Pressed | `rgba(0,0,0,45)` |
| Hairline border on small swatches | `rgba(0,0,0,40)` |

## 3. Layouts

Every tab uses one of two layouts.

### 3.1 Sidebar layout: `ui_style.build_sidebar_layout(self)` → `(sidebar, main)`

For tabs with a data view: plots, a results table, or a log.

```
┌──┬──────────────────────────────┬┬──────────────────────────────────────────┐
│▾ │ [Start] [Stop] │ [Force Stop]││                                          │
│C │ ● Status text                ││   plots / tables / logs                  │
│o │ ▬▬▬▬▬▬▬▬▬ progress           ││   (white main area)                      │
│n │ ETA: 3m 20s                  ││                                          │
│t │ ▾ Section                    ││                                          │
│r │ ┌──────────────────────────┐ ││                                          │
│o │ │ Label:        [ field  ] │ │├──────────────────────────────────────────┤
│l │ └──────────────────────────┘ ││ optional flush footer (Plot options)     │
│s │  (scrolls, 400 px min)       ││                                          │
└──┴──────────────────────────────┴┴──────────────────────────────────────────┘
```

- **Sidebar**: a flush `CollapsibleFrame` (vertical "Controls" strip, rule on
  the right) → `ScrollableFrame(width=400)`. You build into `sidebar`: one
  column, stacked rows.
- **Main**: white, zero margins. Call `main.rowconfigure(r, weight=1)` for
  the row that fills.
- Collapsing the sidebar hands its width to the main area.
- Give the sidebar a trailing empty row with `weight=1`, so content stays
  packed at the top as sections collapse.

Tabs using it: diagnostics, monitored acquisition, timewalk calibration,
parameter sweep, serval config, power supply.

### 3.2 Page layout: `ui_style.build_page_layout(self)` → `page`

For tabs with only forms and actions, where a main area would be empty. It
is the sidebar's content column used as the whole tab: a flush grey,
scrollable page. The column is capped at 640 px (`PAGE_MAX_WIDTH`) and
left-aligned, so fields don't stretch across a wide window. The components
and spacing are the same as in the sidebar.

Tabs using it: collection parameters, stage control.

### 3.3 Main-area content

| Content | Treatment |
|---|---|
| Plots | Flush, edge to edge (`grid_into(glw, main, …)` or `HistogramPlotPanel`) |
| A table that is the main data view | Flush at the top of `main` (Sweep results) |
| A table or log that needs a heading | Card: `ui_style.build_card(main, row, "Title")`, 8 px inset |
| Secondary plot options | Flush `toggle_last` footer (see §5.1) |

### The margin rule

Structural containers reach their parent's real edges. Qt's default
`QGridLayout` margins and spacing leave visible gaps. The layout helpers call
`ui_style.zero_margins(widget, spacing=True)` on the tab root and the main
area. Do the same for any extra structural frame you add, such as a plot
frame. Breathing room inside content comes from explicit `padx`/`pady`, never
from default layout margins.

## 4. Spacing scale

All values are px. Stick to this set: **1, 2, 4, 6, 8, 10, 12**.

| Context | Value |
|---|---|
| Between stacked sidebar/page blocks (status, sections) | `pady=(0, 8)`; button bar `(0, 6)` |
| Form row (label + field) | `pady=4` |
| Label → field gap | `padx=(0, 8)` |
| Read-only stat rows (dense) | `pady=1` |
| Checkbox rows under a form | `pady=(4, 0)`; attached sub-row `(2, 4)` |
| Buttons in a row | `padx=(0, 8)` |
| Before a danger button (after separator) | separator `padx=(0, 10)` |
| Button bar inside a section | `pady=(4, 6)` |
| Field → trailing "..." browse button | `padx=(4, 0)` |
| Badge → status text | `6` |
| Status → progress → ETA | `pady=(4, 0)` each |
| Inline checkbox group items | `padx=(6, 0)` |
| Vertical separator between control groups | `padx=12` |
| Horizontal rule between control rows | `pady=(8, 8)` |
| Boxed section inner margin | `6` all sides |
| Flush section: toggle ↔ panel | `6` (0 for vertical/toggle-last) |
| Card in the main area | `padx=8`, `pady=(8, 0)`; last card `8` |
| Note below its block | `pady=(0, 4)` |
| Table header cells | `pady=(4, 2)`; first column `padx=(6, 4)` |
| Table body cells | `padx=2`, `pady=1`; last row bottom `4` |

## 5. Components

### 5.1 Collapsible section

The main grouping primitive.

| Variant | How | Look | Use for |
|---|---|---|---|
| **Boxed** | `ui_style.build_section(parent, row, "Title", collapsed=False)` → body | `StyledPanel` border, `#d8d8d8` panel, title transparent on the surrounding grey | Every block in a sidebar/page |
| **Flush sidebar** | built by `build_sidebar_layout` | Whole thing `#e9e9e9`, sunken rule on the right, title rotated | The tab's "Controls" sidebar |
| **Flush footer** | `CollapsibleFrame(flush=True, separator="top", toggle_last=True)` | `#e9e9e9`, sunken rule above, toggle pinned to the bottom edge | Secondary options under plots |

`build_section` returns a two-column form body (column 1 stretches). Pass
`columns_stretch=None` for a table body. Pass `pady=0` on the last section.

Toggle button: `▾ Title` when open, `▸ Title` when closed. Bold, borderless,
`padding: 4px`, pointing-hand cursor, hover/pressed overlays from §2.

- Start **collapsed** for set-once or occasional settings: *Histogram bins /
  bounds*, *Hot pixel masking*. Frequently used sections start open.
- Section titles use sentence case: "Acquisition parameters", "Per-shot rates".
- Groups of read-only stats get one boxed section per group (Diagnostics:
  Rates / Per-shot rates / …). No bold sub-headings inside a section.

### 5.2 Card (`ui_style.build_card` → `QGroupBox`)

A titled, bordered group for a data view in the white main area ("Channels",
"Activity", "Dashboard", "Log"). Also used inside a flush footer for groups
that are always visible ("Plot controls"). Use a card when the group should
always be visible, and a boxed section when it should collapse.

### 5.3 Buttons: `ui_style.build_button_bar(parent, row, actions, danger=None)`

- Buttons pack at the **left**, 8 px apart. An empty trailing column takes
  the leftover width. Inside a two-column form, pass `columnspan=2`.
- **Default**: plain `ttk.Button`, Fusion look. Short Title Case labels:
  `Start`, `Stop`, `Reset`, `Show All`, `Set Current as Zero`.
- **Browse**: `...` directly to the right of its path field (built by
  `build_path_field`).
- **Danger**: `danger=(text, command)`. Styled with `DANGER_BUTTON_STYLE`,
  placed **last** after a vertical separator with extra padding, so a
  slightly missed click on the safe action doesn't hit it. At most one per
  bar.
- **Stop naming**: when a tab has both a graceful and a hard stop, the
  graceful one is `Stop` and the hard abort is the danger button `Force Stop`
  (monitored acquisition, parameter sweep). A tab with a single stop uses a
  plain `Stop`.
- Primary actions go in the bar at the top of the sidebar/page. Actions that
  belong to one section go in a bar inside that section (Stage: Move /
  Refresh / Set Current as Zero).

### 5.4 Status block: `ui_style.StatusBlock(parent, row, status_var, progress_var=None, eta_var=None, is_running=None)`

```
● Status text (wraps to the sidebar width)
▬▬▬▬▬▬▬▬▬▬▬▬▬▬ (QProgressBar, text hidden; only if progress_var is given)
ETA: 3m 20s      ← small print; only if eta_var is given
```

Every tab has one, directly under its top button bar.

- **Badge**: 10×10, `border-radius: 5px`, hairline border. The color comes
  from the status text (lower-cased):
  - red `DANGER` if the text contains any of `STATUS_RED_KEYWORDS`: error,
    failed, invalid, required, could not, stopped, canceled/cancelled,
    not installed, unavailable
  - green `RUNNING` if `is_running(text)` is true. By default that is a match
    on `STATUS_GREEN_KEYWORDS`: collecting, finishing, processing, stopping,
    running, accumulating, sweeping, calibrating, started, connecting,
    initializing, homing, moving
  - grey `IDLE` otherwise
- Hardware tabs pass `is_running` so the badge stays green for as long as a
  connection is up (Stage: connected XPS; Power Supply: open serial port).
  Red keywords still take priority.
- **Status text style**: a full sentence ending in a period. Use a trailing
  `...` for in-progress states ("Stopping...", "Connected to COM3.
  Initializing..."). Errors start with `Error:` or name what failed
  ("Config failed: …", "Ping failed: …").

### 5.5 Form rows: `ui_style.add_form_row(parent, row, "Label:", widget)`

Two-column grid, fields stretch. Label `sticky="w"`, `padx=(0, 8)`, `pady=4`.

- Labels end with a colon. Put units in parentheses: `Frame time (s):`,
  `Bias voltage (V):`.
- Short numeric entries: `width=18`. Text fields: no width (stretch).
- Multi-line `Text`: label `sticky="nw"`, field `sticky="nsew"`, `height=4`.
- Checkboxes that apply to the whole form span both columns
  (`columnspan=2, sticky="w"`) and use a descriptive label ("Apply timewalk
  correction before clustering").
- **Metadata exception**: `ui_style.build_metadata_section(parent, row, self)`
  builds the shared Target / Target Pressure / … / Notes block (Collection,
  Monitored Acquisition, Parameter Sweep). Its labels stay in **Title Case**
  on purpose: they mirror the metadata keys written to disk.

### 5.6 Path fields: `ui_style.build_path_field(parent, var, browse_command=None)`

- **Unfocused**: middle-elided text (`Qt.ElideMiddle`), so the drive and the
  filename both stay visible, with the full path as a tooltip.
- **Focused**: the full path, for editing.
- A trailing `...` button when `browse_command` is given. Every editable
  file/folder path should have one.
- Grid the returned row as a form field, or on its own row under a checkbox
  (`columnspan=2, sticky="ew", pady=(2, 4)`).

### 5.7 Read-only stats: `ui_style.add_stat_rows(parent, [(label, var), ...])`

- Same two columns as a form, but `pady=1` (dense), with value labels.
- Placeholder for no value: `--` (two hyphens, never an em dash).
- Long values such as file paths go below the rows as small print:
  `add_note(parent, row, textvariable=var, pady=0)`.

### 5.8 Wrapping text: `ui_style.make_wrapping_label` / `ui_style.add_note`

Use these for any text of unknown length: status messages, paths, helper
notes, fit summaries. They wrap to the width the layout gives them. Grid
them `sticky="ew"`.

**Never** give a label a fixed `wraplength`, and never grid a wrapping label
`sticky="w"`. Qt then guesses the label's height for the wrong width and
clips the text. The helpers also zero the label's minimum width, so a long
note can't push a sidebar wider than its scroll area.

### 5.9 Settings tables

- One table per kind of quantity (spatial vs. temporal), each in its own
  `LabelFrame` or boxed section. Don't combine mixed units in one table.
- Header row in plain labels: `Plot | Bins | Min (unit) | Max (unit)`. When
  rows have different units, put the unit in the row label (`t (ns)`).
- Cells: `ttk.Entry(width=9)`.
- Set-once tables go in a section that starts collapsed.
- Live data tables (Power Supply channels) follow the same cell spacing.
  Colored status text uses the semantic tokens.

### 5.10 Inline control bar

A single card row, split into groups:

```
Log scale: ☐Pixel ☐Cluster …  │  Single plot: [combo ▾] [Focus]
───────────────────────────────────────────────────────────────
Gamma (pixel/cluster maps): ──●────── 1.00 [Reset]
```

- Groups inside a row: vertical `Separator`, `padx=12`.
- Rows: full-width horizontal `Separator`, `pady=(8, 8)`.
- Put each group's label first ("Log scale:", "Single plot:").
- Read-only combos: `state="readonly"`, `width=18`.
- Numeric readout next to a slider: `Label(width=5)`, 2 decimals.

### 5.11 Plots

- White, flush in `main`. Every plot has a title and axis labels with units.
- 2D maps: `ImageItem`, `setAspectLocked(True)` for detector maps, range =
  bin bounds with `padding=0`.
- 1D histograms: plain `plot()` curve, x range fixed to the bin bounds, y
  autorange.
- Every plot uses `ZoomFocusViewBox`: drag a rectangle to zoom; double-click
  focuses in the 6-plot grid.
- Log scale is a per-plot checkbox, not a global switch.

### 5.12 Empty states

Until a plot has data, show a centered grey placeholder that names the next
action:

```python
placeholder = ui_style.add_empty_placeholder(plot_item, "No data — press Start to acquire")
ui_style.show_empty_placeholder(plot_item, placeholder, is_empty)
```

`plot_item.clear()` removes the placeholder, so re-add it after every clear,
after `autoRange()`, so it centers on the final view.

| Tab | Placeholder |
|---|---|
| Diagnostics, Monitored Acquisition, Timewalk (histogram) | No data — press Start to acquire |
| Parameter Sweep | No data — press Start to sweep |
| Timewalk (correction) | No correction yet — press Generate Correction |

## 6. Copy conventions

| Element | Case | Example |
|---|---|---|
| Tab names | lowercase | `monitored acquisition` |
| Section / card titles | Sentence case | `Acquisition parameters`, `Hot pixel masking` |
| Buttons | Title Case, 1–3 words | `Force Stop`, `Show All`, `Reset Accumulation` |
| Field labels | Sentence case + colon + (unit) | `Run duration (s):` (Metadata excepted, §5.5) |
| Status | Sentence, period or `...` | `Collecting 120 frame(s). Writing run.cv4` |
| Missing value | `--` | |

## 7. Checklist for a new or restyled tab

- [ ] Pick the layout: `build_sidebar_layout` if there's a data view, `build_page_layout` if not.
- [ ] Top of the sidebar/page: `build_button_bar` (danger action last, via `danger=`), then `StatusBlock`.
- [ ] Every group is a `build_section`; set-once or occasional groups start collapsed; the last one gets `pady=0`; add a trailing `weight=1` row.
- [ ] Fields via `add_form_row`; paths via `build_path_field` with a browse command; stats via `add_stat_rows`.
- [ ] Unknown-length text via `add_note` / `make_wrapping_label`, never `wraplength`.
- [ ] Main area: plots flush; headed tables/logs in `build_card`.
- [ ] Empty plots show a placeholder that names the next action.
- [ ] Any extra structural frame gets `zero_margins`.
- [ ] Only spacing values from §4 and colors from `ui_style`; add a token there (and here) before adding a new hex value.
