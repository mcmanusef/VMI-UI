# VMI Experiment Control UI

A single desktop application for running a velocity-map imaging (VMI)
experiment built around a Timepix3 (TPX3CAM) detector: it configures and
drives the hardware, collects and clusters data live, and then does the
offline analysis — momentum and m/q calibration, parameter grouping, and
electron–ion coincidence plots — without ever leaving the app.

Everything is one PyQt5 process, organised as three groups of tabs:

| Group | What it is for |
|---|---|
| **Hardware** | Things you connect to and operate: Serval, the Newport XPS stage, the CAEN SY127 HV crate |
| **Acquisition** | Things that collect data: run parameters, diagnostics, monitored acquisition, timewalk calibration, parameter sweeps |
| **Analysis** | Offline work on data already collected: conversion, calibration, grouping, coincidence |

---

## Contents

- [Quick start](#quick-start)
- [Hardware it talks to](#hardware-it-talks-to)
- [The tabs](#the-tabs)
- [Data formats](#data-formats)
- [End-to-end workflows](#end-to-end-workflows)
- [Architecture](#architecture)
- [Configuration and persisted state](#configuration-and-persisted-state)
- [Module map](#module-map)
- [Development](#development)
- [Reference documents](#reference-documents)

---

## Quick start

Requires **Python ≥ 3.13** on Windows (paths, the serial port and the
default data folders all assume Windows; the analysis modules themselves
are platform-independent).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python UI.py
```

`UI.py` is the entry point. (`main.py` is the leftover PyCharm sample
script and is not used by anything.)

**On dependencies:** `requirements.txt` is the authoritative runtime list;
`pyproject.toml` carries the same set as packaging metadata. Install from
`requirements.txt`.

PyQt5 is pinned (`pyqt5==5.15.10`, `pyqt5-qt5==5.15.2`) because
PyQt5-Qt5 5.15.19+ has no `win_amd64` wheel. `pyserial` is imported lazily:
without it the power supply tab still loads, it just refuses to connect.

None of the tabs move, home, or energise anything on their own. Every
motion command and every HV change is behind an explicit button.

---

## Hardware it talks to

| Device | Transport | Module |
|---|---|---|
| ASI **Serval** server / TPX3CAM | HTTP (`requests`) | `serval_client.py` |
| Newport **XPS-D** motion controller | raw TCP socket, port 5001 | `xps_client.py` |
| CAEN **SY127** HV crate | serial (`pyserial`), menu-walking | `power_supply_interface.py` |

Serval calls all go through one shared `requests.Session` with
`trust_env=False` — the server is always local or LAN, so proxy environment
variables should never sit between the app and it, and keep-alive avoids a
fresh TCP handshake per call.

The SY127 has no real command API: a worker thread owns the serial port and
walks the crate's on-screen menus, polling `DISPLAY STATUS` for `GROUP ALL`
about once a second and parsing the fixed-format table. It assumes channels
still have their factory names (`CH00`, `CH01`, …).

---

## The tabs

### Hardware

**serval config** — server URL, BPC/DACS pixel-config files, bias voltage,
destination folder and file pattern; pushes them to Serval and shows a live
dashboard. Owns `server_var`, the server URL every other tab reads.

**stage control** — connect to the XPS-D, initialize, home, and jog to
absolute positions by hand. This tab is the **single owner of the stage
connection**; parameter sweep drives its moves through this connection
rather than opening a second one to the same physical stage. Motion errors
are annotated with the controller's own group-status text, because "Not
allowed action" almost always means "not homed yet".

**power supply** — the SY127 channel table: live voltage/current readout,
per-channel ON/OFF, and ratio-locked channel groups so a set of electrodes
can be ramped together. Groups defined here are what the sweep tab's "HV
during moves" option ramps down.

### Acquisition

**collection parameters** — the run's parameters and metadata (target,
pressures, power, spot size, polarization, wavelength, notes) plus frame
time, run duration and save folder. Writes `run_metadata.json` into the run
folder; the conversion tab reads it back so the metadata follows the data.

**diagnostics** — start Serval collecting into a temp folder and watch the
six live histograms (pixel map, cluster map, counts/pixel, i-ToF, cluster
*t*, e-ToF) with per-frame rate statistics. Nothing is kept.

**monitored acquisition** — the main data-taking tab. One raw `.tpx3` file
per frame, decoded, optionally timewalk-corrected, clustered and appended
into a cv4 (HDF5) file as it arrives, with the same six plots updating
live. Parameters in the sidebar, plots in the main area.

**timewalk calibration** — accumulates a 2D histogram of pixel (ToT, *t*)
and fits a per-ToT time offset out of it. A real arrival peak should not
drift with ToT, so any ToT-dependent drift of the ridge is walk; the curve
is anchored to ≈0 at the high-ToT end. Saves JSON that the diagnostics,
monitored acquisition and conversion tabs can load and apply before
clustering.

**parameter sweep** — moves the stage through a list of positions,
collecting into **one cv4 per physical position** (revisits append to the
same file, so multiple passes accumulate). Frames are triggered one at a
time and handed to a single persistent processing thread, so stage motion
is fully decoupled from processing. In cluster-dwell mode the whole
position list is calibrated once up front to fix each position's frame
count. Optionally ramps a power-supply channel group down before every move
and back up before collecting again.

### Analysis

**conversion** — a folder of raw `.tpx3` in, cv4 out, using exactly the
same per-file pipeline monitored acquisition uses live (`raw_conversion.py`
is shared), so live and offline output are identical. It recognises each
acquisition tab's folder layout and splits the output accordingly.

**momentum calibration** — derives a 3D electron momentum calibration
following `electron_momentum_calibration.md`, from either one circularly
polarized run or an S + P pair of runs. Ctrl+click adds a peak,
Shift+click removes the nearest one.

**m/q calibration** — fits ion time of flight to mass-to-charge. Label
peaks in the i-ToF spectrum with a species (`H2O+`, `N2+`, `Xe++`,
`Xe^2+`) or a plain m/q number, and it fits `t = t₀ + a·√(m/q)`.

**apply calibration** — applies the latest momentum calibration (and
optionally the m/q one) to a cv4 dataset and writes pandas DataFrames into
an HDF5 file. Calibrations reach this tab through a `CalibrationHub`, so
whatever the calibration tabs last fit, loaded or saved is already
selected.

**parameter grouping** — collects a scan's cv4 files into one grouped file,
one HDF5 group per parameter value. The parameter can be the raw stage
position, a QWP angle converted to ellipticity, or a position converted to
a delay in fs.

**coincidence** — electron–ion coincidence plots of a calibrated dataset,
with click-and-drag gates on every quantity. Each electron is joined to the
ions of its own pulse, so m/q is a per-row coincidence quantity. Plots can
show counts, probability density, or forward–backward asymmetry (split by
the sign of `p_x`, the propagation axis), as a 2D heat map or a 1D plot
that can be split into one line per value of a second column. A gate never
cuts into its own plot.

> `quick_monitor_interface.py` — a low-latency live preview that polls
> Serval's own server-side TIFF preview instead of decoding packets — is
> not wired in as a tab right now, but the module is kept intact.

---

## Data formats

```
 .tpx3  ──raw_conversion──▶  .cv4  ──apply_calibration──▶  *_calibrated.h5  ──▶ coincidence
                              │
                              └──parameter_grouping──▶  *_grouped.cv4  ──▶ apply_calibration
```

### `.tpx3` — raw detector stream

A stream of chunks: an 8-byte header (payload length in bytes 6–8,
little-endian) followed by 64-bit packets. Each word has `2**62` subtracted,
then `header = word >> 60` selects pixel packets (`0x7`) or TDC packets
(`0x2`). TDC types are sorted into laser pulses, e-ToF and i-ToF markers by
pulse length. Decoding lives in `tpx_processing.py`; the bit layout is
written out in full in `tpx_file_decoding.md`.

### `.cv4` — clustered events (HDF5)

| Dataset | Shape | Meaning |
|---|---|---|
| `t_pulse` | (N_pulses,) f8 | absolute laser-trigger time (ns, camera clock) |
| `x`, `y`, `t` | (N_cluster,) f8 | cluster position (px) and time relative to its pulse (ns) |
| `cluster_corr` | (N_cluster,) i4 | index into `t_pulse` for each cluster |
| `t_etof` / `etof_corr` | (N_etof,) | e-ToF time relative to its pulse, and its pulse index |
| `t_tof` / `tof_corr` | (N_tof,) | i-ToF time relative to its pulse, and its pulse index |

`t_pulse[0]` is a `-1.0` sentinel so the first real pulse lands at index 1,
matching the reference files. Run metadata sits in the root attributes.

Files are written as `<name>.partial.cv4` and renamed to `<name>.cv4` only
once every raw file has been processed — a stopped or failed run keeps the
`.partial` name. `Cv4Writer` reopens an existing file in place rather than
truncating it, which is what lets every tab close the file after each raw
file and reopen it for the next.

**Folder layouts the conversion tab recognises:**

```
Monitored acquisition:  <run>/raw/*.tpx3                      → <run>/<run>.cv4
Collection parameters:  <run>/*.tpx3                          → <run>/<run>.cv4
Parameter sweep:        <run>/raw/pos_<visit>_<position>/...  → <run>/pos_<index>_<position>.cv4
```

### `*_grouped.cv4` — a parameter scan in one file

Root attributes record the layout, parameter label, mode and settings; one
group per scan file named by its parameter value (`+0.5`, `-3335.64`, with
` (2)` for repeats), each holding that file's datasets and attributes.
Groups are written with HDF5's dataset copy, which streams — memory use
stays flat (~40 MB for an 800 MB file) regardless of scan size.

### `*_calibrated.h5` — calibrated DataFrames

Read back with `pd.read_hdf(path, key)`:

| Key | Contents |
|---|---|
| `electrons` | One row per cluster / e-ToF pair from the same pulse: raw `pulse, t_pulse, x, y, t, t_etof, n_clusters, n_etof` plus calibrated `dt, px, py, pz` (a.u.) and `energy` (eV) |
| `raw/clusters` | Every cluster, paired or not |
| `raw/etof` | Every e-ToF hit |
| `ions` | Every i-ToF hit, plus `mq` when an m/q calibration was given |

Pairs outside the calibration's valid `dt` range get NaN momenta. Grouped
input adds a leading `parameter` column to every table, stored as a data
column so one value can be read on its own:

```python
pd.read_hdf(path, "electrons", where="parameter == 0.5")
```

Output is written one part at a time — each group is calibrated and
appended before the next is read — so memory use is about one group's
worth. Every table's attributes carry the source path, both calibrations as
JSON, and the cv4 run metadata.

---

## End-to-end workflows

**A single run**

1. *serval config* → point at the server, load BPC/DACS, set bias and destination.
2. *collection parameters* → frame time, duration, save folder, metadata.
3. *diagnostics* → confirm the detector looks right before committing.
4. *monitored acquisition* → Start. A cv4 grows as frames arrive.

**A parameter scan**

1. *stage control* → connect, initialize, home.
2. *power supply* → define the channel group to protect during moves (optional).
3. *parameter sweep* → position list, dwell condition, HV guard → Start.
4. *analysis › parameter grouping* → collect the per-position cv4 files into one grouped file.

**From data to coincidence plots**

1. *momentum calibration* → load a circular run (or an S + P pair), pick rings and time peaks, Fit.
2. *m/q calibration* → label i-ToF peaks, Fit.
3. *apply calibration* → pick the cv4 or grouped file, Apply and Save.
4. *coincidence* → Load the `_calibrated.h5`, set gates, plot.

**Recovering a run whose processing fell behind:** raw files are always
kept, so *analysis › conversion* can re-run the exact same pipeline over
the folder afterwards, timewalk correction included.

---

## Architecture

### `qtk.py` — a tkinter-shaped API over real Qt widgets

The UI was originally tkinter. Rather than rewrite every widget call,
`qtk.py` implements exactly the subset of the tkinter/ttk surface this
project used — `grid` geometry, the `Variable` classes with `trace_add`,
the handful of widgets and dialogs — backed by genuine `QWidget`s. Porting
a module was mostly an import change:

```python
import tkinter as tk      →  import qtk as tk
from tkinter import ttk   →  from qtk import ttk
```

It is **not** a general-purpose tkinter emulator. Anything it doesn't
cover — pyqtgraph plots, the scrollable sidebar — is written in native
PyQt5 in its own module (`qt_plots.py`, `scrollable_frame.py`,
`collapsible_frame.py`).

`.grid(...)` adds a widget to a `QGridLayout` created lazily on its Qt
parent. `sticky` maps to per-axis Qt alignment with tkinter's own
semantics: both edge letters on an axis means stretch, one letter anchors,
none centers.

### `ui_style.py` + `STYLE_GUIDE.md` — one look, enforced by helpers

Every tab is built from `ui_style` helpers rather than hand-set margins and
colors: `build_sidebar_layout` (tabs with a data view) or
`build_page_layout` (forms only), then `build_button_bar`, `StatusBlock`,
`build_section`, `add_form_row`, `build_path_field`, `add_stat_rows`,
`add_note`. `STYLE_GUIDE.md` documents the grey/grey/white surface
hierarchy, the spacing scale (1, 2, 4, 6, 8, 10, 12 px only), the semantic
color tokens, and a checklist for adding a tab. **When the guide and the
code disagree, the code wins — update the guide.**

### Cross-tab state

Three separate mechanisms, deliberately:

- **`shared_state.py`** — live-shared Tk variables. Fields that should be
  *the same control* on several tabs (frame time, save folder, metadata,
  histogram bins) are literally the same variable object, so editing it in
  one tab updates the other immediately.
- **`app_settings.py`** — a small JSON store (`app_settings.json`) next to
  the source. `persistent_var(parent, tk.StringVar, "key", default)` gives
  a variable that loads its last value on creation and writes back on every
  change, so fields survive a restart.
- **`CalibrationHub`** (in `calibration_interface.py`) — the calibration
  tabs publish whatever they last fit, loaded or saved; the apply tab
  subscribes. Paths are persisted too, so the latest file reloads after a
  restart.

### `TabCoordinator` — one measurement at a time

Serval has exactly one active trigger/destination configuration, so two
tabs driving it at once would just fight over it. Every tab that can start
a measurement registers with the coordinator, and each tab's `start()` asks
the coordinator to stop every other currently-running tab first.

### Threading model

The pattern is the same everywhere: a background thread does the blocking
work (HTTP, serial, socket, decoding, clustering, HDF5), pushes results
onto a `queue.Queue`, and the Qt main loop drains it via a repeating
`.after` callback. No widget is ever touched off the main thread.

Long acquisitions are deliberately split into bounded batches
(`serval_client.MAX_BATCH_SECONDS`, `LONG_RUN_TRIGGERS = 50_000`) so no
single Serval call is trusted to keep going for an entire run — a lesson
learned from destination-config hangs, which `test_serval_split_strategy.py`
and `test_configure_continuous_destination.py` document.

---

## Configuration and persisted state

| File | Contents | In git? |
|---|---|---|
| `serval_config.json` | Serval defaults: server URL, BPC/DACS paths, bias, destination, file pattern | tracked — shared defaults |
| `timewalk_correction.json` | The default timewalk correction curve (`timewalk.DEFAULT_CORRECTION_PATH`) | tracked |
| `app_settings.json` | Every persisted field: last folders, plot bins and ranges, calibration paths, run metadata | **ignored** — per-machine |

`app_settings.json` is deliberately untracked: it accumulates whatever each
field was last set to, including absolute paths and network shares specific
to one machine. It is created on demand, and every field falls back to its
own default when the file or key is missing, so a fresh clone just works.

`.gitignore` keeps the bulk out: `*.cv4`, `*.h5`, `*.tpx3`, `*.pdf`,
`Timepix Documentation/`, `.venv/`, `__pycache__/`, `.idea/`,
`.ruff_cache/`.

---

## Module map

**Entry point and shell**

| File | Role |
|---|---|
| `UI.py` | The app: builds the three group notebooks and every tab, wires shared state |
| `tab_coordinator.py` | "Only one measurement at a time" |
| `shared_state.py` | Live-shared Tk variable bundles |
| `app_settings.py` | JSON-backed persistent fields |

**UI toolkit**

| File | Role |
|---|---|
| `qtk.py` | tkinter-compatible API over PyQt5 |
| `ui_style.py` | Layout and component helpers (implements `STYLE_GUIDE.md`) |
| `collapsible_frame.py` | Collapsible sections: boxed, flush sidebar, flush footer |
| `scrollable_frame.py` | Scrollable sidebar content |
| `plot_panel.py` | The reusable six-histogram grid + plot options |
| `qt_plots.py` | Shared pyqtgraph utilities: colormap, hist→RGBA, `ZoomFocusViewBox` |

**Hardware clients**

| File | Role |
|---|---|
| `serval_client.py` | Serval HTTP helpers and the shared session |
| `xps_client.py` | Newport XPS-D socket client (connect, home, move, status) |

**Processing (headless — no Qt, importable from a notebook)**

| File | Role |
|---|---|
| `tpx_processing.py` | TPX3 decoding, TDC sorting, clustering, histogram building |
| `raw_conversion.py` | The per-file pipeline and cv4 job planning shared by live and offline paths |
| `cv4_writer.py` | The cv4 HDF5 layout, append/reopen/finalize |
| `timewalk.py` | Timewalk correction curve: generate, save, apply |
| `electron_calibration.py` | 3D electron momentum calibration |
| `mass_calibration.py` | Ion ToF → m/q calibration, formula parsing, isotope masses |
| `apply_calibration.py` | cv4 → calibrated DataFrames in HDF5 |
| `parameter_grouping.py` | Scan cv4 files → one grouped file |
| `coincidence.py` | Coincidence dataset loading, gating, binning, asymmetry |
| `time_estimate.py` | ETA formatting |

**Tab interfaces** — one per tab, named `*_interface.py`:
`serval_`, `stage_`, `power_supply_`, `collection_`, `diagnostics_`,
`acquisition_`, `timewalk_`, `sweep_`, `analysis_` (conversion),
`calibration_` (momentum + m/q), `calibration_apply_`,
`parameter_grouping_`, `coincidence_`, and the unwired
`quick_monitor_interface.py`.

---

## Development

**Linting** — `ruff` 0.15.0, pinned in `requirements.txt`, with default
settings (there is no `[tool.ruff]` section):

```powershell
ruff check .
```

**Tests** — the three `test_*.py` files are **standalone diagnostic
scripts, not a pytest suite**. Each talks to real hardware and is run
directly:

```powershell
python test_xps_connection.py                  # read-only XPS queries; moves nothing
python test_serval_split_strategy.py           # probe Serval's undocumented SplitStrategy field
python test_configure_continuous_destination.py  # time each request in the destination config
```

`test_xps_connection.py` deliberately exercises only read-only queries, so
it is safe to run without knowing the physical stage setup.

**Adding a tab** — work through the checklist at the end of
`STYLE_GUIDE.md`: pick a layout, button bar then `StatusBlock` at the top,
every group a `build_section`, fields via `add_form_row`, unknown-length
text via `add_note` (never a fixed `wraplength`), empty plots get a
placeholder naming the next action, and only spacing values and color
tokens that already exist.

**Keeping live and offline identical** — anything that changes how raw data
becomes a cv4 belongs in `raw_conversion.py` or `tpx_processing.py`, not in
a tab. That is the whole reason monitored acquisition and the conversion
tab produce byte-identical output.

**Docstrings carry the reasoning.** Several modules explain *why* a
non-obvious choice was made — `serval_client.py` on batch limits and
`trust_env`, `qt_plots.py` on histogram orientation,
`quick_monitor_interface.py` on why only one preview channel works,
`sweep_interface.py` on decoupling motion from processing. Read the module
docstring before changing one of those.

---

## Reference documents

In the repo:

- **`STYLE_GUIDE.md`** — the UI style sheet, in full.
- **`tpx_file_decoding.md`** — `.tpx3` layout and the exact decoding recipe.
- **`electron_momentum_calibration.md`** — the momentum calibration model
  and its derivation, which `electron_calibration.py` implements
  section by section.
- **`tpx3_decode.ipynb`**, **`Serval Tests.ipynb`** — scratch notebooks.

Vendor manuals (gitignored, kept locally in `Timepix Documentation/`):
ASI Serval TPX3 manual V1.20 and V3.3, TPX3CAM manual V2.3, and the
Trigger mode user guide.
