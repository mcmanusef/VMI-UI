"""Quick Monitor: a fast, low-latency live view of the detector.

Unlike the other tabs, this does not decode raw .tpx3 files at all. Instead
it tells Serval to run a long, software-started/stopped CONTINUOUS
measurement at a given frame rate, and to serve a live preview image
(Serval's own server-side pixel counting) over its built-in HTTP preview
channel. We just poll GET /measurement/image in a tight loop -- that call
blocks until the next preview image is ready, so it paces itself -- decode
the TIFF body it returns, and render it. This is orders of magnitude
cheaper than decoding/clustering raw packets, which is the point: a quick
look at what the detector is currently seeing, not an analysis.

Note: Serval's http-scheme preview channel only supports the file-image
formats 'tiff', 'pgm', 'png' -- not 'jsonimage'/'jsonhisto' (those need a
tcp:// destination instead). So both the live preview and the optional
on-disk frame saving below use tiff.

Two views, one channel: an earlier version tried to run two *simultaneous*
HTTP preview channels (raw single frame + server-side-integrated), each
given its own port in its Base, on the theory that Serval would open a
listener per port. In practice nothing listens on the extra port -- GET
/measurement/image only ever serves the single default "http://localhost"
channel, so a second channel is just dead config. Only one HTTP preview
channel actually works, so instead we always request raw (non-integrated)
frames from Serval and integrate client-side: every received frame updates
the "Single frame" panel, and is also added into a running numpy sum shown
as the "Integrated" panel. Reset via Stop+Start or the Reset button.
"""
import datetime
import io
import json
import pathlib
import threading
import time
from queue import Queue, Empty

import numpy as np
import requests
import tkinter as tk
from tkinter import ttk, filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image as PILImage

try:
    import cmasher as cmr
except Exception:
    cmr = None

import app_settings
import serval_client
from plot_panel import build_log_norm, build_power_norm

_MODE_CHOICES = [
    ("count", "Count (raw hits)"),
    ("tot", "ToT (time over threshold)"),
    ("toa", "ToA (time of arrival)"),
    ("tof", "ToF (time of flight)"),
]
_MODE_BY_LABEL = {label: key for key, label in _MODE_CHOICES}

_VIEWS = (
    ("single", "Single frame"),
    ("integrated", "Integrated"),
)


def decode_tiff_image(content: bytes) -> np.ndarray:
    """Decode a Serval 'tiff' HTTP response body into a pixel ndarray."""
    with PILImage.open(io.BytesIO(content)) as img:
        return np.array(img)


def _output_channel(base, mode, accumulate, file_pattern=None):
    """Build a Serval destination OutputChannel dict for a tiff channel.

    IntegrationMode must be omitted (not just falsy) when integration is
    disabled -- Serval rejects "IntegrationMode set but IntegrationSize is
    0 or 1" as an invalid config.
    """
    channel = {"Base": base, "Format": "tiff", "Mode": mode}
    if file_pattern is not None:
        channel["FilePattern"] = file_pattern
    if accumulate:
        channel["IntegrationSize"] = -1
        channel["IntegrationMode"] = "sum"
    else:
        channel["IntegrationSize"] = 0
    return channel


class QuickMonitorInterface(ttk.Frame):
    def __init__(self, parent, server_var=None, coordinator=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._server_var = server_var
        self._fallback_server_var = tk.StringVar(self, value="http://localhost:8080")
        self._coordinator = coordinator
        if coordinator is not None:
            coordinator.register(self)

        # Loaded from whatever was last saved via the "Save current as
        # default" button (next to the tab bar); untouched until then.
        self.frame_rate_var = tk.StringVar(self, value=app_settings.get("quick_monitor.frame_rate", "1000"))
        self.preview_period_var = tk.StringVar(self, value=app_settings.get("quick_monitor.preview_period", "0.1"))
        self.mode_choice_var = tk.StringVar(
            self, value=app_settings.get("quick_monitor.mode_choice", _MODE_CHOICES[0][1])
        )
        self.save_tiff_var = tk.BooleanVar(self, value=app_settings.get("quick_monitor.save_tiff", False))
        self.save_folder_var = tk.StringVar(
            self, value=app_settings.get("quick_monitor.save_folder", self._default_save_folder())
        )
        self.log_var = tk.BooleanVar(self, value=False)
        self.gamma_var = tk.DoubleVar(self, value=1.0)
        self.gamma_label_var = tk.StringVar(self, value="1.00")
        self.status_var = tk.StringVar(self, value="Idle")

        self.dash_status_var = tk.StringVar(self, value="--")
        self.dash_frames_var = tk.StringVar(self, value="--")
        self.dash_elapsed_var = tk.StringVar(self, value="--")
        self.dash_rate_var = tk.StringVar(self, value="--")
        self.image_rate_var = tk.StringVar(self, value="--")
        self.integrated_count_var = tk.StringVar(self, value="--")

        # Cheap cluster-rate estimate from the single (non-integrated)
        # frame's raw hit count -- not real clustering (Quick Monitor never
        # decodes packets), just total hits / assumed cluster size /
        # shots-per-frame. Only meaningful in Count mode, since ToT/ToA/ToF
        # pixel values aren't hit counts.
        self.avg_cluster_size_var = tk.StringVar(self, value=app_settings.get("quick_monitor.avg_cluster_size", "3.0"))
        self.repetition_rate_var = tk.StringVar(self, value=app_settings.get("quick_monitor.repetition_rate", "1000"))
        self.cluster_rate_var = tk.StringVar(self, value="--")
        self._active_mode = "count"

        self.avg_cluster_size_var.trace_add("write", lambda *_: self._update_cluster_rate_estimate())
        self.repetition_rate_var.trace_add("write", lambda *_: self._update_cluster_rate_estimate())

        self._stop_event = threading.Event()
        self._queue: Queue = Queue()
        self._fetch_thread = None

        self._last_image = None  # latest raw single frame, as received
        self._accumulated_image = None  # running client-side sum since Start/Reset
        self._accumulated_count = 0
        self._last_recv_time = None
        self._recv_interval_ema = None

        self._axes = {}
        self._canvases = {}

        self._build_ui()
        self._poll_queue()
        self._poll_dashboard()

    def _default_save_folder(self):
        return rf"C:\DATA\{datetime.date.today().strftime('%Y%m%d')}\quick_monitor"

    # ---- UI construction --------------------------------------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        controls.columnconfigure(11, weight=1)

        ttk.Label(controls, text="Frame rate (fps):").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.frame_rate_var, width=10).grid(row=0, column=1, padx=(6, 12))

        ttk.Label(controls, text="Preview refresh (s):").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.preview_period_var, width=10).grid(row=0, column=3, padx=(6, 12))

        ttk.Label(controls, text="Mode:").grid(row=0, column=4, sticky="w")
        mode_box = ttk.Combobox(
            controls,
            textvariable=self.mode_choice_var,
            values=[label for _, label in _MODE_CHOICES],
            state="readonly",
            width=20,
        )
        mode_box.grid(row=0, column=5, padx=(6, 12))

        ttk.Button(controls, text="Start", command=self.start).grid(row=0, column=6, padx=(0, 6))
        ttk.Button(controls, text="Stop", command=self.stop).grid(row=0, column=7, padx=(0, 6))
        ttk.Button(controls, text="Reset accumulation", command=self._reset_accumulation).grid(
            row=0, column=8, padx=(0, 6)
        )

        ttk.Label(controls, textvariable=self.status_var).grid(row=0, column=9, sticky="w")

        save_row = ttk.Frame(self)
        save_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        save_row.columnconfigure(2, weight=1)

        ttk.Checkbutton(save_row, text="Save TIFF frames to:", variable=self.save_tiff_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(save_row, textvariable=self.save_folder_var).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(6, 6))
        ttk.Button(save_row, text="...", width=3, command=self._browse_folder).grid(row=0, column=3)

        rate_row = ttk.LabelFrame(self, text="Estimated cluster rate (Count mode, single frame)")
        rate_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))

        ttk.Label(rate_row, text="Avg cluster size (px):").grid(row=0, column=0, sticky="w", padx=(6, 4), pady=4)
        ttk.Entry(rate_row, textvariable=self.avg_cluster_size_var, width=8).grid(row=0, column=1, padx=(0, 12))

        ttk.Label(rate_row, text="Repetition rate (Hz):").grid(row=0, column=2, sticky="w", padx=(0, 4))
        ttk.Entry(rate_row, textvariable=self.repetition_rate_var, width=8).grid(row=0, column=3, padx=(0, 12))

        ttk.Label(rate_row, text="Est. clusters/shot:").grid(row=0, column=4, sticky="w", padx=(0, 4))
        ttk.Label(rate_row, textvariable=self.cluster_rate_var, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=5, sticky="w", padx=(0, 6)
        )

        opts = ttk.Frame(self)
        opts.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 6))
        ttk.Checkbutton(opts, text="Log color", variable=self.log_var, command=self._redraw).grid(
            row=0, column=0, padx=(0, 12)
        )
        ttk.Label(opts, text="Gamma:").grid(row=0, column=1, sticky="w")
        gamma_scale = ttk.Scale(
            opts, from_=0.1, to=3.0, orient="horizontal", variable=self.gamma_var,
            length=160, command=self._on_gamma_change,
        )
        gamma_scale.grid(row=0, column=2, padx=(6, 6))
        ttk.Label(opts, textvariable=self.gamma_label_var, width=5).grid(row=0, column=3)
        ttk.Button(opts, text="Reset", command=self._reset_gamma).grid(row=0, column=4, padx=(6, 12))

        ttk.Separator(opts, orient="vertical").grid(row=0, column=5, sticky="ns", padx=12)
        stat_cols = [
            ("Status:", self.dash_status_var),
            ("Frames:", self.dash_frames_var),
            ("Elapsed:", self.dash_elapsed_var),
            ("Pixel rate:", self.dash_rate_var),
            ("Image rate:", self.image_rate_var),
            ("Integrated over:", self.integrated_count_var),
        ]
        for col, (label, var) in enumerate(stat_cols):
            ttk.Label(opts, text=label).grid(row=0, column=6 + col * 2, sticky="w", padx=(0, 4))
            ttk.Label(opts, textvariable=var).grid(row=0, column=7 + col * 2, sticky="w", padx=(0, 12))

        ttk.Label(
            self,
            text="Mode takes effect on the next Start. \"Integrated\" is summed client-side from the "
                 "raw frames since Start/Reset (Serval only serves one live HTTP channel at a time).",
            font=("Segoe UI", 8),
        ).grid(row=3, column=0, sticky="w", padx=10, pady=(28, 0))

        plot_frame = ttk.Frame(self)
        plot_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0, 10))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.columnconfigure(1, weight=1)

        for col, (key, title) in enumerate(_VIEWS):
            figure = Figure(figsize=(5, 5), tight_layout=True)
            ax = figure.add_subplot(1, 1, 1)
            canvas = FigureCanvasTkAgg(figure, master=plot_frame)
            pad = (0, 5) if col == 0 else (5, 0)
            canvas.get_tk_widget().grid(row=0, column=col, sticky="nsew", padx=pad)
            self._axes[key] = ax
            self._canvases[key] = canvas

        self._redraw()

    def _on_gamma_change(self, _value=None):
        self.gamma_label_var.set(f"{self.gamma_var.get():.2f}")
        self._redraw()

    def _reset_gamma(self):
        self.gamma_var.set(1.0)
        self._on_gamma_change()

    def _browse_folder(self):
        chosen = filedialog.askdirectory(
            initialdir=self.save_folder_var.get().strip() or ".",
            title="Choose TIFF save folder",
            parent=self,
        )
        if chosen:
            self.save_folder_var.set(chosen)

    def _reset_accumulation(self):
        self._accumulated_image = None
        self._accumulated_count = 0
        self.integrated_count_var.set("0 frames")
        self._redraw()

    # ---- server helpers -----------------------------------------------------

    def _server_url(self):
        url = ""
        if self._server_var is not None:
            url = self._server_var.get().strip()
        if not url:
            url = self._fallback_server_var.get().strip() or "http://localhost:8080"
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        return url

    # ---- start / stop ---------------------------------------------------

    def is_running(self):
        return bool(self._fetch_thread and self._fetch_thread.is_alive())

    def default_fields(self):
        """Fields the global "Save current as default" button can persist."""
        return {
            "quick_monitor.frame_rate": self.frame_rate_var,
            "quick_monitor.preview_period": self.preview_period_var,
            "quick_monitor.mode_choice": self.mode_choice_var,
            "quick_monitor.save_tiff": self.save_tiff_var,
            "quick_monitor.save_folder": self.save_folder_var,
            "quick_monitor.avg_cluster_size": self.avg_cluster_size_var,
            "quick_monitor.repetition_rate": self.repetition_rate_var,
        }

    def start(self):
        if self._coordinator is not None:
            self._coordinator.stop_others(self)

        if self.is_running():
            self.stop()

        try:
            frame_rate = float(self.frame_rate_var.get())
            preview_period = float(self.preview_period_var.get())
            if frame_rate <= 0 or preview_period <= 0:
                raise ValueError("Frame rate and preview refresh must be positive.")
        except ValueError as exc:
            self.status_var.set(f"Invalid setting: {exc}")
            return

        mode = _MODE_BY_LABEL.get(self.mode_choice_var.get(), "count")
        save_tiff = self.save_tiff_var.get()
        server = self._server_url()

        image_channels = []
        if save_tiff:
            folder_text = self.save_folder_var.get().strip()
            if not folder_text:
                self.status_var.set("Save folder is required to save TIFF frames.")
                return
            folder = pathlib.Path(folder_text)
            folder.mkdir(parents=True, exist_ok=True)
            image_channels.append(
                _output_channel(folder.as_uri(), mode, accumulate=False, file_pattern="quick%Hms_")
            )

        # No single CONTINUOUS-mode measurement is left running indefinitely
        # -- each batch is capped to at most max_frames_per_batch(exposure)
        # frames (a minute/exposure worth); _fetch_loop reconfigures +
        # restarts for every subsequent batch itself once the current one's
        # frames are used up.
        exposure = 1.0 / frame_rate
        batch_cap = serval_client.max_frames_per_batch(exposure)

        try:
            config = json.loads(serval_client.SESSION.get(f"{server}/detector/config", timeout=5).text)
            config["TriggerMode"] = "CONTINUOUS"
            config["ExposureTime"] = exposure
            config["TriggerPeriod"] = exposure
            config["nTriggers"] = batch_cap
            serval_client.SESSION.put(f"{server}/detector/config", data=json.dumps(config), timeout=5)

            destination = {
                "Raw": [],
                "Image": image_channels,
                "Preview": {
                    "Period": preview_period,
                    "SamplingMode": "skipOnFrame",
                    # Always raw (non-integrated) -- only one HTTP preview
                    # channel is actually servable via GET
                    # /measurement/image (see module docstring), so
                    # "Integrated" is summed client-side instead of asking
                    # Serval for a second, server-integrated channel.
                    "ImageChannels": [_output_channel("http://localhost", mode, accumulate=False)],
                    "HistogramChannels": [],
                },
            }
            serval_client.SESSION.put(f"{server}/server/destination", data=json.dumps(destination), timeout=5)

            serval_client.SESSION.get(f"{server}/measurement/start", timeout=5)
        except Exception as exc:
            self.status_var.set(f"Config/start failed: {type(exc).__name__}: {exc} (server={server})")
            return

        self._stop_event.clear()
        self._last_image = None
        self._accumulated_image = None
        self._accumulated_count = 0
        self.integrated_count_var.set("0 frames")
        self._last_recv_time = None
        self._recv_interval_ema = None
        self._active_mode = mode
        suffix = ", saving TIFF" if save_tiff else ""
        self.status_var.set(f"Running ({mode}{suffix}).")

        self._fetch_thread = threading.Thread(
            target=self._fetch_loop,
            args=(server, preview_period, exposure, batch_cap),
            daemon=True,
        )
        self._fetch_thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            serval_client.SESSION.get(f"{self._server_url()}/measurement/stop", timeout=5)
        except Exception:
            pass
        self.status_var.set("Stopping...")
        if self._fetch_thread:
            self._fetch_thread.join(timeout=5)
        self.status_var.set("Stopped.")

    # ---- fetch thread: blocking-poll Serval's preview channel -------------

    def _restart_batch(self, server, exposure, n_triggers):
        """Reconfigure nTriggers for a fresh batch and restart -- called
        once the previous batch's frames are used up (Serval will have
        already auto-stopped on its own by then). Only /detector/config
        needs resetting; the destination (channels, save-tiff folder, ...)
        doesn't change between batches."""
        config = json.loads(serval_client.SESSION.get(f"{server}/detector/config", timeout=5).text)
        config["TriggerMode"] = "CONTINUOUS"
        config["ExposureTime"] = exposure
        config["TriggerPeriod"] = exposure
        config["nTriggers"] = n_triggers
        serval_client.SESSION.put(f"{server}/detector/config", data=json.dumps(config), timeout=5)
        serval_client.SESSION.get(f"{server}/measurement/start", timeout=5)

    def _fetch_loop(self, server, preview_period, exposure, batch_cap):
        # /measurement/image blocks server-side until the next preview image
        # is ready, so this loop paces itself off Serval's own preview
        # period -- no client-side timing needed. The timeout is just a
        # safety net (e.g. if the measurement gets stopped mid-wait).
        timeout = max(5.0, preview_period * 5.0 + 5.0)
        frame_no = 0
        start = time.perf_counter()
        # No single start_measurement() (see start()) is left running for
        # more than a minute -- once this batch's frames are used up
        # (Serval auto-stops itself; wall time is used as the client-side
        # proxy for "used up" since preview mode doesn't map 1:1 to raw
        # frames received the way the other tabs' collector loops do),
        # reconfigure + restart for the next one.
        batch_seconds = batch_cap * exposure
        batch_start = time.perf_counter()
        while not self._stop_event.is_set():
            if time.perf_counter() - batch_start >= batch_seconds:
                try:
                    serval_client.wait_for_measurement_idle(server, timeout=10.0)
                    self._restart_batch(server, exposure, batch_cap)
                except Exception as exc:
                    if not self._stop_event.is_set():
                        self._queue.put({"error": f"Batch restart failed: {exc}"})
                batch_start = time.perf_counter()
            try:
                resp = serval_client.SESSION.get(f"{server}/measurement/image", timeout=timeout)
                if resp.status_code != 200 or not resp.content:
                    time.sleep(0.1)
                    continue
                image = decode_tiff_image(resp.content)
                frame_no += 1
                # Serval's tiff response carries no per-frame header (unlike
                # jsonimage), so frame number/elapsed time are tracked here
                # client-side, off however many images we've received.
                header = {"frameNumber": frame_no, "timeAtFrame": time.perf_counter() - start}
                self._queue.put({"header": header, "image": image})
            except requests.exceptions.RequestException as exc:
                if not self._stop_event.is_set():
                    self._queue.put({"error": str(exc)})
                time.sleep(0.5)
            except Exception as exc:
                self._queue.put({"error": f"Decode failed: {exc}"})
                time.sleep(0.2)

    # ---- dashboard polling (cheap, non-blocking) ---------------------------

    def _poll_dashboard(self):
        try:
            resp = serval_client.SESSION.get(f"{self._server_url()}/dashboard", timeout=1)
            data = json.loads(resp.text)
            measurement = data.get("Measurement", {})
            self.dash_status_var.set(str(measurement.get("Status", "--")))
            self.dash_frames_var.set(str(measurement.get("FrameCount", "--")))
            elapsed = measurement.get("ElapsedTime")
            self.dash_elapsed_var.set(f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else "--")
            rate = measurement.get("PixelEventRate")
            self.dash_rate_var.set(f"{rate:,}/s" if isinstance(rate, (int, float)) else "--")
        except Exception:
            pass
        # Runs unconditionally for the app's entire lifetime regardless of
        # which tab is active. Now goes through serval_client.SESSION (a
        # shared, reused requests.Session -- see serval_client.py) rather
        # than a bare requests.get(), which used to open/tear down a fresh
        # TCP connection every call.
        self.after(2000, self._poll_dashboard)

    # ---- main-thread queue handling / drawing -----------------------------

    def _poll_queue(self):
        latest = None
        error = None
        try:
            while True:
                result = self._queue.get_nowait()
                if "error" in result:
                    error = result["error"]
                else:
                    latest = result  # keep only the newest image: never fall behind on display
        except Empty:
            pass

        if error is not None:
            self.status_var.set(f"Error: {error}")

        if latest is not None:
            now = time.perf_counter()
            if self._last_recv_time is not None:
                dt = now - self._last_recv_time
                if dt > 0:
                    inst_rate = 1.0 / dt
                    self._recv_interval_ema = (
                        inst_rate if self._recv_interval_ema is None
                        else 0.8 * self._recv_interval_ema + 0.2 * inst_rate
                    )
                    self.image_rate_var.set(f"{self._recv_interval_ema:.1f} Hz")
            self._last_recv_time = now
            self._last_image = latest

            image = latest["image"].astype(np.float64)
            if self._accumulated_image is None or self._accumulated_image.shape != image.shape:
                self._accumulated_image = image.copy()
            else:
                self._accumulated_image += image
            self._accumulated_count += 1
            self.integrated_count_var.set(f"{self._accumulated_count} frame(s)")

            self._redraw()

        self.after(50, self._poll_queue)

    def _redraw(self):
        for key, title in _VIEWS:
            ax = self._axes[key]
            ax.clear()
            image = self._last_image["image"] if (key == "single" and self._last_image) else None
            if key == "integrated":
                image = self._accumulated_image
            if image is not None:
                cmap = cmr.rainforest if cmr is not None else "viridis"
                norm = build_log_norm(image) if self.log_var.get() else build_power_norm(image, self.gamma_var.get())
                ax.imshow(image, origin="upper", cmap=cmap, norm=norm)
                if key == "single" and self._last_image:
                    header = self._last_image["header"]
                    frame_no = header.get("frameNumber", "?")
                    t_frame = header.get("timeAtFrame", 0.0)
                    ax.set_title(f"{title}: frame {frame_no}  (t = {t_frame:.3f}s)")
                else:
                    ax.set_title(f"{title}: {self._accumulated_count} frame(s)")
            else:
                ax.set_title(f"{title}: no image yet -- press Start")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            self._canvases[key].draw_idle()

        self._update_cluster_rate_estimate()

    def _update_cluster_rate_estimate(self):
        """Cheap clusters-per-shot estimate from the single (non-integrated)
        frame's raw hit count: no real clustering (Quick Monitor never
        decodes packets), just total hits / assumed cluster size /
        shots-per-frame. Recomputed every time a new frame redraws, and
        whenever the size/rate fields change -- there's no separate slower
        timer.
        """
        if self._last_image is None or self._active_mode != "count":
            self.cluster_rate_var.set("--")
            return
        try:
            avg_cluster_size = float(self.avg_cluster_size_var.get())
            repetition_rate = float(self.repetition_rate_var.get())
            frame_rate = float(self.frame_rate_var.get())
            if avg_cluster_size <= 0 or repetition_rate <= 0 or frame_rate <= 0:
                raise ValueError
        except ValueError:
            self.cluster_rate_var.set("--")
            return

        total_hits = float(np.sum(self._last_image["image"]))
        shots_per_frame = repetition_rate / frame_rate
        hits_per_shot = total_hits / shots_per_frame
        clusters_per_shot = hits_per_shot / avg_cluster_size
        self.cluster_rate_var.set(f"{clusters_per_shot:.3g}")
