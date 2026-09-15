"""Writer for the "cv4" HDF5 layout.

Layout (as observed in a reference .cv4 file):

    t_pulse       (N_pulses,)  f8   absolute laser-trigger time (ns, camera clock)
    x, y, t       (N_cluster,) f8   cluster position (detector px) and time
                                     relative to its pulse (ns)
    cluster_corr  (N_cluster,) i4   index into t_pulse for each cluster event
    t_etof        (N_etof,)    f8   e-ToF time relative to its pulse (ns)
    etof_corr     (N_etof,)    i4   index into t_pulse for each e-ToF event
    t_tof         (N_tof,)     f8   i-ToF time relative to its pulse (ns)
    tof_corr      (N_tof,)     i4   index into t_pulse for each i-ToF event

x/y/t/cluster_corr always have matching length, same for t_etof/etof_corr and
t_tof/tof_corr. t_pulse[0] is a -1.0 sentinel (no events reference it; it just
keeps the indexing convention seen in reference files where the first real
pulse lands at index 1).

Run metadata is stored as attributes on the root group.
"""
import pathlib

import h5py
import numpy as np

_DATASETS = {
    "t_pulse": "f8",
    "x": "f8",
    "y": "f8",
    "t": "f8",
    "cluster_corr": "i4",
    "t_etof": "f8",
    "etof_corr": "i4",
    "t_tof": "f8",
    "tof_corr": "i4",
}


class Cv4Writer:
    def __init__(self, path, metadata=None):
        """Opens `path` for appending. If it doesn't exist yet, creates it
        fresh (datasets + the sentinel pulse); if it already does (a
        previous Cv4Writer for the same path was opened, written to, and
        closed), reopens it in place -- existing data is kept, and
        _pulse_count is re-derived from what's already on disk so further
        append_frame() calls keep indexing correctly. This is what lets
        callers close the file after each raw file is processed and reopen
        it for the next one instead of holding one handle open for an
        entire run."""
        self.path = pathlib.Path(path)
        is_new = not self.path.exists()
        self._file = h5py.File(str(self.path), "a")
        self._datasets = {}

        if is_new:
            for name, dtype in _DATASETS.items():
                self._datasets[name] = self._file.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True
                )
            # Sentinel pulse at index 0; no event ever points at it today,
            # but it keeps corr==0 reserved for "no matching pulse" per the
            # cv4 convention.
            self._append("t_pulse", np.array([-1.0], dtype=np.float64))
            self._pulse_count = 1
        else:
            for name in _DATASETS:
                self._datasets[name] = self._file[name]
            self._pulse_count = self._datasets["t_pulse"].shape[0]

        if metadata:
            self.set_attrs(metadata)

    def set_attrs(self, attrs):
        """Set/update root-group attributes after construction -- used to
        record things only known once collection ends (frames collected vs.
        processed, whether it completed naturally or was stopped early)."""
        for key, value in attrs.items():
            if value is None:
                continue
            try:
                self._file.attrs[key] = value
            except TypeError:
                self._file.attrs[key] = str(value)

    def _append(self, name, arr):
        if arr is None or len(arr) == 0:
            return
        ds = self._datasets[name]
        old = ds.shape[0]
        ds.resize(old + len(arr), axis=0)
        ds[old:old + len(arr)] = arr

    def append_frame(self, pulses_sorted, clusters_by_pulse, etof_by_pulse, itof_by_pulse):
        """Append one processed raw-file's worth of events.

        pulses_sorted: sorted list of absolute pulse times (ns) for this frame.
        clusters_by_pulse / etof_by_pulse / itof_by_pulse: dicts keyed by the
        same pulse times, as produced by tpx_processing's grouping helpers
        (etof/itof values are times relative to their pulse).
        """
        n_pulses = len(pulses_sorted)
        if n_pulses == 0:
            return

        base_index = self._pulse_count

        xs, ys, ts, cc = [], [], [], []
        te, ec = [], []
        tt, tc = [], []
        for local_idx, pulse_time in enumerate(pulses_sorted):
            idx = base_index + local_idx
            for cluster in clusters_by_pulse.get(pulse_time, []):
                xs.append(cluster["avg_x"])
                ys.append(cluster["avg_y"])
                ts.append(cluster["avg_t_rel_ns"])
                cc.append(idx)
            for rel_t in etof_by_pulse.get(pulse_time, []):
                te.append(rel_t)
                ec.append(idx)
            for rel_t in itof_by_pulse.get(pulse_time, []):
                tt.append(rel_t)
                tc.append(idx)

        self._append("t_pulse", np.asarray(pulses_sorted, dtype=np.float64))
        self._append("x", np.asarray(xs, dtype=np.float64))
        self._append("y", np.asarray(ys, dtype=np.float64))
        self._append("t", np.asarray(ts, dtype=np.float64))
        self._append("cluster_corr", np.asarray(cc, dtype=np.int32))
        self._append("t_etof", np.asarray(te, dtype=np.float64))
        self._append("etof_corr", np.asarray(ec, dtype=np.int32))
        self._append("t_tof", np.asarray(tt, dtype=np.float64))
        self._append("tof_corr", np.asarray(tc, dtype=np.int32))

        self._pulse_count += n_pulses

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def partial_path_for(final_path):
    """The in-progress filename to actually write to: <name>.partial.cv4
    instead of <name>.cv4. Renamed to the real name only once the caller
    deems the file complete (see finalize_partial_path) -- so anything left
    named .partial.cv4 (including after a crash) is honestly incomplete."""
    final_path = pathlib.Path(final_path)
    return final_path.with_name(final_path.stem + ".partial.cv4")


def finalize_partial_path(partial_path, completed):
    """Drop the .partial suffix if `completed`; otherwise leave the
    .partial name. What counts as complete is up to the caller (e.g. a
    fixed-length run reaching its target, or every raw file that was
    actually collected having been processed). Returns the resulting path."""
    partial_path = pathlib.Path(partial_path)
    if not completed:
        return partial_path
    final_path = partial_path.with_name(partial_path.name[: -len(".partial.cv4")] + ".cv4")
    partial_path.rename(final_path)
    return final_path
