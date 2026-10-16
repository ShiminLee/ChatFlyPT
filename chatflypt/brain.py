"""Sparse event propagation, with the Shiu LIF equations and Brian scheduling.

Units internally are mV and ms. All neurons/edges are retained. Inactive neurons
remain at the exact resting fixed point until their first input, with no pruning.
This is our implementation, not Chetaslua's unpublished simulator.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix
from .paths import ROOT, ARTIFACTS

DT_MS = 0.1


@njit(cache=True, nogil=True)
def simulate(indptr, indices, weights, input_ids, event_ptr, event_ids, steps,
             record_spikes=False, state_record=None):
    n = len(indptr) - 1
    voltage = np.full(n, -52.0)
    conductance = np.zeros(n)
    last_spike = np.full(n, -1000000, np.int64)
    refractory = np.full(n, 22, np.int64)
    active = np.empty(n, np.int32)
    activated = np.zeros(n, np.bool_)
    n_active = 0
    for i in input_ids:
        refractory[i] = 0
        if not activated[i]:
            active[n_active] = i
            activated[i] = True
            n_active += 1
    # Store spike origins, not a dense pending conductance array.
    queue = np.empty((19, n), np.int32)
    sizes = np.zeros(19, np.int32)
    counts = np.zeros(n, np.int32)
    times_out = [np.int32(-1)]
    ids_out = [np.int32(-1)]
    av = np.exp(-0.1 / 20.0)
    ag = np.exp(-0.1 / 5.0)
    coupling = (av - ag) / 3.0
    for step in range(steps):
        slot = step % 19
        sizes[slot] = 0
        for k in range(n_active):
            i = active[k]
            if step - last_spike[i] >= refractory[i]:
                voltage[i] = -52.0 + (voltage[i] + 52.0) * av + conductance[i] * coupling
                conductance[i] *= ag
                # Passive observation: do not alter voltage, synapses or spikes.
                # Rows are positive/negative voltage area (mV ms), then peaks
                # (mV), relative to the model's -52 mV resting potential.
                if state_record is not None:
                    delta = voltage[i] + 52.0
                    if delta > 0:
                        state_record[0, i] += delta * DT_MS
                        state_record[2, i] = max(state_record[2, i], delta)
                    elif delta < 0:
                        state_record[1, i] -= delta * DT_MS
                        state_record[3, i] = max(state_record[3, i], -delta)
                if voltage[i] > -45.0:
                    last_spike[i] = step
                    queue[slot, sizes[slot]] = i
                    sizes[slot] += 1
                    counts[i] += 1
                    if record_spikes:
                        times_out.append(np.int32(step))
                        ids_out.append(np.int32(i))
        # Brian: state update -> thresholds -> delayed synapses -> input -> reset.
        delayed = (step - 18) % 19
        if step >= 18:
            for q in range(sizes[delayed]):
                source = queue[delayed, q]
                for edge in range(indptr[source], indptr[source + 1]):
                    target = indices[edge]
                    if (step != last_spike[target] and
                            step - last_spike[target] >= refractory[target]):
                        conductance[target] += weights[edge]
                        if not activated[target]:
                            activated[target] = True
                            active[n_active] = target
                            n_active += 1
        for p in range(event_ptr[step], event_ptr[step + 1]):
            i = event_ids[p]
            if step != last_spike[i] and step - last_spike[i] >= refractory[i]:
                voltage[i] += 68.75
        for q in range(sizes[slot]):
            i = queue[slot, q]
            voltage[i] = -52.0
            conductance[i] = 0.0
    return counts, np.array(ids_out[1:]), np.array(times_out[1:]), voltage, conductance


@dataclass
class Trial:
    counts: np.ndarray
    spike_ids: np.ndarray
    spike_steps: np.ndarray
    seed: int
    duration_ms: float
    state_record: np.ndarray | None = None


def recorded_trial(brain, tape, seed, duration_ms, record_spikes=True):
    """Observe both spikes and subthreshold voltage, with unchanged dynamics."""
    state = np.zeros((4, len(brain.ids)), dtype=np.float64)
    result = simulate(brain.indptr, brain.indices, brain.weights, *tape,
                      record_spikes, state)
    return Trial(result[0], result[1], result[2], seed, duration_ms, state)


class Brain:
    def __init__(self, version="783"):
        self.version = version
        names = {
            "783": ("Completeness_783.csv", "Connectivity_783.parquet"),
            "630": ("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet"),
        }
        comp, con = names[version]
        self.source_connectivity = ROOT / "vendor/shiu-brain" / con
        self.ids = pd.read_csv(ROOT / "vendor/shiu-brain" / comp, index_col=0).index.to_numpy(np.int64)
        self.lookup = {str(i): j for j, i in enumerate(self.ids)}
        cache = ARTIFACTS / f"graph-{version}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as f:
                self.indptr, self.indices, self.weights = f["indptr"], f["indices"], f["weights"]
        else:
            df = pd.read_parquet(self.source_connectivity,
                columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
            matrix = csr_matrix((df["Excitatory x Connectivity"].to_numpy(np.float64) * .275,
                               (df.Presynaptic_Index, df.Postsynaptic_Index)),
                               shape=(len(self.ids), len(self.ids)))
            self.indptr, self.indices, self.weights = matrix.indptr, matrix.indices, matrix.data
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, indptr=self.indptr, indices=self.indices, weights=self.weights)

    def input_tape(self, groups, duration_ms, seed):
        steps = round(duration_ms / DT_MS)
        rates = {}
        for ids, hz in groups:
            if not 0 <= hz <= 1000:
                raise ValueError("Stimulus rates must be 0..1000 Hz")
            for root_id in ids:
                idx = self.lookup[str(root_id)]  # Fail on absent version-specific IDs.
                if idx in rates:
                    raise ValueError("Overlapping input groups must be resolved explicitly")
                rates[idx] = float(hz)
        inputs = np.array(sorted(rates), np.int32)
        rng = np.random.default_rng(seed)
        t_parts, i_parts = [], []
        for i in inputs:
            ts = np.flatnonzero(rng.random(steps) < rates[i] * DT_MS / 1000)
            t_parts.append(ts)
            i_parts.append(np.full(len(ts), i, np.int32))
        ts = np.concatenate(t_parts) if t_parts else np.empty(0, np.int32)
        idx = np.concatenate(i_parts) if i_parts else np.empty(0, np.int32)
        order = np.argsort(ts, kind="stable")
        ptr = np.concatenate(([0], np.cumsum(np.bincount(ts, minlength=steps)))).astype(np.int64)
        return inputs, ptr, idx[order].astype(np.int32), steps

    def run(self, groups, duration_ms=300.0, seed=0, record_spikes=False):
        if not 0 < duration_ms <= 10000 or abs(duration_ms / DT_MS - round(duration_ms / DT_MS)) > 1e-8:
            raise ValueError("Duration must be a multiple of 0.1 ms in (0, 10000]")
        tape = self.input_tape(groups, duration_ms, seed)
        result = simulate(self.indptr, self.indices, self.weights, *tape, record_spikes)
        return Trial(result[0], result[1], result[2], seed, duration_ms)
