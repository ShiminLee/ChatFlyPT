"""State-preserving form of the existing LIF kernel for bounded co-simulation.

The one-shot kernel in brain.py remains the independent parity reference.
No drive, semantic label, or desired action is stored in neural state.
"""
import numpy as np
from numba import njit

from .brain import DT_MS, Trial


@njit(cache=True, nogil=True)
def advance(indptr, indices, weights, event_ptr, event_ids, steps, offset,
            voltage, conductance, last_spike, refractory, active, activated,
            n_active, queue, sizes, counts, state_record, observation_record):
    times_out = [np.int32(-1)]
    ids_out = [np.int32(-1)]
    av = np.exp(-0.1 / 20.0)
    ag = np.exp(-0.1 / 5.0)
    coupling = (av - ag) / 3.0
    for local in range(steps):
        step = offset + local
        slot = step % 19
        sizes[slot] = 0
        for k in range(n_active):
            i = active[k]
            if step - last_spike[i] >= refractory[i]:
                voltage[i] = -52.0 + (voltage[i] + 52.0) * av + conductance[i] * coupling
                conductance[i] *= ag
                delta = voltage[i] + 52.0
                if delta > 0:
                    state_record[0, i] += delta * DT_MS
                    state_record[2, i] = max(state_record[2, i], delta)
                    observation_record[0, i] += delta * DT_MS
                    observation_record[2, i] = max(observation_record[2, i], delta)
                elif delta < 0:
                    state_record[1, i] -= delta * DT_MS
                    state_record[3, i] = max(state_record[3, i], -delta)
                    observation_record[1, i] -= delta * DT_MS
                    observation_record[3, i] = max(observation_record[3, i], -delta)
                if voltage[i] > -45.0:
                    last_spike[i] = step
                    queue[slot, sizes[slot]] = i
                    sizes[slot] += 1
                    counts[i] += 1
                    times_out.append(np.int32(step))
                    ids_out.append(np.int32(i))
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
        for p in range(event_ptr[local], event_ptr[local + 1]):
            i = event_ids[p]
            if step != last_spike[i] and step - last_spike[i] >= refractory[i]:
                voltage[i] += 68.75
        for q in range(sizes[slot]):
            i = queue[slot, q]
            voltage[i] = -52.0
            conductance[i] = 0.0
    return n_active, np.array(ids_out[1:]), np.array(times_out[1:])


def tape_window(tape, start_step, end_step):
    """Slice events only; keep input membership and the original event order."""
    inputs, ptr, ids, steps = tape
    if not 0 <= start_step < end_step <= steps:
        raise ValueError('Invalid neural tape window')
    begin, end = ptr[start_step], ptr[end_step]
    return inputs, ptr[start_step:end_step + 1] - begin, ids[begin:end], end_step - start_step


class NeuralSession:
    def __init__(self, brain, seed):
        self.brain, self.seed, self.step = brain, seed, 0
        n = len(brain.ids)
        self.voltage = np.full(n, -52.)
        self.conductance = np.zeros(n)
        self.last_spike = np.full(n, -1000000, np.int64)
        self.refractory = np.full(n, 22, np.int64)
        self.active = np.empty(n, np.int32)
        self.activated = np.zeros(n, np.bool_)
        self.n_active = 0
        self.queue = np.zeros((19, n), np.int32)
        self.sizes = np.zeros(19, np.int32)
        self.counts = np.zeros(n, np.int32)
        self.state_record = np.zeros((4, n))
        self.observation_record = np.zeros((4, n))
        self.observation_start = 0
        self.driven = np.zeros(n, np.bool_)
        self.input_activation_step = np.full(n, -1, np.int64)
        self.spike_ids, self.spike_steps = [], []

    def run(self, tape):
        inputs, ptr, ids, steps = tape
        for i in inputs:
            if not self.driven[i]:self.input_activation_step[i]=self.step
            self.driven[i] = True
            self.refractory[i] = 0
            if not self.activated[i]:
                self.active[self.n_active] = i
                self.activated[i] = True
                self.n_active += 1
        self.n_active, origins, times = advance(
            self.brain.indptr, self.brain.indices, self.brain.weights, ptr, ids, steps, self.step,
            self.voltage, self.conductance, self.last_spike, self.refractory, self.active,
            self.activated, self.n_active, self.queue, self.sizes, self.counts, self.state_record,
            self.observation_record)
        self.spike_ids.append(origins)
        self.spike_steps.append(times)
        self.step += steps
        return self.trial()

    def trial(self):
        return Trial(self.counts.copy(), np.concatenate(self.spike_ids), np.concatenate(self.spike_steps),
                     self.seed, self.step * DT_MS, self.state_record.copy())

    def begin_observation(self):
        """Reset only passive measurements, never neural or synaptic state."""
        self.observation_start = self.step
        self.observation_record.fill(0)

    def observation_trial(self):
        trial = self.trial()
        keep = trial.spike_steps >= self.observation_start
        ids = trial.spike_ids[keep]
        return Trial(np.bincount(ids, minlength=len(self.brain.ids)).astype(np.int32),
                     ids, trial.spike_steps[keep] - self.observation_start,
                     self.seed, (self.step-self.observation_start)*DT_MS,
                     self.observation_record.copy())

    def save(self, path):
        np.savez_compressed(path, voltage=self.voltage, conductance=self.conductance,
            last_spike=self.last_spike, refractory=self.refractory, active=self.active[:self.n_active],
            queue=self.queue, queue_sizes=self.sizes, step=self.step, counts=self.counts,
            state_record=self.state_record, driven=self.driven,
            input_activation_step=self.input_activation_step, neuron_ids=self.brain.ids)


def replay_events(brain, events):
    """Replay recorded input timing and changes of directly driven membership.

    A one-shot input union changes refractory rules before late contacts occur.
    Replaying the saved schedule needs only neural IDs/times, never scene labels.
    """
    from .social import to_engine_tape
    tape=to_engine_tape(brain,events)
    activation=np.asarray(events.get('input_activation_steps',np.zeros(len(tape[0]),np.int64)))
    if (activation.shape!=(len(tape[0]),) or activation.dtype.kind not in 'iu' or
            np.any(activation<0) or np.any(activation>=tape[3])):
        raise ValueError('Invalid input activation schedule')
    start_by_id=dict(zip(map(int,events['input_neuron_ids']),activation))
    if any(step<start_by_id[int(neuron)] for step,neuron in zip(events['event_steps'],events['event_neuron_ids'])):
        raise ValueError('Event precedes its input activation')
    boundaries=sorted({0,tape[3],*map(int,activation)})
    session=NeuralSession(brain,int(events['seed']))
    for start,end in zip(boundaries,boundaries[1:]):
        _,ptr,ids,steps=tape_window(tape,start,end)
        session.run((tape[0][activation<=start],ptr,ids,steps))
    return session.trial()
