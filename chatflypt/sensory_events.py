"""Encode explicit sensory items into separate, reproducible dataset tapes.

Only channel membership, annotated side, time and dimensionless intensity are
used. No requested behavioral direction or expected response reaches a decoder.
"""
import hashlib
import numbers

import numpy as np

from .brain import DT_MS


def _number(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not lower <= value <= upper:
        raise ValueError(f'{name} must be in [{lower}, {upper}]')
    return value


def _aligned(value, name):
    ticks = round(value / DT_MS)
    if abs(ticks * DT_MS - value) > 1e-8:
        raise ValueError(f'{name} must align to the {DT_MS} ms clock')
    return ticks


def _profile(channel, steps, onset, duration, intensity):
    rates = np.zeros(steps, dtype=np.float64)
    base = _number(channel['rate_hz'], 'rate_hz', 0., 1000.) * intensity
    elapsed = np.arange(duration, dtype=float) * DT_MS
    values = np.full(duration, base)
    name = channel.get('profile', 'tonic')
    params = channel.get('profile_parameters', {})
    if name == 'touch_transient':
        decay = _number(params.get('decay_ms', 15.), 'decay_ms', DT_MS, 10000.)
        values *= np.exp(-elapsed / decay)
    elif name == 'pulse_envelope':
        interval = _number(params.get('interval_ms', 36.), 'interval_ms', DT_MS, 10000.)
        width = _number(params.get('width_ms', 5.), 'width_ms', DT_MS, interval)
        interval_steps = _aligned(interval, 'interval_ms')
        width_steps = _aligned(width, 'width_ms')
        values *= np.arange(duration) % interval_steps < width_steps
    elif name != 'tonic':
        raise ValueError(f'Unknown profile: {name}')
    rates[onset:onset + duration] = values
    return rates


def encode_inputs(catalogue, items, seed, duration_ms=500., *, window_ms=None):
    """Return ``{dataset: tape}``; each tape fits social.to_engine_tape.

    An empty request returns an empty mapping. Intensity zero returns a dataset
    tape with zero events and zero driven IDs. Unsupported sides, body parts,
    unknown fields or unknown functions fail explicitly rather than broadening
    the selected cells. Dataset/ID namespaces never mix.
    """
    if window_ms is not None:
        duration_ms=_number(window_ms,'window_ms',DT_MS,100.)
    duration_ms = _number(duration_ms, 'duration_ms', DT_MS, 5000.)
    _aligned(duration_ms, 'duration_ms')
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= seed < 2**32:
        raise ValueError('seed must be a nonnegative 32-bit integer')
    if not isinstance(items, (list, tuple)):
        raise TypeError('items must be a list of sensory input specifications')
    # Requested intervals extend the observation window automatically, including
    # 100 ms after the last input. Never silently truncate a later input.
    for item in items:
        if not isinstance(item, dict):
            raise TypeError('Each input specification must be a mapping')
        onset = _number(item.get('onset_ms', 50.), 'onset_ms', 0., 4000.)
        span = _number(item.get('duration_ms', 300.), 'input duration_ms', DT_MS if window_ms is not None else 50., 2000.)
        _aligned(onset, 'onset_ms')
        _aligned(span, 'input duration_ms')
        if window_ms is None:duration_ms = max(duration_ms, onset + span + 100.)
        elif onset+span>duration_ms:raise ValueError('Measured input exceeds its feedback window')
    if duration_ms > 5000.:
        raise ValueError('Input plus 100 ms observation tail exceeds the 5000 ms limit')
    steps = _aligned(duration_ms, 'duration_ms')
    # Rates are aggregated only for selected cells, then sampled once per cell.
    # Deterministic per-cell RNGs make ordering and unrelated datasets immaterial.
    datasets = {}
    for item in items:
        if not isinstance(item, dict):
            raise TypeError('Each input specification must be a mapping')
        unexpected = set(item) - {'channel', 'side', 'intensity', 'onset_ms', 'duration_ms', 'body_part'}
        if unexpected:
            raise ValueError(f'Unsupported input fields: {sorted(unexpected)}')
        cid = item.get('channel')
        if cid not in catalogue['channels']:
            raise ValueError(f'Unknown sensory channel: {cid}')
        channel = catalogue['channels'][cid]
        if channel['status'] not in ['ready', 'candidate']:
            raise ValueError(f'Channel {cid} is unmapped: no executable functional input')
        side = item.get('side', 'bilateral')
        if side not in ['left', 'right', 'bilateral']:
            raise ValueError(f'Unknown entry side: {side}')
        side_map = channel['side_ids']
        part = item.get('body_part')
        if part is not None:
            part_map = channel.get('body_part_ids', {})
            if part not in part_map:
                raise ValueError(f'Body part {part} is not mapped for channel {cid}')
            side_map = part_map[part]
        chosen = side_map[side]
        if not chosen:
            raise ValueError(f'No annotated {side} input cells for channel {cid}')
        if len(set(chosen)) != len(chosen) or not set(chosen) <= set(channel['neuron_ids']):
            raise ValueError('Invalid catalogue side membership')
        intensity = _number(item.get('intensity', 1.), 'intensity', 0., 2.)
        onset_ms = _number(item.get('onset_ms', 50.), 'onset_ms', 0., 4000.)
        span_ms = _number(item.get('duration_ms', 300.), 'input duration_ms', DT_MS if window_ms is not None else 50., 2000.)
        onset = _aligned(onset_ms, 'onset_ms')
        span = _aligned(span_ms, 'input duration_ms')
        if onset + span > steps:
            raise ValueError('Input interval extends past the simulation duration')
        name = channel['dataset']
        dataset = datasets.setdefault(name, {'namespace': channel['id_namespace'], 'rates': {},
                                             'input_count': 0})
        if dataset['namespace'] != channel['id_namespace']:
            raise ValueError('Conflicting ID namespaces inside one dataset')
        dataset['input_count'] += 1
        if intensity == 0 or span == 0:
            continue
        rate = _profile(channel, steps, onset, span, intensity)
        for neuron_id in chosen:
            neuron_id = int(neuron_id)
            if neuron_id not in dataset['rates']:
                # A whole population shares an immutable profile until overlap.
                dataset['rates'][neuron_id] = rate
            else:
                dataset['rates'][neuron_id] = dataset['rates'][neuron_id] + rate

    result = {}
    for name in sorted(datasets):
        row = datasets[name]
        time_parts, origin_parts = [], []
        input_ids = sorted(row['rates'])
        for neuron_id in input_ids:
            rate = row['rates'][neuron_id]
            # Preserve the existing Bernoulli convention, with an explicit
            # domain check instead of silently clipping overlapping stimuli.
            if np.any(rate * DT_MS / 1000. > 1.):
                raise ValueError('Combined rate exceeds one event per simulation bin')
            key = f'{int(seed)}\0{name}\0{row["namespace"]}\0{neuron_id}'.encode()
            subseed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
            rng = np.random.default_rng(subseed)
            times = np.flatnonzero(rng.random(steps) < rate * DT_MS / 1000.)
            time_parts.append(times.astype(np.int32))
            origin_parts.append(np.full(len(times), neuron_id, dtype=np.int64))
        times = np.concatenate(time_parts) if time_parts else np.empty(0, np.int32)
        origins = np.concatenate(origin_parts) if origin_parts else np.empty(0, np.int64)
        order = np.argsort(times, kind='stable')
        result[name] = {'dataset': name, 'id_namespace': row['namespace'],
                        'input_neuron_ids': np.array(input_ids, dtype=np.int64),
                        'event_neuron_ids': origins[order], 'event_steps': times[order],
                        'duration_ms': duration_ms, 'dt_ms': DT_MS, 'seed': int(seed)}
    return result
