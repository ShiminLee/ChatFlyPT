"""Measured leg MN spikes -> bounded tibia targets, alongside the upstream gait.

This is an explicit engineering embodiment, not a calibrated muscle model.
Fe reductor is deliberately NOT redirected to tibia or a fused femur joint.
"""
import hashlib
import json
import math
import numpy as np

SCHEMA = 'chatflypt.tibia-motor.v1'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
CHANNELS = tuple(f'{leg}_{kind}' for leg in LEGS for kind in ('flexor', 'extensor'))
TYPES = {'flexor': ('Ti flexor MN', 'Acc. ti flexor MN'), 'extensor': ('Ti extensor MN',)}
SEGMENTS = {'T1': ('f', 'ProLN'), 'T2': ('m', 'MesoLN'), 'T3': ('h', 'MetaLN')}
GAIN_RAD = .6
SOURCES = ['https://www.nature.com/articles/s41586-024-07389-x',
           'https://elifesciences.org/articles/56754']
LIMITATION = ('新增胫节屈/伸运动神经元到六腿 tibia pitch 的有界工程适配；'
              '侧别、胸节、出神经均须匹配。角度增益未做肌肉生理校准；'
              '不同运动单位等权，未模拟肌肉力长关系或拮抗共同收缩；与 Hybrid 步态叠加未经协同校准。'
              'Fe reductor 功能及独立关节缺失，不代接成抬腿或转向。')


def channel_for(row):
    if row.get('superclass') != 'vnc_motor' or row.get('somaSide') not in ('L', 'R'):
        return None
    segment = SEGMENTS.get(row.get('somaNeuromere'))
    if segment is None or row.get('exitNerve') != segment[1]:
        return None
    kind = next((k for k, types in TYPES.items() if row.get('type') in types), None)
    return f"{row['somaSide'].lower()}{segment[0]}_{kind}" if kind else None


def courses(counts, sizes, ends, smoothing, reference):
    widths = np.diff(np.r_[0., ends])
    rates = counts / np.maximum(sizes, 1)[None, :] / (widths[:, None] / 1000.)
    filtered = np.empty_like(rates)
    state = np.zeros(len(CHANNELS))
    for i, rate in enumerate(rates):
        alpha = 1 - math.exp(-widths[i] / smoothing) if smoothing else 1.
        state += alpha * (rate - state)
        filtered[i] = state
    # Positive tibia pitch folds all six knees in this pinned YAW_PITCH_ROLL
    # skeleton (verified by the geometric included angle, not joint labels).
    # Opposed activity
    # cancels target displacement, not force/stiffness: no muscle model claim.
    normalized = np.clip(filtered / reference, 0, 1)
    offsets = GAIN_RAD * (normalized[:, 0::2] - normalized[:, 1::2])
    return rates, filtered, offsets


def build_joint_motor(rows, ids, counts, spike_indices, steps, excluded, ends,
                      ticks_per_bin, smoothing, reference, dataset, original_selected):
    groups = {k: [] for k in CHANNELS}
    if dataset.lower().startswith('malecns'):
        for i, row in enumerate(rows):
            channel = channel_for(row)
            if channel and i not in excluded:
                groups[channel].append(i)
    binned = np.zeros((len(ends), len(CHANNELS)), dtype=np.int64)
    metadata = []
    for j, (key, members) in enumerate(groups.items()):
        use = np.isin(spike_indices, members)
        binned[:, j] = np.bincount(steps[use] // ticks_per_bin, minlength=len(ends))
        metadata.append({'channel': key, 'neuron_ids': [ids[i] for i in members],
            'cells': [{k: rows[i].get(k, '') for k in ('type', 'somaSide', 'superclass', 'somaNeuromere', 'exitNerve')}
                      for i in members], 'spikes': int(counts[members].sum()),
            'active_neurons': int(np.count_nonzero(counts[members]))})
    rates, filtered, offsets = courses(binned, [len(g) for g in groups.values()], ends, smoothing, reference)
    mapped = {i for g in groups.values() for i in g}
    coverage = motor_coverage(rows, ids, counts, excluded, set(original_selected), mapped)
    return {'schema': SCHEMA, 'channels': metadata, 'spike_counts': binned.tolist(),
        'population_rate_hz': rates.tolist(), 'readouts_hz': filtered.tolist(),
        'offsets_rad': [{leg: float(v) for leg, v in zip(LEGS, row)} for row in offsets],
        'motor_spikes': int(binned.sum()), 'gain_rad': GAIN_RAD, 'biologically_calibrated': False,
        'mapping_sha256': hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
        'sources': SOURCES, 'limitation': LIMITATION, 'coverage': coverage}


def motor_coverage(rows, ids, counts, excluded, original, joint):
    active = [i for i, r in enumerate(rows) if i not in excluded and counts[i] and
              (r.get('superclass') in ('descending_neuron', 'vnc_motor', 'cb_motor') or
               r.get('super_class') in ('descending', 'motor'))]
    missing = [i for i in active if i not in original | joint]
    records = []
    for i in sorted(missing, key=lambda i: (-int(counts[i]), ids[i])):
        r = rows[i]; kind = r.get('type') or r.get('cell_type', '')
        reason = ('Fe reductor 的运动功能未确定；身体无独立转节—股节关节。'
                  if kind == 'Fe reductor MN' else
                  '未建立有依据的该细胞类型/侧别/胸节到现有身体执行器的映射。')
        records.append({'neuron_id': ids[i], 'type': kind, 'side': r.get('somaSide') or r.get('side', ''),
                        'segment': r.get('somaNeuromere', ''), 'spikes': int(counts[i]), 'reason': reason})
    return {'scope': 'Cumulative recorded motor-related activity; directly driven cells excluded.',
        'recorded_motor_spikes': int(counts[active].sum()), 'active_motor_neurons': len(active),
        'unmapped_motor_spikes': int(counts[missing].sum()), 'unmapped_active_neurons': len(missing),
        'upstream_group_spikes': int(counts[list(original)].sum()),
        'joint_group_spikes': int(counts[list(joint)].sum()), 'unmapped': records}


def validate_joint_motor(drive, original_ids):
    """Recompute targets; arbitrary angles cannot bypass recorded-rate checks."""
    payload = drive.get('joint_motor')
    if payload is None:
        return  # Legacy recorded drives preserve their exact old replay.
    if payload.get('schema') != SCHEMA or payload.get('gain_rad') != GAIN_RAD:
        raise ValueError('Invalid joint-motor mapping')
    groups = payload.get('channels', [])
    if [g.get('channel') for g in groups] != list(CHANNELS):
        raise ValueError('Invalid joint-motor channels')
    used = set(original_ids) | set(drive['directly_driven_neuron_ids'])
    sizes = []
    for g in groups:
        ids = g['neuron_ids']; cells = g['cells']
        if len(ids) != len(cells) or any(not isinstance(i, str) or not i.isascii() or not i.isdigit() or str(int(i)) != i for i in ids):
            raise ValueError('Invalid joint motor IDs')
        if len(set(ids)) != len(ids) or used.intersection(ids):
            raise ValueError('Joint motors overlap other groups or directly driven cells')
        if ids and not drive['dataset'].lower().startswith('malecns'):
            raise ValueError('MaleCNS leg IDs cannot be used in another dataset')
        if any(channel_for(c) != g['channel'] for c in cells):
            raise ValueError('Joint motor type/side/segment/nerve mismatch')
        used.update(ids); sizes.append(len(ids))
    n = len(drive['available_at_ms'])
    counts = np.asarray(payload['spike_counts'])
    if counts.shape != (n, len(CHANNELS)) or counts.dtype.kind not in 'iuf' or not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError('Invalid joint motor spike counts')
    if np.any(counts[:, np.asarray(sizes) == 0]) or payload['motor_spikes'] != int(counts.sum()):
        raise ValueError('Invalid joint motor totals')
    for i, g in enumerate(groups):
        if g['spikes'] != int(counts[:, i].sum()) or not 0 <= g['active_neurons'] <= min(g['spikes'], sizes[i]):
            raise ValueError('Invalid joint group counts')
    expected = courses(counts, sizes, drive['available_at_ms'], drive['smoothing_ms'], drive['reference_rate_hz'])
    supplied = (payload['population_rate_hz'], payload['readouts_hz'],
                [[r[k] for k in LEGS] for r in payload['offsets_rad']])
    if any(set(r) != set(LEGS) for r in payload['offsets_rad']):
        raise ValueError('Invalid joint target keys')
    for a, b in zip(expected, supplied):
        b = np.asarray(b)
        if a.shape != b.shape or b.dtype.kind not in 'iuf' or not np.isfinite(b).all() or not np.allclose(a, b, atol=1e-12, rtol=1e-12):
            raise ValueError('Joint targets differ from recorded causal rates')


def offsets_at(drive, row):
    return (dict(drive['joint_motor']['offsets_rad'][row])
            if row >= 0 and 'joint_motor' in drive else {k: 0. for k in LEGS})


def install_joint_adapter(body):
    """Keep pinned source intact and decorate its target construction only."""
    original = body._gesture
    body.joint_motor_offsets = {k: 0. for k in LEGS}
    indices = {d.child.name.split('_')[0]: i for i, d in enumerate(body.order)
               if d.child.is_leg() and d.child.link == 'tibia' and d.axis.value == 'pitch'}
    if set(indices) != set(LEGS):
        raise RuntimeError('Official body lacks the six expected tibia pitch actuators')
    def gesture(target, commands, t):
        grooming = original(target, commands, t)
        for leg, i in indices.items():
            offset = body.joint_motor_offsets[leg]
            if offset:
                # The upstream model has no hard tibia joint stops. Keep this
                # added target inside a conservative 0..pi engineering bound.
                # MuJoCo dynamics, not pose assignment, determine actual angle.
                target[i] = float(np.clip(target[i] + offset, 0., np.pi))
        return grooming
    body._gesture = gesture


def joint_summary(drive, start_ms=0., end_ms=None):
    payload = drive.get('joint_motor')
    if not payload:
        return {}
    end_ms = drive['duration_ms'] if end_ms is None else end_ms
    chosen = held_rows_in_window(drive['available_at_ms'], payload['offsets_rad'],
                                start_ms, end_ms, {k: 0. for k in LEGS})
    return {leg: {'min': min(r[leg] for r in chosen) if chosen else None,
                  'max': max(r[leg] for r in chosen) if chosen else None} for leg in LEGS}


def held_rows_in_window(times, rows, start_ms, end_ms, initial):
    """Values held during [start, end), including a pre-window completed bin."""
    import bisect
    if end_ms <= start_ms:
        return []
    index = bisect.bisect_right(times, start_ms + 1e-8) - 1
    chosen = [rows[index] if index >= 0 else initial]
    chosen.extend(row for time, row in zip(times, rows)
                  if start_ms + 1e-8 < time < end_ms - 1e-8)
    return chosen
