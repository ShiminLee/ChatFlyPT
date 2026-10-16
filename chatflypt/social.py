"""Natural social-signal input catalogue and reproducible sensory event encoder.

Anatomical targets have source evidence; the generated rate envelopes are
explicit simulation proxies, not recorded natural receptor spike trains.
MaleCNS body IDs and FlyWire root IDs never share an ID namespace.
"""
import argparse
import json

import numpy as np
import pandas as pd

from .brain import Brain, DT_MS, simulate
from .paths import ROOT, dump, sha256
from .stimuli import ANNOTATIONS

OUT = ROOT / 'data/derived/social-input'
REPORT = ROOT / 'reports/social-input'
MALE = ROOT / 'data/raw/malecns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather'
TOUCH_SOURCE = 'https://www.nature.com/articles/nature14024'
CVA_SOURCE = 'https://www.nature.com/articles/nature06808'
SONG_SOURCE = 'https://www.nature.com/articles/s41467-017-02453-9'

MESSAGES = {
    '模拟右侧触碰': 'right_foreleg_touch',
    '触碰右前腿': 'right_foreleg_touch',
    '模拟左侧触碰': 'left_foreleg_touch',
    '触碰左前腿': 'left_foreleg_touch',
    '释放cVA信息素': 'cva',
    '发出求偶脉冲鸣声': 'courtship_pulse',
}


def build_catalogue():
    male = pd.read_feather(MALE).fillna('')
    fly = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False).fillna('')
    version_ids = set(pd.read_csv(ROOT / 'vendor/shiu-brain/Completeness_783.csv',
                                 index_col=0).index.astype(str))
    channels = {}
    for side, code in [('right', 'R'), ('left', 'L')]:
        selected = male[
            male.superclass.eq('vnc_sensory') &
            male['class'].eq('mechanosensory_tactile') &
            male.subclass.eq('mechanosensory bristle') &
            male.entryNerve.eq('ProLN') & male.rootSide.eq(code)]
        channels[f'{side}_foreleg_touch'] = {
            'dataset': 'MaleCNS v1.0', 'id_namespace': 'malecns_body_id',
            'neuron_ids': sorted(selected.bodyId.astype(str), key=int),
            'selection': {'superclass': 'vnc_sensory', 'class': 'mechanosensory_tactile',
                          'subclass': 'mechanosensory bristle',
                          'entryNerve': 'ProLN', 'rootSide': code},
            'source': TOUCH_SOURCE,
            'evidence': {
                'signal_and_receptor_class': 'Supported: inter-fly distal-leg touch; mechanosensory sensilla and NOMPC; Ramdya 2015 Fig. 3.',
                'specific_connectome_cells': 'Candidate only: whole foreleg bristle population, not a validated distal sensillum subset.',
                'physiological_spike_encoding': 'Uncalibrated engineering envelope.',
                'human_direction_semantics': 'Proposed only; location-dependent trajectories do not establish a universal direction command.',
                'experimental_context': 'Main experiments: adult females, 2-4 days post-eclosion. Transfer to male CNS needs validation.',
            },
            'biological_signal': f'Conspecific contact with the {side} foreleg',
            'target_status': 'Candidate foreleg bristle population; distal sensillum subset unresolved',
            'profile': 'touch_transient',
            'parameters': {'onset_ms': 50.0, 'end_ms': 150.0,
                           'peak_hz': 150.0, 'decay_ms': 15.0},
            'parameter_evidence': 'All rate/timing parameters are engineering assumptions; no calibrated force-to-spike transfer function.',
            'response_claim': 'Encounter locomotion is supported by the paper; no guaranteed left/right turn from this population.',
            'current_engine': 'Separate MaleCNS v1.0 brain and VNC LIF engine with internal-activity touch readout; FlyWire IDs are not mixed in.',
        }
    selectors = {
        'cva': fly.super_class.eq('sensory') & fly.cell_class.eq('olfactory') & fly.cell_type.eq('ORN_DA1'),
        'courtship_pulse': fly.super_class.eq('sensory') & fly.cell_class.eq('mechanosensory') &
                          fly.cell_sub_class.eq('auditory') & fly.cell_type.str.match(r'^JO-[AB]'),
    }
    for name, mask in selectors.items():
        selected = fly[mask & fly.root_id.astype(str).isin(version_ids)]
        channels[name] = {
            'dataset': 'FlyWire v783', 'id_namespace': 'flywire_root_id',
            'neuron_ids': sorted(selected.root_id.astype(str), key=int),
            'selection': ({'super_class': 'sensory', 'cell_class': 'olfactory', 'cell_type': 'ORN_DA1'}
                          if name == 'cva' else {'super_class': 'sensory', 'cell_class': 'mechanosensory',
                                               'cell_sub_class': 'auditory', 'cell_type_regex': '^JO-[AB]'}),
            'source': CVA_SOURCE if name == 'cva' else SONG_SOURCE,
            'evidence': {
                'signal_and_receptor_class': 'Supported by the cited primary study.',
                'specific_connectome_cells': 'Annotation-based receptor-class selection; not individually response-calibrated.',
                'physiological_spike_encoding': 'Uncalibrated engineering envelope.',
                'human_direction_semantics': 'No natural-language meaning assigned.',
            },
            'biological_signal': 'Male cVA pheromone' if name == 'cva' else 'Courtship pulse song',
            'target_status': 'Annotated receptor-class targets; individual response calibration absent',
            'profile': 'tonic' if name == 'cva' else 'pulse_envelope',
            'parameters': ({'onset_ms': 50.0, 'end_ms': 350.0, 'peak_hz': 100.0}
                           if name == 'cva' else {'onset_ms': 50.0, 'end_ms': 450.0,
                                                'peak_hz': 150.0, 'interval_ms': 36.0, 'width_ms': 5.0}),
            'parameter_evidence': ('Tonic 100 Hz is an engineering proxy, not a calibrated cVA concentration.'
                if name == 'cva' else '36 ms interpulse interval follows the synthetic-song example in Fig. 3; 5 ms gate and 150 Hz are assumptions. No carrier waveform, phase locking or adaptation model.'),
            'response_claim': ('Context- and sex-dependent social cue; no fixed aggression/attraction meaning.'
                               if name == 'cva' else 'Auditory social input; no validated mating decision or language meaning.'),
            'current_engine': 'FlyWire v783 input compatible',
        }
    if any(not c['neuron_ids'] for c in channels.values()):
        raise ValueError('Empty social sensory channel')
    catalogue = {
        'schema': 2, 'evidence_review_date': '2026-09-19',
        'duration_ms': 500.0, 'dt_ms': DT_MS,
        'scope': 'Natural communication-inspired sensory inputs; candidate targets and uncalibrated spike envelopes are explicit.',
        'annotation_sha256': {'malecns': sha256(MALE), 'flywire': sha256(ANNOTATIONS)},
        'messages': MESSAGES, 'channels': channels,
        'legacy_touch_note': 'The old 16-word touch proxy is BM_Taste. It is not a foreleg touch channel and is not reused here.',
    }
    dump(OUT / 'catalogue.json', catalogue)
    return catalogue


def resolve_message(message, catalogue):
    """Small explicit text adapter; unknown/negated sentences are never guessed."""
    message = message.strip().rstrip('。.!！')
    if message in catalogue['channels']:
        return message
    if message in catalogue['messages']:
        return catalogue['messages'][message]
    raise ValueError('Unsupported message; use a documented social-signal phrase or channel name')


def rate_envelope(spec, duration_ms=500.0):
    if not np.isfinite(duration_ms) or not 0 < duration_ms <= 10000:
        raise ValueError('Duration must be in (0, 10000] ms')
    steps = round(duration_ms / DT_MS)
    if abs(steps * DT_MS - duration_ms) > 1e-8:
        raise ValueError('Duration must align to the 0.1 ms simulation clock')
    p = spec['parameters']
    if not all(np.isfinite(x) for x in p.values()):
        raise ValueError('Nonfinite signal parameter')
    if not 0 <= p['onset_ms'] < p['end_ms'] <= duration_ms or not 0 <= p['peak_hz'] <= 1000:
        raise ValueError('Invalid signal timing or rate')
    time = np.arange(steps) * DT_MS
    elapsed = time - p['onset_ms']
    gate = (elapsed >= 0) & (time < p['end_ms'])
    rate = np.zeros(steps)
    if spec['profile'] == 'touch_transient':
        if p['decay_ms'] <= 0:
            raise ValueError('Invalid decay')
        rate[gate] = p['peak_hz'] * np.exp(-elapsed[gate] / p['decay_ms'])
    elif spec['profile'] == 'tonic':
        rate[gate] = p['peak_hz']
    elif spec['profile'] == 'pulse_envelope':
        if not 0 < p['width_ms'] <= p['interval_ms']:
            raise ValueError('Invalid pulse timing')
        rate[gate & ((elapsed % p['interval_ms']) < p['width_ms'])] = p['peak_hz']
    else:
        raise ValueError('Unknown signal profile')
    return rate


def encode(spec, seed, duration_ms=500.0):
    rate = rate_envelope(spec, duration_ms)
    ids = np.array(spec['neuron_ids'], dtype=np.int64)
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Sensory target IDs must be nonempty and unique')
    ids.sort()
    rng = np.random.default_rng(seed)
    steps, origins = [], []
    for root_id in ids:
        times = np.flatnonzero(rng.random(len(rate)) < rate * DT_MS / 1000)
        steps.append(times)
        origins.append(np.full(len(times), root_id, dtype=np.int64))
    steps, origins = np.concatenate(steps), np.concatenate(origins)
    order = np.argsort(steps, kind='stable')
    return {'id_namespace': spec['id_namespace'], 'dataset': spec['dataset'],
            'input_neuron_ids': ids, 'event_neuron_ids': origins[order],
            'event_steps': steps[order].astype(np.int32), 'dt_ms': DT_MS,
            'duration_ms': duration_ms, 'seed': seed}


def to_engine_tape(brain, events):
    dataset=getattr(brain,'dataset',f'FlyWire v{brain.version}')
    namespace=getattr(brain,'id_namespace','flywire_root_id')
    if events['dataset'] != dataset or events['id_namespace'] != namespace:
        raise ValueError('Dataset/ID namespace mismatch; use the matching connectome engine')
    steps = round(events['duration_ms'] / DT_MS)
    times = np.asarray(events['event_steps'])
    if (events['dt_ms'] != DT_MS or np.any(times < 0) or np.any(times >= steps)
            or np.any(np.diff(times) < 0) or len(times) != len(events['event_neuron_ids'])):
        raise ValueError('Invalid event tape')
    targets = {int(x) for x in events['input_neuron_ids']}
    if not set(map(int, events['event_neuron_ids'])).issubset(targets):
        raise ValueError('Event outside selected sensory targets')
    inputs = np.array([brain.lookup[str(x)] for x in events['input_neuron_ids']], np.int32)
    indices = np.array([brain.lookup[str(x)] for x in events['event_neuron_ids']], np.int32)
    ptr = np.concatenate(([0], np.cumsum(np.bincount(times, minlength=steps)))).astype(np.int64)
    return inputs, ptr, indices, steps


def export(catalogue, message, seed=91000):
    name = resolve_message(message, catalogue)
    spec = catalogue['channels'][name]
    events = encode(spec, seed, catalogue['duration_ms'])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{name}-{seed}.npz'
    np.savez_compressed(path, **events)
    # Keep intent/evidence metadata separate from the neural event payload.
    metadata = {'message': message, 'channel': name, 'seed': seed,
                'selected_cells': len(events['input_neuron_ids']),
                'input_events': len(events['event_steps']), 'event_file': str(path),
                'event_sha256': sha256(path), 'specification': spec}
    dump(path.with_suffix('.json'), metadata)
    return metadata


def audit_foreleg_connectivity(catalogue):
    """Check selected IDs against the actual MaleCNS edge table, without dynamics."""
    import pyarrow as pa
    path = ROOT / 'data/raw/malecns-v1.0/connectome-weights-male-cns-v1.0-minconf-0.5.feather'
    names = ['right_foreleg_touch', 'left_foreleg_touch']
    ids = {name: np.array(catalogue['channels'][name]['neuron_ids'], dtype=np.int64) for name in names}
    candidates = np.concatenate(list(ids.values()))
    edges = {name: [] for name in names}
    # Iterate Arrow batches: the complete edge table has >150 million rows.
    with pa.memory_map(str(path), 'r') as source:
        reader = pa.ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            pres = batch.column('body_pre').to_numpy()
            keep = np.isin(pres, candidates)
            if not keep.any():
                continue
            post = batch.column('body_post').to_numpy()[keep]
            weight = batch.column('weight').to_numpy()[keep]
            pres = pres[keep]
            for name in names:
                selected = np.isin(pres, ids[name])
                edges[name].extend(zip(pres[selected].tolist(), post[selected].tolist(),
                                       weight[selected].tolist()))
    result = {'scope': 'Anatomical input connectivity only; not a simulated or observed turn',
              'dataset': 'MaleCNS v1.0',
              'target_count_note': 'Raw edge targets can include small/unclassified fragments; they are not a count of identified downstream neurons.',
              'channels': {}}
    annotations = pd.read_feather(MALE).fillna('')
    classified = set(annotations.loc[annotations.superclass.ne(''), 'bodyId'].astype(int))
    for name in names:
        linked = {edge[0] for edge in edges[name]}
        targets = {edge[1] for edge in edges[name]}
        result['channels'][name] = {
            'selected_sensory_cells': len(ids[name]), 'cells_with_outgoing_edges': len(linked),
            'outgoing_edge_rows': len(edges[name]), 'summed_connection_weight': sum(e[2] for e in edges[name]),
            'unique_raw_target_bodies': len(targets),
            'targets_with_annotated_superclass': len(targets & classified),
            'selected_ids_without_outgoing_edges': sorted(set(map(int, ids[name])) - linked),
        }
    dump(REPORT / 'foreleg-connectivity.json', result)
    return result


def verify(catalogue):
    brain = Brain('783')
    annotations = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False).fillna('')
    rows = annotations.set_index(annotations.root_id.astype(str)).reindex(brain.ids.astype(str)).fillna('')
    nonsensory = ~rows.super_class.isin(['sensory', 'sensory_ascending']).to_numpy()
    result = {'scope': 'Input delivery and downstream activity, not natural behavior validation',
              'flywire_trials': [], 'exports': [],
              'foreleg_dynamic_simulation': 'Separate MaleCNS brain and VNC engine; uncalibrated transferred LIF parameters',
              'malecns_trials': [],
              'natural_touch_calibration': 'pending distal receptor selection and physiological spike calibration'}
    for name in catalogue['channels']:
        item = export(catalogue, name)
        result['exports'].append({k: v for k, v in item.items() if k != 'specification'})
    for name in ['cva', 'courtship_pulse']:
        spec = catalogue['channels'][name]
        for seed in [91000, 91001, 91002]:
            events = encode(spec, seed)
            tape = to_engine_tape(brain, events)
            counts, spike_ids, spike_steps, _, _ = simulate(
                brain.indptr, brain.indices, brain.weights, *tape, True)
            downstream = nonsensory.copy()
            downstream[tape[0]] = False
            before_input = int(np.sum(spike_steps < round(spec['parameters']['onset_ms'] / DT_MS)))
            np.savez_compressed(OUT / f'{name}-{seed}-response.npz', counts=counts,
                                spike_neuron_ids=brain.ids[spike_ids], spike_steps=spike_steps,
                                seed=seed, dt_ms=DT_MS, duration_ms=events['duration_ms'])
            result['flywire_trials'].append({
                'channel': name, 'seed': seed, 'input_events': len(events['event_steps']),
                'total_spikes': int(counts.sum()),
                'nonsensory_spikes': int(counts[downstream].sum()),
                'active_nonsensory_cells': int(np.count_nonzero(counts[downstream])),
                'spikes_before_input': before_input,
                'delivery_passed': bool(counts[downstream].sum() > 0 and before_input == 0)})
    blank = brain.run([], duration_ms=500, seed=91000)
    result['blank_total_spikes'] = int(blank.counts.sum())
    try:
        to_engine_tape(brain, encode(catalogue['channels']['right_foreleg_touch'], 91000))
    except ValueError:
        result['cross_dataset_input_rejected'] = True
    else:
        result['cross_dataset_input_rejected'] = False
    from .malecns import MaleCNS
    male = MaleCNS()
    for name in ['right_foreleg_touch', 'left_foreleg_touch']:
        events = encode(catalogue['channels'][name], 91000)
        trial = male.run_events(events)
        before = int(np.sum(trial.spike_steps < 500))
        result['malecns_trials'].append({'channel': name, 'seed': 91000,
            'input_events': len(events['event_steps']), 'total_spikes': int(trial.counts.sum()),
            'nonsensory_spikes': int(trial.counts[male.nonsensory].sum()),
            'spikes_before_input': before,
            'delivery_passed': bool(trial.counts[male.nonsensory].sum() > 0 and before == 0)})
    result['input_delivery_checks_passed'] = (
        all(x['delivery_passed'] for x in result['flywire_trials']) and
        all(x['delivery_passed'] for x in result['malecns_trials']) and
        result['blank_total_spikes'] == 0 and result['cross_dataset_input_rejected'])
    dump(REPORT / 'verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'encode', 'verify', 'audit-touch'])
    parser.add_argument('--message', default='模拟右侧触碰')
    parser.add_argument('--seed', type=int, default=91000)
    args = parser.parse_args()
    catalogue = build_catalogue()
    if args.action == 'build':
        result = {name: len(c['neuron_ids']) for name, c in catalogue['channels'].items()}
    elif args.action == 'encode':
        result = export(catalogue, args.message, args.seed)
        result.pop('specification')
    elif args.action == 'audit-touch':
        result = audit_foreleg_connectivity(catalogue)
    else:
        result = verify(catalogue)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
