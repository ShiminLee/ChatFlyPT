"""Blind linear readout of foreleg-touch simulations in the MaleCNS graph.

Targets identify the experimental stimulus, not an unobserved subjective state or
turn direction. Input labels/metadata never enter TouchDecoder.decode.
"""
import argparse
import copy
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from .brain import DT_MS, simulate
from .malecns import MaleCNS, OUT, REPORT
from .paths import ROOT, dump, sha256
from .social import OUT as SOCIAL_OUT, encode, to_engine_tape

WORDS = ['right_foreleg_touch', 'left_foreleg_touch']
CONDITIONS = {'right': [WORDS[0]], 'left': [WORDS[1]], 'both': WORDS, 'blank': []}


def protocol():
    """Choose all seeds and strengths before any simulation or evaluation."""
    rng = np.random.default_rng(2026091902)
    cases = []
    for split, repeats, base in [('train', 16, 130000), ('validation', 6, 140000),
                                  ('test', 10, 150000), ('ood', 8, 160000)]:
        for condition, words in CONDITIONS.items():
            for repeat in range(repeats):
                scale = ([.6, 1.4][repeat % 2] if split == 'ood'
                         else float(rng.uniform(.85, 1.15)))
                cases.append({'id': f'{split}-{condition}-{repeat:02d}', 'split': split,
                              'condition': condition, 'words': list(words),
                              'seed': base + len(cases), 'scale': scale, 'duration_ms': 500.})
    return {'name': 'malecns-touch-protocol-v1', 'cases': cases,
            'labels': WORDS, 'label_meaning': 'Known simulated sensory stimulus; no motor direction label',
            'split_rule': 'Disjoint random seeds; OOD strengths 0.6/1.4 outside training 0.85..1.15',
            'feature_rule': 'All annotated sensory classes and directly stimulated cells excluded; training-active cells only',
            'readout': 'sqrt internal spike counts -> Ridge alpha=10; thresholds from validation only',
            'duration_ms': 500., 'source': 'Independent implementation, not Chetaslua source code',
            'calibration': 'Transferred Shiu LIF parameters; candidate foreleg cells and uncalibrated input envelopes'}


def events_for(catalogue, words, seed, scale=1., duration_ms=500.):
    if len(set(words)) != len(words) or any(w not in WORDS for w in words):
        raise ValueError('Only unique left/right foreleg-touch labels are supported')
    if not np.isfinite(scale) or scale < 0:
        raise ValueError('Invalid stimulus scale')
    parts = []
    for word in words:
        spec = copy.deepcopy(catalogue['channels'][word])
        spec['parameters']['peak_hz'] *= scale
        # Fixed label-specific offset makes the two sensory populations independent.
        parts.append(encode(spec, seed + WORDS.index(word) * 1000000, duration_ms))
    def concatenate(key, dtype):
        return np.concatenate([p[key] for p in parts]).astype(dtype) if parts else np.empty(0, dtype)
    times = concatenate('event_steps', np.int32)
    origins = concatenate('event_neuron_ids', np.int64)
    order = np.argsort(times, kind='stable')
    return {'dataset': 'MaleCNS v1.0', 'id_namespace': 'malecns_body_id',
            'input_neuron_ids': concatenate('input_neuron_ids', np.int64),
            'event_neuron_ids': origins[order], 'event_steps': times[order],
            'dt_ms': DT_MS, 'duration_ms': duration_ms, 'seed': seed}


def readout_mask(brain, catalogue):
    mask = brain.nonsensory.copy()
    direct = [brain.lookup[i] for word in WORDS for i in catalogue['channels'][word]['neuron_ids']]
    mask[direct] = False
    return mask


class TouchDecoder:
    def __init__(self, path=None):
        with np.load(path or OUT / 'touch-decoder.npz', allow_pickle=False) as f:
            self.indices = f['indices']
            self.weights = f['weights']
            self.intercept = f['intercept']
            self.thresholds = f['thresholds']
            self.words = f['words'].tolist()
            self.neuron_count = int(f['neuron_count'])

    def scores(self, counts):
        if counts.shape[-1] != self.neuron_count:
            raise ValueError('Wrong count vector size for MaleCNS touch decoder')
        if sparse.issparse(counts):
            x = counts[:, self.indices].astype(float)
            if np.any(~np.isfinite(x.data)) or np.any(x.data < 0):
                raise ValueError('Counts must be finite and nonnegative')
            x.data = np.sqrt(x.data)
            return np.asarray(x @ self.weights.T) + self.intercept
        x = np.asarray(counts)[..., self.indices]
        if np.any(~np.isfinite(x)) or np.any(x < 0):
            raise ValueError('Counts must be finite and nonnegative')
        return np.sqrt(x) @ self.weights.T + self.intercept

    def decode(self, counts):
        values = self.scores(np.asarray(counts))
        if values.ndim != 1:
            raise ValueError('decode expects one neural count vector')
        return {'words': [w for w, v, t in zip(self.words, values, self.thresholds) if v >= t],
                'scores': dict(zip(self.words, map(float, values))),
                'score_type': 'uncalibrated linear scores; not probabilities',
                'label_meaning': 'Decoded simulated sensory stimulus, not a motor command or subjective state'}


def metrics(expected, scores, thresholds):
    predicted = scores >= thresholds
    return {'trials': len(expected),
            'exact_set_accuracy': float(np.mean(np.all(expected == predicted, axis=1))),
            'micro_f1': float(f1_score(expected, predicted, average='micro', zero_division=0)),
            'per_label_f1': dict(zip(WORDS, f1_score(expected, predicted, average=None, zero_division=0).tolist()))}


def activity_summary(brain, counts):
    groups = {}
    for name in sorted(set(brain.rows.superclass)):
        mask = brain.rows.superclass.eq(name).to_numpy()
        groups[name] = {'spikes': int(counts[mask].sum()),
                        'active_cells': int(np.count_nonzero(counts[mask]))}
    return {'total_spikes': int(counts.sum()),
            'nonsensory_spikes': int(counts[brain.nonsensory].sum()),
            'active_nonsensory_cells': int(np.count_nonzero(counts[brain.nonsensory])),
            'groups': groups}


def causal_ablation(brain, catalogue, decoder):
    """Delete the stimulated cells' outgoing synapses in the full graph."""
    result = []
    # Separate copy: training graph never changes.
    weights = brain.weights.copy()
    sources = [brain.lookup[i] for w in WORDS for i in catalogue['channels'][w]['neuron_ids']]
    deleted = 0
    for source in sources:
        start, end = brain.indptr[source:source + 2]
        deleted += int(end - start)
        weights[start:end] = 0
    for j, word in enumerate(WORDS):
        events = events_for(catalogue, [word], 170000 + j)
        intact = brain.run_events(events, record_spikes=False).counts
        disconnected = simulate(brain.indptr, brain.indices, weights,
                                *to_engine_tape(brain, events), False)[0]
        result.append({'word': word, 'seed': 170000 + j, 'deleted_outgoing_edges': deleted,
                       'input_events': len(events['event_steps']),
                       'intact': activity_summary(brain, intact),
                       'disconnected': activity_summary(brain, disconnected),
                       'intact_decoded': decoder.decode(intact),
                       'disconnected_decoded': decoder.decode(disconnected),
                       'passed': bool(intact[brain.nonsensory].sum() > 0 and
                                      disconnected[brain.nonsensory].sum() == 0)})
    return result


def train(workers=4):
    start = time.perf_counter()
    brain = MaleCNS()
    catalogue = json.loads((SOCIAL_OUT / 'catalogue.json').read_text(encoding='utf8'))
    plan = protocol()
    graph = json.loads((OUT / 'graph.json').read_text(encoding='utf8'))
    plan['graph_sha256'] = graph['graph_sha256']
    plan['channel_specs'] = {w: catalogue['channels'][w] for w in WORDS}
    plan['simulation_source_sha256'] = sha256(ROOT / 'chatflypt/brain.py')
    plan['encoder_source_sha256'] = sha256(ROOT / 'chatflypt/social.py')
    plan['fingerprint'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    dump(REPORT / 'touch-protocol.json', plan)
    cache = OUT / 'trials' / plan['fingerprint'][:16]
    cache.mkdir(parents=True, exist_ok=True)
    brain.run_events(events_for(catalogue, [], 0), record_spikes=False)

    def run(case):
        path = cache / (case['id'] + '.npz')
        if not path.exists():
            events = events_for(catalogue, case['words'], case['seed'], case['scale'], case['duration_ms'])
            trial = brain.run_events(events, record_spikes=False)
            np.savez_compressed(path, counts=trial.counts, input_events=len(events['event_steps']))
        return path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, c) for c in plan['cases']]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 16 == 0 or i == len(futures):
                print(f'Touch corpus {i}/{len(futures)} ({time.perf_counter()-start:.1f}s)', flush=True)
    matrices, records = [], []
    for case in plan['cases']:
        path = cache / (case['id'] + '.npz')
        with np.load(path, allow_pickle=False) as f:
            counts = f['counts']
            matrices.append(sparse.csr_matrix(counts[None, :]))
            records.append({**case, 'input_events': int(f['input_events']),
                            'raw_counts_file': str(path.relative_to(ROOT)),
                            **activity_summary(brain, counts)})
    x = sparse.vstack(matrices, format='csr')
    sparse.save_npz(OUT / 'touch-counts.npz', x)
    y = np.array([[int(w in c['words']) for w in WORDS] for c in plan['cases']])
    splits = np.array([c['split'] for c in plan['cases']])
    training, validation = splits == 'train', splits == 'validation'
    allowed = np.flatnonzero(readout_mask(brain, catalogue))
    indices = allowed[np.asarray(x[training][:, allowed].getnnz(axis=0)).ravel() > 0]
    if not len(indices):
        raise RuntimeError('No internal training response: cannot train a sensory-blind readout')
    features = x[:, indices].astype(float)
    features.data = np.sqrt(features.data)
    with threadpool_limits(limits=4):
        model = Ridge(alpha=10., solver='lsqr', tol=1e-6).fit(features[training], y[training])
        val_scores = model.predict(features[validation])
        thresholds = []
        for j in range(len(WORDS)):
            candidates = np.linspace(.15, .85, 71)
            quality = [balanced_accuracy_score(y[validation, j], val_scores[:, j] >= t) for t in candidates]
            best = max(quality)
            thresholds.append(min((t for t, q in zip(candidates, quality) if abs(q-best) < 1e-12),
                                  key=lambda t: abs(t-.5)))
    artifact = OUT / 'touch-decoder.npz'
    np.savez_compressed(artifact, indices=indices, weights=model.coef_, intercept=model.intercept_,
                        thresholds=thresholds, words=np.array(WORDS), neuron_count=len(brain.ids),
                        body_ids=brain.ids[indices], protocol_fingerprint=plan['fingerprint'])
    decoder = TouchDecoder(artifact)
    scores = decoder.scores(x)
    evaluation = {split: metrics(y[splits == split], scores[splits == split], decoder.thresholds)
                  for split in ['train', 'validation', 'test', 'ood']}
    test = splits == 'test'
    shuffled = np.random.default_rng(982).permutation(scores[test])
    zero = np.tile(decoder.scores(np.zeros(len(brain.ids))), (int(test.sum()), 1))
    for i, record in enumerate(records):
        record['predicted'] = [w for w, s, t in zip(WORDS, scores[i], decoder.thresholds) if s >= t]
        record['scores'] = dict(zip(WORDS, map(float, scores[i])))
    report = {'protocol_fingerprint': plan['fingerprint'], 'graph_sha256': plan['graph_sha256'],
              'artifact_sha256': sha256(artifact), 'source': plan['source'],
              'scope': 'Recover simulated left/right/bilateral touch labels from internal neural activity',
              'calibration': plan['calibration'], 'metrics': evaluation,
              'features': len(indices), 'excluded_sensory_cells': int((~brain.nonsensory).sum()),
              'excluded_direct_input_cells': sum(len(catalogue['channels'][w]['neuron_ids']) for w in WORDS),
              'feature_selection': plan['feature_rule'], 'thresholds': dict(zip(WORDS, thresholds)),
              'negative_controls': {
                  'shuffled_test_observations': metrics(y[test], shuffled, decoder.thresholds),
                  'zero_test_observations': metrics(y[test], zero, decoder.thresholds),
                  'full_graph_input_output_ablation': causal_ablation(brain, catalogue, decoder)},
              'raw_count_matrix': 'data/derived/malecns/touch-counts.npz',
              'count_matrix_neuron_ids': 'data/derived/malecns/graph.npz:ids',
              'trials': records, 'elapsed_seconds': time.perf_counter()-start,
              'limitations': ['Labels are stimulus identities; movement and subjective meaning are not decoded.',
                              'All trials reset the model to rest; no physiological calibration or learning in the connectome.',
                              'No random-wiring baseline or biological superiority claim is made.']}
    dump(REPORT / 'touch-readout.json', report)
    print(json.dumps({'features': len(indices), 'metrics': evaluation,
                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    train(args.workers)

