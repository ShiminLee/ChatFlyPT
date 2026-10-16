"""Blind readout for the actual cVA and pulse-envelope input protocol."""
import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from .brain import Brain, simulate
from .experiment import Decoder
from .social import OUT, build_catalogue, encode, to_engine_tape
from .stimuli import ANNOTATIONS
from .paths import ROOT, dump, sha256

LABELS = ['cva', 'courtship_pulse']
MODEL = OUT / 'social-decoder.npz'
REPORT = ROOT / 'reports/full-pathway'


class SocialDecoder(Decoder):
    def __init__(self, path=None):
        super().__init__(path or MODEL)

    def decode(self, counts):
        scores = self.scores(counts)
        if scores.ndim != 1:
            raise ValueError('One neural count vector required')
        return {'words': [w for w, s, t in zip(LABELS, scores, self.thresholds) if s >= t],
                'scores': dict(zip(LABELS, map(float, scores))),
                'score_type': 'uncalibrated linear scores; not probabilities'}


def train():
    table = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False).fillna('')
    table.root_id = table.root_id.astype(str)
    brain = Brain()
    rows = table.set_index('root_id').reindex(brain.ids.astype(str))
    allowed = ~rows.super_class.isin(['sensory', 'sensory_ascending']).to_numpy()
    catalogue = build_catalogue()
    for i, name in enumerate(LABELS):
        direct = [brain.lookup[str(x)] for x in catalogue['channels'][name]['neuron_ids']]
        allowed[direct] = False
    rng = np.random.default_rng(150000)
    cases = []
    for split, repeats in [('train', 16), ('validation', 6), ('test', 10)]:
        for channel in [None] + list(LABELS):
            for j in range(repeats):
                scale = (0.75, 1.0, 1.25)[j % 3] if split == 'test' else float(rng.uniform(.85, 1.15))
                cases.append({'split': split, 'channel': channel, 'seed': 150000 + len(cases), 'scale': scale})
    dump(REPORT / 'social-readout-protocol.json',
         {'cases': cases, 'duration_ms': 500, 'labels': LABELS,
          'train_test_seed_overlap': bool(set(c['seed'] for c in cases if c['split'] == 'train') &
                                          set(c['seed'] for c in cases if c['split'] == 'test')),
          'input_cells_excluded': True})

    def run(c):
        if c['channel'] is None:
            return brain.run([], 500, c['seed']).counts
        spec = copy.deepcopy(catalogue['channels'][c['channel']])
        spec['parameters']['peak_hz'] *= c['scale']
        events = encode(spec, c['seed'], 500)
        tape = to_engine_tape(brain, events)
        return simulate(brain.indptr, brain.indices, brain.weights, *tape, False)[0]

    def metrics(expected, scores, thresholds):
        predicted = scores >= thresholds
        return {'exact_set_accuracy': float(np.mean(np.all(expected == predicted, axis=1))),
                'micro_f1': float(f1_score(expected, predicted, average='micro', zero_division=0))}

    matrices = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, counts in enumerate(pool.map(run, cases), 1):
            matrices.append(sparse.csr_matrix(counts[None, :]))
            if i % 16 == 0 or i == len(cases):
                print('Social readout ' + str(i) + '/' + str(len(cases)), flush=True)
    x = sparse.vstack(matrices, format='csr')
    sparse.save_npz(OUT / 'social-counts.npz', x)
    y = np.array([[int(c['channel'] == w) for w in LABELS] for c in cases])
    tr = np.array([c['split'] == 'train' for c in cases])
    validation = np.array([c['split'] == 'validation' for c in cases])
    test = np.array([c['split'] == 'test' for c in cases])
    indices = np.flatnonzero(allowed)
    indices = indices[np.asarray(x[tr][:, indices].getnnz(axis=0)).ravel() > 0]
    features = x[:, indices].astype(float)
    features.data = np.sqrt(features.data)
    with threadpool_limits(limits=2):
        model = Ridge(alpha=10, solver='lsqr', tol=1e-6).fit(features[tr], y[tr])
        t = model.predict(features)
    grid = np.linspace(.15, .85, 71)
    best = []
    for j in range(len(LABELS)):
        quality = [balanced_accuracy_score(y[validation, j], t[validation, j] >= g) for g in grid]
        top = max(quality)
        best.append(min([g for g, q in zip(grid, quality) if q == top], key=lambda x: abs(x - .5)))
    np.savez_compressed(MODEL, indices=indices, weights=model.coef_, intercept=model.intercept_,
                        thresholds=best)
    s = t[test]
    report = {'dataset': 'FlyWire v783', 'labels': LABELS,
              'trials': len(cases), 'test_trials': int(test.sum()),
              'features': len(indices), 'model_sha256': sha256(MODEL),
              'input_and_all_sensory_cells_excluded': bool(allowed[indices].all()),
              'test': metrics(y[test], s, np.array(best)),
              'controls': {'shuffled_observations': metrics(
                               y[test], np.random.default_rng(15).permutation(s), np.array(best)),
                           'zero_observations': metrics(
                               y[test], np.tile(model.intercept_, (int(test.sum()), 1)), np.array(best))},
              'test_predictions': [dict(c, predicted=[w for w, sv, th in zip(LABELS, t[k], best) if sv >= th])
                                   for k, c in enumerate(cases) if c['split'] == 'test'],
              'meaning': 'Predicted intervention labels from internal spikes; not semantic understanding or observed behavior.'}
    dump(REPORT / 'social-readout.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train'])
    parser.parse_args()
    train()
