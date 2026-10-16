"""Fit the two extra blind slots (delivery pattern, internal activity level).

Preregistration first: band thresholds and qualification gates are written to
`reports/slot-readout/protocol.json` before any validation or test window is
labelled. Ground truth is measured, never intended:

* pattern  - fraction of 10 ms samples in which a receptor event was *actually
             delivered* (from the recorded feedback samples).
* activity - non-sensory, non-input spikes in the window (blind-mask sum).

Both are then decoded from the same blind internal features used by the other
readouts; a slot that fails qualification abstains.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from chatflypt.paths import ROOT, dump, sha256
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.malecns import MaleCNS
from chatflypt.unified_readout import blind_mask, internal_features, model_contract
from chatflypt.slot_readout import (ACTIVITY_LABELS, ACTIVITY_QUANTILES, OUT, PATTERN_LABELS,
                                    SCHEMA, SlotDecoder, activity_labels, pattern_labels, slot_rules)

REPORT = ROOT / 'reports/slot-readout'
CORPUS = ROOT / 'data/derived/unified-readout/runs'
QUALIFICATION = {'minimum_positive_windows': 2, 'minimum_negative_windows': 2,
                 'precision': .9, 'recall': .75}
THRESHOLDS = np.arange(.2, .8 + 1e-9, .01)
_CATALOGUE = None


def protocol(run_id):
    return {'schema': 'chatflypt.slot-readout-protocol.v1', 'run_id': run_id,
            'features': 'sqrt spikes; log1p positive/negative voltage area; normalized to 500 ms',
            'feature_rule': 'All annotated sensory classes and every registered input cell excluded',
            'labels': list(PATTERN_LABELS) + list(ACTIVITY_LABELS),
            'ground_truth': {'pattern': 'fraction of sampled windows with a delivered receptor event',
                             'activity': 'non-sensory non-input spikes in the window'},
            'pattern_transient_max_fraction': .3,
            'activity_quantiles': list(ACTIVITY_QUANTILES),
            'thresholds_from': 'Training split quantiles only, before validation/test labelling',
            'qualification': QUALIFICATION, 'threshold_grid': [.2, .8, .01],
            'candidate_only': True, 'biologically_calibrated': False,
            'explicit_non_claims': [
                'Pattern and activity are measured properties of the receptor events and of internal activity, not emotions or intentions.',
                'A slot that fails qualification abstains; nothing is spoken that was not decoded.',
                'The renderer that consumes these slots may vary phrasing but may not add slots (see generative_renderer.audit_grounding).']}


def corpus_rows(folder):
    rows = []
    for episode in sorted(folder.glob('*')):
        complete = episode / 'complete.json'
        if not complete.is_file():
            continue
        rows.extend(json.loads(complete.read_text(encoding='utf8'))['rows'])
    return rows


def main():
    global _CATALOGUE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--protocol', type=Path,
                        help='Preregistration file (default reports/slot-readout/protocol.json)')
    args = parser.parse_args()
    preregistration = args.protocol or REPORT / 'protocol.json'
    if not preregistration.is_file():
        dump(preregistration, protocol(args.run_id))
        print('preregistered', preregistration, sha256(preregistration))
    plan = json.loads(preregistration.read_text(encoding='utf8'))
    if plan['run_id'] != args.run_id:
        raise SystemExit(f"Preregistration names run {plan['run_id']}; refusing to relabel it as {args.run_id}")

    brain = MaleCNS()
    _CATALOGUE = build_catalogue(save=False)
    allowed = blind_mask(brain, _CATALOGUE)
    rows = corpus_rows(CORPUS / plan['run_id'] / 'trials')
    if not rows:
        raise SystemExit('No corpus rows')
    features, delivery, spikes, splits = [], [], [], []
    for row in rows:
        with np.load(row['record'], allow_pickle=False) as saved:
            counts = saved['counts']; state = saved['state_record']; duration = float(saved['duration_ms'])
        feedback = json.loads(Path(row['feedback']).read_text(encoding='utf8'))
        samples = feedback.get('samples', [])
        delivered = sum(1 for sample in samples
                        if any(item.get('intensity', 1.) > 0 for item in sample.get('items', [])))
        features.append(internal_features(counts, state, duration))
        delivery.append(delivered / max(1, len(samples)))
        spikes.append(float(np.asarray(counts)[allowed].sum()))
        splits.append(row['split'])
    x = np.asarray(features)
    splits = np.asarray(splits)
    training, validation, test = splits == 'train', splits == 'validation', splits == 'test'
    train_windows = [{'delivery_fraction': d, 'nonsensory_spikes': s}
                     for d, s in zip(delivery, spikes) if splits[np.flatnonzero(splits == 'train')[0]] == 'train']
    train_windows = [{'delivery_fraction': d, 'nonsensory_spikes': s}
                     for d, s, is_train in zip(delivery, spikes, training) if is_train]
    rules = slot_rules(train_windows)
    print('window counts train/validation/test:', int(training.sum()), int(validation.sum()), int(test.sum()))
    print('rules:', json.dumps(rules, ensure_ascii=False))

    truth = {'pattern': [], 'activity': []}
    for fraction, spike_count in zip(delivery, spikes):
        truth['pattern'].append(pattern_labels(fraction, rules))
        truth['activity'].append(activity_labels(spike_count, rules))

    active = np.any(x[training][:, :x.shape[1] // 3] > 0, axis=0)
    indices = np.flatnonzero(allowed)[active[np.flatnonzero(allowed)]]
    feature_indices = np.concatenate([indices, indices + len(brain.ids), indices + 2 * len(brain.ids)])
    print('internal features selected:', len(indices))

    def fit_family(labels, family, key_collection):
        y = np.asarray([[1 if label in words else 0 for label in labels] for words in truth[family]], dtype=bool)
        with threadpool_limits(limits=2):
            model = Ridge(alpha=10, solver='cholesky').fit(x[training][:, feature_indices], y[training])
            scores = model.predict(x[:, feature_indices])
        thresholds, qualified = [], []
        for index, label in enumerate(labels):
            target = y[validation, index].astype(bool)
            supported = (target.sum() >= QUALIFICATION['minimum_positive_windows'] and
                         (~target).sum() >= QUALIFICATION['minimum_negative_windows'] and
                         y[training, index].any() and (~y[training, index]).any())
            chosen, ok = .5, False
            if supported:
                best = None
                for threshold in THRESHOLDS:
                    predicted = scores[validation, index] >= threshold
                    tp = int((predicted & target).sum()); fp = int((predicted & ~target).sum())
                    fn = int((~predicted & target).sum())
                    if (tp / max(1, tp + fp) >= QUALIFICATION['precision'] and
                            tp / max(1, tp + fn) >= QUALIFICATION['recall']):
                        if best is None or threshold < best:
                            best = threshold
                if best is not None:
                    chosen, ok = float(best), True
            thresholds.append(chosen); qualified.append(ok)
            print(f'  {label:20s} validation positives={int(target.sum()):3d} qualified={ok} threshold={chosen}')
        return model, scores, labels, thresholds, qualified

    pattern_model, pattern_scores, pattern_labels_, pattern_thresholds, pattern_qualified = fit_family(
        PATTERN_LABELS, 'pattern', None)
    activity_model, activity_scores, activity_labels_, activity_thresholds, activity_qualified = fit_family(
        ACTIVITY_LABELS, 'activity', None)

    candidate = OUT / 'malecns-slots.candidate.npz'
    candidate.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(candidate, indices=indices,
                        pattern_feature_indices=feature_indices, activity_feature_indices=feature_indices,
                        pattern_weights=pattern_model.coef_, pattern_intercept=pattern_model.intercept_,
                        pattern_thresholds=np.asarray(pattern_thresholds),
                        pattern_qualified=np.asarray(pattern_qualified),
                        activity_weights=activity_model.coef_, activity_intercept=activity_model.intercept_,
                        activity_thresholds=np.asarray(activity_thresholds),
                        activity_qualified=np.asarray(activity_qualified),
                        rules=json.dumps(rules),
                        metadata=json.dumps({'schema': SCHEMA, 'contract': model_contract(brain, _CATALOGUE),
                            'protocol_sha256': sha256(preregistration),
                            'tested_durations_ms': sorted({500.}),
                            'label_source': plan['ground_truth']}))
    frozen = sha256(candidate)
    decoder = SlotDecoder(brain, _CATALOGUE, candidate)

    def metrics(family, labels):
        y = np.asarray([[1 if label in words else 0 for label in labels] for words in truth[family]], dtype=bool)
        scores = pattern_scores if family == 'pattern' else activity_scores
        qualified = np.asarray(pattern_qualified if family == 'pattern' else activity_qualified)
        thresholds = np.asarray(pattern_thresholds if family == 'pattern' else activity_thresholds)
        out = {}
        for split, mask in (('validation', validation), ('test', test)):
            predicted = (scores[mask] >= thresholds) & qualified
            target = y[mask]
            out[split] = {'windows': int(mask.sum()),
                          'exact_set_accuracy': float(np.all(predicted == target, axis=1).mean()),
                          'micro_f1': float(f1_score(target, predicted, average='micro', zero_division=0)),
                          'per_label': {label: {'positives': int(target[:, i].sum()),
                                                'predicted': int(predicted[:, i].sum()),
                                                'matched': int((predicted[:, i] & target[:, i]).sum())}
                                        for i, label in enumerate(labels)}}
        return out

    report = {'schema': 'chatflypt.slot-readout-evaluation.v1', 'run_id': plan['run_id'],
              'artifact': str(candidate.relative_to(ROOT)), 'artifact_sha256': frozen,
              'protocol_sha256': sha256(preregistration), 'rules': rules, 'features': int(len(indices)),
              'qualified': {'pattern': [l for l, q in zip(PATTERN_LABELS, pattern_qualified) if q],
                            'activity': [l for l, q in zip(ACTIVITY_LABELS, activity_qualified) if q]},
              'unqualified': {'pattern': [l for l, q in zip(PATTERN_LABELS, pattern_qualified) if not q],
                              'activity': [l for l, q in zip(ACTIVITY_LABELS, activity_qualified) if not q]},
              'metrics': {'pattern': metrics('pattern', PATTERN_LABELS),
                          'activity': metrics('activity', ACTIVITY_LABELS)},
              'candidate_only': True, 'biologically_calibrated': False,
              'limitations': plan['explicit_non_claims']}
    dump(REPORT / 'evaluation.json', report)
    print(json.dumps({'artifact': frozen, 'qualified': report['qualified']}, ensure_ascii=False))
    if args.publish:
        target = OUT / 'malecns-slots.npz'
        target.write_bytes(candidate.read_bytes())
        print('published', target, sha256(target))


if __name__ == '__main__':
    main()
