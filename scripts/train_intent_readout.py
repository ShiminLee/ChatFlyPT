"""Fit the blind motor-side (intent) readout from an already-collected corpus.

Two preregistered modes:

``labels`` (first revision, kept for reproducibility)
    Discrete labels from measured outcomes with training-quantile thresholds and
    per-label precision/recall gates. On conspecific-v4 no label qualified.

``continuous`` (second revision, shape borrowed from the vendored flyvis decoder)
    Regress the measured self-motion vector (yaw change, path length) from the
    same blind internal features, then discretize with fixed rules. Every window
    carries a value, so qualification no longer depends on a handful of rare
    classes. `flyvis/task/decoder.py` regresses a continuous task vector with a
    vector objective (MIT); the *shape* is borrowed, no flyvis code or weights
    are copied.

Rules are written to `reports/intent-readout/protocol.json` before any window is
labelled or scored, and are never tuned on validation or test. The artifact is
written as a candidate and is not published automatically.
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
from chatflypt.intent_readout import (ACTIVITY_QUANTILE, CONTINUOUS_SCHEMA, DATASET, FEATURES,
                                      LABELS, LABEL_MEANING, MOVE_PRECISION, MOVE_R_MIN, MOVE_RECALL,
                                      OUT, SCHEMA, TARGETS, TURN_QUANTILE, TURN_R_MIN, TURN_SIGN_MIN,
                                      ContinuousIntentDecoder, labels_to_vector, measured_outcome,
                                      outcome_labels, outcome_rules, standardize)

REPORT = ROOT / 'reports/intent-readout'
CORPUS = ROOT / 'data/derived/unified-readout/runs'
LABEL_QUALIFICATION = {'minimum_positive_windows': 2, 'minimum_negative_windows': 2,
                       'precision': .9, 'recall': .75}
THRESHOLD_GRID = [.2, .8, .01]
THRESHOLDS = np.arange(THRESHOLD_GRID[0], THRESHOLD_GRID[1] + 1e-9, THRESHOLD_GRID[2])
_CATALOGUE = None


def protocol(run_id, mode):
    base = {'schema': 'chatflypt.intent-readout-protocol.v1', 'run_id': run_id, 'mode': mode,
            'dataset': DATASET, 'input_mode': 'environment_only', 'duration_ms': 500.,
            'split_unit': 'whole_session',
            'label_source': 'Measured body outcome in the same 500 ms window: yaw change and path length',
            'features': FEATURES,
            'feature_rule': 'All annotated sensory classes and every registered input cell excluded; '
                            'training-active internal cells only',
            'readout': 'sqrt spike counts and log1p voltage areas -> Ridge alpha=10',
            'thresholds_from': 'Training split quantiles only, before any validation or test window is scored',
            'turn_quantile_of_abs_yaw': TURN_QUANTILE, 'activity_quantile_of_path': ACTIVITY_QUANTILE,
            'candidate_only': True, 'biologically_calibrated': False,
            'revision_history': [
                'v1 (fixed 5 deg / 0.5 mm discrete rules): zero qualified labels on retinal-axes-v3; right turns absent from validation and "forward" covered ~80% of windows.',
                'v2 (discrete labels, training-quantile thresholds): move:active/move:still qualified on retinal-axes-v3, nothing qualified on conspecific-v4.',
                'v3 (continuous vector regression, this mode): rules fixed before the conspecific-v5 corpus exists.'],
            'explicit_non_claims': [
                'The body is driven by recorded neural activity through a pinned engineering controller; part of what is learned is that controller map.',
                'A label is a measured movement, not a validated internal intention or a decoded command.',
                'No window is labelled from the human sentence, the scene, the planned stimulus or the recorded motor commands.']}
    if mode == 'labels':
        base.update({'labels': list(LABELS), 'label_meaning': LABEL_MEANING,
                     'qualification': LABEL_QUALIFICATION, 'threshold_grid': THRESHOLD_GRID})
    else:
        base.update({'targets': list(TARGETS),
                     'rules': {'turn_r_min': TURN_R_MIN, 'turn_sign_min': TURN_SIGN_MIN,
                               'move_r_min': MOVE_R_MIN, 'move_precision': MOVE_PRECISION,
                               'move_recall': MOVE_RECALL},
                     'qualification': {
                         'turn': 'r(predicted, measured yaw) >= turn_r_min over validation AND sign agreement '
                                 '>= turn_sign_min over validation windows whose |measured yaw| exceeds the '
                                 'training-split turn quantile',
                         'move': 'r(predicted, measured path) >= move_r_min AND a threshold at the training '
                                 'median of path reaches >= move_precision and >= move_recall on validation'},
                     'speech': 'Only qualified components are spoken; the other abstains.'})
    return base


def corpus_rows(folder):
    rows = []
    for episode_dir in sorted(folder.glob('*')):
        complete = episode_dir / 'complete.json'
        if not complete.is_file():
            continue
        for row in json.loads(complete.read_text(encoding='utf8'))['rows']:
            rows.append(row)
    return rows


def read_corpus(plan):
    folder = CORPUS / plan['run_id'] / 'trials'
    rows = corpus_rows(folder)
    if not rows:
        raise SystemExit('No complete corpus rows in ' + str(folder))
    features, outcomes, splits, records = [], [], [], []
    for row in rows:
        with np.load(row['record'], allow_pickle=False) as saved:
            counts, state, duration = saved['counts'], saved['state_record'], float(saved['duration_ms'])
        body = json.loads(Path(row['body']).read_text(encoding='utf8'))
        outcome = measured_outcome(body['trajectory'])
        features.append(internal_features(counts, state, duration))
        outcomes.append(outcome)
        splits.append(row['split'])
        records.append({'episode': row['episode'], 'turn': row['turn'], 'split': row['split'],
                        'session_id': row['session_id'], 'record': row['record'],
                        'record_sha256': row['record_sha256'], 'body': row['body'],
                        'outcome': outcome})
    return np.asarray(features), outcomes, np.asarray(splits), records


def select_features(x, training, brain):
    """Internal, allowed, training-active cells only (same rule in both modes)."""
    active = np.any(x[training][:, :x.shape[1] // 3] > 0, axis=0)
    allowed = blind_mask(brain, _CATALOGUE)
    indices = np.flatnonzero(allowed)[active[np.flatnonzero(allowed)]]
    if not len(indices):
        raise SystemExit('No internal training response: cannot fit a motor-side readout')
    return indices, np.concatenate([indices, indices + len(brain.ids), indices + 2 * len(brain.ids)])


def fit_continuous(plan, preregistration, brain, x, outcomes, splits, records):
    training, validation, test = splits == 'train', splits == 'validation', splits == 'test'
    yaw = np.array([row['yaw_change_deg'] for row in outcomes])
    path = np.array([row['path_length_mm'] for row in outcomes])
    targets = np.stack([yaw, path], axis=1)
    mean = targets[training].mean(axis=0)
    scale = targets[training].std(axis=0, ddof=0)
    if np.any(scale <= 0):
        raise SystemExit('Degenerate target scale in the training split')
    y = standardize(targets, mean, scale)
    indices, feature_indices = select_features(x, training, brain)
    print('windows train/validation/test:', int(training.sum()), int(validation.sum()), int(test.sum()),
          '| internal features:', len(indices))
    with threadpool_limits(limits=2):
        model = Ridge(alpha=10, solver='cholesky').fit(x[training][:, feature_indices], y[training])
        predicted = model.predict(x[:, feature_indices]) * scale + mean
    rules = outcome_rules([row for row, is_train in zip(outcomes, training) if is_train])

    def correlation(mask):
        return {'yaw': float(np.corrcoef(predicted[mask, 0], yaw[mask])[0, 1]) if mask.sum() > 2 else None,
                'path': float(np.corrcoef(predicted[mask, 1], path[mask])[0, 1]) if mask.sum() > 2 else None}

    def sign_accuracy(mask):
        turning = np.abs(yaw[mask]) >= rules['turn_deg']
        if turning.sum() == 0:
            return {'windows': 0, 'accuracy': None}
        agree = np.sign(predicted[mask][turning, 0]) == np.sign(yaw[mask][turning])
        return {'windows': int(turning.sum()), 'accuracy': float(agree.mean())}

    def move_gate(mask):
        predicted_move = predicted[mask, 1] >= rules['move_mm']
        measured_move = path[mask] >= rules['move_mm']
        tp = int((predicted_move & measured_move).sum()); fp = int((predicted_move & ~measured_move).sum())
        fn = int((~predicted_move & measured_move).sum())
        return {'precision': tp / max(1, tp + fp), 'recall': tp / max(1, tp + fn),
                'predicted_active': int(predicted_move.sum()), 'measured_active': int(measured_move.sum())}

    validation_correlation = correlation(validation)
    validation_sign = sign_accuracy(validation)
    validation_move = move_gate(validation)
    turn_ok = (validation_correlation['yaw'] is not None and validation_correlation['yaw'] >= TURN_R_MIN and
               validation_sign['accuracy'] is not None and validation_sign['accuracy'] >= TURN_SIGN_MIN)
    move_ok = (validation_correlation['path'] is not None and validation_correlation['path'] >= MOVE_R_MIN and
               validation_move['precision'] >= MOVE_PRECISION and validation_move['recall'] >= MOVE_RECALL)
    qualified = np.array([turn_ok, move_ok])
    candidate = OUT / 'malecns-intent.candidate.npz'
    candidate.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(candidate, feature_indices=feature_indices, indices=indices,
                        weights=model.coef_, intercept=model.intercept_,
                        target_mean=mean, target_scale=scale, qualified=qualified,
                        rules=json.dumps(rules),
                        metadata=json.dumps({'schema': CONTINUOUS_SCHEMA, 'mode': 'continuous',
                            'contract': model_contract(brain, _CATALOGUE),
                            'input_mode': 'environment_only', 'split_unit': 'whole_session',
                            'protocol_sha256': sha256(preregistration),
                            'tested_durations_ms': sorted({500.}), 'features': FEATURES,
                            'label_source': plan['label_source']}))
    frozen = sha256(candidate)
    # Served-path check: recompute three real test windows through the decoder's
    # own predict(), exactly as the application does, and compare with the fitted
    # matrix. This exercises loading, the stored contract and the feature layout.
    decoder = ContinuousIntentDecoder(brain, _CATALOGUE, candidate)
    test_indices = [i for i, row in enumerate(records) if splits[i] == 'test'][:3]
    for index in test_indices:
        with np.load(records[index]['record'], allow_pickle=False) as saved:
            counts = saved['counts']; state = saved['state_record']; duration = float(saved['duration_ms'])
        served = decoder.predict(counts, state, duration)
        np.testing.assert_allclose(served, predicted[index], atol=1e-6)

    def discretize(mask):
        words = {'turn:left': 0, 'turn:right': 0, 'move:active': 0, 'move:still': 0}
        for index in np.flatnonzero(mask):
            if turn_ok:
                if predicted[index, 0] >= rules['turn_deg']:
                    words['turn:left'] += 1
                elif predicted[index, 0] <= -rules['turn_deg']:
                    words['turn:right'] += 1
            if move_ok:
                words['move:active' if predicted[index, 1] >= rules['move_mm'] else 'move:still'] += 1
        return words

    shuffled = predicted.copy()
    shuffled[test] = predicted[np.random.default_rng(4242).permutation(np.flatnonzero(test))]
    report = {'schema': 'chatflypt.intent-readout-evaluation.v1', 'run_id': plan['run_id'],
              'mode': 'continuous', 'protocol_sha256': sha256(preregistration),
              'artifact': str(candidate.relative_to(ROOT)), 'artifact_sha256': frozen,
              'windows': {'train': int(training.sum()), 'validation': int(validation.sum()), 'test': int(test.sum())},
              'features': int(len(indices)), 'outcome_rules': rules,
              'qualified_targets': [name for name, ok in zip(TARGETS, qualified) if ok],
              'unqualified_targets': [name for name, ok in zip(TARGETS, qualified) if not ok],
              'validation': {'correlation': validation_correlation, 'turn_sign': validation_sign,
                             'move_gate': validation_move},
              'test': {'correlation': correlation(test), 'turn_sign': sign_accuracy(test), 'move_gate': move_gate(test)},
              'test_discretized_words': discretize(test),
              'controls': {'shuffled_test_correlation': {
                  'yaw': float(np.corrcoef(shuffled[test, 0], yaw[test])[0, 1]),
                  'path': float(np.corrcoef(shuffled[test, 1], path[test])[0, 1])}},
              'served_matches_fitted_matrix': True,
              'candidate_only': True, 'biologically_calibrated': False,
              'limitations': plan['explicit_non_claims'] + [
                  'Small-sample environment-only corpus; windows within a session are correlated.',
                  'Ground walking only; no free-flight aerodynamics.']}
    dump(REPORT / 'evaluation.json', report)
    dump(REPORT / 'windows.json',
         [dict(record, prediction={'yaw_change_deg': float(predicted[i, 0]), 'path_length_mm': float(predicted[i, 1])})
          for i, record in enumerate(records)])
    print(json.dumps({'artifact': frozen, 'qualified': report['qualified_targets'],
                      'validation': report['validation'], 'test': report['test'],
                      'test_words': report['test_discretized_words']}, ensure_ascii=False))
    return candidate


def fit_labels(plan, preregistration, brain, x, outcomes, splits, records):
    """First-revision discrete mode, kept so earlier results stay reproducible."""
    training, validation, test = splits == 'train', splits == 'validation', splits == 'test'
    rules = outcome_rules([row for row, is_train in zip(outcomes, training) if is_train])
    for record in records:
        record['labels'] = outcome_labels(record['outcome'], rules)
    y = np.asarray([labels_to_vector(record['labels']) for record in records], dtype=bool)
    indices, feature_indices = select_features(x, training, brain)
    with threadpool_limits(limits=2):
        model = Ridge(alpha=10, solver='cholesky').fit(x[training][:, feature_indices], y[training])
        scores = model.predict(x[:, feature_indices])
    thresholds, qualified = [], []
    for index in range(len(LABELS)):
        target = y[validation, index].astype(bool)
        supported = (target.sum() >= LABEL_QUALIFICATION['minimum_positive_windows'] and
                     (~target).sum() >= LABEL_QUALIFICATION['minimum_negative_windows'] and
                     y[training, index].any() and (~y[training, index]).any())
        chosen, ok = .5, False
        if supported:
            best = None
            for threshold in THRESHOLDS:
                predicted = scores[validation, index] >= threshold
                tp = int((predicted & target).sum()); fp = int((predicted & ~target).sum())
                fn = int((~predicted & target).sum())
                if (tp / max(1, tp + fp) >= LABEL_QUALIFICATION['precision'] and
                        tp / max(1, tp + fn) >= LABEL_QUALIFICATION['recall']):
                    if best is None or threshold < best:
                        best = threshold
            if best is not None:
                chosen, ok = float(best), True
        thresholds.append(chosen); qualified.append(ok)
    candidate = OUT / 'malecns-intent.candidate.npz'
    candidate.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(candidate, feature_indices=feature_indices, indices=indices,
                        weights=model.coef_, intercept=model.intercept_,
                        thresholds=np.asarray(thresholds), qualified=np.asarray(qualified),
                        labels=np.asarray(LABELS), names=np.asarray([LABEL_MEANING[w] for w in LABELS]),
                        metadata=json.dumps({'schema': SCHEMA, 'mode': 'labels',
                            'contract': model_contract(brain, _CATALOGUE),
                            'input_mode': 'environment_only', 'split_unit': 'whole_session',
                            'protocol_sha256': sha256(preregistration),
                            'tested_durations_ms': sorted({500.}), 'features': FEATURES,
                            'label_source': plan['label_source']}))
    frozen = sha256(candidate)

    def metrics(mask):
        predicted = (scores[mask] >= np.asarray(thresholds)) & np.asarray(qualified)
        target = y[mask]
        return {'windows': int(mask.sum()),
                'exact_set_accuracy': float(np.all(predicted == target, axis=1).mean()),
                'micro_f1': float(f1_score(target, predicted, average='micro', zero_division=0)),
                'per_label': {label: {'positives': int(target[:, i].sum()),
                                      'predicted': int(predicted[:, i].sum()),
                                      'matched': int((predicted[:, i] & target[:, i]).sum())}
                              for i, label in enumerate(LABELS)}}

    report = {'schema': 'chatflypt.intent-readout-evaluation.v1', 'run_id': plan['run_id'],
              'mode': 'labels', 'protocol_sha256': sha256(preregistration),
              'artifact': str(candidate.relative_to(ROOT)), 'artifact_sha256': frozen,
              'qualified_labels': [w for w, q in zip(LABELS, qualified) if q],
              'unqualified_labels': [w for w, q in zip(LABELS, qualified) if not q],
              'thresholds': dict(zip(LABELS, map(float, thresholds))), 'outcome_rules': rules,
              'validation': metrics(validation), 'test': metrics(test),
              'candidate_only': True, 'biologically_calibrated': False,
              'limitations': plan['explicit_non_claims']}
    dump(REPORT / 'evaluation.json', report)
    dump(REPORT / 'windows.json', records)
    print(json.dumps({'artifact': frozen, 'qualified': report['qualified_labels'],
                      'test': {k: v for k, v in report['test'].items() if k != 'per_label'}}, ensure_ascii=False))
    return candidate


def main():
    global _CATALOGUE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--mode', choices=('continuous', 'labels'), default='continuous')
    parser.add_argument('--publish', action='store_true',
                        help='Copy the frozen candidate to the live path (manual step, never automatic)')
    parser.add_argument('--protocol', type=Path,
                        help='Preregistration file (default reports/intent-readout/protocol.json)')
    args = parser.parse_args()

    preregistration = args.protocol or REPORT / 'protocol.json'
    if not preregistration.is_file():
        dump(preregistration, protocol(args.run_id, args.mode))
        print('preregistered', preregistration, 'sha256', sha256(preregistration))
    plan = json.loads(preregistration.read_text(encoding='utf8'))
    if plan['run_id'] != args.run_id:
        raise SystemExit(f"Preregistration names run {plan['run_id']}; refusing to relabel it as {args.run_id}")
    if plan.get('mode', 'labels') != args.mode:
        raise SystemExit(f"Preregistration is for mode {plan.get('mode')}; refusing to switch to {args.mode}")

    brain = MaleCNS()
    _CATALOGUE = build_catalogue(save=False)
    x, outcomes, splits, records = read_corpus(plan)
    print('corpus windows:', len(records))
    fitter = fit_continuous if args.mode == 'continuous' else fit_labels
    candidate = fitter(plan, preregistration, brain, x, outcomes, splits, records)
    if args.publish:
        target = OUT / 'malecns-intent.npz'
        target.write_bytes(candidate.read_bytes())
        print('published', target, sha256(target))


if __name__ == '__main__':
    main()
