"""Fit the blind steering-command readout from an already-collected corpus.

Target: the steering command the pinned controller itself issued in the same
window, `mean(right - left)` over the bins that actually acted. Features: the
blind internal set (no sensory class, no registered input cell) **after removing
every mapped motor-group cell**, so the fit cannot restate the pinned formula.
The same fit on the motor groups alone is reported as a ceiling control.

Rules are written to `reports/command-readout/protocol.json` before any window is
scored and are never tuned on validation or test. The artifact is written as a
candidate and is not published automatically.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from chatflypt.paths import ROOT, dump, sha256
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.malecns import MaleCNS
from chatflypt.unified_readout import blind_mask, internal_features, model_contract
from chatflypt.behavior_readout import PreparedMotorAnnotations, build_behavior_drive
from chatflypt.command_readout import (COMMAND_R_MIN, COMMAND_SIGN_MIN, DATASET, FEATURES,
                                       LABELS, LABEL_MEANING, MIN_DIRECTION_SIGN,
                                       MIN_DIRECTION_WINDOWS, MIN_INFORMATIVE_WINDOWS, OUT,
                                       SCHEMA, TARGETS, ZERO_COMMAND_FP_MAX, command_labels,
                                       command_rules, command_target, labels_to_vector,
                                       motor_group_indices, turn_sign)
from chatflypt.intent_readout import standardize

REPORT = ROOT / 'reports/command-readout'
CORPUS = ROOT / 'data/derived/unified-readout/runs'
_CATALOGUE = None


def protocol(run_id):
    return {'schema': 'chatflypt.command-readout-protocol.v1', 'run_id': run_id,
            'dataset': DATASET, 'input_mode': 'environment_only', 'duration_ms': 500.,
            'split_unit': 'whole_session',
            'label_source': 'Recorded steering command of the same window: mean over the window\'s sampled '
                            'frames of (right - left) leg-stepping drive, read from the commands that acted '
                            'in that window\'s own physical run',
            'features': FEATURES,
            'feature_rule': 'Annotated sensory classes and registered input cells excluded by the shared '
                            'blind mask; every mapped motor-group cell additionally excluded; '
                            'training-active internal cells only',
            'readout': 'sqrt spike counts and log1p voltage areas -> Ridge alpha=10',
            'sample_weights': 'v3 revision: Ridge sample weights balanced across the three training classes '
                              '(informative left, informative right, quiet), each class contributing equal total '
                              'weight, computed from the training split only. Declared before the conspecific-v7 '
                              'fit because the v2 training split held 1 informative left window against 21 right, '
                              'so an unweighted least-squares fit predicts the majority sign and can never learn '
                              'the left direction. Features, thresholds, gates and the direction rule are unchanged.',
            'thresholds_from': 'Training split |command| quantile only, before any validation or test window is scored',
            'qualification': {'correlation': f'r(predicted, recorded command) >= {COMMAND_R_MIN} on validation',
                              'sign': f'sign agreement >= {COMMAND_SIGN_MIN} on validation windows whose '
                                      f'|recorded command| exceeds the training quantile',
                              'silence': f'at most {ZERO_COMMAND_FP_MAX:.0%} of validation windows with a '
                                         f'zero command may be predicted as turning',
                              'support': f'at least {MIN_INFORMATIVE_WINDOWS} informative validation windows',
                              'direction': f'a direction word needs >= {MIN_DIRECTION_WINDOWS} windows of that '
                                           f'direction in validation AND sign accuracy >= {MIN_DIRECTION_SIGN} on '
                                           f'them, otherwise it is withheld'},
            'controls': ['Motor-group-only fit (documented ceiling: the excluded shortcut)',
                         'Shuffled-label fit', 'Zero-input prediction', 'Test split opened after freezing'],
            'explicit_non_claims': [
                'The command is a recorded controller output, not a validated intention of the fly.',
                'A qualified readout does not claim that the body followed the command; measured motion is '
                'reported separately.',
                'No window is labelled from the human sentence, the scene, the planned stimulus or the '
                'measured body motion.'],
            'candidate_only': True, 'biologically_calibrated': False,
            'revision_history': [
                'v2 (join round, run conspecific-v6): the elicitation round added follow-source, song/nudge and '
                'cold sessions to make left-turn commands elicit rather than depend on the 43:1 right/left '
                'imbalance. All 72 new sessions are collected and the fit reads 438 windows (294 from the '
                'frozen base run plus 144 from the round). The frozen conspecific-v5 corpus is reused through '
                'a verified extension chain: every base episode keeps its id, split, condition, turns and '
                'environment, and each row keeps the hash of the run it was actually collected under. Note on '
                'rounds like this: after the round, a resume attempt was refused by the model-snapshot guard '
                'because the collection script had been edited for the union support; the corpus was already '
                'complete so nothing was lost, but extending a corpus with changed code needs its own run.',
                'The two earlier intent revisions predicted measured body outcomes and both left the turn '
                'component unqualified. This revision reads the command instead, which is what "转向" means '
                'here: the fly\'s own descending steering output, not the body\'s weak response to it.',
                'Discarded preregistration 33114c0244198c7b843dcd073ad423f827945171d769b599e8f3b292f9e24be5: it '
                'named the persisted drive record as the label source, but that record spans the whole session, '
                'so its first 500 ms are not this window\'s commands. It was replaced before any window was '
                'fitted or scored; labels now come from the commands recorded in the window\'s own physical run.',
                'The direction gate was added before the served artifact existed: the corpus contains 41 '
                'informative right-turn commands and 1 left-turn command, so a left word has no validation '
                'support and is withheld instead of being spoken on an untested direction. This replaced '
                'preregistration e70d0760bb1dd05e12aca6c9c112222eea9564a2f34107685b1a7514f7d3616d, whose '
                'candidate fit the same rules but had no direction gate and was never published.',
                'Threshold bug fixed before this fit: the informative-window quantile was filtered with '
                '"if is_train" on the split names, and a non-empty string is always truthy, so the quantile '
                'covered the whole corpus instead of the training split only. On this corpus the two values '
                'coincide to five decimals (0.011943583629361808), so no reported number moves, but the rule '
                'as written was wrong and is now enforced and unit-tested. This replaced preregistration '
                '377e16dd374190ae8ba8368c56764b31b08525e8463dd7033268c5cb34062a2b, which was also never '
                'used for a published artifact.']}


def jsonable(value, path='report', converted=None):
    """Convert numpy scalars/arrays for the report, and report what was converted.

    A silent conversion could hide a value that should have been a plain number,
    so every conversion is printed.
    """
    converted = [] if converted is None else converted
    if isinstance(value, dict):
        return {key: jsonable(item, f'{path}.{key}', converted) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item, f'{path}[{i}]', converted) for i, item in enumerate(value)]
    if isinstance(value, np.ndarray):
        converted.append((path, f'ndarray{value.shape}', value.dtype.str))
        return value.tolist()
    if isinstance(value, np.generic):
        converted.append((path, type(value).__name__, None))
        return value.item()
    return value


def corpus_rows(folder):
    rows = []
    for episode in sorted(folder.glob('*')):
        complete = episode / 'complete.json'
        if complete.is_file():
            rows.extend(json.loads(complete.read_text(encoding='utf8'))['rows'])
    return rows


def extension_chain(plan, preregistration, base_plan, base_preregistration):
    """Verify that an extended corpus only appends episodes to a frozen one.

    A targeted elicitation round costs hours, so it may extend an existing corpus:
    every base episode must appear in the new plan with the same id, split,
    condition, turns and environment. Seeds are reported, not required - the reused
    windows keep the seeds they were collected with.
    """
    def canonical(episode):
        return json.dumps({key: episode[key] for key in
                           ('id', 'split', 'condition', 'turns', 'environment')},
                          sort_keys=True, ensure_ascii=False)
    old = json.loads(Path(base_preregistration).read_text(encoding='utf8'))
    index = {episode['id']: episode for episode in plan['episodes']}
    changed = [{'id': episode['id'],
                'problem': 'missing' if episode['id'] not in index else 'differs'}
               for episode in old['episodes']
               if episode['id'] not in index or canonical(index[episode['id']]) != canonical(episode)]
    if changed:
        raise SystemExit('Extension protocol is not a superset of the base protocol: '
                         + json.dumps(changed[:5], ensure_ascii=False))
    seeds = [episode['id'] for episode in old['episodes']
             if index[episode['id']]['seed'] != episode['seed']]
    return {'base_run': old.get('run_id'), 'base_protocol_sha256': sha256(base_preregistration),
            'protocol_sha256': sha256(preregistration), 'base_episodes': len(old['episodes']),
            'appended_episodes': len([episode for episode in plan['episodes']
                                      if episode['id'] not in {e['id'] for e in old['episodes']}]),
            'seed_mismatches': len(seeds), 'seed_mismatch_examples': seeds[:5], 'verified': True}


def read_corpus(plan, brain, verify=False, limit=None, base_rows=None):
    folder = CORPUS / plan['run_id'] / 'trials'
    rows = corpus_rows(folder) + list(base_rows or [])
    if not rows:
        raise SystemExit('No complete corpus rows in ' + str(folder))
    if limit:
        rows = rows[:limit]
    features, commands, splits, records, excluded = [], [], [], [], set()
    verified = 0
    # One validated anatomy snapshot instead of rebuilding the 166,700-row
    # annotation view for every window.
    prepared = PreparedMotorAnnotations(brain.ids, brain.rows, brain.dataset)
    for row in rows:
        folder_of_row = Path(row['body']).parent
        body = json.loads(Path(row['body']).read_text(encoding='utf8'))
        drive = json.loads((folder_of_row / 'behavior-drive.json').read_text(encoding='utf8'))
        entry = command_target(body)
        # Tie the sampled commands back to the validated drive record of this window.
        for frame in body['trajectory']:
            index = int(frame['source_command_bin'])
            if index >= 0:
                recorded = drive['commands'][index]
                if any(recorded[key] != frame['commands'][key] for key in frame['commands']):
                    raise SystemExit(f"Window {row['record']} frame commands do not match its drive bin {index}")
        if verify:
            with np.load(folder_of_row / 'cumulative-neural.npz', allow_pickle=False) as saved:
                trial = SimpleNamespace(counts=saved['counts'], spike_ids=saved['spike_ids'],
                                        spike_steps=saved['spike_steps'], seed=int(drive['seed']),
                                        duration_ms=float(drive['duration_ms']))
                driven = [brain.lookup[i] for i in drive['directly_driven_neuron_ids']]
                recomputed = build_behavior_drive(trial, brain.ids, prepared, dataset=brain.dataset,
                                                  driven_indices=driven, context=drive['context'])
            for key in ('left', 'right'):
                np.testing.assert_allclose([c[key] for c in recomputed['commands']],
                                           [c[key] for c in drive['commands']], atol=1e-9)
            if recomputed['source_spikes_sha256'] != drive['source_spikes_sha256']:
                raise SystemExit('Recomputed spikes differ from the recorded drive: ' + row['record'])
            verified += 1
        with np.load(row['record'], allow_pickle=False) as saved:
            counts, state, duration = saved['counts'], saved['state_record'], float(saved['duration_ms'])
        features.append(internal_features(counts, state, duration))
        commands.append(entry)
        splits.append(row['split'])
        excluded.update(motor_group_indices(drive, brain))
        records.append({'episode': row['episode'], 'turn': row['turn'], 'split': row['split'],
                        'session_id': row['session_id'], 'record': row['record'],
                        'record_sha256': row['record_sha256'], 'body': row['body'],
                        'command': entry})
    return np.asarray(features), commands, np.asarray(splits), records, sorted(excluded), verified


def active_features(x, allowed, training, brain):
    """Allowed, training-active cells, expanded to the three feature blocks."""
    active = np.any(x[training][:, :x.shape[1] // 3] > 0, axis=0)
    indices = np.flatnonzero(allowed)[active[np.flatnonzero(allowed)]]
    if not len(indices):
        raise SystemExit('No internal training response: cannot fit a command readout')
    return indices, np.concatenate([indices, indices + len(brain.ids), indices + 2 * len(brain.ids)])


def fit(plan, preregistration, brain, x, commands, splits, records, excluded, verified, union=None):
    training, validation, test = splits == 'train', splits == 'validation', splits == 'test'
    target = np.asarray([row['steer_command'] for row in commands], dtype=float)
    mean = np.array([target[training].mean()])
    scale = np.array([target[training].std(ddof=0)])
    if np.any(scale <= 0):
        raise SystemExit('Degenerate target scale in the training split')
    # y keeps a target axis so Ridge returns (1, features) coefficients, matching
    # the artifact layout the decoder validates.
    y = standardize(target, mean, scale)[:, None]
    rules = command_rules(commands, splits)
    for i, record in enumerate(records):
        record['labels'] = command_labels(commands[i], rules)

    allowed = blind_mask(brain, _CATALOGUE)
    already_blinded = int(np.count_nonzero(~allowed[excluded]))
    stripped = allowed.copy()
    stripped[excluded] = False
    motor_only = allowed & ~stripped
    indices, feature_indices = active_features(x, stripped, training, brain)
    motor_indices, motor_features = active_features(x, motor_only, training, brain)
    print('windows train/validation/test:', int(training.sum()), int(validation.sum()), int(test.sum()),
          '| blind features:', len(indices), '| excluded motor cells:', len(excluded),
          '| motor-only features:', len(motor_indices))

    # Training-class counts and balanced weights, from the training split only.
    threshold = max(rules['informative_abs_command'], rules['zero_epsilon'])
    classes = np.where(target[training] >= threshold, 'left',
                       np.where(target[training] <= -threshold, 'right', 'quiet'))
    class_counts = {name: int((classes == name).sum()) for name in ('left', 'right', 'quiet')}
    balanced_weights = np.array([1. / max(1, class_counts[name]) for name in classes])
    balanced_weights = balanced_weights / balanced_weights.mean()
    print('training classes:', json.dumps(class_counts))

    def fit_predict(feature_index, balanced=True):
        """Fit, then apply the same standardization the artifact stores.

        Ridge stores a 1-D coefficient vector for a single target, so the
        coefficient matrix is rebuilt explicitly here and kept (1, features) for
        both the served artifact and the ceiling control. Sample weights balance
        the three training classes so the majority direction cannot crowd out the
        minority one; class membership comes from the training split alone. The
        ceiling control keeps the plain unweighted fit, exactly as in v2.
        """
        weights = balanced_weights if balanced else np.ones_like(balanced_weights)
        with threadpool_limits(limits=2):
            model = Ridge(alpha=10, solver='cholesky').fit(
                x[training][:, feature_index], y[training], sample_weight=weights)
        coefficients = np.atleast_2d(model.coef_)
        intercept = np.atleast_1d(model.intercept_)
        if coefficients.shape != (1, len(feature_index)) or intercept.shape != (1,):
            raise SystemExit('Unexpected Ridge coefficient layout')
        standardized = x[:, feature_index] @ coefficients.T + intercept
        return coefficients, intercept, (standardized * scale + mean)[:, 0]

    coefficients, intercept, predicted = fit_predict(feature_indices)
    _, _, ceiling = fit_predict(motor_features, balanced=False)

    def correlation(values, mask):
        if mask.sum() < 3 or np.std(values[mask]) == 0 or np.std(target[mask]) == 0:
            return None
        return float(np.corrcoef(values[mask], target[mask])[0, 1])

    def informative(mask):
        return mask & (np.abs(target) >= max(rules['informative_abs_command'], rules['zero_epsilon']))

    def sign_accuracy(values, mask):
        subset = informative(mask)
        if not subset.sum():
            return {'windows': 0, 'accuracy': None}
        agree = np.sign(values[subset]) == np.sign(target[subset])
        return {'windows': int(subset.sum()), 'accuracy': float(agree.mean())}

    def zero_false_positive(values, mask):
        subset = mask & (np.abs(target) <= rules['zero_epsilon'])
        if not subset.sum():
            return {'windows': 0, 'rate': None}
        spoken = np.abs(values[subset]) >= max(rules['informative_abs_command'], rules['zero_epsilon'])
        return {'windows': int(subset.sum()), 'spoken': int(spoken.sum()), 'rate': float(spoken.mean())}

    validation_r = correlation(predicted, validation)
    validation_sign = sign_accuracy(predicted, validation)
    validation_zero = zero_false_positive(predicted, validation)
    qualified = bool(validation_r is not None and validation_r >= COMMAND_R_MIN
                     and validation_sign['accuracy'] is not None and validation_sign['accuracy'] >= COMMAND_SIGN_MIN
                     and validation_zero['rate'] is not None and validation_zero['rate'] <= ZERO_COMMAND_FP_MAX
                     and validation_sign['windows'] >= MIN_INFORMATIVE_WINDOWS)
    # Per-direction support and accuracy: a direction with no validation window is
    # never served, and neither is one the readout gets wrong there.
    support = {split: {label: int(sum(label in records[i]['labels'] for i in np.flatnonzero(mask)))
                       for label in LABELS}
               for split, mask in (('train', training), ('validation', validation), ('test', test))}
    direction_accuracy = {}
    for name, sign in (('left', 1), ('right', -1)):
        subset = [i for i in np.flatnonzero(validation) if sign * target[i] >= rules['informative_abs_command']]
        direction_accuracy[name] = {
            'windows': len(subset),
            'sign_correct': int(sum(1 for i in subset if (predicted[i] > 0) == (sign > 0))),
            'accuracy': (float(sum(1 for i in subset if (predicted[i] > 0) == (sign > 0)) / len(subset))
                         if subset else None)}
    served = {}
    for label, name in (('command:turn-left', 'left'), ('command:turn-right', 'right')):
        accuracy = direction_accuracy[name]['accuracy']
        served[label] = bool(support['validation'][label] >= MIN_DIRECTION_WINDOWS
                             and accuracy is not None and accuracy >= MIN_DIRECTION_SIGN)
    if not any(served.values()):
        print('WARNING: no direction has validation support - withholding all direction words')
        qualified = False
    sign, sign_evidence = turn_sign()
    if sign is None:
        print('WARNING: direction words withheld -', sign_evidence)
        # Without the measured direction calibration a "turn" word cannot be
        # named, so the readout must not claim to have qualified.
        qualified = False

    candidate = OUT / 'malecns-command.candidate.npz'
    candidate.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(candidate, feature_indices=feature_indices, indices=indices,
                        weights=coefficients, intercept=intercept,
                        target_mean=mean, target_scale=scale, qualified=np.array([qualified]),
                        rules=json.dumps(rules),
                        metadata=json.dumps({'schema': SCHEMA, 'mode': 'command',
                            'contract': model_contract(brain, _CATALOGUE),
                            'input_mode': 'environment_only', 'split_unit': 'whole_session',
                            'protocol_sha256': sha256(preregistration),
                            'readout_module_sha256': sha256(Path(__file__).resolve().parents[1]
                                                           / 'chatflypt/command_readout.py'),
                            'tested_durations_ms': sorted({500.}), 'features': FEATURES,
                            'feature_rule': plan['feature_rule'],
                            'label_source': plan['label_source'],
                            'excluded_motor_cells': len(excluded),
                            'served_directions': served,
                            'sign_convention': (sign_evidence or {}).get('convention'),
                            'sign_calibration': sign_evidence if sign is not None else
                                                {'status': 'unavailable', 'reason': sign_evidence},
                            'verified_corpus_windows': verified}))
    frozen = sha256(candidate)
    from chatflypt.command_readout import CommandDecoder
    decoder = CommandDecoder(brain, _CATALOGUE, candidate)
    for index in [i for i, is_test in enumerate(test) if is_test][:3]:
        with np.load(records[index]['record'], allow_pickle=False) as saved:
            served_prediction = decoder.predict(saved['counts'], saved['state_record'],
                                                float(saved['duration_ms']))
        np.testing.assert_allclose(served_prediction, predicted[index], atol=1e-6)
    shuffled = predicted.copy()
    shuffled[test] = predicted[np.random.default_rng(4242).permutation(np.flatnonzero(test))]
    zero_prediction = float(decoder.predict(np.zeros(len(brain.ids)),
                                            np.zeros((4, len(brain.ids))), 500.)[0])
    report = {'schema': 'chatflypt.command-readout-evaluation.v1', 'run_id': plan['run_id'],
              'protocol_sha256': sha256(preregistration), 'artifact': str(candidate.relative_to(ROOT)),
              'artifact_sha256': frozen,
              'windows': {'train': int(training.sum()), 'validation': int(validation.sum()),
                          'test': int(test.sum())},
              'command_distribution': {
                  'nonzero_windows': int((np.abs(target) > rules['zero_epsilon']).sum()),
                  'informative_windows': int(informative(np.ones_like(training, dtype=bool)).sum()),
                  'informative_abs_command': rules['informative_abs_command'],
                  'left_windows': int((target > rules['informative_abs_command']).sum()),
                  'right_windows': int((target < -rules['informative_abs_command']).sum())},
              'features': {'blind_features': int(len(indices)), 'excluded_motor_cells': int(len(excluded)),
                           'motor_cells_already_blinded': already_blinded,
                           'motor_only_features': int(len(motor_indices))},
              'verified_against_recorded_drive': verified,
              'training_class_counts': class_counts,
              'sample_weighting': 'balanced across the three training classes (v3 revision); the motor-group '
                                  'ceiling control keeps the unweighted fit of v2',
              'extension': union,
              'qualified': qualified,
              'direction_support': support, 'served_directions': served,
              'direction_accuracy': direction_accuracy,
              'validation': {'correlation': validation_r, 'turn_sign': validation_sign,
                             'zero_command_false_positive': validation_zero},
              'test': {'correlation': correlation(predicted, test), 'turn_sign': sign_accuracy(predicted, test),
                       'zero_command_false_positive': zero_false_positive(predicted, test),
                       'labels': {label: int(sum(label in records[i]['labels'] for i in np.flatnonzero(test)))
                                  for label in LABELS},
                       'predicted_labels': {label: int(sum(predicted[i] > rules['informative_abs_command']
                                                           if label == 'command:turn-left'
                                                           else predicted[i] < -rules['informative_abs_command']
                                                           for i in np.flatnonzero(test))) for label in LABELS}},
              'controls': {
                  'shuffled_test_correlation': correlation(shuffled, test),
                  'motor_only_ceiling': {'correlation': {'validation': correlation(ceiling, validation),
                                                         'test': correlation(ceiling, test)},
                                         'explanation': 'Same fit restricted to the mapped motor groups the '
                                                        'command is computed from: the shortcut this artifact excludes'},
                  'zero_input_prediction': zero_prediction,
                  'served_matches_fitted_matrix': True},
              'sign_calibration': sign_evidence if sign is not None else {'status': 'unavailable',
                                                                          'reason': sign_evidence},
              'direction_words_available': sign is not None,
              'candidate_only': True, 'biologically_calibrated': False,
              'limitations': plan['explicit_non_claims'] + [
                  'Small environment-only corpus; windows within a session are correlated.',
                  'The command channel only carries steering when stepping is commanded; quiet windows are silent.']}
    REPORT.mkdir(parents=True, exist_ok=True)
    converted = []
    report = jsonable(report, converted=converted)
    if converted:
        print('report values converted from numpy types:', converted[:12], flush=True)
    dump(REPORT / 'evaluation.json', report)
    dump(REPORT / 'windows.json',
         [dict(record, prediction=float(predicted[i]), ceiling=float(ceiling[i]))
          for i, record in enumerate(records)])
    print(json.dumps({'artifact': frozen, 'qualified': qualified,
                      'validation': report['validation'], 'test': report['test'],
                      'controls': report['controls']['motor_only_ceiling'],
                      'direction_words_available': report['direction_words_available']}, ensure_ascii=False))
    return candidate


def main():
    global _CATALOGUE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--verify-labels', action='store_true',
                        help='Recompute every window drive from its recorded spikes and compare')
    parser.add_argument('--limit', type=int, default=None,
                        help='Smoke-test only: use the first N corpus windows')
    parser.add_argument('--publish', action='store_true',
                        help='Copy the frozen candidate to the live path (manual step, never automatic)')
    parser.add_argument('--base-run', help='Frozen readout-corpus run id(s) this round extends, comma separated')
    parser.add_argument('--protocol', type=Path, help='Preregistration file (default reports/command-readout/protocol.json)')
    args = parser.parse_args()
    preregistration = args.protocol or REPORT / 'protocol.json'
    if not preregistration.is_file():
        dump(preregistration, protocol(args.run_id))
        print('preregistered', preregistration, 'sha256', sha256(preregistration))
    plan = json.loads(preregistration.read_text(encoding='utf8'))
    if plan['run_id'] != args.run_id:
        raise SystemExit(f"Preregistration names run {plan['run_id']}; refusing to relabel it as {args.run_id}")
    base_rows = []
    chains = []
    if args.base_run:
        from train_unified_readout import extension_chain as unified_chain
        # The chain is verified between the *corpus* protocols, not this readout's
        # own preregistration: the question is whether the run whose windows we
        # read is a verified extension of the frozen base corpus.
        run_protocol = ROOT / 'reports/unified-readout' / 'runs' / args.run_id / 'protocol.json'
        for name in [item for item in args.base_run.split(',') if item]:
            base_protocol = ROOT / 'reports/unified-readout' / 'runs' / name / 'protocol.json'
            chain = unified_chain(json.loads(run_protocol.read_text(encoding='utf8')), run_protocol,
                                  json.loads(base_protocol.read_text(encoding='utf8')), base_protocol)
            chains.append(chain)
            base_rows.extend(corpus_rows(CORPUS / name / 'trials'))
        print(json.dumps({'extension_chains': chains}, ensure_ascii=False))
    brain = MaleCNS()
    _CATALOGUE = build_catalogue(save=False)
    x, commands, splits, records, excluded, verified = read_corpus(plan, brain, verify=args.verify_labels,
                                                                  limit=args.limit, base_rows=base_rows)
    print('corpus windows:', len(records), '| label verification:', verified)
    candidate = fit(plan, preregistration, brain, x, commands, splits, records, excluded, verified,
                    union=chains or None)
    if args.publish:
        target = OUT / 'malecns-command.npz'
        target.write_bytes(candidate.read_bytes())
        print('published', target, sha256(target))


if __name__ == '__main__':
    main()
