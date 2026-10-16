"""Agreement and stability of a fitted readout across independent sessions.

Field practice reports agreement between independent annotators instead of
averaging disagreement away. Here the independent "annotators" are the sessions:
each test session is a separate physical/neural run of the same individual, and
the two things being compared are the offline measured receptor labels and the
blind readout's predictions.

Reported per label:
  * Cohen's kappa between measured delivery and the readout (chance-corrected),
  * per-session exact-set accuracy for the whole label set,
  * the spread (min / median / max) across sessions, which is the fly analogue of
    between-annotator variation, and
  * how often the readout abstained (unqualified or out-of-duration windows).

Nothing here is a biological accuracy claim: sessions are correlated within a
split, and the labels are measured receptor events, not semantics.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chatflypt.paths import ROOT, dump


def kappa(expected_rows, predicted_rows, labels):
    result = {}
    for label in labels:
        tp = fp = fn = tn = 0
        for expected, predicted in zip(expected_rows, predicted_rows):
            want, got = label in expected, label in predicted
            tp += want and got
            fp += (not want) and got
            fn += want and (not got)
            tn += (not want) and (not got)
        total = tp + fp + fn + tn
        if total == 0:
            result[label] = None
            continue
        observed = (tp + tn) / total
        expected_agree = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (total * total)
        result[label] = None if expected_agree == 1 else (observed - expected_agree) / (1 - expected_agree)
    return result


def accuracy(rows, qualified_only, qualified):
    correct = 0
    for row in rows:
        expected = {label for label in row['expected'] if not qualified_only or label in qualified}
        predicted = {label for label in row['predicted'] if not qualified_only or label in qualified}
        correct += expected == predicted
    return correct / len(rows) if rows else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='conspecific-v4')
    parser.add_argument('--evaluation', type=Path)
    args = parser.parse_args()
    evaluation_path = args.evaluation or (ROOT / 'reports/unified-readout/runs' / args.run_id / 'evaluation.json')
    if not evaluation_path.is_file():
        raise SystemExit('No evaluation at ' + str(evaluation_path))
    evaluation = json.loads(evaluation_path.read_text(encoding='utf8'))
    labels = sorted({label for row in evaluation['test_predictions'] for label in row['expected']} |
                    {label for row in evaluation['test_predictions'] for label in row['predicted']})
    rows = evaluation['test_predictions']
    expected_rows = [row['expected'] for row in rows]
    predicted_rows = [row['predicted'] for row in rows]

    sessions = {}
    for row in rows:
        sessions.setdefault(row['episode'], []).append(row)
    qualified = set(evaluation.get('qualified_labels', []))
    per_session = {}
    for episode, session_rows in sorted(sessions.items()):
        per_session[episode] = {'windows': len(session_rows),
                                'exact_set_accuracy_all_labels': accuracy(session_rows, False, qualified),
                                'exact_set_accuracy_qualified_labels': accuracy(session_rows, True, qualified)}
    spoken = [value['exact_set_accuracy_qualified_labels'] for value in per_session.values()]
    every = [value['exact_set_accuracy_all_labels'] for value in per_session.values()]
    kappas = kappa(expected_rows, predicted_rows, labels)
    report = {'schema': 'chatflypt.readout-agreement.v1', 'run_id': args.run_id,
              'evaluation': str(evaluation_path.relative_to(ROOT)),
              'artifact_sha256': evaluation.get('artifact_sha256'),
              'annotators': 'independent test sessions (separate physical/neural runs of one individual)',
              'compared': {'A': 'offline measured receptor delivery', 'B': 'blind readout prediction'},
              'windows': len(rows), 'sessions': len(sessions),
              'qualified_labels': sorted(qualified),
              'cohens_kappa_by_label': {label: (value if label in qualified else
                                                {'status': 'abstained: label not qualified, so no spoken prediction exists'})
                                        for label, value in kappas.items()},
              'per_session': per_session,
              'session_accuracy': {
                  'qualified_labels': {'min': min(spoken), 'median': statistics.median(spoken), 'max': max(spoken)},
                  'all_labels': {'min': min(every), 'median': statistics.median(every), 'max': max(every),
                                 'note': 'Counts unqualified labels that are deliberately abstained on as misses.'},
                  'note': 'Spread across sessions is between-session variation, reported rather than averaged away.'},
              'limitations': ['Sessions are correlated within a split; this is not independent-animal replication.',
                              'Labels are measured receptor events, not semantic meaning.',
                              'Kappa is undefined when a label never changes value in the test split.']}
    dump(ROOT / 'reports/corpus' / f'{args.run_id}-agreement.json', report)
    print(json.dumps({'sessions': report['sessions'], 'windows': report['windows'],
                      'qualified_labels': sorted(qualified),
                      'session_accuracy': report['session_accuracy'],
                      'kappa': {k: (None if v is None else (v if isinstance(v, dict) else round(v, 3)))
                                for k, v in report['cohens_kappa_by_label'].items()}},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
