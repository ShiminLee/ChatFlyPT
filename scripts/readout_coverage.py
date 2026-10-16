"""Label-by-condition coverage matrix of a collected readout corpus.

For every collected window this reads which channel:side labels were *actually
delivered* (measured receptor events inside the sampled windows) and aggregates
them by condition, split and positive count. The matrix is what an elicitation
round is designed from: a label with too few delivered positives, or with
positives only from bilateral scenes, cannot be qualified however good the
decoder is; a label whose channels have no presentation route stays an explicit
abstention instead of a data gap. Nothing here is a decoder output.

Writes `reports/unified-readout/coverage-<run-id>.json`.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.paths import ROOT, dump
from chatflypt.sensory_catalogue import build_catalogue
from train_unified_readout import CHANNELS, measured_targets

CORPUS = ROOT / 'data/derived/unified-readout/runs'
REPORT = ROOT / 'reports/unified-readout'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    trials = CORPUS / args.run_id / 'trials'
    if not trials.is_dir():
        raise SystemExit('No corpus at ' + str(trials))
    catalogue = build_catalogue(save=False)
    per_label_split = defaultdict(Counter)
    label_conditions = defaultdict(Counter)
    condition_windows = Counter()
    windows = 0
    for episode in sorted(trials.glob('*')):
        complete = episode / 'complete.json'
        if not complete.is_file():
            continue
        condition = episode.name.split('-', 1)[1] if '-' in episode.name else episode.name
        for row in json.loads(complete.read_text(encoding='utf8'))['rows']:
            samples = json.loads(Path(row['feedback']).read_text(encoding='utf8'))['samples']
            with np.load(row['events'], allow_pickle=False) as handle:
                events = {key: handle[key] for key in handle.files}
            labels = measured_targets(samples, events, catalogue, CHANNELS)
            windows += 1
            condition_windows[condition] += 1
            for label in labels:
                per_label_split[label][row['split']] += 1
                label_conditions[label][condition] += 1
    report = {
        'schema': 'chatflypt.readout-coverage.v1', 'run_id': args.run_id, 'windows': windows,
        'conditions': dict(sorted(condition_windows.items())),
        'labels': {label: {'train': counts['train'], 'validation': counts['validation'],
                           'test': counts['test'],
                           'conditions': dict(label_conditions[label].most_common())}
                   for label, counts in sorted(per_label_split.items())},
        'notes': ['Delivered means measured receptor events inside the sampled windows, not the planned scene.',
                  'A label with fewer than two positives in a split cannot pass the per-label gates there.',
                  'Labels whose channels have no presentation route (wing membrane, whole thorax, labellum) '
                  'cannot appear here at all; that is a documented abstention, not missing data.'],
        'biologically_calibrated': False}
    REPORT.mkdir(parents=True, exist_ok=True)
    dump(REPORT / f'coverage-{args.run_id}.json', report)
    print(json.dumps({'windows': windows, 'labels': len(report['labels']),
                      'conditions': len(report['conditions'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
