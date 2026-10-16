"""Why the turn component cannot be read out: is any turning actually commanded?

Scans a collected corpus and reports, per window, the recorded controller command
groups (from the pinned controller's own summary) next to the measured yaw. If the
turning groups are almost never non-zero, then yaw is not command-driven in this
setup and no decoder can be expected to read an intention out of it - the fix
would have to be in the motor path, not in the readout.

Measurement caveat: `command_summary` holds a per-window min/max, and a window's
first sampled frame always carries the zero dict, so a drive that is negative
throughout (backward walking) reports max 0 and is counted here as "no command".
These counts are therefore a lower bound. `scripts/train_command_readout.py` uses
the sampled frames' means instead; the calibration probe that settles the sign is
`scripts/probe_command_sign.py`.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.paths import ROOT, dump
from chatflypt.intent_readout import measured_outcome

CORPUS = ROOT / 'data/derived/unified-readout/runs'
REPORT = ROOT / 'reports/intent-readout'
GROUPS = ('left', 'right', 'escape', 'feeding', 'anterior', 'song', 'antenna', 'rub',
          'flight', 'landing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='conspecific-v5')
    args = parser.parse_args()
    trials = CORPUS / args.run_id / 'trials'
    rows = []
    for episode in sorted(trials.glob('*')):
        complete = episode / 'complete.json'
        if complete.is_file():
            rows.extend(json.loads(complete.read_text(encoding='utf8'))['rows'])
    if not rows:
        raise SystemExit('No corpus rows in ' + str(trials))

    active = Counter()
    nonzero_by_group = Counter()
    yaw_with_turn, yaw_without_turn = [], []
    path_with_turn, path_without_turn = [], []
    per_split = {}
    for row in rows:
        body = json.loads(Path(row['body']).read_text(encoding='utf8'))
        summary = body.get('command_summary') or {}
        maxima = {group: float((summary.get(group) or {}).get('max', 0.)) for group in GROUPS}
        for group, value in maxima.items():
            if value > 0:
                nonzero_by_group[group] += 1
        turning = maxima['left'] > 0 or maxima['right'] > 0
        if turning:
            active['windows_with_turn_command'] += 1
            active['left_only'] += maxima['left'] > 0 >= maxima['right']
            active['right_only'] += maxima['right'] > 0 >= maxima['left']
            active['both'] += maxima['left'] > 0 and maxima['right'] > 0
        active['windows'] += 1
        outcome = measured_outcome(body['trajectory']) if body.get('trajectory') else None
        if outcome is None:
            continue
        (yaw_with_turn if turning else yaw_without_turn).append(abs(outcome['yaw_change_deg']))
        (path_with_turn if turning else path_without_turn).append(outcome['path_length_mm'])
        split = per_split.setdefault(row['split'], {'windows': 0, 'with_turn_command': 0})
        split['windows'] += 1
        split['with_turn_command'] += turning

    def describe(values):
        values = np.asarray(values)
        if not values.size:
            return {'windows': 0}
        return {'windows': int(values.size), 'median': float(np.median(values)),
                'p85': float(np.quantile(values, .85)), 'max': float(values.max())}

    report = {'schema': 'chatflypt.motor-command-audit.v1', 'run_id': args.run_id,
              'windows': active['windows'],
              'windows_with_turn_command': active['windows_with_turn_command'],
              'turn_command_breakdown': {key: active[key] for key in ('left_only', 'right_only', 'both')},
              'nonzero_windows_by_group': dict(nonzero_by_group),
              'abs_yaw_deg': {'with_turn_command': describe(yaw_with_turn),
                              'without_turn_command': describe(yaw_without_turn)},
              'path_mm': {'with_turn_command': describe(path_with_turn),
                          'without_turn_command': describe(path_without_turn)},
              'by_split': per_split,
              'conclusion': ('Turning groups are essentially never commanded in this corpus, so yaw is not '
                             'command-driven and no motor-intent decoder can be expected to read it; the fix '
                             'belongs to the motor path, not to the readout.'
                             if active['windows_with_turn_command'] / max(1, active['windows']) < .05 else
                             'Turn commands do occur; check whether the windows with commands are separable.'),
              'source': 'command_summary recorded by the pinned physical controller in each measured body record'}
    REPORT.mkdir(parents=True, exist_ok=True)
    dump(REPORT / f'{args.run_id}-motor-command-audit.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:2000])


if __name__ == '__main__':
    main()
