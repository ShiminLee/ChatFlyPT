"""Which conditions actually produce one-sided (net) turn commands?

Per condition: how many windows show left-only / right-only / both / neither turn
commands, and what the measured yaw looks like in each case. This points at the
conditions worth enriching if a command-driven turn corpus is to be collected.

Measurement caveat: `command_summary` holds a per-window min/max, and a window's
first sampled frame always carries the zero dict, so a negative-only (backward)
drive reports max 0 and lands in "neither" here. Counts are a lower bound; see
`scripts/train_command_readout.py` (frame means) and `scripts/probe_command_sign.py`
(sign calibration).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.paths import ROOT, dump
from chatflypt.intent_readout import measured_outcome

CORPUS = ROOT / 'data/derived/unified-readout/runs'
REPORT = ROOT / 'reports/intent-readout'


def family(condition):
    for name, markers in (('nudge', ('nudge',)), ('tap', ('tap',)), ('touch', ('touch_',)),
                          ('unilateral_visual', ('unilateral_bright',)),
                          ('unilateral_odor', ('unilateral_yeast',)),
                          ('unilateral_thermal', ('unilateral_heat', 'unilateral_cold')),
                          ('pheromone', ('pheromone',)), ('song', ('song',)),
                          ('platform', ('leg_sugar', 'leg_block', 'mouth_')),
                          ('odor', ('yeast',)), ('thermal', ('heat', 'cold')),
                          ('visual', ('bright', 'dark'))):
        if any(marker in condition for marker in markers):
            return name
    return 'other'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='conspecific-v5')
    args = parser.parse_args()
    trials = CORPUS / args.run_id / 'trials'
    per_condition = defaultdict(lambda: {'windows': 0, 'left_only': 0, 'right_only': 0, 'both': 0,
                                         'neither': 0, 'yaw_left_only': [], 'yaw_right_only': [],
                                         'yaw_both': [], 'yaw_neither': []})
    for episode in sorted(trials.glob('*')):
        complete = episode / 'complete.json'
        if not complete.is_file():
            continue
        condition = episode.name.split('-', 1)[1] if '-' in episode.name else episode.name
        for row in json.loads(complete.read_text(encoding='utf8'))['rows']:
            body = json.loads(Path(row['body']).read_text(encoding='utf8'))
            summary = body.get('command_summary') or {}
            left = float((summary.get('left') or {}).get('max', 0.))
            right = float((summary.get('right') or {}).get('max', 0.))
            outcome = measured_outcome(body['trajectory']) if body.get('trajectory') else None
            entry = per_condition[condition]
            entry['windows'] += 1
            if left > 0 and right > 0:
                key = 'both'
            elif left > 0:
                key = 'left_only'
            elif right > 0:
                key = 'right_only'
            else:
                key = 'neither'
            entry[key] += 1
            if outcome is not None:
                entry[f'yaw_{key}'].append(abs(outcome['yaw_change_deg']))
    families = defaultdict(lambda: dict.fromkeys(('windows', 'left_only', 'right_only', 'both', 'neither'), 0))
    for condition, entry in per_condition.items():
        target = families[family(condition)]
        for key in ('windows', 'left_only', 'right_only', 'both', 'neither'):
            target[key] += entry[key]
    rows = []
    for name, entry in sorted(families.items(), key=lambda item: -(item[1]['left_only'] + item[1]['right_only'])):
        rows.append({'family': name, **entry,
                     'one_sided_fraction': round((entry['left_only'] + entry['right_only'])
                                                 / max(1, entry['windows']), 3)})
    detail = {}
    for condition, entry in sorted(per_condition.items()):
        yaw = {key: {'windows': len(values),
                     'median_abs_yaw_deg': float(np.median(values)) if values else None}
               for key, values in entry.items() if key.startswith('yaw_')}
        detail[condition] = {**{k: v for k, v in entry.items() if not k.startswith('yaw_')}, 'measured': yaw}
    report = {'schema': 'chatflypt.turn-command-by-condition.v1', 'run_id': args.run_id,
              'families': rows, 'conditions': detail,
              'note': 'One-sided commands are what a turn-intent readout needs; both-sided windows have no net turn.'}
    REPORT.mkdir(parents=True, exist_ok=True)
    dump(REPORT / f'{args.run_id}-turn-commands-by-condition.json', report)
    print(f"{'family':20s} {'win':>4s} {'L-only':>7s} {'R-only':>7s} {'both':>5s} {'none':>5s} {'one-sided':>9s}")
    for row in rows:
        print(f"{row['family']:20s} {row['windows']:4d} {row['left_only']:7d} {row['right_only']:7d} "
              f"{row['both']:5d} {row['neither']:5d} {row['one_sided_fraction']:9.3f}")


if __name__ == '__main__':
    main()
