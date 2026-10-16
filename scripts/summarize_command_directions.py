"""Per-direction accuracy of a fitted command readout, derived from its window record.

The gate that decides whether a direction word may be served needs both support and
per-direction sign accuracy on validation. Both are recomputed here from
`reports/command-readout/windows.json` (the run's own per-window predictions and
recorded commands), so the justification for withholding a direction is reproducible
without refitting anything.

Writes `reports/command-readout/direction-summary.json`.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatflypt.paths import ROOT, dump

REPORT = ROOT / 'reports/command-readout'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=Path, default=REPORT / 'windows.json')
    parser.add_argument('--evaluation', type=Path, default=REPORT / 'evaluation.json')
    args = parser.parse_args()
    windows = json.loads(args.windows.read_text(encoding='utf8'))
    evaluation = json.loads(args.evaluation.read_text(encoding='utf8'))
    threshold = evaluation['command_distribution']['informative_abs_command']
    summary = {'schema': 'chatflypt.command-direction-summary.v1',
               'informative_abs_command': threshold,
               'source': 'derived from the run\'s own windows.json; nothing refitted',
               'splits': {}}
    for split in ('train', 'validation', 'test'):
        rows = [row for row in windows if row['split'] == split]
        entry = {}
        for name, sign in (('left', 1), ('right', -1)):
            subset = [row for row in rows if sign * row['command']['steer_command'] >= threshold]
            correct = sum(1 for row in subset if (row['prediction'] > 0) == (sign > 0))
            spoken = sum(1 for row in subset
                         if (row['prediction'] >= threshold if sign > 0 else row['prediction'] <= -threshold))
            entry[name] = {'windows': len(subset), 'sign_correct': correct,
                           'accuracy': (correct / len(subset)) if subset else None, 'spoken': spoken}
        entry['zero_command_windows'] = sum(
            1 for row in rows if abs(row['command']['steer_command']) < threshold)
        summary['splits'][split] = entry
    summary['served_directions'] = evaluation.get('served_directions')
    summary['rule'] = ('A direction is served only when validation holds at least two windows of that '
                       'direction and the readout gets their sign right at least half the time; the '
                       'gate lives in chatflypt/command_readout.py (MIN_DIRECTION_WINDOWS, MIN_DIRECTION_SIGN).')
    dump(REPORT / 'direction-summary.json', summary)
    print(json.dumps(summary['splits'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
