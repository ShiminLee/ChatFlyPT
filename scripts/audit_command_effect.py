"""Do one-sided turn commands actually turn the body? Signed yaw, by command type.

The decisive question for a turn-intent readout: in windows where only the left
turn group is commanded, is the measured yaw actually to the left? If the command
does not move the body consistently, then no decoder - however good - can produce
a behaviour-validated turn label, and the fix would belong to the motor path
rather than to the readout.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='conspecific-v5')
    args = parser.parse_args()
    groups = defaultdict(list)
    for episode in sorted((CORPUS / args.run_id / 'trials').glob('*')):
        complete = episode / 'complete.json'
        if not complete.is_file():
            continue
        for row in json.loads(complete.read_text(encoding='utf8'))['rows']:
            body = json.loads(Path(row['body']).read_text(encoding='utf8'))
            summary = body.get('command_summary') or {}
            left = float((summary.get('left') or {}).get('max', 0.))
            right = float((summary.get('right') or {}).get('max', 0.))
            outcome = measured_outcome(body['trajectory']) if body.get('trajectory') else None
            if outcome is None:
                continue
            key = ('both' if left > 0 and right > 0 else 'left_only' if left > 0 else
                   'right_only' if right > 0 else 'neither')
            groups[key].append({'yaw': outcome['yaw_change_deg'],
                                'left': left, 'right': right,
                                'episode': episode.name, 'turn': row['turn']})

    def describe(rows):
        yaw = np.asarray([row['yaw'] for row in rows])
        if not yaw.size:
            return {'windows': 0}
        left_fraction = float((yaw < -1.).mean())
        right_fraction = float((yaw > 1.).mean())
        return {'windows': int(yaw.size), 'median_yaw_deg': float(np.median(yaw)),
                'mean_yaw_deg': float(yaw.mean()), 'std_yaw_deg': float(yaw.std()),
                'left_beyond_1deg': round(left_fraction, 3), 'right_beyond_1deg': round(right_fraction, 3),
                'largest_left_deg': float(yaw.min()), 'largest_right_deg': float(yaw.max())}

    report = {'schema': 'chatflypt.command-effect-audit.v1', 'run_id': args.run_id,
              'by_command_type': {key: describe(rows) for key, rows in sorted(groups.items())},
              'examples': {key: sorted(rows, key=lambda row: row['yaw'])[:3] + sorted(rows, key=lambda row: -row['yaw'])[:3]
                           for key, rows in sorted(groups.items())},
              'interpretation': ('A left-only command should produce negative yaw. If the left/right share is near '
                                 'chance, the recorded command does not turn the body consistently in this setup, so '
                                 'a behaviour-validated turn label is not available and the fix belongs to the motor path.')}
    REPORT.mkdir(parents=True, exist_ok=True)
    dump(REPORT / f'{args.run_id}-command-effect.json', report)
    print(json.dumps(report['by_command_type'], ensure_ascii=False, indent=1))
    print()
    for key in ('left_only', 'right_only', 'both', 'neither'):
        rows = groups.get(key, [])
        if not rows:
            continue
        ordered = sorted(rows, key=lambda row: row['yaw'])
        print(f"{key}: most-left {ordered[0]['episode']}#{ordered[0]['turn']} yaw={ordered[0]['yaw']:.2f} | "
              f"most-right {ordered[-1]['episode']}#{ordered[-1]['turn']} yaw={ordered[-1]['yaw']:.2f}")


if __name__ == '__main__':
    main()
