"""Engineering calibration of the pinned controller's steering sign.

The turn command readout has to name a direction ("left" / "right"), so this
probe measures which way the pinned Bad Apple Fly controller actually drives the
body when one mapped motor group is active on one side. Spikes here are
synthetic: one mapped cell group at a time, a fixed population rate, no
connectome simulation, no chat session, no user message and no stimulus. It
characterises the pinned controller and body only; it is not a neural experiment
and it is not evidence about fly behaviour.

It also fixes the size of the asymmetry effect: `turn_inside` is multiplicative
on the forward drive, so it needs a walking drive to do anything at all.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.paths import ROOT, dump, sha256
from chatflypt.malecns import MaleCNS
from chatflypt.behavior_readout import build_behavior_drive
from chatflypt.behavior_body import run_behavior_body
from chatflypt.intent_readout import measured_outcome

REPORT = ROOT / 'reports/command-readout'
RAW = ROOT / 'data/derived/command-readout/sign-probe'
DT_MS = .1
DURATION_MS = 500.
RATE_HZ = 80.
SEED = 20261001
# Each condition: (group, side, weight) triples. A weight scales the fixed
# population rate, so norm = weight * RATE_HZ / 100 in the pinned formula.
CONDITIONS = {
    'forward_only': [('forward', 'L', 1.), ('forward', 'R', 1.)],
    'forward_both_sides_plus_context': [('forward', 'L', 1.), ('forward', 'R', 1.),
                                        ('context_walk', 'L', 1.), ('context_walk', 'R', 1.)],
    'forward_turn_inside_L': [('forward', 'L', 1.), ('forward', 'R', 1.), ('turn_inside', 'L', 1.)],
    'forward_turn_inside_R': [('forward', 'L', 1.), ('forward', 'R', 1.), ('turn_inside', 'R', 1.)],
    'forward_turn_inside_both': [('forward', 'L', 1.), ('forward', 'R', 1.),
                                 ('turn_inside', 'L', 1.), ('turn_inside', 'R', 1.)],
    'turn_outside_L_only': [('turn_outside', 'L', 1.)],
    'turn_outside_R_only': [('turn_outside', 'R', 1.)],
    'backward_only': [('backward', 'L', 1.), ('backward', 'R', 1.)],
}


def member_indices(drive, brain):
    """Map the drive's own recorded group populations back to neuron indices."""
    table = {}
    for group in drive['groups']:
        for side, entry in group['sides'].items():
            table[(group['id'], side)] = [brain.lookup[i] for i in entry['neuron_ids']]
    return table


def synthetic_trial(weights, neuron_count, duration_ms=DURATION_MS, rate_hz=RATE_HZ, seed=SEED):
    """Seeded Poisson spike trains, one per cell, at rate_hz * that cell's weight.

    The pinned formula reads a population mean rate, so firing every cell of a
    group at rate_hz gives that group the same readout as a stronger burst in a
    few cells. Rates stay below the 100 Hz reference unless a weight says otherwise.
    """
    rng = np.random.default_rng(seed)
    ticks = round(duration_ms * 10)
    ids, steps = [], []
    for index, weight in sorted(weights.items()):
        count = int(rng.poisson(rate_hz * weight * duration_ms / 1000.))
        ids.extend([index] * count)
        steps.extend(rng.integers(0, ticks, size=count).tolist())
    ids = np.asarray(ids, dtype=np.int64)
    steps = np.asarray(steps, dtype=np.int64)
    order = np.argsort(ids, kind='stable')
    ids, steps = ids[order], steps[order]
    return SimpleNamespace(spike_ids=ids, spike_steps=steps,
                           counts=np.bincount(ids, minlength=neuron_count).astype(np.int64),
                           seed=seed, duration_ms=duration_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--rate-hz', type=float, default=RATE_HZ)
    args = parser.parse_args()
    brain = MaleCNS()
    reference = SimpleNamespace(spike_ids=np.zeros(0, dtype=np.int64), spike_steps=np.zeros(0, dtype=np.int64),
                                counts=np.zeros(len(brain.ids), dtype=np.int64), seed=SEED, duration_ms=DURATION_MS)
    skeleton = build_behavior_drive(reference, brain.ids, brain.rows, dataset=brain.dataset, driven_indices=[])
    table = member_indices(skeleton, brain)
    populations = {f'{group}/{side}': len(cells) for (group, side), cells in sorted(table.items())}
    records = []
    for name, recipe in CONDITIONS.items():
        missing = [f'{group}/{side}' for group, side, _ in recipe if not table.get((group, side))]
        if missing:
            records.append({'condition': name, 'status': 'skipped', 'reason': 'no annotated cells for ' + ', '.join(missing)})
            continue
        weights = {}
        for group, side, weight in recipe:
            for index in table[(group, side)]:
                weights[index] = weights.get(index, 0.) + weight
        for repeat in range(args.repeats):
            seed = SEED + 1000 * repeat + len(records)
            trial = synthetic_trial(weights, len(brain.ids), rate_hz=args.rate_hz, seed=seed)
            drive = build_behavior_drive(trial, brain.ids, brain.rows, dataset=brain.dataset, driven_indices=[])
            folder = RAW / f'{name}-{repeat}'
            body = run_behavior_body(drive, output_dir=folder, timeout=180)
            outcome = measured_outcome(body['trajectory']) if body.get('trajectory') else None
            commands = body.get('command_summary') or {}
            # Mean over the sampled frames, not the summary max: a negative-only
            # command (backward drive) has max 0 because the first frame is
            # always the zero dict, which would read as "no command at all".
            frames = (body.get('trajectory') or [])[1:]
            left = float(np.mean([f['commands']['left'] for f in frames])) if frames else 0.
            right = float(np.mean([f['commands']['right'] for f in frames])) if frames else 0.
            record = {'condition': name, 'repeat': repeat, 'seed': seed,
                      'synthetic_spikes': int(len(trial.spike_ids)),
                      'synthetic_cells': len(weights),
                      'recipe': [{'group': g, 'side': s, 'weight': w} for g, s, w in recipe],
                      'command_left_mean': left, 'command_right_mean': right,
                      'command_left_max': float((commands.get('left') or {}).get('max', 0.)),
                      'command_right_max': float((commands.get('right') or {}).get('max', 0.)),
                      'drive_asymmetry_right_minus_left': right - left,
                      'measured_yaw_deg': None if outcome is None else outcome['yaw_change_deg'],
                      'measured_path_mm': None if outcome is None else outcome['path_length_mm'],
                      'body_status': body.get('status'), 'physics_warnings': (body.get('summary') or {}).get('physics_warnings'),
                      'body_record': str((folder / 'body-result.json').relative_to(ROOT)) if (folder / 'body-result.json').is_file() else None}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    by_condition = {}
    for record in records:
        if record.get('status') == 'skipped':
            by_condition[record['condition']] = record
            continue
        entry = by_condition.setdefault(record['condition'], {'yaw_deg': [], 'path_mm': [],
                                                              'left': [], 'right': [], 'asym': []})
        entry['yaw_deg'].append(record['measured_yaw_deg'])
        entry['path_mm'].append(record['measured_path_mm'])
        entry['left'].append(record['command_left_mean'])
        entry['right'].append(record['command_right_mean'])
        entry['asym'].append(record['drive_asymmetry_right_minus_left'])
    summary = {}
    for name, entry in by_condition.items():
        if 'yaw_deg' not in entry:
            summary[name] = entry
            continue
        yaw = np.asarray(entry['yaw_deg'], dtype=float)
        summary[name] = {'repeats': len(yaw), 'mean_command_left': float(np.mean(entry['left'])),
                         'mean_command_right': float(np.mean(entry['right'])),
                         'mean_asymmetry_right_minus_left': float(np.mean(entry['asym'])),
                         'mean_yaw_deg': float(yaw.mean()), 'yaw_deg': yaw.tolist(),
                         'mean_path_mm': float(np.mean(entry['path_mm'])),
                         'sign_consistent': bool(np.all(yaw > 0) or np.all(yaw < 0))}
    report = {'schema': 'chatflypt.controller-sign-probe.v1',
              'scope': 'Synthetic neural drive into the unchanged pinned controller and body; engineering '
                       'calibration of the command sign, not a neural experiment and not behaviour evidence.',
              'input': 'Synthetic 0.1 ms spike trains in one mapped group/side at a fixed population rate',
              'rate_hz': args.rate_hz, 'duration_ms': DURATION_MS, 'reference_rate_hz': 100.,
              'context': 'ground', 'repeats': args.repeats,
              'group_populations': populations,
              'mapping_source': skeleton['mapping_source'],
              'source_sha256': {'behavior_readout.py': sha256(ROOT / 'chatflypt/behavior_readout.py'),
                                'vendor/badapple-fly/backend/evidence.py': sha256(
                                    ROOT / 'vendor/badapple-fly/backend/evidence.py')},
              'records': records, 'summary': summary,
              'note': 'turn_inside is multiplicative on the forward drive; alone it commands nothing.'}
    REPORT.mkdir(parents=True, exist_ok=True)
    dump(REPORT / 'controller-sign-probe.json', report)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
