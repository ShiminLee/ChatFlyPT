"""Probe: in tethered flight, is the measured yaw actually command-driven?

Ground walking showed commanded left turns of only ~0.3 deg against 3-7 deg of
uncommanded motion. Flight steering is issued by descending neurons, so the same
stimuli in the tethered-flight context may produce yaw that is dominated by the
command - which is what a behaviour-validated turn label needs. This probe runs a
few windows in both contexts and compares the command-to-yaw relationship.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.paths import ROOT, dump
from chatflypt.malecns import MaleCNS
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.environment import empty_environment, probe_environment, validate_objects
from chatflypt.embodied import EmbodiedSession, run_embodied
from chatflypt.intent_readout import measured_outcome
from chatflypt.physical_touch import presentation_span

REPORT = ROOT / 'reports/intent-readout'
PLACEMENT = ROOT / 'reports/unified-readout/conspecific-v4-placement.json'


def conditions(placement, strength=1.):
    """A few declared stimuli: an odor source, a pheromone source, a tap train."""
    yeast = validate_objects([{'id': 'probe_yeast', 'type': 'yeast', 'position_mm': [1., 0., .4],
                               'size_mm': [.5, .5, .5], 'range_mm': 7., 'strength': strength}])
    pheromone = validate_objects([{'id': 'probe_pheromone', 'type': 'pheromone',
                                   'position_mm': placement['local_sources']['pheromone']['left']['position_mm'],
                                   'size_mm': [.05, .05, .05],
                                   'range_mm': placement['local_sources']['pheromone']['left']['range_mm'],
                                   'strength': strength}])
    tap = [{'kind': 'touch', 'side': 'right', 'body_part': 'foreleg', 'strength': strength,
            'onset_ms': 50., 'duration_ms': 60., 'taps': 3, 'interval_ms': 140.}]
    return {'yeast': (yeast, []), 'pheromone_left': (pheromone, []), 'nudge_right': ([], tap)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--seed-base', type=int, default=990000)
    args = parser.parse_args()
    placement = json.loads(PLACEMENT.read_text(encoding='utf8'))
    brain, catalogue = MaleCNS(), build_catalogue(save=False)
    report = {'schema': 'chatflypt.flight-context-probe.v1', 'seeds': args.seeds,
              'conditions': [], 'note': 'Tethered flight fixes the thorax; flight steering comes from '
                                        'descending activity. No free-flight aerodynamics.'}
    for context in ('ground', 'tethered_flight'):
        session = EmbodiedSession(empty_environment(), context, f'probe-{context}')
        try:
            for name, (objects, presentations) in conditions(placement).items():
                environment = empty_environment()
                environment['objects'] = objects
                for index in range(args.seeds):
                    seed = args.base = args.seed_base + index
                    result = run_embodied(brain, brain.rows, catalogue, 'MaleCNS v1.0', [],
                                          environment, seed, context, duration_ms=500.,
                                          session=session, input_mode='environment_only',
                                          presentations=presentations)
                    body = result['body']
                    summary = body.get('command_summary') or {}
                    left = float((summary.get('left') or {}).get('max', 0.))
                    right = float((summary.get('right') or {}).get('max', 0.))
                    outcome = measured_outcome(body['trajectory'])
                    report['conditions'].append({
                        'context': context, 'condition': name, 'seed': seed,
                        'turn_command': {'left': left, 'right': right},
                        'measured_yaw_deg': outcome['yaw_change_deg'],
                        'measured_path_mm': outcome['path_length_mm'],
                        'body_status': body.get('status'), 'physics_warnings': (body.get('summary') or {}).get('physics_warnings')})
        finally:
            session.close()
    rows = report['conditions']
    for context in ('ground', 'tethered_flight'):
        subset = [row for row in rows if row['context'] == context]
        if not subset:
            continue
        yaw = np.asarray([row['measured_yaw_deg'] for row in subset])
        turned = [row for row in subset if abs(row['turn_command']['left'] - row['turn_command']['right']) > 0]
        report[f'{context}_summary'] = {
            'windows': len(subset), 'command_windows': len(turned),
            'abs_yaw_median': float(np.median(np.abs(yaw))), 'abs_yaw_max': float(np.abs(yaw).max()),
            'command_windows_abs_yaw_median': (float(np.median([abs(row['measured_yaw_deg']) for row in turned]))
                                               if turned else None)}
    REPORT.mkdir(parents=True, exist_ok=True)
    dump(REPORT / 'flight-context-probe.json', report)
    print(json.dumps({key: value for key, value in report.items() if key.endswith('_summary')},
                     ensure_ascii=False, indent=1))
    for row in rows:
        print(f"{row['context']:16s} {row['condition']:15s} L={row['turn_command']['left']:.3f} "
              f"R={row['turn_command']['right']:.3f} yaw={row['measured_yaw_deg']:7.3f} "
              f"path={row['measured_path_mm']:.3f} {row['body_status']}")


if __name__ == '__main__':
    main()
