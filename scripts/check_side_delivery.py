"""Probe: does a declared-side following source reach one antenna only, in a real session?

The anchored field model was fixed to honour ``anchor_side`` (the source keeps its
declared offset from its own receptor). The unit test checks the pure field
function; this probe checks the whole path in an actual embodied session: measured
receptor poses, measured events, counted per side. It is an engineering check of
the stimulus geometry, not a neural or behavioural claim, and it never writes into
a corpus.

Usage::

    python scripts/check_side_delivery.py --kinds yeast,cold --seeds 1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from chatflypt.environment import empty_environment, validate_objects
from chatflypt.embodied import EmbodiedSession, run_embodied
from chatflypt.malecns import MaleCNS
from chatflypt.paths import ROOT, dump
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.unified_readout import SEMANTIC_GROUPS

REPORT = ROOT / 'reports/unified-readout'
V3_PLACEMENT = REPORT / 'retinal-axes-v3-placement.json'
V4_PLACEMENT = REPORT / 'conspecific-v4-placement.json'
# Same declared geometry as the v9 elicitation conditions.
REACH = {'yeast': ('funiculus', .16), 'heat': ('arista', .12), 'cold': ('arista', .12)}


def side_event_counts(samples, events, catalogue, channel):
    """Measured events per side for one channel's receptor populations."""
    times = np.asarray(events['event_steps']) * float(events['dt_ms'])
    ids = np.asarray(events['event_neuron_ids']).astype(str)
    spec = catalogue['channels'][channel]
    totals = {'left': 0, 'right': 0}
    per_sample = []
    for sample in samples:
        active = (times >= sample['time_ms']) & (times < sample['until_ms'])
        delivered = ids[active]
        row = {}
        for side in ('left', 'right'):
            population = spec.get('side_ids', {}).get(side, [])
            row[side] = int(np.isin(delivered, np.asarray(population, dtype=str)).sum())
        per_sample.append(row)
        totals['left'] += row['left']
        totals['right'] += row['right']
    return totals, per_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kinds', default='yeast,heat,cold')
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--seed-base', type=int, default=991000)
    args = parser.parse_args()
    local = json.loads(V3_PLACEMENT.read_text(encoding='utf8'))['local_sources']
    conspecific = json.loads(V4_PLACEMENT.read_text(encoding='utf8'))
    receptors = conspecific['neutral_receptors_mm']
    brain, catalogue = MaleCNS(), build_catalogue(save=False)
    rows = []
    for kind in [name for name in args.kinds.split(',') if name]:
        site, reach = REACH[kind]
        for side in ('left', 'right'):
            spec = local[site][side]
            environment = empty_environment()
            environment['objects'] = validate_objects([{
                'id': 'local_source', 'type': kind, 'position_mm': spec['position_mm'],
                'size_mm': [.05, .05, .05], 'range_mm': reach, 'strength': 1.,
                'field': 'anchored', 'anchor': site,
                'anchor_origin_mm': receptors[side + '_' + site], 'anchor_side': side}])
            session = EmbodiedSession(environment, 'ground', f'side-probe-{kind}-{side}')
            try:
                for index in range(args.seeds):
                    result = run_embodied(brain, brain.rows, catalogue, 'MaleCNS v1.0', [],
                                          environment, args.seed_base + index, 'ground', duration_ms=500.,
                                          session=session, input_mode='environment_only', presentations=[])
                    totals, per_sample = side_event_counts(result['feedback']['samples'], result['events'],
                                                           catalogue, 'mc_' + kind)
                    group = SEMANTIC_GROUPS.get('mc_' + kind, kind)
                    own = totals[side]
                    other = totals['left' if side == 'right' else 'right']
                    rows.append({'kind': kind, 'declared_side': side, 'seed': args.seed_base + index,
                                 'own_side_events': own, 'other_side_events': other,
                                 'strictly_one_sided': bool(own > 0 and other == 0),
                                 'per_sample': per_sample, 'group': group})
                    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
            finally:
                session.close()
    summary = {}
    for row in rows:
        entry = summary.setdefault(f"{row['kind']}:{row['declared_side']}",
                                   {'windows': 0, 'one_sided': 0, 'own_events': 0, 'other_events': 0})
        entry['windows'] += 1
        entry['one_sided'] += int(row['strictly_one_sided'])
        entry['own_events'] += row['own_side_events']
        entry['other_events'] += row['other_side_events']
    report = {'schema': 'chatflypt.side-delivery-probe.v1',
              'scope': 'Engineering check of the declared-side following source in a real session; '
                       'no neural or behavioural claim, nothing written to a corpus.',
              'reach_mm': {kind: reach for kind, (_, reach) in REACH.items()},
              'rows': rows, 'summary': summary,
              'biologically_calibrated': False}
    REPORT.mkdir(parents=True, exist_ok=True)
    dump(REPORT / 'side-delivery-probe.json', report)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
