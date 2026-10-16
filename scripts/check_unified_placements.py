"""Physical placement checks before freezing the v3 readout protocol."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from chatflypt.environment import empty_environment, probe_environment, validate_objects, field
from chatflypt.paths import ROOT, dump, sha256
from scripts.train_unified_readout import V3_PLACEMENT


def main():
    out = ROOT / 'reports/unified-readout/placement-checks-v3'
    neutral = probe_environment(empty_environment(), 42, 'ground')
    dump(out / 'neutral-probe-attempt3.json', neutral)
    names = neutral['body_names']
    positions = neutral['frame']['body_positions_mm']
    sites = {side + '_' + part: positions[names.index(prefix + '_' + part)]
             for side, prefix in (('left', 'l'), ('right', 'r'))
             for part in ('funiculus', 'arista')}
    eyes = neutral['vision']['eye_positions_mm']
    sources = {}; checks = []
    for part in ('funiculus', 'arista'):
        sources[part] = {}
        for side, sign in (('left', 1), ('right', -1)):
            # Closer .05 mm cubes collided or perturbed original warmup. Use
            # the first zero-qpos-change position in the saved warmup scan.
            offset = [.12, sign * .08, 0.] if part == 'funiculus' else [0., sign * .08, .04]
            center = (np.array(sites[side + '_' + part]) + offset).tolist()
            sources[part][side] = {'position_mm': center, 'range_mm': .20 if part == 'funiculus' else .16}
            if part != 'funiculus':
                continue
            objects = validate_objects([{'id': 'local_source', 'type': 'yeast',
                'position_mm': center, 'size_mm': [.05, .05, .05], 'range_mm': .20, 'strength': 1.}])
            env = empty_environment(); env['objects'] = objects
            probe = probe_environment(env, 42, 'ground')
            path = out / ('yeast-' + side + '-probe-attempt3.json'); dump(path, probe)
            hits = [c for c in probe['environment_contacts'] if c['object_id'] == 'local_source']
            difference = float(np.max(np.abs(np.asarray(probe['frame']['body_positions_mm']) - positions)))
            checks.append({'side': side, 'contacts': hits, 'max_initial_body_position_change_mm': difference,
                'record': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
            if hits or difference > 1e-9:
                raise ValueError('A local yeast source touched or altered the initial body; placement rejected')
    initial = {}
    for part, sides in sources.items():
        initial[part] = {}
        for side, spec in sides.items():
            obj = {'position_mm': spec['position_mm'], 'range_mm': spec['range_mm'],
                   'yaw_deg': 0., 'strength': 1.}
            values = {receiver: field(sites[receiver + '_' + part], obj) for receiver in ('left', 'right')}
            if values[side] <= 0 or values['right' if side == 'left' else 'left'] != 0:
                raise ValueError('Initial source field is not unilateral')
            initial[part][side] = values
    report = {'schema': 'chatflypt.unified-readout-placement.v1',
        'neutral_receptors_mm': sites, 'neutral_eye_cameras_mm': eyes,
        'eye_position_source': neutral['vision']['eye_position_source'],
        'local_sources': sources, 'verified_no_initial_yeast_contact': True,
        'yeast_geometry_mm': [.05, .05, .05], 'yeast_checks': checks,
        'initial_unilateral_fields': initial,
        'neutral_record': str((out / 'neutral-probe-attempt3.json').relative_to(ROOT)),
        'neutral_record_sha256': sha256(out / 'neutral-probe-attempt3.json'),
        'rationale': ['Odor uses actual funiculus, temperature uses actual arista.',
            'Fixed odor cube center: 0.12 mm forward and 0.08 mm outward from neutral funiculus (0.144 mm center distance; nearest cube point 0.110 mm). Closer checked placements were rejected for contact or altered warmup; engineering range 0.20 mm remains unilateral initially.',
            'Fixed non-solid thermal source center: 0.08 mm outward and 0.04 mm upward from neutral arista.',
            '0.05 mm yeast cube is checked to have no initial contact or body displacement; full trials can still bring a receptor into contact.',
            'Opposing eye shade uses actual compiled camera positions, not merged eye-body proxies.',
            'Movement can end, switch or combine receptor exposure. Targets always use delivered events.'],
        'biologically_calibrated': False}
    dump(V3_PLACEMENT, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
