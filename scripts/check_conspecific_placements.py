"""Physical placement checks for the conspecific-signal (v4) readout protocol.

Both new sources are non-solid engineering fields, so the checks that matter are
(a) the world's initial body pose is untouched, (b) no initial contact is
recorded, and (c) each source's initial field is unilateral at the actual
receptor site: funiculus for cVA (Or67d ORNs) and pedicel for song (Johnston's
organ). No field is calibrated to concentration or sound pressure.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from chatflypt.environment import empty_environment, probe_environment, validate_objects, field
from chatflypt.paths import ROOT, dump, sha256

OUT = ROOT / 'reports/unified-readout/conspecific-v4-placement.json'
CHECKS = ROOT / 'reports/unified-readout/placement-checks-v4'
SIZE_MM = [.05, .05, .05]
OFFSET_MM = .08
RANGE_MM = .20
SITES = {'pheromone': 'funiculus', 'song': 'pedicel'}


def main():
    CHECKS.mkdir(parents=True, exist_ok=True)
    neutral = probe_environment(empty_environment(), 42, 'ground')
    dump(CHECKS / 'neutral-probe.json', neutral)
    names = neutral['body_names']
    positions = neutral['frame']['body_positions_mm']
    sites = {side + '_' + part: positions[names.index(prefix + '_' + part)]
             for side, prefix in (('left', 'l'), ('right', 'r'))
             for part in ('funiculus', 'pedicel', 'arista')}
    sources = {'pheromone': {}, 'song': {}}
    checks = []
    for kind, part in SITES.items():
        for side, sign in (('left', 1), ('right', -1)):
            center = (np.array(sites[side + '_' + part]) + [0., sign * OFFSET_MM, 0.]).tolist()
            sources[kind][side] = {'position_mm': center, 'range_mm': RANGE_MM, 'receptor_site': side + '_' + part}
            objects = validate_objects([{'id': 'local_source', 'type': kind, 'position_mm': center,
                                        'size_mm': SIZE_MM, 'range_mm': RANGE_MM, 'strength': 1.}])
            environment = empty_environment()
            environment['objects'] = objects
            probe = probe_environment(environment, 42, 'ground')
            path = CHECKS / f'{kind}-{side}-probe.json'
            dump(path, probe)
            hits = [c for c in probe['environment_contacts'] if c['object_id'] == 'local_source']
            difference = float(np.max(np.abs(np.asarray(probe['frame']['body_positions_mm']) - positions)))
            checks.append({'kind': kind, 'side': side, 'contacts': hits,
                           'max_initial_body_position_change_mm': difference,
                           'record': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
            if hits:
                raise ValueError(f'{kind} source contacted the body at initialization; placement rejected')
            if difference > 1e-9:
                raise ValueError(f'{kind} source altered the initial body pose; placement rejected')
    initial = {}
    for kind, part in SITES.items():
        initial[kind] = {}
        for side, spec in sources[kind].items():
            obj = {'position_mm': spec['position_mm'], 'range_mm': spec['range_mm'],
                   'yaw_deg': 0., 'strength': 1.}
            values = {receiver: field(sites[receiver + '_' + part], obj)
                      for receiver in ('left', 'right')}
            other = 'right' if side == 'left' else 'left'
            if values[side] <= 0 or values[other] != 0:
                raise ValueError(f'{kind} initial field is not unilateral at the {part}')
            initial[kind][side] = values
    report = {'schema': 'chatflypt.conspecific-placement.v1',
        'neutral_receptors_mm': {k: v for k, v in sites.items()},
        'source_geometry_mm': SIZE_MM, 'source_offset_mm': OFFSET_MM, 'source_range_mm': RANGE_MM,
        'local_sources': sources, 'checks': checks, 'initial_unilateral_fields': initial,
        'neutral_record': str((CHECKS / 'neutral-probe.json').relative_to(ROOT)),
        'neutral_record_sha256': sha256(CHECKS / 'neutral-probe.json'),
        'rationale': [
            'cVA is sampled at the actual funiculus body position, where the antennal Or67d/DA1 ORNs sit; song is sampled at the actual pedicel body position, where Johnston\'s organ sits. Neither site substitutes for another part.',
            'Both sources are non-solid, so they cannot contact the body; the warmup pose is verified bit-identical and zero contacts are recorded.',
            'Offset 0.08 mm from the receptor gives an initial ipsilateral field of about 0.36 with the 0.20 mm engineering range; closer real cubes were rejected in the odor checks for contact, and this offset is not a calibrated concentration or sound pressure.',
            'The pedicel field is static at initialization; song timing (36 ms interpulse interval, 5 ms gate) is applied by the receptor encoder, not by moving the source.',
            'Movement can end, switch or combine receptor exposure; targets always require actually delivered events.'],
        'biologically_calibrated': False}
    dump(OUT, report)
    print(json.dumps({'schema': report['schema'], 'initial_fields': initial,
                      'checks': [{'kind': c['kind'], 'side': c['side'], 'contacts': len(c['contacts']),
                                  'pose_change_mm': c['max_initial_body_position_change_mm']} for c in checks]},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
