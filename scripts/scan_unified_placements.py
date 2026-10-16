"""Read-only pose scan of candidate odor-source geometry; no neural simulation."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from chatflypt.behavior_body import initialize_body
from chatflypt.environment import empty_environment, validate_objects
from chatflypt.paths import ROOT, dump


def main():
    env = empty_environment()
    env['objects'] = validate_objects([{'id': 'local_source', 'type': 'yeast',
        'position_mm': [10., 10., 10.], 'size_mm': [.05, .05, .05]}])
    body, initialization = initialize_body(42, 'ground', env, inspection_contacts=True)
    model, data = body.sim.mj_model, body.sim.mj_data
    mujoco.mj_forward(model, data)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'env_local_source')
    if geom < 0:
        raise ValueError('No source geometry')
    names = [segment.name for segment in body.fly.get_bodysegs_order()]
    points = body.sim.get_body_positions(body.fly.name)
    candidate_rows = []
    for side, prefix, sign in (('left', 'l', 1), ('right', 'r', -1)):
        receptor = points[names.index(prefix + '_funiculus')]
        for dx in np.arange(-.10, .101, .02):
            for dy in np.arange(0., .141, .02):
                for dz in np.arange(-.10, .141, .02):
                    offset = np.array([dx, sign * dy, dz])
                    distance = np.linalg.norm(offset)
                    if not .07 <= distance <= .15:
                        continue
                    model.geom_pos[geom] = receptor + offset
                    mujoco.mj_forward(model, data)
                    distances = []
                    for c in data.contact:
                        if geom in c.geom:
                            distances.append(float(c.dist))
                    # Margin proximity may exert a force too: reject the entire
                    # potential contact list, not only dist <= 0 surfaces.
                    if distances:
                        continue
                    candidate_rows.append({'side': side, 'offset_mm': offset.tolist(),
                        'distance_mm': float(distance), 'position_mm': model.geom_pos[geom].tolist(),
                        'potential_contact_count': len(distances)})
    candidate_rows.sort(key=lambda row: row['distance_mm'])
    result = {'schema': 'chatflypt.placement-geometry-scan.v1',
        'initialization': initialization, 'candidates': candidate_rows,
        'method': 'Moved fixed source geometry in unchanged neutral MuJoCo pose and called mj_forward; no integration or neural activity.'}
    dump(ROOT / 'reports/unified-readout/placement-checks-v3/geometry-scan.json', result)
    print(json.dumps({side: [row for row in candidate_rows if row['side'] == side][:12]
                      for side in ('left', 'right')}), flush=True)


if __name__ == '__main__':
    main()
