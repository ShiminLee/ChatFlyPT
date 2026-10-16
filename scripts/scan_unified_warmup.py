"""Preserve evidence of fixed odor-source placement during original warmup."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from chatflypt.behavior_body import initialize_body
from chatflypt.environment import empty_environment, validate_objects, measure_environment_contacts
from chatflypt.paths import ROOT, dump

out = ROOT / 'reports/unified-readout/placement-checks-v3'
neutral, _ = initialize_body(42, 'ground', empty_environment(), inspection_contacts=True)
names = [s.name for s in neutral.fly.get_bodysegs_order()]
positions = neutral.sim.get_body_positions(neutral.fly.name).copy()
pose = neutral.sim.mj_data.qpos.copy()
result = []
for side, prefix, sign in [('left', 'l', 1), ('right', 'r', -1)]:
    receptor = positions[names.index(prefix + '_funiculus')]
    for offset in [[x, sign * .08, 0.] for x in (.1, .12, .14, .16, .18, .20)]:
        position = receptor + offset
        env = empty_environment()
        env['objects'] = validate_objects([{'id': 'local_source', 'type': 'yeast',
            'position_mm': position.tolist(), 'size_mm': [.05, .05, .05]}])
        body, _ = initialize_body(42, 'ground', env, inspection_contacts=True)
        contacts = measure_environment_contacts(body.sim.mj_model, body.sim.mj_data)
        delta = float(np.max(np.abs(body.sim.mj_data.qpos - pose)))
        row = {'side': side, 'position_mm': position.tolist(), 'offset_mm': offset,
            'contacts': contacts, 'max_qpos_difference': delta,
            'distance_mm': float(np.linalg.norm(offset))}
        result.append(row)
        print(json.dumps(row), flush=True)
        del body
dump(out / 'warmup-scan.json', result)
