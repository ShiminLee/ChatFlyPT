"""Export the unchanged Bad Apple Fly body, including head/mouth/wing joints.

Run only with .venv-body. The old legs-only geometry asset remains untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chatflypt.behavior_body import initialize_body, body_source, COMMAND_KEYS
from scripts.export_body_scene import (
    SCHEMA, export_geometries, frame, compare_to_mujoco, provenance, sha256,
)

DEFAULT_OUTPUT = ROOT / 'chatflypt/web/assets/neuromechfly-behavior-scene.json'


def build_scene():
    body, initialization = initialize_body(seed=1, context='ground')
    initialization['performed'] = True
    sim, fly = body.sim, body.fly
    geoms = export_geometries(sim, fly)
    initial = frame(sim, fly)
    initial['time_ms'] = 0.
    initial['commands'] = {key: 0. for key in COMMAND_KEYS}
    initial['neural_command_active'] = False
    initial['geom_positions_mm'] = sim.mj_data.geom_xpos[body.geometry_indices].tolist()
    initial['geom_rotation_matrices'] = sim.mj_data.geom_xmat[body.geometry_indices].tolist()
    checks = {'initial_after_upstream_warmup': compare_to_mujoco(sim, geoms, initial)}
    points = np.concatenate([
        np.asarray(geom['vertices']).reshape(-1, 3)
        @ sim.mj_data.geom_xmat[geom['geom_index']].reshape(3, 3).T
        + sim.mj_data.geom_xpos[geom['geom_index']] for geom in geoms])
    # Geometry-only validation: these are explicit synthetic controller inputs,
    # not a neural trial, training example or claimed biological response.
    validation_commands = {key: 0. for key in COMMAND_KEYS}
    validation_commands.update(left=.2, right=.3, antenna=.6, feeding=.7, song=.8)
    body.advance(30., validation_commands)
    dynamic = frame(sim, fly)
    dynamic['time_ms'] = body.elapsed * 1000
    checks['synthetic_controller_physical_30ms'] = compare_to_mujoco(sim, geoms, dynamic)
    checks['synthetic_controller_commands'] = validation_commands
    required_parts = ('c_head', 'c_rostrum', 'c_haustellum', 'l_wing', 'r_wing')
    ownership = {}
    for part in required_parts:
        geom = next(g for g in geoms if g['name'].endswith('/'+part))
        if geom['body_name'] != part:
            raise AssertionError(f'Articulated part {part} has unexpected owner {geom["body_name"]}')
        ownership[part] = {'body_index': geom['body_index'], 'body_name': geom['body_name']}
    checks['articulated_part_ownership'] = ownership
    checks['absent_upstream_model_parts'] = ['c_labellum: not present in the official skeleton or mesh; no invented joint']
    checks['physical_state_changed'] = not np.allclose(initial['body_positions_mm'], dynamic['body_positions_mm'], atol=1e-12)
    if not checks['physical_state_changed']:
        raise AssertionError('Dynamic geometry validation did not move the body')
    source = provenance()
    source.update(
        model='Official NeuroMechFly meshes with Bad Apple Fly full behavior articulation',
        factory='vendor/badapple-fly/backend/body.py Body()',
        badapple_reference=body_source(),
        export_script_sha256=sha256(Path(__file__)),
        frame_contract='Matches chatflypt.behavior_body body_names/body_positions_mm/body_quaternions_wxyz order; incompatible with old legs-only local geom poses.',
        limitations=[
            'The static initial frame follows upstream active posture warmup; it is not a neural response.',
            'Full behavior articulation and engineering programs do not establish biological validity.',
            'No aerodynamics; explicit tethered flight only.',
        ])
    return {
        'schema': SCHEMA, 'model_variant': 'badapple-fly-full-behavior',
        'units': 'millimeters', 'up_axis': 'z', 'quaternion_order': 'wxyz',
        'transform_order': 'world_vertex = body_position + R(body_quaternion) * (geom_local_position + R(geom_local_quaternion) * compiled_mesh_vertex)',
        'body_names': [b.name for b in fly.get_bodysegs_order()],
        'geoms': geoms, 'frame': initial,
        'joint_names': [j.name for j in fly.get_jointdofs_order()],
        'geom_indices': body.geometry_indices,
        'initialization': initialization,
        'bounds_mm': {'min': points.min(axis=0).tolist(), 'max': points.max(axis=0).tolist()},
        'ground': {'kind': 'plane', 'position_mm': [0, 0, 0], 'normal': [0, 0, 1]},
        'engine': {'nq': sim.mj_model.nq, 'nv': sim.mj_model.nv, 'nu': sim.mj_model.nu,
                   'timestep_ms': float(sim.timestep*1000)},
        'validation': checks, 'provenance': source,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    scene = build_scene()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scene, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'bytes': args.output.stat().st_size,
                      'bodies': len(scene['body_names']), 'geoms': len(scene['geoms']),
                      'engine': scene['engine'], 'validation': scene['validation']}, indent=2))


if __name__ == '__main__':
    main()
