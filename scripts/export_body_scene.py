"""Export the existing official NeuroMechFly meshes for browser rendering.

Run with .venv-body/Scripts/python.exe scripts/export_body_scene.py. This uses
the same factory/world/reset as chatflypt.body_simulation and only bundled
simplified meshes. No full-resolution downloads or production services are used.
The exported local mesh is transformed by geom-local pose then body-world pose.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'chatflypt/web/assets/neuromechfly-scene.json'
SCHEMA = 'chatflypt.body-scene.v1'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quaternion_matrix(quat) -> np.ndarray:
    """WXYZ quaternion to a column-vector rotation matrix, independent of MuJoCo."""
    w, x, y, z = np.asarray(quat, dtype=float)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def make_scene_simulation():
    """Same public model construction as the body worker; no warming up."""
    from flygym import assets_dir
    from flygym.compose import FlatGroundWorld
    from flygym.simulation import Simulation
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain.common import make_locomotion_fly

    mesh_dir = assets_dir / 'model/neuromechfly/meshes/simplified_max2000faces'
    if not mesh_dir.is_dir() or not any(mesh_dir.glob('*.stl')):
        raise FileNotFoundError(f'Bundled official simplified meshes are missing: {mesh_dir}')
    fly = make_locomotion_fly()
    world = FlatGroundWorld()
    world.add_fly(fly, spawn_position=[0, 0, 1.0],
                  spawn_rotation=Rotation3D('quat', [1, 0, 0, 0]),
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    sim.reset()
    return sim, fly


def frame(sim, fly) -> dict:
    import mujoco

    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    names = [body.name for body in fly.get_bodysegs_order()]
    positions = sim.get_body_positions(fly.name).copy()
    rotations = sim.get_body_rotations(fly.name).copy()
    thorax = names.index('c_thorax')
    quat = rotations[thorax]
    w, x, y, z = quat
    return {'time_ms': float(sim.mj_data.time * 1000),
            'position_mm': positions[thorax].tolist(),
            'quaternion_wxyz': quat.tolist(),
            'yaw_rad': float(math.atan2(2*(w*z + x*y), 1-2*(y*y+z*z))),
            'body_positions_mm': positions.tolist(),
            'body_quaternions_wxyz': rotations.tolist(),
            'joint_angles_rad': sim.get_joint_angles(fly.name).tolist(),
            'feet_mm': [positions[names.index(f'{leg}_tarsus5')].tolist()
                        for leg in ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')]}


def export_geometries(sim, fly) -> list[dict]:
    import mujoco as mj

    model = sim.mj_model
    body_names = [body.name for body in fly.get_bodysegs_order()]
    # A static geom can be compiled into a different owner body (e.g. head
    # into thorax). Ownership must come from geom_bodyid, never from geom name.
    body_lookup = {}
    for index, segment in enumerate(fly.get_bodysegs_order()):
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY,
                              fly.bodyseg_to_mjcfbody[segment].name)
        if body_id >= 0:
            body_lookup[int(body_id)] = index
    geoms = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mj.mjtGeom.mjGEOM_MESH:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        if body_id not in body_lookup:
            raise ValueError(f'Mesh geom {geom_id} has no exported owner body: {body_id}')
        index = body_lookup[body_id]
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start, vertex_count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        face_start, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        vertices = model.mesh_vert[vertex_start:vertex_start+vertex_count].astype(np.float64)
        faces = model.mesh_face[face_start:face_start+face_count].copy()
        if not np.isfinite(vertices).all() or (faces < 0).any() or (faces >= vertex_count).any():
            raise ValueError(f'Invalid mesh arrays: {mesh_id}')
        geoms.append({
            'name': mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id),
            'geom_index': geom_id, 'mesh_index': mesh_id,
            'mesh_name': mj.mj_id2name(model, mj.mjtObj.mjOBJ_MESH, mesh_id),
            'body_name': body_names[index], 'body_index': index,
            'mujoco_body_id': body_id,
            'local_position_mm': model.geom_pos[geom_id].tolist(),
            'local_quaternion_wxyz': model.geom_quat[geom_id].tolist(),
            # Compiled mesh_vert already includes scale and compiler recentering.
            # Do not apply mesh_pos/mesh_quat/mesh_scale for a second time.
            'vertices': np.round(vertices, 8).ravel().tolist(),
            'faces': faces.ravel().tolist(),
            'color': model.geom_rgba[geom_id].astype(float).tolist(),
            'vertex_count': vertex_count, 'face_count': face_count,
        })
    return geoms


def compare_to_mujoco(sim, geoms, exported_frame) -> dict:
    """Check the browser composition against actual compiled geom_xpos/xmat."""
    positions = np.asarray(exported_frame['body_positions_mm'])
    rotations = np.asarray(exported_frame['body_quaternions_wxyz'])
    max_position = max_rotation = max_vertex = 0.
    for geom in geoms:
        index, gid = geom['body_index'], geom['geom_index']
        body_rot = quaternion_matrix(rotations[index])
        predicted_pos = positions[index] + body_rot @ np.asarray(geom['local_position_mm'])
        predicted_rot = body_rot @ quaternion_matrix(geom['local_quaternion_wxyz'])
        actual_pos = sim.mj_data.geom_xpos[gid]
        actual_rot = sim.mj_data.geom_xmat[gid].reshape(3, 3)
        max_position = max(max_position, float(np.max(np.abs(predicted_pos - actual_pos))))
        max_rotation = max(max_rotation, float(np.max(np.abs(predicted_rot - actual_rot))))
        vertices = np.asarray(geom['vertices']).reshape(-1, 3)
        mesh = geom['mesh_index']
        va, vn = int(sim.mj_model.mesh_vertadr[mesh]), int(sim.mj_model.mesh_vertnum[mesh])
        native_vertices = sim.mj_model.mesh_vert[va:va+vn].astype(float)
        actual_world = native_vertices @ actual_rot.T + actual_pos
        exported_world = vertices @ predicted_rot.T + predicted_pos
        max_vertex = max(max_vertex, float(np.max(np.abs(actual_world - exported_world))))
    passed = max_position < 1e-10 and max_rotation < 1e-10 and max_vertex < 2e-8
    if not passed:
        raise AssertionError(f'Geometry pose mismatch: {max_position}, {max_rotation}, {max_vertex}')
    return {'time_ms': exported_frame['time_ms'], 'geoms_checked': len(geoms),
            'max_position_error_mm': max_position, 'max_rotation_matrix_error': max_rotation,
            'max_world_vertex_error_mm': max_vertex, 'passed': passed}


def provenance() -> dict:
    import flygym
    import mujoco

    lock = json.loads((ROOT / 'research/manifests/sources.lock.json').read_text(encoding='utf-8'))
    source = next(item for item in lock if item['id'] == 'flygym')
    repo = ROOT / source['local_path']
    package = repo / 'src/flygym'
    if Path(flygym.__file__).resolve().parent != package.resolve():
        raise RuntimeError('FlyGym import must resolve to the recorded local official source tree')
    mesh_dir = package / 'assets/model/neuromechfly/meshes/simplified_max2000faces'
    paths = [repo / 'LICENSE', repo / 'src/flygym_demo/complex_terrain/common.py',
             package / 'compose/fly/neuromechfly.py', package / 'compose/fly/base_fly.py',
             package / 'assets/model/neuromechfly/rigging.yaml',
             package / 'assets/model/neuromechfly/mujoco_globals.yaml',
             *sorted(mesh_dir.glob('*.stl'))]
    badapple_snapshot = ROOT / 'research/snapshots/badapple-fly-tree.json'
    badapple = json.loads(badapple_snapshot.read_text(encoding='utf-8')) if badapple_snapshot.exists() else {}
    return {
        'model': 'Official NeuroMechFly / FlyGym make_locomotion_fly',
        'repository': source['repository'], 'commit': source['commit'],
        'archive_sha256': source['archive_sha256'],
        'flygym_version': importlib.metadata.version('flygym'), 'mujoco_version': mujoco.__version__,
        'factory': 'flygym_demo.complex_terrain.common.make_locomotion_fly()',
        'mesh_resolution': 'simplified_max2000faces',
        'mesh_specimen': 'Female NeuroMechFly micro-CT body; not registered to the MaleCNS individual.',
        'source_files_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path) for path in paths},
        'license': {'spdx': 'Apache-2.0', 'source_path': 'vendor/flygym/LICENSE',
                    'text': (repo / 'LICENSE').read_text(encoding='utf-8')},
        'badapple_reference': {'repository': 'https://github.com/kevinlinxc/badapple-fly',
            'commit': badapple.get('commit'), 'path': 'backend/body.py geometry()/frame()',
            'scope': 'Reference for browser geometry serialization. Geometry and dynamics are loaded from official FlyGym, not the alternate Bad Apple body/controller.'},
        'downloads_performed': False,
        'export_script_sha256': sha256(Path(__file__)),
        'frame_contract': 'Matches chatflypt.body_simulation body_names/body_positions_mm/body_quaternions_wxyz order.',
        'limitations': ['Geometry visualization does not validate the neural-to-motor engineering controller.',
                       'Leg-only official model; fixed non-leg parts are not invented as animated joints.'],
    }


def build_scene() -> dict:
    import mujoco

    sim, fly = make_scene_simulation()
    body_names = [body.name for body in fly.get_bodysegs_order()]
    geoms = export_geometries(sim, fly)
    initial = frame(sim, fly)
    checks = {'initial_reset': compare_to_mujoco(sim, geoms, initial)}
    points = []
    for geom in geoms:
        gid = geom['geom_index']
        points.append(np.asarray(geom['vertices']).reshape(-1, 3) @ sim.mj_data.geom_xmat[gid].reshape(3, 3).T
                      + sim.mj_data.geom_xpos[gid])
    points = np.concatenate(points)
    # A real second physical state checks movable joints and changing global
    # orientation. This isolated 20 ms gravity/contact run has no neural inputs
    # and is validation only; it is not a user-visible behavioural trajectory.
    sim.mj_model.actuator_gainprm[:] = 0
    sim.mj_model.actuator_biasprm[:] = 0
    sim.mj_data.ctrl[:] = 0
    for _ in range(round(.020 / sim.mj_model.opt.timestep)):
        sim.step()
    dynamic = frame(sim, fly)
    checks['physical_20ms'] = compare_to_mujoco(sim, geoms, dynamic)
    checks['physical_state_changed'] = not np.allclose(initial['body_positions_mm'], dynamic['body_positions_mm'], atol=1e-12)
    if not checks['physical_state_changed']:
        raise AssertionError('Alignment validation did not include a changed physical state')
    return {'schema': SCHEMA, 'units': 'millimeters', 'up_axis': 'z',
            'quaternion_order': 'wxyz',
            'transform_order': 'world_vertex = body_position + R(body_quaternion) * (geom_local_position + R(geom_local_quaternion) * compiled_mesh_vertex)',
            'body_names': body_names, 'geoms': geoms, 'frame': initial,
            'joint_names': [joint.name for joint in fly.get_jointdofs_order()],
            'bounds_mm': {'min': points.min(axis=0).tolist(), 'max': points.max(axis=0).tolist()},
            'ground': {'kind': 'plane', 'position_mm': [0, 0, 0], 'normal': [0, 0, 1]},
            'engine': {'nq': sim.mj_model.nq, 'nv': sim.mj_model.nv, 'nu': sim.mj_model.nu,
                       'timestep_ms': float(sim.mj_model.opt.timestep * 1000)},
            'validation': checks, 'provenance': provenance()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    scene = build_scene()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scene, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'bytes': args.output.stat().st_size,
                      'bodies': len(scene['body_names']), 'geoms': len(scene['geoms']),
                      'vertices': sum(g['vertex_count'] for g in scene['geoms']),
                      'faces': sum(g['face_count'] for g in scene['geoms']),
                      'validation': scene['validation']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
