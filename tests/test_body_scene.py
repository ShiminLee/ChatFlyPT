"""The browser mesh must follow the same body frames as the physical worker."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'chatflypt/web/assets/neuromechfly-scene.json'


def test_scene_has_valid_meshes_and_explicit_owner_body_mapping():
    scene = json.loads(SCENE.read_text(encoding='utf-8'))
    assert scene['schema'] == 'chatflypt.body-scene.v1'
    assert scene['units'] == 'millimeters' and scene['up_axis'] == 'z'
    assert scene['quaternion_order'] == 'wxyz'
    names = scene['body_names']
    assert len(names) == len(set(names))
    initial = scene['frame']
    assert initial['time_ms'] == 0
    assert np.asarray(initial['body_positions_mm']).shape == (len(names), 3)
    assert np.asarray(initial['body_quaternions_wxyz']).shape == (len(names), 4)
    for geom in scene['geoms']:
        vertices = np.asarray(geom['vertices']).reshape(-1, 3)
        faces = np.asarray(geom['faces']).reshape(-1, 3)
        assert len(vertices) == geom['vertex_count'] and len(faces) == geom['face_count']
        assert np.isfinite(vertices).all()
        assert np.issubdtype(faces.dtype, np.integer) and faces.min() >= 0 and faces.max() < len(vertices)
        assert names[geom['body_index']] == geom['body_name']
        assert np.isclose(np.linalg.norm(geom['local_quaternion_wxyz']), 1., atol=1e-12)
    # MuJoCo fused the static head geom into the thorax owner body. Matching by
    # geom name would silently attach it to the wrong body frame.
    head = next(g for g in scene['geoms'] if g['name'].endswith('/c_head'))
    assert head['body_name'] == 'c_thorax'
    assert scene['provenance']['license']['spdx'] == 'Apache-2.0'
    assert 'Apache License' in scene['provenance']['license']['text']
    assert scene['provenance']['downloads_performed'] is False
    assert scene['validation']['initial_reset']['passed']
    assert scene['validation']['physical_20ms']['passed']


def test_actual_worker_frames_place_exported_geoms_at_mujoco_world_poses():
    python = ROOT / '.venv-body' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        pytest.skip('The isolated official body environment is not installed')
    code = r'''
import json
import numpy as np
from pathlib import Path
from scripts.export_body_scene import make_scene_simulation, frame, compare_to_mujoco
from chatflypt.body_simulation import _simulate_physics, SCHEMA, LEGS
scene = json.loads(Path('chatflypt/web/assets/neuromechfly-scene.json').read_text(encoding='utf-8'))
drive = {'schema': SCHEMA, 'dataset': 'MaleCNS v1.0', 'legs': list(LEGS),
 'duration_ms': 20., 'seed': 83107, 'available_at_ms': [10., 20.],
 'amplitudes': [[0.]*6, [0.]*6], 'motor_spikes': 0,
 'reference_rate_hz': 20., 'smoothing_ms': 30., 'bin_ms': 10.}
result = _simulate_physics({'drive': drive, 'duration_ms': 20., 'sample_ms': 20.,
 'passive_control': True, 'output_dir': None, 'render': False})
assert result['physics_run'] and result['status'] == 'passive_control'
assert result['body_names'] == scene['body_names']
for field in ['body_positions_mm', 'body_quaternions_wxyz', 'joint_angles_rad']:
 np.testing.assert_allclose(result['trajectory'][0][field], scene['frame'][field], atol=1e-12, rtol=0)
sim, fly = make_scene_simulation()
frame(sim, fly)
initial = compare_to_mujoco(sim, scene['geoms'], result['trajectory'][0])
sim.mj_model.actuator_gainprm[:] = 0
sim.mj_model.actuator_biasprm[:] = 0
sim.mj_data.ctrl[:] = 0
for _ in range(round(.020 / sim.mj_model.opt.timestep)):
 sim.step()
frame(sim, fly)
dynamic = compare_to_mujoco(sim, scene['geoms'], result['trajectory'][-1])
print(json.dumps({'initial': initial, 'actual_worker_20ms': dynamic}))
'''
    done = subprocess.run([str(python), '-c', code], cwd=ROOT, capture_output=True,
                          text=True, encoding='utf-8', timeout=45,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert done.returncode == 0, done.stdout + '\n' + done.stderr
    checked = json.loads(done.stdout)
    assert checked['initial']['passed'] and checked['actual_worker_20ms']['passed']
    assert checked['actual_worker_20ms']['max_world_vertex_error_mm'] < 2e-8
