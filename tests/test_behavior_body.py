"""Original programs, causal replay, passive forces, and articulated mesh poses."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from chatflypt.behavior_body import run_behavior_body, COMMAND_KEYS, GEOMETRY_ASSET
from chatflypt.behavior_readout import build_behavior_drive
from scripts.export_body_scene import quaternion_matrix

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'chatflypt/web/assets/neuromechfly-behavior-scene.json'
PYTHON = ROOT / '.venv-body' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def make_drive(*, silent=False, last_bin=False, context='ground'):
    # Synthetic neural records exercise named controller pathways; they are
    # test fixtures, not claims about a real sensory experiment or behavior.
    types = ('DNg100', 'DNg100', 'MN9', 'pIP10', 'DNg62', 'DNg02_a')
    rows = [{'type': kind, 'somaSide': 'R' if i == 1 else 'L'} for i, kind in enumerate(types)]
    indices = np.array([] if silent else [i for i in range(6) for _ in range(2)], dtype=np.int32)
    steps = np.array([] if silent else [401 if last_bin else 1, 450 if last_bin else 50]*6, dtype=np.int32)
    trial = SimpleNamespace(counts=np.bincount(indices, minlength=6), spike_ids=indices,
                            spike_steps=steps, duration_ms=60., seed=84217)
    return build_behavior_drive(trial, np.arange(6), rows, dataset='MaleCNS v1.0',
                                context=context, bin_ms=20., smoothing_ms=0.)


def require_body():
    if not PYTHON.is_file():
        pytest.skip('Isolated .venv-body is not installed')


def test_zero_and_unavailable_final_bin_do_not_generate_active_trajectory(tmp_path):
    for drive in (make_drive(silent=True), make_drive(last_bin=True)):
        result = run_behavior_body(drive, output_dir=tmp_path)
        assert result['status'] == 'no_motor_activity'
        assert not result['physics_run'] and result['trajectory'] == []
        assert result['initialization']['performed'] is False
        assert json.loads((tmp_path/'behavior-drive.json').read_text(encoding='utf-8')) == drive
        assert result['source_drive_artifact'] == str((tmp_path/'behavior-drive.json').resolve())
    # A command first available at the truncation endpoint must not leak back.
    result = run_behavior_body(make_drive(), max_duration_ms=20.)
    assert result['status'] == 'no_motor_activity' and result['truncated']


def test_commands_cannot_bypass_neural_formula_validation():
    drive = make_drive(silent=True)
    drive['commands'][0]['feeding'] = 1.
    result = run_behavior_body(drive)
    assert result['status'] == 'error' and not result['physics_run']
    assert not result['trajectory']


@pytest.fixture(scope='module')
def physical():
    require_body()
    result = run_behavior_body(make_drive(), sample_ms=10.)
    assert result['status'] == 'ok', result
    return result


def test_actual_body_reuses_programs_and_discloses_warmup_and_local_feedback(physical):
    result = physical
    assert result['physics_run'] and result['duration_ms'] == 60.
    assert result['controller_id'] == 'badapple_hybrid_v1'
    assert result['geometry_asset'] == GEOMETRY_ASSET
    assert result['initialization']['warmup_ms'] == pytest.approx(50.)
    assert result['initialization']['warmup_active_posture_control']
    assert result['initialization']['constructor_state_discarded']
    assert result['summary']['finite_state'] and result['summary']['physics_warnings'] == 0
    assert result['controls']['neural_feedback'] is False
    assert result['controls']['active_posture_hold_when_commands_zero']
    assert result['controls']['max_force_during_zero_commands'] > 0
    assert result['trajectory'][0]['time_ms'] == 0
    for frame in result['trajectory']:
        assert set(frame['commands']) == set(COMMAND_KEYS)
        assert np.isfinite(frame['body_positions_mm']).all()
        if frame['time_ms'] <= 20.:
            assert not frame['neural_command_active']
            assert frame['source_command_bin'] == -1
        else:
            assert frame['command_available_at_ms'] < frame['time_ms']
    assert any(frame['commands']['feeding'] > 0 for frame in result['trajectory'])
    assert any(frame['commands']['song'] > 0 for frame in result['trajectory'])


def test_mouth_head_wing_meshes_follow_actual_mujoco_world_pose(physical):
    scene = json.loads(SCENE.read_text(encoding='utf-8'))
    assert scene['schema'] == 'chatflypt.body-scene.v1'
    assert scene['frame']['time_ms'] == 0.
    assert scene['initialization']['warmup_ms'] == pytest.approx(50.)
    assert scene['body_names'] == physical['body_names']
    assert scene['geom_indices'] == physical['geom_indices']
    required = ('c_head', 'c_rostrum', 'c_haustellum', 'l_wing', 'r_wing')
    selected = [g for g in scene['geoms'] if g['name'].split('/')[-1] in required]
    assert len(selected) == len(required)
    for geom in selected:
        assert geom['body_name'] == geom['name'].split('/')[-1]
    for frame in physical['trajectory']:
        positions = np.asarray(frame['body_positions_mm'])
        rotations = np.asarray(frame['body_quaternions_wxyz'])
        for geom in scene['geoms']:
            index = geom['body_index']
            rotation = quaternion_matrix(rotations[index])
            expected_position = positions[index] + rotation @ geom['local_position_mm']
            expected_rotation = rotation @ quaternion_matrix(geom['local_quaternion_wxyz'])
            j = physical['geom_indices'].index(geom['geom_index'])
            np.testing.assert_allclose(expected_position, frame['geom_positions_mm'][j], rtol=0, atol=1e-10)
            np.testing.assert_allclose(expected_rotation.ravel(), frame['geom_rotation_matrices'][j], rtol=0, atol=1e-10)
    # Mouth and wing states change physically; positions are not only labels.
    first, last = physical['trajectory'][0], physical['trajectory'][-1]
    for part in ('c_rostrum', 'c_haustellum', 'l_wing'):
        i = physical['body_names'].index(part)
        assert not np.allclose(first['body_quaternions_wxyz'][i], last['body_quaternions_wxyz'][i], atol=1e-8)
    assert scene['validation']['synthetic_controller_physical_30ms']['passed']


def test_passive_control_disables_every_actuator_after_identical_warmup(physical):
    require_body()
    result = run_behavior_body(make_drive(), passive_control=True, sample_ms=10.)
    assert result['status'] == 'passive_control', result
    assert result['physics_run'] and result['duration_ms'] == 60.
    assert result['controls']['all_actuators_disabled_during_replay']
    assert result['controls']['powered_steps'] == 0
    assert result['controls']['max_actuator_force_model_units'] == 0
    assert not result['controls']['active_posture_hold_when_commands_zero']
    assert result['initialization']['warmup_active_posture_control']
    for frame in result['trajectory']:
        assert frame['max_actuator_force_model_units'] == 0
        assert not any(frame['commands'].values())
    np.testing.assert_allclose(result['trajectory'][0]['body_positions_mm'],
                               physical['trajectory'][0]['body_positions_mm'], atol=1e-12, rtol=0)


def test_tethered_context_is_explicit_and_remains_fixed_without_aerodynamics():
    require_body()
    result = run_behavior_body(make_drive(context='tethered_flight'), sample_ms=10.)
    assert result['status'] == 'ok', result
    assert result['context'] == 'tethered_flight'
    assert result['controls']['kinematic_tether'] and not result['controls']['aerodynamics']
    for frame in result['trajectory']:
        np.testing.assert_allclose(frame['position_mm'], [0., 0., 2.], atol=1e-12, rtol=0)
        assert frame['commands']['left'] == frame['commands']['right'] == 0.
    assert any(frame['commands']['flight'] > 0 for frame in result['trajectory'])


def test_causal_adapter_matches_unmodified_source_step_for_step(physical):
    require_body()
    code = r'''
import json, sys
import numpy as np
from chatflypt.behavior_body import initialize_body, COMMAND_KEYS
request=json.load(sys.stdin)
drive=request['drive']; actual=request['actual']
body,_=initialize_body(drive['seed'],drive['context'])
times=np.asarray(drive['available_at_ms'])
dt=body.sim.timestep*1000
sample=round(10./dt)
frames=[body.sim.get_joint_angles(body.fly.name).tolist()]
for step in range(round(60./dt)):
    i=int(np.searchsorted(times,step*dt+1e-8,side='right'))-1
    command=drive['commands'][i] if i>=0 else {k:0. for k in COMMAND_KEYS}
    body.advance(dt,command)
    if (step+1)%sample==0:
        frames.append(body.sim.get_joint_angles(body.fly.name).tolist())
np.testing.assert_allclose(frames,[f['joint_angles_rad'] for f in actual['trajectory']],atol=1e-12,rtol=0)
print(json.dumps({'frames':len(frames),'passed':True}))
'''
    done = subprocess.run([str(PYTHON), '-c', code], cwd=ROOT, capture_output=True,
                          input=json.dumps({'drive': make_drive(), 'actual': physical}),
                          text=True, encoding='utf-8', timeout=45,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert done.returncode == 0, done.stdout+'\n'+done.stderr
    assert json.loads(done.stdout)['passed']
