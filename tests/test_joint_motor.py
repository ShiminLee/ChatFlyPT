"""Causal anatomical projection and preserved abstentions, not biological validation."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from chatflypt.behavior_readout import build_behavior_drive, _validate_behavior_drive
from chatflypt.joint_motor import CHANNELS, channel_for
from chatflypt.behavior_body import run_behavior_body


def build(indices=(0,), steps=(199,), *, excluded=(), kind='Ti flexor MN', duration=100.):
    rows=[dict(type=kind, somaSide=side, somaNeuromere='T1', exitNerve='ProLN', superclass='vnc_motor')
          for side in ('R','L')]
    t=SimpleNamespace(counts=np.bincount(indices,minlength=2),spike_ids=np.array(indices,dtype=np.int32),
        spike_steps=np.array(steps,dtype=np.int32),duration_ms=duration,seed=491701)
    return build_behavior_drive(t,[801,802],rows,dataset='MaleCNS v1.0',driven_indices=excluded)


def test_correct_leg_causal_bins_and_directly_driven_exclusion():
    d=build(indices=(0,1),steps=(199,200));j=d['joint_motor']
    assert j['spike_counts'][0][CHANNELS.index('rf_flexor')]==1
    assert j['spike_counts'][0][CHANNELS.index('lf_flexor')]==0
    assert j['offsets_rad'][0]['rf']>0 and j['offsets_rad'][0]['lf']==0
    assert j['offsets_rad'][1]['lf']>0
    assert d['has_joint_command_in_window'] and not d['has_command_in_window']
    excluded=build(excluded=(0,))
    assert not excluded['has_joint_command_in_window']
    assert excluded['joint_motor']['motor_spikes']==0


def test_nerve_segment_side_and_unknown_function_are_not_guessed():
    base=dict(type='Ti extensor MN',somaSide='R',somaNeuromere='T2',exitNerve='MesoLN',superclass='vnc_motor')
    assert channel_for(base)=='rm_extensor'
    for changes in ({'exitNerve':'ProLN'},{'somaSide':''},{'somaNeuromere':''},{'type':'Fe reductor MN'},{'superclass':'sensory'}):
        assert channel_for(dict(base,**changes)) is None
    d=build(kind='Fe reductor MN')
    assert not d['has_joint_command_in_window']
    assert d['motor_coverage']['unmapped'][0]['type']=='Fe reductor MN'
    assert '独立' in d['motor_coverage']['unmapped'][0]['reason']


@pytest.mark.parametrize('change',[
    lambda d:d['joint_motor']['offsets_rad'][0].__setitem__('rf',.5),
    lambda d:d['joint_motor']['channels'][6]['cells'][0].__setitem__('exitNerve','MetaLN'),
    lambda d:d['joint_motor'].__setitem__('gain_rad',5.),
    lambda d:d['joint_motor']['population_rate_hz'][0].__setitem__(6,999.),
    lambda d:d['joint_motor']['channels'][6].__setitem__('neuron_ids',['802']),
])
def test_forged_joint_drive_is_rejected(change):
    d=build(excluded=(1,));change(d)
    with pytest.raises(ValueError):_validate_behavior_drive(d)


def test_last_bin_cannot_act_in_past():
    d=build(steps=(999,))
    assert not d['has_joint_command_in_window']
    assert run_behavior_body(d)['status']=='no_motor_activity'


def test_real_physics_changes_only_after_available_motor_spikes(tmp_path):
    d=build(indices=(0,0,0),steps=(199,200,300),duration=100.)
    environment={'schema':'chatflypt.environment.v1','revision':0,'objects':[]}
    body=run_behavior_body(d,environment=environment,output_dir=tmp_path/'active')
    assert body['status']=='ok',body
    zero=build(indices=(),steps=(),duration=100.)
    # run_behavior_body skips zero commands with no scene objects; a persistent
    # worker also records the paired zero-command posture/physics control.
    from chatflypt.body_feedback import FeedbackBody
    worker=FeedbackBody({'schema':'chatflypt.environment.v1','revision':0,'objects':[]},491701,'ground',tmp_path/'paired')
    try:
        worker.step(zero);paired=worker.finish()
    finally:worker.close()
    a=np.array([f['joint_angles_rad'] for f in body['trajectory']])
    b=np.array([f['joint_angles_rad'] for f in paired['trajectory']])
    np.testing.assert_array_equal(a[:3],b[:3]) # 0,10,20ms: first command available at20ms.
    assert np.max(np.abs(a[3:]-b[3:]))>1e-5
    assert body['trajectory'][3]['joint_motor_offsets_rad']['rf']>0
    assert body['controls']['powered_steps']>0 and paired['controls']['powered_steps']==0
    assert body['summary']['finite_state'] and not body['summary']['physics_warnings']


def test_all_six_joint_signs_match_geometric_flexion():
    import subprocess
    from chatflypt.behavior_body import ROOT
    code='''
import numpy as np, mujoco
from chatflypt.behavior_body import initialize_body
from chatflypt.joint_motor import LEGS
b,_=initialize_body();m=b.sim.mj_model;d=b.sim.mj_data
names=[s.name for s in b.fly.get_bodysegs_order()]
def included_angle(leg):
    p=b.sim.get_body_positions('fly');k=p[names.index(leg+'_tibia')]
    a=p[names.index(leg+'_trochanterfemur')]-k;c=p[names.index(leg+'_tarsus1')]-k
    return np.arccos(np.clip(np.dot(a,c)/np.linalg.norm(a)/np.linalg.norm(c),-1,1))
for leg in LEGS:
    joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'fly/{leg}_trochanterfemur-{leg}_tibia-pitch')
    assert joint>=0
    adr=m.jnt_qposadr[joint];base=d.qpos[adr];before=included_angle(leg)
    d.qpos[adr]=base+.01;mujoco.mj_forward(m,d)
    assert included_angle(leg)<before-.009,leg
    d.qpos[adr]=base;mujoco.mj_forward(m,d)
print('six geometric flexion signs verified')
'''
    p=subprocess.run([str(ROOT/'.venv-body/Scripts/python.exe'),'-c',code],capture_output=True,text=True,cwd=ROOT,timeout=30)
    assert p.returncode==0,p.stderr


def test_physics_running_does_not_claim_neural_drive():
    from chatflypt.sensory_readout import sensory_response_text
    text=sensory_response_text({'words':['x'],'descriptions':{'x':'触碰'}},{'non_sensory_spikes':5,
        'vnc_motor':{'spikes':5},'body_simulation':{'status':'ok','physics_run':True,'motor_activity_in_window':False,
        'motor_coverage':{'unmapped_active_neurons':3}}})
    assert '已驱动' not in text and '姿态维持' in text and '3 个' in text
