"""Scene physics/retina and semantic isolation, including negative controls."""
import copy
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from chatflypt.environment import (TYPES, validate_objects, sample_environment,
                                  probe_environment, empty_environment)
from chatflypt.expression import expression_report
from chatflypt.behavior_readout import build_behavior_drive
from chatflypt.behavior_body import run_behavior_body
from chatflypt import server


def obj(kind, position=(4,-2,.5), size=(1,1,1), **kwargs):
    return dict(id='sample',type=kind,position_mm=list(position),size_mm=list(size),
                yaw_deg=0,strength=1,range_mm=4,**kwargs)


@pytest.mark.parametrize('kind',list(TYPES))
def test_all_scene_types_are_valid(kind):
    assert validate_objects([obj(kind)])[0]['type']==kind


@pytest.mark.parametrize('change',[{'position_mm':[0,0,-1]}, {'size_mm':[1,0,1]},
    {'strength':float('nan')},{'yaw_deg':True},{'id':'../../bad'},{'range_mm':float('inf')}])
def test_invalid_scene_rejected_before_simulation(change):
    item=obj('sugar');item.update(change)
    with pytest.raises(ValueError):validate_objects([item])


@pytest.fixture(scope='module')
def catalogue():
    return {'channels':{key:{'status':'ready','side_ids':{'left':['l'],'right':['r'],'bilateral':['l','r']}}
                        for key in ('sugar_specific','high_salt','bright','warm','cold','yeast')}}


@pytest.fixture(scope='module')
def probes():
    # Real official body, rendering, and contacts, not input-label fixtures.
    scenes={'empty':[], 'far_sugar':[obj('sugar')],
            'platform':[obj('block',(0,0,.1),(5,4,.2))],
            'dark':[obj('shade',(0,0,3),(8,8,6))],
            'mouth_sugar':[obj('sugar',(1.1,0,.4),(.7,1,.8))]}
    return {name:({'revision':1,'objects':items},probe_environment({'revision':1,'objects':items},seed=42))
            for name,items in scenes.items()}


def test_far_sugar_is_visible_without_being_tasted(probes,catalogue):
    env,probe=probes['far_sugar'];sample=sample_environment(env,probe,catalogue)
    blank=probes['empty'][1]
    assert probe['vision']['status']=='measured'
    assert len(probe['vision']['ommatidia'][0])==721
    assert probe['vision']['ommatidia']!=blank['vision']['ommatidia']
    assert not probe['vision']['object_identity_injected']
    assert any(i['channel']=='bright' for i in sample['items'])
    assert not any(i['channel']=='sugar_specific' for i in sample['items'])


def test_coarse_mouth_contact_does_not_claim_labellar_taste(probes,catalogue):
    env,probe=probes['far_sugar'];probe=copy.deepcopy(probe)
    probe['environment_contacts']=[{'object_id':'sample','fly_geom':'fly/rf_tarsus5'}]
    assert not any(i['channel']=='sugar_specific' for i in sample_environment(env,probe,catalogue)['items'])
    probe['environment_contacts']=[{'object_id':'sample','fly_geom':'fly/c_haustellum'}]
    sample=sample_environment(env,probe,catalogue)
    assert not any(i['channel']=='sugar_specific' for i in sample['items'])
    assert any(i['channel']=='sugar_specific' and 'labellum' in i['reason'] for i in sample['unmapped'])
    real_env,real_probe=probes['mouth_sugar']
    assert any('rostrum' in c['fly_geom'] for c in real_probe['environment_contacts'])
    sample=sample_environment(real_env,real_probe,catalogue)
    assert not any(i['channel']=='sugar_specific' for i in sample['items'])
    assert any(i['channel']=='sugar_specific' for i in sample['unmapped'])


def test_dark_region_attenuates_retina_without_inventing_dark_neurons(probes,catalogue):
    env,probe=probes['dark']
    assert probe['vision']['mean_by_eye']=={'left':0,'right':0}
    assert not sample_environment(env,probe,catalogue)['items']


@pytest.mark.parametrize('kind,channel',[('yeast','yeast'),('heat','warm'),('cold','cold')])
def test_fields_depend_on_measured_receptor_position_and_range(probes,catalogue,kind,channel):
    probe=probes['empty'][1]
    site_name='l_funiculus' if kind=='yeast' else 'l_arista'
    site=probe['frame']['body_positions_mm'][probe['body_names'].index(site_name)]
    item=obj(kind,site,(.05,.05,.05));item['range_mm']=.1
    env={'revision':1,'objects':[item]}
    near=[x for x in sample_environment(env,probe,catalogue)['items'] if x['channel']==channel]
    assert len(near)==1 and near[0]['side']=='left' and near[0]['intensity']==1.
    item['position_mm']=[20,20,20]
    assert not any(x['channel']==channel for x in sample_environment(env,probe,catalogue)['items'])


def test_missing_side_mapping_is_reported_without_substituting_other_side(probes,catalogue):
    env,probe=probes['empty'];catalogue=copy.deepcopy(catalogue)
    catalogue['channels']['bright']['side_ids']['left']=[]
    result=sample_environment(env,probe,catalogue)
    assert not any(x['side']=='left' for x in result['items'])
    assert any(x['side']=='left' for x in result['unmapped'])


def test_real_platform_collides_and_supports_body(probes):
    env,probe=probes['platform']
    assert 'env_sample' in probe['environment_geom_names']
    assert probe['environment_contacts']
    assert probe['frame']['position_mm'][2]>probes['empty'][1]['frame']['position_mm'][2]+.1
    trial=SimpleNamespace(counts=np.array([0]),spike_ids=np.array([],dtype=np.int32),
        spike_steps=np.array([],dtype=np.int32),duration_ms=60.,seed=42)
    drive=build_behavior_drive(trial,np.array([1]),[{'type':'DNg100','somaSide':'L'}],dataset='MaleCNS v1.0')
    body=run_behavior_body(drive,environment=env)
    assert body['physics_run'] and body['summary']['finite_state'],body
    assert body['controls']['powered_steps']==0
    assert any(f['environment_contacts'] for f in body['trajectory'])


def test_input_or_motor_labels_without_blind_predictions_do_not_generate_speech():
    report=expression_report({'id':'test','decoded':{'words':[]},
        'reception':{'words':['sugar','courtship_pulse']},
        'reaction':{'non_sensory_spikes':10000},'scene':{'body':{'commands':{'song':1}}}})
    assert report['text'] is None and not report['messages']
    assert report['status']=='unidentified'


def test_failed_apply_and_stale_revision_leave_applied_scene_unchanged(tmp_path,monkeypatch):
    app=server.Application.__new__(server.Application);app.environment=empty_environment()
    app.environment_lock=threading.RLock()
    def forbidden(*args,**kwargs):raise AssertionError('No physics on stale request')
    monkeypatch.setattr('chatflypt.environment.probe_environment',forbidden)
    with pytest.raises(ValueError,match='版本'):app.apply_environment([],2)
    assert app.environment==empty_environment()
    def failed(*args,**kwargs):raise ValueError('Probe failed')
    monkeypatch.setattr('chatflypt.environment.probe_environment',failed)
    with pytest.raises(ValueError,match='Probe failed'):app.apply_environment([],0)
    assert app.environment==empty_environment()
