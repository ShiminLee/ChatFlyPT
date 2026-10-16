import copy
import math
import threading

import pytest

from chatflypt import conversation, server
from chatflypt.environment import empty_environment, sample_environment, validate_objects
from chatflypt.scene_language import parse_scene_request, ground_scene_request, evaluate_goal


def obj(kind='sugar',oid='food',position=(3,-2,.15)):
    return validate_objects([{'id':oid,'type':kind,'position_mm':list(position),'size_mm':[1,1,.3]}])[0]


@pytest.fixture
def catalogue():
    channels={}
    for key in ('foreleg_touch','midleg_touch','hindleg_touch','sugar_specific','bright','leg_sweet_local','leg_appetitive_ascending'):
        channels[key]={'status':'candidate','source':'source','side_ids':{'left':['1'],'right':['2'],'bilateral':['1','2']},
            'body_part_ids':{p:{'left':[p+'L'],'right':[p+'R'],'bilateral':[p+'L',p+'R']} for p in ('foreleg','midleg','hindleg')}}
    return {'channels':channels}


@pytest.fixture
def probe():
    return {'frame':{'position_mm':[0,0,1],'body_quaternions_wxyz':[[1,0,0,0]],'body_positions_mm':[[0,0,1]]},
            'body_names':['c_thorax'],'environment_contacts':[]}


@pytest.mark.parametrize('text,kind,target',[
    ('糖在哪里','query','sugar'),('哪里有酵母？','query','yeast'),('方块在哪儿','query','block'),
    ('走到糖的位置去','goal','sugar'),('请靠近左边的热源','goal','heat'),('去最近的光源那里','goal','light'),
    ('看看周围','observe',None),('观察高盐','observe','high_salt'),
])
def test_general_scene_grammar(text,kind,target):
    request=parse_scene_request(text);assert request['kind']==kind and request['object_type']==target


@pytest.mark.parametrize('text',['不要走到糖的位置去','如果去糖那里会怎样','走到糖那里然后发出声音','去糖那里还是酵母那里','糖在哪里，左转','左转','模拟右前腿触碰'])
def test_partial_or_negated_scene_requests_do_not_execute(text):
    assert parse_scene_request(text) is None


def test_goal_uses_body_relative_opposite_touch_not_body_command(catalogue,probe):
    environment={'revision':3,'objects':[obj()]}
    plan=ground_scene_request(parse_scene_request('走到糖的位置去'),environment,probe,catalogue)
    assert plan['items']==[]
    assert plan['presentations'][0]['kind']=='touch' and plan['presentations'][0]['side']=='left'
    assert plan['referent']['side']=='right' and not plan['translation']['forces_body_action']
    assert not any(i['channel']=='sugar_specific' for i in plan['items'])
    probe['frame']['body_quaternions_wxyz']=[[0,0,0,1]]
    rotated=ground_scene_request(parse_scene_request('去糖那里'),environment,probe,catalogue)
    assert rotated['items']==[] and rotated['presentations'][0]['side']=='right'


def test_question_does_not_add_touch_or_remote_taste(catalogue,probe):
    plan=ground_scene_request(parse_scene_request('糖在哪里'),{'revision':1,'objects':[obj()]},probe,catalogue)
    assert plan['kind']=='scene_observation' and plan['items']==[]


def test_ambiguous_goal_requires_disambiguation(catalogue,probe):
    environment={'revision':1,'objects':[obj(),obj(oid='other',position=(3,2,.15))]}
    ambiguous=ground_scene_request(parse_scene_request('去糖那里'),environment,probe,catalogue)
    assert ambiguous['kind']=='unsupported' and not ambiguous['items']
    resolved=ground_scene_request(parse_scene_request('去左边的糖那里'),environment,probe,catalogue)
    assert resolved['referent']['object_id']=='other'


def test_actual_leg_contact_uses_same_leg_and_never_labellar_cells(catalogue,probe):
    env={'revision':1,'objects':[obj()]}
    assert not any(i['channel'].startswith('leg_') for i in sample_environment(env,probe,catalogue)['items'])
    probe['environment_contacts']=[{'object_id':'food','fly_geom':'fly/rh_tarsus5'}]
    result=sample_environment(env,probe,catalogue)
    inputs=result['items'];tastes=[i for i in inputs if i['channel'].startswith('leg_')]
    assert len(tastes)==2 and all(i['side']=='right' and i['body_part']=='hindleg' for i in tastes)
    assert not any(i['channel']=='sugar_specific' for i in inputs)
    assert any(i['channel']=='hindleg_touch' and i['side']=='right' for i in inputs)
    assert all(x['neuron_ids']==['hindlegR'] for x in result['evidence'] if x['mode']=='mujoco_leg_taste_candidate')


def test_leg_salt_never_reuses_sugar_or_mouth(catalogue,probe):
    probe['environment_contacts']=[{'object_id':'food','fly_geom':'fly/lf_tibia'}]
    result=sample_environment({'revision':1,'objects':[obj(kind='high_salt')]},probe,catalogue)
    assert not any(i['channel'].startswith('leg_') for i in result['items'])
    assert any(x['channel']=='leg_high_salt' for x in result['unmapped'])


def test_goal_evaluation_is_only_posthoc(catalogue,probe):
    plan=ground_scene_request(parse_scene_request('去糖那里'),{'revision':1,'objects':[obj()]},probe,catalogue)
    observations=[{'id':'obs','dataset':'MaleCNS v1.0','scene':{'body':{'physics_run':True,'trajectory':[
        {'time_ms':0,'position_mm':[0,0,1],'environment_contacts':[]},
        {'time_ms':500,'position_mm':[1,0,1],'environment_contacts':[]}],'controls':{'powered_steps':3}}}}]
    result=evaluate_goal(plan,observations)
    assert result['trials'][0]['distance_change_mm']>0 and result['trials'][0]['status']=='not_reached'
    assert not result['used_for_control'] and not result['goal_completed']


def test_environment_apply_runs_neural_path_and_saves_response(tmp_path,monkeypatch):
    app=server.Application.__new__(server.Application);app.environment=empty_environment()
    app.environment_lock=threading.RLock();app.body_enabled=True
    calls=[];app.input_catalogue=lambda:{'channels':{}}
    def converse(**kwargs):
        calls.append(kwargs)
        return {'neural_simulation_run':True,'observations':[{'scene':{'body':{'physics_run':True,'status':'ok'}}}]}
    app.converse=converse
    monkeypatch.setattr(server,'ROOT',tmp_path)
    monkeypatch.setattr('chatflypt.environment.probe_environment',lambda *a,**k:{})
    monkeypatch.setattr('chatflypt.environment.sample_environment',lambda *a,**k:{'items':[],'source':'measured'})
    result=app.apply_environment([obj()],0)
    assert calls[0]['kind']=='environment' and calls[0]['environment_sample']['source']=='measured'
    assert result['environment']['last_record']['kind']=='neural' and result['neural_simulation_run']
    assert app.environment_response(result['environment']['last_record']['id'])==result
    before=copy.deepcopy(app.environment)
    app.converse=lambda **kwargs:{'observations':[{'scene':{'body':{'status':'error','reason':'failed'}}}]}
    with pytest.raises(ValueError,match='环境未应用'):app.apply_environment([],1)
    assert app.environment==before
