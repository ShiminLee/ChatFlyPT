import copy
from types import SimpleNamespace
import pytest

from chatflypt import conversation
from chatflypt.embodied import run_embodied
from chatflypt.environment import empty_environment, sample_environment
from chatflypt.environment_message import parse_message, ground_message
from chatflypt.physical_touch import validate_presentations


@pytest.fixture(scope='module')
def catalogue():
    from chatflypt.sensory_catalogue import build_catalogue
    return build_catalogue(save=False)


@pytest.fixture
def probe():
    return {'frame':{'position_mm':[0,0,1],'body_quaternions_wxyz':[[1,0,0,0]],
                     'body_positions_mm':[[0,0,1]]},'body_names':['c_thorax'],'environment_contacts':[]}


def test_food_description_places_a_real_object_without_taste_or_input_ids(catalogue,probe):
    before=empty_environment()
    plan,env=ground_message(parse_message('右边有糖',catalogue),before,probe,catalogue)
    assert before['objects']==[] and env['revision']==1
    assert env['objects'][0]['position_mm'][1]==-3 and env['objects'][0]['type']=='sugar'
    assert plan['items']==[] and plan['presentations']==[]
    sampled=sample_environment(env,probe,catalogue)
    assert sampled['items']==[]  # A label and distance cannot activate taste.
    assert not sampled['evidence'][0]['activated']


def test_turn_is_a_physical_probe_without_a_receptor_item(catalogue,probe):
    env=empty_environment()
    plan,after=ground_message(parse_message('左转',catalogue),env,probe,catalogue)
    assert after==env and plan['items']==[]
    assert plan['presentations']==[{'kind':'touch','side':'right','body_part':'foreleg',
        'strength':1.,'onset_ms':50.,'duration_ms':300.}]
    assert not plan['translation']['forces_body_action']


def test_physical_modality_gap_never_falls_back_to_neural_stimulation(catalogue,probe):
    plan,env=ground_message(parse_message('醋',catalogue),empty_environment(),probe,catalogue)
    assert plan['kind']=='unsupported' and plan['items']==[]
    assert not env['objects'] and not plan['presentations']


@pytest.mark.parametrize('text',['不要在右边放糖','如果有糖呢','在右边放糖然后左转'])
def test_partial_scene_instructions_do_not_execute(catalogue,text):
    assert parse_message(text,catalogue)['kind'] in ('unsupported','question')


def test_world_only_guard_rejects_direct_input_before_loading_a_brain(tmp_path):
    with pytest.raises(ValueError,match='禁止直接神经刺激'):
        run_embodied(None,None,None,'MaleCNS v1.0',[{'channel':'foreleg_touch'}],empty_environment(),1,'ground',
                     input_mode='environment_only',output_dir=tmp_path)


def test_probe_protocol_rejects_neural_and_motor_instructions():
    with pytest.raises(ValueError,match='不接受神经或动作指令'):
        validate_presentations([{'kind':'touch','side':'right','body_part':'foreleg','channel':'foreleg_touch'}])


def test_chat_never_passes_even_legacy_plan_items_to_neurons(tmp_path,monkeypatch):
    monkeypatch.setattr(conversation,'ROOT',tmp_path)
    seen={}
    def prepare(*args,**kwargs):
        return {'plan':{'kind':'scene_observation','items':[{'channel':'sugar_specific'}],
                        'unresolved':[]},'environment':empty_environment(),'sample':None,'changed':False}
    def stimulate(items,seed,**kwargs):
        seen.update(items=items,**kwargs)
        obs={'id':'world-only','dataset':'MaleCNS v1.0','decoded':{'words':[]},
             'reaction':{'non_sensory_spikes':0},'reception':{},'scene':{'body':{}}}
        return {'handled':True,'observations':[obs],'observation':obs,'pathway':{}}
    app=SimpleNamespace(prepare_scene_message=prepare,stimulate=stimulate)
    conversation.run_turn(app,message='糖')
    assert seen['items']==[] and seen['input_mode']=='environment_only'


@pytest.fixture
def placement_catalogue():
    """Parsing/placement needs identities, not a full connectome on disk."""
    definitions={'yeast':('酵母','olfaction'),'warm':('温暖','temperature'),
                 'bright':('明亮','vision'),'foreleg_touch':('触碰','touch')}
    return {'channels':{key:{'id':key,'aliases':[word],'modality':modality,
        'status':'candidate','body_parts':['foreleg'] if key=='foreleg_touch' else [],
        'side_ids':{'left':['1'],'right':['2'],'bilateral':['1','2']}}
        for key,(word,modality) in definitions.items()}}


@pytest.mark.parametrize(('message','kind'),[
    ('两侧酵母','yeast'),('双侧放糖','sugar'),('左右两边放盐块','high_salt'),
    ('两侧温暖','heat'),('左侧和右侧明亮','light')])
def test_explicit_bilateral_objects_become_two_world_presentations(placement_catalogue,probe,message,kind):
    before=empty_environment()
    plan,environment=ground_message(parse_message(message,placement_catalogue),before,probe,placement_catalogue)
    assert plan['kind']=='environment_presentation',plan
    assert plan['items']==[] and plan['presentations']==[]
    assert [(o['type'],o['position_mm'][:2]) for o in environment['objects']]==[(kind,[0,3]),(kind,[0,-3])]
    assert {o['id'] for o in environment['objects']}=={'chat_'+kind+'_left','chat_'+kind+'_right'}
    assert not before['objects'] and not plan['translation']['direct_neural_input']


@pytest.mark.parametrize('message',['温暖','酵母','给它酵母','明亮'])
def test_omitted_object_side_still_defaults_to_one_front_object(placement_catalogue,probe,message):
    plan,environment=ground_message(parse_message(message,placement_catalogue),empty_environment(),probe,placement_catalogue)
    assert plan['kind']=='environment_presentation' and not plan['items']
    assert len(environment['objects'])==1 and environment['objects'][0]['position_mm'][:2]==[3,0]
    assert environment['objects'][0]['id'].endswith('_front')


def test_bilateral_metadata_stays_with_its_clause_and_rotates_with_body(placement_catalogue,probe):
    import math
    probe['frame']['body_quaternions_wxyz']=[[math.sqrt(.5),0,0,math.sqrt(.5)]]
    plan,environment=ground_message(parse_message('两侧酵母和温暖',placement_catalogue),empty_environment(),probe,placement_catalogue)
    by_id={o['id']:o for o in environment['objects']}
    assert len(by_id)==3
    assert by_id['chat_yeast_left']['position_mm'][:2]==pytest.approx([-3,0])
    assert by_id['chat_yeast_right']['position_mm'][:2]==pytest.approx([3,0])
    assert by_id['chat_heat_front']['position_mm'][:2]==pytest.approx([0,3])
    assert plan['items']==[]
