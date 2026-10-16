"""The active conversation cannot construct a model or send action feedback."""
import json
import pytest
from chatflypt import conversation, language, model_runtime

class FakeApp:
    def __init__(self):
        self.calls=[]
    def stimulate(self,items,seed,**kwargs):
        self.calls.append(('stimulate',items,seed))
        obs={'id':'initial','dataset':'MaleCNS v1.0','decoded':{'words':['foreleg_touch:right']},
             'reaction':{'non_sensory_spikes':3},'reception':{},'scene':{'body':{'source':'neural-only'}},
             'response_text':'TECHNICAL_ONLY'}
        return {'handled':True,'observation':obs,'observations':[obs],'pathway':{}}
    def observe(self,words,seed,**kwargs):
        return self.stimulate(words,seed)['observation']
    def feedback(self,*args,**kwargs):
        pytest.fail('No external action feedback is allowed')

@pytest.fixture
def app(tmp_path,monkeypatch):
    monkeypatch.setattr(conversation,'ROOT',tmp_path)
    def forbidden(*args,**kwargs):
        pytest.fail('No language model may run in the fly pathway')
    monkeypatch.setattr(language.LLM,'complete',forbidden)
    monkeypatch.setattr(model_runtime,'make_model',forbidden)
    return FakeApp()

@pytest.mark.parametrize('backend',['configured','local','off'])
@pytest.mark.parametrize('feedback',[True,False])
def test_even_stale_enabled_client_settings_cannot_call_model_or_feedback(app,backend,feedback):
    result=conversation.run_turn(app,kind='stimuli',items=[{'input':'sentinel'}],seed=42,
                                 backend=backend,feedback_enabled=feedback)
    assert len(app.calls)==1 and len(result['observations'])==1
    assert result['observation']['reception']['sensation']['text']=='我感到右前腿被碰到了。'
    assert result['observation']['fly_text']=='我感到右前腿被碰到了。'
    assert result['reply']==result['messages'][0]['text']==result['observation']['fly_text']
    assert result['messages'][0]['source']=='blind_neural_readout'
    assert result['observation']['scene']['body']=={'source':'neural-only'}
    assert result['dialogue']['max_feedback_rounds']==0
    assert not result['dialogue']['feedback_enabled'] and not result['dialogue']['llm_called']
    assert not result['pathway']['communication_round_trip']
    saved=json.loads((conversation.ROOT/result['dialogue_record']).read_text(encoding='utf8'))
    assert saved['reply']==result['reply'] and 'model_response' not in saved['dialogue']

@pytest.mark.parametrize('kind',['stimuli','words'])
def test_default_is_neural_expression_only(app,kind):
    result=conversation.run_turn(app,kind=kind)
    assert result['dialogue']['backend']=='off'
    assert result['reply_source']=='neural_expression'
    assert result['dialogue']['status']=='neural_only'
    assert result['pathway']['expression_translation_completed']
    assert result['pathway']['completion_scope']=='bounded_neural_readout_turn'

@pytest.mark.parametrize('kwargs',[{'backend':'unknown'},{'feedback_enabled':'yes'},
    {'seed':True},{'seed':-1},{'kind':'unknown'},{'body_context':'free_flight'}])
def test_bad_options_do_not_run_a_trial(app,kwargs):
    with pytest.raises(ValueError):
        conversation.run_turn(app,**dict({'kind':'stimuli'},**kwargs))
    assert not app.calls
