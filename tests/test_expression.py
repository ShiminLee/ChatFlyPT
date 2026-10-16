"""Neural labels can speak; input plans, motor activity and advice cannot."""
import pytest

from chatflypt.expression import expression_report
from chatflypt import conversation


def observation(words=('sugar',), spikes=3, **extra):
    return {'id':'test','dataset':'FlyWire v783','decoded':{'words':list(words)},
            'reaction':{'non_sensory_spikes':spikes},'reception':{},**extra}


def test_author_example_has_only_decoded_sensation_no_question_or_advice():
    report=expression_report(observation(['vinegar','yeast','sugar','warm','bright']))
    assert set(report['labels'])=={'vinegar','yeast','sugar','warm','bright'}
    assert report['text']=='我察觉到了光。我尝到了甜味。我闻到了醋相关气味。我感到温暖。我闻到了酵母相关气味。'
    assert report['status']=='decoded' and not report['llm_called']
    assert not report['outward_signal_decoded']


def test_input_description_history_motor_and_saved_text_cannot_change_translation():
    obs=observation()
    expected=expression_report(obs)
    obs.update(intent='去吃盐',history=['向左'],sentence='我想飞',response_text='飞起来',
        sensation={'text':'我不开心'},fly_text='停下',expression={'text':'你好'},
        scene={'body':{'commands':{'turn':1,'song':1}}},
        reception={'description':'模拟冷风','intent':'向左'},
        outward_signals={'emitted_sound':{'text':'我要飞'}})
    obs['decoded']['descriptions']={'sugar':'我很快乐'}
    obs['decoded']['scores']={'cold':100}
    assert expression_report(obs)==expected


@pytest.mark.parametrize('words,spikes,status',[
    (['sugar'],0,'no_signal'),([],3,'unidentified'),
    (['invented'],3,'unidentified'),(['command:left'],3,'command'),
    (['warm','cold'],3,'ambiguous')])
def test_abstention_is_a_system_status_not_fabricated_fly_speech(words,spikes,status):
    report=expression_report(observation(words,spikes))
    assert report['status']==status and report['text'] is None and not report['messages']
    assert report['reason']


def test_uncalibrated_and_unqualified_labels_are_not_promoted():
    obs=observation();obs['decoded']['status']='uncalibrated'
    assert expression_report(obs)['text'] is None
    obs['decoded']={'words':['sugar','bitter','command:left'],'unqualified_words':['bitter']}
    report=expression_report(obs)
    assert report['text']=='我尝到了甜味。' and report['labels']==['sugar']


def test_feedback_prediction_is_qualified_even_when_saved_sensation_claims_certainty():
    obs=observation(embodied_feedback={'sampling_ms':100},sensation={'status':'decoded'})
    report=expression_report(obs)
    assert report['status']=='candidate'
    assert report['text']=='候选神经读出（连续反馈尚未校准）：我尝到了甜味。'


def test_independent_datasets_keep_separate_messages_and_candidate_status(tmp_path,monkeypatch):
    first=observation(embodied_feedback={'sampling_ms':100})
    second=observation(['foreleg_touch:right'],id='second',dataset='MaleCNS v1.0')
    class App:
        def stimulate(self,*args,**kwargs):
            return {'observation':first,'observations':[first,second],'pathway':{}}
    monkeypatch.setattr(conversation,'ROOT',tmp_path)
    result=conversation.run_turn(App(),kind='stimuli')
    assert [m['dataset'] for m in result['messages']]==['FlyWire v783','MaleCNS v1.0']
    assert [m['status'] for m in result['messages']]==['candidate','decoded']
    assert result['pathway']['expression_translation_completed']
    assert not result['pathway']['readout_qualified'] and not result['pathway']['completed']
    assert first['reception']['sensation']['status']=='candidate'
