import numpy as np
import pytest

from chatflypt.brain import Trial
from chatflypt.feedback_readout import FeedbackDecoder,normalized_features
from chatflypt.paths import sha256


class Base:
    indices=np.array([1,2]);feature_indices=np.array([1,5,9])
    labels=['sugar','bright'];names={'sugar':'甜味','bright':'光'}
    catalogue_signature='catalogue-fixture'
    def decode_trial(self,trial):
        words=['bright'] if trial.counts[2]>0 else []
        return {'words':words,'scores':{'sugar':0.,'bright':float(bool(words))},
            'descriptions':{w:self.names[w] for w in words},'unqualified_words':[],'status':'decoded'}


def reader(tmp_path,**changes):
    base_path=tmp_path/'base.npz';base_path.write_bytes(b'base-model')
    path=tmp_path/'adapter.npz'
    params=dict(feature_indices=Base.feature_indices,weights=np.array([[1.,0.,0.]]),
        intercept=np.array([.7]),thresholds=np.array([1.]),enabled=np.array([True]),labels=['sugar'],
        base_sha256=sha256(base_path),protocol_sha256='protocol-fixture',catalogue_signature=Base.catalogue_signature)
    params.update(changes);np.savez(path,**params)
    return FeedbackDecoder(Base(),path,base_path)


def trial(counts,duration=500.,state=None):
    return Trial(np.asarray(counts),np.empty(0,np.int32),np.empty(0,np.int32),42,duration,
                 np.zeros((4,4)) if state is None else state)


def test_feedback_probe_ignores_direct_inputs_and_preserves_other_readout_labels(tmp_path):
    decoder=reader(tmp_path)
    a=decoder.decode_trial(trial([0,4,1,0]))
    b=decoder.decode_trial(trial([100000,4,1,99999]))
    assert a==b and a['words']==['sugar','bright']
    assert a['feedback_adaptation']['applied']
    assert not a['feedback_adaptation']['globally_calibrated']


def test_intercept_cannot_make_silence_speak_and_unknown_duration_uses_base(tmp_path):
    decoder=reader(tmp_path,intercept=np.array([10.]))
    assert not decoder.decode_trial(trial([0,0,0,0]))['words']
    other=decoder.decode_trial(trial([0,4,1,0],duration=300.))
    assert other['words']==['bright'] and not other['feedback_adaptation']['applied']


def test_failed_validation_never_overrides_existing_labels(tmp_path):
    decoder=reader(tmp_path,enabled=np.array([False]))
    result=decoder.decode_trial(trial([0,4,1,0]))
    assert result['words']==['bright'] and not result['feedback_adaptation']['applied']


def test_limited_feedback_validation_cannot_remove_chat_candidate_warning(tmp_path):
    from chatflypt.expression import expression_report
    decoded=reader(tmp_path).decode_trial(trial([0,4,0,0]))
    report=expression_report({'decoded':decoded,'reaction':{'non_sensory_spikes':4},'embodied_feedback':{'interval_ms':100}})
    assert report['status']=='candidate'
    assert report['text']=='候选神经读出（连续反馈仅有限验证）：我尝到了甜味。'


def test_stale_optional_adapter_preserves_base_with_an_explicit_reason(tmp_path,monkeypatch):
    from chatflypt import feedback_readout
    from chatflypt import state_readout
    reader(tmp_path)
    (tmp_path/'adapter.npz').rename(tmp_path/'flywire-adapter.npz')
    (tmp_path/'base.npz').rename(tmp_path/'flywire-decoder.npz')
    (tmp_path/'flywire-decoder.npz').write_bytes(b'updated-base')
    monkeypatch.setattr(feedback_readout,'OUT',tmp_path)
    monkeypatch.setattr(state_readout,'OUT',tmp_path)
    base=Base();selected,note=feedback_readout.load_feedback_decoder(base,'FlyWire v783')
    assert selected is base and not note['applied'] and 'different base model' in note['reason']


def test_disabled_adapter_stays_off_even_if_offline_fitting_recreates_it(tmp_path,monkeypatch):
    from chatflypt import feedback_readout,state_readout
    reader(tmp_path)
    (tmp_path/'adapter.npz').rename(tmp_path/'flywire-adapter.npz')
    (tmp_path/'base.npz').rename(tmp_path/'flywire-decoder.npz')
    (tmp_path/'flywire-disabled.json').write_text('{"reason":"failed heat negative controls"}')
    monkeypatch.setattr(feedback_readout,'OUT',tmp_path)
    monkeypatch.setattr(state_readout,'OUT',tmp_path)
    base=Base();selected,note=feedback_readout.load_feedback_decoder(base,'FlyWire v783')
    assert selected is base and not note['applied'] and 'disabled' in note['reason']


@pytest.mark.parametrize('changes',[
    {'base_sha256':'different'}, {'catalogue_signature':'different'},
    {'feature_indices':np.array([0,4,8])}, {'labels':['invented']},
    {'weights':np.array([[np.nan,0,0]])}, {'enabled':np.array([1.])},
])
def test_stale_or_unblinded_adapters_cannot_load(tmp_path,changes):
    with pytest.raises(ValueError):reader(tmp_path,**changes)


def test_duration_normalization_uses_neural_measurements_only():
    a=np.array([0.,4.,2.,1.]);state=np.ones((4,4))
    np.testing.assert_allclose(normalized_features(a,state,500.),normalized_features(a*2,state*2,1000.))
    with pytest.raises(ValueError):normalized_features(a,state,0)


def test_benchmark_splits_whole_sessions_and_includes_withdrawal():
    from scripts.benchmark_feedback_readout import protocol
    episodes=protocol()['episodes']
    assert len({e['id'] for e in episodes})==len(episodes)==18
    seeds=[e['seed']+i for e in episodes for i,_ in enumerate(e['turns'])]
    assert len(set(seeds))==len(seeds)==48
    for episode in episodes:
        if episode['dataset']=='MaleCNS v1.0':
            assert episode['turns'][1]==episode['turns'][3]==[]
            assert episode['turns'][0][0]['side']!=episode['turns'][2][0]['side']


def test_evaluation_labels_require_actual_events_and_correct_population():
    from scripts.fit_feedback_readout import measured_targets
    catalogue={'channels':{'foreleg_touch':{'dataset':'MaleCNS v1.0',
        'side_ids':{'left':['1'],'right':['2'],'bilateral':['1','2']}}}}
    row={'dataset':'MaleCNS v1.0','human_items':[{'channel':'foreleg_touch','side':'left'}],'environment_items':[]}
    assert measured_targets(row,catalogue,['2'])==set()
    assert measured_targets(row,catalogue,['1'])=={'foreleg_touch:left'}
