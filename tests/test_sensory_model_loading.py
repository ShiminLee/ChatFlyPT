"""A long-lived service must see a later trained model and reject stale maps."""
from types import SimpleNamespace
from chatflypt import server
from chatflypt.sensory_readout import UncalibratedDecoder, fingerprint


def test_environment_requires_current_feedback_model(monkeypatch):
    from chatflypt import unified_readout
    app=server.Application.__new__(server.Application)
    brain,catalogue=object(),{'channels':{}}
    missing={'applied':False,'reason':'stale physical model'}
    monkeypatch.setattr(unified_readout,'load_unified_decoder',lambda b,c:(None,missing))
    decoder,note=app._environment_decoder(brain,catalogue)
    assert isinstance(decoder,UncalibratedDecoder) and note is missing
    qualified=object()
    monkeypatch.setattr(unified_readout,'load_unified_decoder',lambda b,c:(qualified,None))
    assert app._environment_decoder(brain,catalogue)==(qualified,None)


def test_unqualified_labels_cannot_be_rendered_as_a_sensation(tmp_path):
    import numpy as np
    from chatflypt.sensory_readout import SensoryDecoder,sensory_response_text
    path=tmp_path/'probe.npz'
    np.savez(path,indices=[0],weights=[[1.],[1.]],intercept=[0.,0.],thresholds=[.5,.5],
             labels=['candidate_a:left','candidate_b:right'],names=['左侧候选A','右侧候选B'],
             qualified=[True,False],catalogue_fingerprint='unit-only')
    decoder=SensoryDecoder('FlyWire v783',path)
    decoded=decoder.decode(np.array([1]))
    assert decoded['words']==['candidate_a:left']
    assert decoded['unqualified_words']==['candidate_b:right']
    assert decoded['scores']['candidate_b:right']==1.
    reply=sensory_response_text(decoded,{'non_sensory_spikes':1})
    assert '左侧候选A' in reply and '右侧候选B' not in reply
    assert '未通过单通道验证' in reply


def test_missing_model_can_become_available_without_restart(tmp_path,monkeypatch):
    app=server.Application.__new__(server.Application)
    app.brain=object();app.view=object()
    app.sensory_catalogue={'channels':{}}
    app.sensory_decoders={};app.sensory_decoder_signatures={}
    monkeypatch.setattr(server,'SENSORY_OUT',tmp_path)
    _,_,before=app._sensory_model('FlyWire v783',True)
    assert isinstance(before,UncalibratedDecoder)
    assert 'FlyWire v783' not in app.sensory_decoders
    matching=SimpleNamespace(catalogue_fingerprint=fingerprint(app.sensory_catalogue))
    monkeypatch.setattr(server,'SensoryDecoder',lambda dataset:matching)
    path=tmp_path/'flywire-decoder.npz';path.write_bytes(b'model-v1')
    assert app._sensory_model('FlyWire v783',True)[2] is matching
    mismatched=SimpleNamespace(catalogue_fingerprint='different-receptor-map')
    monkeypatch.setattr(server,'SensoryDecoder',lambda dataset:mismatched)
    path.write_bytes(b'model-v2-larger')
    assert isinstance(app._sensory_model('FlyWire v783',True)[2],UncalibratedDecoder)


def test_state_model_invalidation_cannot_reuse_state_object_as_count_fallback(tmp_path,monkeypatch):
    from chatflypt import state_readout
    app=server.Application.__new__(server.Application)
    app.brain=object();app.view=object();app.sensory_catalogue={'channels':{}}
    app.sensory_decoders={};app.sensory_decoder_signatures={}
    count_dir=tmp_path/'count';count_dir.mkdir();state_dir=tmp_path/'state';state_dir.mkdir()
    monkeypatch.setattr(server,'SENSORY_OUT',count_dir)
    monkeypatch.setattr(state_readout,'OUT',state_dir)
    (count_dir/'flywire-decoder.npz').write_bytes(b'count')
    count=SimpleNamespace(catalogue_fingerprint=fingerprint(app.sensory_catalogue))
    monkeypatch.setattr(server,'SensoryDecoder',lambda dataset:count)
    assert app._sensory_model('FlyWire v783',True)[2] is count
    (state_dir/'flywire-decoder.npz').write_bytes(b'state')
    state=SimpleNamespace(catalogue_signature=state_readout.signature(app.sensory_catalogue,'FlyWire v783'))
    monkeypatch.setattr(state_readout,'StateDecoder',lambda dataset:state)
    assert app._sensory_model('FlyWire v783',True)[2] is state
    state.catalogue_signature='stale'
    assert app._sensory_model('FlyWire v783',True)[2] is count
