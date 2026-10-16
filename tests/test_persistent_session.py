"""Continuity includes latent neural state, body state and input provenance."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from chatflypt.neural_session import NeuralSession, tape_window, replay_events
from chatflypt.embodied import EmbodiedSession, run_embodied


def test_observation_boundary_does_not_reset_delayed_activity_or_count_old_spikes():
    graph=csr_matrix(([100.],([0],[1])),shape=(2,2))
    brain=SimpleNamespace(ids=np.arange(2),indptr=graph.indptr,indices=graph.indices,weights=graph.data)
    tape=(np.array([0],np.int32),np.r_[0,np.ones(40,np.int64)],np.array([0],np.int32),40)
    split=NeuralSession(brain,3);whole=NeuralSession(brain,3)
    split.run(tape_window(tape,0,10));first=split.observation_trial()
    voltage=split.voltage.copy();queue=split.queue.copy()
    split.begin_observation()
    np.testing.assert_array_equal(split.voltage,voltage);np.testing.assert_array_equal(split.queue,queue)
    split.run(tape_window(tape,10,40));second=split.observation_trial();whole.run(tape)
    assert first.duration_ms==1 and second.duration_ms==3
    assert first.counts[1]==0 and second.counts[1]>0
    for name in ('counts','spike_ids','spike_steps','state_record'):
        np.testing.assert_array_equal(getattr(split.trial(),name),getattr(whole.trial(),name))
    np.testing.assert_array_equal(first.counts+second.counts,whole.counts)


def test_session_registry_never_mixes_dataset_environment_or_closed_state():
    from chatflypt.server import Application
    app=Application.__new__(Application);env={'revision':1,'objects':[]};sid='a'*32
    male,_=app.embodied_session(sid,'MaleCNS v1.0',env,'ground')
    male.probe={'identity':'male'}
    fly,_=app.embodied_session(sid,'FlyWire v783',env,'ground')
    fly.probe={'identity':'fly'}
    assert app.session_probe(sid,env,'ground','MaleCNS v1.0')=={'identity':'male'}
    assert app.session_probe(sid,env,'ground','FlyWire v783')=={'identity':'fly'}
    replacement,reason=app.embodied_session(sid,'MaleCNS v1.0',dict(env,revision=2),'ground')
    assert not male.closed and replacement is male and reason is None
    assert replacement.environment_update['state_preserved']
    assert not fly.closed
    replacement.close()
    with pytest.raises(ValueError,match='停止'):app.embodied_session(sid,'MaleCNS v1.0',dict(env,revision=2),'ground')
    with pytest.raises(ValueError,match='ID'):app._session_map('../invalid')


def test_partial_failed_turn_invalidates_all_dataset_states(monkeypatch):
    from chatflypt.server import Application
    from chatflypt import conversation
    app=Application.__new__(Application);env={'revision':1,'objects':[]};sid='b'*32
    male,_=app.embodied_session(sid,'MaleCNS v1.0',env,'ground')
    fly,_=app.embodied_session(sid,'FlyWire v783',env,'ground')
    male.neural=SimpleNamespace(step=10);fly.neural=SimpleNamespace(step=20)
    def failed(*args,**kwargs):
        fly.neural.step+=10
        raise ValueError('Second dataset failed')
    monkeypatch.setattr(conversation,'run_turn',failed)
    with pytest.raises(ValueError,match='Second'):app.converse(session_id=sid)
    assert male.closed and fly.closed


def test_refresh_restores_record_and_new_chat_closes_state(tmp_path,monkeypatch):
    import threading
    from chatflypt import server
    app=server.Application.__new__(server.Application);app.session_lock=threading.RLock()
    env={'revision':1,'objects':[]};sid='c'*32
    state,_=app.embodied_session(sid,'MaleCNS v1.0',env,'ground')
    record={'dialogue':{'session_id':sid},'observation':{'id':'recorded'}}
    (tmp_path/'response.json').write_text(json.dumps(record),encoding='utf8')
    monkeypatch.setattr(server,'ROOT',tmp_path);app.session_results={sid:'response.json'}
    assert app.session_status(sid)['response']==record
    assert app.reset_session(sid)['neural_and_body_reset'] and state.closed
    assert sid not in app.embodied_sessions and sid not in app.session_results


def test_real_two_messages_equal_one_uninterrupted_physical_trial(tmp_path):
    from chatflypt.malecns import MaleCNS
    from chatflypt.sensory_catalogue import build_catalogue
    from chatflypt.environment import validate_objects
    from chatflypt.paths import ROOT
    brain=MaleCNS();catalogue=build_catalogue(save=False)
    env={'schema':'chatflypt.environment.v1','revision':1,'objects':validate_objects([
        {'id':'continuous_sugar','type':'sugar','position_mm':[0,0,.1],'size_mm':[5,4,.2]}])}
    folder=ROOT/'reports/persistent-session';folder.mkdir(parents=True,exist_ok=True)
    state=EmbodiedSession(env,'ground')
    try:
        first=run_embodied(brain,brain.rows,catalogue,'MaleCNS v1.0',[],env,91505,'ground',
            duration_ms=100,output_dir=folder/'first',session=state)
        second=run_embodied(brain,brain.rows,catalogue,'MaleCNS v1.0',[],env,91506,'ground',
            duration_ms=200,output_dir=folder/'second',session=state)
    finally:state.close()
    reference=run_embodied(brain,brain.rows,catalogue,'MaleCNS v1.0',[],env,91505,'ground',
        duration_ms=300,output_dir=folder/'uninterrupted')
    assert second['feedback']['session_start_ms']==100 and second['feedback']['session_end_ms']==300
    assert second['feedback']['neural_reset_count']==second['feedback']['body_reset_count']==0
    assert first['body']['initialization']['performed'] and not first['body']['initialization']['resumed']
    assert not second['body']['initialization']['performed'] and second['body']['initialization']['resumed']
    assert second['feedback']['initial_neural_state_sha256']==first['feedback']['neural_state_sha256']
    np.testing.assert_array_equal(first['body']['trajectory'][-1]['position_mm'],second['body']['trajectory'][0]['position_mm'])
    for name in ('counts','spike_ids','spike_steps','state_record'):
        np.testing.assert_array_equal(getattr(state.neural.trial(),name),getattr(reference['trial'],name))
    np.testing.assert_array_equal(first['trial'].counts+second['trial'].counts,reference['trial'].counts)
    np.testing.assert_array_equal(second['body']['trajectory'][-1]['position_mm'],reference['body']['trajectory'][-1]['position_mm'])
    with np.load(folder/'second/input-events.npz',allow_pickle=False) as raw:
        events={k:raw[k].item() if raw[k].ndim==0 else raw[k] for k in raw.files}
    replay=replay_events(brain,events)
    for name in ('counts','spike_ids','spike_steps','state_record'):
        np.testing.assert_array_equal(getattr(replay,name),getattr(state.neural.trial(),name))
    result={'neural_bitwise_equal_to_uninterrupted':True,'physical_position_bitwise_equal_to_uninterrupted':True,
        'complete_input_prefix_replay_identical':True,'second_turn_neural_resets':0,'second_turn_body_resets':0,
        'initial_state_hash_matches_previous_final':True,'session_duration_ms':300,
        'spikes':int(replay.counts.sum()),'first_spikes':int(first['trial'].counts.sum()),
        'second_spikes':int(second['trial'].counts.sum()),'second_body':second['body']['summary']}
    (folder/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf8')
