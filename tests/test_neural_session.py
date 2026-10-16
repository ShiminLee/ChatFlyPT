from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from chatflypt.brain import simulate
from chatflypt.neural_session import NeuralSession, tape_window


@pytest.mark.parametrize('cuts', [[1, 2, 18, 19, 22, 35, 70, 180, 250], [100, 200, 250], [250]])
def test_chunk_boundaries_preserve_delays_refractory_and_subthreshold_state(cuts):
    # Recurrent excitation/inhibition and late input cross queue/refractory boundaries.
    graph=csr_matrix(([44.,-12.,26.,12.,32.],([0,1,2,3,1],[1,2,3,1,0])),shape=(4,4))
    brain=SimpleNamespace(ids=np.arange(4),indptr=graph.indptr,indices=graph.indices,weights=graph.data)
    times=np.array([0,16,17,18,20,33,99,119,198,230])
    ptr=np.r_[0,np.cumsum(np.bincount(times,minlength=250))].astype(np.int64)
    tape=(np.array([0],np.int32),ptr,np.zeros(len(times),np.int32),250)
    reference_state=np.zeros((4,4))
    reference=simulate(graph.indptr,graph.indices,graph.data,*tape,True,reference_state)
    session=NeuralSession(brain,7);previous=0
    for end in cuts:
        trial=session.run(tape_window(tape,previous,end));previous=end
    for actual,expected in [(trial.counts,reference[0]),(trial.spike_ids,reference[1]),
            (trial.spike_steps,reference[2]),(session.voltage,reference[3]),
            (session.conductance,reference[4]),(trial.state_record,reference_state)]:
        np.testing.assert_array_equal(actual,expected)
    assert trial.duration_ms==25 and trial.counts[1:].sum()>0


def test_input_after_pause_updates_same_state_instead_of_resetting():
    graph=csr_matrix(([100.],([0],[1])),shape=(2,2))
    brain=SimpleNamespace(ids=np.arange(2),indptr=graph.indptr,indices=graph.indices,weights=graph.data)
    session=NeuralSession(brain,1)
    session.run((np.array([0],np.int32),np.array([0,1],np.int64),np.array([0],np.int32),1))
    assert session.voltage[0]>-45
    trial=session.run((np.array([],np.int32),np.zeros(41,np.int64),np.array([],np.int32),40))
    assert trial.spike_steps[0]==1 and trial.counts[1]>0


def test_feedback_window_never_repeats_human_cue_or_crosses_id_namespaces():
    from chatflypt.embodied import combine_window
    brain=SimpleNamespace(version='783',lookup={'1':0,'2':1})
    def events(times,ids):
        return {'dataset':'FlyWire v783','id_namespace':'flywire_root_id',
            'input_neuron_ids':np.array(sorted(set(ids))),
            'event_steps':np.array(times,np.int32),'event_neuron_ids':np.array(ids,np.int64)}
    human=events([2,11,19],[1,1,1]);environmental=events([2,4],[2,2])
    tape,times,origins,_=combine_window(brain,human,environmental,10,20,2,42,'FlyWire v783','flywire_root_id')
    assert times.tolist()==[11,12,14,19] and origins.tolist()==[1,2,2,1]
    assert tape[3]==10
    environmental['dataset']='MaleCNS v1.0'
    with pytest.raises(ValueError,match='namespaces'):
        combine_window(brain,human,environmental,10,20,2,42,'FlyWire v783','flywire_root_id')


def test_late_receptor_activation_is_replayable_without_early_refractory_change():
    from chatflypt.neural_session import replay_events
    from chatflypt.social import to_engine_tape
    graph=csr_matrix(([180.,80.],([0,1],[1,0])),shape=(2,2))
    brain=SimpleNamespace(ids=np.array([1,2]),version='783',lookup={'1':0,'2':1},
        indptr=graph.indptr,indices=graph.indices,weights=graph.data)
    events={'dataset':'FlyWire v783','id_namespace':'flywire_root_id','seed':7,'dt_ms':.1,'duration_ms':20.,
        'input_neuron_ids':np.array([1,2]),'input_activation_steps':np.array([0,80]),
        'event_steps':np.array([0,5,10,15,60,82,90]),'event_neuron_ids':np.array([1,1,1,1,1,2,2])}
    tape=to_engine_tape(brain,events);session=NeuralSession(brain,7)
    _,ptr,ids,steps=tape_window(tape,0,80);session.run((np.array([0],np.int32),ptr,ids,steps))
    assert session.refractory[1]==22 and session.input_activation_step[1]==-1
    expected=session.run(tape_window(tape,80,200));actual=replay_events(brain,events)
    assert session.input_activation_step.tolist()==[0,80]
    for name in ('counts','spike_ids','spike_steps','state_record'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
    bad=dict(events,input_activation_steps=np.array([0,95]))
    with pytest.raises(ValueError,match='precedes'):replay_events(brain,bad)
