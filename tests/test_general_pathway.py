"""General sensory composition and provenance; no held-out language example."""
import json
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from chatflypt import server
from chatflypt.paths import ROOT
from chatflypt.sensory_readout import SensoryDecoder, sensory_response_text


@pytest.fixture(scope='module')
def app():
    if not all((ROOT/f'data/derived/general-sensory/{key}-decoder.npz').exists()
               for key in ['flywire','malecns']):
        pytest.skip('General readouts are not trained')
    with threadpool_limits(limits=1):
        instance=server.Application(body_enabled=False)
        from chatflypt.environment import empty_environment
        instance.environment=empty_environment()
        yield instance


def test_cross_dataset_inputs_keep_distinct_event_and_neuron_spaces(app,tmp_path,monkeypatch):
    monkeypatch.setattr(server,'ROOT',tmp_path)
    result=app.stimulate([
        {'channel':'cold','side':'left','intensity':1,'duration_ms':200},
        {'channel':'hindleg_touch','side':'right','intensity':1,'onset_ms':400,'duration_ms':250},
    ],812301)
    assert result['handled'] and result['neural_simulation_run']
    assert result['pathway']['independent_datasets']
    assert len(result['observations'])==2
    assert result['reception']['neuron_ids']==[]
    assert sum(map(len,result['reception']['neuron_ids_by_dataset'].values()))==result['reception']['neuron_count']
    assert {o['dataset'] for o in result['observations']}=={'FlyWire v783','MaleCNS v1.0'}
    for obs in result['observations']:
        assert obs['duration_ms']==750
        with np.load(tmp_path/obs['event_record']['file'],allow_pickle=False) as events:
            assert str(events['dataset'])==obs['dataset']
            assert np.all(events['event_steps']<7500)
            assert set(events['event_neuron_ids'])<=set(events['input_neuron_ids'])
        with np.load(tmp_path/obs['raw_record']['file'],allow_pickle=False) as record:
            decoder=app.sensory_decoders[obs['dataset']]
            if hasattr(decoder,'decode_trial'):
                assert decoder.decode(record['counts'],record['state_record'])==obs['decoded']
            else:
                assert isinstance(decoder,SensoryDecoder)
                assert decoder.decode(record['counts'])==obs['decoded']
            assert sensory_response_text(obs['decoded'],obs['reaction'])==obs['response_text']
            assert not set(decoder.indices)&set(record['driven_indices'])
        assert obs['scene']['body']['trajectory']==[]


def test_unmapped_stimulus_is_not_sent_to_configured_chat_api(app,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('An unmapped sensory request must not trigger LLM or neural execution')
    monkeypatch.setattr(server,'local_config',lambda:{'OPENAI_API_KEY':'fake-key','CHATFLYPT_MODEL':'fake-model'})
    monkeypatch.setattr(app,'chat',forbidden)
    monkeypatch.setattr(app,'stimulate',forbidden)
    response=app.communicate('施加疼痛刺激',812302)
    assert response['reply_source']=='input_status'
    assert response['input_plan']['kind']=='unsupported'
    assert not response['neural_simulation_run']


def test_zero_intensity_has_no_neural_or_interpreted_response(app,tmp_path,monkeypatch):
    monkeypatch.setattr(server,'ROOT',tmp_path)
    result=app.stimulate([{'channel':'bitter','side':'left','intensity':0}],812303)
    observation=result['observation']
    assert observation['spikes']==0
    assert observation['decoded']['words']==[]
    assert observation['reception']['input_events']==0
    assert '没有记录到' in result['reply']


def test_advanced_annotation_input_runs_without_fabricated_semantic_readout(app,tmp_path,monkeypatch):
    monkeypatch.setattr(server,'ROOT',tmp_path)
    spec=next(s for s in app.input_catalogue()['channels'].values()
              if s.get('advanced') and s['status']!='unmapped' and s['side_ids']['bilateral'])
    result=app.stimulate([{'channel':spec['id'],'side':'bilateral','intensity':1}],812304)
    assert result['handled'] and result['neural_simulation_run']
    assert not result['pathway']['completed']
    assert result['observation']['decoded']['status']=='uncalibrated'
    assert result['observation']['decoded']['words']==[]


def test_public_catalogue_exposes_status_without_large_id_tables(app):
    public=app.inputs()
    assert len(public['channels'])>100
    assert all('neuron_ids' not in spec and 'side_ids' not in spec for spec in public['channels'])
    for dataset,row in public['coverage'].items():
        assert row['accounted_cells']==row['sensory_cells_in_engine']
        assert row['not_executable_cells']>0
        assert not any(k.endswith('_ids') for k in row)


@pytest.mark.parametrize('items',[
    [{'channel':'co2','side':'right','intensity':float('nan')}],
    [{'channel':'co2','side':'front'}],
    [{'channel':'hunger','side':'bilateral'}],
    [{'channel':'co2','side':'right','onset_ms':4000,'duration_ms':2000}],
])
def test_invalid_stimulus_cannot_produce_partial_trial(app,monkeypatch,items):
    monkeypatch.setattr(server,'recorded_trial',lambda *args:pytest.fail('invalid event input reached simulator'))
    before=set(app.observations)
    with pytest.raises((ValueError,TypeError)):
        app.stimulate(items,812305)
    assert set(app.observations)==before
