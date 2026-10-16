"""Integration checks using the installed connectomes and trained readouts.

Each supported case runs once with a fresh seed. These tests verify provenance
and isolation across the full pathway, not a classifier's statistical accuracy.
"""
import copy
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from chatflypt import server
from chatflypt.brain import Trial
from chatflypt.communication import response_text
from chatflypt.dialogue import sensation_text
from chatflypt.expression import expression_report
from chatflypt.sensory_readout import sensory_response_text, OUT as SENSORY_OUT
from chatflypt.malecns import OUT as MALE_OUT
from chatflypt.paths import ARTIFACTS, sha256
from chatflypt.reactions import summarize_reaction
from chatflypt.social import OUT as SOCIAL_OUT


CASES = {
    'right': ('有人碰了你的右前腿', 2026091901),
    'left': ('模拟左前腿触碰', 2026091902),
    'cva': ('释放信息素', 2026091903),
    'song': ('播放求偶鸣声', 2026091904),
    'five': ('我闻到了醋和酵母，尝到了糖，温暖而且明亮。', 2026091905),
    'blank': ('停止刺激', 2026091906),
}


@pytest.fixture(scope='module')
def pathway():
    required = [ARTIFACTS / 'decoder.npz', ARTIFACTS / 'stimuli.json',
                ARTIFACTS / 'graph-783.npz', SOCIAL_OUT / 'social-decoder.npz',
                MALE_OUT / 'graph.npz', MALE_OUT / 'touch-decoder.npz']
    required += [SENSORY_OUT / 'flywire-decoder.npz', SENSORY_OUT / 'malecns-decoder.npz']
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip('Integration assets are not installed: ' + ', '.join(missing))
    with threadpool_limits(limits=1):
        instance=server.Application(body_enabled=False)
        from chatflypt.environment import empty_environment
        instance.environment=empty_environment()
        yield instance


@pytest.fixture(scope='module')
def observations(pathway, tmp_path_factory):
    output = tmp_path_factory.mktemp('pathway-records')
    results = {}
    # Input/model source paths are absolute module constants. Only server output
    # destinations move, so this uses the same models and execution path as the UI.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(server, 'ROOT', output)
        patch.setattr(server, 'REPORTS', output / 'reports/reconstruction')
        for name, (message, seed) in CASES.items():
            results[name] = pathway.communicate(message, seed)
    return output, results


def model_for(app, name):
    if name in {'right', 'left'}:
        return app.male, app.sensory_decoders['MaleCNS v1.0'], app.male_view
    return app.brain, app.sensory_decoders['FlyWire v783'], app.view


@pytest.mark.parametrize('name', CASES)
def test_fresh_trial_reply_and_raw_record_have_the_same_neural_source(pathway, observations, name):
    output, results = observations
    result = results[name]
    observation = result['observation']
    brain, decoder, view = model_for(pathway, name)
    assert result['handled'] and result['pathway']['sensory_processing_completed']
    assert result['pathway']['expression_translation_completed']==bool(result['messages'])
    assert result['pathway']['completion_scope']=='bounded_neural_readout_turn'
    assert result['pathway']['fresh_neural_trial']
    assert result['pathway']['stages'] == [
        'text', 'sensory_events', 'connectome_activity', 'blind_readout', 'text']
    assert result['reply_source'] == ('neural_expression' if result['messages'] else 'expression_status')
    assert all(o['reception']['sensation']['text']==sensation_text(o['decoded'],o['reaction']) for o in result['observations'])
    assert observation['expression']==expression_report(observation)
    assert observation['response_text'] == sensory_response_text(observation['decoded'], observation['reaction'])
    assert observation['seed'] == CASES[name][1]

    path = output / observation['raw_record']['file']
    assert path.is_file() and sha256(path) == observation['raw_record']['sha256']
    with np.load(path, allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw['neuron_ids'], brain.ids)
        counts = raw['counts']
        # Saved counts must be recoverable from the saved spike event stream.
        np.testing.assert_array_equal(np.bincount(raw['spike_indices'],
                                                  minlength=len(brain.ids)), counts)
        assert int(raw['seed']) == CASES[name][1]
        assert float(raw['duration_ms']) == observation['duration_ms']
        assert len(raw['spike_indices']) == len(raw['spike_steps'])
        assert int(counts.sum()) == observation['spikes']
        assert int(np.count_nonzero(counts)) == observation['active_neurons']
        state = raw['state_record'] if 'state_record' in raw else None
        decoded_again = decoder.decode(counts,state) if hasattr(decoder,'decode_trial') else decoder.decode(counts)
        assert decoded_again['words'] == observation['decoded']['words']
        assert decoded_again['scores'].keys() == observation['decoded']['scores'].keys()
        np.testing.assert_allclose(list(decoded_again['scores'].values()),
                                   list(observation['decoded']['scores'].values()),
                                   rtol=1e-12, atol=1e-12)
        replayed_trial = Trial(counts, raw['spike_indices'], raw['spike_steps'],
                               int(raw['seed']), float(raw['duration_ms']),state)
        reaction_again = summarize_reaction(brain, replayed_trial,
                                            raw['driven_indices'], view.rows)
        assert reaction_again == observation['reaction']
        assert observation['response_text'] == sensory_response_text(decoded_again, reaction_again)
        # The chat sentence is regenerated solely from the saved neural result.
        assert observation['expression']==expression_report({
            'id':observation['id'],'dataset':observation['dataset'],
            'decoded':decoded_again,'reaction':reaction_again})
    # Actual movement is not manufactured from a sensory label or motor spikes.
    assert observation['scene']['body']['trajectory'] == []
    if name == 'blank':
        assert observation['spikes'] == observation['reaction']['non_sensory_spikes'] == 0
        assert '没有收到新的感觉信号' in observation['reception']['sensation']['text']
        assert not result['messages'] and observation['expression']['text'] is None


@pytest.mark.parametrize('name', ['right', 'cva'])
def test_installed_readouts_ignore_all_sensory_and_driven_cell_counts(pathway, observations, name):
    output, results = observations
    brain, decoder, view = model_for(pathway, name)
    sensory = view.rows.super_class.str.contains('sensory').to_numpy(copy=True)
    with np.load(output / results[name]['observation']['raw_record']['file'],
                 allow_pickle=False) as raw:
        original = raw['counts']
        sensory[raw['driven_indices']] = True
        assert sensory.any() and not sensory[decoder.indices].any()
        altered = original.copy()
        altered[sensory] += 10000
        state=raw['state_record'].copy() if 'state_record' in raw else None
    assert not np.array_equal(original, altered)
    if hasattr(decoder,'decode_trial'):
        altered_state=state.copy();altered_state[:,sensory]+=10000
        assert decoder.decode(original,state) == decoder.decode(altered,altered_state)
    else:
        np.testing.assert_array_equal(decoder.scores(original), decoder.scores(altered))
        assert decoder.decode(original) == decoder.decode(altered)


@pytest.mark.parametrize('message', [
    '你好', '不要向左', '如果触碰右前腿会怎样', '向左然后向右',
    '糖尿病', '不要释放信息素', '假设播放求偶鸣声',
])
def test_unsupported_language_without_api_never_runs_a_neural_trial(pathway, monkeypatch, message):
    def unexpected(*args, **kwargs):
        pytest.fail('Unsupported text must not start a neural trial or an API call')

    monkeypatch.setattr(server, 'local_config', lambda: {})
    monkeypatch.setattr(server, 'LLM', unexpected)
    monkeypatch.setattr(pathway.brain, 'run', unexpected)
    monkeypatch.setattr(pathway, 'observe', unexpected)
    before = set(pathway.observations)
    result = pathway.communicate(message, 2026091910)
    assert not result['handled'] and not result['neural_simulation_run']
    assert result['reply_source'] == 'input_status'
    assert set(pathway.observations) == before


def test_question_uses_selected_blind_readout_without_any_model(pathway,observations,monkeypatch):
    _,results=observations
    obs=copy.deepcopy(results['right']['observation'])
    obs['response_text']='TECHNICAL_SENTINEL'
    obs['reception']={'intent':'INPUT_SENTINEL'}
    monkeypatch.setitem(pathway.observations,obs['id'],obs)
    def forbidden(*args,**kwargs):
        pytest.fail('Question must not run a model or new neural trial')
    monkeypatch.setattr(server,'LLM',forbidden)
    monkeypatch.setattr(pathway,'stimulate',forbidden)
    result=pathway.communicate('你感受到了什么？',42,observation_id=obs['id'])
    assert result['sensation']['text']=='我感到右前腿被碰到了。'
    assert result['expression']['text']=='我感到右前腿被碰到了。'
    assert result['reply_source']=='neural_expression' and result['reused_observation']
    assert not result['llm_called'] and not result['neural_simulation_run']
    assert 'SENTINEL' not in json.dumps(result)
