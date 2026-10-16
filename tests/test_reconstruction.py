import json
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from chatflypt.brain import simulate, Brain
from chatflypt.stimuli import FIRST_SENTENCE, WORDS, groups_for
from chatflypt.language import verbalize
from chatflypt.experiment import protocol, Decoder
from chatflypt.demo import REFERENCE_SENTENCE
from chatflypt.paths import ARTIFACTS


def test_zero_input_no_spontaneous_spikes():
    graph=csr_matrix(np.array([[0,100.0],[100.0,0]]))
    result=simulate(graph.indptr,graph.indices,graph.data,np.array([],np.int32),
                    np.zeros(101,np.int64),np.array([],np.int32),100,True)
    assert not result[0].any()


def test_engine_matches_brian_on_recurrent_inhibitory_graph():
    from chatflypt.verify import verify_small
    assert verify_small()['passed']


def test_protocol_no_seed_or_mixture_leakage():
    cases=protocol()['cases']
    assert len({c['seed'] for c in cases})==len(cases)
    seen={}
    for case in cases:
        if case['kind']!='mixture':continue
        combination=tuple(case['words'])
        assert combination!=tuple(sorted(FIRST_SENTENCE))
        assert combination not in seen
        seen[combination]=case['split']


def test_sentence_contains_only_decoded_evidence():
    assert verbalize(FIRST_SENTENCE)==REFERENCE_SENTENCE
    assert 'vinegar' not in verbalize(['sugar'])
    assert 'sugar' not in verbalize([])
    assert 'cannot identify' in verbalize([])


def test_input_mixture_overlap_explicitly_sums():
    manifest={'words':{'a':{'neuron_ids':['123'],'rate_hz':50},
                       'b':{'neuron_ids':['123','456'],'rate_hz':100}}}
    assert groups_for(manifest,['a','b'])==[(['123'],150.0),(['456'],100.0)]
    with pytest.raises(ValueError):groups_for(manifest,['a','a'])


def test_version_specific_missing_id_rejected():
    brain=Brain.__new__(Brain);brain.lookup={'123':0}
    with pytest.raises(KeyError):brain.input_tape([(['missing'],100)],100,1)


def test_decoder_excludes_all_sensory_and_input_neurons():
    if not (ARTIFACTS/'decoder.npz').exists():pytest.skip('Train model first')
    manifest=json.loads((ARTIFACTS/'stimuli.json').read_text(encoding='utf8'))
    decoder=Decoder()
    assert set(decoder.indices)<=set(manifest['readout_indices'])
    brain=Brain()
    inputs={brain.lookup[i] for s in manifest['words'].values() for i in s['neuron_ids']}
    assert not inputs.intersection(decoder.indices)
    baseline=np.zeros(len(brain.ids))
    altered=baseline.copy();altered[list(inputs)]=1000
    np.testing.assert_array_equal(decoder.scores(baseline),decoder.scores(altered))


def test_humidity_targets_match_independent_moist_annotation():
    """Prevent the old VP1l cooling / moist biological mapping regression."""
    import pandas as pd
    from chatflypt.stimuli import ANNOTATIONS
    manifest=json.loads((ARTIFACTS/'stimuli.json').read_text(encoding='utf8'))
    annotation=pd.read_csv(ANNOTATIONS,sep='\t',low_memory=False)
    selected=annotation[annotation.root_id.astype(str).isin(manifest['words']['humid']['neuron_ids'])]
    assert len(selected)>0
    assert selected.cell_sub_class.eq('moist').all()
    assert not set(selected.root_id.astype(str)) & set(manifest['words']['cold']['neuron_ids'])


def test_decoder_saved_artifact_roundtrip(tmp_path):
    path=tmp_path/'readout.npz'
    np.savez(path,indices=[0,2],weights=np.ones((16,2)),intercept=np.zeros(16),thresholds=np.ones(16))
    d=Decoder(path)
    assert d.decode(np.array([0,9000,0]))['words']==[]
    assert d.decode(np.array([1,0,0]))['words']==WORDS


def test_missing_chatgpt_credentials_do_not_fabricate_reply(monkeypatch):
    import chatflypt.language as language
    monkeypatch.setattr(language,'local_config',lambda:{})
    with pytest.raises(language.LLMUnavailable):language.LLM(provider='openai')


def test_openai_response_parsing_and_secret_not_in_process_args(monkeypatch):
    import chatflypt.language as language
    from types import SimpleNamespace
    monkeypatch.setattr(language,'local_config',lambda:{'OPENAI_API_KEY':'test-secret',
        'CHATFLYPT_MODEL':'test-gpt','OPENAI_BASE_URL':'https://api.openai.com/v1'})
    def run(args,**kwargs):
        assert 'test-secret' not in ' '.join(args)
        assert '/v1/v1/' not in kwargs['input']
        assert '/v1/responses' in kwargs['input']
        return SimpleNamespace(returncode=0,stdout=json.dumps({'id':'test-response','model':'test-gpt',
            'output':[{'type':'reasoning','summary':[]},{'type':'message','content':[{'type':'output_text','text':'left'}]}]}))
    monkeypatch.setattr(language.subprocess,'run',run)
    result=language.LLM(provider='openai').complete('Wall ahead','One direction only')
    assert result['text']=='left'
    assert 'test-secret' not in json.dumps(result)


def test_api_failure_cannot_be_counted_as_a_model_answer(monkeypatch):
    import chatflypt.language as language
    from types import SimpleNamespace
    monkeypatch.setattr(language,'local_config',lambda:{'OPENAI_API_KEY':'test-secret','CHATFLYPT_MODEL':'test-gpt'})
    monkeypatch.setattr(language.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,
        stdout=json.dumps({'error':{'type':'invalid_api_key','message':'secret-response-body'}})))
    with pytest.raises(language.LLMUnavailable,match='invalid_api_key') as exc:
        language.LLM(provider='openai').complete('test','test')
    assert 'secret-response-body' not in str(exc.value)
