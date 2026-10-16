import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from chatflypt.malecns import MaleCNS, OUT, REPORT
from chatflypt.social import to_engine_tape
from chatflypt.touch_readout import TouchDecoder, WORDS, events_for, protocol


def small_model():
    brain = MaleCNS.__new__(MaleCNS)
    brain.version = 'malecns-test'
    brain.ids = np.array([101, 202, 303])
    brain.lookup = {str(body): i for i, body in enumerate(brain.ids)}
    graph = csr_matrix(np.array([[0., 100., 0.], [0., 0., 100.], [0., 0., 0.]]))
    brain.indptr, brain.indices, brain.weights = graph.indptr, graph.indices, graph.data
    brain.rows = pd.DataFrame({'superclass': ['vnc_sensory', 'vnc_intrinsic', 'vnc_motor']})
    brain.nonsensory = np.array([False, True, True])
    return brain


def one_event():
    return {'dataset': 'MaleCNS v1.0', 'id_namespace': 'malecns_body_id',
            'input_neuron_ids': np.array([101]), 'event_neuron_ids': np.array([101]),
            'event_steps': np.array([0]), 'dt_ms': .1, 'duration_ms': 50., 'seed': 1}


def test_malecns_uses_separate_namespace_and_real_graph_propagation():
    model = small_model()
    events = one_event()
    trial = model.run_events(events)
    assert trial.counts[0] == 1
    assert np.all(trial.counts[1:] > 0)
    first = {i: trial.spike_steps[trial.spike_ids == i].min() for i in range(3)}
    assert first[0] < first[1] < first[2]
    assert first[1] - first[0] >= 18
    # Delete the sensory source's output; it still receives its external input.
    model.weights[:model.indptr[1]] = 0
    disconnected = model.run_events(events)
    assert disconnected.counts[0] == 1
    assert not disconnected.counts[1:].any()
    bad = copy.deepcopy(events)
    bad['dataset'], bad['id_namespace'] = 'FlyWire v783', 'flywire_root_id'
    with pytest.raises(ValueError, match='namespace'):
        to_engine_tape(model, bad)
    for duration in [0, -.1, .15, float('nan')]:
        bad = copy.deepcopy(events)
        bad['duration_ms'] = duration
        with pytest.raises(ValueError, match='Duration'):
            model.run_events(bad)


def test_touch_protocol_holds_out_seeds_and_strengths():
    cases = protocol()['cases']
    assert len(cases) == 160
    assert len({c['seed'] for c in cases}) == 160
    for case in cases:
        if case['split'] == 'ood':
            assert case['scale'] in [.6, 1.4]
        else:
            assert .85 <= case['scale'] <= 1.15
    blank = events_for({'channels': {}}, [], 123)
    assert blank['input_neuron_ids'].size == blank['event_steps'].size == 0
    assert blank['id_namespace'] == 'malecns_body_id'


def test_touch_readout_is_blind_to_sensory_count_changes(tmp_path):
    artifact = tmp_path / 'decoder.npz'
    np.savez(artifact, indices=[1], weights=[[1.], [-1.]], intercept=[0., 0.],
             thresholds=[.5, .5], words=WORDS, neuron_count=3)
    decoder = TouchDecoder(artifact)
    assert decoder.decode(np.array([9000, 1, 0]))['words'] == [WORDS[0]]
    np.testing.assert_array_equal(decoder.scores(np.array([0, 1, 0])),
                                  decoder.scores(np.array([9000, 1, 0])))
    assert decoder.decode(np.zeros(3))['words'] == []
    with pytest.raises(ValueError, match='size'):
        decoder.decode(np.zeros(4))


def test_trained_touch_features_exclude_every_sensory_class_and_input():
    if not (OUT / 'touch-decoder.npz').exists():
        pytest.skip('Train touch decoder first')
    model = MaleCNS()
    decoder = TouchDecoder()
    plan = json.loads((REPORT / 'touch-protocol.json').read_text(encoding='utf8'))
    assert model.nonsensory[decoder.indices].all()
    inputs = {model.lookup[i] for word in WORDS for i in plan['channel_specs'][word]['neuron_ids']}
    assert not inputs.intersection(decoder.indices)
    counts = np.zeros(len(model.ids))
    altered = counts.copy()
    altered[~model.nonsensory] = 1000
    altered[list(inputs)] = 1000
    np.testing.assert_array_equal(decoder.scores(counts), decoder.scores(altered))
