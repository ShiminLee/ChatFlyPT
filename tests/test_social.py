import copy
import numpy as np
import pytest

from chatflypt.brain import Brain
from chatflypt.social import encode, rate_envelope, resolve_message, to_engine_tape


def fixture_spec():
    return {'dataset': 'FlyWire v783', 'id_namespace': 'flywire_root_id',
            'neuron_ids': ['123', '456'], 'profile': 'pulse_envelope',
            'parameters': {'onset_ms': 50., 'end_ms': 450., 'peak_hz': 150.,
                           'width_ms': 5., 'interval_ms': 36.}}


def test_encoder_preserves_quiet_periods_timing_targets_and_seed():
    spec = fixture_spec()
    first = encode(spec, 1234)
    again = encode(spec, 1234)
    changed = encode(spec, 5678)
    np.testing.assert_array_equal(first['event_steps'], again['event_steps'])
    assert not np.array_equal(first['event_steps'], changed['event_steps'])
    assert set(first['event_neuron_ids']) <= {123, 456}
    assert len(first['event_steps']) > 0
    assert np.all(rate_envelope(spec)[first['event_steps']] > 0)
    assert np.all(np.diff(first['event_steps']) >= 0)
    assert first['event_steps'].min() >= 500
    assert first['event_steps'].max() < 4500


def test_engine_adapter_rejects_namespace_and_non_target_injection():
    brain = Brain.__new__(Brain)
    brain.version = '783'
    brain.lookup = {'123': 7, '456': 9}
    events = encode(fixture_spec(), 1234)
    inputs, ptr, ids, steps = to_engine_tape(brain, events)
    assert set(inputs) == {7, 9}
    assert set(ids) <= {7, 9}
    assert len(ptr) == steps + 1
    assert ptr[-1] == len(ids)
    bad = copy.deepcopy(events)
    bad['id_namespace'] = 'malecns_body_id'
    with pytest.raises(ValueError, match='namespace'):
        to_engine_tape(brain, bad)
    bad = copy.deepcopy(events)
    bad['event_neuron_ids'][0] = 999
    with pytest.raises(ValueError, match='outside'):
        to_engine_tape(brain, bad)


def test_text_adapter_cannot_drop_negation_or_invent_supported_semantics():
    catalogue = {'channels': {'right_foreleg_touch': {}},
                 'messages': {'模拟右侧触碰': 'right_foreleg_touch'}}
    assert resolve_message('模拟右侧触碰。', catalogue) == 'right_foreleg_touch'
    for message in ['不要模拟右侧触碰', '向左', '告诉它明天有糖']:
        with pytest.raises(ValueError):
            resolve_message(message, catalogue)


def test_invalid_signal_parameters_fail_before_encoding():
    spec = fixture_spec()
    spec['parameters']['peak_hz'] = float('nan')
    with pytest.raises(ValueError):
        encode(spec, 123)
    spec = fixture_spec()
    spec['parameters']['width_ms'] = 50.
    with pytest.raises(ValueError):
        encode(spec, 123)
