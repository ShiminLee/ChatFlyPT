import copy

import numpy as np
import pandas as pd
import pytest

from chatflypt.sensory_catalogue import build_catalogue, FW, MC
from chatflypt.sensory_events import encode_inputs
from chatflypt.social import MALE, to_engine_tape
from chatflypt.stimuli import ANNOTATIONS


@pytest.fixture(scope='module')
def catalogue():
    return build_catalogue(save=False)


def miniature_catalogue():
    one = {'id': 'a', 'dataset': FW, 'id_namespace': 'flywire_root_id',
           'status': 'candidate', 'neuron_ids': ['101', '202'],
           'side_ids': {'left': ['101'], 'right': ['202'], 'bilateral': ['101', '202']},
           'body_part_ids': {'foreleg': {'left': ['101'], 'right': [], 'bilateral': ['101']}},
           'profile': 'tonic', 'rate_hz': 500., 'profile_parameters': {}}
    other = copy.deepcopy(one)
    other.update(id='b', dataset=MC, id_namespace='malecns_body_id')
    unmapped = copy.deepcopy(one)
    unmapped.update(id='unknown', status='unmapped')
    return {'channels': {'a': one, 'b': other, 'unknown': unmapped}}


def test_real_registry_accounts_for_sensory_cells_without_fabricating_ids(catalogue):
    fly = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False).fillna('')
    male = pd.read_feather(MALE).fillna('')
    all_ids = {FW: set(fly.loc[fly.super_class.str.contains('sensory'), 'root_id'].astype(str)),
               MC: set(male.loc[male.superclass.str.contains('sensory'), 'bodyId'].astype(str))}
    for dataset, coverage in catalogue['coverage'].items():
        assert coverage['accounted_cells'] == coverage['sensory_cells_in_engine']
        assert coverage['unaccounted_ids'] == []
        assert coverage['not_executable_cells'] > 0
    for spec in catalogue['channels'].values():
        assert set(spec['neuron_ids']) <= all_ids[spec['dataset']]
        assert set(spec['side_ids']['left']).isdisjoint(spec['side_ids']['right'])
        assert set(spec['side_ids']['left']) | set(spec['side_ids']['right']) <= set(spec['side_ids']['bilateral'])
        assert set(spec['side_ids']['bilateral']) <= set(spec['neuron_ids'])
        assert not set(spec['unassigned_side_ids']) & set(spec['side_ids']['bilateral'])
        if spec['status'] != 'unmapped':
            assert spec['neuron_ids'] and spec['side_ids']['bilateral']
    for word in ['hunger', 'thirst', 'nociception']:
        assert catalogue['channels'][word]['status'] == 'unmapped'
        assert not catalogue['channels'][word]['neuron_ids']


def test_sugar_water_and_lateralization_follow_annotations_not_old_unilateral_lists(catalogue):
    fly = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False).fillna('')
    fly.index = fly.root_id.astype(str)
    sweet = catalogue['channels']['sweet_water']
    selected = fly.loc[sweet['neuron_ids']]
    assert selected.cell_sub_class.eq('sugar/water').all()
    for side in ['left', 'right']:
        assert sweet['side_ids'][side]
        assert fly.loc[sweet['side_ids'][side], 'side'].eq(side).all()
    moist = fly.loc[catalogue['channels']['moist']['neuron_ids']]
    assert moist.cell_sub_class.eq('moist').all()
    assert not set(moist.root_id.astype(str)) & set(fly.loc[fly.cell_type.eq('HRN_VP1l'), 'root_id'].astype(str))
    male = pd.read_feather(MALE).fillna('')
    male.index = male.bodyId.astype(str)
    for channel, nerve in [('foreleg_touch', 'ProLN'), ('midleg_touch', 'MesoLN'), ('hindleg_touch', 'MetaLN')]:
        spec = catalogue['channels'][channel]
        for side, code in [('left', 'L'), ('right', 'R')]:
            selected = male.loc[spec['side_ids'][side]]
            assert selected.rootSide.eq(code).all()
            assert selected.entryNerve.eq(nerve).all()


def test_events_preserve_real_sides_temporal_windows_and_dataset_boundaries():
    cat = miniature_catalogue()
    items = [{'channel': 'a', 'side': 'left', 'onset_ms': 1800., 'duration_ms': 500.},
             {'channel': 'b', 'side': 'right', 'onset_ms': 20., 'duration_ms': 50.}]
    encoded = encode_inputs(cat, items, 821)
    assert set(encoded) == {FW, MC}
    assert encoded[FW]['duration_ms'] == encoded[MC]['duration_ms'] == 2400.
    assert set(encoded[FW]['event_neuron_ids']) == {101}
    assert set(encoded[MC]['event_neuron_ids']) == {202}
    assert encoded[FW]['event_steps'].min() >= 18000
    assert encoded[FW]['event_steps'].max() < 23000
    assert encoded[MC]['event_steps'].min() >= 200
    assert encoded[MC]['event_steps'].max() < 700
    assert encoded[FW]['id_namespace'] != encoded[MC]['id_namespace']
    reordered = encode_inputs(cat, list(reversed(items)), 821)
    for name in encoded:
        for field in ['input_neuron_ids', 'event_steps', 'event_neuron_ids']:
            np.testing.assert_array_equal(encoded[name][field], reordered[name][field])


def test_overlap_is_combined_before_sampling_and_body_parts_are_not_broadened():
    cat = miniature_catalogue()
    items = [{'channel': 'a', 'side': 'left', 'intensity': .5},
             {'channel': 'a', 'side': 'left', 'intensity': .5}]
    mixture = encode_inputs(cat, items, 61)[FW]
    once = encode_inputs(cat, [{'channel': 'a', 'side': 'left', 'intensity': 1.}], 61)[FW]
    np.testing.assert_array_equal(mixture['event_steps'], once['event_steps'])
    np.testing.assert_array_equal(mixture['event_neuron_ids'], once['event_neuron_ids'])
    restricted = encode_inputs(cat, [{'channel': 'a', 'side': 'bilateral', 'body_part': 'foreleg'}], 61)[FW]
    assert set(restricted['input_neuron_ids']) == {101}
    with pytest.raises(ValueError, match='No annotated'):
        encode_inputs(cat, [{'channel': 'a', 'side': 'right', 'body_part': 'foreleg'}], 61)
    with pytest.raises(ValueError, match='Body part'):
        encode_inputs(cat, [{'channel': 'a', 'body_part': 'wing'}], 61)


def test_zero_input_and_unmapped_or_invalid_requests_do_not_stimulate():
    cat = miniature_catalogue()
    assert encode_inputs(cat, [], 0) == {}
    silent = encode_inputs(cat, [{'channel': 'a', 'intensity': 0}], 0)[FW]
    assert len(silent['input_neuron_ids']) == len(silent['event_steps']) == 0
    bad_items = [{'channel': 'unknown'}, {'channel': 'missing'}, {'channel': 'a', 'side': 'unknown'},
                 {'channel': 'a', 'intensity': 3}, {'channel': 'a', 'intensity': float('nan')},
                 {'channel': 'a', 'duration_ms': 40}, {'channel': 'a', 'duration_ms': 2001},
                 {'channel': 'a', 'onset_ms': 4001}, {'channel': 'a', 'onset_ms': 4000, 'duration_ms': 1000},
                 {'channel': 'a', 'onset_ms': 50.05}, {'channel': 'a', 'direction': 'left'}]
    for item in bad_items:
        with pytest.raises(ValueError):
            encode_inputs(cat, [item], 0)


def test_pulse_envelope_and_tape_adapter_agree_on_event_timing():
    cat = miniature_catalogue()
    cat['channels']['a'].update(profile='pulse_envelope', profile_parameters={'interval_ms': 36., 'width_ms': 5.})
    events = encode_inputs(cat, [{'channel': 'a', 'side': 'left'}], 72)[FW]
    assert len(events['event_steps']) > 0
    assert np.all((events['event_steps'] - 500) % 360 < 50)
    from types import SimpleNamespace
    brain = SimpleNamespace(dataset=FW, id_namespace='flywire_root_id', version='783', lookup={'101': 0, '202': 1})
    inputs, ptr, ids, steps = to_engine_tape(brain, events)
    assert inputs.tolist() == [0]
    assert ptr[-1] == len(events['event_steps'])
    assert set(ids) == {0} and steps == 5000
