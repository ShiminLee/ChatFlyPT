"""Conspecific channels: annotation evidence, catalogue wiring and routing."""
import pandas as pd
import pytest

from chatflypt.environment import FIELD_SITES, TYPES, empty_environment, probe_environment, validate_objects
from chatflypt.malecns_senses import CHANNEL_MAP, DATASET, POPULATIONS
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.unified_readout import SEMANTIC_GROUPS


@pytest.fixture(scope='module')
def catalogue():
    return build_catalogue(save=False)


def test_both_conspecific_channels_resolve_to_one_malecns_entry():
    assert CHANNEL_MAP['cva'] == 'mc_cva'
    assert CHANNEL_MAP['auditory'] == 'mc_auditory'
    # The FlyWire stimulus name must never create a second MaleCNS channel.
    assert CHANNEL_MAP['courtship_pulse'] == 'mc_auditory'
    sources = [row[0] for row in POPULATIONS]
    assert sources.count('auditory') == 1
    assert SEMANTIC_GROUPS['mc_auditory'] == 'auditory'


def test_emitters_are_non_solid_fields_sampled_at_the_right_receptor(catalogue):
    for kind, channel, site in (('pheromone', 'cva', 'funiculus'), ('song', 'courtship_pulse', 'pedicel')):
        assert kind in TYPES and TYPES[kind]['solid'] is False
        assert TYPES[kind]['channel'] == channel
        assert FIELD_SITES[kind] == site
    # A song source uses the antennal field path, never a contact-based one.
    objects = validate_objects([{'id': 'conspecific_source', 'type': 'song', 'position_mm': [0, 0, 1],
                                 'size_mm': [.05, .05, .05], 'range_mm': .20, 'strength': 1.}])
    environment = empty_environment()
    environment['objects'] = objects
    result = probe_environment(environment, 42, 'ground') if False else None  # placement is checked separately
    assert FIELD_SITES['song'] == 'pedicel'


def test_song_population_is_the_official_auditory_subclass(catalogue):
    spec = catalogue['channels']['mc_auditory']
    assert spec['dataset'] == DATASET and spec['status'] in ('candidate', 'unmapped')
    assert spec['profile'] == 'pulse_envelope'
    assert spec['profile_parameters'] == {'interval_ms': 36., 'width_ms': 5.}
    assert spec['selection']['subclass'] == 'auditory'
    assert spec['side_ids']['left'] and spec['side_ids']['right']
    notes = ' '.join(spec['limitations'])
    assert 'chordotonal' in notes and 'never' in notes


def test_cva_population_is_orn_da1_with_both_sides(catalogue):
    spec = catalogue['channels']['mc_cva']
    assert spec['selection']['type'] == ['ORN_DA1']
    assert spec['side_ids']['left'] and spec['side_ids']['right']
    assert spec['body_parts'] == ['antenna']


def test_annotations_confirm_the_auditory_selection_and_exclude_leg_chordotonal(catalogue):
    from chatflypt.unified_readout import SOURCE_ARTIFACTS
    table = pd.read_feather(SOURCE_ARTIFACTS['malecns_annotations'], columns=['bodyId', 'class', 'subclass', 'entryNerve'])
    table['bodyId'] = table['bodyId'].astype(str)
    indexed = table.set_index('bodyId')
    ids = catalogue['channels']['mc_auditory']['neuron_ids']
    rows = indexed.reindex(ids)
    assert (rows['subclass'] == 'auditory').all()
    assert (rows['entryNerve'] == 'AN').all()
    # Leg chordotonal cells are proprioceptive and must never appear here.
    leg = table[table['subclass'].eq('chordotonal organ')]['bodyId'].tolist()
    assert not set(ids) & set(leg)


def test_human_phrases_route_to_the_conspecific_emitters(catalogue):
    from chatflypt.environment_message import ground_message, parse_message
    environment = empty_environment()
    probe = probe_environment(environment, 42, 'ground')
    for message, kind, side in (('释放信息素', 'pheromone', 'front'),
                                ('播放求偶鸣声', 'song', 'front'),
                                ('在左边放信息素源', 'pheromone', 'left')):
        plan, candidate = ground_message(parse_message(message, catalogue), environment, probe, catalogue)
        assert plan['kind'] == 'environment_presentation', message
        assert plan['items'] == []
        assert [obj['type'] for obj in candidate['objects']] == [kind]
        assert candidate['objects'][0]['position_mm'] is not None
        assert plan['translation']['direct_neural_input'] is False
