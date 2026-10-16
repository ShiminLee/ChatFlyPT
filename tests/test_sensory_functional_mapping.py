"""Functional candidates must retain their independently annotated anatomy."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from chatflypt.paths import ROOT
from chatflypt.sensory_catalogue import MC, FW, build_catalogue
from chatflypt.sensory_events import encode_inputs
from chatflypt.social import MALE


@pytest.fixture(scope='module')
def mapping():
    snapshot = ROOT / 'data/derived/sensory/catalogue.json'
    before = hashlib.sha256(snapshot.read_bytes()).hexdigest() if snapshot.exists() else None
    result = build_catalogue(save=False)
    after = hashlib.sha256(snapshot.read_bytes()).hexdigest() if snapshot.exists() else None
    assert before == after, 'Reading the revised registry must not overwrite the frozen training snapshot'
    return result


def test_new_taste_candidates_use_disjoint_real_subtypes_and_entry_sides(mapping):
    male = pd.read_feather(MALE).fillna('')
    male.index = male.bodyId.astype(str)
    populations = {}
    for cid, types, expected_counts in [
        ('water_specific', ['LB3a'], (9, 8)),
        ('sugar_specific', ['LB3b', 'LB3c'], (17, 17)),
        ('high_salt', ['LB3d'], (11, 15)),
        ('fatty_acid_taste', ['claw_tpGRN'], (25, 25)),
    ]:
        spec = mapping['channels'][cid]
        populations[cid] = set(spec['neuron_ids'])
        expected = male[male['class'].eq('gustatory') & male.type.isin(types)]
        assert populations[cid] == set(expected.index)
        assert spec['dataset'] == MC
        assert spec['status'] == 'candidate' and not spec['advanced']
        assert spec['evidence']['individual_receptor_measurements'] is False
        for side, code, count in zip(['left', 'right'], ['L', 'R'], expected_counts):
            assert len(spec['side_ids'][side]) == count
            assert male.loc[spec['side_ids'][side], 'rootSide'].eq(code).all()
        assert populations[cid] == set(spec['side_ids']['bilateral'])
    for a, b in [('water_specific', 'sugar_specific'), ('water_specific', 'high_salt'),
                 ('sugar_specific', 'high_salt')]:
        assert populations[a].isdisjoint(populations[b])
    assert all(populations['fatty_acid_taste'].isdisjoint(populations[name])
               for name in ['water_specific', 'sugar_specific', 'high_salt'])
    assert mapping['channels']['sugar_specific']['missing_entry_nerve_ids'] == ['957530']
    # Existing broad population and aliases remain a distinct historical input.
    broad = mapping['channels']['sweet_water']
    assert broad['dataset'] == FW and len(broad['neuron_ids']) == 129
    assert not {'糖', '水'} & set(broad['aliases'])
    from chatflypt.input_language import parse_input
    assert parse_input('左侧糖和左侧水',mapping)['items']==[
        {'channel':'sugar_specific','side':'left','intensity':1.,'onset_ms':50.,'duration_ms':300.},
        {'channel':'water_specific','side':'left','intensity':1.,'onset_ms':50.,'duration_ms':300.}]
    fatty = mapping['channels']['fatty_acid_taste']
    assert fatty['proxy_tastant'] == 'hexanoic acid'
    assert fatty['not_distinguished_from']
    assert 'https://www.nature.com/articles/s41467-018-06453-1' in fatty['sources']


def test_taste_events_do_not_copy_the_other_side_or_merge_datasets(mapping):
    channels = mapping['channels']
    for cid in ['water_specific', 'sugar_specific', 'high_salt', 'fatty_acid_taste']:
        for side in ['left', 'right']:
            encoded = encode_inputs(mapping, [{'channel': cid, 'side': side}], seed=48131)
            assert set(encoded) == {MC}
            tape = encoded[MC]
            assert set(map(str, tape['input_neuron_ids'])) == set(channels[cid]['side_ids'][side])
            assert set(map(str, tape['event_neuron_ids'])) <= set(channels[cid]['side_ids'][side])
            assert len(tape['event_steps']) > 0
            assert np.all(tape['event_steps'] >= 500) and np.all(tape['event_steps'] < 3500)
    assert mapping['coverage'][MC]['unaccounted_ids'] == []
