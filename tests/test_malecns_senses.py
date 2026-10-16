"""Verify same-brain population identity without asserting biological efficacy."""
import copy

import numpy as np
import pandas as pd
import pytest

from chatflypt.malecns_senses import (
    CHANNEL_MAP, DATASET, POPULATIONS, graph_membership,
    remap_environment_inputs, remap_environment_sample, target_channel,
)
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.sensory_events import encode_inputs
from chatflypt.sensory_readout import canonical_specs
from chatflypt.social import MALE


@pytest.fixture(scope='module')
def catalogue():
    return build_catalogue(save=False)


def selected_rows(rows, cls, types):
    """Both declared selector forms: exact class+types, or class+subclass."""
    if isinstance(cls, dict):
        return rows[rows['class'].eq(cls['class']) & rows['subclass'].eq(cls['subclass'])]
    return rows[rows['class'].eq(cls) & rows.type.isin(types)]


def test_exact_official_types_sides_and_outgoing_edges(catalogue):
    rows = pd.read_feather(MALE).fillna('')
    rows = rows[rows.superclass.str.contains('sensory')]
    outgoing, _ = graph_membership()
    for source, cls, types, *_ in POPULATIONS:
        spec = catalogue['channels'][CHANNEL_MAP[source]]
        selected = selected_rows(rows, cls, types)
        assert set(spec['neuron_ids']) == set(selected.bodyId.astype(str))
        assert spec['dataset'] == DATASET
        assert spec['id_namespace'] == 'malecns_body_id'
        for side, root in [('left', 'L'), ('right', 'R')]:
            expected = {str(i) for i in selected[selected.rootSide.eq(root)].bodyId
                        if outgoing.get(str(i), 0) > 0}
            assert set(spec['side_ids'][side]) == expected
        assert set(spec['side_ids']['bilateral']) == set(spec['side_ids']['left'] + spec['side_ids']['right'])
        assert not (set(spec['unassigned_side_ids']) & set(spec['side_ids']['bilateral']))
        assert all(outgoing[i] > 0 for i in spec['side_ids']['bilateral'])


def test_new_populations_do_not_enter_old_label_protocol(catalogue):
    old = {s['id'] for s in canonical_specs(catalogue, DATASET)}
    assert not (set(CHANNEL_MAP.values()) & old)
    sources = {}
    for source, cid in CHANNEL_MAP.items():
        sources.setdefault(cid, set()).add(source)
    for cid, aliases in sources.items():
        spec = catalogue['channels'][cid]
        assert spec['advanced'] and spec['status'] == 'candidate'
        assert spec['readout_group'] == cid
        assert spec['readout_status'] == 'untrained_new_population'
        # One MaleCNS channel may be reachable from several source names, but its
        # recorded source must be one of them (never an unrelated registry name).
        assert spec['source_channel'] in aliases
        assert not spec['functional_mapping']['physical_transduction_calibrated']


def test_measured_inputs_resolve_into_one_id_namespace(catalogue):
    items = [{'channel': c, 'side': 'left', 'intensity': 1., 'onset_ms': 0., 'duration_ms': 100.}
             for c in ('bright', 'yeast', 'warm', 'cold', 'foreleg_touch', 'sugar_specific')]
    before = copy.deepcopy(items)
    mapped, exclusions = remap_environment_inputs(catalogue, items)
    assert not exclusions and items == before
    assert [i['channel'] for i in mapped[:4]] == ['mc_bright', 'mc_yeast', 'mc_warm', 'mc_cold']
    tapes = encode_inputs(catalogue, mapped, seed=991811, duration_ms=200.)
    assert set(tapes) == {DATASET}
    tape = tapes[DATASET]
    assert len(tape['event_steps']) > 0
    outgoing, _ = graph_membership()
    assert set(map(str, tape['event_neuron_ids'])) <= set(outgoing)
    assert np.all(tape['event_steps'] < 1000)


def test_missing_modality_never_falls_back_to_other_brain(catalogue):
    # FlyWire-only modalities that still have no MaleCNS counterpart must fail
    # closed instead of being resolved onto an unrelated population.
    for channel in ('sweet_water', 'vision_r7'):
        items = [{'channel': channel, 'side': 'right', 'intensity': 1.}]
        mapped, exclusions = remap_environment_inputs(catalogue, items)
        assert mapped == [] and len(exclusions) == 1
        assert exclusions[0]['channel'] == channel
        with pytest.raises(ValueError, match='counterpart'):
            target_channel(catalogue, channel)
    # The conspecific song modality, by contrast, now has a same-individual entry.
    assert target_channel(catalogue, 'auditory') == 'mc_auditory'
    assert target_channel(catalogue, 'courtship_pulse') == 'mc_auditory'


def test_no_invented_side_for_unknowns(catalogue):
    yeast = catalogue['channels']['mc_yeast']
    assert len(yeast['neuron_ids']) == 54
    assert len(yeast['unassigned_side_ids']) == 1
    assert len(yeast['side_ids']['bilateral']) == 53
    bright = catalogue['channels']['mc_bright']
    assert len(bright['graph_audit']['without_modeled_outgoing_ids']) == 18


def test_sample_provenance_matches_new_population(catalogue):
    sample = {'items': [{'channel': 'bright', 'side': 'right', 'intensity': .5}],
              'evidence': [{'channel': 'bright', 'side': 'right', 'neuron_ids': ['old']},
                           {'mode': 'official_retina', 'mean_by_eye': {'right': .5}}],
              'unmapped': [], 'environment_revision': 8}
    before = copy.deepcopy(sample)
    result = remap_environment_sample(catalogue, sample)
    assert sample == before
    assert result['items'][0]['channel'] == 'mc_bright'
    assert result['evidence'][0]['neuron_ids'] == catalogue['channels']['mc_bright']['side_ids']['right']
    assert result['evidence'][0]['source_channel'] == 'bright'
    assert result['evidence'][1] == sample['evidence'][1]
    assert result['neural_dataset'] == DATASET
    assert result['sensory_projections'][0]['readout_status'] == 'untrained_new_population'
