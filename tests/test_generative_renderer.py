"""Compositional renderer and the two extra blind slots."""
import numpy as np
import pytest

from chatflypt.generative_renderer import (ACTIVITY_PHRASES, PATTERN_PHRASES, audit_grounding,
                                           render_state)
from chatflypt.slot_readout import (ACTIVITY_LABELS, PATTERN_LABELS, activity_labels,
                                    load_slot_decoders, pattern_labels, slot_rules)


def feeling(channel, side, text):
    return {'channel': channel, 'side': side, 'text': text}


def test_renderer_is_deterministic_in_seed_and_slots():
    args = dict(feelings=[feeling('bright', 'left', '我在左侧察觉到了光。')],
                activity='activity:quiet', pattern='pattern:sustained', seed=7)
    assert render_state(**args)['text'] == render_state(**args)['text']
    other = render_state(**dict(args, seed=8))['text']
    assert other is not None


def test_renderer_only_speaks_slots_it_was_given():
    rendered = render_state(feelings=[feeling('bright', 'left', '我在左侧察觉到了光。')],
                            activity='activity:busy', seed=3)
    text = rendered['text']
    assert '活跃' in text and '光' in text
    # A slot family that was not passed can never appear in the sentence.
    for phrase in PATTERN_PHRASES['pattern:sustained'] + PATTERN_PHRASES['pattern:transient']:
        assert phrase not in text
    assert set(rendered['slots']) == {'activity:busy', 'bright:left'}


def test_renderer_abstains_without_slots():
    rendered = render_state(seed=1)
    assert rendered['text'] is None and rendered['slots'] == []
    assert '不生成陈述' in rendered['reason']


def test_phrasing_varies_without_changing_the_claim():
    texts = {render_state(feelings=[feeling('bright', 'left', '我在左侧察觉到了光。')],
                          activity='activity:quiet', seed=seed)['text'] for seed in range(40)}
    assert len(texts) > 1, 'the renderer must not be a single fixed template'
    for text in texts:
        assert any(phrase in text for phrase in ACTIVITY_PHRASES['activity:quiet'])
        assert '我在左侧察觉到了光' in text


def test_audit_counts_invented_and_contradicted_slots_separately():
    measured = {'delivered_items': {'mc_bright:left': 30, 'mc_bright:right': 30},
                'activity_band': 'activity:busy', 'pattern_band': 'pattern:sustained'}
    good = render_state(feelings=[feeling('bright', 'left', '我在左侧察觉到了光。')],
                        activity='activity:busy', pattern='pattern:sustained', seed=2)
    audit = audit_grounding(good, measured)
    assert audit == {'invented': [], 'unsupported': [], 'contradicted': [], 'audited_slots': 3}
    misread = render_state(feelings=[feeling('bright', 'left', '我在左侧察觉到了光。')],
                           activity='activity:quiet', pattern='pattern:sustained', seed=2)
    audit = audit_grounding(misread, measured)
    assert audit['contradicted'] == [{'slot': 'activity:quiet', 'measured': 'activity:busy'}]
    unsupported = render_state(feelings=[feeling('cold', 'left', '我在左侧感到寒冷。')], seed=2)
    assert audit_grounding(unsupported, measured)['unsupported'] == ['cold:left']


def test_slot_bands_come_from_the_training_split():
    windows = [{'delivery_fraction': .05, 'nonsensory_spikes': 10.},
               {'delivery_fraction': .9, 'nonsensory_spikes': 100.},
               {'delivery_fraction': .9, 'nonsensory_spikes': 1000.}]
    rules = slot_rules(windows)
    assert rules['pattern_transient_max_fraction'] == .3
    assert pattern_labels(.05, rules) == ['pattern:transient']
    assert pattern_labels(.9, rules) == ['pattern:sustained']
    low, high = rules['activity_spike_quantiles']
    assert activity_labels(1., rules) == ['activity:quiet']
    assert activity_labels(high + 1., rules) == ['activity:busy']
    assert activity_labels((low + high) / 2, rules) == ['activity:moderate']
    assert set(ACTIVITY_LABELS) | set(PATTERN_LABELS)


def test_slot_loading_abstains_without_an_artifact(tmp_path):
    class FakeBrain:
        ids = np.array([1, 2])
        dataset = 'MaleCNS v1.0'
    decoder, note = load_slot_decoders(FakeBrain(), {'channels': {}}, tmp_path / 'missing.npz')
    assert decoder is None and note['applied'] is False
    assert 'not been trained' in note['reason']
