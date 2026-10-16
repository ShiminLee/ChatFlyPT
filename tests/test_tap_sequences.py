"""Conspecific contact trains: probe timing, validation and language routing."""
import pytest

from chatflypt.environment_message import ground_message, parse_message
from chatflypt.input_language import parse_input
from chatflypt.physical_touch import (MIN_TAP_MS, presentation_span, ramp_ms, tap_schedule,
                                      validate_presentations)


def probe(**extra):
    return dict({'kind': 'touch', 'side': 'right', 'body_part': 'foreleg',
                 'strength': 1., 'onset_ms': 50., 'duration_ms': 300.}, **extra)


def test_single_tap_stays_backwards_compatible():
    rows = validate_presentations([probe()])
    assert rows[0]['taps'] == 1
    assert rows[0]['duration_ms'] == 300.
    assert tap_schedule(rows[0]) == [50.]


def test_tap_train_schedule_is_onset_to_onset():
    rows = validate_presentations([probe(taps=3, duration_ms=60., interval_ms=140.)])
    assert tap_schedule(rows[0]) == [50., 190., 330.]
    assert presentation_span(rows[0]) == pytest.approx(330. + 60. + ramp_ms(rows[0]))


def test_short_tap_floor_and_spacing_rules():
    with pytest.raises(ValueError, match='范围'):
        validate_presentations([probe(duration_ms=MIN_TAP_MS - 1, taps=2)])
    with pytest.raises(ValueError, match='间隔'):
        validate_presentations([probe(taps=2, duration_ms=60., interval_ms=70.)])
    with pytest.raises(ValueError, match='重叠'):
        validate_presentations([probe(taps=3, duration_ms=60., interval_ms=100.),
                                probe(taps=1, onset_ms=120., duration_ms=60.)])
    assert validate_presentations([probe(taps=8, duration_ms=30., interval_ms=60.)])[0]['taps'] == 8
    with pytest.raises(ValueError, match='范围'):
        validate_presentations([probe(taps=9)])


def test_tap_train_must_fit_the_bounded_window():
    # 8 taps x 400 ms with 700 ms onset spacing spans > 5 s.
    with pytest.raises(ValueError, match='5 秒'):
        validate_presentations([probe(taps=8, duration_ms=400., interval_ms=700.)])


def fake_catalogue():
    """Minimal catalogue carrying the vocabulary the parser needs."""
    return {'channels': {'foreleg_touch': {
        'status': 'candidate', 'label': '前腿触碰候选群', 'aliases': ['foreleg_touch'],
        'quality_aliases': ['触碰', '接触', '碰', 'touch'],
        'side_ids': {'left': ['a'], 'right': ['b'], 'bilateral': ['a', 'b']},
        'body_parts': ['foreleg'], 'body_part_ids': {}}}}


def test_language_parses_tap_counts_and_keeps_timing_engineering():
    catalogue = fake_catalogue()
    result = parse_input('碰三下右前腿', catalogue)
    assert result['items'][0]['taps'] == 3
    assert result['items'][0]['duration_ms'] == 60.
    result = parse_input('连碰三次左前腿', catalogue)
    assert result['items'][0]['taps'] == 3 and result['items'][0]['side'] == 'left'


def test_ground_message_expands_taps_into_presentations():
    catalogue = fake_catalogue()
    raw = parse_message('连碰三次右前腿', catalogue)
    assert raw['items'][0]['taps'] == 3
    from chatflypt.environment import empty_environment, probe_environment
    environment = empty_environment()
    probe_frame = probe_environment(environment, 42, 'ground')
    plan, candidate = ground_message(raw, environment, probe_frame, catalogue)
    assert plan['kind'] == 'environment_presentation'
    assert plan['items'] == []
    presentation = plan['presentations'][0]
    assert presentation['taps'] == 3 and presentation['duration_ms'] == 60.
    assert candidate['objects'] == []
