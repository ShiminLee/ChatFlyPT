"""Steering-command readout: label rules, reporting, direction gate, fail-closed loading."""
import numpy as np
import pytest

from chatflypt.command_readout import (LABELS, command_labels, command_report, command_rules,
                                       command_target, load_command_decoder, turn_sign)


def body_frame(left, right, index):
    return {'commands': {'left': left, 'right': right}, 'source_command_bin': index}


def test_target_uses_the_sampled_frames_and_skips_the_pre_window_frame():
    # The first frame still holds the previous completed bin, so it is excluded.
    body = {'trajectory': [body_frame(9., 9., -1), body_frame(0., .2, 0), body_frame(.1, .1, 1)]}
    entry = command_target(body)
    assert entry['steer_command'] == pytest.approx((0.2 - 0.0 + 0.1 - 0.1) / 2)
    assert entry['walk_drive'] == pytest.approx(((0. + .2) / 2 + (.1 + .1) / 2) / 2)
    assert entry['sampled_frames'] == 2
    assert entry['command_bins'] == [0, 1]


def test_rules_come_from_the_training_split_only_and_labels_respect_them():
    commands = [{'steer_command': value} for value in (0., .02, -.03, 0., .05, -.04)]
    split = np.array(['train', 'train', 'train', 'validation', 'validation', 'validation'])
    rules = command_rules(commands, split)
    # Split names must be compared, not treated as truthy: only the three
    # training windows may set the threshold.
    assert rules['informative_abs_command'] == pytest.approx(float(np.quantile([0., .02, .03], .85)))
    assert command_rules(commands, split == 'train')['informative_abs_command'] == \
        rules['informative_abs_command']
    quiet = {'steer_command': 0.}
    assert command_labels(quiet, rules) == []
    assert command_labels({'steer_command': .5}, rules) == ['command:turn-left']
    assert command_labels({'steer_command': -.5}, rules) == ['command:turn-right']


def test_report_keeps_the_qualifier_and_never_speaks_without_a_decoded_label():
    empty = command_report({})
    assert empty['status'] == 'unavailable' and empty['text'] is None
    uncalibrated = command_report({'status': 'uncalibrated', 'words': ['command:turn-right']})
    assert uncalibrated['text'] is None
    spoken = command_report({'status': 'decoded', 'words': ['command:turn-right'],
                             'command_readout': {'served_directions': {'command:turn-right': True}}})
    assert '向右' in spoken['text'] and '未经身体行为验证' in spoken['text']
    assert spoken['labels'] == ['command:turn-right']
    assert spoken['command_readout']['served_directions'] == {'command:turn-right': True}
    with_outcome = command_report({'status': 'decoded', 'words': ['command:turn-left']},
                                  {'yaw_change_deg': -1.5, 'path_length_mm': .2})
    assert '向左' in with_outcome['text'] and '不作为行为验证' in with_outcome['text']


def test_loading_abstains_when_no_artifact_is_published(tmp_path):
    class FakeBrain:
        ids = np.array([1, 2])
        dataset = 'MaleCNS v1.0'
    decoder, note = load_command_decoder(FakeBrain(), {'channels': {}}, tmp_path / 'missing.npz')
    assert decoder is None
    assert note['applied'] is False and 'not been trained' in note['reason']


def test_direction_words_are_gated_by_the_measured_calibration():
    sign, evidence = turn_sign()
    if sign is None:
        assert isinstance(evidence, str)
        return
    assert sign == 1
    assert evidence['convention'].startswith('steer = mean(right - left)')
    # Left turns come from attenuating the left strides, so they must have been
    # measured as positive yaw; the probe records that, not an assumption.
    assert evidence['conditions']['forward_turn_inside_L']['mean_yaw_deg'] > 0
    assert evidence['conditions']['forward_turn_inside_R']['mean_yaw_deg'] < 0
