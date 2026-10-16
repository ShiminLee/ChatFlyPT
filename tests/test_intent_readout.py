"""Motor-side (intent) readout: label rules, reporting and fail-closed loading."""
import numpy as np

from chatflypt.intent_readout import (ACTIVITY_QUANTILE, LABELS, TURN_QUANTILE, intent_report,
                                      labels_to_vector, load_intent_decoder, measured_outcome,
                                      outcome_labels, outcome_rules)


def outcome(yaw, path):
    return {'yaw_change_deg': yaw, 'path_length_mm': path}


def test_rules_come_from_the_training_split_quantiles_only():
    training = [outcome(yaw, path) for yaw, path in
                [(0., 0.), (0., 1.), (0., 2.), (0., 3.), (0., 4.),
                 (10., .5), (-10., .5), (2., .5), (-2., .5), (1., .5)]]
    rules = outcome_rules(training)
    assert rules['turn_quantile'] == TURN_QUANTILE and rules['activity_quantile'] == ACTIVITY_QUANTILE
    assert rules['turn_deg'] == float(np.quantile([0., 0., 0., 0., 0., 10., 10., 2., 2., 1.], TURN_QUANTILE))
    assert rules['move_mm'] == float(np.quantile([0., 1., 2., 3., 4., .5, .5, .5, .5, .5], ACTIVITY_QUANTILE))


def test_labels_mark_turn_and_activity_independently():
    rules = {'turn_deg': 5., 'move_mm': 1.}
    assert outcome_labels(outcome(6., 2.), rules) == ['turn:left', 'move:active']
    assert outcome_labels(outcome(-6., .5), rules) == ['turn:right', 'move:still']
    assert outcome_labels(outcome(.1, .5), rules) == ['move:still']
    assert labels_to_vector(['turn:left', 'move:active']).tolist() == [True, False, True, False]
    assert labels_to_vector(['move:still']).dtype == bool


def test_measured_outcome_reads_the_recorded_trajectory():
    frames = []
    for index, (x, yaw) in enumerate([(0., 0.), (1., 0.), (2., np.pi / 4)]):
        frames.append({'body_positions_mm': [[x, 0., 0.]],
                       'body_quaternions_wxyz': [[np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]]})
    result = measured_outcome(frames)
    assert result['path_length_mm'] == 2.
    assert 40. < result['yaw_change_deg'] < 50.


def test_report_never_speaks_unless_decoded_and_qualified():
    empty = intent_report({})
    assert empty['status'] == 'unavailable' and empty['text'] is None
    uncalibrated = intent_report({'status': 'uncalibrated', 'words': ['turn:left']})
    assert uncalibrated['text'] is None
    decoded = intent_report({'status': 'decoded', 'words': ['move:active']})
    assert decoded['text'].startswith('候选运动意图') and '走动' in decoded['text']
    assert decoded['measured_outcome'] is None
    ambiguous = intent_report({'status': 'decoded', 'words': ['turn:left', 'turn:right']})
    assert ambiguous['status'] == 'ambiguous' and ambiguous['text'] is None


def test_measured_outcome_is_evidence_not_a_substitute_for_a_prediction():
    report = intent_report({'status': 'decoded', 'words': []}, outcome(9., 3.))
    assert report['text'] is None and report['measured_outcome'] == outcome(9., 3.)


def test_loading_abstains_when_no_artifact_is_published(tmp_path):
    class FakeBrain:
        ids = np.array([1, 2])
        dataset = 'MaleCNS v1.0'
    decoder, note = load_intent_decoder(FakeBrain(), {'channels': {}}, tmp_path / 'missing.npz')
    assert decoder is None
    assert note['applied'] is False and 'not been trained' in note['reason']
    assert note['globally_calibrated'] is False and note['biologically_calibrated'] is False
