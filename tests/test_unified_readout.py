import json
from types import SimpleNamespace

import numpy as np
import pytest

from chatflypt.unified_readout import (SCHEMA, DATASET, UnifiedDecoder, blind_mask,
    internal_features, load_unified_decoder, model_contract)
from scripts.train_unified_readout import (choose_thresholds, measured_targets,
    protocol, validate_splits, write_observation)


def brain():
    return SimpleNamespace(dataset=DATASET, ids=np.arange(6, dtype=np.int64),
        lookup={str(i): i for i in range(6)}, nonsensory=np.array([False, True, True, True, True, True]),
        indptr=np.zeros(7, np.int32), indices=np.array([], np.int32), weights=np.array([], float))


def catalogue():
    return {'channels': {
        'mc_bright': {'id': 'mc_bright', 'dataset': DATASET, 'label': '光',
            'neuron_ids': ['0'], 'side_ids': {'left': ['0'], 'right': [], 'bilateral': ['0']},
            'rate_hz': 100., 'profile': 'constant', 'advanced': True},
        'proprioception': {'id': 'proprioception', 'dataset': DATASET, 'label': '本体感觉',
            'neuron_ids': ['1'], 'side_ids': {'left': ['1'], 'right': [], 'bilateral': ['1']},
            'body_part_ids': {'foreleg': {'right': ['2']}}, 'advanced': True}}}


def decoder(tmp_path, changes=None):
    b, c = brain(), catalogue()
    metadata = {'schema': SCHEMA, 'contract': model_contract(b, c),
        'input_mode': 'environment_only', 'split_unit': 'whole_session',
        'protocol_sha256': 'test-protocol', 'tested_durations_ms': [500.]}
    saved = dict(feature_indices=np.array([3, 10, 17]), indices=np.array([3, 4, 5]),
        weights=np.array([[1., 0., 0.], [0., 1., 0.]]), intercept=np.array([1., 1.]),
        thresholds=np.array([1.5, 1.5]), qualified=np.array([True, False]),
        labels=['bright:left', 'yeast:left'], names=['左侧光', '左侧酵母气味'],
        metadata=json.dumps(metadata))
    saved.update(changes or {})
    path = tmp_path / 'model.npz'; np.savez(path, **saved)
    return UnifiedDecoder(b, c, path), b, c, path


def test_blind_mask_includes_advanced_proprioception_and_body_part_inputs():
    np.testing.assert_array_equal(blind_mask(brain(), catalogue()), [False, False, False, True, True, True])
    c = catalogue(); c['channels']['proprioception']['neuron_ids'].append('999')
    with pytest.raises(ValueError, match='absent'):
        blind_mask(brain(), c)


def test_direct_input_cannot_change_prediction_and_unqualified_label_is_local(tmp_path):
    reader, _, _, _ = decoder(tmp_path)
    counts = np.array([0, 0, 0, 4, 0, 0])
    state = np.zeros((4, 6)); state[0, 4] = 10.; state[2, 4] = 2.
    a = reader.decode(counts, state)
    counts[:3] = 999999; state[:, :3] = 999999
    b = reader.decode(counts, state)
    assert a == b
    assert a['words'] == ['bright:left']
    assert a['unqualified_words'] == ['yeast:left']
    assert a['feedback_adaptation']['applied']
    assert not a['feedback_adaptation']['biologically_calibrated']


def test_intercepts_cannot_make_silence_speak(tmp_path):
    reader, _, _, _ = decoder(tmp_path, {'intercept': np.array([100., 100.])})
    decoded = reader.decode(np.zeros(6), np.zeros((4, 6)))
    assert decoded['words'] == decoded['unqualified_words'] == []
    counts = np.array([123, 123, 123, 0, 0, 0])
    state = np.zeros((4, 6)); state[:, :3] = 999.
    assert reader.decode(counts, state)['words'] == []


def test_unknown_duration_abstains_and_trial_interface_has_no_scene_parameter(tmp_path):
    reader, _, _, _ = decoder(tmp_path)
    trial = SimpleNamespace(counts=np.ones(6), state_record=np.ones((4, 6)), duration_ms=300.)
    assert reader.decode_trial(trial)['status'] == 'uncalibrated'
    with pytest.raises(TypeError):
        reader.decode(trial.counts, trial.state_record, user_message='左转')


@pytest.mark.parametrize('change', [
    {'feature_indices': np.array([0, 10, 17]), 'indices': np.array([0, 4, 5])},
    {'feature_indices': np.array([3, 3, 17]), 'indices': np.array([3, 5])},
    {'weights': np.array([[np.nan, 0., 0.], [0., 1., 0.]])},
    {'qualified': np.array([1, 0])}, {'indices': np.array([3, 4])},
    {'labels': ['bright:left', 'bright:left']},
])
def test_corrupt_or_unblinded_artifact_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        decoder(tmp_path, change)


def test_changed_input_mapping_and_new_input_mask_reject_old_weights(tmp_path):
    _, b, c, path = decoder(tmp_path)
    c['channels']['mc_bright']['rate_hz'] = 150.
    with pytest.raises(ValueError, match='stale'):
        UnifiedDecoder(b, c, path)
    c = catalogue(); c['channels']['proprioception']['neuron_ids'].append('3')
    selected, reason = load_unified_decoder(b, c, path)
    assert selected is None and 'stale' in reason['reason']
    assert not reason['biologically_calibrated']


def test_neuron_order_and_graph_are_bound_to_artifact(tmp_path):
    _, _, c, path = decoder(tmp_path)
    b = brain(); b.ids = b.ids[::-1].copy()
    with pytest.raises(ValueError, match='stale'):
        UnifiedDecoder(b, c, path)
    b = brain(); b.weights = np.array([.3]); b.indices = np.array([3])
    with pytest.raises(ValueError, match='stale'):
        UnifiedDecoder(b, c, path)


def test_missing_model_explicitly_abstains(tmp_path):
    selected, reason = load_unified_decoder(brain(), catalogue(), tmp_path / 'missing.npz')
    assert selected is None and 'not been trained' in reason['reason']


@pytest.mark.parametrize('counts,state,duration', [
    (np.array([-1]), np.zeros((4, 1)), 500.),
    (np.zeros(1), np.full((4, 1), np.nan), 500.),
    (np.zeros(1), np.zeros((3, 1)), 500.),
    (np.zeros(1), np.zeros((4, 1)), 0.),
])
def test_invalid_neural_records_rejected(counts, state, duration):
    with pytest.raises(ValueError): internal_features(counts, state, duration)


def test_offline_targets_require_correct_actual_population_and_sample_time():
    c = catalogue()
    sample = {'time_ms': 0., 'until_ms': 100., 'items': [{'channel': 'mc_bright', 'side': 'left'}]}
    events = {'dataset': DATASET, 'dt_ms': .1, 'event_steps': np.array([1500]), 'event_neuron_ids': np.array([0])}
    assert measured_targets([sample], events, c, ['mc_bright']) == set()
    events['event_steps'] = np.array([500])
    assert measured_targets([sample], events, c, ['mc_bright']) == {'bright:left'}
    events['event_neuron_ids'] = np.array([1])
    assert measured_targets([sample], events, c, ['mc_bright']) == set()


def test_actual_session_identity_cannot_be_split_under_different_names(tmp_path):
    rows = [{'episode': split, 'session_id': 'one-session' if split != 'test' else 'another',
             'split': split, 'turn': 0, 'record': str(tmp_path / (split + '.npz'))}
            for split in ('train', 'validation', 'test')]
    with pytest.raises(ValueError, match='session'):
        validate_splits(rows)
    rows[1]['session_id'] = 'validation-session'
    assert len(validate_splits(rows)) == 3
    rows[1]['record'] = rows[0]['record']
    with pytest.raises(ValueError, match='reused'):
        validate_splits(rows)


def test_selection_qualifies_individual_labels_using_negatives():
    truth = np.array([[1, 1], [1, 1], [0, 0], [0, 0]], bool)
    scores = np.array([[.9, .9], [.8, .9], [.1, .9], [.1, .9]])
    thresholds, qualified = choose_thresholds(scores, truth, np.ones(4, bool), truth,
        {'minimum_positive_windows': 2, 'minimum_negative_windows': 2, 'precision': .9, 'recall': .75})
    assert qualified.tolist() == [True, False]
    assert .2 <= thresholds[0] <= .8


def test_protocol_uses_physical_presentations_and_keeps_whole_sessions_separate():
    plan = protocol()
    assert plan['input_mode'] == 'environment_only'
    assert len(plan['episodes']) == 36
    assert len({e['id'] for e in plan['episodes']}) == 36
    seeds = [e['seed'] + i for e in plan['episodes'] for i in range(len(e['turns']))]
    assert len(set(seeds)) == len(seeds)
    for episode in plan['episodes']:
        for turn in episode['turns']:
            for presentation in turn:
                assert presentation['kind'] == 'touch' and 'channel' not in presentation
        if episode['condition'].startswith('touch_'):
            assert episode['turns'][1] == []


def test_direct_neural_historical_records_cannot_be_relabelled_environmental(tmp_path):
    result = {'feedback': {'input_mode': 'neural_intervention', 'direct_neural_intervention': True},
              'body': {}, 'trial': None}
    with pytest.raises(ValueError, match='environment-only'):
        write_observation(result, tmp_path, {'id': 'trial', 'split': 'train'}, 0, {}, 'sha')


def test_episode_lock_rejects_overlapping_collectors(tmp_path, monkeypatch):
    from scripts import train_unified_readout as training
    monkeypatch.setattr(training, 'OUT', tmp_path)
    with training.episode_lock('same-trial', 'fixed-protocol'):
        with pytest.raises(FileExistsError):
            with training.episode_lock('same-trial', 'fixed-protocol'):
                pass
    assert not (tmp_path / 'locks/same-trial.lock').exists()


def test_live_camera_geometry_dependency_change_invalidates_decoder(tmp_path, monkeypatch):
    from chatflypt import spatial_vision
    source = tmp_path / 'current-camera-source.yaml'
    source.write_text('eye_axes: initial', encoding='utf8')
    monkeypatch.setattr(spatial_vision, 'spatial_source_paths', lambda: (source,), raising=False)
    _, b, c, path = decoder(tmp_path)
    source.write_text('eye_axes: different physical camera', encoding='utf8')
    selected, reason = load_unified_decoder(b, c, path)
    assert selected is None and 'stale' in reason['reason']


def test_v3_adds_physical_controls_without_changing_original_scenes_or_seeds(tmp_path, monkeypatch):
    from scripts import train_unified_readout as training
    placement = tmp_path / 'placement.json'
    placement.write_text(json.dumps({'neutral_receptors_mm': {},
        'neutral_eye_cameras_mm': {'left': [0., .4, 1.], 'right': [0., -.4, 1.]},
        'verified_no_initial_yeast_contact': True,
        'local_sources': {part: {side: {'position_mm': [1., sign * .2, 1.], 'range_mm': .16}
            for side, sign in (('left', 1), ('right', -1))} for part in ('funiculus', 'arista')}}), encoding='utf8')
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    monkeypatch.setattr(training, 'V3_PLACEMENT', placement)
    baseline = protocol(); augmented = protocol('retinal-axes-v3')
    assert augmented['episodes'][:36] == baseline['episodes']
    assert len(augmented['episodes']) == 72
    assert sum(len(e['turns']) for e in augmented['episodes']) == 156
    seeds = [e['seed'] + i for e in augmented['episodes'] for i, _ in enumerate(e['turns'])]
    assert len(set(seeds)) == 156
    dark_sugar = next(e for e in augmented['episodes'] if e['id'] == 'train-dark_leg_sugar_control')
    dark_block = next(e for e in augmented['episodes'] if e['id'] == 'train-dark_leg_block_control')
    assert dark_sugar['environment']['objects'][0] == dark_block['environment']['objects'][0]
    assert dark_sugar['environment']['objects'][1]['size_mm'] == dark_block['environment']['objects'][1]['size_mm']
    left_light = next(e for e in augmented['episodes'] if e['id'] == 'train-unilateral_bright_left')
    assert left_light['environment']['objects'][0]['position_mm'][1] < 0
