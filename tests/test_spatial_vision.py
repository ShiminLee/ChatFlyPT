"""Spatial discrimination of measured samples, not anatomical calibration."""
import json

import numpy as np
import pandas as pd
import pytest

from chatflypt.paths import ROOT, dump
from chatflypt.social import MALE
from chatflypt.spatial_vision import (
    DATASET, PROVENANCE, encode_spatial_vision, hex_cartesian,
    load_spatial_map, project_spatial_vision,
)


@pytest.fixture(scope='module')
def catalogue():
    # Exact audited same-connectome population; no full graph is needed to
    # exercise deterministic geometric projection of real-shaped eye samples.
    saved = ROOT / 'reports/malecns-senses-20260924/populations.json'
    return {'channels': json.loads(saved.read_text(encoding='utf-8'))['channels']}


def vision():
    return {'status': 'measured', 'ommatidia': np.zeros((2, 721, 2)),
            'mean_by_eye': {'left': 0., 'right': 0.}}


def test_column_map_has_verified_same_eye_downstream_evidence(catalogue):
    arrays, report = load_spatial_map()
    rows = pd.read_feather(MALE).set_index('bodyId')
    assert len(arrays['body_ids']) == report['mapped_count'] == 2322
    assert report['rejected_count'] == 1055
    assert [int(np.sum(arrays['eye_indices'] == i)) for i in (0, 1)] == [713, 1609]
    assert report['registration_calibrated'] is False
    assert report['rejected_count'] > 0
    for mapped in report['mapped']:
        source = rows.loc[int(mapped['body_id'])]
        targets = rows.loc[list(map(int, mapped['target_body_ids']))]
        assert source['type'] == 'R1-R6'
        assert source.rootSide == ('L' if mapped['side'] == 'left' else 'R')
        assert targets.somaSide.eq(source.rootSide).all()
        assert targets.type.isin(['L1', 'L2', 'L3']).all()
        assert mapped['column_fraction'] >= .5
        assert any((targets.assignedOlHex1 == mapped['hex'][0]) &
                   (targets.assignedOlHex2 == mapped['hex'][1]))
        assert mapped['angular_error_deg'] < 3.7
        assert len(mapped['author_head_ray']) == 3


def test_author_column_crosswalk_and_out_of_field_rejection_are_explicit():
    arrays, report = load_spatial_map()
    left = report['column_crosswalks']['left']
    assert len(left['accepted']) == 872 and len(left['rejected']) == 5
    assert sum(r['current_hex'] != r['author_hex'] for r in left['accepted']) == 352
    for side, proof in report['column_crosswalks'].items():
        accepted = proof['accepted']
        assert len({tuple(r['current_hex']) for r in accepted}) == len(accepted)
        assert len({tuple(r['author_hex']) for r in accepted}) == len(accepted)
        for r in accepted:
            assert len({w['type'] for w in r['witnesses']}) >= 2
            assert all(w['author_hex'] == r['author_hex'] for w in r['witnesses'])
    reasons = {r['reason'] for r in report['rejected']}
    assert {'no_author_atlas_column', 'no_unambiguous_body_id_column_crosswalk',
            'behind_eye_camera', 'outside_rectilinear_camera', 'outside_compound_eye_mask'} <= reasons
    assert 'hex_center_xy' not in report['transforms']['L']
    assert len(json.dumps(report['transforms'])) < 4000


def test_source_change_invalidates_an_already_cached_map(tmp_path, monkeypatch):
    import chatflypt.spatial_vision as spatial
    from chatflypt.paths import sha256
    source = tmp_path / 'vision.yaml'; source.write_text('initial geometry')
    model = tmp_path / 'map.npz'; np.savez(model, body_ids=np.array([1]))
    metadata = tmp_path / 'map.json'
    dump(metadata, {'map_sha256': sha256(model),
        'registration_source_sha256': {'vision.yaml': sha256(source)}})
    monkeypatch.setattr(spatial, 'ROOT', tmp_path)
    monkeypatch.setattr(spatial, 'MAP', model)
    monkeypatch.setattr(spatial, 'PROVENANCE', metadata)
    monkeypatch.setattr(spatial, 'spatial_source_paths', lambda: (source,))
    spatial._load_map.cache_clear()
    spatial.load_spatial_map()
    source.write_text('changed geometry with a different size')
    with pytest.raises(ValueError, match='registration source changed'):
        spatial.load_spatial_map()
    source.unlink()
    with pytest.raises(RuntimeError, match='source missing'):
        spatial.load_spatial_map()
    spatial._load_map.cache_clear()


def test_build_rejects_camera_snapshot_from_changed_geometry(monkeypatch):
    import chatflypt.paths as paths
    from chatflypt.retinal_registration import prepare_registration, EYE_ASSETS
    real_sha = paths.sha256
    monkeypatch.setattr(paths, 'sha256', lambda p: 'changed' if p == EYE_ASSETS / 'rigging.yaml' else real_sha(p))
    with pytest.raises(ValueError, match='Captured eye-camera geometry sources changed'):
        prepare_registration(None, None)


def test_build_rejects_modified_author_data_even_before_rebuilding(monkeypatch):
    import chatflypt.paths as paths
    from chatflypt.retinal_registration import prepare_registration, ATLAS
    real_sha = paths.sha256
    monkeypatch.setattr(paths, 'sha256', lambda p: 'changed' if p == ATLAS / 'pqxyztp_left.xlsx' else real_sha(p))
    with pytest.raises(ValueError, match='Author retinal source checksum mismatch'):
        prepare_registration(None, None)


def test_official_hex_axes_have_documented_vertical_and_horizontal_signs():
    center, q, p = hex_cartesian([[18, 19], [19, 19], [18, 20]])
    np.testing.assert_array_equal(center, [0, 0])
    assert q[0] > 0 and p[0] < 0 and q[1] > 0 and p[1] > 0


def test_equal_eye_means_different_local_samples_reach_different_cells(catalogue):
    arrays, _ = load_spatial_map()
    left_ommatidia = np.unique(arrays['ommatidium_indices'][arrays['eye_indices'] == 0])
    a, b = int(left_ommatidia[0]), int(left_ommatidia[-1])
    first, second = vision(), vision()
    first['ommatidia'][0, a, 0] = 1.
    second['ommatidia'][0, b, 0] = 1.
    p1, p2 = project_spatial_vision(first, catalogue), project_spatial_vision(second, catalogue)
    active1 = {i for i, x in zip(p1['cell_ids'], p1['intensities']) if x > 0}
    active2 = {i for i, x in zip(p2['cell_ids'], p2['intensities']) if x > 0}
    assert active1 and active2 and active1.isdisjoint(active2)
    assert first['ommatidia'].mean() == second['ommatidia'].mean()
    assert not p1['mean_fallback'] and not p1['registration_calibrated']
    t1, t2 = (encode_spatial_vision(p, 3181991, 1000.) for p in (p1, p2))
    assert len(t1['event_steps']) > 0 and len(t2['event_steps']) > 0
    assert set(t1['event_neuron_ids']) <= active1
    assert set(t2['event_neuron_ids']) <= active2
    assert p1['per_eye']['right']['has_visual_drive'] is False
    dump(ROOT / 'reports/malecns-senses-20260924/spatial-acceptance.json', {
        'scope': 'Synthetic local pixel perturbations verify candidate bridge implementation; not anatomical registration or normal-chat environmental validation.',
        'equal_global_mean': float(first['ommatidia'].mean()),
        'first_ommatidium': a, 'second_ommatidium': b,
        'first_active_cells': sorted(active1), 'second_active_cells': sorted(active2),
        'disjoint_input_cells': True, 'first_events': len(t1['event_steps']), 'second_events': len(t2['event_steps']),
        'map_sha256': p1['map_sha256'], 'registration_calibrated': False,
        'first_per_eye': p1['per_eye'], 'second_per_eye': p2['per_eye']})


def test_darkness_does_not_fall_back_to_nonzero_eye_mean(catalogue):
    dark = vision()
    dark['mean_by_eye'] = {'left': 1., 'right': 1.}
    projection = project_spatial_vision(dark, catalogue)
    tape = encode_spatial_vision(projection, 31, 100.)
    assert len(tape['input_neuron_ids']) == len(tape['event_steps']) == 0
    assert all(not p['has_visual_drive'] for p in projection['per_eye'].values())


def test_excluded_receptors_and_bad_samples_never_gain_mean_input(catalogue):
    arrays, report = load_spatial_map()
    sample = vision(); sample['ommatidia'][:] = .25
    projection = project_spatial_vision(sample, catalogue)
    rejected = {int(r['body_id']) for r in report['rejected']}
    assert not rejected & set(projection['cell_ids'])
    assert projection['per_eye']['left']['rejected_cell_ids']
    sample['ommatidia'] = np.zeros((2, 720, 2))
    with pytest.raises(ValueError, match='ommatidial'):
        project_spatial_vision(sample, catalogue)
    sample['ommatidia'] = np.full((2, 721, 2), np.nan)
    with pytest.raises(ValueError, match='finite'):
        project_spatial_vision(sample, catalogue)


def test_spatial_encoder_deterministic_and_no_unmeasured_signal(catalogue):
    sample = vision(); sample['ommatidia'][1, :, 1] = .4
    projection = project_spatial_vision(sample, catalogue)
    first, second = (encode_spatial_vision(projection, 311, 50.) for _ in range(2))
    np.testing.assert_array_equal(first['event_steps'], second['event_steps'])
    np.testing.assert_array_equal(first['event_neuron_ids'], second['event_neuron_ids'])
    assert set(map(str, first['event_neuron_ids'])) <= set(catalogue['channels']['mc_bright']['side_ids']['right'])
    assert first['dataset'] == DATASET
    unmeasured = project_spatial_vision({'status': 'unavailable'}, catalogue)
    assert encode_spatial_vision(unmeasured, 311, 50.) is None


def test_official_retina_roundoff_allowed_but_real_out_of_range_rejected(catalogue):
    sample = vision()
    sample['ommatidia'][0, :, 0] = 1.000000000000004
    sample['ommatidia'][1, :, 0] = -4e-15
    projected = project_spatial_vision(sample, catalogue)
    intensities = np.asarray(projected['intensities'])
    eyes = np.asarray(projected['eye_indices'])
    np.testing.assert_array_equal(intensities[eyes == 0], 1.)
    np.testing.assert_array_equal(intensities[eyes == 1], 0.)
    assert sample['ommatidia'][0, 0, 0] == 1.000000000000004
    for bad in (1. + 1e-8, -1e-8, np.inf):
        sample['ommatidia'][0, 0, 0] = bad
        with pytest.raises(ValueError, match='finite official'):
            project_spatial_vision(sample, catalogue)
