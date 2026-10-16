"""Bounded offline evidence checks; never imports or runs the brain."""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from registration import sha, FROZEN, ATLAS, xlsx_numeric


def test_live_map_unchanged_and_candidate_has_no_edge_fallback():
    report = json.loads((HERE/'registration-summary.json').read_text())
    assert sha(FROZEN) == report['frozen_map_json_sha256']
    assert sha(FROZEN.with_suffix('.npz')) == report['frozen_map_npz_sha256']
    mapping = json.loads((HERE/'candidate-receptor-map.json').read_text())
    assert len(mapping['mapped']) == 2322
    assert len(mapping['rejected']) == 1007
    assert {r['body_id'] for r in mapping['mapped']}.isdisjoint(r['body_id'] for r in mapping['rejected'])
    for side, count in [('left',713),('right',1609)]:
        rows = [r for r in mapping['mapped'] if r['side']==side]
        assert len(rows)==count
        assert all(0<=r['ommatidium_index']<721 for r in rows)
        assert max(r['angular_error_deg'] for r in rows)<3.7
    assert {'behind_eye_camera','outside_rectilinear_camera','outside_compound_eye_mask',
            'no_author_atlas_column','no_unambiguous_body_id_column_crosswalk'} == {r['reason'] for r in mapping['rejected']}


def test_crosswalk_uses_same_ids_distinct_types_and_bijective_columns():
    for side in ['left','right']:
        data = json.loads((HERE/f'column-crosswalk-{side}.json').read_text())
        rows = data['accepted']
        assert len(set(tuple(r['current_hex']) for r in rows))==len(rows)
        assert len(set(tuple(r['author_hex']) for r in rows))==len(rows)
        for row in rows:
            assert len({w['type'] for w in row['witnesses']})>=2
            assert all(w['author_hex']==row['author_hex'] for w in row['witnesses'])
    assert len(json.loads((HERE/'column-crosswalk-left.json').read_text())['rejected'])==5


def test_author_xyz_and_signed_phi_are_not_confused():
    for side in ['left','right']:
        a=xlsx_numeric(ATLAS/f'pqxyztp_{side}.xlsx')
        np.testing.assert_allclose(np.linalg.norm(a[['x','y','z']],axis=1),1.,atol=1e-14)
        np.testing.assert_array_equal(a.p,a.hex2-19)
        np.testing.assert_array_equal(a.q,a.hex1-18)
        # Explicit regression for the negative-atan2 display phi.
        np.testing.assert_allclose((a.phi+np.degrees(np.arctan2(a.y,a.x))+180)%360-180,0,atol=2e-10)


def test_actual_renderer_distinguishes_sides_and_author_axes_receive_light():
    scan=json.loads((HERE/'renderer-scan.json').read_text())
    assert scan['cases']==150
    assert scan['author_axis_cases']==scan['author_axis_received_light']==40
    assert scan['peak_ray_error_deg']['p95']<3.
    indexed={(r['side'],r['azimuth_left_deg'],r['elevation_deg']):r for r in scan['records'] if r['kind']=='head_direction_grid'}
    assert indexed['left',90,0]['renderer_visible']
    assert not indexed['right',90,0]['renderer_visible']
    assert indexed['right',-90,0]['renderer_visible']
    assert not indexed['left',-90,0]['renderer_visible']
    assert all(indexed[side,0,0]['renderer_visible'] for side in ['left','right'])
