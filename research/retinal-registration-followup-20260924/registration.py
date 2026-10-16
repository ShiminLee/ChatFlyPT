"""Offline author atlas / actual FlyGym camera audit. Never writes live maps.

Run from repository root with .venv/Scripts/python.exe PATH/registration.py.
The separate camera_scan.py uses .venv-body for actual renderer checks.
"""
from pathlib import Path
import hashlib
import json
import re
import warnings
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / 'vendor/flygym/src/flygym/assets/model/neuromechfly'
ATLAS = HERE / 'eyemap-archive/maps/eyemap_mcns_f20240701'
ANN = ROOT / 'data/raw/malecns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather'
FROZEN = ROOT / 'reports/unified-pathway-20260924/retinal-bbox-baseline/receptor-map.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def xlsx_numeric(path, sheet=1):
    """Read the first simple author sheet without executing spreadsheet code.

    Preserves full xlsx numbers; the author's CSV exports round vectors to 6 dp.
    All used sources have a single ordinary value table; formulas are rejected.
    """
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rows = []
        for row in ET.fromstring(z.read(f'xl/worksheets/sheet{sheet}.xml')).findall('.//m:row', ns):
            values = {}
            for cell in row.findall('m:c', ns):
                if cell.find('m:f', ns) is not None:
                    raise ValueError('Unexpected formula in author data')
                ix = 0
                for char in re.match('[A-Z]+', cell.attrib['r'])[0]:
                    ix = 26 * ix + ord(char) - 64
                v = cell.find('m:v', ns)
                if v is None:
                    value = ''.join(cell.itertext())
                elif cell.attrib.get('t') == 's':
                    value = strings[int(v.text)]
                else:
                    value = float(v.text)
                values[ix - 1] = value
            rows.append(values)
    return pd.DataFrame([{rows[0][k]: v for k, v in row.items()} for row in rows[1:]])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=-1), -1, 1)))


def quantiles(values):
    a = np.asarray(values, float)
    return dict(zip(('min', 'p50', 'p95', 'max'), np.quantile(a, [0, .5, .95, 1]).tolist())) if a.size else None


def retina_rays(camera, config, id_map):
    """Mean angular ray for each official ommatidial footprint.

    Reproduce exact destination->source pixel mapping and integer truncation
    of pinned Retina._correct_fisheye; use the renderer's pixel centers.
    The returned rays use +x anterior, +y left, +z dorsal head coordinates.
    No eye reflection is added: compiled camera matrices include both eyes.
    """
    h, w = id_map.shape
    row, col = np.indices(id_map.shape)
    dn_r = (2 * row - h) / h / config['fisheye_zoom']
    dn_c = (2 * col - w) / w / config['fisheye_zoom']
    den = 1 - config['fisheye_distortion_coefficient'] * (dn_r**2 + dn_c**2) + 1e-6
    sr = np.trunc((dn_r / den + 1) * h / 2).astype(int)
    sc = np.trunc((dn_c / den + 1) * w / 2).astype(int)
    valid = (sr >= 0) & (sr < h) & (sc >= 0) & (sc < w) & (id_map > 0)
    f = h / (2 * np.tan(np.radians(camera['fovy_deg']) / 2))
    camera_rays = unit(np.stack(((sc + .5 - w / 2) / f, -(sr + .5 - h / 2) / f, -np.ones_like(sr)), axis=-1))
    head_rays = camera_rays @ np.asarray(camera['head_rotation']).T
    means, spreads, fractions = [], [], []
    for label in range(1, int(id_map.max()) + 1):
        mask = valid & (id_map == label)
        mean = unit(head_rays[mask].mean(axis=0))
        means.append(mean)
        spreads.append(float(angle(head_rays[mask], mean).max()))
        fractions.append(float(mask.sum() / np.sum(id_map == label)))
    return np.asarray(means), np.asarray(spreads), np.asarray(fractions), head_rays, valid


def atlas_pixel(ray, camera, config, id_map):
    """Invert the continuous fisheye mapping to check actual retinal support.

    This avoids assigning unseen rear/dorsal directions to the nearest edge.
    A subpixel edge remains discretized and is checked by camera_scan.py.
    """
    r = np.asarray(ray) @ np.asarray(camera['head_rotation'])
    if r[2] >= 0:
        return None, 'behind_eye_camera'
    h, w = id_map.shape
    f = h / (2 * np.tan(np.radians(camera['fovy_deg']) / 2))
    src = np.array([f * r[0] / -r[2] * 2 / w, -f * r[1] / -r[2] * 2 / h])
    if np.any(np.abs(src) >= 1):
        return None, 'outside_rectilinear_camera'
    rho = np.linalg.norm(src)
    k = config['fisheye_distortion_coefficient']
    # +1e-6 is present in the actual forward denominator.
    rad = 2 * rho * (1 + 1e-6) / (1 + np.sqrt(1 + 4 * k * rho**2 * (1 + 1e-6)))
    dst = src * (rad / rho if rho else 1.) * config['fisheye_zoom']
    c, y = np.rint((dst + 1) * np.array([w, h]) / 2).astype(int)
    if not (0 <= c < w and 0 <= y < h) or id_map[y, c] == 0:
        return None, 'outside_compound_eye_mask'
    return int(id_map[y, c]) - 1, None


def main():
    # Only read scalar top-level numeric calibration entries, no YAML objects.
    config = {k: float(v) for k, v in re.findall(r'^([a-z_]+): ([0-9.]+)$',
        (ASSETS / 'vision.yaml').read_text(), flags=re.M)}
    geometry = json.loads((HERE / 'camera-geometry.json').read_text())
    with np.load(ASSETS / 'compound_eye.npz', allow_pickle=False) as z:
        id_map = z['ommatidia_id_map']
    frozen = json.loads(FROZEN.read_text(encoding='utf8'))
    ann = pd.read_feather(ANN).set_index('bodyId')
    report = {'schema': 'offline.retinal_registration.v1', 'live_map_modified': False,
        'author_atlas_commit': '503c7f055d5491a48b60b49ade8c71798d24d8f1',
        'frozen_map_json_sha256': sha(FROZEN), 'frozen_map_npz_sha256': sha(FROZEN.with_suffix('.npz')),
        'annotations_sha256': sha(ANN), 'compound_eye_sha256': sha(ASSETS / 'compound_eye.npz'),
        'vision_yaml_sha256': sha(ASSETS / 'vision.yaml'), 'camera_geometry_sha256': sha(HERE / 'camera-geometry.json'),
        'coordinate_convention': '+x anterior, +y left, +z dorsal; vector fields used, not signed phi',
        'status': 'author_atlas_registration_with_engineering_angular_resampling_not_physiological_calibration',
        'sides': {}}
    rows, rejected, column_rows = [], [], []
    arrays = {}
    for side, letter in [('left', 'L'), ('right', 'R')]:
        atlas = xlsx_numeric(ATLAS / f'pqxyztp_{side}.xlsx')
        assert not atlas.duplicated(['hex1', 'hex2']).any()
        assert np.all(atlas.p == atlas.hex2 - 19) and np.all(atlas.q == atlas.hex1 - 18)
        assert np.max(np.abs(np.linalg.norm(atlas[['x', 'y', 'z']], axis=1) - 1)) < 1e-10
        atlas.to_csv(HERE / f'author-atlas-{side}-full-precision.csv', index=False)
        pins = xlsx_numeric(HERE / 'visualpathways/params' / ('ME_L_columnar-cells_location.xlsx' if side == 'left' else 'ME_columnar-cells_location.xlsx'))
        matches, mismatches, missing_annotations = [], [], []
        crosswalk_witnesses = {}
        for _, p in pins.iterrows():
            for kind in ['L1', 'L2', 'L3', 'Mi1']:
                bid = p.get(kind, np.nan)
                if not np.isfinite(bid):
                    continue
                if int(bid) not in ann.index:
                    missing_annotations.append(int(bid)); continue
                a = ann.loc[int(bid)]
                actual = [a.assignedOlHex1, a.assignedOlHex2]
                expected = [p.hex1_id, p.hex2_id]
                good = actual == expected and a.somaSide == letter and a.type == kind
                (matches if good else mismatches).append({'body_id': int(bid), 'type': kind,
                    'author_hex': list(map(int, expected)),
                    'annotation_hex': [int(x) if np.isfinite(x) else None for x in actual], 'annotation_type': a.type,
                    'annotation_side': str(a.somaSide)})
                if np.isfinite(actual).all() and a.somaSide == letter and a.type == kind:
                    current = tuple(map(int, actual)); source = tuple(map(int, expected))
                    crosswalk_witnesses.setdefault(current, []).append({'body_id': int(bid), 'type': kind, 'author_hex': source})
        # Do not assume coordinate numbers survived the left-column revision.
        # Require agreement of at least two distinct identified columnar types,
        # then reject forward or inverse collisions in the author->current map.
        crosswalk, crosswalk_rejected = {}, {}
        for current, witnesses in crosswalk_witnesses.items():
            choices = {tuple(w['author_hex']) for w in witnesses}
            if len(choices) == 1 and len({w['type'] for w in witnesses}) >= 2:
                crosswalk[current] = next(iter(choices))
            else:
                crosswalk_rejected[current] = 'conflicting_or_insufficient_body_id_witnesses'
        inverse = {}
        for current, source in crosswalk.items(): inverse.setdefault(source, []).append(current)
        for source, currents in inverse.items():
            if len(currents) > 1:
                for current in currents:
                    crosswalk.pop(current); crosswalk_rejected[current] = 'non_bijective_body_id_crosswalk'
        dump(HERE / f'column-crosswalk-{side}.json', {
            'method': 'same body IDs across author column table and current annotation; >=2 distinct types, all agree, bijective',
            'author_atlas_to_author_column_table_relation': 'shared documented MaleCNS hex coordinate namespace; private atlas generator version not separately supplied',
            'accepted': [{'current_hex': list(k), 'author_hex': list(v), 'witnesses': crosswalk_witnesses[k]} for k,v in sorted(crosswalk.items())],
            'rejected': [{'current_hex': list(k), 'reason': v, 'witnesses': crosswalk_witnesses[k]} for k,v in sorted(crosswalk_rejected.items())]})
        camera = geometry['cameras'][side]
        rays, spreads, valid_fraction, _, _ = retina_rays(camera, config, id_map)
        arrays[f'{side}_ommatidium_head_rays'] = rays
        arrays[f'{side}_ommatidium_spread_deg'] = spreads
        tree = cKDTree(rays)
        byhex = {}
        for _, a in atlas.iterrows():
            ray = a[['x', 'y', 'z']].to_numpy(float)
            pixel_id, reason = atlas_pixel(ray, camera, config, id_map)
            _, nearest = tree.query(ray)
            rec = {'side': side, 'hex1': int(a.hex1), 'hex2': int(a.hex2), 'author_head_ray': ray.tolist(),
                'pixel_containing_ommatidium': pixel_id, 'ommatidium_index': int(nearest) if not reason else None,
                'rejected_reason': reason, 'angular_error_deg': float(angle(ray, rays[nearest])),
                'nearest_ommatidium_spread_deg': float(spreads[nearest])}
            column_rows.append(rec); byhex[(int(a.hex1), int(a.hex2))] = rec
        old = [r for r in frozen['mapped'] if r['side'] == side]
        for r in old:
            source_hex = crosswalk.get(tuple(r['hex']))
            match = byhex.get(source_hex) if source_hex else None
            rec = dict(body_id=r['body_id'], side=side, eye_index=r['eye_index'], hex=r['hex'],
                column_fraction=r['column_fraction'], column_synapses=r['column_synapses'],
                old_ommatidium_index=r['ommatidium_index'])
            if not source_hex:
                rejected.append(dict(rec, reason='no_unambiguous_body_id_column_crosswalk'))
            elif not match:
                rejected.append(dict(rec, reason='no_author_atlas_column'))
            elif match['rejected_reason']:
                rejected.append(dict(rec, reason=match['rejected_reason'], author_head_ray=match['author_head_ray']))
            else:
                rec.update(ommatidium_index=match['ommatidium_index'], author_head_ray=match['author_head_ray'],
                    author_hex=list(source_hex),
                    angular_error_deg=match['angular_error_deg'],
                    old_angular_error_deg=float(angle(match['author_head_ray'], rays[r['ommatidium_index']])) )
                rows.append(rec)
        accepted = [r for r in rows if r['side'] == side]
        rejected_here = [r for r in rejected if r['side'] == side]
        atlas_keys = set(byhex)
        pin_keys = set(zip(pins.hex1_id.astype(int), pins.hex2_id.astype(int)))
        valid_cols = [r for r in column_rows if r['side'] == side and not r['rejected_reason']]
        report['sides'][side] = {'author_columns': len(atlas), 'author_columns_on_pins': len(atlas_keys & pin_keys),
            'author_columns_not_on_pins': [list(x) for x in sorted(atlas_keys - pin_keys)],
            'official_body_id_hex_checks': {'matched': len(matches), 'mismatched': len(mismatches), 'absent_body_ids': len(missing_annotations)},
            'column_crosswalk': {'accepted': len(crosswalk), 'changed_coordinates': sum(k!=v for k,v in crosswalk.items()), 'rejected': len(crosswalk_rejected)},
            'author_columns_in_retina_field': len(valid_cols), 'mapped_receptors': len(accepted),
            'rejected_receptors': len(rejected_here), 'rejection_counts': pd.Series([r['reason'] for r in rejected_here]).value_counts().to_dict(),
            'unique_sampled_ommatidia': len(set(r['ommatidium_index'] for r in accepted)),
            'mean_ray_nearest_error_deg': quantiles([r['angular_error_deg'] for r in accepted]),
            'old_bbox_error_deg_on_same_receptors': quantiles([r['old_angular_error_deg'] for r in accepted]),
            'retina_footprint_radius_deg': quantiles(spreads), 'retina_valid_pixel_fraction': quantiles(valid_fraction),
            'source_xlsx_sha256': sha(ATLAS / f'pqxyztp_{side}.xlsx')}
        dump(HERE / f'body-id-column-check-{side}.json', {'matched': matches, 'mismatched': mismatches, 'absent': missing_annotations})
    rows.sort(key=lambda r: int(r['body_id']))
    arrays.update(body_ids=np.array([int(r['body_id']) for r in rows], np.int64),
        eye_indices=np.array([r['eye_index'] for r in rows], np.int8),
        ommatidium_indices=np.array([r['ommatidium_index'] for r in rows], np.int32),
        hex=np.array([r['hex'] for r in rows], np.int32),
        author_head_rays=np.array([r['author_head_ray'] for r in rows]),
        column_fraction=np.array([r['column_fraction'] for r in rows]))
    np.savez_compressed(HERE / 'candidate-receptor-map.npz', **arrays)
    dump(HERE / 'candidate-receptor-map.json', {'metadata': report, 'mapped': rows, 'rejected': rejected, 'columns': column_rows})
    dump(HERE / 'registration-summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
