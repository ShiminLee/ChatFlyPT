"""Measured Retina -> author optical axes -> MaleCNS column candidates.

Body-ID crosswalks and actual eye-camera geometry replace the former bounding
rectangle hypothesis. Optical registration does not calibrate physiology.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import ROOT, dump, sha256
from .social import MALE
from .retinal_registration import (
    ATLAS_COMMIT, COLUMNS_COMMIT, source_paths, prepare_registration, registered_column,
)

DATASET = 'MaleCNS v1.0'
NAMESPACE = 'malecns_body_id'
OUT = ROOT / 'data/derived/malecns/spatial-vision'
MAP = OUT / 'receptor-map.npz'
PROVENANCE = OUT / 'receptor-map.json'
EYE_ASSET = ROOT / 'vendor/flygym/src/flygym/assets/model/neuromechfly/compound_eye.npz'
EYE_CONFIG = EYE_ASSET.with_name('vision.yaml')
GRAPH = ROOT / 'data/derived/malecns/graph.npz'
MIN_COLUMN_SYNAPSES = 3
MIN_COLUMN_FRACTION = .5


def spatial_source_paths():
    """Current geometry/build inputs for map invalidation and model contract."""
    return (*source_paths(), MALE, GRAPH)


def _source_signature():
    try:
        return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in spatial_source_paths())
    except OSError as exc:
        raise RuntimeError('Spatial registration source missing; explicitly rebuild the map') from exc


def hex_cartesian(hexes):
    """Official q=hex1, p=hex2; +q right/up, +p left/up (paper diagram)."""
    h = np.asarray(hexes, dtype=float) - np.array([18., 19.])
    return np.column_stack((np.sqrt(3.) / 2. * (h[:, 0] - h[:, 1]),
                            .5 * (h[:, 0] + h[:, 1])))


def retina_centroids(id_map):
    """Locations follow pixel labels from the official asset, never ID sorting."""
    id_map = np.asarray(id_map)
    y, x = np.indices(id_map.shape)
    n = int(id_map.max())
    centers = np.array([[x[id_map == i].mean(), y[id_map == i].mean()]
                        for i in range(1, n + 1)])
    if not np.isfinite(centers).all():
        raise ValueError('Retina labels must all have actual pixels')
    lower, upper = centers.min(axis=0), centers.max(axis=0)
    # Horizontal camera pixels point right, rows point down.
    return 2. * (centers - (lower + upper) / 2.) / (upper - lower)


def build_spatial_map(save=True):
    """Rebuild from current graph evidence and fixed author optics, no old map."""
    raw = pd.read_feather(MALE)
    with np.load(GRAPH, allow_pickle=False) as graph:
        ids, ptr, targets, weights = (graph[k] for k in ('ids', 'indptr', 'indices', 'weights'))
    rows = raw.set_index('bodyId').reindex(ids)
    types = rows.type.fillna('').to_numpy()
    roots = rows.rootSide.fillna('').to_numpy()
    somas = rows.somaSide.fillna('').to_numpy()
    hexes = rows[['assignedOlHex1', 'assignedOlHex2']].to_numpy(float)
    has_hex = np.isfinite(hexes).all(axis=1)
    downstream = np.isin(types, ['L1', 'L2', 'L3']) & has_hex
    with np.load(EYE_ASSET, allow_pickle=False) as asset:
        id_map = asset['ommatidia_id_map']
        centroids = retina_centroids(id_map)
    registered = prepare_registration(raw.set_index('bodyId'), id_map)
    chosen, rejected, transforms = [], [], {}
    for eye_index, root in enumerate(('L', 'R')):
        side = 'left' if root == 'L' else 'right'
        registration = registered[side]
        crosswalk = registration['crosswalk']
        transforms[root] = {'method': 'author_optical_axes_body_id_crosswalk_actual_retina_rays',
            'atlas_commit': ATLAS_COMMIT, 'author_columns': len(registration['columns']),
            'body_id_crosswalk_columns': len(crosswalk),
            'crosswalk_changed_coordinates': sum(k != v for k, v in crosswalk.items()),
            'coordinate_basis': '+x anterior, +y left, +z dorsal; xyz used, not display phi',
            'camera_to_head_rotation': registration['camera']['head_rotation'],
            'fovy_deg': registration['camera']['fovy_deg'],
            'out_of_field_policy': 'reject; no edge, bbox or mean fallback',
            'registration_calibrated': False}
        for index in np.flatnonzero((types == 'R1-R6') & (roots == root)):
            posts = targets[ptr[index]:ptr[index + 1]]
            synapses = np.abs(weights[ptr[index]:ptr[index + 1]]) / .275
            keep = downstream[posts] & (somas[posts] == root)
            posts, synapses = posts[keep], synapses[keep]
            record = {'body_id': str(ids[index]), 'side': side}
            if not len(posts):
                rejected.append({**record, 'reason': 'No modeled same-eye L1/L2/L3 targets with official column labels'})
                continue
            votes = {}
            for post, strength in zip(posts, synapses):
                col = tuple(hexes[post].astype(int))
                votes[col] = votes.get(col, 0.) + float(strength)
            ranked = sorted(votes.items(), key=lambda x: (-x[1], x[0]))
            best, support = ranked[0]
            fraction = support / sum(votes.values())
            if (support < MIN_COLUMN_SYNAPSES or fraction < MIN_COLUMN_FRACTION or
                    (len(ranked) > 1 and np.isclose(support, ranked[1][1]))):
                rejected.append({**record, 'reason': 'Insufficient or ambiguous column support',
                    'best_column': list(map(int, best)), 'best_synapses': support, 'best_fraction': fraction})
                continue
            evidence = {**record, 'eye_index': eye_index, 'hex': list(map(int, best)),
                'column_synapses': support, 'column_fraction': fraction,
                'target_body_ids': [str(ids[p]) for p in posts],
                'column_candidates': [{'hex': list(map(int, h)), 'synapses': float(w)} for h, w in ranked]}
            optics, reason = registered_column(registration, best)
            if reason:
                rejected.append({**evidence, **(optics or {}), 'reason': reason})
            else:
                chosen.append({**evidence, **optics})
        accepted_eye = [r for r in chosen if r['side'] == side]
        errors = [r['angular_error_deg'] for r in accepted_eye]
        transforms[root].update(mapped_cells=len(accepted_eye),
            unique_sampled_ommatidia=len({r['ommatidium_index'] for r in accepted_eye}),
            angular_sampling_error_deg={'median': float(np.median(errors)) if errors else None,
                'p95': float(np.quantile(errors, .95)) if errors else None,
                'max': float(max(errors)) if errors else None})
    order = sorted(chosen, key=lambda r: int(r['body_id']))
    arrays = {'body_ids': np.array([int(r['body_id']) for r in order], np.int64),
              'eye_indices': np.array([r['eye_index'] for r in order], np.int8),
              'ommatidium_indices': np.array([r['ommatidium_index'] for r in order], np.int32),
              'hex': np.array([r['hex'] for r in order], np.int32),
              'column_fraction': np.array([r['column_fraction'] for r in order]),
              'retina_centroids': centroids,
              'author_head_rays': np.asarray([r['author_head_ray'] for r in order], float),
              'author_hex': np.asarray([r['author_hex'] for r in order], np.int32)}
    for side, registration in registered.items():
        arrays[f'{side}_ommatidium_head_rays'] = registration['ommatidium_head_rays']
    provenance = {'schema': 'chatflypt.spatial_vision.v2', 'dataset': DATASET,
        'source_sha256': {'annotations': sha256(MALE), 'graph': json.loads(GRAPH.with_suffix('.json').read_text(encoding='utf-8'))['graph_sha256'],
                          'compound_eye': sha256(EYE_ASSET), 'vision_config': sha256(EYE_CONFIG)},
        'registration_source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in spatial_source_paths()},
        'column_inference': {'targets': ['L1', 'L2', 'L3'], 'same_eye_soma_side_required': True,
            'min_column_synapses': MIN_COLUMN_SYNAPSES, 'min_column_fraction': MIN_COLUMN_FRACTION,
            'tie_policy': 'reject', 'weight_basis': 'Absolute current signed graph weight divided by unchanged 0.275 mV/synapse gain'},
        'transforms': transforms, 'registration_calibrated': False,
        'status': 'author_optical_axis_registration_candidate',
        'author_atlas_commit': ATLAS_COMMIT, 'author_columns_commit': COLUMNS_COMMIT,
        'data_license': 'CC BY-SA 4.0; Arthur Zhao / Reiser Lab; derived optical-axis/column map with current-body-ID crosswalk and FlyGym resampling',
        'column_crosswalks': {side: data['crosswalk_evidence'] for side, data in registered.items()},
        'author_optical_columns': {side: list(data['columns'].values()) for side, data in registered.items()},
        'sources': ['https://male-cns.janelia.org/download/',
            'https://www.nature.com/articles/s41586-025-08746-0',
            f'https://github.com/artxz/eyemap-archive/tree/{ATLAS_COMMIT}',
            f'https://github.com/reiserlab/visualpathways/tree/{COLUMNS_COMMIT}',
            'https://www.nature.com/articles/s41586-025-09276-5',
            'https://github.com/NeLy-EPFL/flygym', 'https://www.nature.com/articles/s41592-024-02497-y'],
        'limitations': ['Column inference is a synaptic connectivity hypothesis, not an individually traced photoreceptor receptive field.',
            'Author precomputed optical axes are registered across specimens; their private generating pipeline and individual biological error bars were not reproduced.',
            'Author and current column numbers are joined through unanimous same-body-ID witnesses; the left eye contains a documented column revision.',
            'The regular whole-eye atlas is used. The author DRA-named left alternative differs by about 4.82 degrees median; its detailed generation/selection rule is not public.',
            'Actual camera/Retina geometry passed controlled sphere-rendering checks; angular resampling is not physiological calibration, spectral sensitivity or graded photoreceptor dynamics.',
            'No invented columns or receptors; unassigned receptors receive no mean-light fallback.',
            'Local image variation is retained, but visual recognition and natural directional behavior are unvalidated.'],
        'mapped_count': len(order), 'rejected_count': len(rejected), 'mapped': order, 'rejected': rejected}
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(MAP, **arrays)
        provenance['map_sha256'] = sha256(MAP)
        dump(PROVENANCE, provenance)
    return arrays, provenance


@lru_cache(maxsize=2)
def _load_map(path, modified_ns, size, metadata_signature, source_signature):
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    metadata = json.loads(PROVENANCE.read_text(encoding='utf-8'))
    if metadata.get('map_sha256') != sha256(path):
        raise ValueError('Spatial receptor mapping checksum mismatch')
    recorded = metadata.get('registration_source_sha256', {})
    expected = {str(p.relative_to(ROOT)).replace('\\', '/'): p for p in spatial_source_paths()}
    if set(recorded) != set(expected):
        raise ValueError('Spatial registration source manifest mismatch; explicitly rebuild the map')
    for name, source in expected.items():
        if recorded[name] != sha256(source):
            raise ValueError(f'Spatial registration source changed: {name}; explicitly rebuild the map')
    return arrays, metadata


def load_spatial_map():
    if not MAP.exists() or not PROVENANCE.exists():
        raise RuntimeError('MaleCNS spatial receptor map must be explicitly built before use')
    stat, metadata_stat = MAP.stat(), PROVENANCE.stat()
    return _load_map(str(MAP), stat.st_mtime_ns, stat.st_size,
                     (metadata_stat.st_mtime_ns, metadata_stat.st_size), _source_signature())


def project_spatial_vision(vision, catalogue):
    """Accept only official measured ommatidial samples, with no object labels."""
    if vision.get('status') != 'measured':
        return {'status': 'unmeasured', 'dataset': DATASET, 'cell_ids': [], 'intensities': [],
                'registration_calibrated': False, 'per_eye': {}}
    arrays, metadata = load_spatial_map()
    values = np.asarray(vision.get('ommatidia'), dtype=float)
    expected = (2, len(arrays['retina_centroids']), 2)
    # Official Retina averages many pixels; roundoff can yield e.g.
    # 1.000000000000004 for an otherwise normalized measurement.
    tolerance = 1e-10
    if (values.shape != expected or not np.isfinite(values).all() or
            np.any(values < -tolerance) or np.any(values > 1. + tolerance)):
        raise ValueError(f'Expected finite official ommatidial samples with shape {expected} in [0,1]')
    values = np.clip(values, 0., 1.)
    spec = catalogue['channels']['mc_bright']
    if spec['dataset'] != DATASET or spec['id_namespace'] != NAMESPACE:
        raise ValueError('Spatial vision may only target MaleCNS photoreceptors')
    permitted = set(spec['side_ids']['bilateral'])
    keep = np.array([str(i) in permitted for i in arrays['body_ids']])
    ids = arrays['body_ids'][keep]
    eyes = arrays['eye_indices'][keep]
    ommatidia = arrays['ommatidium_indices'][keep]
    intensity = values.sum(axis=-1)[eyes, ommatidia]
    per_eye = {}
    for eye_index, side in enumerate(('left', 'right')):
        active = eyes == eye_index
        known = set(spec.get('anatomical_side_ids', spec['side_ids'])[side])
        mapped = {str(i) for i in ids[active]}
        per_eye[side] = {'mapped_cells': int(active.sum()), 'illuminated_cells': int(np.count_nonzero(intensity[active] > 0)),
            'has_visual_drive': bool(np.any(intensity[active] > 0)),
            'rejected_cell_ids': sorted(known - mapped, key=int),
            'unique_sampled_ommatidia': int(len(np.unique(ommatidia[active])))}
    return {'status': 'candidate_spatial_projection', 'dataset': DATASET, 'id_namespace': NAMESPACE,
        'channel': 'mc_bright', 'rate_hz': spec['rate_hz'],
        'cell_ids': ids.tolist(), 'intensities': intensity.tolist(),
        'eye_indices': eyes.tolist(), 'ommatidium_indices': ommatidia.tolist(),
        'per_eye': per_eye, 'registration_calibrated': False,
        'map_sha256': metadata['map_sha256'], 'transforms': metadata['transforms'],
        'object_identity_injected': False, 'mean_fallback': False,
        'limitations': metadata['limitations']}


def encode_spatial_vision(projection, seed, window_ms):
    """Encode locally measured receptor intensities without adding a baseline."""
    from .brain import DT_MS
    ticks = round(window_ms / DT_MS)
    if not 0 < window_ms <= 5000 or abs(ticks * DT_MS - window_ms) > 1e-8:
        raise ValueError('Visual window must align to the neural clock')
    if projection.get('dataset') != DATASET:
        raise ValueError('Visual projection dataset mismatch')
    if projection.get('status') == 'unmeasured':
        return None
    ids = np.asarray(projection['cell_ids'], dtype=np.int64)
    intensity = np.asarray(projection['intensities'], dtype=float)
    if (ids.shape != intensity.shape or not np.isfinite(intensity).all() or
            np.any(intensity < 0) or np.any(intensity > 2) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Invalid visual receptor intensities')
    rate = float(projection['rate_hz'])
    if not np.isfinite(rate) or not 0 <= rate <= 1000:
        raise ValueError('Invalid visual receptor rate')
    times, origins, driven = [], [], []
    for cell, value in zip(ids, intensity):
        if value <= 0:
            continue
        key = f'{int(seed)}\0{DATASET}\0{NAMESPACE}\0{cell}'.encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
        at = np.flatnonzero(rng.random(ticks) < rate * value * DT_MS / 1000.)
        times.append(at.astype(np.int32)); origins.append(np.full(len(at), cell, np.int64)); driven.append(cell)
    ts = np.concatenate(times) if times else np.empty(0, np.int32)
    cells = np.concatenate(origins) if origins else np.empty(0, np.int64)
    order = np.argsort(ts, kind='stable')
    return {'dataset': DATASET, 'id_namespace': NAMESPACE,
        'input_neuron_ids': np.asarray(driven, np.int64), 'event_steps': ts[order],
        'event_neuron_ids': cells[order], 'seed': int(seed), 'duration_ms': float(window_ms), 'dt_ms': DT_MS}


if __name__ == '__main__':
    _, report = build_spatial_map()
    print(json.dumps({k: report[k] for k in ('mapped_count', 'rejected_count', 'map_sha256')}, indent=2))
