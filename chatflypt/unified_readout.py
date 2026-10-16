"""Blind readout for one MaleCNS brain under measured environmental feedback.

The inference boundary is counts, membrane statistics and observation duration.
Inputs, scene objects, intended actions and receptor events are never features.
New populations require new measured-feedback training; this module does not
transfer the FlyWire decoder or call an adviser.
"""
import hashlib
import json
from pathlib import Path
from functools import lru_cache

import numpy as np

from .paths import ROOT, sha256

DATASET = 'MaleCNS v1.0'
OUT = ROOT / 'data/derived/unified-readout'
SCHEMA = 'chatflypt.unified-neural-readout.v1'
MIN_VOLTAGE_MV = 1e-6
FEATURES = 'sqrt spikes; log1p positive/negative voltage area; normalized to 500 ms'
SOURCE_ARTIFACTS = {
    'malecns_annotations': ROOT / 'data/raw/malecns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'spatial_vision_receptor_map': ROOT / 'data/derived/malecns/spatial-vision/receptor-map.npz',
    'spatial_vision_mapping_provenance': ROOT / 'data/derived/malecns/spatial-vision/receptor-map.json',
}
# These are output meanings learned afresh within MaleCNS, not shared weights.
SEMANTIC_GROUPS = {f'mc_{key}': key for key in (
    'bright', 'yeast', 'vinegar', 'geosmin', 'cva', 'co2', 'warm', 'cold', 'moist', 'dry',
    'auditory')}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _array_digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _file_digest(path, modified_ns, size):
    return sha256(path)


def file_fingerprint(path):
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    return _file_digest(str(path), stat.st_mtime_ns, stat.st_size)


def source_artifact_paths():
    """Bind current eye geometry and source code, not only a cached projection."""
    paths = dict(SOURCE_ARTIFACTS)
    from . import spatial_vision
    if hasattr(spatial_vision, 'spatial_source_paths'):
        for raw_path in spatial_vision.spatial_source_paths():
            path = Path(raw_path).resolve()
            key = str(path.relative_to(ROOT)).replace('\\', '/') if path.is_relative_to(ROOT) else str(path)
            paths['spatial:' + key] = path
    return paths


def blind_mask(brain, catalogue):
    """Exclude ALL sensory and registered input cells, including advanced groups.

    A training/serving mismatch must be visible: silently dropping an unknown
    registered input could admit an answer-bearing feature after later edits.
    """
    allowed = np.asarray(brain.nonsensory, dtype=bool).copy()
    if allowed.shape != (len(brain.ids),):
        raise ValueError('MaleCNS nonsensory annotation mask is required')
    for spec in catalogue['channels'].values():
        if spec['dataset'] != DATASET:
            continue
        ids = set(map(str, spec.get('neuron_ids', [])))
        for values in spec.get('side_ids', {}).values():
            ids.update(map(str, values))
        for sides in spec.get('body_part_ids', {}).values():
            for values in sides.values():
                ids.update(map(str, values))
        for body_id in ids:
            if body_id not in brain.lookup:
                raise ValueError('Registered MaleCNS input absent from current graph: ' + body_id)
            allowed[brain.lookup[body_id]] = False
    return allowed


def model_contract(brain, catalogue):
    """Fingerprint actual graph, neuron order, input mappings and LIF code.

    Hash arrays afresh so even an in-place experimental graph edit cannot
    accidentally retain a decoder that belongs to the previous dynamics.
    """
    if getattr(brain, 'dataset', None) != DATASET:
        raise ValueError('Unified decoder requires one MaleCNS graph')
    allowed = blind_mask(brain, catalogue)
    graph = {key: _array_digest(getattr(brain, key))
             for key in ('ids', 'indptr', 'indices', 'weights')}
    keys = ('id', 'dataset', 'neuron_ids', 'side_ids', 'body_part_ids', 'rate_hz',
            'profile', 'profile_parameters', 'readout_group', 'status')
    mappings = [{key: spec.get(key) for key in keys}
                for _, spec in sorted(catalogue['channels'].items()) if spec['dataset'] == DATASET]
    engine = {name: sha256(ROOT / 'chatflypt' / name)
              for name in ('brain.py', 'malecns.py', 'neural_session.py', 'sensory_events.py', 'social.py')}
    # A new receptor/physics background changes the training distribution even
    # when the connectome itself is unchanged. UI and HTTP code are excluded.
    physical_names = ('environment.py', 'embodied.py', 'body_feedback.py',
                      'physical_touch.py', 'malecns_senses.py', 'proprioception.py', 'spatial_vision.py',
                      'behavior_readout.py', 'behavior_body.py', 'joint_motor.py', 'live_environment.py',
                      'body_measurement.py')
    transduction = {name: sha256(ROOT / 'chatflypt' / name)
                   if (ROOT / 'chatflypt' / name).is_file() else None for name in physical_names}
    sources = {name: file_fingerprint(path) for name, path in source_artifact_paths().items()}
    readout_modules = {name: sha256(ROOT / 'chatflypt' / name)
                       for name in ('unified_readout.py', 'intent_readout.py', 'plume_field.py')}
    return {'schema': SCHEMA, 'dataset': DATASET, 'neurons': len(brain.ids),
            'graph': graph, 'input_mapping_sha256': _digest(mappings),
            'blind_mask_sha256': _array_digest(allowed), 'engine': engine,
            'physical_transduction': transduction,
            'readout_modules': readout_modules,
            'source_artifacts': sources,
            'readout_features_sha256': file_fingerprint(__file__),
            'features': FEATURES}


def internal_features(counts, state_record, duration_ms):
    """Only passive neural measurements; no stimulus metadata is accepted."""
    counts = np.asarray(counts)
    state = np.asarray(state_record)
    if (counts.ndim != 1 or state.shape != (4, len(counts)) or
            not np.isfinite(duration_ms) or duration_ms <= 0 or
            not np.isfinite(counts).all() or not np.isfinite(state).all() or
            np.any(counts < 0) or np.any(state < 0)):
        raise ValueError('Finite nonnegative neural counts and membrane statistics required')
    scale = 500. / duration_ms
    return np.concatenate((np.sqrt(counts * scale), np.log1p(state[0] * scale),
                           np.log1p(state[1] * scale)))


class UnifiedDecoder:
    """A finite, per-label-qualified environmental neural-state decoder."""

    unified = True

    def __init__(self, brain, catalogue, path=None):
        path = Path(path or OUT / 'malecns-decoder.npz')
        self.contract = model_contract(brain, catalogue)
        self.allowed = blind_mask(brain, catalogue)
        self.neurons = len(brain.ids)
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            for key in ('feature_indices', 'indices', 'weights', 'intercept', 'thresholds', 'qualified'):
                setattr(self, key, saved[key])
            if (saved['labels'].ndim != 1 or saved['names'].shape != saved['labels'].shape or
                    saved['labels'].dtype.kind not in 'US' or saved['names'].dtype.kind not in 'US'):
                raise ValueError('Readout labels and names must be aligned strings')
            self.labels = saved['labels'].tolist()
            self.names = dict(zip(self.labels, saved['names'].tolist()))
        if metadata.get('contract') != self.contract:
            raise ValueError('Unified readout graph, input mappings or blind mask is stale')
        self.metadata = metadata
        labels, features = len(self.labels), len(self.feature_indices)
        if (metadata.get('schema') != SCHEMA or metadata.get('input_mode') != 'environment_only' or
                metadata.get('split_unit') != 'whole_session' or
                not metadata.get('protocol_sha256') or
                len(set(self.labels)) != labels or labels == 0 or
                self.feature_indices.ndim != 1 or self.feature_indices.dtype.kind not in 'iu' or
                self.indices.ndim != 1 or self.indices.dtype.kind not in 'iu' or
                len(np.unique(self.feature_indices)) != features or features == 0 or
                np.any(self.feature_indices < 0) or np.any(self.feature_indices >= 3 * self.neurons) or
                not np.array_equal(self.indices, np.unique(self.feature_indices % self.neurons)) or
                not np.all(self.allowed[self.indices]) or
                self.weights.shape != (labels, features) or
                any(a.shape != (labels,) for a in (self.intercept, self.thresholds, self.qualified)) or
                self.qualified.dtype.kind != 'b' or
                not all(np.isfinite(a).all() for a in (self.weights, self.intercept, self.thresholds))):
            raise ValueError('Invalid or unblinded unified readout artifact')
        durations = metadata.get('tested_durations_ms', [])
        if not durations or any(isinstance(x, bool) or not isinstance(x, (float, int)) or
                                not np.isfinite(x) or not 0 < x <= 5000 for x in durations):
            raise ValueError('Artifact needs explicit tested observation durations')
        self.catalogue_signature = self.contract['input_mapping_sha256']
        self.artifact_sha256 = sha256(path)
        self.artifact = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    def decode_trial(self, trial):
        return self.decode(trial.counts, trial.state_record, trial.duration_ms)

    def decode(self, counts, state_record, duration_ms=500.):
        values = internal_features(counts, state_record, duration_ms)
        if len(counts) != self.neurons:
            raise ValueError('Observation belongs to a different neuron index')
        note = {'applied': False, 'artifact': self.artifact, 'sha256': self.artifact_sha256,
                'protocol_sha256': self.metadata['protocol_sha256'],
                'scope': 'Session-separated measured environmental simulation; finite candidate readout',
                'globally_calibrated': False, 'biologically_calibrated': False}
        if duration_ms not in self.metadata['tested_durations_ms']:
            note['reason'] = 'Outside validated observation durations; no extrapolated labels'
            return {'words': [], 'scores': {}, 'descriptions': {}, 'unqualified_words': [],
                    'status': 'uncalibrated', 'feedback_adaptation': note}
        scores = self.weights @ values[self.feature_indices] + self.intercept
        has_signal = bool(np.any(np.asarray(counts)[self.indices]) or
                          np.any(np.asarray(state_record)[2:, self.indices] > MIN_VOLTAGE_MV))
        positive = (scores >= self.thresholds) & has_signal
        words = [w for w, yes, q in zip(self.labels, positive, self.qualified) if yes and q]
        note.update(applied=bool(self.qualified.any()), labels=[w for w, q in zip(self.labels, self.qualified) if q])
        return {'words': words, 'scores': dict(zip(self.labels, map(float, scores))),
                'descriptions': {w: self.names[w] for w in words},
                'unqualified_words': [w for w, yes, q in zip(self.labels, positive, self.qualified) if yes and not q],
                'status': 'decoded', 'score_type': 'Linear neural-state scores, not probabilities',
                'scope': 'Finite candidate internal readout; no intention, language understanding or biological calibration',
                'feedback_adaptation': note}


def load_unified_decoder(brain, catalogue, path=None):
    """Absent/stale weights remain explicit; never infer from scene labels."""
    path = Path(path or OUT / 'malecns-decoder.npz')
    if not path.is_file():
        return None, {'applied': False, 'reason': 'Unified environmental readout has not been trained',
                      'globally_calibrated': False, 'biologically_calibrated': False}
    try:
        return UnifiedDecoder(brain, catalogue, path), None
    except (ValueError, KeyError, OSError) as error:
        return None, {'applied': False, 'reason': 'Unified readout unavailable: ' + str(error),
                      'globally_calibrated': False, 'biologically_calibrated': False}
