"""Blind motor-side readout: does internal activity predict the body's own motion?

The inference boundary is the same as the sensory readout: spike counts, passive
membrane statistics and the observation duration. Nothing about the user's
message, the scene, the intended stimulus or the recorded commands is a feature.

Labels are measured body outcomes (yaw change and path length actually produced
by the pinned physical controller), never the human's instruction and never the
decoder's own scores. The body is driven by recorded neural activity through a
pinned engineering controller, so part of what this decoder can learn is that
controller's input-to-output map; a qualified label is therefore a candidate
prediction of ongoing motion, not proof of innate intent.

Artifacts bind the same model contract as the sensory readout, so a graph,
input-mapping or transduction change makes the decoder abstain instead of
silently reusing stale weights.
"""
import json
from pathlib import Path
from functools import lru_cache

import numpy as np

from .paths import ROOT, sha256
from .unified_readout import (DATASET, SCHEMA as SENSORY_SCHEMA, blind_mask,
                              internal_features, model_contract, _array_digest)

OUT = ROOT / 'data/derived/intent-readout'
SCHEMA = 'chatflypt.intent-readout.v1'
MIN_VOLTAGE_MV = 1e-6
FEATURES = 'sqrt spikes; log1p positive/negative voltage area; normalized to 500 ms'
# Measured-outcome label definitions. The thresholds are quantiles taken from
# the TRAINING split alone and preregistered before validation or test windows
# are labelled; they are never tuned on validation or test.
TURN_QUANTILE = .85
ACTIVITY_QUANTILE = .5
LABELS = ('turn:left', 'turn:right', 'move:active', 'move:still')
LABEL_MEANING = {
    'turn:left': '实测身体朝向左转，超过训练集转向幅度分位阈值',
    'turn:right': '实测身体朝向右转，超过训练集转向幅度分位阈值',
    'move:active': '实测位移高于训练集中位数',
    'move:still': '实测位移低于训练集中位数',
}


def measured_outcome(trajectory):
    """Compute the body's own outcome from recorded poses, not from commands."""
    first, last = trajectory[0], trajectory[-1]
    yaw = []
    for frame in (first, last):
        w, x, y, z = frame['body_quaternions_wxyz'][0]
        yaw.append(np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))))
    delta = (yaw[1] - yaw[0] + 180.) % 360. - 180.
    path = 0.
    for a, b in zip(trajectory, trajectory[1:]):
        path += float(np.linalg.norm(np.asarray(b['body_positions_mm'][0]) - np.asarray(a['body_positions_mm'][0])))
    return {'yaw_change_deg': float(delta), 'path_length_mm': float(path)}


def outcome_rules(outcomes):
    """Fix outcome thresholds from the training split's own distribution."""
    yaw = np.abs([row['yaw_change_deg'] for row in outcomes])
    path = np.array([row['path_length_mm'] for row in outcomes])
    return {'turn_deg': float(np.quantile(yaw, TURN_QUANTILE)),
            'move_mm': float(np.quantile(path, ACTIVITY_QUANTILE)),
            'turn_quantile': TURN_QUANTILE, 'activity_quantile': ACTIVITY_QUANTILE,
            'rule': 'Quantiles of the measured outcome distribution in the training split only'}


def outcome_labels(outcome, rules):
    """Map one measured outcome to its labels; a window may carry several."""
    words = []
    if outcome['yaw_change_deg'] >= rules['turn_deg']:
        words.append('turn:left')
    elif outcome['yaw_change_deg'] <= -rules['turn_deg']:
        words.append('turn:right')
    words.append('move:active' if outcome['path_length_mm'] >= rules['move_mm'] else 'move:still')
    return words


def labels_to_vector(words, labels=LABELS):
    return np.array([1 if label in words else 0 for label in labels], dtype=bool)


# ---------------------------------------------------------------------------
# Continuous mode: predict the measured self-motion vector, then discretize.
#
# Borrowed idea (not code): the vendored flyvis decoder head regresses a
# *continuous* task vector (optical flow) from population activity and scores it
# with a vector objective (`flyvis/task/objectives.py`, MIT). Applying the same
# shape here removes the "too few positive windows" bottleneck of discrete labels:
# every window carries a yaw and a path value, so qualification can use the
# regression across all windows instead of a handful of rare classes.
#
# Preregistered rules (fixed before the v5 corpus exists, see the protocol file):
#   turn  qualifies if r(predicted yaw, measured yaw) >= TURN_R_MIN over
#         validation AND the sign of the prediction matches the measured sign in
#         at least TURN_SIGN_MIN of the "turning" windows (|yaw| >= the training
#         split's TURN_QUANTILE).
#   move  qualifies if r(predicted path, measured path) >= MOVE_R_MIN AND a
#         threshold at the training median reaches precision >= MOVE_PRECISION and
#         recall >= MOVE_RECALL on validation.
# A component that fails abstains; it never falls back to the other component.
CONTINUOUS_SCHEMA = 'chatflypt.intent-continuous.v1'
TARGETS = ('yaw_change_deg', 'path_length_mm')
TURN_R_MIN = .5
TURN_SIGN_MIN = .75
MOVE_R_MIN = .5
MOVE_PRECISION = .85
MOVE_RECALL = .70


def standardize(values, mean, scale):
    return (np.asarray(values, dtype=float) - mean) / scale


class ContinuousIntentDecoder:
    """Regresses the measured self-motion vector from blind internal activity."""

    intent = True
    continuous = True

    def __init__(self, brain, catalogue, path=None):
        path = Path(path or OUT / 'malecns-intent.npz')
        self.contract = model_contract(brain, catalogue)
        self.allowed = blind_mask(brain, catalogue)
        self.neurons = len(brain.ids)
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            self.feature_indices = saved['feature_indices']
            self.indices = saved['indices']
            self.weights = saved['weights']
            self.intercept = saved['intercept']
            self.target_mean = saved['target_mean']
            self.target_scale = saved['target_scale']
            self.rules = json.loads(str(saved['rules']))
            self.qualified = saved['qualified']
        if metadata.get('contract') != self.contract:
            raise ValueError('Intent readout graph, input mappings or blind mask is stale')
        if metadata.get('schema') != CONTINUOUS_SCHEMA or metadata.get('mode') != 'continuous':
            raise ValueError('Not a continuous intent artifact')
        labels, features = len(TARGETS), len(self.feature_indices)
        if (self.weights.shape != (labels, features) or self.intercept.shape != (labels,) or
                self.target_mean.shape != (labels,) or self.target_scale.shape != (labels,) or
                self.qualified.shape != (labels,) or self.qualified.dtype.kind != 'b' or
                not all(np.isfinite(a).all() for a in
                        (self.weights, self.intercept, self.target_mean, self.target_scale)) or
                np.any(self.target_scale <= 0)):
            raise ValueError('Invalid continuous intent artifact')
        durations = metadata.get('tested_durations_ms', [])
        if not durations or any(isinstance(x, bool) or not isinstance(x, (float, int)) or
                                not np.isfinite(x) or not 0 < x <= 5000 for x in durations):
            raise ValueError('Artifact needs explicit tested observation durations')
        self.metadata = metadata
        self.artifact_sha256 = sha256(path)
        self.artifact = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    def predict(self, counts, state_record, duration_ms=500.):
        values = internal_features(counts, state_record, duration_ms)
        if len(counts) != self.neurons:
            raise ValueError('Observation belongs to a different neuron index')
        standardized = self.weights @ values[self.feature_indices] + self.intercept
        return standardized * self.target_scale + self.target_mean

    def decode(self, counts, state_record, duration_ms=500.):
        note = {'applied': False, 'artifact': self.artifact, 'sha256': self.artifact_sha256,
                'scope': 'Candidate motor-side regression from internal activity; measured-outcome targets',
                'globally_calibrated': False, 'biologically_calibrated': False,
                'qualified_targets': [name for name, q in zip(TARGETS, self.qualified) if q]}
        if duration_ms not in self.metadata['tested_durations_ms']:
            note['reason'] = 'Outside validated observation durations; no extrapolated intent'
            return {'words': [], 'prediction': None, 'status': 'uncalibrated', 'intent_adaptation': note}
        prediction = self.predict(counts, state_record, duration_ms)
        yaw, path = float(prediction[0]), float(prediction[1])
        has_signal = bool(np.any(np.asarray(counts)) or
                          np.any(np.asarray(state_record)[2:] > MIN_VOLTAGE_MV))
        words = []
        turn_qualified, move_qualified = bool(self.qualified[0]), bool(self.qualified[1])
        if turn_qualified and has_signal:
            if yaw >= self.rules['turn_deg']:
                words.append('turn:left')
            elif yaw <= -self.rules['turn_deg']:
                words.append('turn:right')
        if move_qualified and has_signal:
            words.append('move:active' if path >= self.rules['move_mm'] else 'move:still')
        note['applied'] = bool(self.qualified.any())
        return {'words': words, 'prediction': {'yaw_change_deg': yaw, 'path_length_mm': path},
                'descriptions': {w: LABEL_MEANING[w] for w in words},
                'unqualified_words': [name for name, q in zip(TARGETS, self.qualified) if not q],
                'status': 'decoded', 'score_type': 'Regression of measured self-motion, not a probability',
                'scope': 'Finite candidate motor-side readout; not innate intent or biological calibration',
                'intent_adaptation': note}


class IntentDecoder:
    """A finite, per-label-qualified motor-side decoder. Reads internal activity only."""

    intent = True

    def __init__(self, brain, catalogue, path=None):
        path = Path(path or OUT / 'malecns-intent.npz')
        self.contract = model_contract(brain, catalogue)
        self.allowed = blind_mask(brain, catalogue)
        self.neurons = len(brain.ids)
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            for key in ('feature_indices', 'indices', 'weights', 'intercept', 'thresholds', 'qualified'):
                setattr(self, key, saved[key])
            self.labels = saved['labels'].tolist()
            self.names = dict(zip(self.labels, saved['names'].tolist()))
        if metadata.get('contract') != self.contract:
            raise ValueError('Intent readout graph, input mappings or blind mask is stale')
        self.metadata = metadata
        labels, features = len(self.labels), len(self.feature_indices)
        if (metadata.get('schema') != SCHEMA or metadata.get('input_mode') != 'environment_only' or
                metadata.get('split_unit') != 'whole_session' or not metadata.get('protocol_sha256') or
                len(set(self.labels)) != labels or labels == 0 or
                self.feature_indices.ndim != 1 or self.feature_indices.dtype.kind not in 'iu' or
                len(np.unique(self.feature_indices)) != features or features == 0 or
                np.any(self.feature_indices < 0) or np.any(self.feature_indices >= 3 * self.neurons) or
                self.weights.shape != (labels, features) or
                any(a.shape != (labels,) for a in (self.intercept, self.thresholds, self.qualified)) or
                self.qualified.dtype.kind != 'b' or
                not all(np.isfinite(a).all() for a in (self.weights, self.intercept, self.thresholds))):
            raise ValueError('Invalid intent readout artifact')
        durations = metadata.get('tested_durations_ms', [])
        if not durations or any(isinstance(x, bool) or not isinstance(x, (float, int)) or
                                not np.isfinite(x) or not 0 < x <= 5000 for x in durations):
            raise ValueError('Artifact needs explicit tested observation durations')
        self.artifact_sha256 = sha256(path)
        self.artifact = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    def decode(self, counts, state_record, duration_ms=500.):
        values = internal_features(counts, state_record, duration_ms)
        if len(counts) != self.neurons:
            raise ValueError('Observation belongs to a different neuron index')
        note = {'applied': False, 'artifact': self.artifact, 'sha256': self.artifact_sha256,
                'protocol_sha256': self.metadata['protocol_sha256'],
                'scope': 'Candidate motor-side prediction from internal activity; measured-outcome labels',
                'globally_calibrated': False, 'biologically_calibrated': False}
        if duration_ms not in self.metadata['tested_durations_ms']:
            note['reason'] = 'Outside validated observation durations; no extrapolated intent'
            return {'words': [], 'scores': {}, 'status': 'uncalibrated', 'intent_adaptation': note}
        scores = self.weights @ values[self.feature_indices] + self.intercept
        has_signal = bool(np.any(np.asarray(counts)) or
                          np.any(np.asarray(state_record)[2:] > MIN_VOLTAGE_MV))
        positive = (scores >= self.thresholds) & has_signal
        words = [w for w, yes, q in zip(self.labels, positive, self.qualified) if yes and q]
        note.update(applied=bool(self.qualified.any()),
                    labels=[w for w, q in zip(self.labels, self.qualified) if q])
        return {'words': words, 'scores': dict(zip(self.labels, map(float, scores))),
                'descriptions': {w: self.names[w] for w in words},
                'unqualified_words': [w for w, yes, q in zip(self.labels, positive, self.qualified) if yes and not q],
                'status': 'decoded', 'score_type': 'Linear neural-state scores, not probabilities',
                'scope': 'Finite candidate motor-side readout; not innate intent or biological calibration',
                'intent_adaptation': note}


def load_intent_decoder(brain, catalogue, path=None):
    """Absent or stale weights stay explicit; no label is inferred from behaviour."""
    path = Path(path or OUT / 'malecns-intent.npz')
    if not path.is_file():
        return None, {'applied': False, 'reason': 'Intent readout has not been trained',
                      'globally_calibrated': False, 'biologically_calibrated': False}
    try:
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
        decoder_class = ContinuousIntentDecoder if metadata.get('mode') == 'continuous' else IntentDecoder
        return decoder_class(brain, catalogue, path), None
    except (ValueError, KeyError, OSError, TypeError) as error:
        return None, {'applied': False, 'reason': 'Intent readout unavailable: ' + str(error),
                      'globally_calibrated': False, 'biologically_calibrated': False}


def intent_report(decoded, outcome=None):
    """Render qualified intent labels as a separate, explicitly candidate statement.

    Measured outcome is only ever used as evidence alongside the prediction; it is
    never a decoder input and never silently replaces an abstention.
    """
    result = {'schema': 'chatflypt.intent-expression.v1', 'source': 'blind_motor_readout',
              'status': 'unavailable', 'text': None, 'labels': [], 'llm_called': False,
              'measured_outcome': outcome, 'prediction': None,
              'limitations': ['文字来自内部活动的有限运动侧读出，不是已证明的主观意图。',
                              '身体由记录到的神经活动经固定工程控制器驱动；候选标签可能部分反映该控制器映射。']}
    if not isinstance(decoded, dict):
        return result
    result['status'] = decoded.get('status', 'unavailable')
    result['prediction'] = decoded.get('prediction')
    words = [w for w in decoded.get('words', []) if w in LABEL_MEANING]
    if result['status'] != 'decoded' or not words:
        return result
    # Conflicting directions cannot be asserted; competing labels are reported.
    if 'turn:left' in words and 'turn:right' in words:
        result['status'] = 'ambiguous'
        return result
    phrases = {'turn:left': '它正在向左转', 'turn:right': '它正在向右转',
               'move:active': '它在走动', 'move:still': '它基本停着不动'}
    result['labels'] = words
    result['text'] = '候选运动意图（有限验证）：' + '，'.join(phrases[w] for w in words) + '。'
    return result
