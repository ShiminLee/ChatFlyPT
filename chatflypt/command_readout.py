"""Blind readout of the steering command the fly itself issued.

This is the motor-command readout, not the measured-motion readout in
`intent_readout.py`. The target is the pinned controller's own steering command
recovered from the recorded `left`/`right` leg-stepping drives of the same
window: `steer = mean(right - left)` over the bins that actually acted. Nothing
about the user's message, the scene, the planned stimulus, the measured body
motion or the decoder's own scores is a feature or a target.

Two guardrails keep this from becoming a restatement of the pinned formula:

* Features are the blind internal set (no annotated sensory class, no registered
  input cell) **minus every cell of the mapped motor groups**. The command is a
  pure function of those groups' filtered rates, so reading it from them would
  prove nothing. Reading it from the rest of the brain is the actual question.
* The same fit is also run on the motor groups alone and reported as a ceiling
  control, so the shortcut that was excluded is visible instead of implied.

The direction word ("left"/"right") is calibrated by measurement, not assumed:
`reports/command-readout/controller-sign-probe.json` records which way the
pinned body moves when one mapped steering group is active on one side.
"""
import json
from pathlib import Path

import numpy as np

from .paths import ROOT, sha256
from .unified_readout import DATASET, blind_mask, internal_features, model_contract

OUT = ROOT / 'data/derived/command-readout'
SCHEMA = 'chatflypt.command-readout.v1'
FEATURES = 'sqrt spikes; log1p positive/negative voltage area; normalized to 500 ms'
TARGETS = ('steer_command',)
MIN_VOLTAGE_MV = 1e-6
# Preregistered before any window was scored (see reports/command-readout/protocol.json).
COMMAND_R_MIN = .5
COMMAND_SIGN_MIN = .75
ZERO_COMMAND_FP_MAX = .10
MIN_INFORMATIVE_WINDOWS = 6
# A direction word is only served if the validation split actually contains that
# direction AND the readout gets its sign right there: the join round showed a
# direction can have label support while never being predicted (left: 0 of 4
# validation windows), which must not become a spoken word.
MIN_DIRECTION_WINDOWS = 2
MIN_DIRECTION_SIGN = .5
SIGN_PROBE = ROOT / 'reports/command-readout/controller-sign-probe.json'
LABELS = ('command:turn-left', 'command:turn-right')
LABEL_MEANING = {
    'command:turn-left': '它自己下发的踏步驱动不对称指向左转（指令读出，未经身体行为验证）',
    'command:turn-right': '它自己下发的踏步驱动不对称指向右转（指令读出，未经身体行为验证）',
}
SCOPE = ('它自己下发的转向指令（候选）：读出记录到的下行动作息不对称；'
         '不是实测转动，也不是已证明的主观意图。')


def turn_sign():
    """Which sign of (right - left) is a left turn, from the measured probe.

    Both steering groups attenuate the same-side strides (DNa02) or lengthen the
    contralateral ones (DNg13), so the fly turns away from the side whose drive
    is reduced. The probe measures that on the pinned body instead of assuming
    it. Fail-closed: without the calibration record the sign is unknown and the
    direction words are withheld rather than guessed.
    """
    if not SIGN_PROBE.is_file():
        return None, 'controller sign probe has not been run'
    record = json.loads(SIGN_PROBE.read_text(encoding='utf8'))
    summary = record.get('summary') or {}
    evidence = {}
    # Required: the two inside-stride conditions, where both repeats must agree
    # with the convention and with each other.
    for condition, sign in (('forward_turn_inside_L', +1), ('forward_turn_inside_R', -1)):
        entry = summary.get(condition) or {}
        yaw = entry.get('yaw_deg')
        if not yaw:
            return None, 'controller sign probe is missing condition ' + condition
        mean = float(np.mean(yaw))
        evidence[condition] = {'mean_yaw_deg': mean, 'repeats': len(yaw), 'required': True}
        if (sign > 0 and mean <= 0) or (sign < 0 and mean >= 0):
            return None, (f'controller sign probe disagrees with the inside-stride convention: '
                          f'{condition} yaw {mean:+.3f}')
    # Supporting evidence, recorded but not required: the contralateral
    # outside-stride conditions and the no-asymmetry control (its spread is the
    # measured floor below which a yaw difference says nothing).
    for condition in ('turn_outside_L_only', 'turn_outside_R_only', 'forward_only',
                      'forward_both_sides_plus_context'):
        entry = summary.get(condition) or {}
        if entry.get('yaw_deg'):
            evidence[condition] = {'mean_yaw_deg': float(np.mean(entry['yaw_deg'])),
                                   'sign_consistent': entry.get('sign_consistent'),
                                   'repeats': len(entry['yaw_deg']), 'required': False}
    return +1, {'probe': str(SIGN_PROBE.relative_to(ROOT)), 'sha256': sha256(SIGN_PROBE),
                'convention': 'steer = mean(right - left) over the window frames; positive steer = left turn',
                'conditions': evidence}


def command_target(body):
    """The steering command that acted in one window, from its own physical run.

    Every sampled frame of the recorded body replay carries the command that was
    in force at that instant and the index of the drive bin it came from, so the
    label is read straight off the measurement instead of being re-derived from
    an indexing assumption. The first frame is the state before the window's
    first integration step (it still holds the previous completed bin) and is
    excluded; the remaining frames cover the window.
    """
    frames = body['trajectory'][1:] or body['trajectory']
    left = np.asarray([frame['commands']['left'] for frame in frames], dtype=float)
    right = np.asarray([frame['commands']['right'] for frame in frames], dtype=float)
    return {'steer_command': float(np.mean(right - left)),
            'walk_drive': float(np.mean((left + right) / 2)),
            'left_max': float(left.max()), 'right_max': float(right.max()),
            'sampled_frames': int(len(frames)),
            'command_bins': sorted({int(frame['source_command_bin']) for frame in frames})}


def training_mask(split):
    """Accept either a boolean split mask or the split names themselves."""
    array = np.asarray(split)
    if array.dtype.kind == 'b':
        return array
    return np.asarray([str(name) == 'train' for name in array])


def command_rules(commands, split):
    """Fix the informative-window threshold from the training split alone.

    The split argument may be the boolean training mask or the split names; a
    name is only the training split when it reads exactly "train", never merely
    because a non-empty string is truthy.
    """
    mask = training_mask(split)
    values = np.abs([row['steer_command'] for row, is_train in zip(commands, mask) if is_train])
    quantile = float(np.quantile(values, .85)) if values.size else 0.
    return {'informative_abs_command': quantile, 'quantile': .85,
            'zero_epsilon': 1e-12,
            'rule': 'Quantile of |steer command| in the training split only; the threshold is fixed '
                    'before any validation or test window is scored'}


def command_labels(entry, rules):
    """Name one window's command; silence and sub-threshold asymmetry stay unnamed."""
    if abs(entry['steer_command']) < max(rules['informative_abs_command'], rules['zero_epsilon']):
        return []
    return ['command:turn-left'] if entry['steer_command'] > 0 else ['command:turn-right']


def labels_to_vector(words, labels=LABELS):
    return np.array([1 if label in words else 0 for label in labels], dtype=bool)


def motor_group_indices(drive, brain):
    """Every cell of the mapped motor groups, i.e. the command's own input set."""
    return sorted({brain.lookup[i] for group in drive['groups']
                   for entry in group['sides'].values() for i in entry['neuron_ids']})


class CommandDecoder:
    """Regresses the recorded steering command from blind internal activity."""

    command = True
    continuous = True

    def __init__(self, brain, catalogue, path=None):
        path = Path(path or OUT / 'malecns-command.npz')
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
            raise ValueError('Command readout graph, input mappings or blind mask is stale')
        if metadata.get('schema') != SCHEMA or metadata.get('mode') != 'command':
            raise ValueError('Not a command readout artifact')
        if metadata.get('readout_module_sha256') != sha256(Path(__file__)):
            raise ValueError('Command readout module changed after the artifact was fitted')
        # A direction word is only meaningful under the calibration it was fitted
        # with; a missing or changed probe withholds it instead of serving a stale sign.
        sign, calibration = turn_sign()
        if sign is None:
            raise ValueError('Command direction calibration unavailable: ' + str(calibration))
        if metadata.get('sign_convention') != calibration['convention']:
            raise ValueError('Command direction calibration changed since the artifact was fitted')
        labels, features = len(TARGETS), len(self.feature_indices)
        if (self.weights.shape != (labels, features) or self.intercept.shape != (labels,) or
                self.target_mean.shape != (labels,) or self.target_scale.shape != (labels,) or
                self.qualified.shape != (labels,) or self.qualified.dtype.kind != 'b' or
                not all(np.isfinite(a).all() for a in
                        (self.weights, self.intercept, self.target_mean, self.target_scale)) or
                np.any(self.target_scale <= 0)):
            raise ValueError('Invalid command readout artifact')
        durations = metadata.get('tested_durations_ms', [])
        if not durations or any(isinstance(x, bool) or not isinstance(x, (float, int)) or
                                not np.isfinite(x) or not 0 < x <= 5000 for x in durations):
            raise ValueError('Artifact needs explicit tested observation durations')
        served = metadata.get('served_directions')
        if (not isinstance(served, dict) or set(served) != set(LABELS)
                or not all(isinstance(value, bool) for value in served.values())):
            raise ValueError('Artifact must declare which command directions the validation split supports')
        self.served_directions = served
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
                'scope': SCOPE, 'globally_calibrated': False, 'biologically_calibrated': False,
                'label_source': self.metadata.get('label_source'),
                'sign_calibration': self.metadata.get('sign_calibration'),
                'served_directions': self.served_directions,
                'excluded_motor_cells': self.metadata.get('excluded_motor_cells'),
                'feature_exclusion': self.metadata.get('feature_rule')}
        if duration_ms not in self.metadata['tested_durations_ms']:
            note['reason'] = 'Outside validated observation durations; no extrapolated command'
            return {'words': [], 'prediction': None, 'status': 'uncalibrated', 'command_readout': note}
        if not bool(self.qualified[0]):
            note['reason'] = 'The steering command did not qualify on the validation split'
            return {'words': [], 'prediction': None, 'status': 'unqualified', 'command_readout': note}
        value = float(self.predict(counts, state_record, duration_ms)[0])
        has_signal = bool(np.any(np.asarray(counts)) or
                          np.any(np.asarray(state_record)[2:] > MIN_VOLTAGE_MV))
        words = []
        if has_signal and abs(value) >= self.rules['informative_abs_command']:
            label = 'command:turn-left' if value > 0 else 'command:turn-right'
            if self.served_directions.get(label):
                words.append(label)
            else:
                # The fit may point left, but this corpus never validated a left
                # command, so no direction word is spoken for it.
                note['withheld_direction'] = label
                note['reason'] = 'No validation window supports this direction; the word is withheld'
        note['applied'] = True
        return {'words': words, 'prediction': {'steer_command': value,
                                               'informative_abs_command': self.rules['informative_abs_command']},
                'descriptions': {w: LABEL_MEANING[w] for w in words},
                'status': 'decoded', 'score_type': 'Regression of the recorded steering command, not a probability',
                'scope': SCOPE, 'command_readout': note}


def load_command_decoder(brain, catalogue, path=None):
    """Absent or stale weights abstain; no direction is inferred from behaviour."""
    path = Path(path or OUT / 'malecns-command.npz')
    if not path.is_file():
        return None, {'applied': False, 'reason': 'Command readout has not been trained',
                      'globally_calibrated': False, 'biologically_calibrated': False}
    try:
        return CommandDecoder(brain, catalogue, path), None
    except (ValueError, KeyError, OSError, TypeError) as error:
        return None, {'applied': False, 'reason': 'Command readout unavailable: ' + str(error),
                      'globally_calibrated': False, 'biologically_calibrated': False}


def command_report(decoded, outcome=None):
    """Render the commanded direction as a labelled line, never as measured motion."""
    result = {'schema': 'chatflypt.command-expression.v1', 'source': 'blind_command_readout',
              'status': 'unavailable', 'text': None, 'labels': [], 'llm_called': False,
              'measured_outcome': outcome, 'prediction': None,
              'limitations': ['文字来自内部活动的有限命令读出：读出的是它自己下发的踏步驱动不对称。',
                              '不是实测转动，也不证明它有意转向；身体是否跟随由物理仿真单独记录。',
                              '读出特征已排除全部已映射运动群细胞，避免直接复述固定控制公式。']}
    if not isinstance(decoded, dict):
        return result
    result['status'] = decoded.get('status', 'unavailable')
    result['prediction'] = decoded.get('prediction')
    # Carry the qualification note into the served line so the record shows which
    # directions were allowed and whether one was withheld, not just the words.
    if decoded.get('command_readout') is not None:
        result['command_readout'] = decoded['command_readout']
    words = [w for w in decoded.get('words', []) if w in LABEL_MEANING]
    if result['status'] != 'decoded' or not words:
        return result
    if 'command:turn-left' in words and 'command:turn-right' in words:
        result['status'] = 'ambiguous'
        return result
    phrases = {'command:turn-left': '向左', 'command:turn-right': '向右'}
    result['labels'] = words
    # The qualifier rides with the sentence itself, not only with the limitations:
    # this line is a command readout, not evidence that the body turned.
    sentence = '它自己下发的转向指令（候选）：' + '、'.join(phrases[w] for w in words) + '（命令读出，未经身体行为验证）。'
    if isinstance(outcome, dict):
        sentence += f"同期实测偏航 {outcome['yaw_change_deg']:+.2f}°；该身体的转向跟随很弱，这一项不作为行为验证。"
    result['text'] = sentence
    return result
