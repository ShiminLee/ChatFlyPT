"""Two extra blind slots that let the fly's speech be composed, not templated.

The chat's words are still read out of internal activity only. These decoders add
two slot families to the single-channel sensations, so an utterance can be
*composed* for combinations that never appeared as a fixed sentence:

* ``pattern``  - was delivery in this window a brief transient or a sustained
                 presence? Ground truth is measured: the fraction of 10 ms
                 samples in which any receptor event was actually delivered.
* ``activity`` - how busy was the internal state? Ground truth is measured: the
                 number of non-sensory spikes in the window.

Both bands come from the *training split's* quantiles and are preregistered before
validation or test windows are labelled. A slot that fails qualification abstains;
nothing is spoken that was not decoded and qualified. The renderer that consumes
these slots is in `generative_renderer.py`; grounding is auditable there.
"""
import json
from pathlib import Path

import numpy as np

from .paths import ROOT, sha256
from .unified_readout import (DATASET, blind_mask, internal_features, model_contract,
                              MIN_VOLTAGE_MV)

OUT = ROOT / 'data/derived/slot-readout'
SCHEMA = 'chatflypt.slot-readout.v1'
FEATURES = 'sqrt spikes; log1p positive/negative voltage area; normalized to 500 ms'

# Slot definitions. Bands are quantiles of the measured training distribution.
PATTERN_BANDS = {'transient_max_fraction': .3}
ACTIVITY_QUANTILES = (.33, .66)
PATTERN_LABELS = ('pattern:transient', 'pattern:sustained')
ACTIVITY_LABELS = ('activity:quiet', 'activity:moderate', 'activity:busy')
LABEL_MEANING = {
    'pattern:transient': '受体事件只在窗口的一小部分时间里出现（实测投递样本占比低于训练分位阈值）',
    'pattern:sustained': '受体事件几乎贯穿整个窗口（实测投递样本占比不低于训练分位阈值）',
    'activity:quiet': '窗口内非感觉脉冲数低于训练集下分位',
    'activity:moderate': '窗口内非感觉脉冲数处于训练集中间三分位',
    'activity:busy': '窗口内非感觉脉冲数高于训练集上分位',
}


def slot_rules(training_windows):
    """Fix band thresholds from the training split alone."""
    fractions = np.array([w['delivery_fraction'] for w in training_windows])
    spikes = np.array([w['nonsensory_spikes'] for w in training_windows])
    return {'pattern_transient_max_fraction': float(PATTERN_BANDS['transient_max_fraction']),
            'activity_spike_quantiles': [float(np.quantile(spikes, q)) for q in ACTIVITY_QUANTILES],
            'training_windows': int(len(training_windows)),
            'rule': 'Transient/sustained split is fixed at 0.3 of sampled windows; activity bands are the '
                    'training split\'s 33rd and 66th percentiles of measured non-sensory spikes.'}


def pattern_labels(fraction, rules):
    return ['pattern:sustained' if fraction >= rules['pattern_transient_max_fraction']
            else 'pattern:transient']


def activity_labels(spikes, rules):
    low, high = rules['activity_spike_quantiles']
    if spikes < low:
        return ['activity:quiet']
    if spikes >= high:
        return ['activity:busy']
    return ['activity:moderate']


class SlotDecoder:
    """One-versus-rest linear decoders over the shared blind internal features."""

    slots = True

    def __init__(self, brain, catalogue, path=None):
        path = Path(path or OUT / 'malecns-slots.npz')
        self.contract = model_contract(brain, catalogue)
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            for key in ('pattern_feature_indices', 'activity_feature_indices', 'indices',
                        'pattern_weights', 'pattern_intercept', 'pattern_thresholds', 'pattern_qualified',
                        'activity_weights', 'activity_intercept', 'activity_thresholds', 'activity_qualified'):
                setattr(self, key, saved[key])
            self.rules = json.loads(str(saved['rules']))
        if metadata.get('contract') != self.contract:
            raise ValueError('Slot readout graph, input mappings or blind mask is stale')
        if metadata.get('schema') != SCHEMA:
            raise ValueError('Not a slot artifact')
        self.neurons = len(brain.ids)
        self.metadata = metadata
        self.artifact_sha256 = sha256(path)
        self.artifact = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    def decode(self, counts, state_record, duration_ms=500.):
        note = {'applied': False, 'artifact': self.artifact, 'sha256': self.artifact_sha256,
                'scope': 'Candidate blind slot readout; measured delivery pattern and measured internal activity level',
                'globally_calibrated': False, 'biologically_calibrated': False}
        if duration_ms not in self.metadata.get('tested_durations_ms', []):
            note['reason'] = 'Outside validated observation durations; no extrapolated slots'
            return {'words': [], 'status': 'uncalibrated', 'slots': note}
        values = internal_features(counts, state_record, duration_ms)
        has_signal = bool(np.any(np.asarray(counts)) or
                          np.any(np.asarray(state_record)[2:] > MIN_VOLTAGE_MV))
        words = []
        if has_signal:
            pattern = self.pattern_weights @ values[self.pattern_feature_indices] + self.pattern_intercept
            hits = [label for label, score, threshold, ok in zip(
                PATTERN_LABELS, pattern, self.pattern_thresholds, self.pattern_qualified)
                if ok and score >= threshold]
            if len(hits) == 1:
                words.extend(hits)
            activity = self.activity_weights @ values[self.activity_feature_indices] + self.activity_intercept
            hits = [label for label, score, threshold, ok in zip(
                ACTIVITY_LABELS, activity, self.activity_thresholds, self.activity_qualified)
                if ok and score >= threshold]
            if len(hits) == 1:
                words.extend(hits)
        note['applied'] = bool(self.pattern_qualified.any() or self.activity_qualified.any())
        return {'words': words,
                'descriptions': {w: LABEL_MEANING[w] for w in words},
                'status': 'decoded', 'score_type': 'Linear neural-state scores, not probabilities',
                'slot_qualification': {'pattern': self.pattern_qualified.tolist(),
                                       'activity': self.activity_qualified.tolist()},
                'scope': 'Finite candidate slot readout; not subjective state or biological calibration',
                'slots': note}


def load_slot_decoders(brain, catalogue, path=None):
    """Absent or stale weights abstain rather than falling back."""
    path = Path(path or OUT / 'malecns-slots.npz')
    if not path.is_file():
        return None, {'applied': False, 'reason': 'Slot readout has not been trained',
                      'globally_calibrated': False, 'biologically_calibrated': False}
    try:
        return SlotDecoder(brain, catalogue, path), None
    except (ValueError, KeyError, OSError, TypeError) as error:
        return None, {'applied': False, 'reason': 'Slot readout unavailable: ' + str(error),
                      'globally_calibrated': False, 'biologically_calibrated': False}
