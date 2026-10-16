"""Candidate readout adaptation for measured 500 ms feedback windows.

Inference consumes only recorded neural activity. Session/environment labels,
targets, input events and user text are confined to the offline benchmark.
Passing its small validation set never makes a label biologically calibrated.
"""
from pathlib import Path
import numpy as np

from .paths import ROOT, sha256
from .state_readout import MIN_VOLTAGE_MV

OUT=ROOT/'data/derived/feedback-readout'


def load_feedback_decoder(base,dataset):
    """Fall back explicitly if an optional saved adapter is absent or stale."""
    from .sensory_readout import dataset_key
    from .state_readout import OUT as BASE_OUT
    if not hasattr(base,'feature_indices'):return base,None
    key=dataset_key(dataset);path=OUT/f'{key}-adapter.npz'
    disabled=OUT/f'{key}-disabled.json'
    if disabled.is_file():
        return base,{'applied':False,'reason':'Feedback adapter disabled after failed independent negative controls.',
                     'disable_record':str(disabled),'globally_calibrated':False,'biologically_calibrated':False}
    if not path.exists():return base,None
    try:return FeedbackDecoder(base,path,BASE_OUT/f'{key}-decoder.npz'),None
    except (ValueError,KeyError,OSError) as error:
        return base,{'applied':False,'reason':'Saved feedback adapter unavailable: '+str(error),
                     'globally_calibrated':False,'biologically_calibrated':False}


def normalized_features(counts,state,duration_ms):
    if not np.isfinite(duration_ms) or duration_ms<=0:
        raise ValueError('Positive finite observation duration required')
    counts=np.asarray(counts);state=np.asarray(state)
    if counts.ndim!=1 or state.shape!=(4,len(counts)):
        raise ValueError('Recorded counts and four membrane statistics required')
    scale=500./duration_ms
    return np.concatenate((np.sqrt(counts*scale),np.log1p(state[0]*scale),np.log1p(state[1]*scale)))


class FeedbackDecoder:
    def __init__(self,base,path,base_path):
        self.base=base;self.indices=base.indices
        with np.load(path,allow_pickle=False) as saved:
            for name in ('feature_indices','weights','intercept','thresholds','enabled'):
                setattr(self,name,saved[name])
            self.labels=saved['labels'].tolist()
            self.base_sha256=str(saved['base_sha256'])
            self.protocol_sha256=str(saved['protocol_sha256'])
            self.catalogue_signature=str(saved['catalogue_signature'])
        if sha256(base_path)!=self.base_sha256 or base.catalogue_signature!=self.catalogue_signature:
            raise ValueError('Feedback adapter belongs to a different base model or sensory catalogue')
        if not np.array_equal(self.feature_indices,base.feature_indices):
            raise ValueError('Feedback adapter must use the fixed blind internal feature mask')
        if not set(self.labels)<=set(base.labels):
            raise ValueError('Adapter cannot invent labels outside the base readout')
        if (len(set(self.labels))!=len(self.labels) or
                self.weights.shape!=(len(self.labels),len(self.feature_indices)) or
                any(a.shape!=(len(self.labels),) for a in (self.intercept,self.thresholds,self.enabled)) or
                self.enabled.dtype.kind!='b' or
                not all(np.isfinite(a).all() for a in (self.weights,self.intercept,self.thresholds))):
            raise ValueError('Invalid feedback adapter parameters')
        self.artifact_sha256=sha256(path)
        self.artifact=str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else str(path)

    def decode_trial(self,trial):
        decoded=self.base.decode_trial(trial)
        # This experiment contains 500 ms windows only. Other durations retain
        # the existing explicitly uncalibrated feedback prediction.
        if trial.duration_ms!=500.:
            decoded['feedback_adaptation']={'applied':False,'reason':'Outside the tested 500 ms observation window'}
            return decoded
        x=normalized_features(trial.counts,trial.state_record,trial.duration_ms)[self.feature_indices]
        values=self.weights@x+self.intercept
        has_signal=bool(np.any(trial.counts[self.indices]) or np.any(trial.state_record[2:,self.indices]>MIN_VOLTAGE_MV))
        adapted={label:bool(score>=threshold and has_signal) for label,score,threshold,use in
                 zip(self.labels,values,self.thresholds,self.enabled) if use}
        words=set(decoded['words'])-set(adapted)
        words.update(label for label,positive in adapted.items() if positive)
        decoded['words']=[w for w in self.base.labels if w in words]
        decoded['descriptions']={w:self.base.names[w] for w in decoded['words']}
        decoded['unqualified_words']=[w for w in decoded.get('unqualified_words',[]) if w not in adapted]
        decoded['scores'].update({label:float(v) for label,v,use in zip(self.labels,values,self.enabled) if use})
        decoded['feedback_adaptation']={'applied':bool(adapted),'labels':list(adapted),
            'artifact':self.artifact,'sha256':self.artifact_sha256,'protocol_sha256':self.protocol_sha256,
            'scope':'Small session-separated simulation benchmark; still candidate neural readout',
            'globally_calibrated':False,'biologically_calibrated':False}
        return decoded
