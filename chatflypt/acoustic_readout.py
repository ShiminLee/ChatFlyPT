"""Recorded waveform -> published DAS acoustic classes. No semantic decoding.

Loading is lazy and numeric (ONNX). Incoming sensations, human text and motor
commands are not arguments to this component and cannot stand in for sound.
"""
import json
from pathlib import Path
import numpy as np
from .paths import ROOT,sha256

MODEL=ROOT/'data/derived/acoustic-readout/das-model.onnx'
RATE=10000
CLASSES=('noise','sine','pulse')
WINDOW=4096
# Longest path: 3 * (1+2+4+8+16) convolutions of size 32.
# SAME padding is asymmetric for dilation=1: total left=1440, right=1443.
LEFT_CONTEXT=1440
RIGHT_CONTEXT=1443
STRIDE=WINDOW-LEFT_CONTEXT-RIGHT_CONTEXT


def validate_waveform(samples,sample_rate):
    if isinstance(sample_rate,bool) or sample_rate!=RATE:raise ValueError('DAS requires 10000 Hz; resample explicitly')
    x=np.asarray(samples,dtype=np.float32)
    if x.ndim!=1 or not 1<=len(x)<=600*RATE or not np.isfinite(x).all():
        raise ValueError('Expected finite mono waveform, 1 sample to 600 seconds')
    return x


class AcousticReadout:
    def __init__(self,path=MODEL):
        import onnxruntime as ort
        self.path=Path(path);self.receipt=json.loads(self.path.with_suffix('.json').read_text(encoding='utf8'))
        if sha256(self.path)!=self.receipt['onnx_sha256']:raise ValueError('Acoustic model hash mismatch')
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(self.path),sess_options=options,providers=['CPUExecutionProvider'])

    def predict_batch(self,batch):
        batch=np.asarray(batch,dtype=np.float32)
        if batch.ndim!=3 or batch.shape[1:]!=(WINDOW,1) or not np.isfinite(batch).all():
            raise ValueError('DAS batches must have shape (batch, 4096, 1) and finite values')
        result=self.session.run(None,{'input_1':batch})[0]
        if not np.isfinite(result).all():raise ValueError('Acoustic inference produced non-finite scores')
        return result

    def predict(self,samples,sample_rate):
        x=validate_waveform(samples,sample_rate)
        padded=np.pad(x,(LEFT_CONTEXT,RIGHT_CONTEXT+WINDOW))
        score=np.empty((len(x),3),np.float32)
        for start in range(0,len(x),STRIDE):
            prediction=self.predict_batch(padded[start:start+WINDOW][None,:,None])[0]
            count=min(STRIDE,len(x)-start)
            score[start:start+count]=prediction[LEFT_CONTEXT:LEFT_CONTEXT+count]
        return score

    def analyze(self,samples,sample_rate,*,source='unverified_external_recording'):
        x=validate_waveform(samples,sample_rate);scores=self.predict(x,sample_rate)
        labels=scores.argmax(axis=1);labels[scores.max(axis=1)<.5]=-1
        labels[:min(LEFT_CONTEXT,len(labels))]=-1
        labels[max(0,len(labels)-RIGHT_CONTEXT):]=-1
        boundaries=np.r_[0,np.flatnonzero(np.diff(labels))+1,len(labels)];intervals=[]
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            label=int(labels[a])
            if label<=0:continue
            intervals.append({'class':CLASSES[label],
                'description_zh':'检测到与果蝇'+('持续鸣声' if label==1 else '脉冲鸣声')+'模型相符的声学片段。',
                'start_ms':a/10,'end_ms':b/10,
                'mean_score':float(scores[a:b,label].mean()),'minimum_score':float(scores[a:b,label].min())})
        return {'schema':'chatflypt.acoustic-readout.v2','status':'acoustic_classes_only',
            'model':'DAS Dmel_single_v1.0 original weights','source':source,
            'sample_rate_hz':RATE,'duration_ms':len(x)/10,'class_names':list(CLASSES),
            'sample_counts':{name:int(np.sum(labels==i)) for i,name in enumerate(CLASSES)},
            'abstained_samples':int(np.sum(labels<0)),'intervals':intervals,
            'lookahead_ms':RIGHT_CONTEXT/10,'left_context_ms':LEFT_CONTEXT/10,
            'model_sha256':self.receipt['onnx_sha256'],'llm_called':False,
            'semantics_decoded':False,'expression_messages':[],
            'limitations':['Classifies noise/sine/pulse acoustic patterns; no utterance meanings.',
                'External recording classification does not establish a signal emitted by this simulation.',
                'Release example evaluation has unknown training overlap; not an independent-animal test.',
                'Noncausal model requires 144.3 ms future context; global edges are unclassified.',
                'Current body has no calibrated acoustic emitter; chat cannot use these external recordings as its voice.']}


def acoustic_capability():
    return {'decoder':'DAS Dmel_single_v1.0','installed':MODEL.is_file(),
        'scope':'acoustic_classification_only','semantic_decoder':False,
        'body_signal_producer':False,'validation_report':'reports/acoustic-readout/das-acceptance.json'}
