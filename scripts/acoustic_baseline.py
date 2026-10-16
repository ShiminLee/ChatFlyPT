"""Waveform-only, bounded acoustic classification; never an utterance decoder.

The saved forest contains numeric arrays, not executable/pickled model objects.
Inference needs neither annotations, neural input labels nor language models.
The training recording is one fly-song recording; scores are NOT calibrated
probabilities or evidence of generalization to a simulated acoustic emitter.
"""
from pathlib import Path
import json
import numpy as np

from chatflypt.paths import ROOT, sha256

MODEL = ROOT / 'data/derived/acoustic-readout/model.npz'
SAMPLE_RATE = 10000
WINDOW = 200
HOP = 50
CLASSES = ('unannotated_background', 'pulse', 'sine')
NAMES = ('未标注背景', '脉冲鸣声候选', '持续鸣声候选')


def waveform(value, sample_rate):
    if sample_rate != SAMPLE_RATE or isinstance(sample_rate, bool):
        raise ValueError('Acoustic readout requires 10000 Hz; resample explicitly before analysis')
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 1 or len(x) < WINDOW or len(x) > SAMPLE_RATE * 600:
        raise ValueError('Supply a mono waveform between 20 ms and 600 s')
    if not np.isfinite(x).all():
        raise ValueError('Waveform must be finite')
    return x


def features(clips):
    """20 ms spectral shape, temporal energy and waveform statistics."""
    x = np.asarray(clips, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != WINDOW or not np.isfinite(x).all():
        raise ValueError('Expected finite 200-sample clips')
    x = x - x.mean(axis=1, keepdims=True)
    energy = np.mean(x*x, axis=1)
    norm = x / np.sqrt(energy[:, None] + 1e-30)
    power = abs(np.fft.rfft(norm * np.hanning(WINDOW), n=512))**2
    power /= power.sum(axis=1, keepdims=True) + 1e-30
    # Entire spectrum retained in bins; energy scale is a separate feature.
    spectrum = np.stack([power[:, a:b].sum(axis=1) for a,b in
                         zip(range(0,256,4),range(4,260,4))], axis=1)
    temporal = np.mean(norm.reshape(-1, 8, 25)**2, axis=2)
    stats = np.column_stack((np.log10(energy+1e-30), np.mean(norm**4,axis=1),
                             abs(norm).max(axis=1), np.mean(np.diff(np.signbit(x),axis=1),axis=1)))
    return np.column_stack((spectrum,temporal,stats)).astype(np.float32)


class AcousticReadout:
    def __init__(self, path=MODEL):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as model:
            self.left = model['left']; self.right = model['right']
            self.feature = model['feature']; self.threshold = model['threshold']
            self.value = model['value']; self.accept = model['accept']
            self.metadata = json.loads(str(model['metadata']))
        if (self.metadata['classes'] != list(CLASSES) or
            self.metadata['sample_rate_hz'] != SAMPLE_RATE or self.metadata['window_samples'] != WINDOW):
            raise ValueError('Unsupported acoustic model contract')

    def scores(self, clips):
        x = features(clips)
        score = np.zeros((len(x),len(CLASSES)))
        for left,right,feature,threshold,value in zip(self.left,self.right,self.feature,self.threshold,self.value):
            nodes = np.zeros(len(x),dtype=np.int32)
            # Exported trees have a fixed finite maximum depth.
            for _ in range(self.metadata['max_depth']+1):
                active = left[nodes] >= 0
                if not active.any(): break
                rows = np.flatnonzero(active); current = nodes[active]
                go_left = x[rows,feature[current]] <= threshold[current]
                nodes[active] = np.where(go_left,left[current],right[current])
            score += value[nodes]
        return score / len(self.left)

    def classify(self, clips):
        score = self.scores(clips); labels = score.argmax(axis=1)
        accepted = score[np.arange(len(score)),labels] >= self.accept[labels]
        # A background classification is not proof of acoustic silence.
        return score, np.where(accepted, labels, -1)

    def analyze(self, samples, sample_rate, *, source='unverified_recording'):
        x = waveform(samples,sample_rate)
        starts = np.arange(0,len(x)-WINDOW+1,HOP)
        scores=[]; classes=[]
        for block in np.array_split(starts,max(1,int(np.ceil(len(starts)/2048)))):
            clips = x[block[:,None]+np.arange(WINDOW)]
            score, label = self.classify(clips)
            scores.append(score); classes.append(label)
        scores=np.concatenate(scores); classes=np.concatenate(classes)
        intervals=[]
        # Intervals describe the supported center/hop bins, not exact event bounds.
        changes=np.r_[0,np.flatnonzero(np.diff(classes))+1,len(classes)]
        for begin,end in zip(changes[:-1],changes[1:]):
            label=int(classes[begin])
            if label <= 0: continue
            intervals.append({'class':CLASSES[label],'label':NAMES[label],
                'start_ms':float((starts[begin]+WINDOW//2-HOP//2)/10),
                'end_ms':float((starts[end-1]+WINDOW//2+HOP//2)/10),
                'minimum_score':float(scores[begin:end,label].min()),
                'mean_score':float(scores[begin:end,label].mean())})
        return {'schema':'chatflypt.acoustic-readout.v1','status':'research_candidate',
            'source':source,'sample_rate_hz':SAMPLE_RATE,'duration_ms':len(x)/10,
            'analysis_window_ms':WINDOW/10,'hop_ms':HOP/10,'lookahead_ms':WINDOW/20,
            'unscored_edge_ms_each':(WINDOW-HOP)/20,
            'model_sha256':sha256(self.path),'class_names':list(CLASSES),
            'frame_counts':{name:int(np.sum(classes==i)) for i,name in enumerate(CLASSES)},
            'abstained_frames':int(np.sum(classes==-1)),'intervals':intervals,
            'llm_called':False,'semantics_decoded':False,'expression_messages':[],
            'limitations':['Acoustic classes only; no intentions, sentence meanings or subjective state.',
                'One recording; no independent animal/microphone or simulation validation.',
                'Random-forest vote scores are not calibrated probabilities.',
                'Missing annotations are background for this benchmark, not verified silence.',
                'Overlapping analysis windows do not define individual pulse events.',
                'No acoustic emitter is implemented in the current body model.']}


def acoustic_capability():
    """Do not load a model or claim a measurement merely because it is installed."""
    return {'decoder':'waveform_pulse_sine_classifier','installed':MODEL.is_file(),
            'scope':'research_acoustic_classes','semantic_decoder':False,
            'body_signal_producer':False,'validation_report':'reports/acoustic-readout/acceptance.json'}
