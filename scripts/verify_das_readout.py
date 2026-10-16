"""Reference computation and release-recording evaluation, never a chat demo."""
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from chatflypt.acoustic_readout import AcousticReadout,CLASSES,RATE,WINDOW,LEFT_CONTEXT,RIGHT_CONTEXT
from chatflypt.paths import sha256,dump
from convert_das_model import numpy_reference


def metrics(actual,predicted):
    confusion=np.zeros((3,3),np.int64)
    np.add.at(confusion,(actual,predicted),1)
    per={}
    for i,name in enumerate(CLASSES):
        tp=confusion[i,i];precision=tp/max(1,confusion[:,i].sum());recall=tp/max(1,confusion[i].sum())
        per[name]={'precision':float(precision),'recall':float(recall),
            'f1':float(2*precision*recall/max(1e-30,precision+recall)),'samples':int(confusion[i].sum())}
    return {'confusion_matrix':confusion.tolist(),'row_and_column_classes':list(CLASSES),
            'classes':per,'accuracy':float(np.mean(actual==predicted)),
            'macro_f1':float(np.mean([c['f1'] for c in per.values()]))}


def main():
    from threadpoolctl import threadpool_limits
    source=ROOT/'research/outward-signals/das-pretrained/data.npz'
    z=np.load(source,allow_pickle=False);x=z['x'].ravel().astype(np.float32)
    assert list(z['class_names'])==list(CLASSES) and int(z['samplerate_Hz'])==RATE
    reader=AcousticReadout();start=time.perf_counter()
    batch=np.stack((x[50000:50000+WINDOW],x[60000:60000+WINDOW],np.zeros(WINDOW)))[...,None]
    with threadpool_limits(limits=2):reference=numpy_reference(batch)
    prediction=reader.predict_batch(batch)
    delta=float(abs(reference-prediction).max())
    assert delta<2e-4,delta
    print('NumPy/ONNX max error:',delta,flush=True)
    # Compare shifted full windows on their overlap after excluding true context.
    offset=173
    windows=np.stack((x[50000:50000+WINDOW],x[50000+offset:50000+offset+WINDOW]))[...,None]
    scores=reader.predict_batch(windows)
    a=scores[0,LEFT_CONTEXT+offset:WINDOW-RIGHT_CONTEXT]
    b=scores[1,LEFT_CONTEXT:WINDOW-RIGHT_CONTEXT-offset]
    shift_error=float(abs(a-b).max());assert shift_error<2e-4,shift_error
    print('Context/stitching max error:',shift_error,flush=True)
    scores=reader.predict(x,RATE);truth=z['y_true'].argmax(axis=1)
    valid=slice(LEFT_CONTEXT,len(x)-RIGHT_CONTEXT)
    report={'schema':'chatflypt.das-acceptance.v1','weights':'published, not retrained or tuned',
        'example_recording':str(source.relative_to(ROOT)),'example_sha256':sha256(source),
        'model_sha256':sha256(reader.path),'numeric_reference_max_absolute_error':delta,
        'shifted_window_max_absolute_error':shift_error,'numeric_reference':'independent NumPy graph implementation',
        'original_tensorflow_executed':False,'duration_s':len(x)/RATE,
        'context_samples':{'left':LEFT_CONTEXT,'right':RIGHT_CONTEXT},
        'model_training_overlap_with_example':'unknown; do not claim independent test',
        'metrics':metrics(truth[valid],scores[valid].argmax(axis=1)),
        'semantic_translation':False,'body_emitted_this_recording':False,'llm_called':False}
    noise=np.random.default_rng(45).normal(0,.01,WINDOW)
    controls=reader.predict_batch(np.stack((np.zeros(WINDOW),noise))[...,None])
    report['controls']={name:{'song_argmax_samples':int(np.sum(controls[i,LEFT_CONTEXT:WINDOW-RIGHT_CONTEXT].argmax(axis=1)>0)),
                             'tested_samples':WINDOW-LEFT_CONTEXT-RIGHT_CONTEXT}
                        for i,name in enumerate(('silence','white_noise'))}
    report['wall_seconds']=time.perf_counter()-start
    folder=ROOT/'reports/acoustic-readout'
    dump(folder/'das-acceptance.json',report)
    np.savez_compressed(folder/'das-release-predictions.npz',scores=scores,truth=truth,sample_rate_hz=RATE)
    # Generate ordinary-language signal descriptions, never first-person speech.
    dump(folder/'das-waveform-analysis.json',reader.analyze(x[50000:70000],RATE,source='external_DAS_release_recording_5_to_7_seconds'))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
