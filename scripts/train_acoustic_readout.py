"""Train/evaluate a bounded acoustic classifier on disjoint time intervals.

Author's pulse clip is [t-100,t+100); sine clip is [t,t+200).
No random clip split: temporal overlap across partitions is disallowed, including
all samples used in features. Missing labels are *unannotated* background.
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from acoustic_baseline import AcousticReadout,features,CLASSES,WINDOW,HOP,MODEL
from chatflypt.paths import dump,sha256

SOURCE=ROOT/'research/outward-signals/flies.npz'
EXPECTED_HASH='75aa388dc554a6cb24bd765f78ac808f410d723e2a8e1e5173737298b7dbc113'


def partition(x,pulses,sines,start,end):
    centers=np.arange(start+WINDOW//2,end-WINDOW//2+1,HOP,dtype=int)
    # Pulse centers use a +-5 ms positive core; sine annotations define 20 ms clips.
    labels=np.zeros(len(centers),int)
    for t in sines: labels[(centers>=t)&(centers<t+WINDOW)]=2
    for t in pulses: labels[abs(centers-t)<=50]=1
    return centers, labels, x[centers[:,None]+np.arange(-100,100)]


def metrics(actual,predicted):
    matrix=np.zeros((3,4),int)
    for a,b in zip(actual,predicted): matrix[a,b if b>=0 else 3]+=1
    per={}
    for i,name in enumerate(CLASSES):
        tp=matrix[i,i];precision=tp/max(1,matrix[:,i].sum());recall=tp/max(1,matrix[i].sum())
        per[name]={'precision':float(precision),'recall':float(recall),
            'f1':float(2*precision*recall/max(1e-30,precision+recall)),
            'support':int(matrix[i].sum())}
    return {'confusion_matrix':matrix.tolist(),'rows':list(CLASSES),
        'columns':list(CLASSES)+['abstain'],'classes':per,
        'accuracy':float(np.mean(actual==predicted)),
        'coverage':float(np.mean(predicted>=0)),'macro_f1':float(np.mean([v['f1'] for v in per.values()]))}


def export(forest,accept,metadata):
    trees=[e.tree_ for e in forest.estimators_];nodes=max(t.node_count for t in trees)
    shape=(len(trees),nodes)
    left=np.full(shape,-1,np.int32);right=left.copy();feature=left.copy()
    threshold=np.zeros(shape);value=np.zeros((*shape,3))
    for i,t in enumerate(trees):
        n=t.node_count;left[i,:n]=t.children_left;right[i,:n]=t.children_right
        feature[i,:n]=t.feature;threshold[i,:n]=t.threshold
        v=t.value[:,0];value[i,:n]=v/v.sum(axis=1,keepdims=True)
    MODEL.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(MODEL,left=left,right=right,feature=feature,threshold=threshold,
                        value=value,accept=accept,metadata=json.dumps(metadata))


def main():
    from sklearn.ensemble import RandomForestClassifier
    assert sha256(SOURCE)==EXPECTED_HASH
    z=np.load(SOURCE,allow_pickle=False);x=z['recording'].ravel();rate=int(z['samplerate'])
    assert rate==10000
    limits={'train':(0,600000),'validation':(600000,800000),'test':(800000,len(x))}
    sets={k:partition(x,z['pulsetimes'],z['sinetimes'],*v) for k,v in limits.items()}
    train,validation,test=[sets[k] for k in ('train','validation','test')]
    forest=RandomForestClassifier(n_estimators=96,max_depth=12,min_samples_leaf=3,
             class_weight='balanced_subsample',random_state=827,n_jobs=2)
    forest.fit(features(train[2]),train[1])
    meta={'schema':'chatflypt.acoustic-forest.v1','classes':list(CLASSES),'sample_rate_hz':rate,
          'window_samples':WINDOW,'hop_samples':HOP,'max_depth':12,'source_sha256':EXPECTED_HASH,
          'partitions_samples':limits,'semantic_decoder':False,'independent_animals':False}
    # Choose abstention thresholds from validation only. Never inspect test for tuning.
    vs=forest.predict_proba(features(validation[2]));best=vs.argmax(axis=1);thresholds=[]
    for i in range(3):
        choices=[]
        for t in np.arange(.50,.961,.02):
            keep=(best==i)&(vs[:,i]>=t);tp=np.sum(keep&(validation[1]==i))
            if keep.sum()>=10 and tp/keep.sum()>=.90:
                choices.append((int(tp),float(t)))
        thresholds.append(max(choices,key=lambda r:(r[0],-r[1]))[1] if choices else 1.01)
    export(forest,np.array(thresholds),meta);reader=AcousticReadout()
    report={'schema':'chatflypt.acoustic-acceptance.v1','source':str(SOURCE.relative_to(ROOT)),
        'source_sha256':EXPECTED_HASH,'model_sha256':sha256(MODEL),'threshold_selection':'validation precision >= 0.90 with at least 10 predictions; otherwise reject class',
        'acceptance_thresholds':dict(zip(CLASSES,thresholds)),'partition_seconds':{k:[a/rate,b/rate] for k,(a,b) in limits.items()},
        'overlapping_samples_between_partitions':0,'random_clip_split':False,'independent_animal_validation':False,
        'label_scope':'pulse +/-5ms core, sine author 20ms intervals; all other centers unannotated background',
        'semantic_translation':False,'llm_called':False,'metrics':{},'arrays':{}}
    for name,(centers,labels,clips) in sets.items():
        scores,predicted=reader.classify(clips)
        reference=forest.predict_proba(features(clips))
        error=float(abs(scores-reference).max())
        assert error<1e-12
        report['metrics'][name]=metrics(labels,predicted)
        report['metrics'][name]['export_max_score_error']=error
        report['arrays'][name]={'first_window_start':int(centers[0]-100),'last_window_end_exclusive':int(centers[-1]+100),'frames':len(centers)}
    silent=np.zeros((5,WINDOW));_,silent_labels=reader.classify(silent)
    rng=np.random.default_rng(45);_,noise_labels=reader.classify(rng.normal(0,.01,(200,WINDOW)))
    report['controls']={'silent_song_false_positives':int(np.sum(silent_labels>0)),
        'white_noise_song_false_positives':int(np.sum(noise_labels>0)),'noise_windows':200}
    folder=ROOT/'reports/acoustic-readout';dump(folder/'acceptance.json',report)
    analysis=reader.analyze(x[limits['test'][0]:],rate,source='external_DAS_recording_test_partition')
    analysis['recording_offset_ms']=limits['test'][0]/10
    dump(folder/'blind-waveform-test.json',analysis)
    np.savez_compressed(folder/'heldout-predictions.npz',centers=test[0],labels=test[1],scores=reader.scores(test[2]),predicted=reader.classify(test[2])[1])
    print(json.dumps({'thresholds':thresholds,'validation':report['metrics']['validation'],
                      'test':report['metrics']['test'],'controls':report['controls']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
