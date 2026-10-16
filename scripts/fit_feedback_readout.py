"""Fit on training sessions, select thresholds on validation, then open test."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.linear_model import Ridge
from chatflypt.paths import ROOT,dump,sha256
from chatflypt.state_readout import StateDecoder,OUT as BASE_OUT,MIN_VOLTAGE_MV
from chatflypt.feedback_readout import normalized_features,OUT
from chatflypt.feedback_readout import FeedbackDecoder
from chatflypt.brain import Trial

REPORT=ROOT/'reports/feedback-readout'
TARGETS={'flywire':['bright:left','bright:right','yeast:left','yeast:right'],
         'malecns':['foreleg_touch:left','foreleg_touch:right']}


def measured_targets(row,catalogue,event_ids):
    """Evaluation target is actual delivery, never what the user hoped to do."""
    delivered=set(map(str,event_ids));labels=set()
    for item in row['human_items']+row['environment_items']:
        if item.get('intensity',1)<=0:continue
        spec=catalogue['channels'][item['channel']]
        if spec['dataset']!=row['dataset']:continue
        side_map=spec.get('body_part_ids',{}).get(item.get('body_part'),spec['side_ids'])
        sides=('left','right') if item.get('side','bilateral')=='bilateral' else (item['side'],)
        for side in sides:
            if delivered.intersection(map(str,side_map.get(side,[]))):
                labels.add(spec.get('readout_group',item['channel'])+':'+side)
    return labels


def metric(truth,predicted,labels):
    truth=np.asarray(truth,bool);predicted=np.asarray(predicted,bool)
    tp=(truth&predicted).sum(0);fp=(~truth&predicted).sum(0);fn=(truth&~predicted).sum(0)
    return {'trials':len(truth),'exact_set_accuracy':float(np.all(truth==predicted,axis=1).mean()),
        'micro_f1':float(2*tp.sum()/max(1,2*tp.sum()+fp.sum()+fn.sum())),
        'per_label':{w:{'tp':int(t),'fp':int(f),'fn':int(n),'positives':int(y.sum()),
            'negatives':int((~y).sum()),'precision':float(t/max(1,t+f)),
            'recall':float(t/max(1,t+n)),'f1':float(2*t/max(1,2*t+f+n))}
            for w,t,f,n,y in zip(labels,tp,fp,fn,truth.T)}}


def fit(key,catalogue,plan):
    dataset={'flywire':'FlyWire v783','malecns':'MaleCNS v1.0'}[key]
    base=StateDecoder(dataset);labels=TARGETS[key]
    rows=[]
    for episode in plan['episodes']:
        if episode['dataset']!=dataset:continue
        complete=OUT/episode['id']/'complete.json'
        data=json.loads(complete.read_text(encoding='utf8'))
        if data['protocol_sha256']!=sha256(REPORT/'protocol.json'):raise ValueError('Protocol mismatch')
        rows.extend(data['records'])
    groups={split:{r['episode'] for r in rows if r['split']==split} for split in ('train','validation','test')}
    assert not groups['train']&groups['validation'] and not groups['train']&groups['test'] and not groups['validation']&groups['test']

    def load(split):
        vectors=[];truth=[];baseline=[];signal=[];selected=[];outside=[]
        for row in rows:
            if row['split']!=split:continue
            path=ROOT/row['record'];events=ROOT/row['events']
            assert sha256(path)==row['record_sha256'] and sha256(events)==row['events_sha256']
            with np.load(events,allow_pickle=False) as raw:expected=measured_targets(row,catalogue,raw['event_neuron_ids'])
            with np.load(path,allow_pickle=False) as raw:
                counts=raw['counts'];state=raw['state_record']
                np.testing.assert_array_equal(np.bincount(raw['spike_indices'],minlength=len(counts)),counts)
                assert not np.intersect1d(base.indices,raw['driven_indices']).size
                vectors.append(normalized_features(counts,state,float(raw['duration_ms']))[base.feature_indices])
                original=base.decode(counts,state)['words']
                baseline.append([w in original for w in labels])
                outside.append(sorted(set(original)-set(labels)))
                signal.append(bool(np.any(counts[base.indices]) or np.any(state[2:,base.indices]>MIN_VOLTAGE_MV)))
            truth.append([w in expected for w in labels]);selected.append(row)
        return np.asarray(vectors),np.asarray(truth,bool),np.asarray(baseline,bool),np.asarray(signal),selected,outside

    xt,yt,_,_,_,_=load('train')
    xv,yv,bv,sv,_,_=load('validation')
    model=Ridge(alpha=10,solver='cholesky').fit(xt,yt.astype(float))
    values=model.predict(xv);thresholds=[];enabled=[];before=metric(yv,bv,labels)
    for k,label in enumerate(labels):
        candidates=[]
        for threshold in np.linspace(.2,.8,61):
            pred=(values[:,k]>=threshold)&sv
            scores=metric(yv[:,k,None],pred[:,None],[label])['per_label'][label]
            supported=scores['positives']>=2 and scores['negatives']>=2
            passes=supported and scores['precision']>=.9 and scores['recall']>=.75
            candidates.append((passes,scores['f1'],-abs(threshold-.5),threshold))
        best=max(candidates);thresholds.append(best[-1])
        enabled.append(bool(best[0] and best[1]>=before['per_label'][label]['f1']))
    thresholds=np.asarray(thresholds);enabled=np.asarray(enabled,bool)
    path=OUT/f'{key}-adapter.npz'
    np.savez_compressed(path,feature_indices=base.feature_indices,weights=model.coef_,intercept=model.intercept_,
        thresholds=thresholds,enabled=enabled,labels=labels,base_sha256=sha256(BASE_OUT/f'{key}-decoder.npz'),
        protocol_sha256=sha256(REPORT/'protocol.json'),catalogue_signature=base.catalogue_signature)
    # Artifact and thresholds are fixed before any held-out neural features or
    # targets are loaded. Test failures are reported, never retuned here.
    locked_hash=sha256(path)
    xe,ye,be,se,test_rows,outside=load('test')
    predicted=(model.predict(xe)>=thresholds)&se[:,None]
    served=np.where(enabled,predicted,be)
    adapted_validation=np.where(enabled,(values>=thresholds)&sv[:,None],bv)
    reader=FeedbackDecoder(base,path,BASE_OUT/f'{key}-decoder.npz')
    with np.load(ROOT/rows[0]['record'],allow_pickle=False) as raw:n=len(raw['counts'])
    zero=reader.decode_trial(Trial(np.zeros(n,np.int32),np.empty(0,np.int32),np.empty(0,np.int32),0,500.,np.zeros((4,n))))
    assert not zero['words']
    # Check the actual serving class, not only the training matrix calculation.
    for row,expected in zip(test_rows,served):
        with np.load(ROOT/row['record'],allow_pickle=False) as raw:
            trial=Trial(raw['counts'],raw['spike_indices'],raw['spike_steps'],row['seed'],float(raw['duration_ms']),raw['state_record'])
            output=reader.decode_trial(trial)
            np.testing.assert_array_equal([w in output['words'] for w in labels],expected)
    report={'dataset':dataset,'protocol_sha256':sha256(REPORT/'protocol.json'),
        'artifact':str(path.relative_to(ROOT)),'artifact_sha256':locked_hash,
        'model_selection':'Training sessions only; validation thresholds and label selection locked before loading test',
        'train_trials':len(yt),'validation_trials':len(yv),'test_trials':len(ye),'groups':{k:sorted(v) for k,v in groups.items()},
        'enabled_labels':[w for w,e in zip(labels,enabled) if e],'candidate_only':True,
        'baseline_validation':before,'adapted_validation':metric(yv,adapted_validation,labels),
        'baseline_test':metric(ye,be,labels),'adapted_test':metric(ye,served,labels),
        'shuffled_test':metric(ye,served[np.random.default_rng(4101).permutation(len(ye))],labels),
        'zero_feature_false_positives':len(zero['words']),
        'serving_class_matches_test_predictions':True,
        'unscored_baseline_predictions':outside,
        'test_predictions':[{'episode':r['episode'],'turn':r['turn_index'],'record':r['record'],
            'expected':[w for w,y in zip(labels,truth) if y],'baseline':[w for w,y in zip(labels,old) if y],
            'adapted':[w for w,y in zip(labels,new) if y]}
            for r,truth,old,new in zip(test_rows,ye,be,served)],
        'limitations':['Small synthetic simulation benchmark, not biological calibration.',
            'Visual and odor scenes mainly co-activate both sides; lateral discrimination is not established.',
            'Only 500 ms windows and the named channels are adapted; unrelated baseline labels are not evaluated here.',
            'Current-window input identity is the target; residual activity is not a decoded memory or intent.']}
    assert sha256(path)==locked_hash
    dump(REPORT/f'{key}-evaluation.json',report)
    print(json.dumps({k:report[k] for k in ('dataset','train_trials','validation_trials','test_trials','enabled_labels','baseline_test','adapted_test')},ensure_ascii=False),flush=True)


def main():
    from chatflypt.sensory_catalogue import build_catalogue
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['all','flywire','malecns'],default='all')
    args=parser.parse_args()
    plan=json.loads((REPORT/'protocol.json').read_text(encoding='utf8'))
    # Separate fitting protocol explicitly fixes labels and criteria before data access.
    dump(REPORT/'fit-protocol.json',{'labels':TARGETS,'alpha':10,'threshold_grid':[.2,.8,.01],
        'minimum_validation_positives':2,'minimum_validation_negatives':2,'precision':.9,'recall':.75,
        'selection':'validation F1 at least baseline; test never tunes model','candidate_only':True})
    catalogue=build_catalogue(save=False)
    with threadpool_limits(limits=1):
        for key in (TARGETS if args.dataset=='all' else [args.dataset]):fit(key,catalogue,plan)


if __name__=='__main__':main()
