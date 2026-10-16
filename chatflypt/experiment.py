"""Generate a locked protocol, train blind linear probes, and evaluate examples."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import time
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from .brain import Brain
from .stimuli import WORDS, FIRST_SENTENCE, build_manifest, groups_for, notebook_literal
from .paths import ROOT, ARTIFACTS, REPORTS, dump, sha256


def protocol():
    rng=np.random.default_rng(20260919)
    cases=[]
    seed=10000
    def add(split,words,scale,kind):
        nonlocal seed
        cases.append({'id':f'trial-{seed}','split':split,'words':sorted(words),
                      'scale':float(scale),'seed':seed,'duration_ms':300,'kind':kind})
        seed+=1
    # Single-channel repeat splits measure independent Poisson realizations.
    for split,repeats in [('train',12),('validation',3),('test',6)]:
        for w in WORDS:
            for j in range(repeats):
                add(split,[w],rng.uniform(.85,1.15),'single')
        for j in range(8):
            add(split,[],1,'silence')
    # No mixture configuration appears in more than one split, or in the demo.
    seen={tuple(sorted(FIRST_SENTENCE))}
    for split,total in [('train',224),('validation',64),('test',64),('ood',64)]:
        for j in range(total):
            while True:
                combination=tuple(sorted(rng.choice(WORDS,int(rng.integers(2,7)),replace=False)))
                if combination not in seen:
                    seen.add(combination)
                    break
            scale=rng.uniform(.85,1.15) if split!='ood' else [.6,1.4][j%2]
            add(split,list(combination),scale,'mixture')
    return {'name':'chatflypt-independent-protocol-v1','author_exact_protocol':False,
            'words':WORDS,'duration_ms':300,'model':'Shiu LIF equations; full FlyWire v783 graph',
            'all_sensory_cells_excluded':True,'train_test_seed_overlap':False,
            'mixture_configurations_disjoint':True,'held_out_demo':FIRST_SENTENCE,
            'readout':'Linear ridge regression, alpha=10; thresholds chosen on validation only',
            'criteria':{'single_accuracy_target':1.0,'motor_accuracy_reference':.85,
                        'first_sentence_exact_recovery':True,'command_success_reference':[37,37]},
            'caveat':'Targets quote reported demo metrics; vocabulary, sampling, dynamics settings and reverse code are our independent choices.',
            'cases':cases}


def generate(workers=4):
    brain=Brain()
    manifest=build_manifest(brain)
    plan=protocol()
    fingerprint=hashlib.sha256((json.dumps(plan,sort_keys=True)+json.dumps(manifest,sort_keys=True)
        +sha256(ROOT/'chatflypt/brain.py')).encode()).hexdigest()
    plan['fingerprint']=fingerprint
    dump(REPORTS/'protocol.json',plan)
    cache=ARTIFACTS/'trials'/fingerprint[:16]
    cache.mkdir(parents=True,exist_ok=True)
    brain.run([],1,0)  # Compile once before worker threads.
    start=time.perf_counter()
    def run(case):
        path=cache/(case['id']+'.npz')
        if not path.exists():
            trial=brain.run(groups_for(manifest,case['words'],case['scale']),
                            case['duration_ms'],case['seed'])
            np.savez_compressed(path,counts=trial.counts)
        return case['id']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run,c) for c in plan['cases']]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%16==0 or i==len(futures):
                print(f'Corpus {i}/{len(futures)}, {time.perf_counter()-start:.1f}s',flush=True)
    matrices=[]
    for c in plan['cases']:
        with np.load(cache/(c['id']+'.npz'),allow_pickle=False) as f:
            matrices.append(sparse.csr_matrix(f['counts'][None,:]))
    sparse.save_npz(ARTIFACTS/'counts.npz',sparse.vstack(matrices,format='csr'))
    dump(ARTIFACTS/'corpus.json',plan)
    print('Corpus saved',flush=True)


class Decoder:
    def __init__(self,path=None):
        with np.load(path or ARTIFACTS/'decoder.npz',allow_pickle=False) as f:
            self.indices=f['indices']
            self.weights=f['weights']
            self.intercept=f['intercept']
            self.thresholds=f['thresholds']
    def scores(self,counts):
        # No stimulus, labels, seed, or trial metadata enter this interface.
        if sparse.issparse(counts):
            x=counts[:,self.indices].astype(float)
            x.data=np.sqrt(x.data)
            return np.asarray(x@self.weights.T)+self.intercept
        return np.sqrt(np.asarray(counts)[...,self.indices])@self.weights.T+self.intercept
    def decode(self,counts):
        values=self.scores(counts)
        if values.ndim!=1:
            raise ValueError('decode expects one neural count vector')
        return {'words':[w for w,v,t in zip(WORDS,values,self.thresholds) if v>=t],
                'scores':{w:float(v) for w,v in zip(WORDS,values)},
                'score_type':'uncalibrated linear scores; not probabilities'}


def metrics(y,scores,thresholds,kinds):
    from sklearn.metrics import accuracy_score, f1_score
    pred=scores>=thresholds
    one=np.array([k=='single' for k in kinds])
    result={'trials':len(y),'exact_set_accuracy':float(np.mean(np.all(pred==y,axis=1))),
            'micro_f1':float(f1_score(y,pred,average='micro',zero_division=0)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'per_word_f1':dict(zip(WORDS,f1_score(y,pred,average=None,zero_division=0).tolist()))}
    if one.any():
        result['single_word_top1_accuracy']=float(accuracy_score(y[one].argmax(1),scores[one].argmax(1)))
        result['single_word_trials']=int(one.sum())
    return result


def fit_probe(x,y,train,val,indices,out):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import balanced_accuracy_score
    features=x[:,indices].astype(float)
    features.data=np.sqrt(features.data)
    model=Ridge(alpha=10,solver='lsqr',tol=1e-6)
    model.fit(features[train],y[train])
    val_scores=model.predict(features[val])
    thresholds=[]
    for j in range(len(WORDS)):
        grid=np.linspace(.15,.85,71)
        quality=[balanced_accuracy_score(y[val,j],val_scores[:,j]>=t) for t in grid]
        best=max(quality)
        candidates=[t for t,q in zip(grid,quality) if abs(q-best)<1e-12]
        thresholds.append(min(candidates,key=lambda t:abs(t-.5)))
    np.savez_compressed(out,indices=indices,weights=model.coef_,intercept=model.intercept_,thresholds=thresholds)
    return Decoder(out)


def train():
    plan=json.loads((ARTIFACTS/'corpus.json').read_text(encoding='utf8'))
    manifest=json.loads((ARTIFACTS/'stimuli.json').read_text(encoding='utf8'))
    x=sparse.load_npz(ARTIFACTS/'counts.npz')
    y=np.array([[int(w in c['words']) for w in WORDS] for c in plan['cases']])
    splits=np.array([c['split'] for c in plan['cases']])
    train,val=splits=='train',splits=='validation'
    kinds=np.array([c['kind'] for c in plan['cases']])
    report={'protocol_fingerprint':plan['fingerprint'],'original_author_replication':False,
            'feature_selection':'All sensory/sensory_ascending and stimulated cells excluded before fitting',
            'models':{}}
    with threadpool_limits(limits=4):
        for name,key in [('all_non_sensory','readout_indices'),('descending_motor','motor_indices')]:
            indices=np.array(manifest[key])
            # Feature selection uses only training activity, never held-out data.
            indices=indices[np.asarray(x[train][:,indices].getnnz(axis=0)).ravel()>0]
            out=ARTIFACTS/('decoder.npz' if name=='all_non_sensory' else 'decoder-motor.npz')
            decoder=fit_probe(x,y,train,val,indices,out)
            scores=decoder.scores(x)
            summary={'features':len(indices),'artifact_sha256':sha256(out),'metrics':{}}
            for split in ['train','validation','test','ood']:
                mask=splits==split
                summary['metrics'][split]=metrics(y[mask],scores[mask],decoder.thresholds,kinds[mask])
            report['models'][name]=summary
            print(name,json.dumps(summary['metrics']['test']),flush=True)
            if name=='all_non_sensory':
                test=splits=='test'
                shuffled=np.random.default_rng(898).permutation(scores[test])
                report['controls']={'shuffled_neural_observations':metrics(y[test],shuffled,decoder.thresholds,kinds[test]),
                    'zero_neural_observations':metrics(y[test],np.tile(decoder.intercept,(test.sum(),1)),decoder.thresholds,kinds[test]),
                    'interpretation':'Negative controls test whether predictions depend on neural observations. No claim of connectome superiority over random wiring is made.'}
                dump(REPORTS/'heldout-predictions.json',[
                    {'trial_id':c['id'],'split':c['split'],'expected':c['words'],
                     'predicted':[w for w,s,t in zip(WORDS,scores[i],decoder.thresholds) if s>=t]}
                    for i,c in enumerate(plan['cases']) if c['split'] in ['test','ood']])
    dump(REPORTS/'readout-evaluation.json',report)


def physiology():
    brain=Brain('630')
    sugar,bitter=notebook_literal('neu_sugar'),notebook_literal('neu_bitter')
    mn9=brain.lookup[str(notebook_literal('id_mn9'))]
    trials=[]
    for seed in range(10):
        row={'seed':70000+seed}
        for name,groups in [('sugar',[(sugar,150)]),('sugar_bitter',[(sugar,150),(bitter,150)])]:
            r=brain.run(groups,1000,70000+seed)
            row[name+'_mn9_hz']=int(r.counts[mn9])
        trials.append(row)
    sweet=np.array([t['sugar_mn9_hz'] for t in trials])
    bitter_rate=np.array([t['sugar_bitter_mn9_hz'] for t in trials])
    difference=sweet-bitter_rate
    rng=np.random.default_rng(61)
    boot=np.mean(rng.choice(difference,(10000,len(difference))),axis=1)
    result={'dataset':'FlyWire v630','input_source':'Shiu figures.ipynb','duration_ms':1000,
            'rate_hz':150,'trials':trials,'sugar_mean_hz':float(sweet.mean()),
            'sugar_bitter_mean_hz':float(bitter_rate.mean()),
            'paired_mean_suppression_hz':float(difference.mean()),
            'paired_bootstrap_95_ci_hz':np.quantile(boot,[.025,.975]).tolist(),
            'direction_passed':bool(np.quantile(boot,.025)>0),'author_reported_hz':[78,3],
            'exact_numeric_replication':False,'note':'Independent prespecified protocol; no parameter fitting to reported numbers.'}
    dump(REPORTS/'mn9-evaluation.json',result)
    print(result,flush=True)


def verify_full():
    from .verify import brian_reference
    from .brain import simulate
    brain=Brain('630')
    results=[]
    sugar,bitter=notebook_literal('neu_sugar'),notebook_literal('neu_bitter')
    for name,groups in [('sugar',[(sugar,150)]),('sugar_bitter',[(sugar,150),(bitter,150)])]:
        tape=brain.input_tape(groups,100,481)
        ours=simulate(brain.indptr,brain.indices,brain.weights,*tape,True)
        ref=brian_reference(brain.indptr,brain.indices,brain.weights,tape)
        match=sorted(zip(ours[2],ours[1]))==sorted(zip(ref[1],ref[0]))
        result={'condition':name,'neurons':len(brain.ids),'edges':len(brain.indices),
                'custom_spikes':len(ours[1]),'brian_spikes':len(ref[0]),'spikes_identical':match,
                'voltage_max_error_mv':float(np.max(np.abs(ours[3]-ref[2])))}
        results.append(result)
        print(result,flush=True)
        if not match:
            raise AssertionError('Full-graph engine comparison failed')
    dump(REPORTS/'engine-verification-full.json',{'passed':True,'cases':results,
         'upstream_model_sha256':sha256(ROOT/'vendor/shiu-brain/model.py')})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['verify-full','physiology','generate','train'])
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.stage=='generate':generate(args.workers)
    elif args.stage=='train':train()
    elif args.stage=='physiology':physiology()
    else:verify_full()


if __name__=='__main__':main()
