"""New neural trials after the membrane-state models are frozen."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from .paths import ROOT,dump,sha256
from .sensory_readout import canonical_specs,targets
from .state_readout import DATASETS,OUT,REPORT,StateDecoder,setup,run_case,signature


def run(workers=6, keys=None):
    catalogue=json.loads((ROOT/'data/derived/sensory/catalogue.json').read_text(encoding='utf8'))
    keys=list(DATASETS) if keys is None else keys
    plans={}; previous={}; results={}
    # Separate invocations can evaluate each frozen model as it becomes ready.
    # Retain another dataset only when its frozen artifact still matches.
    protocol_path=REPORT/'fresh-state-protocol.json'
    result_path=REPORT/'fresh-state-results.json'
    if protocol_path.exists() and result_path.exists():
        prior_plans=json.loads(protocol_path.read_text(encoding='utf8'))['datasets']
        prior_results=json.loads(result_path.read_text(encoding='utf8'))
        for key,plan in prior_plans.items():
            path=OUT/f'{key}-decoder.npz'
            if key not in keys and path.exists() and sha256(path)==plan['model_sha256']:
                previous[key]=plan
                if key in prior_results:results[key]=prior_results[key]
    for dataset_index,(key,dataset) in enumerate(DATASETS.items()):
        if key not in keys:continue
        rng=np.random.default_rng(9842042+dataset_index)
        decoder=StateDecoder(dataset)
        if decoder.catalogue_signature!=signature(catalogue,dataset):raise ValueError('Stale model')
        specs=canonical_specs(catalogue,dataset);cases=[]
        def add(items,kind):
            cases.append({'seed':4800000+len(cases),'kind':kind,'items':[
                dict(item,intensity=1.,onset_ms=50.,duration_ms=300.) for item in items]})
        for spec in specs:
            for side in ['left','right']:
                if spec['side_ids'][side]:add([{'channel':spec['id'],'side':side}],'single')
        old=json.loads((REPORT/f'{key}-state-protocol.json').read_text(encoding='utf8'))
        used={tuple(sorted((i['channel'],i['side']) for i in c['items'])) for c in old['cases']}
        for _ in range(10):
            while True:
                chosen=rng.choice(len(specs),int(rng.integers(2,5)),replace=False)
                items=[{'channel':specs[i]['id'],'side':str(rng.choice([s for s in ['left','right','bilateral'] if specs[i]['side_ids'][s]]))} for i in chosen]
                identity=tuple(sorted((i['channel'],i['side']) for i in items))
                if identity not in used:used.add(identity);break
            add(items,'mixture')
        add([],'blank')
        plans[key]={'dataset':dataset,'cases':cases,'model_sha256':sha256(OUT/f'{key}-decoder.npz')}
    dump(REPORT/'fresh-state-protocol.json',{'models_frozen_before_trials':True,
        'purpose':'Fresh seeds and mixtures; every registered primary label stays in the metric denominator.',
        'datasets':{**previous,**plans}})
    for key,plan in plans.items():
        brain,allowed=setup(plan['dataset']);decoder=StateDecoder(plan['dataset'])
        directory=OUT/'fresh-check'/key;directory.mkdir(parents=True,exist_ok=True)
        def one(case):
            trial=run_case(brain,catalogue,plan['dataset'],case)
            decoded=decoder.decode_trial(trial)
            path=directory/f"{case['seed']}.npz"
            np.savez_compressed(path,counts=trial.counts,state_record=trial.state_record,seed=case['seed'])
            return {**case,'decoded':decoded,'raw_record':str(path.relative_to(ROOT))}
        rows=[]
        with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=workers) as pool:
            for i,future in enumerate(as_completed([pool.submit(one,c) for c in plan['cases']]),1):
                rows.append(future.result())
                if i%10==0:print(f'{key} fresh state {i}/{len(plan["cases"])}',flush=True)
        rows.sort(key=lambda r:r['seed'])
        y=np.array([targets(c,catalogue,decoder.labels) for c in rows])
        pred=np.array([[int(w in c['decoded']['words']) for w in decoder.labels] for c in rows])
        results[key]={'trials':len(rows),'exact_set_accuracy':float(np.mean(np.all(y==pred,axis=1))),
            'micro_f1':float(f1_score(y,pred,average='micro',zero_division=0)),
            'per_label_f1':dict(zip(decoder.labels,f1_score(y,pred,average=None,zero_division=0).tolist())),
            'model_unchanged':sha256(OUT/f'{key}-decoder.npz')==plan['model_sha256'],
            'model_sha256':plan['model_sha256'],'cases':rows}
        dump(REPORT/'fresh-state-results.json',results)
        print(json.dumps({k:v for k,v in results[key].items() if k not in ['per_label_f1','cases']}),flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    run(args.workers,[args.dataset] if args.dataset else None)
