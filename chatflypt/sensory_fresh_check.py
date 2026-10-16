"""Fresh neural trials after the universal validation gate; no language cases."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from .brain import Brain, simulate
from .malecns import MaleCNS
from .paths import ROOT,dump,sha256
from .sensory_catalogue import build_catalogue
from .sensory_events import encode_inputs
from .sensory_readout import (OUT,REPORT,DATASETS,SensoryDecoder,canonical_specs,targets)
from .social import to_engine_tape


def run():
    catalogue=build_catalogue(save=False)
    rng=np.random.default_rng(8842026)
    plans={}
    for key,dataset in DATASETS.items():
        specs=canonical_specs(catalogue,dataset)
        cases=[]
        def add(items,kind):
            cases.append({'seed':2800000+len(cases),'kind':kind,'items':[
                dict(i,intensity=1.,onset_ms=50.,duration_ms=300.) for i in items]})
        for spec in specs:
            for side in ['left','right']:
                if spec['side_ids'][side]:add([{'channel':spec['id'],'side':side}],'single')
        old=json.loads((REPORT/f'{key}-protocol.json').read_text(encoding='utf8'))
        used={tuple(sorted((i['channel'],i['side']) for i in c['items'])) for c in old['cases']}
        for _ in range(10):
            while True:
                selected=rng.choice(len(specs),int(rng.integers(2,5)),replace=False)
                items=[{'channel':specs[k]['id'],'side':str(rng.choice(['left','right','bilateral']))} for k in selected]
                identity=tuple(sorted((i['channel'],i['side']) for i in items))
                if identity not in used:used.add(identity);break
            add(items,'mixture')
        add([],'blank')
        plans[key]={'dataset':dataset,'cases':cases,'model_sha256':sha256(OUT/f'{key}-decoder.npz')}
    dump(REPORT/'fresh-neural-protocol.json',{
        'purpose':'Fresh post-correction neural tests, separate from the now-known language acceptance.',
        'models_frozen_before_new_counts':True,'seed_overlap_with_original_protocol':False,'datasets':plans})
    results={}
    for key,plan in plans.items():
        brain=Brain() if key=='flywire' else MaleCNS()
        decoder=SensoryDecoder(plan['dataset'])
        brain.run([],1,0)
        directory=OUT/'fresh-check'/key
        directory.mkdir(parents=True,exist_ok=True)
        def one(case):
            if case['items']:
                events=encode_inputs(catalogue,case['items'],case['seed'])[plan['dataset']]
                tape=to_engine_tape(brain,events)
                counts=simulate(brain.indptr,brain.indices,brain.weights,*tape,False)[0]
            else:counts=brain.run([],500,case['seed']).counts
            with threadpool_limits(limits=1):
                decoded=decoder.decode(counts)
            raw=directory/f"{case['seed']}.npz"
            np.savez_compressed(raw,counts=counts,seed=case['seed'])
            return {**case,'decoded':decoded,'raw_record':str(raw.relative_to(ROOT))}
        rows=[]
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(one,c) for c in plan['cases']]
            for i,future in enumerate(as_completed(futures),1):
                rows.append(future.result())
                if i%10==0:print(f'{key} fresh check {i}/{len(futures)}',flush=True)
        rows.sort(key=lambda r:r['seed'])
        y=np.array([targets(c,catalogue,decoder.labels) for c in rows])
        pred=np.array([[int(label in r['decoded']['words']) for label in decoder.labels] for r in rows])
        qualified=np.array(decoder.qualified)
        result={'trials':len(rows),'exact_set_accuracy_all_registered_labels':float(np.mean(np.all(y==pred,axis=1))),
                'micro_f1_all_registered_labels':float(f1_score(y,pred,average='micro',zero_division=0)),
                'micro_f1_qualified_labels':float(f1_score(y[:,qualified],pred[:,qualified],average='micro',zero_division=0)),
                'qualified_label_count':int(qualified.sum()),'total_label_count':len(qualified),
                'per_label_f1':dict(zip(decoder.labels,f1_score(y,pred,average=None,zero_division=0).tolist())),
                'predictions':rows,'model_sha256':plan['model_sha256']}
        results[key]=result
        dump(REPORT/'fresh-neural-check.json',results)
        print(json.dumps({k:v for k,v in result.items() if k not in ['predictions','per_label_f1']},ensure_ascii=False),flush=True)
    return results


if __name__=='__main__':run()
