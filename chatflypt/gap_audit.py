"""Re-examine zero-spike cases with voltage observations; retain original audit."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import numpy as np
from .paths import ROOT, dump, sha256
from .sensory_events import encode_inputs
from .social import to_engine_tape
from .brain import recorded_trial
from .state_readout import setup, DATASETS, REPORT, OUT


def run(workers=3):
    catalogue=json.loads((ROOT/'data/derived/sensory/catalogue.json').read_text(encoding='utf8'))
    original=json.loads((ROOT/'reports/general-sensory/channel-delivery.json').read_text(encoding='utf8'))
    previous={r['channel']:r for r in original['cases']}
    cases=[]; results=[]
    for i,spec in enumerate(sorted(catalogue['channels'].values(),key=lambda s:(s['dataset'],s['id']))):
        if spec['status']=='unmapped':continue
        old=previous.get(spec['id'])
        # Existing positive spike evidence is still positive electrical evidence.
        # Do not invent voltage measurements for trials that did not record them.
        if old and old['input_delivery_passed']:
            results.append(dict(old,electrical_response=True,evidence_type='archived_spike_record'))
        else:cases.append({'channel':spec['id'],'dataset':spec['dataset'],
                          'seed':old['seed'] if old else 3900000+i,'previous':old})
    dump(REPORT/'electrical-audit-protocol.json',{'cases':cases,
        'reused_positive_spike_trials':len(results),'code_sha256':sha256(ROOT/'chatflypt/brain.py'),
        'criterion':'A recorded internal spike OR peak absolute LIF voltage deviation > 1e-6 mV; excludes sensory/direct cells.',
        'note':'Reclassification measures electrical response, not identification or natural physiology.'})
    for key,dataset in DATASETS.items():
        todo=[c for c in cases if c['dataset']==dataset]
        if not todo:continue
        brain,allowed=setup(dataset)
        directory=OUT/'electrical-audit'/key;directory.mkdir(parents=True,exist_ok=True)
        def one(case):
            events=encode_inputs(catalogue,[{'channel':case['channel'],'side':'bilateral'}],case['seed'])[dataset]
            tape=to_engine_tape(brain,events)
            trial=recorded_trial(brain,tape,case['seed'],events['duration_ms'],False)
            mask=allowed.copy();mask[tape[0]]=False
            internal_cells=int(np.count_nonzero(trial.state_record[2:,mask].max(axis=0)>1e-6))
            counts=trial.counts;raw=directory/(hashlib.sha256(case['channel'].encode()).hexdigest()[:16]+'.npz')
            np.savez_compressed(raw,counts=counts,state=trial.state_record,seed=case['seed'],
                                input_neuron_ids=events['input_neuron_ids'])
            old=case.get('previous');same=None
            if old:
                with np.load(ROOT/old['raw_record']) as f:same=bool(np.array_equal(f['counts'],counts))
            return {'channel':case['channel'],'dataset':dataset,'seed':case['seed'],
                'input_events':len(events['event_steps']),'driven_spikes':int(counts[tape[0]].sum()),
                'non_sensory_spikes':int(counts[mask].sum()),'membrane_internal_cells':internal_cells,
                'electrical_response':bool(np.any(counts[mask]) or internal_cells),
                'evidence_type':'new_passive_voltage_record','same_spikes_as_original':same,
                'max_positive_mv':float(trial.state_record[2,mask].max(initial=0)),
                'max_negative_mv':float(trial.state_record[3,mask].max(initial=0)),
                'raw_record':str(raw.relative_to(ROOT))}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i,f in enumerate(as_completed([pool.submit(one,c) for c in todo]),1):
                results.append(f.result())
                if i%10==0:print(f'{key} voltage recheck {i}/{len(todo)}',flush=True)
    result={'total':len(results),'electrical_response':sum(r['electrical_response'] for r in results),
            'new_voltage_trials':len(cases),'archived_positive_spike_trials':len(results)-len(cases),
            'original_counts_preserved':all(r.get('same_spikes_as_original') is not False for r in results),
            'cases':sorted(results,key=lambda r:(r['dataset'],r['channel']))}
    dump(REPORT/'electrical-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    return result


if __name__=='__main__':run()
