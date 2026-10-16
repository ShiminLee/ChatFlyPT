"""Execute the published examples with explicit reconstruction assumptions.

These are neural-only reconstruction experiments (blind readout, no external
model in the loop). The historical LLM adviser and action-feedback loop were
removed on 2026-09-28 and are not part of any current interface.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from .brain import Brain
from .experiment import Decoder
from .stimuli import FIRST_SENTENCE, groups_for
from .language import verbalize
from .paths import ARTIFACTS, REPORTS, dump, sha256


REFERENCE_SENTENCE='I smell vinegar and yeast. I taste sugar. It is warm and bright. What should I do?'


def sentence_examples(brain,manifest,workers=4):
    decoder=Decoder()
    raw=ARTIFACTS/'demo-spikes'
    raw.mkdir(exist_ok=True)
    def trial(seed):
        r=brain.run(groups_for(manifest,FIRST_SENTENCE),300,seed,record_spikes=True)
        decoded=decoder.decode(r.counts)
        sentence=verbalize(decoded['words'])
        path=raw/f'first-sentence-{seed}.npz'
        np.savez_compressed(path,neuron_indices=r.spike_ids,steps=r.spike_steps,counts=r.counts)
        return {'seed':seed,'expected_words':FIRST_SENTENCE,'decoded':decoded,
                'sentence':sentence,'exact_words':set(decoded['words'])==set(FIRST_SENTENCE),
                'exact_sentence':sentence==REFERENCE_SENTENCE,'spikes':int(r.counts.sum()),
                'raw_spikes':str(path.relative_to(ARTIFACTS.parent.parent.parent)),
                'raw_sha256':sha256(path)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        trials=list(pool.map(trial,range(81000,81020)))
    report={'reference_sentence':REFERENCE_SENTENCE,'trials':trials,
            'exact_sentence_successes':sum(x['exact_sentence'] for x in trials),'total':len(trials),
            'zero_signal_sentence':verbalize(decoder.decode(np.zeros(len(brain.ids)))['words']),
            'training_excluded_this_combination':True,
            'semantics':'Engineering sensory proxies. The final question is a fixed template, not a decoded intention.'}
    dump(REPORTS/'first-sentence.json',report)
    print('First sentence',report['exact_sentence_successes'],'/',len(trials),flush=True)
    return report


def command_examples(brain,manifest,command='left',workers=4):
    labels=['left','right','stop','forward']
    if command not in labels:
        raise ValueError('Unsupported auditory command')
    def groups(direction):
        if direction=='stop':return []
        if direction=='forward':
            return [(manifest['auditory_commands']['left']+manifest['auditory_commands']['right'],100)]
        return [(manifest['auditory_commands'][direction],100)]
    def simulate_case(item):
        label,seed=item
        return brain.run(groups(label),300,seed).counts
    train_cases=[(label,82000+li*100+j) for li,label in enumerate(labels) for j in range(20)]
    test_cases=[(label,83000+li*100+j) for li,label in enumerate(labels) for j in range(37)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        train_counts=sparse.csr_matrix(np.stack(list(pool.map(simulate_case,train_cases))))
        test_counts=sparse.csr_matrix(np.stack(list(pool.map(simulate_case,test_cases))))
    y=np.repeat(np.eye(len(labels)),20,axis=0)
    report={'command':command,'training_trials':len(train_cases),'test_trials':len(test_cases),
            'encoding':'Artificial code: left/right = left/right auditory cells; forward = both; stop = silence.',
            'controller':'Decoded label becomes a turn in a toy grid. This is not direct biomechanical steering.',
            'by_readout':{}}
    for name,key in [('non_sensory','readout_indices'),('descending_motor','motor_indices')]:
        idx=np.array(manifest[key])
        idx=idx[np.asarray(train_counts[:,idx].getnnz(axis=0)).ravel()>0]
        x=train_counts[:,idx].astype(float);x.data=np.sqrt(x.data)
        xt=test_counts[:,idx].astype(float);xt.data=np.sqrt(xt.data)
        if len(idx)==0:
            report['by_readout'][name]={'status':'no_downstream_activity','correct':0,'total':len(test_cases)}
            continue
        model=Ridge(alpha=10,solver='lsqr',tol=1e-6).fit(x,y)
        predicted=np.argmax(model.predict(xt),axis=1)
        expected=np.repeat(np.arange(len(labels)),37)
        successes={label:int(np.sum(predicted[expected==i]==i)) for i,label in enumerate(labels)}
        report['by_readout'][name]={'features':len(idx),'successes_by_command':successes,
                                  'trials_per_command':37,'correct':int(np.sum(predicted==expected)),
                                  'predictions':[labels[i] for i in predicted]}
        np.savez_compressed(ARTIFACTS/f'command-decoder-{name}.npz',indices=idx,
                            weights=model.coef_,intercept=model.intercept_,labels=labels)
    # Independently inspect anatomical turn-related neurons; do not overwrite
    # this measurement with the learned external command decoder's decision.
    turn={'left':['720575940644438551','720575940604737708'],
          'right':['720575940627787609','720575940629327659']}
    rates={side:np.asarray(test_counts[:,[brain.lookup[x] for x in ids]].sum(axis=1)).ravel()/0.3
           for side,ids in turn.items()}
    rows=[]
    for i,(intended,seed) in enumerate(test_cases):
        lateral='left' if rates['left'][i]>rates['right'][i] else ('right' if rates['right'][i]>rates['left'][i] else 'tie')
        rows.append({'seed':seed,'input_command':intended,'DNa01_DNa02_left_summed_hz':float(rates['left'][i]),
                     'DNa01_DNa02_right_summed_hz':float(rates['right'][i]),'larger_side':lateral})
    report['anatomical_laterality']=rows
    report['anatomical_direction_counts']={label:{side:sum(r['larger_side']==side for r in rows if r['input_command']==label)
                                                for side in ['left','right','tie']} for label in labels}
    report['native_motor_behavior_validated']=False
    report['note']='DNa laterality is only a neural measurement; this experiment has no body physics.'
    dump(REPORTS/'auditory-command.json',report)
    np.savez_compressed(ARTIFACTS/'command-heldout-counts.npz',counts=test_counts.toarray(),
                        intended=[c[0] for c in test_cases],seeds=[c[1] for c in test_cases])
    print('Auditory command',json.dumps({k:v.get('successes_by_command') for k,v in report['by_readout'].items()}),flush=True)
    return report


def summary():
    def read(name):
        p=REPORTS/name
        return json.loads(p.read_text(encoding='utf8')) if p.exists() else None
    mn9,readout,first,auditory=[read(n) for n in ['mn9-evaluation.json','readout-evaluation.json',
                                                 'first-sentence.json','auditory-command.json']]
    rows=[]
    if mn9:rows.append({'example':'Sugar -> MN9, bitter suppresses it','status':'passed_direction' if mn9['direction_passed'] else 'failed',
        'measured':[mn9['sugar_mean_hz'],mn9['sugar_bitter_mean_hz']],'reference':[78,3],
        'exact_author_replication':False})
    if readout:
        for name,target in [('all_non_sensory',1.0),('descending_motor',.85)]:
            value=readout['models'][name]['metrics']['test']['single_word_top1_accuracy']
            rows.append({'example':'16 proxy words: '+name,'status':'target_met' if value>=target else 'target_not_met',
                         'measured':value,'reference':target,'exact_author_replication':False})
    if first:rows.append({'example':'Vinegar + yeast + sugar + warm + bright sentence',
        'status':'passed' if first['exact_sentence_successes']==first['total'] else 'partial_or_failed',
        'measured':[first['exact_sentence_successes'],first['total']],'exact_author_replication':False})
    if auditory:
        val=auditory['by_readout']['descending_motor'].get('successes_by_command',{}).get('left',0)
        rows.append({'example':'Auditory left code -> external motor-population readout',
                     'status':'target_met' if val==37 else 'target_not_met','measured':[val,37],
                     'native_left_turn_validated':False,'exact_author_replication':False})
        native=[r for r in auditory['anatomical_laterality'] if r['input_command']=='left']
        rows.append({'example':'Native DNa01/DNa02 directionality under the left auditory code',
                     'status':'not_demonstrated','left_larger_trials':sum(r['larger_side']=='left' for r in native),
                     'tied_trials':sum(r['larger_side']=='tie' for r in native),'total':len(native),
                     'note':'The external command decoder success does not establish native turning.'})
    rows.append({'example':'External LLM advice and action feedback',
                 'status':'removed','removed_on':'2026-09-28',
                 'note':'The adviser loop is outside the current interface; body action reads recorded neural activity only.'})
    report={'kind':'independent functional reconstruction','exact_author_replication':False,'examples':rows}
    dump(REPORTS/'acceptance.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    brain=Brain();manifest=json.loads((ARTIFACTS/'stimuli.json').read_text(encoding='utf8'))
    sentence_examples(brain,manifest,args.workers)
    command_examples(brain,manifest,workers=args.workers)
    print(json.dumps(summary(),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
