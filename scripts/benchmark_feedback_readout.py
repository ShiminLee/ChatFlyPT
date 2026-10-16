"""Collect actual closed-loop episodes; keep whole sessions out of training.

The protocol is saved before collection. Receptor events provide evaluation
targets only; model features are recorded internal neural activity. No LLM or
target-seeking controller runs. This benchmark does not calibrate biology.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import traceback
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from chatflypt.paths import ROOT, dump, sha256

OUT=ROOT/'reports/feedback-readout'
RAW=ROOT/'data/derived/feedback-readout'


def protocol():
    from chatflypt.environment import validate_objects
    episodes=[]
    for split,index in [('train',0),('validation',1),('test',2)]:
        for dark,odor in [(False,False),(True,False),(False,True),(True,True)]:
            objects=[]
            if dark:
                objects.append({'id':'shade','type':'shade','position_mm':[0,0,10],
                                'size_mm':[20,20,20],'strength':1})
            if odor:
                objects.append({'id':'yeast','type':'yeast','position_mm':[4+index*.25,.2-index*.2,.35],
                    'size_mm':[.6,.6,.6],'range_mm':7.,'strength':[1.,.85,1.15][index]})
            eid=f'flywire-{split}-{int(dark)}{int(odor)}'
            episodes.append({'id':eid,'dataset':'FlyWire v783','split':split,'seed':320000+len(episodes)*10,
                'environment':{'schema':'chatflypt.environment.v1','revision':1,'objects':validate_objects(objects)},
                'turns':[[],[]]})
        for order in [('left','right'),('right','left')]:
            cue=lambda side:[{'channel':'foreleg_touch','side':side,'intensity':[1.,.85,1.15][index],
                               'onset_ms':50.,'duration_ms':300.}]
            episodes.append({'id':f'malecns-{split}-{order[0]}','dataset':'MaleCNS v1.0','split':split,
                'seed':320000+len(episodes)*10,'environment':{'schema':'chatflypt.environment.v1','revision':0,'objects':[]},
                'turns':[cue(order[0]),[],cue(order[1]),[]]})
    return {'schema':'chatflypt.feedback-readout-protocol.v1','duration_ms':500.,'interval_ms':100.,
        'episodes':episodes,'seed_overlap':False,'split_unit':'whole independent neural/body session',
        'target':'Sensory population with actual input events during the current observation window',
        'scope':'Bounded synthetic visual/odor fields and foreleg cues, including cue removal and switched sides',
        'spatial_vision_calibrated':False,'natural_perception_calibrated':False,
        'qualification':'No global qualification from this small benchmark; all live feedback labels remain candidates.',
        'selected_features':'Existing fixed internal-neuron feature mask, duration-normalized counts and voltage areas',
        'fit':'Ridge alpha=10; validation-only threshold selection; test untouched until final evaluation'}


def collect(plan,dataset,split='all'):
    from chatflypt.state_readout import setup
    from chatflypt.stimuli import ANNOTATIONS
    from chatflypt.embodied import EmbodiedSession,run_embodied
    from chatflypt.sensory_catalogue import build_catalogue
    import pandas as pd
    brain,_=setup(dataset);catalogue=build_catalogue(save=False)
    rows=brain.rows if dataset=='MaleCNS v1.0' else pd.read_csv(ANNOTATIONS,sep='\t',low_memory=False,dtype={'root_id':str}).fillna('')
    if dataset!='MaleCNS v1.0':
        rows=rows.set_index(rows.root_id.astype(str)).reindex(brain.ids.astype(str)).fillna('')
        rows['root_id']=rows.index
    for episode in plan['episodes']:
        if episode['dataset']!=dataset or (split!='all' and episode['split']!=split):continue
        dest=RAW/episode['id'];complete=dest/'complete.json'
        if complete.exists():
            saved=json.loads(complete.read_text(encoding='utf8'))
            if saved.get('protocol_sha256')!=sha256(OUT/'protocol.json'):raise ValueError('Stale cached protocol')
            continue
        state=EmbodiedSession(episode['environment'],'ground',episode['id'])
        records=[]
        try:
            for index,items in enumerate(episode['turns']):
                folder=dest/f'turn-{index}'
                result=run_embodied(brain,rows,catalogue,dataset,items,episode['environment'],
                    episode['seed']+index,'ground',duration_ms=plan['duration_ms'],output_dir=folder,session=state)
                trial=result['trial'];body=result['body'];feedback=result['feedback']
                assert body['status']=='ok' and body['summary']['finite_state']
                assert not body['summary']['physics_warnings']
                record=folder/'observation.npz'
                np.savez_compressed(record,counts=trial.counts,state_record=trial.state_record,
                    spike_indices=trial.spike_ids,spike_steps=trial.spike_steps,driven_indices=result['driven'],
                    duration_ms=trial.duration_ms)
                event_file=folder/'window-input-events.npz'
                np.savez_compressed(event_file,**result['events'])
                row={'episode':episode['id'],'split':episode['split'],'dataset':dataset,'turn_index':index,
                    'seed':episode['seed']+index,'human_items':items,
                    'environment_items':result['environment_items'],
                    'record':str(record.relative_to(ROOT)),'record_sha256':sha256(record),
                    'events':str(event_file.relative_to(ROOT)),'events_sha256':sha256(event_file),
                    'feedback_record':str((folder/'feedback.json').relative_to(ROOT)),
                    'body_record':str((folder/'body-result.json').relative_to(ROOT)),
                    'spikes':int(trial.counts.sum()),'physics_run':body['physics_run'],
                    'session_start_ms':feedback['session_start_ms'],'session_end_ms':feedback['session_end_ms'],
                    'neural_resets':feedback['neural_reset_count'],'body_resets':feedback['body_reset_count'],
                    'llm_called':False,'goal_controller':False}
                if index:
                    assert feedback['neural_reset_count']==feedback['body_reset_count']==0
                    assert feedback['initial_neural_state_sha256']==feedback['previous_checkpoint']['neural_state_sha256']
                dump(folder/'observation.json',row);records.append(row)
                print(json.dumps({'episode':episode['id'],'turn':index,'spikes':row['spikes'],'progress':'recorded'},ensure_ascii=False),flush=True)
        finally:state.close()
        dump(complete,{'protocol_sha256':sha256(OUT/'protocol.json'),'records':records})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--collect',action='store_true')
    parser.add_argument('--dataset',choices=['flywire','malecns','all'],default='all')
    parser.add_argument('--split',choices=['all','train','validation','test'],default='all')
    args=parser.parse_args();plan=protocol()
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8'))!=plan:raise ValueError('Do not overwrite an existing protocol')
    dump(path,plan)
    if args.collect:
        datasets={'flywire':'FlyWire v783','malecns':'MaleCNS v1.0'}
        chosen=list(datasets.values()) if args.dataset=='all' else [datasets[args.dataset]]
        def work(dataset):
            try:collect(plan,dataset,args.split)
            except Exception:
                traceback.print_exc()
                raise
        with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=len(chosen)) as pool:
            list(pool.map(work,chosen))


if __name__=='__main__':main()
