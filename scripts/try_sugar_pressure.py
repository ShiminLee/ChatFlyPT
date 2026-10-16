"""Exploratory sugar/contact/intensity trials with unchanged neural/body models."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from chatflypt.paths import ROOT,dump,sha256

REPORT=ROOT/'reports/sugar-pressure-20260924'
RAW=ROOT/'data/derived/sugar-pressure-20260924'
SEEDS=[491701,491702]


def protocol():
    from chatflypt.environment import validate_objects
    def item(kind,position,size):
        return {'id':'sample','type':kind,'position_mm':position,'size_mm':size,'strength':1,'range_mm':4}
    far=item('sugar',[4,-2,.5],[1,1,1])
    mouth=lambda kind:item(kind,[1.1,0,.4],[.7,1,.8])
    male=[('empty','空环境',[],['']),('far_sugar','远处糖',[far],['']),
          ('mouth_block','口器位置普通方块',[mouth('block')],['']),
          ('mouth_sugar','口器位置糖',[mouth('sugar')],['']),
          ('mouth_high_salt','口器位置高盐',[mouth('high_salt')],['']),
          ('leg_sugar','脚下糖平台',[item('sugar',[0,0,.1],[5,4,.2])],['']),
          ('sweet_cue','人类消息：糖',[],['糖']),
          ('weak_touch','弱触碰右前腿',[],['轻轻触碰右前腿']),
          ('strong_touch','强触碰右前腿及撤除',[],['非常强触碰右前腿','停止输入']),
          ('sugar_strong_touch','口器糖与强触碰',[mouth('sugar')],['非常强触碰右前腿'])]
    fly=[('empty','空环境',[],['']),('far_sugar','远处糖',[far],['']),
         ('heat','位置相关热源',[item('heat',[1,0,1],[.2,.2,.2])],[''])]
    cases=[]
    for key,dataset,conditions in [('malecns','MaleCNS v1.0',male),('flywire','FlyWire v783',fly)]:
        for seed in SEEDS:
            for name,label,objects,messages in conditions:
                cases.append({'id':f'{key}-{seed}-{name}','condition':name,'label':label,'dataset':dataset,
                    'seed':seed,'messages':messages,
                    'environment':{'schema':'chatflypt.environment.v1','revision':1,'objects':validate_objects(objects)}})
    return {'schema':'chatflypt.sugar-pressure-trials.v1','duration_ms':500.,'interval_ms':100.,
        'seeds':SEEDS,'cases':cases,'comparison':'Matched initialization seed across conditions; independent sessions.',
        'scope':'Exploratory virtual stimulation, not biological reward/punishment or semantic obedience.',
        'model_fit':False,'llm_called':False,'controller_changed':False,'intensity_units':'Engineering rate multiplier, not physical force or temperature.'}


def run_dataset(key,plan,workers):
    import pandas as pd
    from chatflypt.state_readout import setup,StateDecoder,OUT as BASE_OUT
    from chatflypt.feedback_readout import load_feedback_decoder
    from chatflypt.sensory_readout import UncalibratedDecoder
    from chatflypt.sensory_catalogue import build_catalogue
    from chatflypt.stimuli import ANNOTATIONS
    from chatflypt.input_language import parse_input
    from chatflypt.embodied import EmbodiedSession,run_embodied
    from chatflypt.reactions import summarize_reaction
    from chatflypt.expression import expression_report
    dataset={'malecns':'MaleCNS v1.0','flywire':'FlyWire v783'}[key]
    brain,_=setup(dataset);catalogue=build_catalogue(save=False)
    if key=='malecns':
        rows=brain.rows
        reaction_rows=pd.DataFrame({'cell_type':rows['type'],'cell_class':rows['class'],
            'super_class':rows.superclass,'side':rows.somaSide.replace({'L':'left','R':'right'})})
    else:
        rows=pd.read_csv(ANNOTATIONS,sep='\t',low_memory=False,dtype={'root_id':str}).fillna('')
        rows=rows.set_index(rows.root_id.astype(str)).reindex(brain.ids.astype(str)).fillna('')
        rows['root_id']=rows.index;reaction_rows=rows
    reader,note=load_feedback_decoder(StateDecoder(dataset),dataset)
    assert note is None
    def trained(item):
        spec=catalogue['channels'][item['channel']]
        if spec.get('advanced'):return False
        side=item.get('side','bilateral')
        ids=spec.get('body_part_ids',{}).get(item.get('body_part'),spec['side_ids'])
        return set(ids.get(side,[]))==set(spec['side_ids'].get(side,[]))
    def run(case):
        folder=RAW/case['id'];complete=folder/'complete.json'
        if complete.exists():
            saved=json.loads(complete.read_text(encoding='utf8'))
            assert saved['protocol_sha256']==sha256(REPORT/'protocol.json')
            return saved['records']
        state=EmbodiedSession(case['environment'],'ground',case['id']);records=[]
        try:
            for turn,message in enumerate(case['messages']):
                parsed=parse_input(message,catalogue) if message else {'kind':'silence','items':[]}
                assert parsed['kind'] in ('stimulus','silence'),parsed
                dest=folder/f'turn-{turn}'
                result=run_embodied(brain,rows,catalogue,dataset,parsed['items'],case['environment'],
                    case['seed']+turn,'ground',duration_ms=500.,output_dir=dest,session=state)
                trial=result['trial'];feedback=result['feedback'];body=result['body']
                assert body['status']=='ok' and body['summary']['finite_state'] and not body['summary']['physics_warnings']
                selected=parsed['items']+result['environment_items']
                calibrated=all(trained(i) for i in selected)
                decoder=reader if calibrated else UncalibratedDecoder()
                decoded=decoder.decode_trial(trial) if calibrated else decoder.decode(trial.counts)
                if calibrated:assert not np.intersect1d(decoder.indices,result['driven']).size
                reaction=summarize_reaction(brain,trial,result['driven'],reaction_rows)
                expression=expression_report({'id':case['id']+f'-{turn}','dataset':dataset,
                    'decoded':decoded,'reaction':reaction,'embodied_feedback':feedback})
                raw=dest/'observation.npz'
                np.savez_compressed(raw,counts=trial.counts,state_record=trial.state_record,
                    spike_indices=trial.spike_ids,spike_steps=trial.spike_steps,driven_indices=result['driven'],
                    duration_ms=trial.duration_ms,seed=case['seed']+turn)
                np.testing.assert_array_equal(np.bincount(trial.spike_ids,minlength=len(brain.ids)),trial.counts)
                np.savez_compressed(dest/'window-events.npz',**result['events'])
                if turn:
                    assert feedback['neural_reset_count']==feedback['body_reset_count']==0
                    assert feedback['initial_neural_state_sha256']==feedback['previous_checkpoint']['neural_state_sha256']
                frames=body['trajectory'];yaw=np.unwrap([f['yaw_rad'] for f in frames])
                contacts=sorted({c['fly_geom'] for f in frames for c in f['environment_contacts']})
                evidence=[dict(time_ms=s['time_ms'],**e) for s in feedback['samples'] for e in s['evidence'] if e.get('mode')!='official_retina']
                report={'case':case['id'],'condition':case['condition'],'label':case['label'],'turn':turn,
                    'dataset':dataset,'seed':case['seed'],'message':message,'input_plan':parsed,
                    'spikes':int(trial.counts.sum()),'reaction':reaction,'decoded':decoded,'expression':expression,
                    'environment_items':result['environment_items'],'contact_geoms':contacts,'contact_evidence':evidence,
                    'unmapped':[u for s in feedback['samples'] for u in s['unmapped']],
                    'body_summary':body['summary'],'command_summary':body['command_summary'],
                    'powered_steps':body['controls']['powered_steps'],
                    'yaw_change_deg':float(np.rad2deg(yaw[-1]-yaw[0])),
                    'feedback':{k:feedback[k] for k in ['sample_count','session_start_ms','session_end_ms','neural_reset_count','body_reset_count','llm_called','goal_controller']},
                    'raw_record':str(raw.relative_to(ROOT)),'raw_sha256':sha256(raw),
                    'body_record':str((dest/'body-result.json').relative_to(ROOT)),
                    'feedback_record':str((dest/'feedback.json').relative_to(ROOT)),
                    'base_readout_sha256':sha256(BASE_OUT/f'{key}-decoder.npz')}
                dump(dest/'result.json',report);records.append(report)
                print(json.dumps({'case':case['id'],'turn':turn,'internal_spikes':reaction['non_sensory_spikes'],
                    'words':decoded['words'],'expression_status':expression['status'],'powered_steps':report['powered_steps'],
                    'path_mm':body['summary']['path_length_mm'],'contacts':contacts},ensure_ascii=False),flush=True)
        finally:state.close()
        dump(complete,{'protocol_sha256':sha256(REPORT/'protocol.json'),'records':records})
        return records
    chosen=[c for c in plan['cases'] if c['dataset']==dataset]
    with ThreadPoolExecutor(max_workers=workers) as pool:groups=list(pool.map(run,chosen))
    dump(REPORT/f'{key}-results.json',{'protocol_sha256':sha256(REPORT/'protocol.json'),
         'records':[r for group in groups for r in group]})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['malecns','flywire'],required=True)
    parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
    plan=protocol();path=REPORT/'protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf8'))==plan
    else:dump(path,plan)
    with threadpool_limits(limits=1):run_dataset(args.dataset,plan,args.workers)


if __name__=='__main__':main()
