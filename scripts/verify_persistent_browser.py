"""Verify recorded UI turns: continuity and independent full-prefix replay."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from chatflypt.paths import dump
from chatflypt.neural_session import replay_events


def main():
    records=[]
    receipt=ROOT/'reports/persistent-session/browser-acceptance.json'
    if receipt.exists():
        paths=[ROOT/t['record'] for t in json.loads(receipt.read_text(encoding='utf8'))['turns']]
        records=[(p,json.loads(p.read_text(encoding='utf8'))) for p in paths]
    else:
        for path in sorted((ROOT/'reports/dialogue').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
            r=json.loads(path.read_text(encoding='utf8'))
            if r.get('dialogue',{}).get('session_id'):
                records.append((path,r))
                if len(records)==3:break
        records.reverse()
    if len(records)!=3:raise ValueError('Need three actual browser turns')
    sid=records[0][1]['dialogue']['session_id']
    assert all(r['dialogue']['session_id']==sid for _,r in records)
    folder=ROOT/'reports/persistent-session';previous={};summary=[]
    latest={}
    for n,(path,r) in enumerate(records,1):
        dump(folder/f'browser-turn-{n}.json',r)
        row={'record':str(path.relative_to(ROOT)),'reply':r['reply'],'llm_called':r['dialogue']['llm_called'],
             'observations':[]}
        for o in r['observations']:
            f=o['embodied_feedback'];body=o['scene']['body'];dataset=o['dataset']
            old=previous.get(dataset)
            if old:
                assert f['session_start_ms']==old['embodied_feedback']['session_end_ms']
                assert f['initial_neural_state_sha256']==old['embodied_feedback']['neural_state_sha256']
                assert f['neural_reset_count']==f['body_reset_count']==0
                np.testing.assert_array_equal(body['trajectory'][0]['position_mm'],old['scene']['body']['trajectory'][-1]['position_mm'])
            assert body['summary']['finite_state'] and body['summary']['physics_warnings']==0
            previous[dataset]=latest[dataset]=o
            row['observations'].append({'id':o['id'],'dataset':dataset,'start_ms':f['session_start_ms'],
                'end_ms':f['session_end_ms'],'resets':f['neural_reset_count'],'spikes':o['spikes'],
                'active_neurons':o['active_neurons'],'position_continues_previous':bool(old),
                'body_summary':body['summary'],'outward_expression':o['expression']['status']})
        summary.append(row)
    replayed=[]
    for dataset,obs in latest.items():
        if dataset.startswith('FlyWire'):
            from chatflypt.brain import Brain
            brain=Brain()
        else:
            from chatflypt.malecns import MaleCNS
            brain=MaleCNS()
        p=ROOT/obs['embodied_feedback']['output_dir']
        with np.load(p/'input-events.npz',allow_pickle=False) as z:
            events={k:z[k].item() if z[k].ndim==0 else z[k] for k in z.files}
        trial=replay_events(brain,events)
        with np.load(p/'cumulative-neural.npz',allow_pickle=False) as expected:
            for name in ('counts','spike_ids','spike_steps','state_record'):
                np.testing.assert_array_equal(getattr(trial,name),expected[name])
        replayed.append({'dataset':dataset,'duration_ms':trial.duration_ms,'bitwise_equal':True})
        del brain,trial
    result={'date':'2026-09-24','session_id':sid,'turns':summary,'cumulative_input_replay':replayed,
        'neural_and_physical_continuity_verified':True,'full_free_chat_complete':False}
    dump(folder/'browser-acceptance.json',result)
    print(json.dumps({'turns':len(summary),'replayed':replayed,'continuity':True},ensure_ascii=False))


if __name__=='__main__':main()
