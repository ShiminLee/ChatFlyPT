"""Real recorded replay, independent closed-loop retest and a salt sign ablation.

No direct-MN activation is called a sensory experiment. The LB3d sign hypothesis
is isolated here, not silently substituted into the production connectome.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from threadpoolctl import threadpool_limits
from chatflypt.paths import ROOT,dump,sha256

REPORT=ROOT/'reports/motor-repair-20260924'
RAW=ROOT/'data/derived/motor-repair-20260924'
OLD=ROOT/'data/derived/sugar-pressure-20260924'


def closed_loop():
    import scripts.try_sugar_pressure as runner
    runner.REPORT=REPORT;runner.RAW=RAW/'closed-loop'
    plan=runner.protocol()
    chosen={'empty','mouth_sugar','mouth_high_salt','leg_sugar','strong_touch'}
    plan['cases']=[dict(c,messages=c['messages'][:1]) for c in plan['cases']
                   if c['dataset']=='MaleCNS v1.0' and c['condition'] in chosen]
    plan.update(controller_changed=True,controller_extension='anatomical tibia MN target adapter v1',
                comparison='Same seeds and scene geometry as archived baseline; neural parameters unchanged.')
    dump(REPORT/'protocol.json',plan)
    runner.run_dataset('malecns',plan,2)


def replay():
    from chatflypt.malecns import MaleCNS
    from chatflypt.behavior_readout import build_behavior_drive
    from chatflypt.behavior_body import run_behavior_body
    brain=MaleCNS();cases=[]
    for seed in (491701,491702):
        for condition in ('mouth_sugar','mouth_high_salt','leg_sugar'):
            source=OLD/f'malecns-{seed}-{condition}'/'turn-0'
            with np.load(source/'observation.npz') as a:
                t=SimpleNamespace(counts=a['counts'],spike_ids=a['spike_indices'],spike_steps=a['spike_steps'],
                                  seed=int(a['seed']),duration_ms=float(a['duration_ms']))
                drive=build_behavior_drive(t,brain.ids,brain.rows,dataset=brain.dataset,driven_indices=a['driven_indices'])
            old=json.loads((source/'body-result.json').read_text(encoding='utf8'))
            legacy=dict(drive)
            for k in ('joint_motor','motor_coverage','has_joint_command_in_window'):legacy.pop(k,None)
            for name,d in [('legacy',legacy),('tibia',drive)]:
                cases.append((seed,condition,name,d,old['environment']))
    def run(case):
        seed,condition,name,drive,env=case;folder=RAW/'replay'/f'{seed}-{condition}'/name
        path=folder/'body-result.json'
        if path.exists():r=json.loads(path.read_text(encoding='utf8'))
        else:r=run_behavior_body(drive,output_dir=folder,environment=env,timeout=120)
        assert r['status']=='ok' and r['summary']['finite_state'] and r['summary']['physics_warnings']==0,r
        print(f'Replay {seed} {condition} {name}: {r["summary"]["path_length_mm"]:.6f} mm',flush=True)
        return {'seed':seed,'condition':condition,'controller':name,'summary':r['summary'],
                'record':str(path.relative_to(ROOT)),'source_spikes_sha256':drive['source_spikes_sha256']}
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(run,cases))
    dump(REPORT/'paired-replay.json',records)


def salt_sign():
    from chatflypt.malecns import MaleCNS,TRANSMITTERS
    import pandas as pd
    from chatflypt.neural_session import replay_events
    from chatflypt.behavior_readout import build_behavior_drive
    from chatflypt.behavior_body import run_behavior_body
    brain=MaleCNS()
    nt=pd.read_feather(TRANSMITTERS).set_index('body')
    selected=np.flatnonzero(brain.rows.type.eq('LB3d').to_numpy())
    assert len(selected)==26 and nt.reindex(brain.ids[selected]).consensus_nt.eq('acetylcholine').all()
    official=brain.weights;alternative=brain.weights.copy();edges=0
    for i in selected:
        a,b=brain.indptr[i:i+2];assert np.all(alternative[a:b]>0)
        alternative[a:b]*=-1;edges+=b-a
    source='https://www.sciencedirect.com/science/article/abs/pii/S0092867426009438'
    records=[]
    # Full archived input tape held fixed: compare network assumptions, not a
    # controller that inspects the salt label or imposes a rejection response.
    for seed in (491701,491702):
        for condition in ('mouth_sugar','mouth_high_salt'):
            original=OLD/f'malecns-{seed}-{condition}'/'turn-0'
            events=dict(np.load(original/'input-events.npz',allow_pickle=False))
            old=json.loads((original/'body-result.json').read_text(encoding='utf8'))
            for profile,weights in [('official_prediction',official),('LB3d_inhibitory_hypothesis',alternative)]:
                folder=RAW/'salt-sign'/f'{seed}-{condition}-{profile}';folder.mkdir(parents=True,exist_ok=True)
                brain.weights=weights
                t=replay_events(brain,events)
                if profile=='official_prediction':
                    with np.load(original/'observation.npz') as recorded:
                        np.testing.assert_array_equal(t.counts,recorded['counts'])
                        np.testing.assert_array_equal(t.spike_ids,recorded['spike_indices'])
                        np.testing.assert_array_equal(t.spike_steps,recorded['spike_steps'])
                driven=[brain.lookup[str(int(i))] for i in events['input_neuron_ids']]
                drive=build_behavior_drive(t,brain.ids,brain.rows,dataset=brain.dataset,driven_indices=driven)
                body=run_behavior_body(drive,environment=old['environment'],output_dir=folder,timeout=120)
                assert body['status']=='ok' and body['summary']['finite_state'] and not body['summary']['physics_warnings'],body
                np.savez_compressed(folder/'neural.npz',counts=t.counts,spike_ids=t.spike_ids,spike_steps=t.spike_steps,state_record=t.state_record)
                mn=np.flatnonzero(brain.rows.type.eq('MN9').to_numpy())
                record={'seed':seed,'condition':condition,'profile':profile,'MN9_spikes':int(t.counts[mn].sum()),
                    'feeding_max':body['command_summary']['feeding']['max'],'body_summary':body['summary'],
                    'neural_record':str((folder/'neural.npz').relative_to(ROOT)),
                    'body_record':str((folder/'body-result.json').relative_to(ROOT)),
                    'events_sha256':sha256(original/'input-events.npz')}
                records.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
    dump(REPORT/'salt-transmitter-ablation.json',{'source':source,'default_changed':False,
        'scope':'Counterfactual replay of identical archived input events, not independent biological validation or closed-loop equivalent.',
        'changed_presynaptic_cells':[str(brain.ids[i]) for i in selected],'changed_edges':int(edges),
        'reference':'Cell 2026 Fig 2D/E, putative population-level glutamatergic identity; raw EM classifier predicts ACh.',
        'records':records})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['closed-loop','replay','salt-sign']);args=parser.parse_args()
    with threadpool_limits(limits=1):
        {'closed-loop':closed_loop,'replay':replay,'salt-sign':salt_sign}[args.mode]()
