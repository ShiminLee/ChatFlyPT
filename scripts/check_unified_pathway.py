"""Real single-MaleCNS feedback and geometry-edit continuity acceptance."""
import json
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from chatflypt.paths import ROOT, dump
from chatflypt.embodied import EmbodiedSession, run_embodied
from chatflypt.malecns import MaleCNS
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.environment import validate_objects
from chatflypt.neural_session import replay_events
from chatflypt.unified_readout import model_contract


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/unified-pathway-20260924/final-validation')
    folder=parser.parse_args().output
    brain=MaleCNS();catalogue=build_catalogue(save=False)
    contract=model_contract(brain,catalogue)
    environment={'schema':'chatflypt.environment.v1','revision':0,'objects':[]}
    session=EmbodiedSession(environment,'ground')
    reports=[];started=time.perf_counter()
    try:
        for step in range(3):
            if step==1:
                environment={'schema':'chatflypt.environment.v1','revision':1,'objects':validate_objects([
                    {'id':'far_food','type':'sugar','position_mm':[4,-2,.5],'size_mm':[1,1,1]}])}
            if step==2:environment={'schema':'chatflypt.environment.v1','revision':2,'objects':[]}
            result=run_embodied(brain,brain.rows,catalogue,'MaleCNS v1.0',[],environment,
                924980,'ground',duration_ms=100.,session=session,input_mode='environment_only',
                output_dir=folder/f'turn-{step}')
            feedback=result['feedback'];samples=feedback['samples']
            assert feedback['interval_ms']==10 and len(samples)==10
            assert feedback['session_start_ms']==step*100
            assert feedback['neural_reset_count']==feedback['body_reset_count']==int(step==0)
            assert all(s['proprioception']['neural_feedback_applied'] for s in samples)
            assert all(catalogue['channels'][i['channel']]['dataset']=='MaleCNS v1.0' for s in samples for i in s['items'])
            assert not any(i['channel'] in ('sugar_specific','leg_sweet_local') for s in samples for i in s['items'])
            assert not result['body']['summary']['physics_warnings']
            if step:
                assert feedback['initial_neural_state_sha256']==reports[-1]['neural_state_sha256']
                update=feedback['environment_update']
                assert update['state_preserved'] and not update['body_reset']
                assert update['integration_state_before_sha256']==update['integration_state_after_sha256']
            row={'turn':step,'revision':environment['revision'],'session_start_ms':feedback['session_start_ms'],
                'session_end_ms':feedback['session_end_ms'],'input_events':len(result['events']['event_steps']),
                'recorded_spikes':int(result['trial'].counts.sum()),
                'neural_state_sha256':feedback['neural_state_sha256'],
                'proprioception_events':sum(s['proprioception']['encoded_event_count'] for s in samples),
                'environment_update':feedback['environment_update'],
                'channels':sorted({i['channel'] for s in samples for i in s['items']}),
                'body_summary':result['body']['summary']}
            reports.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        with np.load(folder/'turn-2/input-events.npz',allow_pickle=False) as z:
            events={k:z[k].item() if z[k].ndim==0 else z[k] for k in z.files}
        replay=replay_events(brain,events)
        for name in ('counts','spike_ids','spike_steps','state_record'):
            np.testing.assert_array_equal(getattr(replay,name),getattr(session.neural.trial(),name))
        assert model_contract(brain,catalogue)==contract, 'Core model changed during acceptance'
        report={'single_individual':True,'dataset':'MaleCNS v1.0','turns':reports,
            'model_contract':contract,
            'full_prefix_replay_bitwise_identical':True,'environment_updates_preserve_state':True,
            'elapsed_seconds':time.perf_counter()-started,'biologically_calibrated':False}
        dump(folder/'acceptance.json',report)
        print('Unified physical feedback and replay passed',flush=True)
    finally:session.close()


if __name__=='__main__':main()
