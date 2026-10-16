"""Real physical acceptance for the environment-only message path."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from chatflypt.paths import ROOT, dump
from chatflypt.malecns import MaleCNS
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.environment import empty_environment,validate_objects
from chatflypt.embodied import run_embodied
from chatflypt.neural_session import replay_events

brain=MaleCNS();catalogue=build_catalogue(save=False)
output=ROOT/'reports/environment-only-20260924';output.mkdir(parents=True,exist_ok=True)
summaries=[]
for name,objects,presentations in [
    ('no_touch',[],[]),
    ('physical_right_touch',[],[{'kind':'touch','side':'right','body_part':'foreleg','strength':1.,'onset_ms':50.,'duration_ms':300.}]),
    ('distant_sugar',validate_objects([{'id':'sugar','type':'sugar','position_mm':[0,-4,.2],'size_mm':[.6,.6,.4]}]),[]),
]:
    env=empty_environment();env['objects']=objects
    result=run_embodied(brain,brain.rows,catalogue,'MaleCNS v1.0',[],env,492401,'ground',
        output_dir=output/name,input_mode='environment_only',presentations=presentations)
    feedback=result['feedback'];samples=feedback['samples'];body=result['body']
    inputs=[{'time_ms':s['time_ms'],'channels':[i['channel']+':'+i['side'] for i in s['items']],
             'contacts':[e for e in s['evidence'] if e.get('mode')=='mujoco_leg_contact']} for s in samples]
    replay=replay_events(brain,result['events'])
    for key in ('counts','spike_ids','spike_steps','state_record'):
        np.testing.assert_array_equal(getattr(result['trial'],key),getattr(replay,key))
    summary={'case':name,'input_mode':feedback['input_mode'],'direct_neural_intervention':feedback['direct_neural_intervention'],
        'events':len(result['events']['event_steps']),'recorded_spikes':int(result['trial'].counts.sum()),
        'samples':inputs,'body':body['summary'],'replay_bitwise_identical':True}
    summaries.append(summary);dump(output/'acceptance.json',summaries)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    assert not feedback['direct_neural_intervention']
    if name!='physical_right_touch':assert len(result['events']['event_steps'])==0
    else:assert any(i['channel']=='foreleg_touch' for s in samples for i in s['items'])
