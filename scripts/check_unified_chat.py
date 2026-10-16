"""Exercise real HTTP chat and scene edits without changing the saved user scene."""
import json
import argparse
from pathlib import Path
import sys
import threading
import urllib.request
import uuid
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chatflypt.server import Application, create_server
from chatflypt.environment import empty_environment
from chatflypt.paths import ROOT, dump, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decoder',type=Path,help='Evaluate a frozen candidate without publishing it')
    args=parser.parse_args()
    saved=ROOT/'data/local/environment.json'
    before=sha256(saved) if saved.is_file() else None
    app=Application();app.environment=empty_environment()
    if args.decoder:
        from chatflypt.unified_readout import load_unified_decoder
        decoder,note=load_unified_decoder(app.male,app.input_catalogue(),args.decoder)
        assert decoder is not None,note
        app._environment_decoder=lambda brain,catalogue:(decoder,None)
    # Execute the real commit protocol, but publish this test scene in memory.
    app._publish_environment=lambda candidate:setattr(app,'environment',candidate)
    server=create_server(0,app)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_address[1])
    sid=uuid.uuid4().hex;rows=[];end_state=None;end_ms=0.
    extra_sid=None
    def comparison(observation):
        # Post-hoc evaluation only; this evidence never enters the decoder.
        from train_unified_readout import measured_targets,CHANNELS
        with np.load(ROOT/observation['event_record']['file'],allow_pickle=False) as events:
            expected=measured_targets(observation['embodied_feedback']['samples'],events,
                                      app.input_catalogue(),CHANNELS)
        predicted=set(observation['decoded']['words'])
        return {'actual_receptor_labels':sorted(expected),'predicted':sorted(predicted),
                'false_positive':sorted(predicted-expected),'missed':sorted(expected-predicted),
                'source':'offline comparison only; not a decoder input'}
    def post(path,payload):
        request=urllib.request.Request(base+path,data=json.dumps(payload).encode('utf8'),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=600) as response:
            return json.load(response)
    try:
        with urllib.request.urlopen(base+'/api/status',timeout=30) as response:
            status=json.load(response)
        assert status['dataset']=='MaleCNS v1.0' and status['neurons']==166700
        assert status['single_individual'] and not status['llm_in_pathway']
        messages=('看看环境','右边有糖','左转','看看环境')
        for index,message in enumerate(messages):
            result=post('/api/converse',{'message':message,'session_id':sid,'seed':924720+index})
            assert result['pathway']['input_mode']=='environment_only'
            assert result['pathway']['single_individual'] and len(result['observations'])==1
            assert result['input_plan']['items']==[]
            assert not result['dialogue']['llm_called'] and not result['dialogue']['external_action_control']
            observation=result['observations'][0]
            assert observation['dataset']=='MaleCNS v1.0'
            assert observation['readout_model']=='general_sensory_unified'
            feedback=observation['embodied_feedback'];body=observation['scene']['body']
            assert feedback['interval_ms']==10 and len(feedback['samples'])==50
            assert not feedback['direct_neural_intervention']
            assert feedback['session_start_ms']==end_ms
            if index:
                assert feedback['neural_reset_count']==feedback['body_reset_count']==0
                assert feedback['initial_neural_state_sha256']==end_state
            end_state=feedback['neural_state_sha256'];end_ms=feedback['session_end_ms']
            assert not body['summary']['physics_warnings'] and body['summary']['finite_state']
            assert observation['reception']['proprioception']['applied']
            assert all(s['visual_projection']['status']=='candidate_spatial_projection' for s in feedback['samples'])
            if message=='右边有糖':
                assert result['environment_event']['label']=='环境改动'
                assert feedback['environment_update']['state_preserved']
                assert not feedback['environment_update']['body_reset']
            # The neural output is not required to agree with the user's words.
            rows.append({'message':message,'record':result['dialogue_record'],
                'dataset':observation['dataset'],'readout_model':observation['readout_model'],
                'start_ms':feedback['session_start_ms'],'end_ms':end_ms,
                'neural_resets':feedback['neural_reset_count'],'body_resets':feedback['body_reset_count'],
                'environment_update':feedback['environment_update'],
                'input_events':observation['reception']['input_events'],
                'physical_presentation':observation['reception'].get('physical_presentation'),
                'decoded':observation['decoded']['words'],'reply':result['reply'],
                'offline_evaluation':comparison(observation),
                'body_summary':body['summary'],'body_commands':body['command_summary']})
            print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
        # The editor endpoint must retain exactly the same live individual too.
        result=post('/api/environment/apply',{'objects':[], 'revision':app.environment['revision'],
            'seed':924724,'session_id':sid,'body_context':'ground'})
        f=result['observations'][0]['embodied_feedback']
        assert f['session_start_ms']==end_ms and f['initial_neural_state_sha256']==end_state
        assert f['environment_update']['state_preserved']
        assert f['neural_reset_count']==f['body_reset_count']==0
        rows.append({'editor_removed_objects':True,'record':result['dialogue_record'],
            'environment_update':f['environment_update'],'start_ms':f['session_start_ms'],
            'end_ms':f['session_end_ms'],'reply':result['reply']})
        # A fresh individual in the user's unchanged, real eight-object scene.
        # This combination is outside the training protocol; retain failures.
        if saved.is_file():
            scene=json.loads(saved.read_text(encoding='utf8'))
            app.environment={k:scene[k] for k in ('schema','revision','objects')}
            extra_sid=uuid.uuid4().hex
            result=post('/api/converse',{'message':'看看环境','session_id':extra_sid,'seed':924730})
            observation=result['observations'][0]
            feedback=observation['embodied_feedback'];body=observation['scene']['body']
            assert len(result['observations'])==1 and result['pathway']['single_individual']
            assert observation['readout_model']=='general_sensory_unified'
            assert feedback['session_start_ms']==0 and not feedback['direct_neural_intervention']
            assert not body['summary']['physics_warnings'] and body['summary']['finite_state']
            rows.append({'held_out_saved_scene':True,'objects':len(scene['objects']),
                'environment_revision':scene['revision'],'record':result['dialogue_record'],
                'reply':result['reply'],'offline_evaluation':comparison(observation),
                'body_summary':body['summary'],'body_commands':body['command_summary']})
            print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
        dump(ROOT/'reports/unified-pathway-20260924/chat-acceptance.json',
             {'single_individual':True,'real_http':True,'saved_environment_unchanged':True,
              'llm_called':False,'test_turns':rows,'biologically_calibrated':False})
    finally:
        app.reset_session(sid)
        if extra_sid:app.reset_session(extra_sid)
        server.shutdown();server.server_close();thread.join(timeout=5)
        assert (sha256(saved) if saved.is_file() else None)==before


if __name__=='__main__':main()
