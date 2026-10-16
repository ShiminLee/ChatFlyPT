"""Verify the saved real-feedback corpus and record its provenance."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from chatflypt.paths import ROOT,dump,sha256


def main():
    report=ROOT/'reports/feedback-readout'
    raw=ROOT/'data/derived/feedback-readout'
    protocol=report/'protocol.json'
    plan=json.loads(protocol.read_text(encoding='utf8'))
    records=[];continuations=0;sample_count=0
    for episode in plan['episodes']:
        complete=json.loads((raw/episode['id']/'complete.json').read_text(encoding='utf8'))
        assert complete['protocol_sha256']==sha256(protocol)
        assert len(complete['records'])==len(episode['turns'])
        previous=None
        for row in complete['records']:
            folder=(ROOT/row['record']).parent
            feedback=json.loads((folder/'feedback.json').read_text(encoding='utf8'))
            body=json.loads((folder/'body-result.json').read_text(encoding='utf8'))
            assert sha256(ROOT/row['record'])==row['record_sha256']
            assert sha256(ROOT/row['events'])==row['events_sha256']
            assert feedback['interval_ms']==100. and feedback['sample_count']==len(feedback['samples'])==5
            assert feedback['session_end_ms']-feedback['session_start_ms']==500.
            assert body['physics_run'] and body['status']=='ok'
            assert body['summary']['finite_state'] and not body['summary']['physics_warnings']
            assert not feedback['llm_called'] and not feedback['goal_controller'] and not feedback['human_cue_repeated']
            for field,file in [('initial_neural_state_sha256','initial-neural-state.npz'),
                               ('neural_state_sha256','neural-state.npz'),
                               ('physical_state_sha256','physical-state.npz'),
                               ('input_events_sha256','input-events.npz')]:
                assert feedback[field]==sha256(folder/file)
            if previous is not None:
                continuations+=1
                assert feedback['neural_reset_count']==feedback['body_reset_count']==0
                assert feedback['initial_neural_state_sha256']==previous['neural_state_sha256']
                assert feedback['previous_checkpoint']['physical_state_sha256']==previous['physical_state_sha256']
                assert feedback['session_start_ms']==previous['session_end_ms']
            previous=feedback;sample_count+=len(feedback['samples'])
            records.append({'episode':row['episode'],'turn':row['turn_index'],'split':row['split'],
                'observation':row['record'],'sha256':row['record_sha256'],
                'feedback_sha256':sha256(folder/'feedback.json'),'body_sha256':sha256(folder/'body-result.json')})
    assert len(records)==48 and continuations==30 and sample_count==240
    source_names=['scripts/benchmark_feedback_readout.py','scripts/fit_feedback_readout.py',
        'scripts/audit_feedback_readout.py','chatflypt/feedback_readout.py','chatflypt/embodied.py',
        'chatflypt/neural_session.py','chatflypt/body_feedback.py','chatflypt/brain.py',
        'chatflypt/malecns.py','chatflypt/environment.py','chatflypt/behavior_readout.py']
    audit={'protocol_sha256':sha256(protocol),'episodes':len(plan['episodes']),'windows':len(records),
        'feedback_samples':sample_count,'continuations_without_reset':continuations,
        'simulation_ms':len(records)*500,'all_physics_finite':True,'physics_warnings':0,
        'state_and_input_hashes_verified':True,'llm_called':False,'goal_controller':False,
        'sources_at_audit':{p:sha256(ROOT/p) for p in source_names},'records':records,
        'note':'Source hashes captured after collection. Initial failed FlyWire ID-loading attempt was fixed and rerun; no failed window is included.'}
    dump(report/'corpus-audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('sources_at_audit','records')},ensure_ascii=False))


if __name__=='__main__':main()
