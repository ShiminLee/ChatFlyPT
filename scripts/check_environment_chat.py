"""Exercise real chat without publishing over the user's saved environment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import uuid
from chatflypt.server import Application
from chatflypt.environment import empty_environment
from chatflypt.paths import ROOT, dump, sha256

saved=ROOT/'data/local/environment.json';before=sha256(saved)
app=Application();app.environment=empty_environment()
# Same environment commit path and physical checks, but this test's scene is
# in memory. The real user scene file is never replaced or restored.
app._publish_environment=lambda candidate:setattr(app,'environment',candidate)
sid=uuid.uuid4().hex;reports=[]
try:
    for text in ('右边有糖','左转','看看环境'):
        result=app.converse(message=text,seed=492402,session_id=sid)
        assert result['pathway']['input_mode']=='environment_only'
        assert result['input_plan']['items']==[]
        rows=[]
        for obs in result['observations']:
            f=obs['embodied_feedback']
            assert not f['direct_neural_intervention']
            if text!='右边有糖':assert f['neural_reset_count']==f['body_reset_count']==0
            channels=sorted({i['channel']+':'+i['side'] for s in f['samples'] for i in s['items']})
            assert not any('sugar' in c or 'sweet' in c for c in channels)
            rows.append({'dataset':obs['dataset'],'channels':channels,'input_events':obs['reception']['input_events'],
                'physical_presentation':obs['reception'].get('physical_presentation'),
                'start_ms':f['session_start_ms'],'end_ms':f['session_end_ms'],
                'neural_resets':f['neural_reset_count'],'body_resets':f['body_reset_count'],
                'finite':obs['scene']['body']['summary']['finite_state']})
        report={'message':text,'dialogue_record':result['dialogue_record'],'environment_revision':app.environment['revision'],
                'environment_event':result.get('environment_event'),'observations':rows,'reply':result['reply']}
        reports.append(report);print(json.dumps(report,ensure_ascii=False),flush=True)
finally:
    app.reset_session(sid)
    assert sha256(saved)==before
dump(ROOT/'reports/environment-only-20260924/chat-acceptance.json',{'saved_environment_unchanged':True,'turns':reports})
