"""Scene editing and passive mechanics only; no neural trials or LLM calls."""
import copy
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from chatflypt import environment_inspection as inspection
from chatflypt.environment import empty_environment, validate_objects
from chatflypt import server


def item(oid='platform', kind='block', position=(0,0,.1), size=(5,4,.2)):
    return validate_objects([{'id':oid,'type':kind,'position_mm':list(position),'size_mm':list(size)}])[0]


@pytest.mark.parametrize('change',[
    {'seed':True},{'seed':-1},{'seed':2**32},{'context':'unknown'},
    {'duration_ms':99},{'duration_ms':101},{'duration_ms':float('nan')},
    {'duration_ms':1001},{'environment':{'revision':True,'objects':[]}},
    {'environment':{'revision':0,'objects':[item(),item()]}},
])
def test_invalid_request(change):
    args={'environment':empty_environment(),'seed':42,'context':'ground','duration_ms':100}
    args.update(change)
    with pytest.raises(ValueError):inspection.validate_request(**args)


def test_contact_transitions_use_observed_times_and_deduplicate():
    contact={'object_id':'platform','fly_geom':'fly/rf_tarsus5','distance_mm':-.01,'position_mm':[0,0,0]}
    frames=[{'time_ms':0,'environment_contacts':[]},
        {'time_ms':20,'environment_contacts':[contact,dict(contact,distance_mm=-.02)]},
        {'time_ms':40,'environment_contacts':[contact]}, {'time_ms':60,'environment_contacts':[]}]
    events=inspection.contact_events(frames)
    assert [(e['kind'],e['time_ms']) for e in events]==[('contact_started',20),('contact_ended',60)]
    assert events[0]['distance_mm']==-.02


def test_object_changes():
    before={'objects':[item('removed'),item('moved'),item('same')]}
    after=copy.deepcopy(before);after['objects'][0]=item('added');after['objects'][1]['yaw_deg']=90
    changes=inspection.environment_changes(before,after)
    assert [c['kind'] for c in changes]==['added','removed','updated']
    assert changes[-1]['fields']==['yaw_deg']
    assert changes[-1]['previous']['yaw_deg']==0


@pytest.fixture
def app(tmp_path,monkeypatch):
    result=server.Application.__new__(server.Application)
    result.environment=empty_environment();result.environment_lock=threading.RLock()
    monkeypatch.setattr(server,'ROOT',tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('Scene inspection must not run neural or language code')
    for method in ('converse','stimulate','observe','input_catalogue','chat'):
        setattr(result,method,forbidden)
    return result


@pytest.fixture
def stub(monkeypatch):
    calls=[]
    def inspect(env,seed=42,context='ground',duration_ms=500):
        inspection.validate_request(env,seed,context,duration_ms)
        calls.append(copy.deepcopy(env))
        return {'id':'a'*32,'environment':copy.deepcopy(env),'neural_simulation_run':False,'llm_called':False}
    monkeypatch.setattr(inspection,'inspect_environment',inspect)
    return calls


def test_inspection_commits_physics_record_without_neural_trial(app,stub,tmp_path):
    response=app.inspect_applied_environment()
    assert response['environment']['revision']==0 and len(stub)==1
    assert response['neural_simulation_run'] is False and response['llm_called'] is False
    assert json.loads((tmp_path/'data/local/environment.json').read_text())==app.environment
    assert app.environment_inspection('a'*32)==response['inspection']


def test_inspect_keeps_revision_and_objects(app,stub):
    app.environment={'revision':7,'objects':[item()]}
    result=app.inspect_applied_environment(duration_ms=100)
    assert result['environment']['revision']==7
    assert result['environment']['objects']==[item()]
    assert result['environment_event']['label']=='物理观察'


def test_stale_and_failed_apply_never_publish(app,stub,monkeypatch,tmp_path):
    with pytest.raises(ValueError,match='版本'):app.apply_environment([item()],2)
    assert not stub
    def failed(*args,**kwargs):raise ValueError('Physics failed')
    monkeypatch.setattr(inspection,'inspect_environment',failed)
    with pytest.raises(ValueError,match='Physics failed'):app.inspect_applied_environment()
    assert app.environment==empty_environment()
    assert not (tmp_path/'data/local/environment.json').exists()


def test_persistence_failure_never_publishes(app,stub,monkeypatch):
    def failed(*args,**kwargs):raise OSError('disk full')
    monkeypatch.setattr(server,'dump',failed)
    with pytest.raises(OSError):app.inspect_applied_environment()
    assert app.environment==empty_environment()


@pytest.mark.parametrize('value',['../data/local/environment','/etc/passwd','A'*32,'a'*31,'a'*32])
def test_record_lookup_rejects_invalid_or_missing_ids(app,value):
    with pytest.raises(ValueError):app.environment_inspection(value)


def test_http_inspection_and_record_routes(app,stub):
    httpd=server.create_server(0,app);thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(httpd.server_port)
    try:
        def post(path,body):
            req=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req) as r:return json.load(r)
        result=post('/api/environment/inspect',{'duration_ms':100})
        assert result['environment']['revision']==0
        with urllib.request.urlopen(base+'/api/environment/inspection?record='+'a'*32) as response:
            assert json.load(response)==result['inspection']
        with urllib.request.urlopen(base+'/api/environment/inspection?record='+'a'*32+'&download=1') as response:
            assert response.headers['Content-Disposition']=='attachment; filename="environment-'+('a'*32)+'.json"'
            assert json.load(response)==result['inspection']
        with pytest.raises(urllib.error.HTTPError) as exc:urllib.request.urlopen(base+'/api/environment/inspection?record=../x')
        assert exc.value.code==400
    finally:httpd.shutdown();httpd.server_close();thread.join()


def test_real_passive_platform_and_optics(tmp_path):
    environment={'revision':1,'objects':[item()]}
    record=inspection.inspect_environment(environment,duration_ms=100)
    assert record['status']=='ok' and record['isolation']['neural_or_language_modules_loaded']==[]
    assert record['neural_simulation_run'] is False and record['neural_input_generated'] is False
    assert record['llm_called'] is False
    summary=record['summary'];assert summary['physics_warnings']==0 and summary['finite_state']
    assert summary['max_actuator_force_model_units']==0 and summary['frame_count']==6
    assert summary['contact_pairs']>0 and record['events']
    assert summary['minimum_thorax_height_mm']>0
    frames=record['body']['trajectory']
    assert [f['time_ms'] for f in frames]==[0,20,40,60,80,100]
    assert all(np.isfinite(f['body_positions_mm']).all() and f['actuator_force_max']==0 for f in frames)
    assert [s['time_ms'] for s in record['camera_samples']]==[0,50,100]
    for sample in record['camera_samples']:
        vision=sample['vision'];assert np.asarray(vision['ommatidia']).shape==(2,721,2)
        assert vision['neural_input_generated'] is False
        for kind in ('eye_images','mosaic_images'):
            assert all(vision[kind][side].startswith('data:image/png;base64,') for side in ('left','right'))
    # Keep an actual physical acceptance record without changing the live scene.
    path=Path(__file__).resolve().parents[1]/'reports/environment-inspection/platform-acceptance.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,ensure_ascii=False),encoding='utf8')


def test_passive_body_cannot_sink_through_ground_after_falling():
    record=inspection.inspect_environment(empty_environment(),duration_ms=500)
    assert record['summary']['minimum_thorax_height_mm']>0
    assert record['summary']['physics_warnings']==0
    assert record['summary']['max_actuator_force_model_units']==0
    assert record['body']['initialization']['inspection_full_body_ground_contacts']
