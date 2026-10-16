"""Scene edits preserve a moving body's state and change real eye/contact data."""
import json
import os
import subprocess
from chatflypt.paths import ROOT


def test_real_scene_edits_keep_body_controller_and_clock(tmp_path):
    code=r'''
import hashlib,json,pickle
import numpy as np
from chatflypt.body_feedback import BodySession
from chatflypt.environment import validate_objects
from chatflypt.behavior_body import COMMAND_KEYS
empty={'revision':0,'objects':[]}
b=BodySession({'seed':42,'context':'ground','environment':empty})
b.body.advance(20,{k:0. for k in COMMAND_KEYS})
before=b.probe();controller=b.body.controller
controller_state=hashlib.sha256(pickle.dumps(controller)).hexdigest()
q=b.data.qpos.copy();v=b.data.qvel.copy();clock=b.data.time;elapsed=b.body.elapsed
objects=validate_objects([{'id':'new_food','type':'sugar','position_mm':[4,-2,.5],'size_mm':[1,1,1]}])
records=[]
for revision,items in [(1,objects),(2,[]),(3,objects)]:
    r=b.update_environment({'revision':revision,'objects':items})
    np.testing.assert_array_equal(q,b.data.qpos);np.testing.assert_array_equal(v,b.data.qvel)
    assert b.data.time==clock and b.body.elapsed==elapsed
    assert b.body.controller is controller
    assert hashlib.sha256(pickle.dumps(controller)).hexdigest()==controller_state
    assert r['integration_state_before_sha256']==r['integration_state_after_sha256']
    if items:assert r['probe']['vision']['ommatidia']!=before['vision']['ommatidia']
    else:np.testing.assert_array_equal(r['probe']['vision']['ommatidia'],before['vision']['ommatidia'])
    records.append({k:v for k,v in r.items() if k!='probe'})
platform=validate_objects([{'id':'surface','type':'block','position_mm':[0,0,.1],'size_mm':[5,4,.2]}])
r=b.update_environment({'revision':4,'objects':platform})
assert any(c['object_id']=='surface' for c in r['probe']['environment_contacts'])
np.testing.assert_array_equal(q,b.data.qpos)
b.body.advance(10,{k:0. for k in COMMAND_KEYS});b.check()
print(json.dumps({'records':records,'real_contact':True,'finite_after_edit':True}))
'''
    python=ROOT/'.venv-body'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    done=subprocess.run([str(python),'-X','utf8','-c',code],cwd=ROOT,
                        capture_output=True,text=True,encoding='utf8',timeout=120)
    assert done.returncode==0,done.stderr
    report=json.loads(done.stdout)
    from chatflypt.paths import dump
    dump(ROOT/'reports/unified-pathway-20260924/live-environment.json',report)
