"""A near-contact record in MuJoCo must not become a taste or touch input."""
import json
import os
import subprocess

from chatflypt.paths import ROOT


def test_real_mujoco_margin_and_gap_do_not_create_phantom_reception():
    code=r'''
import json
import mujoco
from chatflypt.environment import measure_environment_contacts

xml="""<mujoco><worldbody>
<geom name="env_food" type="plane" size="1 1 .1"/>
<body pos="0 0 .15"><freejoint/>
<geom name="fly/rf_tarsus5" type="sphere" size=".1" margin=".2" gap=".1"/>
</body></worldbody></mujoco>"""
model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
cases=[]
for z in (.35,.15,.05):
    data.qpos[2]=z;mujoco.mj_forward(model,data)
    raw=[{'dist':float(c.dist),'active':int(c.efc_address)>=0} for c in data.contact]
    measured=measure_environment_contacts(model,data)
    cases.append({'z':z,'raw':raw,'measured':measured})
    if z>.1:assert not measured
    else:
        assert measured and measured[0]['surface_contact']
        assert measured[0]['normal_force_model_units']>0
        assert not measured[0]['force_calibrated']
assert any(c['active'] for row in cases[:2] for c in row['raw'])
assert any(not c['active'] for row in cases[:2] for c in row['raw'])
print(json.dumps({'mujoco_version':mujoco.__version__,'cases':cases}))
'''
    python=ROOT/'.venv-body'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    done=subprocess.run([str(python),'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stderr
    report=json.loads(done.stdout)
    from chatflypt.paths import dump
    dump(ROOT/'reports/reality-audit-20260924/contact-physics.json',report)
