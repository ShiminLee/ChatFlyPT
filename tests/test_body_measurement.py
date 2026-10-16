"""Real MuJoCo static-fusion / receptor-location regression, no neural run."""
import json
from pathlib import Path
import subprocess


def test_fused_body_recovery_optical_shade_and_unchanged_physics(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = r'''
import json,hashlib
import numpy as np,mujoco
from chatflypt.behavior_body import initialize_body,COMMAND_KEYS
from chatflypt.body_measurement import eye_camera_positions
from chatflypt.environment import measure_vision
from chatflypt.live_environment import update_body_environment
env={'objects':[],'revision':0}
b,_=initialize_body(environment=env);sim=b.sim;m,d=sim.mj_model,sim.mj_data
mujoco.mj_forward(m,d)
names=[s.name for s in b.fly.get_bodysegs_order()]
positions=sim.get_body_positions('fly');raw=sim._chatflypt_raw_body_positions('fly')
assert sim._internal_bodyids_by_fly['fly'][names.index('l_eye')]==-1
assert np.array_equal(raw[names.index('l_eye')],raw[names.index('rh_tarsus5')])
head=m.body('fly/c_head').id
expected=d.xpos[head]+d.xmat[head].reshape(3,3)@np.array([.231,-9.31e-10,.017])
np.testing.assert_allclose(positions[names.index('l_eye')],expected,atol=1e-12)
keep=np.array([i for i,n in enumerate(names) if n!='l_eye'])
np.testing.assert_array_equal(positions[keep],raw[keep])
eyes=eye_camera_positions(sim,'fly')
assert eyes['left'][1]>.3 and eyes['right'][1]<-.3
before=measure_vision(sim,'fly',env,names,positions)
shade={'id':'shade','type':'shade','position_mm':eyes['left'].tolist(),'size_mm':[.1,.1,.1],'yaw_deg':0,'strength':1,'range_mm':1}
shaded=measure_vision(sim,'fly',{'objects':[shade]},names,positions)
base=np.array(before['ommatidia']);sample=np.array(shaded['ommatidia'])
assert base[0].sum()>0 and base[1].sum()>0
assert sample[0].sum()==0
np.testing.assert_array_equal(sample[1],base[1])
new={'revision':1,'objects':[{'id':'block','type':'block','position_mm':[7,7,1],'size_mm':[1,1,1],'yaw_deg':0,'strength':1,'range_mm':1}]}
changed=update_body_environment(b,env,new)
assert changed['state_preserved']
np.testing.assert_array_equal(sim.get_body_positions('fly'),positions)
np.testing.assert_array_equal(eye_camera_positions(sim,'fly')['left'],eyes['left'])
# Compare entire qpos/qvel time series with original accessors. No physics or
# controller modification is allowed by this observation-layer repair.
traces=[]
for repaired in [False,True]:
 body,_=initialize_body(environment=env)
 if not repaired:
  body.sim.get_body_positions=body.sim._chatflypt_raw_body_positions
  body.sim.get_body_rotations=body.sim._chatflypt_raw_body_rotations
 trace=[]
 for step in range(1000):
  command={k:0. for k in COMMAND_KEYS}
  if 250<=step<500:command['escape']=.2
  body.advance(.1,command)
  trace.append(np.r_[body.sim.mj_data.qpos,body.sim.mj_data.qvel])
 traces.append(np.array(trace))
np.testing.assert_array_equal(*traces)
print(json.dumps({'l_eye_original_id':-1,'static_body_recovered':True,'other_frames_bitwise_unchanged':True,'eye_positions_mm':{k:v.tolist() for k,v in eyes.items()},'shade_only_left_eye':True,'scene_update_preserves_frames':True,'full_physical_qpos_qvel_bitwise_equal':True,'trajectory_sha256':hashlib.sha256(traces[0].tobytes()).hexdigest()}))
'''
    result = subprocess.run([str(root/'.venv-body/Scripts/python.exe'), '-c', script],
        cwd=root, capture_output=True, text=True, encoding='utf8', timeout=90)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data['shade_only_left_eye'] and data['full_physical_qpos_qvel_bitwise_equal']
    (root/'reports/unified-pathway-20260924/body-measurement-repair.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
