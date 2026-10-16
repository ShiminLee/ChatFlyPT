"""Non-neural scene inspection: rigid-body mechanics and rendered camera data.

No connectome, sensory encoder, decoder, language model, or action policy is used.
The measured window is passive after the existing fixed-posture body warmup.
"""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def validate_request(environment, seed, context, duration_ms):
    from chatflypt.environment import validate_objects
    if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**32:
        raise ValueError('随机种子必须为非负 32 位整数')
    if context not in ('ground','tethered_flight'):
        raise ValueError('未知身体环境')
    if isinstance(duration_ms,bool) or not isinstance(duration_ms,(int,float)) or not math.isfinite(duration_ms) or not 100<=duration_ms<=1000 or duration_ms%20:
        raise ValueError('观察时间须为 100–1000 ms，且是 20 ms 的整数倍')
    if not isinstance(environment,dict):
        raise ValueError('环境必须是对象')
    revision=environment.get('revision')
    if not isinstance(revision,int) or isinstance(revision,bool) or revision<0:
        raise ValueError('环境版本无效')
    return {'schema':'chatflypt.environment.v1','revision':revision,
            'objects':validate_objects(environment.get('objects'))}


def inspect_environment(environment, seed=42, context='ground', duration_ms=500):
    environment=validate_request(environment,seed,context,duration_ms)
    python=ROOT/'.venv-body'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    request={'environment':environment,'seed':seed,'context':context,'duration_ms':duration_ms}
    try:
        done=subprocess.run([str(python),str(Path(__file__).resolve()),'--worker'],
            input=json.dumps(request),capture_output=True,text=True,encoding='utf8',timeout=90,cwd=ROOT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('物理观察超时，环境没有应用') from exc
    if done.returncode:
        raise ValueError('物理观察失败：'+done.stderr[-600:])
    result=json.loads(done.stdout)
    if result.get('status')!='ok':
        raise ValueError('物理观察失败：'+result.get('reason','unknown'))
    return result


def contact_events(frames):
    """Transitions at the sampled times, not invented exact collision times."""
    previous={};events=[]
    for frame in frames:
        current={}
        for contact in frame['environment_contacts']:
            key=(contact['object_id'],contact['fly_geom'])
            if key not in current or contact['distance_mm']<current[key]['distance_mm']:
                current[key]=contact
        for key in sorted(current.keys()-previous.keys()):
            events.append(dict(current[key],kind='contact_started',time_ms=frame['time_ms']))
        for key in sorted(previous.keys()-current.keys()):
            events.append({'object_id':key[0],'fly_geom':key[1],'kind':'contact_ended','time_ms':frame['time_ms']})
        previous=current
    return events


def environment_changes(before,after):
    old={o['id']:o for o in before['objects']};new={o['id']:o for o in after['objects']}
    changes=[]
    for oid in sorted(new.keys()-old.keys()):changes.append({'kind':'added','object':new[oid]})
    for oid in sorted(old.keys()-new.keys()):changes.append({'kind':'removed','object':old[oid]})
    for oid in sorted(new.keys()&old.keys()):
        fields=[key for key in new[oid] if new[oid][key]!=old[oid].get(key)]
        if fields:changes.append({'kind':'updated','object':new[oid],'previous':old[oid],'fields':fields})
    return changes


def _simulate(request):
    import numpy as np
    import mujoco
    import base64
    import io
    from PIL import Image
    from chatflypt.behavior_body import initialize_body, GEOMETRY_ASSET
    from chatflypt.environment import measure_environment_contacts, measure_vision

    env=validate_request(request['environment'],request['seed'],request['context'],request['duration_ms'])
    body,initialization=initialize_body(request['seed'],request['context'],env,inspection_contacts=True)
    initialization['inspection_full_body_ground_contacts']=True
    sim,fly=body.sim,body.fly;model,data=sim.mj_model,sim.mj_data
    if sum(int(w.number) for w in data.warning):raise ValueError('初始化出现物理数值警告')
    # Disable every actuator in the reported window. Gravity, joints, friction,
    # collision constraints and an explicitly selected tether are the only forces.
    model.actuator_gainprm[:]=0;model.actuator_biasprm[:]=0;data.ctrl[:]=0
    names=[segment.name for segment in fly.get_bodysegs_order()]
    thorax=names.index('c_thorax');feet=[names.index(f'{leg}_tarsus5') for leg in ('lf','lm','lh','rf','rm','rh')]
    timestep_ms=float(sim.timestep)*1000;steps=int(round(request['duration_ms']/timestep_ms))
    sample_steps=max(1,int(round(20/timestep_ms)));elapsed=0.
    max_force=0.

    def frame():
        mujoco.mj_forward(model,data)
        positions=sim.get_body_positions(fly.name);rotations=sim.get_body_rotations(fly.name)
        return {'time_ms':round(elapsed,7),'position_mm':positions[thorax].tolist(),
            'quaternion_wxyz':rotations[thorax].tolist(),'body_positions_mm':positions.tolist(),
            'body_quaternions_wxyz':rotations.tolist(),'feet_mm':positions[feet].tolist(),
            'joint_angles_rad':sim.get_joint_angles(fly.name).tolist(),
            'geom_positions_mm':data.geom_xpos[body.geometry_indices].tolist(),
            'geom_rotation_matrices':data.geom_xmat[body.geometry_indices].tolist(),
            'environment_contacts':measure_environment_contacts(model,data),
            'contact_count':int(data.ncon),'actuator_force_max':float(np.max(np.abs(data.actuator_force),initial=0)),
            'neural_command_active':False}

    def eyes():
        vision=measure_vision(sim,fly.name,env,names,sim.get_body_positions(fly.name))
        vision.update(bridge='none; camera inspection only',neural_input_generated=False)
        mosaic={}
        for i,side in enumerate(('left','right')):
            values=np.sum(np.asarray(vision['ommatidia'][i]),axis=-1)
            pixels=sim.retina.hex_pxls_to_human_readable(values,color_8bit=True)
            output=io.BytesIO();Image.fromarray(pixels).convert('RGB').resize((240,240)).save(output,format='PNG')
            mosaic[side]='data:image/png;base64,'+base64.b64encode(output.getvalue()).decode('ascii')
        vision['mosaic_images']=mosaic
        return {'time_ms':round(elapsed,7),'vision':vision}

    frames=[frame()];camera_samples=[eyes()]
    middle=max(1,steps//2)
    for step in range(1,steps+1):
        data.ctrl[:]=0;sim.step();elapsed=step*timestep_ms
        if request['context']=='tethered_flight':
            data.qpos[:7]=[0,0,2,1,0,0,0];data.qvel[:6]=0
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError('物理状态非有限数值')
        max_force=max(max_force,float(np.max(np.abs(data.actuator_force),initial=0)))
        if step%sample_steps==0 or step==steps:frames.append(frame())
        if step in (middle,steps):
            mujoco.mj_forward(model,data);camera_samples.append(eyes())
    warnings=int(sum(int(w.number) for w in data.warning))
    if warnings or max_force!=0:raise ValueError('被动物理观察未通过数值/执行器检查')
    events=contact_events(frames);positions=np.asarray([f['position_mm'] for f in frames])
    summary={'duration_ms':request['duration_ms'],'frame_count':len(frames),'sample_ms':20,
        'minimum_thorax_height_mm':float(positions[:,2].min()),
        'displacement_mm':(positions[-1]-positions[0]).tolist(),
        'path_length_mm':float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
        'contact_pairs':len({(c['object_id'],c['fly_geom']) for f in frames for c in f['environment_contacts']}),
        'contact_events':len(events),'physics_warnings':warnings,'max_actuator_force_model_units':max_force,
        'finite_state':True}
    record={'schema':'chatflypt.environment-inspection.v1','id':uuid.uuid4().hex,'status':'ok',
        'created_at':datetime.now(timezone.utc).isoformat(),'environment':env,'seed':request['seed'],
        'neural_simulation_run':False,'neural_input_generated':False,'llm_called':False,
        'body':{'schema':'chatflypt.body.v1','status':'passive_control','physics_run':True,
            'duration_ms':request['duration_ms'],'context':request['context'],'trajectory':frames,
            'geometry_asset':GEOMETRY_ASSET,'body_names':names,'controller_id':'passive_environment_inspector',
            'initialization':initialization,'environment':env,'summary':summary,
            'controls':{'all_actuators_disabled_during_replay':True,'neural_feedback':False,
                        'kinematic_tether':request['context']=='tethered_flight'}},
        'summary':summary,'events':events,'camera_samples':camera_samples,
        'limitations':['只记录机械接触、位移与渲染画面，不推断感觉、意图或表达。',
            '记录窗口前使用原固定姿态预热；窗口内关闭执行器，身体可能自然下落或倒伏。',
            '接触事件按 20 ms 采样，可能遗漏采样间的短暂接触；力保留模型单位。',
            '画面为场景相机与官方复眼采样，没有发送到神经网络。']}
    record['text']=f"物理观察 {request['duration_ms']:g} ms：记录到 {summary['contact_pairs']} 组物品接触，身体路径长度 {summary['path_length_mm']:.3f} mm；已保存 3 组双眼画面。"
    forbidden={'chatflypt.brain','chatflypt.malecns','chatflypt.sensory_events','chatflypt.conversation',
               'chatflypt.experiment','chatflypt.language','chatflypt.behavior_readout'}
    loaded=sorted(forbidden.intersection(sys.modules))
    if loaded:raise RuntimeError('物理观察意外加载了神经/语言模块：'+', '.join(loaded))
    record['isolation']={'neural_or_language_modules_loaded':loaded}
    return record


if __name__=='__main__':
    sys.path.insert(0,str(ROOT));sys.stdin.reconfigure(encoding='utf8');sys.stdout.reconfigure(encoding='utf8')
    if sys.argv[1:]!=['--worker']:raise SystemExit('Internal physics inspector')
    try:print(json.dumps(_simulate(json.load(sys.stdin)),ensure_ascii=False,allow_nan=False))
    except Exception as exc:print(json.dumps({'status':'error','reason':str(exc),'error_type':type(exc).__name__},ensure_ascii=False))
