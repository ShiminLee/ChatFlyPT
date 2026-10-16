"""Persistent isolated MuJoCo worker, advanced only by recorded neural drives."""
from pathlib import Path
import json
import os
import queue
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]


class FeedbackBody:
    def __init__(self, environment, seed, context, output_dir):
        python=ROOT/'.venv-body'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        self.inbox=queue.Queue();self.process=None
        self.output_dir=Path(output_dir);self.output_dir.mkdir(parents=True,exist_ok=True)
        self.stderr=(self.output_dir/'body-worker.log').open('w',encoding='utf8')
        try:
            self.process=subprocess.Popen([str(python),str(Path(__file__).resolve()),'--worker'],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,encoding='utf8',
                cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            def read():
                for line in self.process.stdout:self.inbox.put(line)
                self.inbox.put(None)
            self.reader=threading.Thread(target=read,daemon=True);self.reader.start()
            self.initial=self.call({'op':'initialize','environment':environment,'seed':seed,'context':context})
        except Exception:
            self.close();raise

    def call(self, request):
        self.process.stdin.write(json.dumps(request,ensure_ascii=False)+'\n');self.process.stdin.flush()
        try:line=self.inbox.get(timeout=120)
        except queue.Empty:raise RuntimeError('连续物理工作进程响应超时')
        if line is None:raise RuntimeError('连续物理工作进程提前退出；参见 body-worker.log')
        response=json.loads(line)
        if response.get('status')=='error':raise RuntimeError(response['reason'])
        return response

    def step(self, drive, refresh_vision=True):
        return self.call({'op':'advance','drive':drive,'refresh_vision':refresh_vision})

    def finish(self, output_dir=None, start_ms=0., continuous=False):
        return self.call({'op':'finish','output_dir':str(output_dir or self.output_dir),
                          'start_ms':start_ms,'continuous':continuous})

    def begin_observation(self):
        return self.call({'op':'begin_observation'})

    def present(self, presentations):
        return self.call({'op':'present','presentations':presentations})

    def update_environment(self, environment):
        return self.call({'op':'update_environment','environment':environment})

    def close(self):
        if self.process:
            try:self.process.stdin.close()
            except (OSError,ValueError):pass
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
            if hasattr(self,'reader'):self.reader.join(timeout=1)
            self.process.stdout.close()
        self.stderr.close()


class BodySession:
    def __init__(self, request):
        import numpy as np
        from chatflypt.behavior_body import initialize_body, COMMAND_KEYS, BodyControlStatistics
        self.environment=request['environment'];self.np=np
        self.body,self.initialization=initialize_body(request['seed'],request['context'],self.environment,inspection_contacts=True)
        self.initialization['performed']=True
        self.names=[s.name for s in self.body.fly.get_bodysegs_order()]
        self.sim=self.body.sim;self.model,self.data=self.sim.mj_model,self.sim.mj_data
        from chatflypt.physical_touch import PhysicalTouch
        self.touch=PhysicalTouch(self.model,self.data)
        self.current={k:0. for k in COMMAND_KEYS};self.row=-1;self.drive=None
        self.trajectory=[];self.statistics=BodyControlStatistics()
        self.last_vision=None
        self.feedback_interval_ms=100.
        self.check();self.trajectory.append(self.frame())

    def check(self):
        if not self.np.isfinite(self.data.qpos).all() or not self.np.isfinite(self.data.qvel).all():
            raise RuntimeError('MuJoCo produced non-finite state')
        if any(w.number for w in self.data.warning):raise RuntimeError('MuJoCo reported numerical/physics warnings')

    def frame(self):
        import math
        import mujoco
        from chatflypt.environment import measure_environment_contacts, measure_body_contacts
        from chatflypt.behavior_body import LEGS, command_execution
        mujoco.mj_forward(self.model,self.data)
        positions=self.sim.get_body_positions(self.body.fly.name)
        rotations=self.sim.get_body_rotations(self.body.fly.name)
        thorax=self.names.index('c_thorax');w,x,y,z=rotations[thorax]
        geom=self.body.geometry_indices
        return {'time_ms':round(self.body.elapsed*1000,7),'position_mm':positions[thorax].tolist(),
            'quaternion_wxyz':rotations[thorax].tolist(),'yaw_rad':math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)),
            'feet_mm':positions[[self.names.index(f'{leg}_tarsus5') for leg in LEGS]].tolist(),
            'joint_angles_rad':self.sim.get_joint_angles(self.body.fly.name).tolist(),
            'joint_positions_rad':dict(zip((j.name for j in self.body.fly.get_jointdofs_order()),
                                           self.sim.get_joint_angles(self.body.fly.name).tolist())),
            'joint_velocities_rad_s':dict(zip((j.name for j in self.body.fly.get_jointdofs_order()),
                                             self.sim.get_joint_velocities(self.body.fly.name).tolist())),
            'body_contacts':measure_body_contacts(self.model,self.data),
            'body_positions_mm':positions.tolist(),'body_quaternions_wxyz':rotations.tolist(),
            'geom_positions_mm':self.data.geom_xpos[geom].tolist(),'geom_rotation_matrices':self.data.geom_xmat[geom].tolist(),
            'contact_count':int(self.data.ncon),'environment_contacts':measure_environment_contacts(self.model,self.data),
            'physical_probes':self.touch.frame(),
            'commands':dict(self.current),'drive':dict(self.current),'source_command_bin':self.row,
            'command_available_at_ms':self.drive['available_at_ms'][self.row] if self.row>=0 else None,
            'joint_motor_offsets_rad':dict(self.body.joint_motor_offsets),
            **command_execution(self.current,self.body.joint_motor_offsets),
            'max_actuator_force_model_units':float(self.np.max(self.np.abs(self.data.actuator_force),initial=0))}

    def probe(self, refresh_vision=True):
        from chatflypt.environment import measure_vision
        frame=self.frame()
        if refresh_vision or self.last_vision is None:
            self.last_vision=measure_vision(self.sim,self.body.fly.name,self.environment,self.names,
                                           self.np.asarray(frame['body_positions_mm']))
            self.last_vision['sampled_at_ms']=self.body.elapsed*1000
        vision=dict(self.last_vision,age_ms=self.body.elapsed*1000-self.last_vision['sampled_at_ms'])
        return {'status':'ok','frame':frame,'body_names':self.names,'vision':vision,
            'environment_contacts':frame['environment_contacts'],'initialization':self.initialization}

    def update_environment(self, environment):
        from chatflypt.live_environment import update_body_environment
        from chatflypt.physical_touch import PhysicalTouch
        result=update_body_environment(self.body,self.environment,environment)
        program=self.touch.program
        self.model,self.data=self.sim.mj_model,self.sim.mj_data
        self.touch=PhysicalTouch(self.model,self.data);self.touch.program=program
        self.touch.update(self.body.elapsed*1000)
        self.environment=environment
        result['probe']=self.probe()
        # The boundary has the same body state but the new visible environment.
        if self.trajectory and abs(self.trajectory[-1]['time_ms']-self.body.elapsed*1000)<1e-5:
            self.trajectory[-1]=result['probe']['frame']
        return result

    def advance(self, drive, refresh_vision=True):
        from chatflypt.behavior_readout import _validate_behavior_drive
        from chatflypt.behavior_body import COMMAND_KEYS
        _validate_behavior_drive(drive)
        now=self.body.elapsed*1000;end=drive['duration_ms'];dt=self.sim.timestep*1000
        if not now<end<=60000 or end-now>100.00001:raise ValueError('Feedback advance must be within 100 ms and session within 60 s')
        if drive['context']!=self.body.context:raise ValueError('Body context changed during a trial')
        self.feedback_interval_ms=end-now
        self.drive=drive;times=self.np.asarray(drive['available_at_ms'])
        steps=round((end-now)/dt);sample_every=round(10/dt)
        if abs(steps*dt-(end-now))>1e-5:raise ValueError('Physics clock mismatch')
        for step in range(steps):
            self.touch.update(self.body.elapsed*1000)
            self.row=int(self.np.searchsorted(times,self.body.elapsed*1000+1e-8,side='right'))-1
            self.current=dict(drive['commands'][self.row]) if self.row>=0 else {k:0. for k in COMMAND_KEYS}
            from chatflypt.joint_motor import offsets_at
            self.body.joint_motor_offsets=offsets_at(drive,self.row)
            self.body.advance(dt,self.current)
            force=float(self.np.max(self.np.abs(self.data.actuator_force),initial=0))
            self.statistics.record(self.current,self.body.joint_motor_offsets,force)
            if (step+1)%sample_every==0 or step+1==steps:
                self.check();self.trajectory.append(self.frame())
        self.touch.update(self.body.elapsed*1000)
        return self.probe(refresh_vision)

    def begin_observation(self):
        from chatflypt.behavior_body import BodyControlStatistics
        self.statistics=BodyControlStatistics(self.body.elapsed*1000)
        return {'status':'ok','time_ms':self.body.elapsed*1000}

    def finish(self, output_dir, start_ms=0., continuous=False):
        import mujoco
        from chatflypt.behavior_body import _base_result, _write_result, COMMAND_KEYS
        self.check();result=_base_result(self.drive)
        if abs(self.statistics.start_ms-start_ms)>1e-5:
            raise ValueError('Body control statistics do not match the requested observation window')
        import copy
        frames=copy.deepcopy([f for f in self.trajectory if f['time_ms']>=start_ms-1e-7])
        if not frames or abs(frames[0]['time_ms']-start_ms)>1e-5:raise ValueError('Missing body observation boundary')
        for frame in frames:
            frame['session_time_ms']=frame['time_ms'];frame['time_ms']-=start_ms
        positions=self.np.asarray([f['position_mm'] for f in frames])
        duration=round(self.body.elapsed*1000,7)
        result.update(status='ok',physics_run=True,environment=self.environment,trajectory=frames,
            duration_ms=duration-start_ms,requested_duration_ms=duration-start_ms,truncated=False,body_names=self.names,
            session_start_ms=start_ms,session_end_ms=duration,
            joint_names=[j.name for j in self.body.fly.get_jointdofs_order()],geom_indices=self.body.geometry_indices,
            geom_names=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in self.body.geometry_indices],
            initialization=dict(self.initialization,performed=start_ms==0,resumed=start_ms>0),
            engine={'model':'Persistent Bad Apple Fly Body / FlyGym','timestep_ms':self.sim.timestep*1000},
            summary={'displacement_mm':(positions[-1]-positions[0]).tolist(),
                'path_length_mm':float(self.np.linalg.norm(self.np.diff(positions,axis=0),axis=1).sum()),
                'max_contacts':max(f['contact_count'] for f in frames),'finite_state':True,'physics_warnings':0})
        measured=self.statistics.report()
        result['controls'].update(measured.pop('controls'))
        result.update(measured)
        result['controls'].update(neural_feedback=True,feedback_interval_ms=self.feedback_interval_ms,
            controller_local_feedback='Official Hybrid local feedback; measured receptors separately return to neural state',
            active_posture_hold_when_commands_zero=True,passive_control=False,
            aerodynamics=False,
            kinematic_tether=self.body.context=='tethered_flight')
        result['limitations']=[s for s in result['limitations']
            if not any(old in s for old in ('没有反馈到连接组','没有地面接触向神经网络的感觉反馈'))]
        result['limitations'].append(f'物理反馈每 {self.feedback_interval_ms:g} ms 更新，眼图每 100 ms 更新；'+('同一会话跨消息保留状态，等待消息时仿真时钟暂停。' if continuous else '独立试次之间重新初始化。'))
        result['limitations'].append('MaleCNS 环境通路已回传实测腿角和角速度的 claw/club 候选群；负荷仅记录，hook 方向和 campaniform 转导尚未建立，群体活性到事件率未经生理标定。')
        output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
        self.np.savez_compressed(output/'physical-state.npz',qpos=self.data.qpos,qvel=self.data.qvel,
                                act=self.data.act,ctrl=self.data.ctrl,time=self.data.time)
        _write_result(result,output,self.drive)
        return result


def worker():
    sys.path.insert(0,str(ROOT));sys.stdin.reconfigure(encoding='utf8');sys.stdout.reconfigure(encoding='utf8')
    body=None
    for line in sys.stdin:
        try:
            request=json.loads(line);op=request['op']
            if op=='initialize':
                if body is not None:raise ValueError('Body already initialized')
                body=BodySession(request);result=body.probe()
            elif op=='advance':result=body.advance(request['drive'],request.get('refresh_vision',True))
            elif op=='present':
                result=body.touch.present(request['presentations'],body.body.elapsed*1000)
                result['probe']=body.probe()
            elif op=='update_environment':result=body.update_environment(request['environment'])
            elif op=='begin_observation':
                result=body.begin_observation()
            elif op=='finish':result=body.finish(request['output_dir'],request.get('start_ms',0.),request.get('continuous',False))
            else:raise ValueError('Unknown physics operation')
            print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
        except Exception as exc:
            print(json.dumps({'status':'error','reason':str(exc),'error_type':type(exc).__name__}),flush=True)
            break


if __name__=='__main__':
    if sys.argv[1:]!=['--worker']:raise SystemExit('Internal JSONL protocol only')
    worker()
