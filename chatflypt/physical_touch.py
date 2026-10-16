"""A physical probe, never a neural or motor command.

A presentation describes probe geometry and timing only. Each tap is aimed at
the target leg's current pose when that tap begins, so a multi-tap train follows
the moving leg; every tap is still a fixed world-space trajectory once aimed.
It never steers the fly. Only MuJoCo contacts measured elsewhere can become
sensory events.

Tap timing is an engineering parameter set: the accessible Ramdya et al. 2015
text supports gentle distal-leg contact between flies and a contralateral turn
tendency, but does not report contact durations in milliseconds. Defaults are
explicit assumptions, not measurements.
"""
import math

LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
RADIUS_MM = .06
PARK = [0., 0., 35.]
MIN_TAP_MS = 30.
MAX_TAPS = 8
PROBE_KEYS = {'kind', 'side', 'body_part', 'onset_ms', 'duration_ms', 'strength', 'taps', 'interval_ms'}


def add_probe_geometries(world):
    import mujoco
    for leg in LEGS:
        body = world.mjcf_root.worldbody.add_body(name='touch_probe_'+leg, mocap=True, pos=PARK)
        geom = body.add_geom(name='env_message_probe_'+leg, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                             size=[RADIUS_MM, 0., 0.], rgba=[.9, .6, .3, 0.], contype=0, conaffinity=0)
        world.ground_geoms.append(geom)
        world.environment_geoms.append(geom)


def tap_schedule(p):
    """Onset of each tap in a train, relative to the presentation onset."""
    return [p['onset_ms'] + i*p['interval_ms'] for i in range(int(p['taps']))]


def ramp_ms(p):
    # Short taps ramp faster so the tip actually reaches full indentation.
    return min(50., max(5., p['duration_ms']/2))


def presentation_span(p):
    """Last simulated millisecond a presentation can still touch the fly."""
    return tap_schedule(p)[-1] + p['duration_ms'] + ramp_ms(p)


def validate_presentations(presentations):
    if not isinstance(presentations, list) or len(presentations)>16:
        raise ValueError('每条消息最多 16 次物理触碰')
    result=[]
    for p in presentations:
        if not isinstance(p,dict) or set(p)-PROBE_KEYS:
            raise ValueError('物理呈现只接受探针几何与时序，不接受神经或动作指令')
        if p.get('kind')!='touch' or p.get('side') not in ('left','right') or p.get('body_part') not in ('foreleg','midleg','hindleg'):
            raise ValueError('尚未实现此部位的物理触碰探针')
        q=dict(p)
        for key,default,lo,hi in [('onset_ms',50.,0.,4000.),('duration_ms',300.,MIN_TAP_MS,2000.),
                                  ('strength',1.,0.,2.),('taps',1,1,MAX_TAPS),
                                  ('interval_ms',None,0.,1000.)]:
            v=p.get(key,default)
            if v is None:
                v=q['duration_ms']+80. if key=='interval_ms' else default
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
                raise ValueError('物理探针参数超出范围')
            q[key]=int(v) if key=='taps' else float(v)
        if q['taps']>1 and q['interval_ms']<q['duration_ms']+ramp_ms(q):
            raise ValueError('敲击间隔必须大于单次触碰加回撤时间')
        if presentation_span(q)+100>5000:raise ValueError('物理呈现窗口超过 5 秒')
        for other in result:
            if (q['side'],q['body_part'])==(other['side'],other['body_part']) and max(q['onset_ms'],other['onset_ms'])<min(presentation_span(q),presentation_span(other)):
                raise ValueError('同一个物理探针的呈现时段不能重叠')
        result.append(q)
    return result


class PhysicalTouch:
    def __init__(self,model,data):
        import mujoco
        self.model=model;self.data=data;self.program=[]
        self.ids={}
        for leg in LEGS:
            bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'touch_probe_'+leg)
            gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'env_message_probe_'+leg)
            if bid<0 or gid<0:raise ValueError('Physical touch geometry missing')
            self.ids[leg]=(int(model.body_mocapid[bid]),gid)

    def _aim(self,leg,side,strength):
        """Freeze one tap's trajectory from the target leg's current world pose."""
        import mujoco
        thorax=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'fly/c_thorax')
        left=self.data.xmat[thorax].reshape(3,3)[:,1].copy()
        gid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'fly/'+leg+'_tarsus3')
        if gid<0:raise ValueError('Physical target geometry missing')
        direction=left*(1 if side=='left' else -1)
        center=self.data.geom_xpos[gid].copy()
        # Bounded 0.36–0.40 mm approach. "Strength" changes physical indentation,
        # never event rate.
        end=center+direction*(.06-.02*strength)
        start=center+direction*.42
        start[2]=max(RADIUS_MM,start[2]);end[2]=max(RADIUS_MM,end[2])
        return start.tolist(),end.tolist()

    def present(self,presentations,at_ms):
        rows=validate_presentations(presentations)
        self.program=[]
        for p in rows:
            leg=('l' if p['side']=='left' else 'r')+{'foreleg':'f','midleg':'m','hindleg':'h'}[p['body_part']]
            self.program.append(dict(p,leg=leg,aimed=[None]*int(p['taps']),
                                     onsets=[at_ms+t for t in tap_schedule(p)]))
        self.update(at_ms)
        return {'status':'ok','source':'fixed_world_mocap_probe','program':self.program,
                'force_calibrated':False,'tracks_body':True,
                'aiming':'per_tap_at_onset','neural_input_written':False}

    def active_taps(self,at_ms):
        """Every tap active now as (program, tap index, phase amount), by leg."""
        active={}
        for p in self.program:
            if not p['strength']:continue
            for i,onset in enumerate(p['onsets']):
                t=at_ms-onset;duration=p['duration_ms'];ramp=ramp_ms(p)
                if 0.<=t<=duration+ramp:
                    amount=min(1.,t/ramp) if t<duration else max(0.,1.-(t-duration)/ramp)
                    active.setdefault(p['leg'],(p,i,amount))
        return active

    def update(self,at_ms):
        import numpy as np
        for mocap,gid in self.ids.values():
            self.data.mocap_pos[mocap]=PARK;self.model.geom_rgba[gid,3]=0.
        for leg,(p,index,amount) in self.active_taps(at_ms).items():
            if p['aimed'][index] is None:
                p['aimed'][index]=self._aim(leg,p['side'],p['strength'])
            start,end=p['aimed'][index]
            mocap,gid=self.ids[leg]
            self.data.mocap_pos[mocap]=np.array(start)*(1-amount)+np.array(end)*amount
            self.model.geom_rgba[gid,3]=1.

    def frame(self):
        return [{'id':'message_probe_'+leg,'position_mm':self.data.mocap_pos[mocap].tolist(),
                 'radius_mm':RADIUS_MM,'visible':bool(self.model.geom_rgba[gid,3])}
                for leg,(mocap,gid) in self.ids.items()]
