"""Versioned scene objects and explicit, uncalibrated environmental transduction.

Only applied objects reach this module. Drafts are a browser concern. Receptor
positions and taste contacts come from the same MuJoCo initialization as replay.
"""
import json
import math
import re
import subprocess
import os

from .paths import ROOT

TYPES = {
    'sugar': {'label': '糖', 'color': '#efe6b1', 'solid': True, 'channel': 'sugar_specific'},
    'yeast': {'label': '酵母', 'color': '#c59a62', 'solid': True, 'channel': 'yeast'},
    'high_salt': {'label': '高盐', 'color': '#c5dce1', 'solid': True, 'channel': 'high_salt'},
    'heat': {'label': '热源', 'color': '#f07a50', 'solid': False, 'channel': 'warm'},
    'cold': {'label': '冷源', 'color': '#6cbce6', 'solid': False, 'channel': 'cold'},
    'light': {'label': '光源', 'color': '#f4d86a', 'solid': False, 'channel': 'bright'},
    'shade': {'label': '暗区 / 遮光区域', 'color': '#596174', 'solid': False, 'channel': None},
    'block': {'label': '可站立方块', 'color': '#7c9e8c', 'solid': True, 'channel': None},
    # Conspecific signals. Both are non-solid engineering fields: cVA is sampled
    # at the funiculus (where Or67d ORNs sit) and song at the pedicel (where
    # Johnston's organ sits). Neither is a calibrated concentration or sound
    # pressure, and neither emits anything without a measured field.
    'pheromone': {'label': 'cVA 信息素源', 'color': '#c8a2c8', 'solid': False, 'channel': 'cva'},
    'song': {'label': '求偶鸣声源', 'color': '#9fd8cb', 'solid': False, 'channel': 'courtship_pulse'},
}
# Receptor site used by each measured surface field. Funiculus/pedicel are real
# model body positions, not substitutes: no site falls back to another part.
FIELD_SITES = {'heat': 'arista', 'cold': 'arista', 'pheromone': 'funiculus', 'song': 'pedicel'}
# Field models for non-solid emitters:
#   range    - fixed finite-range engineering field (default, uncalibrated)
#   plume    - borrowed upstream precomputed plume grid (see plume_field.py)
#   anchored - the source keeps its initial offset but follows the measured
#              receptor pose ("a partner that stays with the animal")
FIELD_MODELS = ('range', 'plume', 'anchored')
ANCHOR_SITES = ('funiculus', 'pedicel', 'arista')
LIMITATIONS = [
    '气味、温度及光照使用位置相关的工程场，强度不是浓度、摄氏度或照度的生理标定。',
    '糖实际接触跗节可触发 LgLG4/LgAG2 候选群，按真实腿及侧别选择，未定位单根味毛。胫节虽有味毛，但尚未核实这两群的胫节映射。高盐腿部通路仍未映射。',
    '当前身体没有独立 labellum/唇部味毛表面；rostrum、haustellum 接触不能代替唇部味觉，保留未映射记录。',
    '热/冷工程场在各侧实际 arista 位置按正负相加后再转导，未模拟真实热传导、摄氏温度或动态适应。气味在 funiculus 采样；VP3 亚群的外周对应仍未逐细胞标定。',
    '翼缘触毛、后部背板及其他未配准体表的真实接触保留部位和原因，不把整个翼膜或胸部碰撞冒充已定位的感觉毛。',
    '暗区衰减局部光照，不是发出黑光，也没有单独注入 OFF 视觉通道。',
    '正常聊天把官方小眼采样投射到 MaleCNS 感光细胞及其连接推定的视网膜列；覆盖不完整，跨标本尺度、朝向与眼位仍为候选配准，没有物体身份识别。旧 FlyWire 实验保留分眼亮度代理。',
    '正常聊天每 10 ms 采样身体和接触、每 100 ms 更新眼图；消息与物品增删移动保留同一神经、身体和控制器状态。更换身体场景、新对话或服务重启初始化；独立实验不共享聊天状态。',
    '正常环境通路使用单个 MaleCNS 神经和身体会话；旧独立连接组实验不能合并成同一个体。',
    '实测腿角与角速度反馈到 claw/club 候选本体入口；钙活性到事件率的增益未经生理标定，负荷记录尚未转导为 campaniform 放电。',
    'cVA 信息素与求偶鸣声是工程距离场，不是浓度或声压；发射器不会自行发出信号，只有实测场强超过阈值才产生候选入口事件。鸣声的时间结构沿用文献示例的 36 ms 脉冲间隔，未实现载波、相位锁定或适应。',
]


def empty_environment():
    return {'schema': 'chatflypt.environment.v1', 'revision': 0, 'objects': []}


def validate_objects(objects):
    if not isinstance(objects, list) or len(objects) > 32:
        raise ValueError('环境最多支持 32 个物品')
    result, ids = [], set()
    for obj in objects:
        if not isinstance(obj, dict) or obj.get('type') not in TYPES:
            raise ValueError('未知环境物品')
        oid = obj.get('id')
        if not isinstance(oid, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', oid) or oid in ids or oid.startswith('message_probe_'):
            raise ValueError('环境物品 ID 必须唯一且有效')
        ids.add(oid)
        def number(value, lo, hi):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError('环境坐标、大小或强度超出范围')
            return float(value)
        def vector(key, default, lo, hi):
            values = obj.get(key, default)
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError('坐标与大小必须为三维数组')
            return [number(v, lo, hi) for v in values]
        position = vector('position_mm', [3, 0, .5], -40, 40)
        size = vector('size_mm', [1, 1, 1], .05, 20)
        if position[2] < size[2]/2 - 1e-8:
            raise ValueError('物品底面不能低于地面，请增大 Z 或减小高度')
        model = obj.get('field', 'range')
        if model not in FIELD_MODELS:
            raise ValueError('未知的场模型（可选 range/plume/anchored）')
        anchor = obj.get('anchor')
        anchor_side = obj.get('anchor_side')
        origin = None
        if model == 'anchored':
            if anchor not in ANCHOR_SITES:
                raise ValueError('跟随源必须声明 anchor（funiculus/pedicel/arista）')
            if anchor_side is not None and anchor_side not in ('left', 'right'):
                raise ValueError('anchor_side 只能是 left 或 right')
            # The frozen relative offset is defined against the receptor's
            # neutral position, so the source follows the measured pose with a
            # fixed geometry instead of drifting with it. Declaring the side says
            # which receptor the source belongs to; without it the legacy
            # bilateral behaviour applies and is recorded as such.
            origin = vector('anchor_origin_mm', None, -40, 40) if obj.get('anchor_origin_mm') is not None else None
            if origin is None:
                raise ValueError('跟随源必须声明 anchor_origin_mm（受体中立位置）')
        elif anchor is not None or anchor_side is not None:
            raise ValueError('只有跟随源可以声明 anchor / anchor_side')
        entry = {'id': oid, 'type': obj['type'], 'position_mm': position, 'size_mm': size,
                 'yaw_deg': number(obj.get('yaw_deg', 0), -360, 360),
                 'strength': number(obj.get('strength', 1), 0, 2),
                 'range_mm': number(obj.get('range_mm', 4), .1, 30),
                 'field': model}
        if model == 'anchored':
            entry.update(anchor=anchor, anchor_origin_mm=origin)
            if anchor_side:
                # Only a declared side is recorded on the object, so previously
                # frozen scenes validate to exactly the same payload as before.
                entry.update(anchor_side=anchor_side)
        result.append(entry)
    return result


def probe_environment(environment, seed=42, context='ground'):
    """Measure actual initialization poses/contacts in the isolated physics runtime."""
    python = ROOT / '.venv-body' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    request = {'environment': environment, 'probe_only': True, 'seed': seed, 'context': context}
    done = subprocess.run([str(python), str(ROOT/'chatflypt/behavior_body.py'), '--worker'],
        input=json.dumps(request), capture_output=True, text=True, encoding='utf8', timeout=90,
        cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if done.returncode:
        raise ValueError('环境物理初始化失败：' + done.stderr[-500:])
    result = json.loads(done.stdout)
    if result.get('status') != 'ok':
        raise ValueError('环境物理初始化失败：' + result.get('reason', 'unknown'))
    return result


def local_coordinates(point, obj):
    x, y, z = [a-b for a,b in zip(point, obj['position_mm'])]
    angle = math.radians(obj['yaw_deg'])
    return [math.cos(angle)*x+math.sin(angle)*y, -math.sin(angle)*x+math.cos(angle)*y, z]


def field(point, obj):
    # Smooth finite-range engineering field; explicitly not a calibrated plume.
    distance = math.sqrt(sum(v*v for v in local_coordinates(point, obj)))
    return obj['strength'] * max(0., 1-distance/obj['range_mm'])**2


def measured_field(model, point, obj, time_ms, anchor_position=None):
    """Value of a non-solid emitter at one measured receptor position.

    Three declared models, all uncalibrated and all evaluated at the receptor's
    *measured* position:

    * ``range``    - the finite-range engineering field above.
    * ``plume``    - the borrowed upstream precomputed plume grid; a sample that
                     falls outside the grid is no input (never zero).
    * ``anchored`` - the source keeps its initial offset from **its own declared
                     receptor** and follows that receptor's measured pose, i.e. a
                     partner that stays with the animal on one side. Without a
                     declared ``anchor_side`` the model falls back to the legacy
                     behaviour of re-placing the source next to whichever receptor
                     is sampled - which gives every receptor the same field and
                     cannot express a side at all. That legacy mode is kept only so
                     that previously frozen scenes stay reproducible; it is
                     reported as ``anchor_mode='legacy_bilateral'``.
    """
    if model == 'plume':
        from .plume_field import sample as plume_sample
        value = plume_sample(point, time_ms, obj['strength'])
        return 0. if value is None else value
    if model == 'anchored':
        # Preserve the declared relative geometry: the source sits at the offset
        # it was declared with, measured from the receptor it belongs to.
        offset = [a - b for a, b in zip(obj['position_mm'], obj['anchor_origin_mm'])]
        origin = anchor_position if anchor_position is not None else point
        moved = {'position_mm': [a + b for a, b in zip(origin, offset)],
                 'yaw_deg': obj['yaw_deg'], 'strength': obj['strength'],
                 'range_mm': obj['range_mm']}
        return field(point, moved)
    return field(point, obj)

def sample_environment(environment, probe, catalogue, *, dataset=None, time_ms=0.):
    """Measure the applied scene at the current receptor poses.

    ``time_ms`` is the simulated time (ms since session start) used by
    time-indexed field models such as the borrowed upstream plume grid.
    """
    sites = dict(zip(probe['body_names'], probe['frame']['body_positions_mm']))
    contacts = probe.get('environment_contacts', [])
    totals, evidence, unsupported = {}, [], []
    thermal_components={'left':[],'right':[]}
    # Mechanical contact and taste use distinct receptor surfaces. A collision
    # with an anatomical segment does not locate a single sensory hair.
    leg_contacts={}
    for contact in contacts:
        match=re.fullmatch(r'fly/([lr])([fmh])_(?:coxa|trochanterfemur|tibia|tarsus[1-5])',contact['fly_geom'])
        if match:
            side='left' if match[1]=='l' else 'right'
            part={'f':'foreleg','m':'midleg','h':'hindleg'}[match[2]]
            leg_contacts.setdefault((contact['object_id'],side,part),[]).append(contact)
        else:
            wing=re.fullmatch(r'fly/([lr])_wing',contact['fly_geom'])
            if wing:
                side='left' if wing[1]=='l' else 'right'
                channel,part='wing_touch','wing'
                reason='ADMN 触毛对应翼缘；当前翼网格未配准翼缘感器，不能将任意翼膜接触视为触毛偏转'
            elif contact['fly_geom']=='fly/c_thorax':
                side=None;channel,part='body_touch','notum'
                reason='PDMN 候选仅覆盖后部背板；当前接触点未配准背板感器区，不能把整个胸部碰撞映射到该群'
            else:
                side=None;channel,part='surface_touch',contact['fly_geom'].removeprefix('fly/')
                reason='已测量该体表接触，但尚无经核实的 MaleCNS 部位—触毛映射'
            unsupported.append({'object_id':contact['object_id'],'channel':channel,'side':side,
                                'body_part':part,'contacts':[contact],'reason':reason})
            evidence.append({'object_id':contact['object_id'],'mode':'mujoco_surface_contact_unmapped',
                             'side':side,'body_part':part,'contacts':[contact],'activated':False,'reason':reason})
    for (oid,side,part),hits in leg_contacts.items():
        channel=part+'_touch';spec=catalogue['channels'].get(channel,{})
        if spec.get('status') in ('ready','candidate') and spec.get('side_ids',{}).get(side):
            totals[channel,side,None]=1.
        else:unsupported.append({'object_id':oid,'channel':channel,'side':side,'reason':'真实腿接触缺少可用机械感觉入口'})
        evidence.append({'object_id':oid,'mode':'mujoco_leg_contact','side':side,'body_part':part,'contacts':hits,
                         'physical_force_calibrated':False})
    for obj in environment['objects']:
        kind = obj['type']; channel = TYPES[kind]['channel']
        if channel is None or kind == 'light':
            continue
        if kind in ('sugar', 'high_salt'):
            actual = [c for c in contacts if c['object_id'] == obj['id']]
            # The official model has rostrum/haustellum but no separately
            # registered labellum. Do not infer a distal taste surface by an
            # arbitrary coordinate threshold on either coarse mouth mesh.
            mouth = [c for c in actual if c['fly_geom']=='fly/c_labellum']
            coarse_mouth = [c for c in actual if c['fly_geom'] in ('fly/c_rostrum','fly/c_haustellum')]
            values = {'bilateral': obj['strength'] if mouth else 0.}
            evidence.append({'object_id': obj['id'], 'mode': 'mujoco_mouth_contact', 'contacts': mouth,
                             'unmapped_mouth_contacts':coarse_mouth,
                             'activated': bool(mouth), 'intensities': values,'receptor_surface':'labellum',
                             'surface_registration_calibrated':False})
            if coarse_mouth:
                unsupported.append({'object_id':obj['id'],'channel':channel,'body_part':'mouth',
                    'contacts':coarse_mouth,'reason':'实测 rostrum/haustellum 接触不能定位 labellum 味毛；官方身体缺少已注册的唇部表面，未激活唇部味觉'})
            for (oid,side,part),hits in leg_contacts.items():
                if oid!=obj['id']:continue
                if kind=='high_salt':
                    unsupported.append({'object_id':oid,'channel':'leg_high_salt','side':side,'body_part':part,'reason':'高盐腿部味觉尚无可执行受体映射'})
                    continue
                tarsal_hits=[c for c in hits if re.fullmatch(r'fly/[lr][fmh]_tarsus[1-5]',c['fly_geom'])]
                other_hits=[c for c in hits if c not in tarsal_hits]
                if other_hits:
                    unsupported.append({'object_id':oid,'channel':'leg_sweet_surface','side':side,'body_part':part,
                        'contacts':other_hits,'reason':'该腿接触不在已支持的跗节候选表面；胫节虽有味毛，当前 LgLG4/LgAG2 的非跗节对应尚未核实'})
                if not tarsal_hits:continue
                for leg_channel in ('leg_sweet_local','leg_appetitive_ascending'):
                    spec=catalogue['channels'].get(leg_channel,{})
                    ids=spec.get('body_part_ids',{}).get(part,{}).get(side,[])
                    if spec.get('status') in ('ready','candidate') and ids:
                        totals[leg_channel,side,part]=min(2.,totals.get((leg_channel,side,part),0.)+obj['strength'])
                        evidence.append({'object_id':oid,'mode':'mujoco_leg_taste_candidate','channel':leg_channel,
                            'side':side,'body_part':part,'neuron_ids':ids,'contacts':tarsal_hits,'source':spec['source'],
                            'receptor_surface':'tarsus','surface_registration_calibrated':False,
                            'functional_mapping':spec.get('functional_mapping'),'readout_calibrated':False})
                    else:unsupported.append({'object_id':oid,'channel':leg_channel,'side':side,'body_part':part,'reason':'该接触腿缺少味觉候选入口'})
        else:
            values, receptor_sites = {}, {}
            model = obj.get('field', 'range')
            anchor_side = obj.get('anchor_side') if model == 'anchored' else None
            anchor_point = None
            if anchor_side:
                anchor_name = ('l' if anchor_side == 'left' else 'r') + '_' + obj['anchor']
                anchor_point = sites.get(anchor_name)
                if anchor_point is None:
                    unsupported.append({'object_id':obj['id'],'channel':channel,'side':anchor_side,
                        'reason':'物理记录缺少跟随源所属受体的位置 '+anchor_name+'，不以其他部位代替'})
                    continue
            for side, prefix in [('left', 'l'), ('right', 'r')]:
                site_name=prefix+'_'+(obj['anchor'] if model=='anchored' else FIELD_SITES.get(kind,'funiculus'))
                if site_name not in sites:
                    unsupported.append({'object_id':obj['id'],'channel':channel,'side':side,
                                        'reason':'物理记录缺少实际感受器位置 '+site_name+'，不以其他部位代替'})
                    continue
                point = sites[site_name]
                receptor_sites[side]={'body_name':site_name,'position_mm':list(point)}
                values[side] = measured_field(model, point, obj, time_ms, anchor_position=anchor_point)
            evidence.append({'object_id': obj['id'],
                             'mode': ('temperature_field_component' if kind in ('heat','cold') else
                                      'upstream_plume_grid' if model=='plume' else
                                      'following_source_field' if model=='anchored' else 'position_field'),
                             'anchor_side': obj.get('anchor_side') if model == 'anchored' else None,
                             'anchor_mode': (('declared_side' if obj.get('anchor_side') else 'legacy_bilateral')
                                             if model == 'anchored' else None),
                             'field_model': model, 'sampled_at_ms': float(time_ms),
                             'intensities': values,
                             'receptor_sites':receptor_sites,'field_calibrated':False,
                             'concentration_units':False})
            if kind in ('heat','cold'):
                for side,value in values.items():
                    thermal_components[side].append({'object_id':obj['id'],
                        'signed_field':value if kind=='heat' else -value,
                        'receptor_site':receptor_sites[side]})
                # Opposing fields combine at the receptor before choosing a
                # population. Object labels must not independently deliver
                # simultaneous opposite temperature deviations at one site.
                continue
        for side, value in values.items():
            if value <= .0001:
                continue
            spec = catalogue['channels'].get(channel, {})
            if spec.get('status') not in ('ready', 'candidate') or not spec.get('side_ids', {}).get(side):
                unsupported.append({'object_id': obj['id'], 'channel': channel, 'side': side, 'reason': '该入口缺少可用映射，不以另一侧代替'})
                continue
            totals[channel, side, None] = min(2., totals.get((channel, side, None), 0.)+value)
    for side,components in thermal_components.items():
        if not components:continue
        net=math.fsum(row['signed_field'] for row in components)
        channel=('warm' if net>0 else 'cold') if abs(net)>.0001 else None
        evidence.append({'mode':'net_temperature_field','side':side,'channel':channel,
            'net_signed_field':net,'components':components,'field_calibrated':False,
            'units':'dimensionless engineering temperature deviation',
            'receptor_site':components[0]['receptor_site']})
        if channel is None:continue
        spec=catalogue['channels'].get(channel,{})
        if spec.get('status') not in ('ready','candidate') or not spec.get('side_ids',{}).get(side):
            unsupported.append({'channel':channel,'side':side,'reason':'该入口缺少可用映射，不以另一侧代替'})
            continue
        totals[channel,side,None]=min(2.,abs(net))
    vision = probe.get('vision', {})
    if vision.get('status') == 'measured':
        for side, value in vision['mean_by_eye'].items():
            spec = catalogue['channels'].get('bright', {})
            if value > .0001 and spec.get('side_ids', {}).get(side):
                totals['bright', side, None] = min(2., value)
            elif value > .0001:
                unsupported.append({'channel': 'bright', 'side': side, 'reason': '该眼没有可用连接组入口'})
        evidence.append({'mode': 'official_retina', 'mean_by_eye': vision['mean_by_eye'],
                         'object_identity_injected': False, 'spatial_registration': False})
    items = [dict({'channel': c, 'side': s, 'intensity': v, 'onset_ms': 50., 'duration_ms': 300.},
                  **({'body_part':part} if part else {})) for (c,s,part),v in sorted(totals.items()) if v>0]
    sample={'items': items, 'evidence': evidence, 'unmapped': unsupported,
            'dataset':dataset,'single_individual':dataset is not None,
            'environment_revision': environment['revision'], 'sensor_pose': probe['frame'],
            'sampling': 'mujoco_initial_pose_and_contacts', 'continuous_feedback': False,
            'vision': vision, 'limitations': list(LIMITATIONS)}
    if dataset is not None:
        from .malecns_senses import remap_environment_sample
        return remap_environment_sample(catalogue,sample,dataset)
    return sample


def add_environment_to_world(base_class, environment, *, full_body_contacts=False):
    """Extend the world before compilation; preserve upstream body/controller code."""
    import mujoco
    objects = validate_objects(environment['objects'])
    class ObjectWorld(base_class):
        def __init__(self):
            super().__init__()
            self.environment_geoms = []
            from .physical_touch import add_probe_geometries
            add_probe_geometries(self)
            for obj in objects:
                if obj['type'] == 'light':
                    self.mjcf_root.worldbody.add_light(name='light_'+obj['id'],
                        type=mujoco.mjtLightType.mjLIGHT_POINT, pos=obj['position_mm'],
                        diffuse=[obj['strength']]*3, attenuation=[1.,0.,1./obj['range_mm']**2],
                        castshadow=True)
                if not TYPES[obj['type']]['solid']:
                    continue
                color = TYPES[obj['type']]['color'].lstrip('#')
                rgba = [int(color[i:i+2],16)/255 for i in (0,2,4)] + [1.]
                a = math.radians(obj['yaw_deg'])/2
                geom = self.mjcf_root.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                    name='env_'+obj['id'], pos=obj['position_mm'], size=[v/2 for v in obj['size_mm']],
                    quat=[math.cos(a),0,0,math.sin(a)], rgba=rgba, contype=0, conaffinity=0)
                self.ground_geoms.append(geom); self.environment_geoms.append(geom)

        def _attach_fly_mjcf(self, fly, *args, **kwargs):
            names = super()._attach_fly_mjcf(fly, *args, **kwargs)
            # Existing ground pairs cover legs. Enable physical mouth contact
            # with objects only; do not substitute leg sensation with mouth GRNs.
            for segment, geoms in fly.bodyseg_to_mjcfgeom.items():
                if full_body_contacts:
                    # The upstream preset already pairs tibiae/tarsi with every
                    # ground surface. Passive inspection also needs the trunk,
                    # head, wings and other links to collide instead of sinking.
                    link=segment.name.split('_')[-1]
                    if link=='tibia' or link.startswith('tarsus'):continue
                elif segment.name not in ('c_rostrum', 'c_haustellum', 'c_head', 'c_thorax', 'c_abdomen12'):
                    continue
                for geom in geoms:
                    for objgeom in (self.ground_geoms if full_body_contacts else self.environment_geoms):
                        self.mjcf_root.add_pair(geomname1=geom.name, geomname2=objgeom.name,
                            name=geom.name+'-'+objgeom.name+'-object', friction=[1.,1.,.005,.0001,.0001])
            return names
    return ObjectWorld


def measure_environment_contacts(model, data):
    """Return active surface contacts, excluding proximity-only records.

    MuJoCo may retain gap contacts without an active constraint, or generate
    forces inside an inflated margin before the actual mesh surfaces touch.
    Neither is evidence that food contacted a taste sensillum. Force is saved
    in native model units; no unvalidated force-to-firing curve is inferred.
    """
    import mujoco
    import numpy as np
    contacts = []
    for index,contact in enumerate(data.contact):
        if contact.efc_address < 0 or contact.dist > 0:
            continue
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or '' for i in contact.geom]
        for i, name in enumerate(names):
            if name.startswith('env_') and names[1-i].startswith('fly/'):
                force=np.zeros(6)
                mujoco.mj_contactForce(model,data,index,force)
                contacts.append({'object_id': name[4:], 'fly_geom': names[1-i],
                                 'distance_mm': float(contact.dist), 'position_mm': contact.pos.tolist(),
                                 'active_constraint':True,'surface_contact':True,
                                 'normal_force_model_units':float(force[0]),
                                 'tangential_force_model_units':float(np.linalg.norm(force[1:3])),
                                 'force_calibrated':False})
    return contacts


def measure_body_contacts(model, data):
    """Actual external surface loads, including ground, for body sensation."""
    import mujoco
    import numpy as np
    measured=[]
    for index,contact in enumerate(data.contact):
        if contact.efc_address<0 or contact.dist>0:continue
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or '' for g in contact.geom]
        for i,name in enumerate(names):
            if not name.startswith('fly/') or names[1-i].startswith('fly/'):continue
            force=np.zeros(6);mujoco.mj_contactForce(model,data,index,force)
            measured.append({'body_geom':name,'other_geom':names[1-i],
                'distance_mm':float(contact.dist),'active_constraint':True,'surface_contact':True,
                'normal_force_model_units':float(force[0]),
                'tangential_force_model_units':float(np.linalg.norm(force[1:3])),
                'position_mm':contact.pos.tolist(),'force_calibrated':False})
    return measured


def measure_vision(sim, fly_name, environment, names, positions):
    """Use official eye cameras/Retina, without telling the brain object labels."""
    import base64
    import io
    import numpy as np
    from PIL import Image
    from .body_measurement import eye_camera_positions
    images = sim.get_raw_vision(fly_name)
    eye_positions = eye_camera_positions(sim, fly_name)
    values, pictures, means = [], {}, {}
    for i, side in enumerate(('left','right')):
        image = images[i].copy()
        attenuation = 1.
        for shade in environment['objects']:
            if shade['type']=='shade' and all(abs(v)<=s/2 for v,s in zip(local_coordinates(eye_positions[side],shade),shade['size_mm'])):
                attenuation *= max(0.,1-min(1.,shade['strength']))
        image = (image.astype(float)*attenuation).astype(np.uint8)
        ommatidia = sim.retina.raw_image_to_hex_pxls(image)
        values.append(ommatidia.tolist())
        # One nonzero yellow/pale channel per ommatidium. No object-name input.
        means[side] = float(np.sum(ommatidia,axis=-1).mean())
        output=io.BytesIO();Image.fromarray(image).resize((240,240)).save(output,format='PNG')
        pictures[side]='data:image/png;base64,'+base64.b64encode(output.getvalue()).decode('ascii')
    sim.eye_renderer.close();sim.eye_renderer=None
    return {'status':'measured','source':'FlyGym get_raw_vision + Retina.raw_image_to_hex_pxls',
            'ommatidia':values,'mean_by_eye':means,'eye_images':pictures,
            'eye_positions_mm': {side: position.tolist() for side, position in eye_positions.items()},
            'eye_position_source':'compiled MuJoCo camera frames',
            'bridge':'official local ommatidial measurements; downstream author-axis column candidate mapping',
            'object_identity_injected':False,'spatial_registration':False}
