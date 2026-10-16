"""Ground human scene references before proposing bounded sensory inputs.

Object coordinates are used to interpret the human's referent and to evaluate
the recorded trajectory. They are never decoder features or body commands.
"""
import math
import re
import unicodedata

ALIASES={
    'sugar':('糖块','糖','sugar'), 'yeast':('酵母','yeast'),
    'high_salt':('高盐','盐块','盐','salt'), 'heat':('热源',),
    'cold':('冷源',), 'light':('光源','灯'),
    'shade':('遮光区域','暗区','暗源'), 'block':('可站立方块','方块','平台'),
    'pheromone':('cVA 信息素源','cVA信息素源','信息素源','cVA'), 'song':('求偶鸣声源','鸣声源'),
}
_REFERENT='(?:最近的|左边的|右边的|左侧的|右侧的|前面的|后面的)?(?:'+ '|'.join(re.escape(x) for names in ALIASES.values() for x in names)+')'
_GO=re.compile(r'(?:请你|请|你)?(?:走到|走向|前往|靠近|去|到|朝|向)(?P<target>'+_REFERENT+r')(?:的)?(?:位置|那里|那边|旁边)?(?:去|走|走去|移动|靠近)?')
_WHERE=re.compile(r'(?:请问|请告诉我|告诉我)?(?:(?P<target>'+_REFERENT+r')(?:在)?(?:哪里|哪儿|哪边|什么位置)|(?:哪里|哪儿|哪边)有(?P<other>'+_REFERENT+r'))')
_LOOK=re.compile(r'(?:请你|请|你)?(?:看看|看一下|观察|感知)(?P<target>'+_REFERENT+r'|周围|环境|四周)')


def parse_scene_request(message):
    if not isinstance(message,str) or not message.strip() or len(message)>4000:return None
    text=unicodedata.normalize('NFKC',message).casefold()
    text=re.sub(r'[\s。！？?!,，]+$','',text.strip())
    match=_GO.fullmatch(text);kind='goal'
    if not match:match=_WHERE.fullmatch(text);kind='query'
    if not match:match=_LOOK.fullmatch(text);kind='observe'
    if not match:return None
    target=match.groupdict().get('target') or match.groupdict().get('other')
    objtype=next((kind for kind,names in ALIASES.items() if any(target.endswith(x) for x in names)),None)
    return {'kind':kind,'source_text':message,'referent':target,'object_type':objtype}


def relative_position(frame,names,obj):
    """Actual thorax pose; body local +X forward and +Y left."""
    origin=frame['position_mm'];index=names.index('c_thorax')
    w,x,y,z=frame['body_quaternions_wxyz'][index]
    norm=math.sqrt(w*w+x*x+y*y+z*z)
    if not math.isfinite(norm) or norm<1e-8:raise ValueError('身体朝向无效')
    w,x,y,z=[v/norm for v in (w,x,y,z)]
    delta=[v-o for v,o in zip(obj['position_mm'],origin)]
    # Inverse quaternion rotation: world displacement into thorax coordinates.
    forward=(1-2*y*y-2*z*z)*delta[0]+(2*x*y+2*w*z)*delta[1]+(2*x*z-2*w*y)*delta[2]
    left=(2*x*y-2*w*z)*delta[0]+(1-2*x*x-2*z*z)*delta[1]+(2*y*z+2*w*x)*delta[2]
    bearing=math.degrees(math.atan2(left,forward))
    side='left' if bearing>15 else 'right' if bearing< -15 else 'front'
    return {'bearing_deg':bearing,'side':side,'distance_mm':math.sqrt(sum(v*v for v in delta)),
            'forward_mm':forward,'left_mm':left}


def ground_scene_request(request,environment,probe,catalogue):
    plan={'kind':'scene_observation','items':[],'presentations':[],'scene_request':request,'assumptions':[
        '物品名称及位置用于理解人的指代，不作为神经读出器的答案。',
        '感官输入来自本次实际物理姿态、双眼渲染和接触采样。'],
        'translation':{'role':'human_to_sensory_input','forces_body_action':False,
            'description':'重新采样已应用环境，呈现实际视觉、气味、温度和接触输入'},
        'environment_revision':environment['revision'],'unresolved':[]}
    if request['kind']=='query':
        plan['assumptions'].append('位置问句触发新的环境观察；尚无把“在哪里”这一问句含义传给果蝇的感觉码本，不能把采样成功当作它理解了问题。')
        plan['translation']['question_semantics_encoded']=False
    candidates=[o for o in environment['objects'] if o['type']==request['object_type']]
    measured=[(o,relative_position(probe['frame'],probe['body_names'],o)) for o in candidates]
    ref=request['referent']
    if ref.startswith(('左边的','左侧的')):measured=[x for x in measured if x[1]['left_mm']>0]
    if ref.startswith(('右边的','右侧的')):measured=[x for x in measured if x[1]['left_mm']<0]
    if ref.startswith('前面的'):measured=[x for x in measured if x[1]['forward_mm']>0]
    if ref.startswith('后面的'):measured=[x for x in measured if x[1]['forward_mm']<0]
    if ref.startswith('最近的') and measured:measured=[min(measured,key=lambda x:x[1]['distance_mm'])]
    if len(measured)==1:
        obj,relative=measured[0]
        plan['referent']={'object_id':obj['id'],'object_type':obj['type'],
            'source':'applied_scene_for_human_reference_only','position_mm':obj['position_mm'],**relative}
    elif request['object_type']:
        reason='已应用环境中没有符合指代的物品。' if not measured else '环境里有多个符合指代的物品；需要用左边、右边或最近的区分。'
        plan['unresolved'].append(reason)
        if request['kind']=='goal':plan['kind']='unsupported'
    if request['kind']=='goal' and plan.get('referent'):
        # Reuse only the user-approved direction-to-opposite-foreleg encoding.
        # One initial cue; never issue a sequence of target-seeking body commands.
        side=plan['referent']['side']
        if side in ('left','right'):
            entry='right' if side=='left' else 'left';spec=catalogue['channels'].get('foreleg_touch',{})
            if spec.get('status') not in ('candidate','ready') or not spec.get('side_ids',{}).get(entry):
                plan['kind']='unsupported';plan['unresolved'].append('目标方向所需的前腿触碰入口不可用。')
            else:
                plan['presentations']=[{'kind':'touch','side':entry,'body_part':'foreleg',
                                'strength':1.,'onset_ms':50.,'duration_ms':300.}]
                plan['proposed']=True
                plan['translation'].update(mapping_status='engineering_proposal',description='用真实碰撞探针尝试触碰'+('右' if entry=='right' else '左')+'前腿；只有实测接触才生成触觉输入，同时采样已应用环境')
                plan['assumptions'].append('人的目标方位译为一次对侧前腿触碰；不向身体写入方向、速度或路径，也不保证接近目标。')
        else:
            plan['assumptions'].append('目标位于身体前方；没有经过验证的“向前走”感觉编码，因此只呈现实际环境，不伪造前进指令。')
    plan['reason']='；'.join(plan['unresolved']) if plan['kind']=='unsupported' else '已解析人的场景指代；执行环境感觉采样与神经响应。'
    return plan


def evaluate_goal(plan,observations):
    """Post-hoc trajectory check, never a control policy or speech decoder."""
    if plan.get('scene_request',{}).get('kind')!='goal' or not plan.get('referent'):return None
    target=plan['referent'];rows=[]
    for obs in observations:
        body=obs.get('scene',{}).get('body',{});frames=body.get('trajectory',[])
        if not body.get('physics_run') or not frames:continue
        distance=lambda f:math.dist(f['position_mm'],target['position_mm'])
        first,last=distance(frames[0]),distance(frames[-1]);progress=first-last
        touches=[{'time_ms':f['time_ms'],'fly_geom':c['fly_geom']} for f in frames
                 for c in f.get('environment_contacts',[]) if c['object_id']==target['object_id']]
        commands=body.get('controls',{}).get('powered_steps',0)
        rows.append({'observation_id':obs['id'],'dataset':obs['dataset'],'initial_distance_mm':first,
            'final_distance_mm':last,'distance_change_mm':progress,'target_contacts':touches,
            'motor_command_steps':commands,'status':'contact_recorded' if touches else 'not_reached'})
    return {'source':'recorded_physics_only','object_id':target['object_id'],'trials':rows,
        'used_for_control':False,'goal_completed':False,
        'limitations':['距离变化可能包含姿态维持、重力和接触的影响；不能单凭接近或接触认定已理解或完成指令。']}
