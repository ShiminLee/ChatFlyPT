"""Translate language into world presentations, without selecting neural IDs."""
import copy
import math
import re

from .environment import TYPES, validate_objects
from .input_language import parse_input
from .scene_language import ALIASES, parse_scene_request, ground_scene_request

OBJECT_CHANNELS={'sugar_specific':'sugar','yeast':'yeast','high_salt':'high_salt',
                 'warm':'heat','cold':'cold','bright':'light',
                 'cva':'pheromone','courtship_pulse':'song','auditory':'song'}
PARTS={'foreleg_touch':'foreleg','midleg_touch':'midleg','hindleg_touch':'hindleg'}
_OBJECT_WORDS={alias:kind for kind,names in ALIASES.items() for alias in names}
_PLACE=re.compile(r'(?:请)?(?:在)?(?P<side>左右两侧|左右两边|两侧|双侧|两边|左右|左边|右边|左侧|右侧|前方|前面|后方|后面)?'
                  r'(?:放置|放|摆放|加入|添加|有|给它|给果蝇|给)?(?:一块|一些|一个)?'
                  r'(?P<object>'+ '|'.join(re.escape(x) for x in sorted(_OBJECT_WORDS,key=len,reverse=True))+r')')


def parse_message(message,catalogue):
    request=parse_scene_request(message)
    if request:return {'kind':'scene_reference','scene_request':request}
    text=message.strip().strip('。！？?!') if isinstance(message,str) else ''
    match=_PLACE.fullmatch(text)
    if match:
        side={'左边':'left','左侧':'left','右边':'right','右侧':'right','后方':'back','后面':'back',
              **dict.fromkeys(('左右两侧','左右两边','两侧','双侧','两边','左右'),'bilateral')}.get(match['side'],'front')
        return {'kind':'world_objects','objects':[{'type':_OBJECT_WORDS[match['object']],'side':side,'strength':1.}],
                'items':[],'unresolved':[]}
    return parse_input(message,catalogue,include_side_metadata=True)


def ground_message(raw,environment,probe,catalogue):
    if raw['kind']=='scene_reference':
        return ground_scene_request(raw['scene_request'],environment,probe,catalogue),environment
    plan=copy.deepcopy(raw)
    if plan['kind'] in ('question','unsupported'):return plan,environment
    proposals=plan.pop('items',[])
    objects=list(plan.pop('objects',[]));presentations=[];unresolved=[]
    for item in proposals:
        channel=item['channel'];strength=item.get('intensity',1.)
        if channel in PARTS:
            sides=('left','right') if item.get('side','bilateral')=='bilateral' else (item['side'],)
            train={'taps':item['taps']} if item.get('taps',1)>1 else {}
            if train and item.get('interval_ms') is not None:
                train['interval_ms']=item['interval_ms']
            presentations.extend({'kind':'touch','side':side,'body_part':PARTS[channel],
                'strength':strength,'onset_ms':item.get('onset_ms',50.),'duration_ms':item.get('duration_ms',300.),
                **train} for side in sides)
        elif channel in OBJECT_CHANNELS and not item.get('body_part'):
            if item.get('onset_ms',50.)!=50. or item.get('duration_ms',300.)!=300.:
                unresolved.append('物品的定时出现/消失尚未实现，请在环境编辑器中应用或移除。')
            # The sensory parser defaults absent sides to bilateral; physical
            # placement defaults to the front only when no side was expressed.
            side=item.get('side','front') if item.get('side_explicit',True) else 'front'
            objects.append({'type':OBJECT_CHANNELS[channel],'side':side,'strength':strength})
        else:
            unresolved.append(channel+' 尚无对应的物理环境呈现；未改用直接神经刺激。')
    plan.update(items=[],presentations=presentations,unresolved=unresolved,input_mode='environment_only')
    plan['assumptions']=['正常聊天只改变或采样物理环境；感觉事件由实际眼图、感受器位置和碰撞决定。',
        '方向到对侧腿触碰仍是工程交流约定，物理探针不追踪身体、不保证接触，也不保证转向。',
        '物品默认放在指定方位 3 mm 处；未指定方位时放前方。应用物品更新时保留当前神经、身体与控制器状态。']
    objects=[dict(obj,side=side) for obj in objects
             for side in (('left','right') if obj['side']=='bilateral' else (obj['side'],))]
    candidate=copy.deepcopy(environment);description=[]
    for obj in objects:
        side=obj['side'];kind=obj['type']
        offset={'left':(0,3),'right':(0,-3),'back':(-3,0)}.get(side,(3,0))
        frame=probe['frame'];idx=probe['body_names'].index('c_thorax')
        w,x,y,z=frame['body_quaternions_wxyz'][idx]
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        dx,dy=offset;origin=frame['position_mm']
        position=[origin[0]+math.cos(yaw)*dx-math.sin(yaw)*dy,origin[1]+math.sin(yaw)*dx+math.cos(yaw)*dy,.2]
        if not TYPES[kind]['solid']:position[2]=max(.5,origin[2])
        oid='chat_'+kind+'_'+side
        placed={'id':oid,'type':kind,'position_mm':position,'size_mm':[.6,.6,.4],
                'strength':obj['strength'],'range_mm':6.}
        candidate['objects']=[o for o in candidate['objects'] if o['id']!=oid]+[placed]
        description.append('在'+{'left':'左侧','right':'右侧','back':'后方'}.get(side,'前方')+'环境放置'+TYPES[kind]['label']+'（距身体约 3 mm）')
    if objects:
        candidate['objects']=validate_objects(candidate['objects'])
        if candidate['objects']!=environment['objects']:candidate['revision']+=1
    for p in presentations:
        description.append('用真实碰撞探针尝试触碰'+('左' if p['side']=='left' else '右')+{'foreleg':'前腿','midleg':'中腿','hindleg':'后腿'}[p['body_part']])
    if unresolved:
        plan.update(kind='unsupported',presentations=[],reason='；'.join(unresolved))
        return plan,environment
    plan.update(kind='environment_presentation' if objects or presentations else 'scene_observation',
                reason='已转换为环境呈现；是否收到感觉由物理采样决定。')
    plan['translation']={'role':'human_to_environment','forces_body_action':False,
        'description':'；'.join(description) or '继续采样当前实际环境',
        'contact_required':True,'direct_neural_input':False}
    return plan,candidate
