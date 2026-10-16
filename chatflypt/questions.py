"""Question routing over measured readouts: answer only what was actually measured.

A recognised question is routed to the one readout that measures its topic in the
last recorded window, and the answer is that readout's own sentence with its own
qualifiers and abstentions. A topic the record does not measure is refused
explicitly - never answered from the message, an intention, the scene plan or a
guess. An unrecognised question keeps the earlier behaviour: re-express the last
blind readout and say that it is a re-expression, not an answer.

Routing is not comprehension: it selects a measurement, it does not understand
the question as language. The chosen topic and the readout it used are always
reported with the answer.
"""
import re

TOPICS = (
    {'id': 'motion', 'label': '实测身体运动', 'slot': 'scene.intent',
     'keywords': ('在动', '走动', '走路', '移动', '走了', '动吗', '停下', '停着', '速度'),
     'note': '它读的是实测身体运动（走/停），不是输入意图。'},
    {'id': 'steering', 'label': '自身下发的转向指令', 'slot': 'scene.command',
     'keywords': ('往哪转', '想往哪', '转向', '拐', '转吗', '转不转'),
     'note': '它读的是它自己下发的转向指令，未经身体行为验证。'},
    {'id': 'sensation', 'label': '盲神经感觉读出', 'slot': 'sensation',
     'keywords': ('感觉到', '感觉', '闻到', '尝到', '看到', '听到', '痛', '刺激',
                  '冷不冷', '热不热', '甜', '气味'),
     'note': '它读的是内部活动对应的有限感觉标签，不是主观体验。'},
    {'id': 'contact', 'label': '实测接触', 'slot': 'contacts',
     'keywords': ('碰到', '接触', '摸到', '踩到', '撞到', '碰到了'),
     'note': '它报的是物理探针实测到的接触次数，不是接触强度或它的反应。'},
    {'id': 'body', 'label': '实测身体状态', 'slot': 'body',
     'keywords': ('身体', '姿态', '腿怎么', '翅膀', '站姿', '站稳'),
     'note': '身体由固定工程控制器驱动，这一项是实测位姿，不算神经指令证据。'},
)
LIMITATIONS = [
    '问题路由只选择一项实测读出，不代表问题被当作语言理解了。',
    '读出弃权时问题也弃权：没有实测记录就不给答案，也不从问句本身推断。',
]

_AMBIGUOUS_MOTION = ('走动', '停着', '运动')


def match_topic(question):
    """Longest keyword match wins, so '往哪转' is not read as a motion question."""
    if not isinstance(question, str):
        return None
    compact = re.sub(r'\s+', '', question)
    best = None
    for topic in TOPICS:
        for keyword in topic['keywords']:
            if keyword in compact and (best is None or len(keyword) > best[1]):
                best = (topic, len(keyword), keyword)
    return None if best is None else {'topic': best[0], 'keyword': best[2]}


def _measurement(observation, topic_id):
    """The measured record behind one topic, or why it is missing."""
    scene = observation.get('scene') or {}
    if topic_id == 'motion':
        block = scene.get('intent') or {}
        return {'status': block.get('status'), 'text': block.get('text'),
                'source': 'blind_motor_readout', 'measured_outcome': block.get('measured_outcome'),
                'labels': block.get('labels'), 'prediction': block.get('prediction'),
                'reason': block.get('reason')}
    if topic_id == 'steering':
        block = scene.get('command') or {}
        note = block.get('command_readout') or {}
        return {'status': block.get('status'), 'text': block.get('text'),
                'source': 'blind_command_readout', 'labels': block.get('labels'),
                'prediction': block.get('prediction'), 'reason': block.get('reason') or note.get('reason'),
                'withheld_direction': note.get('withheld_direction'),
                'served_directions': note.get('served_directions')}
    if topic_id == 'sensation':
        block = observation.get('sensation') or {}
        return {'status': block.get('status'), 'text': block.get('text'),
                'source': 'blind_neural_readout', 'labels': block.get('labels'),
                'reason': block.get('reason')}
    if topic_id == 'contact':
        presentation = (observation.get('reception') or {}).get('physical_presentation') or {}
        if not presentation:
            return {'status': 'unavailable', 'text': None, 'source': 'measured_presentation',
                    'reason': '本轮没有施加物理探针，因此没有可报告的实测接触。'}
        contacts = presentation.get('measured_leg_contacts')
        bouts = presentation.get('measured_contact_bouts')
        if presentation.get('contact_recorded'):
            text = (f'本轮物理探针实测到 {contacts} 次腿接触记录（{bouts} 段接触），'
                    '这是实测接触，不代表接触强度或它的反应。')
        else:
            text = '本轮物理探针没有实测到任何腿接触（探针已发出，未测得碰撞）。'
        return {'status': 'decoded', 'text': text, 'source': 'measured_presentation',
                'presentation': presentation}
    if topic_id == 'body':
        body = scene.get('body') or {}
        summary = body.get('summary') or {}
        if not body.get('physics_run'):
            return {'status': 'unavailable', 'text': None, 'source': 'measured_body',
                    'reason': '本轮没有物理仿真记录，因此没有可报告的身体实测。'}
        text = (f"本轮实测位移 {summary.get('displacement_mm')} mm、路径 {summary.get('path_length_mm')} mm，"
                f"最大接触数 {summary.get('max_contacts')}；身体由固定工程控制器驱动，姿态未做生理标定。")
        return {'status': 'decoded', 'text': text, 'source': 'measured_body', 'summary': summary,
                'limitations': body.get('limitations')}
    raise ValueError('Unknown question topic')


def route_question(question, observation):
    """Answer a question from one measured readout, or refuse it explicitly."""
    result = {'schema': 'chatflypt.question-route.v1', 'question': question,
              'status': 'unrouted', 'topic': None, 'matched_keyword': None, 'text': None,
              'source': None, 'measurement': None, 'llm_called': False,
              'scope': 'Selects one measured readout; it does not understand the question as language',
              'limitations': list(LIMITATIONS)}
    matched = match_topic(question)
    if matched is None:
        return result
    topic = matched['topic']
    result.update(topic=topic['id'], topic_label=topic['label'], matched_keyword=matched['keyword'])
    if not isinstance(observation, dict):
        result.update(status='no_measurement',
                      text='还没有可引用的实测记录，所以这一项没有答案。')
        return result
    measurement = _measurement(observation, topic['id'])
    result['measurement'] = measurement
    result['source'] = measurement.get('source')
    result['observation_id'] = observation.get('id')
    if measurement.get('text'):
        result.update(status='answered',
                      text=f"（据上一轮实测记录 · {topic['label']}）{measurement['text']} {topic['note']}")
        return result
    # A readout that says nothing has three distinct reasons; report the right one
    # instead of collapsing them into "not qualified".
    reason = measurement.get('reason')
    if not reason and measurement.get('status') == 'decoded':
        prediction = measurement.get('prediction')
        threshold = (prediction or {}).get('informative_abs_command') if isinstance(prediction, dict) else None
        reason = ('本轮该读出的预测幅度低于信息阈值，没有达到可发音的强度。'
                  if threshold is not None else '本轮该读出没有给出可发音的结果。')
    reason = reason or '本轮该读出弃权（没有达到合格条件）。'
    result.update(status='no_measurement',
                  text=f"（{topic['label']}）本轮没有可用的实测结果：{reason} {topic['note']}")
    return result
