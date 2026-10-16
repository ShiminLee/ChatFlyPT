"""Sensory input -> recorded neural activity -> expression, without an adviser."""
import uuid

from .paths import ROOT, dump


def validate_options(backend, feedback_enabled, seed):
    # Old clients may still send these fields; they cannot enable an adviser.
    if backend not in ('configured', 'local', 'off'):
        raise ValueError('Unknown legacy backend')
    if not isinstance(feedback_enabled, bool):
        raise ValueError('feedback_enabled must be boolean')
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError('Seed must be a nonnegative 32-bit integer')


def looks_like_question(text):
    """Question shape only decides whether the measured-readout router may answer.

    It is never taken as understanding: an unrecognised question still gets the
    parser's refusal, and a routed topic must have its own measurement.
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if stripped.endswith(('?', '？')):
        return True
    return any(marker in stripped for marker in ('吗', '呢', '什么', '怎么', '哪', '多少', '是否'))


def run_turn(app, *, kind='message', message='', words=None, items=None, seed=42,
             history=None, observation_id='', body_context='ground',
             backend='off', feedback_enabled=False, environment=None, environment_sample=None,session_id=None):
    from .dialogue import observation_sensation
    from .expression import expression_report
    from .server import validate_body_context
    validate_options(backend, feedback_enabled, seed)
    validate_body_context(body_context)
    if kind == 'message':
        session_options={'session_id':session_id} if session_id else {}
        scene_request=app.prepare_scene_message(message,seed,body_context,**session_options)
        plan = scene_request['plan']
        if plan['kind'] == 'question':
            return app.answer_question(message, history, observation_id)
        if plan['kind'] == 'unsupported':
            # A question about a word the stimulus parser cannot resolve ("它想往哪转？"
            # reads as a turn instruction) is still a question: let the measured-readout
            # router answer it, and only then fall back to the parser's refusal.
            if observation_id and looks_like_question(message):
                routed = app.answer_question(message, history, observation_id)
                route = routed.get('question_route') or {}
                if route.get('status') == 'answered' or (route.get('status') == 'no_measurement'
                                                         and route.get('topic')):
                    return routed
            return {'handled': False, 'input_plan': plan, 'reply_source': 'input_status',
                    'neural_simulation_run': False, 'reply': plan['reason']}
        extra={'environment':scene_request['environment'],'environment_sample':scene_request['sample']} if scene_request else {}
        result = app.stimulate([], seed, input_plan=plan, intent=message, body_context=body_context,
                              input_mode='environment_only',presentations=plan.get('presentations',[]),**extra,**session_options)
    elif kind in ('stimuli','environment'):
        extra = {'environment':environment,'environment_sample':environment_sample} if environment is not None else {}
        if kind=='environment' and session_id:extra['session_id']=session_id
        result = app.stimulate([] if items is None else items, seed, body_context=body_context,
                              intent='环境改动' if kind=='environment' else None,
                              input_mode='environment_only' if kind=='environment' else 'neural_intervention', **extra)
    elif kind == 'words':
        observation = app.observe([] if words is None else words, seed, body_context=body_context)
        result = {'handled': True, 'neural_simulation_run': True, 'observation': observation,
                  'observations': [observation], 'reception': observation['reception'],
                  'pathway': {'closed_loop': False}}
    else:
        raise ValueError('Choose message, stimuli or words')

    observations = result.get('observations') or [result['observation']]
    messages = []
    intent_messages = []
    for observation in observations:
        observation['dialogue_stage'] = 'initial'
        observation['sensation'] = observation_sensation(observation)
        observation['reception']['sensation'] = observation['sensation']
        observation['expression'] = expression_report(observation)
        observation['fly_text'] = observation['expression']['text']
        messages.extend(observation['expression']['messages'])
        # The motor-side readout is a separate statement: it predicts the body's
        # own measured motion and never repeats the human message as intent.
        intention = (observation.get('scene') or {}).get('intent') or {}
        observation['intent'] = intention
        if intention.get('text'):
            intent_messages.append({'kind': 'intent', 'text': intention['text'],
                'source': intention.get('source'), 'status': intention.get('status'),
                'labels': intention.get('labels', []), 'dataset': observation.get('dataset'),
                'observation_id': observation.get('id')})
    dialogue = {'id': uuid.uuid4().hex, 'backend': 'off', 'max_feedback_rounds': 0,
                'status': 'neural_only', 'initial_observation_ids': [obs['id'] for obs in observations],
                'feedback_enabled': False, 'feedback': {'status': 'removed'},
                'llm_called': False, 'external_action_control': False,
                'expression_source': 'blind_neural_readout',
                'body_source': 'recorded_motor_activity',
                'reset_between_trials': not all(o.get('embodied_feedback',{}).get('between_messages') for o in observations),
                'session_id':session_id,'embodied_neural_feedback': all(bool(o.get('embodied_feedback')) for o in observations)}
    reply = '\n'.join(m['text'] for m in messages) if len(observations)==1 else '\n'.join(
        m['dataset']+'：'+m['text'] for m in messages)
    result.update(reply_source='neural_expression' if messages else 'expression_status', messages=messages, dialogue=dialogue,
                  observations=observations, reply=reply or observations[0]['expression']['reason'])
    result['intent'] = intent_messages[0] if len(intent_messages) == 1 else None
    result['intent_messages'] = intent_messages
    result['pathway']['intent_readout_completed'] = bool(intent_messages)
    result['pathway']['closed_loop'] = dialogue['embodied_neural_feedback']
    result['pathway']['closed_loop_scope'] = ('persistent_session_measured_feedback' if not dialogue['reset_between_trials'] else 'within_trial_measured_feedback') if dialogue['embodied_neural_feedback'] else None
    result['pathway']['feedback_interval_ms_by_dataset']={o['dataset']:o.get('embodied_feedback',{}).get('interval_ms') for o in observations}
    result['pathway']['sensory_processing_completed'] = result['pathway'].get('completed',True)
    result['pathway']['expression_translation_completed'] = bool(messages)
    result['pathway']['readout_qualified'] = all(o['expression']['status']=='decoded' for o in observations)
    result['pathway']['completion_scope'] = 'bounded_neural_readout_turn'
    result['pathway']['completed'] = bool(result['pathway']['sensory_processing_completed'] and messages and result['pathway']['readout_qualified'])
    result['pathway']['communication_round_trip'] = kind=='message' and result['pathway']['completed']
    if kind=='environment':
        result['pathway']['stages']=['applied_scene','physics_receptors','sensory_events','connectome_activity','sensory_readout']
    if kind=='message' and scene_request:
        from .scene_language import evaluate_goal
        result['scene_request']=plan.get('scene_request');result['goal_evaluation']=evaluate_goal(plan,observations)
        result['pathway']['scene_grounding_completed']=plan['kind']!='unsupported'
        result['pathway']['stages']=['human_language','environment_presentation','physics_receptors','sensory_events','connectome_activity','sensory_readout','recorded_body']
        scene_status='已重新采样环境并记录神经与身体响应。'
        if result['goal_evaluation']:
            checks=result['goal_evaluation']['trials']
            reports=[]
            for check in checks:
                distance=f"{check['final_distance_mm']:.2f}"
                motor='；没有形成控制器可用的运动命令' if check['motor_command_steps']==0 else ''
                reports.append(check['dataset']+'：与目标中心距离 '+f"{check['initial_distance_mm']:.2f}"+' → '+distance+' mm，'+('有目标物品接触' if check['target_contacts'] else '没有目标物品接触')+motor)
            cue='已按目标方位呈现一次物理探针触碰，接收情况见实测碰撞。' if plan.get('presentations') else '目标在身体前方；当前只呈现实际环境，未编码前进命令。'
            body_report='；'.join(reports)+'。尚不能把距离变化解释为理解或完成了指令。' if reports else '本轮没有可用于判断接近或接触目标的有效身体轨迹。'
            scene_status=cue+body_report
        if plan['unresolved']:scene_status='；'.join(plan['unresolved'])+' 已观察当前环境；未虚构目标。'
        result['status_messages']=[scene_status]
        if scene_request.get('changed'):app.commit_message_environment(scene_request,result)
    if kind in ('stimuli','words'):
        result['experimental_neural_intervention']=True
        result['status_messages']=['实验性直接神经干预：此记录不是环境自然感知，也不是正常聊天通路。']
    result['dialogue_record'] = f"reports/dialogue/{dialogue['id']}.json"
    dump(ROOT / result['dialogue_record'], result)
    return result
