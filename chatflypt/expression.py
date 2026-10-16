"""Finite neural readout -> language, following the published demonstration.

This renders the saved blind decoder's labels, not the input plan, body commands,
an emitted acoustic signal, or an LLM's guess. First person is a display convention.
"""
from .dialogue import observation_sensation, _ambiguous, _dataset


def expression_report(observation):
    report = observation_sensation(observation)
    result = {
        'schema': 'chatflypt.expression.v2', 'source': 'blind_neural_readout',
        'status': report['status'], 'text': None, 'messages': [], 'labels': [],
        'observation_id': observation.get('id'), 'dataset': _dataset(observation.get('dataset')),
        'activity': report['activity'], 'llm_called': False,
        'scope': 'finite_neural_labels_to_language',
        'outward_signal_decoded': False,
        'limitations': report['limitations'] + [
            '文字来自有限词汇的内部活动读出，不证明任意语言理解、主观体验或真实发声。'],
    }
    reasons = {
        'no_signal': '本轮没有记录到可供翻译的内部神经反应。',
        'uncalibrated': '记录到内部神经反应，但对应读出器尚未校准，暂不能译为具体内容。',
        'unavailable': '本轮没有可用的内部神经读出。',
        'unidentified': '记录到内部神经反应，但有限词汇读出器未识别出内容。',
        'command': '仅读出了历史工程回传命令，不能将它作为果蝇的感觉或行动意图。',
    }
    result['reason'] = reasons.get(report['status'], '')
    if report['status'] not in ('decoded', 'candidate'):
        return result
    if _ambiguous(report):
        result.update(status='ambiguous', reason='神经读出包含相互冲突的感觉标签，暂不生成确定陈述。')
        return result
    # Old command-code experiments must never introduce advice or intent here.
    feelings = [f for f in report['feelings'] if f['channel'] != 'command']
    labels = [w for w in report['labels'] if not w.startswith('command:')]
    slots = observation.get('slots') or (observation.get('scene') or {}).get('slots') or {}
    slot_words = [w for w in slots.get('words', []) if isinstance(w, str)]
    activity = next((w for w in slot_words if w.startswith('activity:')), None)
    pattern = next((w for w in slot_words if w.startswith('pattern:')), None)
    if (activity or pattern) and (feelings or activity or pattern):
        # Compose the sentence from decoded slots; the renderer varies wording but
        # can never add a slot that was not decoded and qualified. Without slot
        # readouts the per-channel template join is kept, so the author-example
        # acceptance sentence stays byte-identical.
        from .generative_renderer import render_state
        rendered = render_state(feelings=feelings, activity=activity, pattern=pattern,
                                seed=int(observation.get('seed') or 0))
        text = rendered['text'] or ''
        result['render'] = {'slots': rendered['slots'], 'reason': rendered['reason'],
                            'source': 'compositional_renderer'}
        result['slot_labels'] = slot_words
    else:
        text = ''.join(dict.fromkeys(f['text'] for f in feelings))
        result['render'] = {'slots': [], 'reason': '仅使用逐通道模板（无可用槽位读出）。',
                            'source': 'per_channel_templates'}
        result['slot_labels'] = []
    if not text:
        result.update(status='unidentified', reason='本轮没有可组合的合格槽位，不生成陈述。')
        return result
    if report['status'] == 'candidate':
        text = ('候选神经读出（连续反馈仅有限验证）：' if report.get('feedback_partially_validated') else '候选神经读出（连续反馈尚未校准）：') + text
    result.update(text=text, labels=labels, reason='已将内部神经读出翻译为有限词汇的自然语言。')
    result['messages'] = [{
        'kind': 'expression', 'text': text, 'source': result['source'],
        'status': result['status'], 'dataset': result['dataset'],
        'observation_id': result['observation_id'], 'labels': labels,
    }]
    return result
