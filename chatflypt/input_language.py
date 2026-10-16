"""Composable, bounded language adapter for a sensory catalogue.

This module emits proposed simulation inputs, never decoder targets. A caller
must execute only ``kind == 'stimulus'`` or ``kind == 'silence'``. Items retained
in an unsupported result are explanatory previews, not executable instructions.
Anatomical aliases come from the catalogue; grammar handles side, body part,
relative intensity, parallel inputs and temporal ordering independently.
"""
from collections import defaultdict
from collections.abc import Mapping
import math
import re
import unicodedata


_BODY = {
    'foreleg': ('前腿', '前足', '前肢'),
    'midleg': ('中腿', '中足', '中肢'),
    'hindleg': ('后腿', '后足', '后肢'),
    'head': ('头部', '头'), 'eye': ('复眼', '眼部', '眼睛'),
    'wing': ('翅膀', '翅部', '翅'), 'body': ('躯干', '身体表面', '体壁'),
    'leg': ('腿部', '腿', '足部'), 'haltere': ('平衡棒',),
    'antenna': ('触角',), 'labellum': ('口器', '唇瓣', '喙'),
}
_SIDE = {
    'bilateral': ('左右两侧', '左右两边', '左右侧', '两侧', '双侧', '两边', '双边', '左右'),
    'left': ('左侧', '左边', '左'), 'right': ('右侧', '右边', '右'),
}
_DIRECTION = {
    'front': ('正前方', '前方', '前面', '前侧'),
    'back': ('正后方', '后方', '后面', '后侧'),
    'up': ('上方', '上面', '上侧'), 'down': ('下方', '下面', '下侧'),
}
_INTENSITY = {
    .25: ('非常微弱', '极弱'), .5: ('微弱', '轻微', '轻轻', '弱', '轻'),
    1.: ('中等', '适中', '正常'), 1.5: ('强烈', '强', '重'),
    2.: ('非常强', '极强', '很强'),
}
_QUESTION = re.compile(r'如果|假如|假设|是否|为什么|为何|怎样|如何|什么|能否|'
                       r'哪里|哪儿|哪个|哪些|哪边|哪侧|多少|'
                       r'能不能|可不可以|可以吗|吗(?:[。！？?!\s]*$)|\b(?:why|how|whether|if)\b')
_DISCUSSION = re.compile(r'^(?:请你|请|麻烦你|我想)?(?:解释|说明|介绍|聊聊|讲讲|谈谈|分析)')
_ADDITIONAL_INPUT = re.compile(r'(?:然后|接着|随后|再|并且|同时|以及|并|[，,;；])\s*'
                               r'(?:请)?(?:施加|模拟|输入|呈现|给予|播放|释放|触碰|给)')
_GREETING = re.compile(r'(?:你好|您好|嗨|早上好|下午好|晚上好|再见|hello|hi|thanks|'
                       r'谢谢(?:你|你的解释|你的说明|你的帮助)?|多谢)')
_NEGATION = re.compile(r'不要|不再|不许|不必|不用|不能|没有|没|禁止|别|不|撤销|撤回|取消|停止|移除|'
                       r'\b(?:not|never|without|cancel|stop)\b')
_WORLD_SCENE = re.compile(r'远处|远方|距离|物体|放在|放置|摆在|摆放|搁在|附近有|'
                         r'\d+(?:\.\d+)?\s*(?:厘米|毫米|米)(?!秒)')
_SEQUENCE = re.compile(r'然后|接着|随后|再|\bthen\b')
_PARALLEL = re.compile(r'[，,、；;]|以及|并且|而且|同时|和|与|又|且|并|\band\b')
_PUNCTUATION = re.compile(r'[\s，。！？、；：,.!?;:（）()“”"「」『』=]+')
_TURN_ZH = re.compile(r'(?:请(?:你)?|麻烦(?:你)?)?(?:试着|试试)?(?:向|往|朝)?'
                      r'([左右])(?:边|侧)?(?:转弯|转|走|移动)?(?:一下|一点点|一点)?(?:吧)?')
_TURN_EN = re.compile(r'(?:please\s+)?(?:turn\s+)?(left|right)(?:\s+please)?')
_TIME_UNIT = r'(毫秒|ms|秒|s)'
_DURATION = re.compile(r'(?:持续|维持|时长(?:为)?|持续时间(?:为)?)\s*(\d+(?:\.\d+)?)\s*' + _TIME_UNIT)
_ONSET = re.compile(r'(?:在|从)\s*(\d+(?:\.\d+)?)\s*' + _TIME_UNIT + r'\s*(?:时|开始)')
_DELAY = re.compile(r'(?:延迟|等待|间隔)\s*(\d+(?:\.\d+)?)\s*' + _TIME_UNIT)
_NUMERIC_INTENSITY = re.compile(r'(?:强度|倍率)(?:为|是|=)?\s*(\d+(?:\.\d+)?)\s*(?:倍)?')
# Conspecific contact is a train of taps, not one long press. A count word or
# digit plus a classifier selects a train; the train's timing stays an
# engineering parameter set (see physical_touch).
_TAP_WORDS = {'一': 1, '两': 2, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8}
_TAPS = re.compile(r'(?:连续|连着|连|重复|轻轻)?\s*(?:触碰|轻触|敲击|碰|敲|拍)?\s*'
                   r'(?P<count>[一二两三四五六七八]|\d{1,2})\s*(?:次|下|遍|回)(?:触碰|轻触|敲击|碰|敲|拍)?')
_MODALITY_TERMS = {
    'vision': ('看到', '看见', '视觉'),
    'olfaction': ('闻到', '闻见', '嗅觉', '气味'),
    'taste': ('尝到', '尝出', '味觉'),
    'audition': ('听到', '听见', '听觉'),
    'touch': ('触觉',), 'proprioception': ('本体感觉',),
    'temperature': ('温度',), 'humidity': ('湿度',),
}
_MODALITY_IDS = {
    'vision': {'bright', 'vision_r7', 'vision_r8', 'vision_dra', 'vision_ocelli'},
    'olfaction': {'vinegar', 'yeast', 'geosmin', 'cva', 'co2'},
    'taste': {'sweet_water', 'bitter', 'low_salt'},
    'audition': {'auditory'}, 'temperature': {'warm', 'cold', 'evaporative_cooling'},
    'humidity': {'moist', 'dry'},
    'touch': {'head_touch', 'eye_touch', 'foreleg_touch', 'midleg_touch',
              'hindleg_touch', 'wing_touch', 'body_touch'},
    'proprioception': {'chordotonal_proprioception', 'hair_plate_proprioception',
                       'campaniform_proprioception', 'haltere_proprioception', 'leg_proprioception'},
}
_FILLERS = (
    '向指定入口', '向感觉入口', '实验呈现', '感觉输入', '感觉刺激', '输入刺激',
    '我感受到', '我感觉到', '我感到', '我闻到', '我尝到', '我听到', '我看到',
    '果蝇感受到', '果蝇感觉到', '让果蝇', '给果蝇', '这只果蝇', '一只果蝇',
    '模拟果蝇', '给它', '给你', '向它', '向你', '它的', '你的',
    '感受到', '感觉到', '感到', '闻到', '闻见', '尝到', '尝出', '听到', '听见',
    '看到', '看见', '受到', '接收到', '接收', '模拟', '呈现', '施加', '给予',
    '提供', '播放', '释放', '输入', '出现', '环境中', '环境', '感觉',
    '入口', '刺激', '信号', '一点点', '一些', '一点', '请你', '请',
    '有人', '被', '的', '了', '到', '在', '向', '从', '有', '是', '很', '稍微',
)


def _normal(text):
    return unicodedata.normalize('NFKC', text).casefold()


def _result(kind, items=None, reason='', assumptions=None, unresolved=None):
    return {'kind': kind, 'items': items or [], 'reason': reason,
            'assumptions': list(dict.fromkeys(assumptions or [])),
            'unresolved': list(dict.fromkeys(unresolved or []))}


def _channel_map(catalogue):
    raw = catalogue.get('channels', {}) if isinstance(catalogue, Mapping) else {}
    if isinstance(raw, list):
        return {str(spec['id']): spec for spec in raw if isinstance(spec, Mapping) and 'id' in spec}
    return {str(key): value for key, value in raw.items() if isinstance(value, Mapping)} if isinstance(raw, Mapping) else {}


def _pattern(alias):
    escaped = re.escape(alias)
    return r'(?<![a-z0-9_])' + escaped + r'(?![a-z0-9_])' if alias.isascii() else escaped


def _lexicon(channels):
    aliases = defaultdict(set)
    for channel, spec in channels.items():
        display=spec.get('label','') if not spec.get('advanced',False) else ''
        # Registered display names are also usable vocabulary. Remove only
        # generic taxonomy suffixes, never an unknown substantive modifier.
        short=re.sub(r'(?:候选通道|候选群|感觉群|通道|输入|候选)$','',display) if isinstance(display,str) else ''
        for alias in [channel, display, short, *spec.get('aliases', []), *spec.get('quality_aliases', [])]:
            if isinstance(alias, str) and alias.strip():
                aliases[_normal(alias.strip())].add(channel)
    return aliases


def _matches(text, lexicon):
    """Longest non-overlapping lexemes; duplicate aliases retain all candidates."""
    found = []
    for alias, labels in lexicon.items():
        for match in re.finditer(_pattern(alias), text):
            found.append({'span': match.span(), 'text': alias, 'labels': set(labels)})
    kept = []
    for item in sorted(found, key=lambda row: (-(row['span'][1] - row['span'][0]), row['span'][0])):
        if not any(item['span'][0] < previous['span'][1] and previous['span'][0] < item['span'][1]
                   for previous in kept):
            kept.append(item)
    return sorted(kept, key=lambda row: row['span'][0])


def _slot_matches(text, mapping):
    aliases = defaultdict(set)
    for key, words in mapping.items():
        for word in words:
            aliases[word].add(key)
    return _matches(text, aliases)


def _mark(mask, span):
    mask[span[0]:span[1]] = [True] * (span[1] - span[0])


def _split(text, separator, lexicon, unresolved=None):
    protected = [item['span'] for item in _matches(text, lexicon)]
    start, pieces, has_left, pending = 0, [], False, None
    for match in separator.finditer(text):
        if any(a <= match.start() < b for a, b in protected):
            continue
        piece = text[start:match.start()].strip()
        pieces.append(piece)
        if _PUNCTUATION.sub('', piece):
            has_left, pending = True, None
        connector = _PUNCTUATION.sub('', match.group())
        if connector:
            # Punctuation may terminate a sentence, but a written conjunction
            # requires a following clause. "同时" also starts a complete plan.
            if unresolved is not None:
                if pending is not None:
                    unresolved.append(f'连接词 {pending} 与 {connector} 之间缺少感觉分句。')
                elif not has_left and connector != '同时':
                    unresolved.append(f'连接词 {connector} 前缺少感觉分句。')
            pending = connector
        start = match.end()
    tail = text[start:].strip()
    pieces.append(tail)
    if pending is not None and not _PUNCTUATION.sub('', tail) and unresolved is not None:
        unresolved.append(f'连接词 {pending} 后缺少感觉分句。')
    return [piece for piece in pieces if _PUNCTUATION.sub('', piece)]


def _milliseconds(match):
    number, unit = float(match.group(1)), match.group(2)
    return number * (1000 if unit in {'秒', 's'} else 1)


def _family(channel, spec):
    for family, ids in _MODALITY_IDS.items():
        if channel in ids:
            return family
    aliases = {'visual': 'vision', 'olfactory': 'olfaction', 'gustatory': 'taste', 'gustation': 'taste',
               'auditory': 'audition', 'thermosensory': 'temperature',
               'hygrosensory': 'humidity', 'tactile': 'touch'}
    modality = str(spec.get('modality', ''))
    return aliases.get(modality, modality)


def presentation_end(item):
    """Last simulated millisecond an item can still be active, including taps."""
    taps = item.get('taps', 1)
    spacing = item.get('interval_ms', item.get('duration_ms', 300.) + 80.)
    return item.get('onset_ms', 50.) + item.get('duration_ms', 300.) + (taps - 1) * spacing


def _validate_item(item, channels, unresolved):
    channel, side = item['channel'], item['side']
    spec = channels[channel]
    if spec.get('status') == 'unmapped':
        unresolved.append(f'{channel} 尚无可执行的神经入口映射。')
    body = item.get('body_part')
    targets = spec.get('side_ids', {})
    if body:
        by_part = spec.get('body_part_ids', {})
        if body in by_part:
            targets = by_part[body]
        elif body not in spec.get('body_parts', []):
            unresolved.append(f'{channel} 没有 {body} 部位映射。')
        elif len(spec.get('body_parts', [])) > 1:
            unresolved.append(f'{channel} 尚不能单独选择 {body} 的入口。')
    if not targets.get(side):
        unresolved.append(f'{channel} 没有已标定身份的 {side} 侧入口，未改用另一侧。')
    direction = item.get('direction')
    if direction and direction not in spec.get('calibrated_directions', []):
        unresolved.append(f'{channel} 未校准空间方向 {direction} 到神经入口的映射。')
    if not math.isfinite(item['intensity']) or not 0 <= item['intensity'] <= 2:
        unresolved.append('强度倍率必须在 0 到 2 之间。')
    if not 30 <= item['duration_ms'] <= 2000:
        unresolved.append('每项持续时间必须为 30 到 2000 毫秒（触碰序列支持短敲击）。')
    taps = item.get('taps', 1)
    if not isinstance(taps, int) or isinstance(taps, bool) or not 1 <= taps <= 8:
        unresolved.append('触碰次数必须为 1 到 8 次。')
    if taps > 1:
        spacing = item.get('interval_ms', item['duration_ms'] + 80.)
        if not math.isfinite(spacing) or spacing < item['duration_ms'] + 20 or spacing > 1000:
            unresolved.append('敲击间隔必须大于单次触碰时长且不超过 1000 毫秒。')
    if not 0 <= item['onset_ms'] <= 4000 or presentation_end(item) + 100 > 5000:
        unresolved.append('开始时间须为 0 到 4000 毫秒，含 100 毫秒尾窗的整段仿真不得超过 5000 毫秒。')
    for key in ['onset_ms', 'duration_ms']:
        value = item[key]
        if not math.isfinite(value) or abs(round(value / .1) * .1 - value) > 1e-8:
            unresolved.append('开始时间和持续时间必须对齐 0.1 毫秒的仿真时钟。')


def _intensities(text):
    numeric = list(_NUMERIC_INTENSITY.finditer(text))
    words = [row for row in _slot_matches(text, _INTENSITY)
             if not any(a.start() <= row['span'][0] < a.end() for a in numeric)]
    values = set().union(*(row['labels'] for row in words)) if words else set()
    values.update(float(match.group(1)) for match in numeric)
    return values, [row['span'] for row in words] + [match.span() for match in numeric]


def _parse_piece(text, mentions, channels, onset, assumptions, unresolved, inherited_text='', side_metadata=False):
    mask = [False] * len(text)
    for mention in mentions:
        if not mention.get('inherited'):
            _mark(mask, mention['span'])
    slots = {}
    for name, mapping in [('side', _SIDE), ('body', _BODY), ('direction', _DIRECTION),
                          ('modality', _MODALITY_TERMS)]:
        rows = _slot_matches(text, mapping)
        slots[name] = set().union(*(row['labels'] for row in rows)) if rows else set()
        for row in rows:
            _mark(mask, row['span'])
    sides = slots['side']
    side = next(iter(sides)) if len(sides) == 1 else 'bilateral'
    if not sides:
        assumptions.append('未指定入口侧别时，向目录中左右已知入口的并集呈现。')
    elif len(sides) > 1:
        assumptions.append('同一并列感觉项中的左右入口按两侧同时呈现。')
    bodies, directions = slots['body'], slots['direction']
    if len(bodies) > 1:
        unresolved.append(f'“{text}”包含多个身体部位，请用并列项分别描述。')
    if len(directions) > 1:
        unresolved.append(f'“{text}”包含多个未区分的空间方向。')
    intensity_values, intensity_spans = _intensities(text)
    for span in intensity_spans:
        _mark(mask, span)
    if not intensity_values and inherited_text:
        intensity_values = _intensities(inherited_text)[0]
    if len(intensity_values) > 1:
        unresolved.append(f'“{text}”包含相冲突的强度，请分成并列项。')
    intensity = next(iter(intensity_values)) if len(intensity_values) == 1 else 1.
    duration_values, onset_values, delay_values = [], [], []
    for pattern, values in [(_DURATION, duration_values), (_ONSET, onset_values), (_DELAY, delay_values)]:
        for match in pattern.finditer(text):
            values.append(_milliseconds(match))
            _mark(mask, match.span())
        if not values and inherited_text and pattern is _DURATION:
            values.extend(_milliseconds(match) for match in pattern.finditer(inherited_text))
    if any(len(set(values)) > 1 for values in [duration_values, onset_values, delay_values]):
        unresolved.append(f'“{text}”中的时间参数不唯一。')
    tap_values = []
    for match in _TAPS.finditer(text):
        token = match.group('count')
        count = _TAP_WORDS.get(token, int(token) if token.isdigit() else None)
        if count is not None:
            tap_values.append(count)
        _mark(mask, match.span())
    if len(set(tap_values)) > 1:
        unresolved.append(f'“{text}”中的触碰次数不唯一。')
    taps = tap_values[0] if tap_values else 1
    # A tap is a brief contact; an unqualified touch press stays at 300 ms, but
    # an unqualified tap train defaults to a short engineering tap length.
    default_duration = 60. if taps > 1 else 300.
    duration = duration_values[0] if duration_values else default_duration
    interval = None
    if taps > 1 and delay_values:
        # In a tap train an explicit 间隔 is the onset-to-onset spacing, not a
        # delay of the whole train.
        interval = delay_values[0]
        delay_values = []
    start = (onset_values[0] if onset_values else onset) + (delay_values[0] if delay_values else 0)
    for filler in sorted(_FILLERS, key=len, reverse=True):
        for match in re.finditer(re.escape(filler), text):
            _mark(mask, match.span())
    remainder = ''.join(' ' if used else char for char, used in zip(text, mask))
    unknown = _PUNCTUATION.sub(' ', remainder).split()
    if unknown:
        unresolved.append('未解释的文字：' + '、'.join(unknown) + '。')
    if not mentions:
        unresolved.append(f'“{text}”未指定目录支持的感觉品质。')
        return []
    selected = []
    for mention in mentions:
        candidates = set(mention['labels'])
        if bodies and len(candidates) > 1:
            candidates = {channel for channel in candidates if bodies.intersection(
                set(channels[channel].get('body_parts', [])) | set(channels[channel].get('body_part_ids', {})))}
        if slots['modality']:
            compatible = {channel for channel in candidates if _family(channel, channels[channel]) in slots['modality']}
            if len(candidates) == 1 and not compatible:
                unresolved.append(f'“{mention["text"]}”的通道不属于句中指定的感觉系统。')
            else:
                candidates = compatible
        ordinary = {channel for channel in candidates if not channels[channel].get('advanced')}
        if len(candidates) > 1 and ordinary:
            candidates = ordinary
        if len(candidates) != 1:
            unresolved.append(f'感觉品质“{mention["text"]}”与部位/感觉系统未能唯一对应通道。')
            continue
        channel = next(iter(candidates))
        if channel in selected:
            continue
        selected.append(channel)
    items = []
    for channel in selected:
        item = {'channel': channel, 'side': side, 'intensity': intensity,
                'onset_ms': start, 'duration_ms': duration}
        if taps > 1:
            item['taps'] = taps
            if interval is not None:
                item['interval_ms'] = interval
        if side_metadata:
            item['side_explicit'] = bool(sides)
        if len(bodies) == 1:
            item['body_part'] = next(iter(bodies))
        if directions:
            item['direction'] = next(iter(directions)) if len(directions) == 1 else sorted(directions)
        _validate_item(item, channels, unresolved)
        items.append(item)
    return items


def parse_input(text, catalogue, *, include_side_metadata=False):
    """Parse bounded slot combinations without supplying labels to any decoder.

    Questions and cancellations contain no input items. Unsupported plans may
    retain interpretable items for display but must not execute. Intensity is an
    engineering rate multiplier, not a concentration, force or brightness unit.
    World-object placement and uncalibrated spatial directions are unresolved.
    Environment translation may request explicit-side metadata. The default
    event-encoder item schema remains unchanged for independent experiments.
    """
    if not isinstance(text, str) or not text.strip():
        return _result('unsupported', reason='请输入感觉描述。', unresolved=['输入为空。'])
    text = _normal(text.strip())
    if len(text) > 1000:
        return _result('unsupported', reason='输入过长。', unresolved=['每次输入最多 1000 字符。'])
    compact = _PUNCTUATION.sub('', text)
    if _DISCUSSION.match(compact) and _ADDITIONAL_INPUT.search(text):
        return _result('unsupported', reason='请将自由对话与新的刺激请求分开发送。',
                       unresolved=['检测到对话后的新增刺激指令，未将其忽略或交给自由问答执行。'])
    if _QUESTION.search(text) or _DISCUSSION.match(compact) or _GREETING.fullmatch(compact):
        return _result('question', reason='这是对话、问题或假设，不改变神经仿真状态。')
    if re.fullmatch(r'(?:请)?(?:停止|取消|撤销|撤回|移除)(?:当前|所有|全部)?(?:感觉)?(?:刺激|输入)', compact) or compact in {
            '保持安静', '无输入', '不输入', 'silence'}:
        return _result('silence', reason='停止施加输入。',
                       assumptions=['本次不施加感觉输入；不把它解释为果蝇停止运动。'])
    if _NEGATION.search(text):
        return _result('unsupported', reason='检测到否定或撤销内容，未施加刺激。',
                       unresolved=['否定不转换成相反刺激；混合撤销与新增输入需要分开发送。'])
    channels = _channel_map(catalogue)
    if not channels:
        return _result('unsupported', reason='感觉目录不可用。', unresolved=['没有可查询的感觉通道。'])
    assumptions = ['简单感觉陈述按“向指定感觉入口呈现刺激”的实验指令解释，不声称物体已被真实接触。',
                   '左右表示入口的解剖侧别；不自动换算现实声源、光源或物体的空间角度。',
                   '强弱是工程输入倍率：默认 1、弱 0.5、强 1.5；未校准物理强度。']
    unresolved, items = [], []
    direction = _TURN_ZH.fullmatch(compact)
    english_direction = _TURN_EN.fullmatch(re.sub(r'[.!?。！？]+$', '', text).strip())
    if direction or english_direction:
        left = direction.group(1) == '左' if direction else english_direction.group(1)=='left'
        if 'foreleg_touch' not in channels:
            return _result('unsupported', reason='语言翻译所需感觉通道不可用。', unresolved=['缺少前腿触碰通道。'])
        item = {'channel': 'foreleg_touch', 'side': 'right' if left else 'left',
                'body_part': 'foreleg', 'intensity': 1., 'onset_ms': 50., 'duration_ms': 300.}
        if include_side_metadata:
            item['side_explicit'] = True
        _validate_item(item, channels, unresolved)
        assumptions.append('人类方向表达译为对侧前腿触碰；这是感觉编码约定，不是给身体控制器的转向命令。')
        if any(word in compact for word in ('一下','一点','试着','试试')):
            assumptions.append('“一下／一点／试着”表示一次尝试；仍使用同一感觉提示，不解释为已标定的角度、距离或更弱刺激。')
        result = _result('unsupported' if unresolved else 'stimulus', [item],
                         '已将人类方向表达译为感觉输入。', assumptions, unresolved)
        result['proposed'] = True
        result['translation'] = {'source_text': text, 'description': '模拟果蝇的'+('右' if left else '左')+'前腿被触碰',
            'role': 'human_to_sensory_input', 'mapping_status': 'engineering_proposal',
            'forces_body_action': False}
        return result
    if _WORLD_SCENE.search(text):
        unresolved.append('这是物体位置、距离或摆放场景，尚不能自动转换成感觉入口刺激；请明确实际感知或入口呈现。')
    # Coordinated side slots may precede a shared noun without forming two inputs.
    text = re.sub(r'左(?:侧|边)?\s*(?:和|与)\s*右(?:侧|边)?', '两侧', text)
    text = re.sub(r'右(?:侧|边)?\s*(?:和|与)\s*左(?:侧|边)?', '两侧', text)
    text = re.sub(r'^(?:请)?先', '', text)
    lexicon = _lexicon(channels)
    stages = _split(text, _SEQUENCE, lexicon, unresolved)
    onset, preceding, preceding_text = 50., [], ''
    for stage in stages:
        pieces = _split(stage, _PARALLEL, lexicon, unresolved)
        mentions_by_piece = [_matches(piece, lexicon) for piece in pieces]
        stage_items = []
        for index, piece in enumerate(pieces):
            mentions = mentions_by_piece[index]
            inherited_text = ''
            if not mentions and (_slot_matches(piece, _BODY) or _slot_matches(piece, _SIDE)):
                neighbors = sorted((j for j, rows in enumerate(mentions_by_piece) if rows),
                                   key=lambda j: abs(index - j))
                source = mentions_by_piece[neighbors[0]] if neighbors else preceding
                if source:
                    mentions = [dict(row, inherited=True) for row in source]
                    inherited_text = pieces[neighbors[0]] if neighbors else preceding_text
                    assumptions.append('并列或相邻时段中省略的感觉品质、强度和时长按相邻明示品质补全，侧别和部位保持各自描述。')
            stage_items.extend(_parse_piece(piece, mentions, channels, onset, assumptions, unresolved,
                                             inherited_text, side_metadata=include_side_metadata))
            if mentions:
                preceding = mentions
                preceding_text = piece
        items.extend(stage_items)
        if stage_items:
            onset = max(presentation_end(item) for item in stage_items) + 50.
    if len(stages) > 1:
        assumptions.append('时间先后按上一段结束后间隔 50 毫秒开始下一段；每项默认持续 300 毫秒。')
    else:
        assumptions.append('未指定时间时从 50 毫秒开始，每项持续 300 毫秒。')
    if not items and not unresolved:
        unresolved.append('没有可执行的感觉项。')
    if len(items) > 16:
        unresolved.append('每次最多执行 16 个感觉项；请拆分当前输入计划。')
    kind = 'unsupported' if unresolved else 'stimulus'
    return _result(kind, items, '存在未解决的输入含义，未施加刺激。' if unresolved else '已解析为明确的感觉输入计划。',
                   assumptions, unresolved)
