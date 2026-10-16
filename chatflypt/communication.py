"""Bounded message parsing and Chinese rendering of measured neural readouts.

The parser proposes model inputs; it is not a general language-to-biology
translator. The response renderer deliberately has no original-message or
stimulus-configuration argument. Only predicted labels and measured internal
activity may determine its answer.
"""
from collections.abc import Mapping
import math
import re
import unicodedata

from .stimuli import WORDS


_ENVIRONMENT_ALIASES = {
    'sugar': ('糖', '甜味', '甜的', 'sugar'),
    'bitter': ('苦味', '苦的', '苦', 'bitter'),
    'water': ('水', 'water'),
    'salt': ('盐味', '咸味', '盐', 'salt'),
    'vinegar': ('醋味', '醋', 'vinegar'),
    'yeast': ('酵母味', '酵母', 'yeast'),
    'geosmin': ('土臭素', '土腥味', '土臭味', 'geosmin'),
    'pheromone': ('pheromone',),
    'warm': ('温暖', '暖和', '变暖', 'warm'),
    'cold': ('寒冷', '冷的', '变冷', '冷', 'cold'),
    'humid': ('潮湿', '湿润', '变湿', 'humid'),
    'dry': ('干燥', '变干', 'dry'),
    'bright': ('明亮', '亮光', '强光', '变亮', 'bright'),
    'touch': ('味觉触毛触碰', '头部味觉触毛触碰', 'touch'),
    'sound': ('声音', '声响', 'sound'),
    'wind': ('吹风', '有风', '风', 'wind'),
}
_SOCIAL_ALIASES = {
    'cva': ('雄性cva信息素', 'cva信息素', 'cva'),
    'courtship_pulse': ('求偶脉冲鸣声', '求偶脉冲声', '求偶鸣声',
                        '求偶歌声', '求偶声音', 'courtship_pulse'),
    'right_foreleg_touch': ('right_foreleg_touch',),
    'left_foreleg_touch': ('left_foreleg_touch',),
}
_FILLERS = (
    '模拟果蝇', '另外一只果蝇', '另一只果蝇', '一只果蝇', '这只果蝇',
    '给果蝇', '给它', '给你', '给我', '让果蝇', '让它', '让你',
    '我想要', '我想', '我闻到了', '我尝到了', '我感受到了',
    '我闻到', '我尝到', '我感到',
    '感受到了', '感觉到了', '听到了', '闻到了', '尝到了',
    '感受到', '感觉到', '听到', '闻到', '尝到', '模拟', '接收到',
    '这里有', '环境中有', '环境', '现在有', '有人', '一点点', '一些',
    '一点', '你的', '它的', '我的', '周围', '气味', '刺激', '信号',
    '输入', '播放', '发出', '释放', '提供', '给予', '以及', '而且',
    '同时', '伴随', '请你', '请', '和', '与', '又', '有', '了', '的',
    '是', '很', '被', '受到', '发生', '触碰', '碰到', '碰了', '碰',
)
_PUNCTUATION = re.compile(r'[\s，。！？、；：,.!?;:（）()“”"「」『』]+')
_UNSAFE_SCOPE = re.compile(
    r'不要|不能|不准|不用|不必|没有|没|未|禁止|别|不|'
    r'如果|假如|假设|假装|想象|要是|例如|比如|是否|会怎样|会发生|'
    r'为什么|为何|怎么|如何|什么|或者|还是|要不要|'
    r'\b(?:not|no|never|without|if|unless|perhaps|maybe|or)\b')
_SEQUENCE = re.compile(r'然后|接着|之后|随后|先.+再|先.+后|\bthen\b')


def _result(kind, *, words=(), channel=None, proposed=False, reason=''):
    return {'kind': kind, 'words': list(words), 'channel': channel,
            'proposed': proposed, 'reason': reason}


def _unsupported(reason):
    return _result('unsupported', reason=reason)


def _matches(text, aliases):
    """Remove complete aliases, keeping English word boundaries intact."""
    found = set()
    pairs = sorted(((alias, label) for label, names in aliases.items()
                    for alias in names), key=lambda item: len(item[0]), reverse=True)
    for alias, label in pairs:
        pattern = re.escape(alias)
        if alias.isascii():
            pattern = r'(?<![a-z0-9_])' + pattern + r'(?![a-z0-9_])'
        text, count = re.subn(pattern, ' ', text)
        if count:
            found.add(label)
    return found, text


def _only_fillers(text):
    for filler in sorted(_FILLERS, key=len, reverse=True):
        text = text.replace(filler, '')
    # Basic English separators are useful when entering the documented labels.
    text = re.sub(r'\band\b', '', text)
    return not _PUNCTUATION.sub('', text)


def parse_message(text):
    """Return an explicit, JSON-ready request for one supported input family.

    ``kind`` is environment, social, silence or unsupported. Environment requests
    contain ``words`` in the established vocabulary order; social requests contain
    ``channel``. ``proposed`` is true only for the tentative direction-to-touch
    code. Negated, hypothetical, sequential and partly unrecognised messages never
    produce an executable request. Silence stops input, not physical movement.
    """
    if not isinstance(text, str) or not text.strip():
        return _unsupported('请输入一个明确的感觉输入。')
    text = unicodedata.normalize('NFKC', text).strip().casefold()
    if len(text) > 280:
        return _unsupported('单次输入过长；请用一句话描述感觉刺激。')
    compact = _PUNCTUATION.sub('', text)
    if compact in {'停止输入', '停止刺激', '安静', '保持安静', '无输入',
                   '不输入', '不施加刺激', 'silence'}:
        return _result('silence', reason='本次不施加感觉输入；这不是停止动作指令。')
    if _UNSAFE_SCOPE.search(text) or '?' in text or '？' in text or text.endswith(('吗', '呢')):
        return _unsupported('这句话包含否定、假设、提问或选择，未施加刺激。')
    if _SEQUENCE.search(text):
        return _unsupported('尚不执行多步刺激；请每次只给出一个输入。')
    # This exact phrase is the existing UI button label for the cVA channel.
    # Do not extend it to arbitrary pheromones or longer, partially parsed text.
    if compact == '释放信息素':
        return _result('social', channel='cva',
                       reason='按现有界面按钮定义使用 cVA 信息素通道；具体刺激显示在接收记录中。')
    direction = re.fullmatch(r'(?:请你|请)?(?:向|往|朝)?([左右])(?:边|侧)?(?:转弯|转|走|移动)?', compact)
    if direction or text in {'left', 'right', 'turn left', 'turn right'}:
        left = direction.group(1) == '左' if direction else text.endswith('left')
        return _result('social', channel='right_foreleg_touch' if left else 'left_foreleg_touch',
                       proposed=True, reason='人类方向表达译为感觉输入；不直接指定身体动作。')

    social, residue = _matches(text, _SOCIAL_ALIASES)
    if 'cva' in social:
        residue = residue.replace('信息素', '')
    # Leg-side words alone do not create an input: an explicit contact verb is required.
    if re.search(r'触碰|碰到|碰了|碰|接触', residue):
        for side, label in [('右', 'right_foreleg_touch'), ('左', 'left_foreleg_touch')]:
            residue, count = re.subn(side + r'(?:侧前腿|前腿|腿|侧)', ' ', residue)
            if count:
                social.add(label)
        residue = residue.replace('接触', '触碰')
    words, residue = _matches(residue, _ENVIRONMENT_ALIASES)
    if not _only_fillers(residue):
        return _unsupported('这句话有尚未定义的含义；请明确描述糖、气味、声音或某侧前腿触碰。')
    if len(social) > 1 or (social and words):
        return _unsupported('暂不混合不同社会通道或社会通道与环境输入，请分开发送。')
    if social:
        return _result('social', channel=next(iter(social)),
                       reason='已解析为显式感觉通道，刺激参数使用当前仿真代理。')
    if words:
        if {'warm', 'cold'} <= words or {'humid', 'dry'} <= words:
            return _unsupported('单次输入包含相反的环境条件，请先明确一种条件。')
        return _result('environment', words=[word for word in WORDS if word in words],
                       reason='已解析为支持的感觉代理标签；未推断未描述的刺激。')
    return _unsupported('这句话没有可执行的感觉输入。')


def _nonnegative_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def response_text(decoded, reaction):
    """Render predicted labels and measured non-sensory activity only.

    ``decoded`` is a mapping with ``words`` (predicted labels, never target labels).
    Besides the 16 environment labels, it may contain the four social channel
    labels. ``reaction`` supplies ``non_sensory_spikes`` and optionally
    ``active_non_sensory_neurons`` and ``motor_evidence`` (rows with ``spikes``).
    Other fields are ignored, including original text, target labels and input
    configuration. Missing/zero internal activity cannot support a felt response.
    No motor intention or completed movement is inferred from output-cell spikes.
    """
    if not isinstance(decoded, Mapping) or not isinstance(reaction, Mapping):
        raise TypeError('decoded and reaction must be mappings')
    spikes = _nonnegative_number(reaction.get('non_sensory_spikes'))
    if spikes is None:
        return '尚未取得可读出的内部神经活动。'
    if spikes == 0:
        return '本次没有记录到可读出的内部神经活动。'
    predicted = decoded.get('words', [])
    if not isinstance(predicted, (list, tuple, set, frozenset)):
        raise TypeError('decoded.words must be a collection of predicted labels')
    words = {word for word in predicted if isinstance(word, str)}
    sentences = []
    tastes = [('sugar', '甜味'), ('bitter', '苦味'), ('water', '水相关味觉信号'), ('salt', '盐味')]
    odors = [('vinegar', '醋相关气味'), ('yeast', '酵母相关气味'), ('geosmin', '土臭素相关气味')]
    conditions = [('warm', '温暖'), ('cold', '寒冷'), ('humid', '潮湿'), ('dry', '干燥'), ('bright', '明亮')]
    for verb, labels in [('我感受到了', tastes), ('我检测到了', odors), ('我感受到环境', conditions)]:
        descriptions = [description for word, description in labels if word in words]
        if descriptions:
            sentences.append(verb + '、'.join(descriptions) + '。')
    if words & {'cva', 'pheromone'}:
        sentences.append('我检测到了 cVA 信息素对应的信号。')
    for label, sentence in [
        ('right_foreleg_touch', '我感到右前腿受到触碰。'),
        ('left_foreleg_touch', '我感到左前腿受到触碰。'),
        ('touch', '我检测到了味觉触毛的触碰信号。'),
        ('courtship_pulse', '我检测到了求偶脉冲声对应的听觉信号。'),
        ('wind', '我检测到了风相关信号。'),
    ]:
        if label in words:
            sentences.append(sentence)
    if 'sound' in words and 'courtship_pulse' not in words:
        sentences.append('我检测到了声音相关信号。')
    if not sentences:
        sentences.append('记录到内部神经活动，但当前读出器没有识别出支持的感觉标签。')
    if sentences[0] != '记录到内部神经活动，但当前读出器没有识别出支持的感觉标签。':
        sentences.append('刺激后，内部神经网络出现了活动。')
    rows = reaction.get('motor_evidence', [])
    if isinstance(rows, (list, tuple)):
        motor_spikes = sum(_nonnegative_number(row.get('spikes')) or 0
                           for row in rows if isinstance(row, Mapping))
        if motor_spikes > 0:
            sentences.append('同时记录到了运动相关输出细胞的活动；动作结果尚未验证。')
    return ''.join(sentences)
