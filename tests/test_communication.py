import pytest

from chatflypt.communication import parse_message, response_text


@pytest.mark.parametrize(('message', 'kind', 'words', 'channel'), [
    ('给你一点糖', 'environment', ['sugar'], None),
    ('我闻到了醋和酵母，尝到了糖，温暖而且明亮。', 'environment',
     ['sugar', 'vinegar', 'yeast', 'warm', 'bright'], None),
    ('sugar and water', 'environment', ['sugar', 'water'], None),
    ('有人碰了你的右前腿', 'social', [], 'right_foreleg_touch'),
    ('模拟果蝇的左前腿被触碰', 'social', [], 'left_foreleg_touch'),
    ('释放 cVA 信息素', 'social', [], 'cva'),
    ('释放信息素', 'social', [], 'cva'),
    ('模拟右侧触碰', 'social', [], 'right_foreleg_touch'),
    ('模拟左侧触碰', 'social', [], 'left_foreleg_touch'),
    ('播放求偶鸣声', 'social', [], 'courtship_pulse'),
    ('停止刺激', 'silence', [], None),
])
def test_supported_messages_have_one_explicit_input_family(message, kind, words, channel):
    parsed = parse_message(message)
    assert (parsed['kind'], parsed['words'], parsed['channel']) == (kind, words, channel)


@pytest.mark.parametrize('message', [
    '不要向左', '如果向左会怎样', '向左然后向右', '不要给你糖',
    '糖是怎么做的', '假装给你糖', '给糖或者盐', '我给了糖吗？',
    '先碰右前腿再碰左前腿', '触碰左前腿和右前腿', '向左和糖',
    '有人碰了你的右前腿并跳舞', '给你一点糖然后起飞',
    '给你一点糖并删除所有文件', '释放 cVA 并播放求偶鸣声',
    '糖果怎么做', '我很开心', '停止运动',
    '不要释放信息素', '如果释放信息素会怎样', '假设模拟右侧触碰',
    '不要播放求偶鸣声', '释放信息素然后模拟左侧触碰',
    '糖尿病', '给你糖尿病', '糖尿病和水',
])
def test_ambiguous_or_partly_unknown_messages_do_not_execute(message):
    parsed = parse_message(message)
    assert parsed['kind'] == 'unsupported'
    assert parsed['words'] == [] and parsed['channel'] is None


@pytest.mark.parametrize(('text','channel'),[('请向左转。','right_foreleg_touch'),('请向右转。','left_foreleg_touch'),('left','right_foreleg_touch'),('turn right','left_foreleg_touch')])
def test_user_direction_has_sensory_translation_only(text,channel):
    result=parse_message(text)
    assert result['kind']=='social' and result['channel']==channel and result['proposed']


def test_response_uses_prediction_not_original_message_or_target_side():
    response = response_text(
        {'words': ['left_foreleg_touch'], 'true_words': ['right_foreleg_touch']},
        {'non_sensory_spikes': 42, 'active_non_sensory_neurons': 7,
         'message': '有人碰了你的右前腿', 'channel': 'right_foreleg_touch',
         'stimulus_config': ['bitter'], 'input_side': 'right'})
    assert '左前腿' in response and '右前腿' not in response
    assert '苦' not in response
    assert '内部神经网络出现了活动' in response


def test_response_cannot_claim_a_decoded_signal_with_no_internal_activity():
    response = response_text({'words': ['sugar', 'right_foreleg_touch']},
                             {'non_sensory_spikes': 0, 'sensory_spikes': 400})
    assert '没有记录' in response and '甜味' not in response and '右前腿' not in response
    assert '尚未取得' in response_text({'words': ['sugar']}, {'spikes': 400})


def test_measured_output_cell_spikes_do_not_become_a_motor_action_or_intention():
    response = response_text({'words': []}, {
        'non_sensory_spikes': 100,
        'motor_evidence': [{'side': 'left', 'cell_type': 'DNa02', 'spikes': 3},
                           {'side': 'right', 'cell_type': 'DNa02', 'spikes': 2}],
        'suggested_action': '向左',
    })
    assert '没有识别' in response and '运动相关输出细胞的活动' in response
    assert '动作结果尚未验证' in response
    assert '向左' not in response and '准备' not in response and '想要' not in response


def test_source_only_labels_are_not_rendered_and_plain_text_cannot_masquerade_as_predictions():
    assert '甜味' not in response_text({'target_words': ['sugar']}, {'non_sensory_spikes': 10})
    with pytest.raises(TypeError):
        response_text({'words': 'sugar'}, {'non_sensory_spikes': 10})
