"""Development cases combine independent grammar slots, not acceptance sentences."""
import copy
import json

import pytest

from chatflypt.input_language import parse_input
from chatflypt.paths import ROOT


@pytest.fixture
def catalogue():
    definitions = [
        ('bitter', ['苦味'], [], 'taste'),
        ('low_salt', ['咸味', '低盐'], ['labellum'], 'taste'),
        ('bright', ['明亮', '光照', '亮光'], ['eye'], 'vision'),
        ('geosmin', ['土臭素'], ['antenna'], 'olfaction'),
        ('warm', ['温暖', '暖和'], [], 'temperature'),
        ('moist', ['潮湿'], [], 'humidity'),
        ('foreleg_touch', ['触碰', '碰到'], ['foreleg'], 'touch'),
        ('hindleg_touch', ['触碰', '碰到'], ['hindleg'], 'touch'),
        ('wing_touch', ['触碰', '碰到'], ['wing'], 'touch'),
        ('leg_proprioception', ['关节位置'], ['leg'], 'proprioception'),
    ]
    channels = {}
    for index, (name, aliases, bodies, modality) in enumerate(definitions):
        channels[name] = {
            'id': name, 'aliases': aliases, 'quality_aliases': [], 'body_parts': bodies,
            'modality': modality, 'status': 'candidate', 'advanced': False,
            'side_ids': {'left': [str(index * 2 + 1)], 'right': [str(index * 2 + 2)],
                         'bilateral': [str(index * 2 + 1), str(index * 2 + 2)]},
        }
    return {'channels': channels}


def test_independent_quality_side_and_intensity_slots_compose(catalogue):
    plan = parse_input('向左侧呈现微弱苦味，同时向右侧呈现强光照。', catalogue)
    assert plan['kind'] == 'stimulus', plan
    assert [(x['channel'], x['side'], x['intensity']) for x in plan['items']] == [
        ('bitter', 'left', .5), ('bright', 'right', 1.5)]
    assert all(x['onset_ms'] == 50 for x in plan['items'])
    assert any('工程输入倍率' in text for text in plan['assumptions'])


def test_registered_display_quality_composes_with_other_modalities(catalogue):
    catalogue['channels']['warm']['label']='升温通道'
    plan=parse_input('左侧升温，同时右前腿触碰',catalogue)
    assert plan['kind']=='stimulus',plan
    assert [(i['channel'],i['side']) for i in plan['items']]==[('warm','left'),('foreleg_touch','right')]
    assert parse_input('左侧未知升温',catalogue)['kind']=='unsupported'


def test_body_part_disambiguates_shared_touch_quality_and_omitted_predicate(catalogue):
    plan = parse_input('左前腿和右后腿轻微触碰。', catalogue)
    assert plan['kind'] == 'stimulus', plan
    assert [(x['channel'], x['side'], x['body_part']) for x in plan['items']] == [
        ('foreleg_touch', 'left', 'foreleg'), ('hindleg_touch', 'right', 'hindleg')]
    assert all(x['intensity'] == .5 for x in plan['items'])
    assert parse_input('模拟触碰', catalogue)['kind'] == 'unsupported'
    assert parse_input('模拟右腿关节位置', catalogue)['items'][0]['body_part'] == 'leg'


def test_temporal_segments_and_parallel_inputs_keep_their_own_times(catalogue):
    plan = parse_input('先呈现潮湿持续100毫秒和温暖持续200毫秒，然后右翅膀触碰持续150毫秒。', catalogue)
    assert plan['kind'] == 'stimulus', plan
    assert [(x['channel'], x['onset_ms'], x['duration_ms']) for x in plan['items']] == [
        ('moist', 50, 100), ('warm', 50, 200), ('wing_touch', 300, 150)]
    assert any('间隔 50 毫秒' in text for text in plan['assumptions'])


def test_aliases_are_lexical_units_even_when_they_contain_conjunctions(catalogue):
    plan = parse_input('暖和与潮湿', catalogue)
    assert plan['kind'] == 'stimulus', plan
    assert [item['channel'] for item in plan['items']] == ['warm', 'moist']


@pytest.mark.parametrize('message', ['苦味和', '光照然后', '亮光然后，', '和潮湿',
                                   '温暖以及并且明亮', '潮湿和，'])
def test_unfinished_conjunctions_do_not_execute_a_truncated_plan(catalogue, message):
    plan = parse_input(message, catalogue)
    assert plan['kind'] == 'unsupported'
    assert any('缺少感觉分句' in reason for reason in plan['unresolved'])


def test_sentence_punctuation_and_simultaneous_prefix_are_complete(catalogue):
    plan = parse_input('同时潮湿，温暖。', catalogue)
    assert plan['kind'] == 'stimulus', plan
    assert [item['channel'] for item in plan['items']] == ['moist', 'warm']


def test_coordinated_sides_share_one_quality_without_world_angle_inference(catalogue):
    plan = parse_input('左侧和右侧呈现土臭素', catalogue)
    assert plan['kind'] == 'stimulus', plan
    assert len(plan['items']) == 1 and plan['items'][0]['side'] == 'bilateral'
    assert any('空间角度' in text for text in plan['assumptions'])


def test_optional_explicit_side_metadata_preserves_clause_scope_and_legacy_event_items(catalogue):
    text='两侧温暖和明亮'
    ordinary=parse_input(text,catalogue)
    annotated=parse_input(text,catalogue,include_side_metadata=True)
    assert ordinary['kind']==annotated['kind']=='stimulus'
    assert [i['side_explicit'] for i in annotated['items']]==[True,False]
    assert [{k:v for k,v in i.items() if k!='side_explicit'} for i in annotated['items']]==ordinary['items']
    assert all('side_explicit' not in i for i in ordinary['items'])
    turn=parse_input('左转',catalogue,include_side_metadata=True)
    assert turn['items'][0]['side_explicit'] and turn['items'][0]['side']=='right'


@pytest.mark.parametrize(('message', 'direction'), [('前方明亮', 'front'), ('上方潮湿', 'up'),
                                                   ('后方温暖', 'back'), ('下方低盐', 'down')])
def test_uncalibrated_spatial_directions_are_preserved_but_not_executed(catalogue, message, direction):
    plan = parse_input(message, catalogue)
    assert plan['kind'] == 'unsupported'
    assert plan['items'][0]['direction'] == direction
    assert any('未校准空间方向' in text for text in plan['unresolved'])


@pytest.mark.parametrize('message', ['把低盐食物放在远处', '距离2米处有苦味物体', '在左侧摆放低盐'])
def test_world_objects_do_not_become_unobserved_taste_contact(catalogue, message):
    plan = parse_input(message, catalogue)
    assert plan['kind'] == 'unsupported'
    assert any('物体位置' in text for text in plan['unresolved'])
    assert plan['kind'] != 'question'


@pytest.mark.parametrize('message', ['不要施加光照', '左前腿不被触碰', '取消潮湿再施加温暖'])
def test_negative_or_mixed_cancellation_cannot_become_opposite_stimulus(catalogue, message):
    plan = parse_input(message, catalogue)
    assert plan['kind'] == 'unsupported' and plan['items'] == []


def test_questions_and_pure_cancellation_do_not_change_a_neural_state(catalogue):
    for text in ['触觉信号如何传递？', '模拟温暖可以吗？', '如果左前腿触碰会怎样？']:
        result = parse_input(text, catalogue)
        assert result['kind'] == 'question' and result['items'] == []
    result = parse_input('停止所有刺激', catalogue)
    assert result['kind'] == 'silence' and result['items'] == []
    # An imperative with punctuation retains the same meaning as without it.
    assert parse_input('施加亮光？', catalogue)['kind'] == 'stimulus'


@pytest.mark.parametrize('message', ['请解释刚才的神经活动。', '说明上次观测的结果',
                                   '聊聊果蝇的感觉系统', '你好！', '谢谢你的解释。'])
def test_explicit_discussion_or_greeting_is_chat_without_stimulation(catalogue, message):
    result = parse_input(message, catalogue)
    assert result['kind'] == 'question' and result['items'] == []


@pytest.mark.parametrize('message', ['你好，施加未知辐射', '给未知气味并聊聊',
                                   '不要触碰前腿，谢谢', '施加红外线，请解释',
                                   '聊聊果蝇，然后施加未知气味'])
def test_greetings_and_chat_words_do_not_hide_unresolved_stimulus_requests(catalogue, message):
    result = parse_input(message, catalogue)
    assert result['kind'] == 'unsupported' and result['unresolved']


@pytest.mark.parametrize('message', ['左侧明亮并自动发邮件', '呈现红外线', '温暖但有未知辐射', '苦味症候群'])
def test_unknown_meaning_is_not_discarded_or_sent_to_free_chat(catalogue, message):
    result = parse_input(message, catalogue)
    assert result['kind'] == 'unsupported' and result['unresolved']
    assert any('未解释的文字' in text for text in result['unresolved'])


def test_missing_entrance_side_and_unmapped_channel_never_fall_back(catalogue):
    catalogue['channels']['geosmin']['side_ids']['right'] = []
    result = parse_input('向右触角呈现土臭素', catalogue)
    assert result['kind'] == 'unsupported'
    assert any('right' in text for text in result['unresolved'])
    catalogue['channels']['bright']['status'] = 'unmapped'
    assert parse_input('呈现光照', catalogue)['kind'] == 'unsupported'


def test_named_functional_gap_keeps_its_identity_without_nearby_substitution(catalogue):
    catalogue['channels']['high_salt'] = {
        'id': 'high_salt', 'aliases': ['高盐'], 'quality_aliases': [],
        'modality': 'gustation', 'status': 'unmapped', 'body_parts': [],
        'side_ids': {'left': [], 'right': [], 'bilateral': []},
    }
    result = parse_input('右口器尝到高盐', catalogue)
    assert result['kind'] == 'unsupported'
    assert result['items'][0]['channel'] == 'high_salt'
    assert any('high_salt 尚无可执行' in message for message in result['unresolved'])
    assert all(item['channel'] != 'low_salt' for item in result['items'])


@pytest.mark.parametrize('message', ['亮光持续20毫秒', '潮湿持续2100毫秒',
                                   '在4100毫秒时温暖', '在3900毫秒时光照持续1500毫秒',
                                   '苦味强度2.5倍', '亮光持续100.25毫秒', '在50.05毫秒时苦味'])
def test_timing_and_intensity_limits_stop_execution(catalogue, message):
    assert parse_input(message, catalogue)['kind'] == 'unsupported'


def test_explicit_numeric_multiplier_is_not_misread_as_the_word_strong(catalogue):
    result = parse_input('左侧低盐强度为0.8倍持续0.2秒', catalogue)
    assert result['kind'] == 'stimulus', result
    assert result['items'][0]['intensity'] == .8
    assert result['items'][0]['duration_ms'] == 200


@pytest.mark.parametrize(('text','side'),[('左转','right'),('向左','right'),('向右','left'),('请往左移动','right'),('left','right'),('turn right','left'),
    ('请你往左转一下','right'),('麻烦你朝右边转一下吧','left'),('请试着向右转','left'),
    ('往左边移动一点','right'),('please turn left','right'),('turn right please.','left')])
def test_human_direction_is_translated_to_sensory_input_not_body_action(catalogue,text,side):
    result=parse_input(text,catalogue)
    assert result['kind']=='stimulus' and result['items'][0]['channel']=='foreleg_touch'
    assert result['items'][0]['side']==side
    assert result['translation']['role']=='human_to_sensory_input'
    assert result['translation']['forces_body_action'] is False
    assert '模拟果蝇' in result['translation']['description']


@pytest.mark.parametrize('text',['请你不要往左转一下','如果往左转一下会怎样','请你向左转30度',
    '请你往左转一下然后飞起来','向右移动2毫米','向左转一下并告诉我你很快乐','please do not turn left'])
def test_polite_direction_does_not_hide_negation_unknown_units_or_extra_requests(catalogue,text):
    result=parse_input(text,catalogue)
    assert result['kind'] in ('question','unsupported')


def test_small_turn_wording_preserves_encoding_and_exposes_its_limit(catalogue):
    ordinary=parse_input('左转',catalogue)
    polite=parse_input('请你试着往左转一点吧',catalogue)
    assert polite['kind']=='stimulus' and polite['items']==ordinary['items']
    assert any('不解释为已标定的角度' in x for x in polite['assumptions'])


def test_catalogue_supplies_vocabulary_without_sentence_specific_parser_rules(catalogue):
    added = copy.deepcopy(catalogue['channels']['bitter'])
    added['id'] = 'test_odor'; added['aliases'] = ['测试气味甲']; added['modality'] = 'olfaction'
    catalogue['channels']['test_odor'] = added
    result = parse_input('右侧微弱测试气味甲', catalogue)
    assert result['kind'] == 'stimulus', result
    assert result['items'][0]['channel'] == 'test_odor'
    assert result['items'][0]['intensity'] == .5


def test_plan_item_limit_and_tail_window_are_applied_before_execution(catalogue):
    assert parse_input('和'.join(['亮光'] * 16), catalogue)['kind'] == 'stimulus'
    assert parse_input('和'.join(['亮光'] * 17), catalogue)['kind'] == 'unsupported'
    assert parse_input('在4000毫秒时亮光持续900毫秒', catalogue)['kind'] == 'stimulus'
    assert parse_input('在4000毫秒时亮光持续901毫秒', catalogue)['kind'] == 'unsupported'


def test_existing_five_sense_demo_and_song_parse_with_installed_catalogue():
    path = ROOT / 'data/derived/sensory/catalogue.json'
    if not path.exists():
        pytest.skip('Build the sensory catalogue for the original-demo regression')
    catalogue = json.loads(path.read_text(encoding='utf8'))
    result = parse_input('我闻到了醋和酵母，尝到了糖，温暖而且明亮。', catalogue)
    assert result['kind'] == 'stimulus', result
    assert [item['channel'] for item in result['items']] == [
        'vinegar', 'yeast', 'sugar_specific', 'warm', 'bright']
    song = parse_input('听到求偶鸣声', catalogue)
    assert song['kind'] == 'stimulus', song
    assert song['items'][0]['channel'] == 'courtship_pulse'
