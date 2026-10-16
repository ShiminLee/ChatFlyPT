"""Question routing: a question is answered only from a measured readout."""
from chatflypt.questions import TOPICS, match_topic, route_question

OBSERVATION = {
    'id': 'obs-1', 'seed': 5,
    'sensation': {'status': 'decoded', 'text': '它好像在闻东西。', 'labels': ['yeast:left']},
    'scene': {
        'intent': {'status': 'decoded', 'text': '候选运动意图（有限验证）：它在走动。',
                   'labels': ['move:active'], 'measured_outcome': {'yaw_change_deg': 0.4, 'path_length_mm': 2.2}},
        'command': {'status': 'decoded', 'labels': [], 'text': None,
                    'command_readout': {'reason': 'No validation window supports this direction; the word is withheld',
                                        'withheld_direction': 'command:turn-left'}},
        'body': {'physics_run': True, 'summary': {'displacement_mm': [0.4, 0.1, 0.0],
                                                  'path_length_mm': 1.6, 'max_contacts': 3},
                 'limitations': ['命令不等于动作。']},
    },
    'reception': {'physical_presentation': {'requested_taps': 3, 'measured_leg_contacts': 2,
                                            'measured_contact_bouts': 1, 'contact_recorded': True}},
}


def test_topics_cover_the_readouts_that_exist():
    ids = {topic['id'] for topic in TOPICS}
    assert ids == {'motion', 'steering', 'sensation', 'contact', 'body'}
    slots = {topic['slot'] for topic in TOPICS}
    assert {'scene.intent', 'scene.command', 'sensation', 'contacts', 'body'} == slots


def test_longest_keyword_wins_so_steering_is_not_read_as_motion():
    assert match_topic('它想往哪转？')['topic']['id'] == 'steering'
    assert match_topic('它在动吗')['topic']['id'] == 'motion'
    assert match_topic('今天天气怎么样') is None


def test_motion_question_answers_from_the_measured_motion_readout():
    route = route_question('它在动吗？', OBSERVATION)
    assert route['status'] == 'answered' and route['topic'] == 'motion'
    assert route['source'] == 'blind_motor_readout'
    assert '它在走动' in route['text'] and '实测' in route['text']
    assert route['measurement']['measured_outcome']['path_length_mm'] == 2.2


def test_a_withheld_direction_refuses_instead_of_inventing_one():
    route = route_question('它想往哪转？', OBSERVATION)
    assert route['status'] == 'no_measurement' and route['topic'] == 'steering'
    assert '没有可用的实测结果' in route['text']
    assert 'withheld' in route['measurement']['reason']


def test_contact_and_body_answers_come_from_measured_records():
    contact = route_question('它碰到什么了吗？', OBSERVATION)
    assert contact['status'] == 'answered' and '2 次腿接触记录' in contact['text']
    body = route_question('它身体怎么样？', OBSERVATION)
    assert body['status'] == 'answered' and '1.6' in body['text'] and '固定工程控制器' in body['text']


def test_missing_observation_and_unrouted_questions_never_guess():
    empty = route_question('它在动吗？', None)
    assert empty['status'] == 'no_measurement' and '还没有可引用的实测记录' in empty['text']
    unrouted = route_question('给我讲个笑话', OBSERVATION)
    assert unrouted['status'] == 'unrouted' and unrouted['text'] is None
    assert unrouted['topic'] is None and unrouted['llm_called'] is False


def test_every_answer_keeps_its_scope_limitations():
    for question in ('它在动吗', '它想往哪转', '它感觉到什么了吗', '它碰到什么了吗', '它身体怎么样'):
        route = route_question(question, OBSERVATION)
        assert route['limitations'], question
        assert '不代表问题被当作语言理解了' in route['limitations'][0]
        assert route['scope'].startswith('Selects one measured readout')
