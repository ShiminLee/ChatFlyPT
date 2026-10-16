"""Live acceptance for question routing: real HTTP, answers only from measurements.

One statement turn produces the observation, then several questions are asked
against it. Checks that a routed answer cites a measured readout, that a topic the
record cannot answer refuses instead of inventing one, and that no language model
is involved.
"""
import json
import sys
import threading
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatflypt.environment import empty_environment
from chatflypt.paths import ROOT, dump
from chatflypt.server import Application, create_server

QUESTIONS = ('它在动吗？', '它想往哪转？', '它感觉到什么了吗？')


def main():
    app = Application()
    app.environment = empty_environment()
    app._publish_environment = lambda candidate: setattr(app, 'environment', candidate)
    server = create_server(0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = 'http://127.0.0.1:' + str(server.server_address[1])
    sid = uuid.uuid4().hex
    rows = []

    def post(path, payload):
        request = urllib.request.Request(base + path, data=json.dumps(payload).encode('utf8'),
                                        headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    try:
        statement = post('/api/converse', {'message': '右侧有酵母', 'session_id': sid, 'seed': 924780})
        observation = statement['observations'][0]
        assert observation['readout_model'] == 'general_sensory_unified'
        observation_id = observation['id']
        for index, question in enumerate(QUESTIONS):
            result = post('/api/converse', {'message': question, 'session_id': sid,
                                            'observation_id': observation_id,
                                            'seed': 924790 + index})
            route = result.get('question_route') or {}
            assert route.get('status') in ('answered', 'no_measurement', 'unrouted'), result.get('reply')
            assert not result.get('llm_called') and not result.get('external_action_control')
            if route.get('status') == 'answered':
                assert route['source'] in ('blind_motor_readout', 'blind_command_readout',
                                           'blind_neural_readout', 'measured_presentation', 'measured_body')
                assert '据上一轮实测记录' in route['text']
            else:
                assert '没有可用的实测结果' in route['text'] or route.get('text') is None
            rows.append({'question': question, 'status': route.get('status'), 'topic': route.get('topic'),
                         'source': route.get('source'), 'reply': result.get('reply'),
                         'reply_source': result.get('reply_source'),
                         'neural_simulation_run': result.get('neural_simulation_run')})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        topics = {row['topic'] for row in rows if row['topic']}
        assert len(topics) >= 2, 'questions did not route to distinct readouts'
        dump(ROOT / 'reports/command-readout/question-route-acceptance.json',
             {'schema': 'chatflypt.question-route-acceptance.v1', 'real_http': True, 'single_individual': True,
              'llm_called': False, 'observation_id': observation_id, 'turns': rows,
              'biologically_calibrated': False})
        print('routed topics:', sorted(topics))
    finally:
        app.reset_session(sid)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == '__main__':
    main()
