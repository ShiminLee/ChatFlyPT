"""Live acceptance for the steering-command readout over real HTTP chat.

Sends a few messages into one MaleCNS environment-only session and checks the
served line: the block is present, a spoken direction must be one the validation
split supports, the wording must keep saying it is not behaviour-validated, and a
direction that the fit pointed at without support must stay unspoken. Writes
`reports/command-readout/chat-acceptance.json`.
"""
import argparse
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

MESSAGES = ('看看环境', '右边有糖', '左转', '看看环境', '右侧有酵母', '看看环境')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--messages', nargs='*', default=None)
    args = parser.parse_args()
    messages = tuple(args.messages) if args.messages else MESSAGES
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

    def served_line(observation):
        block = observation['scene'].get('command')
        assert isinstance(block, dict), 'the live scene has no command block'
        note = block.get('command_readout') or {}
        served = note.get('served_directions') or {}
        labels = block.get('labels') or []
        assert block['status'] in ('decoded', 'unqualified', 'uncalibrated', 'unavailable'), block['status']
        if block.get('text'):
            assert block['status'] == 'decoded'
            assert '未经身体行为验证' in block['text'], 'a served direction must stay labelled'
            assert labels and all(served.get(label) for label in labels), \
                'spoke a direction the validation split does not support'
            assert 'withheld_direction' not in note, 'spoke while also withholding a direction'
        else:
            assert not labels
        return {'status': block['status'], 'text': block['text'], 'labels': labels,
                'reason': note.get('reason'), 'withheld_direction': note.get('withheld_direction'),
                'served_directions': served,
                'prediction': block.get('prediction'),
                'measured_outcome': block.get('measured_outcome'),
                'limitations': block.get('limitations')}

    try:
        with urllib.request.urlopen(base + '/api/status', timeout=30) as response:
            status = json.load(response)
        assert status['dataset'] == 'MaleCNS v1.0' and status['single_individual']
        assert not status['llm_in_pathway']
        for index, message in enumerate(messages):
            result = post('/api/converse', {'message': message, 'session_id': sid, 'seed': 924750 + index})
            assert result['pathway']['input_mode'] == 'environment_only'
            assert len(result['observations']) == 1 and not result['dialogue']['llm_called']
            observation = result['observations'][0]
            body = observation['scene']['body']
            assert not body['summary']['physics_warnings'] and body['summary']['finite_state']
            row = {'message': message, 'readout_model': observation['readout_model'],
                   'command': served_line(observation),
                   'intent': (observation['scene'].get('intent') or {}).get('text'),
                   'body_commands': body['command_summary'],
                   'record': result['dialogue_record']}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        spoken = [row for row in rows if row['command']['text']]
        dump(ROOT / 'reports/command-readout/chat-acceptance.json',
             {'schema': 'chatflypt.command-chat-acceptance.v1', 'real_http': True,
              'single_individual': True, 'llm_called': False, 'messages': list(messages),
              'windows': len(rows), 'windows_with_spoken_direction': len(spoken),
              'served_directions': (rows[0]['command']['served_directions'] if rows else None),
              'turns': rows, 'biologically_calibrated': False})
        print('windows', len(rows), 'spoken', len(spoken))
    finally:
        app.reset_session(sid)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == '__main__':
    main()
