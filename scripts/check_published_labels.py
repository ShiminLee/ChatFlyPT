"""Live check: what does the newly published readout actually say in a fresh session?"""
import json
import sys
import threading
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatflypt.environment import empty_environment
from chatflypt.server import Application, create_server


def main():
    app = Application()
    app.environment = empty_environment()
    app._publish_environment = lambda candidate: setattr(app, 'environment', candidate)
    server = create_server(0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = 'http://127.0.0.1:' + str(server.server_address[1])
    sid = uuid.uuid4().hex
    try:
        for index, message in enumerate(('左边有点热', '右侧有酵母')):
            request = urllib.request.Request(base + '/api/converse',
                data=json.dumps({'message': message, 'session_id': sid, 'seed': 924900 + index}).encode('utf8'),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.load(response)
            if not result.get('observations'):
                print(json.dumps({'message': message, 'handled': result.get('handled'),
                                  'input_kind': (result.get('input_plan') or {}).get('kind'),
                                  'reply_source': result.get('reply_source'),
                                  'reply': (result.get('reply') or '')[:160]}, ensure_ascii=False), flush=True)
                continue
            observation = result['observations'][0]
            print(json.dumps({'message': message,
                              'decoded_words': observation['decoded']['words'],
                              'sensation': (observation.get('sensation') or {}).get('text'),
                              'sensation_labels': (observation.get('sensation') or {}).get('labels'),
                              'intent': ((observation.get('scene') or {}).get('intent') or {}).get('text'),
                              'command': ((observation.get('scene') or {}).get('command') or {}).get('text'),
                              'readout_model': observation['readout_model']}, ensure_ascii=False), flush=True)
    finally:
        app.reset_session(sid)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == '__main__':
    main()
