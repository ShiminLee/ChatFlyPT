"""HTTP /api/chat re-reads neural evidence without any model in the path."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from chatflypt import model_runtime, server


@pytest.fixture
def chat_server(monkeypatch):
    calls = []

    class Model:
        def __init__(self, backend):
            self.backend = backend

        def complete(self, user, system):
            calls.append((self.backend, json.loads(user)))
            return {"text": "仅测试回复", "requested_model": self.backend}

    def remote(*args, **kwargs):
        calls.append(("configured_factory", kwargs))
        return Model("configured")

    def local(*args, **kwargs):
        calls.append(("local_factory", kwargs))
        assert kwargs["config"]["OPENAI_API_KEY"] == "local-inference"
        return Model("local")

    monkeypatch.setattr(server, "LLM", remote)
    monkeypatch.setattr(model_runtime, "LLM", local)
    monkeypatch.setattr(model_runtime, "local_model_config", lambda: {
        "OPENAI_API_KEY": "local-inference", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
        "CHATFLYPT_MODEL": "fixture-local", "CHATFLYPT_API_STYLE": "chat_completions"})
    app = server.Application.__new__(server.Application)
    app.observations = {}
    http = server.create_server(0, app=app)
    worker = threading.Thread(target=lambda: http.serve_forever(poll_interval=.01), daemon=True)
    worker.start()

    def post(body):
        request = urllib.request.Request(
            f"http://127.0.0.1:{http.server_address[1]}/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        yield app, calls, post
    finally:
        http.shutdown()
        http.server_close()
        worker.join(timeout=2)


@pytest.mark.parametrize("backend",["configured","local","off"])
def test_http_chat_never_calls_model_even_with_legacy_backend(chat_server,backend):
    app,calls,post=chat_server
    app.observations['saved']={'id':'saved','dataset':'MaleCNS v1.0',
        'decoded':{'words':['foreleg_touch:right'],'scores':{'sugar':100}},
        'reaction':{'non_sensory_spikes':3},'response_text':'RAW_SENTINEL'}
    status,result=post({'message':'我觉得很快乐。你呢？','backend':backend,'observation_id':'saved'})
    assert status==200 and result['sensation']['text']=='我感到右前腿被碰到了。'
    assert result['expression']['text']==result['reply']=='我感到右前腿被碰到了。'
    assert result['messages'][0]['reused_observation'] and result['reused_observation']
    assert calls==[] and not result['llm_called'] and not result['neural_simulation_run']
    assert 'RAW_SENTINEL' not in json.dumps(result) and 'sugar' not in json.dumps(result)

@pytest.mark.parametrize('backend',['unknown','',None,False])
def test_invalid_backend_never_falls_back_to_model(chat_server,backend):
    _,calls,post=chat_server
    status,result=post({'message':'你好。','backend':backend})
    assert status==400 and 'error' in result and not calls

def test_no_observation_does_not_invent_fly_speech(chat_server):
    _,calls,post=chat_server
    status,result=post({'message':'你好。'})
    assert status==200 and result['reply_source']=='input_status'
    assert result['reply']=='还没有可读出的表达记录。' and not calls
