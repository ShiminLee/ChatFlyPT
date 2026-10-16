import json
import os
from types import SimpleNamespace
import threading
import urllib.request
import urllib.error

import pytest
from chatflypt import api_settings as settings, language, server

SECRET = 'unit-test-secret-not-real'


@pytest.fixture
def local_file(tmp_path, monkeypatch):
    path = tmp_path / 'api-settings.json'
    monkeypatch.setattr(settings, 'SETTINGS_FILE', path)
    return path


def configuration(**changes):
    values = {'base_url': 'https://gateway.example', 'api_key': SECRET,
              'model': 'example-model', 'api_style': 'chat_completions'}
    values.update(changes)
    return values


def test_persistence_keeps_key_private_and_blank_edit_preserves_it(local_file):
    public = settings.save_settings(configuration(), {})
    assert SECRET not in json.dumps(public)
    assert public['key_configured'] and 'api_key' not in public
    if os.name == 'nt':
        assert SECRET not in local_file.read_text()
        assert public['storage'] == 'windows-dpapi'
    else:
        assert local_file.stat().st_mode & 0o777 == 0o600
    loaded = settings.read_saved_settings()
    assert loaded['OPENAI_API_KEY'] == SECRET
    updated = settings.save_settings(configuration(api_key='', model='new-model'), loaded)
    assert updated['model'] == 'new-model'
    assert settings.read_saved_settings()['OPENAI_API_KEY'] == SECRET


def test_changed_destination_cannot_reuse_hidden_credential(local_file):
    current = settings.draft_config(configuration(), {})
    with pytest.raises(ValueError, match='更换 API 地址'):
        settings.draft_config(configuration(base_url='https://other.example', api_key=''), current)
    with pytest.raises(ValueError, match='更换 API 地址'):
        settings.draft_config(configuration(base_url='https://gateway.example/other', api_key=''), current)
    assert settings.draft_config(configuration(base_url='https://other.example'), current)['OPENAI_API_KEY'] == SECRET


@pytest.mark.parametrize('url', ['http://remote.example', 'https://user:pass@host.example',
    'https://host.example?api_key=x', 'https://host.example/#x', 'file:///tmp/key',
    'https://host.example\nheader=x', 'https://host.example:bad'])
def test_invalid_destination_rejected(url):
    with pytest.raises(ValueError):
        settings.normalize_base(url)


def test_custom_paths_and_full_endpoint_are_normalized_once():
    assert settings.normalize_base('https://host.example') == 'https://host.example/v1'
    assert settings.normalize_base('https://host.example/v1/') == 'https://host.example/v1'
    assert settings.normalize_base('https://host.example/custom/v2/chat/completions') == 'https://host.example/custom/v2'
    assert settings.normalize_base('http://127.0.0.1:1234/v1') == 'http://127.0.0.1:1234/v1'


def test_chat_completions_parsing_and_secret_transport(monkeypatch):
    config = settings.draft_config(configuration(), {})
    def run(args, **kw):
        assert SECRET not in ' '.join(args)
        assert 'https://gateway.example/v1/chat/completions' in kw['input']
        assert SECRET in kw['input']
        assert '--location' not in args
        result = {'model':'example-model', 'id':'test-response',
                  'choices':[{'message':{'content':'Hello'}}]}
        return SimpleNamespace(returncode=0, stdout=json.dumps(result)+'\n200')
    monkeypatch.setattr(language.subprocess, 'run', run)
    result = language.LLM(provider='openai', config=config).complete('Hi', 'Be brief')
    assert result['text'] == 'Hello' and result['api_style'] == 'chat_completions'
    assert SECRET not in json.dumps(result)


def test_transport_error_body_and_bad_credentials_never_echo(monkeypatch):
    def run(*args, **kw):
        return SimpleNamespace(returncode=0, stdout=json.dumps(
            {'error': {'type': 'invalid_api_key', 'message': SECRET}})+'\n401')
    monkeypatch.setattr(language.subprocess, 'run', run)
    with pytest.raises(language.LLMUnavailable, match='401') as exc:
        language.request_json('https://gateway.example/v1/models', ['Authorization: Bearer '+SECRET])
    assert SECRET not in str(exc.value)


def test_local_settings_override_env_without_returning_secrets(local_file, monkeypatch):
    monkeypatch.setattr(language, 'ROOT', local_file.parent)
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://old.example/v1')
    settings.save_settings(configuration(), {})
    config = language.local_config()
    assert config['OPENAI_BASE_URL'] == 'https://gateway.example/v1'
    assert config['OPENAI_API_KEY'] == SECRET


def test_http_settings_lifecycle_and_cross_site_rejection(local_file, monkeypatch):
    monkeypatch.setattr(server, 'local_config', settings.read_saved_settings)
    http = server.create_server(0, app=SimpleNamespace())
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{http.server_port}'
    def call(body=None, **headers):
        req = urllib.request.Request(base+'/api/settings',
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type':'application/json', **headers})
        with urllib.request.urlopen(req) as response:
            assert response.headers['Cache-Control'] == 'no-store'
            raw = response.read().decode()
            assert SECRET not in raw
            return json.loads(raw)
    try:
        assert not call()['key_configured']
        assert call(configuration())['key_configured']
        assert call()['model'] == 'example-model'
        for headers in [{'Origin':'https://unrelated.example'}, {'Host':'unrelated.example'},
                        {'Sec-Fetch-Site':'cross-site'}]:
            with pytest.raises(urllib.error.HTTPError) as exc:
                call(configuration(model='must-not-save'), **headers)
            assert exc.value.code == 403
        assert call()['model'] == 'example-model'
    finally:
        http.shutdown()
        http.server_close()
        worker.join(timeout=2)
