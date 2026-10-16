import json

import pytest

from chatflypt import model_runtime
from chatflypt.language import LLMUnavailable


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    monkeypatch.setattr(model_runtime, 'ROOT', tmp_path)
    path = tmp_path/'data/local/local-model.json'
    path.parent.mkdir(parents=True)
    return path


def test_local_model_keeps_remote_credentials_out_of_configuration(config_path):
    config_path.write_text(json.dumps({'base_url': 'http://127.0.0.1:8081/v1', 'model': 'local-fixture'}))
    config = model_runtime.local_model_config()
    assert config['OPENAI_API_KEY'] == 'local-inference'
    assert config['CHATFLYPT_MODEL'] == 'local-fixture'
    assert model_runtime.local_model_info() == {
        'configured': True, 'model': 'local-fixture', 'base_url': 'http://127.0.0.1:8081/v1'}
    assert 'api_key' not in config_path.read_text()


@pytest.mark.parametrize('base', ['https://example.com/v1', 'http://192.168.1.2/v1',
                                  'http://secret@localhost/v1', 'http://localhost/v1?key=secret'])
def test_local_backend_rejects_external_or_credential_bearing_urls(config_path, base):
    config_path.write_text(json.dumps({'base_url': base, 'model': 'fixture'}))
    with pytest.raises(LLMUnavailable):
        model_runtime.local_model_config()
    assert not model_runtime.local_model_info()['configured']


def test_absent_local_model_is_explicit(config_path):
    assert model_runtime.local_model_info() == {'configured': False, 'model': None}
    with pytest.raises(LLMUnavailable):
        model_runtime.make_model('local')


def test_windows_powershell_utf8_bom_configuration_is_accepted(config_path):
    config_path.write_text(json.dumps({'base_url': 'http://127.0.0.1:8081/v1',
                                       'model': 'local-fixture'}), encoding='utf-8-sig')
    assert model_runtime.local_model_info()['model'] == 'local-fixture'
