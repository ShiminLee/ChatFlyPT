"""Separate local inference configuration; never overwrite the remote API key."""
import json
from urllib.parse import urlsplit

from .paths import ROOT
from .api_settings import normalize_base
from .language import LLM, LLMUnavailable


def local_model_config():
    path = ROOT / 'data/local/local-model.json'
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        base = normalize_base(data['base_url'])
        if urlsplit(base).hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('Local inference must use a loopback address')
        model = data['model']
        if not isinstance(model, str) or not model.strip() or len(model) > 200:
            raise ValueError('Invalid local model name')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LLMUnavailable('本地模型尚未准备好，请使用已配置 API 或仅神经读出。') from exc
    return {'OPENAI_BASE_URL': base, 'OPENAI_API_KEY': 'local-inference',
            'CHATFLYPT_MODEL': model, 'CHATFLYPT_API_STYLE': 'chat_completions',
            'CHATFLYPT_PROVIDER': 'openai'}


def local_model_info():
    try:
        config = local_model_config()
        return {'configured': True, 'model': config['CHATFLYPT_MODEL'],
                'base_url': config['OPENAI_BASE_URL']}
    except LLMUnavailable:
        return {'configured': False, 'model': None}


def make_model(backend):
    if backend == 'configured':
        return LLM(provider='openai')
    if backend == 'local':
        return LLM(provider='openai', config=local_model_config())
    raise ValueError('Choose configured or local inference')
