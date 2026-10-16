"""Local API settings; Windows credentials are protected for the current user."""
import base64
import ctypes
import json
import os
from pathlib import Path
import tempfile
import threading
from urllib.parse import urlsplit, urlunsplit
from .paths import ROOT

SETTINGS_FILE = ROOT / 'data/local/api-settings.json'
SETTINGS_LOCK = threading.Lock()


def normalize_base(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError('请输入 API 地址。')
    value = value.strip().rstrip('/')
    if any(ord(c) < 33 for c in value):
        raise ValueError('API 地址不能包含空格或控制字符。')
    url = urlsplit(value)
    if (not url.hostname or url.username or url.password or url.query or url.fragment or
            url.scheme not in ('http', 'https')):
        raise ValueError('请输入完整 API 地址，不包含密钥、查询参数或锚点。')
    if url.scheme != 'https' and url.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('远程 API 请使用 HTTPS；HTTP 仅用于本机服务。')
    try:
        url.port
    except ValueError:
        raise ValueError('API 端口无效。') from None
    path = url.path.rstrip('/')
    for suffix in ('/chat/completions', '/responses', '/models'):
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    return urlunsplit((url.scheme, url.netloc.lower(), path or '/v1', '', ''))


def _dpapi(data, decrypt=False):
    from ctypes import wintypes
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    buf = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    fn = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    fn.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    fn.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if not fn(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise ValueError('无法读取或保护本机密钥，请重新配置。')
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel.LocalFree(ctypes.cast(target.data, ctypes.c_void_p))


def read_saved_settings():
    if not SETTINGS_FILE.exists():
        return {}
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding='utf8'))
        if saved['storage'] == 'windows-dpapi':
            key = _dpapi(base64.b64decode(saved['key_ciphertext']), decrypt=True).decode('utf8')
        elif saved['storage'] == 'owner-file' and os.name != 'nt':
            key = saved['api_key']
        else:
            raise ValueError()
        return {'OPENAI_BASE_URL': saved['base_url'], 'OPENAI_API_KEY': key,
                'CHATFLYPT_MODEL': saved['model'], 'CHATFLYPT_API_STYLE': saved['api_style'],
                'CHATFLYPT_PROVIDER': 'openai'}
    except (KeyError, ValueError, OSError):
        raise ValueError('本机 API 配置不可读，请重新保存 API 设置。') from None


def public_settings(config):
    return {'base_url': normalize_base(config.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'),
            'model': config.get('CHATFLYPT_MODEL', ''),
            'api_style': config.get('CHATFLYPT_API_STYLE') or 'responses',
            'key_configured': bool(config.get('OPENAI_API_KEY')),
            'storage': ('windows-dpapi' if os.name == 'nt' else 'owner-file') if SETTINGS_FILE.exists() else 'environment'}


def draft_config(body, current, require_model=True):
    if not isinstance(body, dict):
        raise ValueError('API 设置必须是一个对象。')
    base = normalize_base(body.get('base_url', current.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'))
    key = body.get('api_key', '')
    if not isinstance(key, str) or len(key) > 4096 or any(ord(c) < 32 for c in key):
        raise ValueError('API 密钥格式无效。')
    key = key.strip()
    if not key:
        if base != normalize_base(current.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'):
            raise ValueError('更换 API 地址时，请填写该地址使用的密钥。')
        key = current.get('OPENAI_API_KEY', '')
    if not key:
        raise ValueError('请填写 API 密钥。')
    model = body.get('model', current.get('CHATFLYPT_MODEL', ''))
    if not isinstance(model, str) or len(model) > 200 or any(ord(c) < 32 for c in model):
        raise ValueError('模型名称格式无效。')
    model = model.strip()
    if require_model and not model:
        raise ValueError('请填写或选择模型。')
    style = body.get('api_style', current.get('CHATFLYPT_API_STYLE') or 'responses')
    if style not in ('responses', 'chat_completions'):
        raise ValueError('请选择 Responses 或 Chat Completions 协议。')
    return {'OPENAI_API_KEY': key, 'OPENAI_BASE_URL': base, 'CHATFLYPT_MODEL': model,
            'CHATFLYPT_API_STYLE': style, 'CHATFLYPT_PROVIDER': 'openai'}


def save_settings(body, current):
    config = draft_config(body, current)
    saved = {'base_url': config['OPENAI_BASE_URL'], 'model': config['CHATFLYPT_MODEL'],
             'api_style': config['CHATFLYPT_API_STYLE'], 'schema': 1}
    if os.name == 'nt':
        saved.update(storage='windows-dpapi', key_ciphertext=base64.b64encode(
            _dpapi(config['OPENAI_API_KEY'].encode('utf8'))).decode('ascii'))
    else:
        saved.update(storage='owner-file', api_key=config['OPENAI_API_KEY'])
    with SETTINGS_LOCK:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=SETTINGS_FILE.parent,
                                             prefix='.api-', suffix='.tmp', delete=False) as f:
                tmp = Path(f.name)
                os.chmod(tmp, 0o600)
                json.dump(saved, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, SETTINGS_FILE)
        finally:
            if tmp and tmp.exists():
                tmp.unlink()
    return public_settings(config)
