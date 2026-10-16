"""Language rendering from decoded neural observations, plus optional real APIs."""
import json
import os
import subprocess
import re
from urllib.parse import urlparse
from .paths import ROOT
from .api_settings import read_saved_settings, normalize_base


def local_config():
    """Read only this app's named settings; never persist or display secrets."""
    names=['OPENAI_API_KEY','OPENAI_BASE_URL','CHATFLYPT_MODEL','CHATFLYPT_PROVIDER','CHATFLYPT_API_STYLE',
           'ANTHROPIC_AUTH_TOKEN','ANTHROPIC_API_KEY','ANTHROPIC_BASE_URL']
    values={n:os.environ[n] for n in names if os.environ.get(n)}
    if os.name=='nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as key:
                for name in names:
                    try:
                        value,_=winreg.QueryValueEx(key,name)
                        if value:values[name]=value
                    except FileNotFoundError:pass
        except FileNotFoundError:pass
    env=ROOT/'.env'
    if env.exists():
        for line in env.read_text(encoding='utf-8-sig').splitlines():
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line:continue
            name,value=line.split('=',1);name=name.strip();value=value.strip()
            if name in names:
                if len(value)>=2 and value[0]==value[-1] and value[0] in ['"',"'"]:value=value[1:-1]
                values[name]=value
    values.update(read_saved_settings())
    return values


def verbalize(words, ask=True):
    words=set(words)
    odors=[w for w in ['vinegar','yeast','geosmin','pheromone'] if w in words]
    tastes=[w for w in ['sugar','bitter','water','salt'] if w in words]
    conditions=[w for w in ['warm','cold','humid','dry','bright'] if w in words]
    sentences=[]
    def join(items):
        return items[0] if len(items)==1 else ', '.join(items[:-1])+' and '+items[-1]
    if odors:sentences.append('I smell '+join(odors)+'.')
    if tastes:sentences.append('I taste '+join(tastes)+'.')
    if conditions:sentences.append('It is '+join(conditions)+'.')
    for word,text in [('touch','I detect touch.'),('sound','I hear sound.'),('wind','I detect wind.')]:
        if word in words:sentences.append(text)
    if not sentences:sentences.append('I cannot identify a supported sensory signal.')
    if ask:sentences.append('What should I do?')
    return ' '.join(sentences)


class LLMUnavailable(RuntimeError):
    pass


def request_json(endpoint, headers, payload=None):
    # Secrets enter curl via stdin only; no redirects or response bodies in errors.
    lines=['url = '+json.dumps(endpoint),'request = '+json.dumps('GET' if payload is None else 'POST')]
    lines += ['header = '+json.dumps(h) for h in headers]
    if payload is not None:
        lines += ['data-binary = '+json.dumps(json.dumps(payload,ensure_ascii=True))]
    try:
        proc=subprocess.run(['curl.exe' if os.name=='nt' else 'curl','--silent','--show-error',
            '--max-time','55','--write-out','\n%{http_code}','--config','-'],
            input='\n'.join(lines)+'\n',text=True,capture_output=True,timeout=60,encoding='utf8')
    except (OSError,subprocess.TimeoutExpired,UnicodeError):
        raise LLMUnavailable('API 连接失败或超时；未保存外部错误正文。') from None
    if proc.returncode:
        if proc.returncode==28:
            raise LLMUnavailable('API 请求超时（55 秒）；请稍后重试或更换模型。')
        raise LLMUnavailable(f'API transport failed (curl {proc.returncode}); credentials omitted')
    raw,_,tail=proc.stdout.rpartition('\n')
    code=int(tail) if re.fullmatch(r'\d{3}',tail) else 200
    if code==200 and not re.fullmatch(r'\d{3}',tail):raw=proc.stdout
    try:
        result=json.loads(raw)
    except (json.JSONDecodeError,RecursionError):
        raise LLMUnavailable(f'API 未返回 JSON（HTTP {code}），请检查地址和协议。') from None
    if not isinstance(result,dict):raise LLMUnavailable('API 返回格式无效。')
    if 'error' in result or not 200<=code<300:
        err=result.get('error',{})
        kind=err.get('type','unknown') if isinstance(err,dict) else 'unknown'
        if not isinstance(kind,str) or not re.fullmatch(r'[a-zA-Z0-9_.-]{1,60}',kind) or 'sk-' in kind:
            kind='unknown'
        hint=('网关上游暂时不可用，请稍后重试或更换模型' if code>=500 else
              '密钥或模型访问权限被拒绝，请检查配置' if code in (401,403) else
              '请求频率或额度受限，请检查服务商账户' if code==429 else
              'API 拒绝请求，请检查模型和接口协议')
        raise LLMUnavailable(f'{hint}（HTTP {code}, {kind}）。')
    return result


def list_models(config):
    result=request_json(normalize_base(config['OPENAI_BASE_URL'])+'/models',
                        ['Authorization: Bearer '+config['OPENAI_API_KEY']])
    rows=result.get('data')
    if not isinstance(rows,list):raise LLMUnavailable('网关未提供标准模型列表；可手动填写模型。')
    models=sorted({row['id'] for row in rows if isinstance(row,dict) and
        isinstance(row.get('id'),str) and 0<len(row['id'])<=200 and
        config['OPENAI_API_KEY'] not in row['id']})
    return {'models':models[:1000]}


class LLM:
    def __init__(self,provider=None,model=None,config=None):
        config=local_config() if config is None else config
        self.provider=provider or config.get('CHATFLYPT_PROVIDER','openai')
        if self.provider=='openai':
            self.key=config.get('OPENAI_API_KEY','')
            self.base=normalize_base(config.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1')
            self.model=model or config.get('CHATFLYPT_MODEL','')
            self.api_style=config.get('CHATFLYPT_API_STYLE') or 'responses'
            if self.api_style not in ('responses','chat_completions'):raise ValueError('Unsupported API style')
        elif self.provider=='anthropic':
            self.key=config.get('ANTHROPIC_AUTH_TOKEN') or config.get('ANTHROPIC_API_KEY','')
            self.base=config.get('ANTHROPIC_BASE_URL','https://api.anthropic.com/v1').rstrip('/')
            self.model=model or config.get('CHATFLYPT_MODEL','claude-sonnet-4-6')
        else:
            raise ValueError('Unsupported provider')
        if self.provider=='anthropic' and not self.base.endswith('/v1'):
            self.base+='/v1'
        if not self.key or not self.model:
            raise LLMUnavailable('Configure API credentials and CHATFLYPT_MODEL locally; no offline answer is substituted.')

    def complete(self,user,system):
        if self.provider=='openai':
            if self.api_style=='chat_completions':
                endpoint=self.base+'/chat/completions'
                payload={'model':self.model,'messages':[{'role':'system','content':system},
                    {'role':'user','content':user}],'max_completion_tokens':1024,'stream':False,'store':False}
            else:
                endpoint=self.base+'/responses'
                payload={'model':self.model,'input':user,'instructions':system,'max_output_tokens':1024,'store':False}
            headers=['Content-Type: application/json','Authorization: Bearer '+self.key]
        else:
            endpoint=self.base+'/messages'
            payload={'model':self.model,'max_tokens':512,'system':system,
                     'messages':[{'role':'user','content':user}]}
            headers=['Content-Type: application/json','x-api-key: '+self.key,
                     'Authorization: Bearer '+self.key,'anthropic-version: 2023-06-01']
        result=request_json(endpoint,headers,payload)
        try:
            if self.provider=='openai':
                if self.api_style=='chat_completions':
                    choices=result.get('choices') or []
                    answer=choices[0].get('message',{}).get('content','') if choices else ''
                    if isinstance(answer,list):answer=''.join(c.get('text','') for c in answer if c.get('type')=='text')
                else:
                    answer=''.join(c.get('text','') for item in result.get('output',[]) if item.get('type')=='message'
                                   for c in item.get('content',[]) if c.get('type')=='output_text')
            else:
                answer=''.join(c.get('text','') for c in result.get('content',[]) if c.get('type')=='text')
        except (AttributeError,IndexError,KeyError,TypeError,ValueError):
            # A gateway may return HTTP 200 with malformed nested API fields.
            # Preserve the caller's completed neural observation and never put
            # response fragments or credentials in a schema-error message.
            raise LLMUnavailable('API 响应结构无效，无法读取模型文本；未保存外部错误正文。') from None
        if not isinstance(answer,str) or not answer.strip():
            raise LLMUnavailable('Model returned no text')
        return {'text':answer.strip().replace(self.key,'[redacted]'),'provider_protocol':self.provider,
                'api_style':getattr(self,'api_style','messages'),'provider_host':urlparse(self.base).hostname,
                'requested_model':self.model,'returned_model':result.get('model'),
                'response_id':result.get('id'),'usage':result.get('usage'),
                'provenance':('live response from a local inference endpoint' if urlparse(self.base).hostname in ['127.0.0.1','localhost','::1'] else
                    'live API response; third-party gateway model identity is not independently attested' if urlparse(self.base).hostname not in ['api.openai.com','api.anthropic.com'] else 'live official API response')}

