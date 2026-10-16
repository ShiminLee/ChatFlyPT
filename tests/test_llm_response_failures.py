"""Broken HTTP-200 API schemas remain recoverable and redacted in the client."""
import json
from types import SimpleNamespace

import pytest

from chatflypt import conversation, language


SECRET = "fixture-secret-never-display"
BODY_SENTINEL = "untrusted-api-response-body"


def model(protocol):
    config = {"OPENAI_API_KEY": SECRET, "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
              "CHATFLYPT_MODEL": "fixture-model", "CHATFLYPT_API_STYLE": protocol,
              "ANTHROPIC_API_KEY": SECRET, "ANTHROPIC_BASE_URL": "http://127.0.0.1:9/v1"}
    return language.LLM(provider="anthropic" if protocol == "messages" else "openai", config=config)


BROKEN_RESPONSES = [
    ("responses", {"output": None}),
    ("responses", {"output": 7}),
    ("responses", {"output": [BODY_SENTINEL]}),
    ("responses", {"output": [{"type": "message", "content": None}]}),
    ("responses", {"output": [{"type": "message", "content": [{"type": "output_text", "text": {SECRET: BODY_SENTINEL}}]}]}),
    ("chat_completions", {"choices": {SECRET: BODY_SENTINEL}}),
    ("chat_completions", {"choices": [{"message": None}]}),
    ("chat_completions", {"choices": [BODY_SENTINEL]}),
    ("chat_completions", {"choices": [{"message": {"content": [None]}}]}),
    ("chat_completions", {"choices": [{"message": {"content": [{"type": "text", "text": None}]}}]}),
    ("messages", {"content": None}),
    ("messages", {"content": [BODY_SENTINEL]}),
    ("messages", {"content": [{"type": "text", "text": {SECRET: BODY_SENTINEL}}]}),
]


@pytest.mark.parametrize("protocol,response", BROKEN_RESPONSES)
def test_malformed_200_response_is_redacted_in_standalone_api_client(protocol,response,monkeypatch):
    def curl(*args,**kwargs):
        return SimpleNamespace(returncode=0,stdout=json.dumps(response)+"\n200",stderr="")
    monkeypatch.setattr(language.subprocess,'run',curl)
    with pytest.raises(language.LLMUnavailable) as error:
        model(protocol).complete('standalone fixture','standalone test')
    assert SECRET not in str(error.value) and BODY_SENTINEL not in str(error.value)


@pytest.mark.parametrize("protocol,response", [
    ("responses", {"output": [{"type": "reasoning", "summary": []},
                               {"type": "message", "content": [{"type": "output_text", "text": "回复"}]}]}),
    ("chat_completions", {"choices": [{"message": {"content": "回复"}}]}),
    ("chat_completions", {"choices": [{"message": {"content": [{"type": "text", "text": "回复"}]}}]}),
    ("messages", {"content": [{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "回复"}]}),
])
def test_valid_response_and_existing_request_fields_are_unchanged(protocol, response, monkeypatch):
    captured = []

    def request(endpoint, headers, payload):
        captured.append((endpoint, payload))
        return response

    monkeypatch.setattr(language, "request_json", request)
    result = model(protocol).complete("input-exact", "system-exact")
    assert result["text"] == "回复"
    endpoint, payload = captured[0]
    if protocol == "responses":
        assert endpoint.endswith("/responses")
        assert payload == {"model": "fixture-model", "input": "input-exact", "instructions": "system-exact",
                           "max_output_tokens": 1024, "store": False}
    elif protocol == "chat_completions":
        assert endpoint.endswith("/chat/completions")
        assert payload == {"model": "fixture-model", "messages": [
            {"role": "system", "content": "system-exact"}, {"role": "user", "content": "input-exact"}],
            "max_completion_tokens": 1024, "stream": False, "store": False}
    else:
        assert endpoint.endswith("/messages")
        assert payload == {"model": "fixture-model", "max_tokens": 512, "system": "system-exact",
                           "messages": [{"role": "user", "content": "input-exact"}]}


def test_invalid_response_encoding_is_redacted_as_unavailable(monkeypatch):
    def broken(*args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, SECRET)
    monkeypatch.setattr(language.subprocess, "run", broken)
    with pytest.raises(language.LLMUnavailable) as error:
        model("responses").complete("input", "system")
    assert SECRET not in str(error.value)
