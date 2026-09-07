import json
from unittest.mock import MagicMock, patch

import financial_llm_engine as fle


def _response(status=200, content="nvidia response"):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    response.text = "error"
    return response


def test_nvidia_is_primary_when_configured(monkeypatch):
    monkeypatch.setattr(fle, "NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(fle, "check_llm_connectivity", lambda: False)
    monkeypatch.setattr(fle, "_cache_load", lambda *args, **kwargs: None)
    with patch.object(fle.requests, "post", return_value=_response()) as post:
        assert fle._call_llm("prompt") == "nvidia response"
    assert post.call_count == 1
    assert post.call_args.args[0] == fle.NVIDIA_LLM_API_URL
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert post.call_args.kwargs["json"]["model"] == fle.NVIDIA_LLM_MODEL


def test_lm_studio_is_fallback_after_nvidia_failure(monkeypatch):
    monkeypatch.setattr(fle, "NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(fle, "check_llm_connectivity", lambda: True)
    monkeypatch.setattr(fle, "_cache_load", lambda *args, **kwargs: None)
    with patch.object(
        fle.requests,
        "post",
        side_effect=[_response(status=503), _response(content="local response")],
    ) as post:
        assert fle._call_llm("prompt") == "local response"
    assert post.call_count == 2
    assert post.call_args_list[1].args[0] == fle.LLM_API_URL


def test_without_nvidia_key_only_local_provider_is_attempted(monkeypatch):
    monkeypatch.setattr(fle, "NVIDIA_API_KEY", "")
    monkeypatch.setattr(fle, "check_llm_connectivity", lambda: True)
    monkeypatch.setattr(fle, "_cache_load", lambda *args, **kwargs: None)
    with patch.object(fle.requests, "post", return_value=_response(content="local response")) as post:
        assert fle._call_llm("prompt") == "local response"
    assert post.call_count == 1
    assert post.call_args.args[0] == fle.LLM_API_URL
