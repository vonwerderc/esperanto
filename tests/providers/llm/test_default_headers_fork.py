"""Header-forwarding behavior for OpenAICompatibleLanguageModel.

Covers the fork feature: optional ``default_headers`` merged under the
provider-controlled headers for direct HTTP requests and forwarded to
LangChain ``ChatOpenAI``.
"""

from unittest.mock import Mock, patch

import httpx
import pytest

from esperanto.providers.llm.openai_compatible import OpenAICompatibleLanguageModel


def _make_model(**kwargs) -> OpenAICompatibleLanguageModel:
    defaults = {
        "api_key": "test-key",
        "base_url": "http://localhost:8080",
    }
    defaults.update(kwargs)
    return OpenAICompatibleLanguageModel(**defaults)


def _chat_completion_payload() -> dict:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class TestDefaultHeaders:
    """default_headers: merge, precedence, LangChain forwarding, isolation."""

    def test_get_headers_preserves_auth_and_custom(self):
        model = _make_model(config={"default_headers": {"x-opencode-session": "ses_abc"}})
        headers = model._get_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"
        assert headers["x-opencode-session"] == "ses_abc"

    def test_get_headers_caller_data_cannot_override_controlled_headers(self):
        model = _make_model(
            config={
                "default_headers": {
                    "Authorization": "Bearer attacker-chosen",
                    "Content-Type": "text/plain",
                }
            }
        )
        headers = model._get_headers()
        # Provider-controlled headers always win over caller-supplied data.
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    def test_get_headers_new_mapping_each_call(self):
        model = _make_model(config={"default_headers": {"x-opencode-session": "ses_abc"}})
        h1 = model._get_headers()
        h2 = model._get_headers()
        h1["x-opencode-session"] = "mutated"
        assert h2["x-opencode-session"] == "ses_abc"

    def test_chat_complete_sends_default_headers_direct_http(self):
        model = _make_model(config={"default_headers": {"x-opencode-session": "ses_http"}})
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json=_chat_completion_payload())

        model.client = httpx.Client(transport=httpx.MockTransport(handler))
        model.async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        response = model.chat_complete([{"role": "user", "content": "hi"}])
        assert response.choices[0].message.content == "ok"
        assert captured["headers"]["x-opencode-session"] == "ses_http"
        assert captured["headers"]["authorization"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_achat_complete_sends_default_headers_direct_http(self):
        model = _make_model(config={"default_headers": {"x-opencode-session": "ses_async"}})
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json=_chat_completion_payload())

        model.async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        response = await model.achat_complete([{"role": "user", "content": "hi"}])
        assert response.choices[0].message.content == "ok"
        assert captured["headers"]["x-opencode-session"] == "ses_async"
        assert captured["headers"]["authorization"] == "Bearer test-key"

    def test_to_langchain_forwards_default_headers_when_present(self):
        model = _make_model(config={"default_headers": {"x-opencode-session": "ses_lc"}})
        with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
            mock_chat_openai.return_value = Mock()
            model.to_langchain()
            call_args = mock_chat_openai.call_args[1]
            assert call_args["default_headers"] == {"x-opencode-session": "ses_lc"}

    def test_to_langchain_omits_default_headers_when_absent(self):
        model = _make_model()
        with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
            mock_chat_openai.return_value = Mock()
            model.to_langchain()
            call_args = mock_chat_openai.call_args[1]
            assert "default_headers" not in call_args

    def test_to_langchain_omits_default_headers_when_empty(self):
        model = _make_model(config={"default_headers": {}})
        with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
            mock_chat_openai.return_value = Mock()
            model.to_langchain()
            call_args = mock_chat_openai.call_args[1]
            assert "default_headers" not in call_args

    def test_missing_default_headers_preserves_baseline_direct_http(self):
        model = _make_model()
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json=_chat_completion_payload())

        model.client = httpx.Client(transport=httpx.MockTransport(handler))
        model.async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        response = model.chat_complete([{"role": "user", "content": "hi"}])
        assert response.choices[0].message.content == "ok"
        assert "x-opencode-session" not in captured["headers"]
        assert captured["headers"]["authorization"] == "Bearer test-key"

    def test_input_mapping_not_mutated(self):
        mapping = {"x-opencode-session": "ses_orig"}
        model = _make_model(config={"default_headers": mapping})
        headers = model._get_headers()
        headers["x-opencode-session"] = "changed"
        assert mapping == {"x-opencode-session": "ses_orig"}

    def test_default_headers_via_direct_attribute(self):
        model = _make_model()
        model.default_headers = {"x-opencode-session": "ses_attr"}
        headers = model._get_headers()
        assert headers["x-opencode-session"] == "ses_attr"
        assert headers["Authorization"] == "Bearer test-key"
