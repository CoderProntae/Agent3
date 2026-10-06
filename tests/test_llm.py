"""Tests for the token estimator and the Ollama transport client."""

from __future__ import annotations

import json
import threading

import pytest
import requests

from agent3.core.config import OllamaSettings
from agent3.llm.messages import ChatMessage, Role, Usage
from agent3.llm.ollama_client import OllamaClient, OllamaError, OllamaUnavailableError
from agent3.llm.tokenizer import count_message_tokens, estimate_tokens, truncate_to_tokens


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code=200, lines=None, payload=None, text=""):
        self.status_code = status_code
        self._lines = lines or []
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.closed = False

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            yield line

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def close(self):
        self.closed = True


class FakeSession:
    """Records requests and replays scripted responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.headers = {}

    def request(self, method, url, json=None, stream=False, timeout=None):
        self.calls.append({"method": method, "url": url, "json": json, "stream": stream})
        item = self._responses.pop(0) if self._responses else FakeResponse(200, payload={})
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass

    def mount(self, *args, **kwargs):
        pass


def make_client(responses, **overrides) -> OllamaClient:
    settings = OllamaSettings(max_retries=2, retry_backoff=0.01, **overrides)
    client = OllamaClient(settings)
    client._session = FakeSession(responses)  # noqa: SLF001 - deliberate test seam
    return client


class TestTokenizer:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_monotonic(self):
        assert estimate_tokens("hello world") < estimate_tokens("hello world " * 20)

    def test_plausible_magnitude(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = estimate_tokens(text)
        assert len(text) / 6 < tokens < len(text) / 2

    def test_non_ascii_costs_more(self):
        assert estimate_tokens("ağırlıklı değerlendirme") > estimate_tokens("weighted evaluation")

    def test_message_overhead(self):
        messages = [ChatMessage.user("hi"), ChatMessage.assistant("hello")]
        assert count_message_tokens(messages) >= estimate_tokens("hi") + estimate_tokens("hello")

    def test_truncate_keeps_head_and_tail(self):
        text = "START" + ("x" * 10000) + "END"
        trimmed = truncate_to_tokens(text, 100)
        assert trimmed.startswith("START") and trimmed.endswith("END")
        assert "truncated" in trimmed
        assert len(trimmed) < len(text)

    def test_truncate_noop_for_small_text(self):
        assert truncate_to_tokens("short", 100) == "short"


class TestMessages:
    def test_payload_shape(self):
        payload = ChatMessage(Role.USER, "hi").to_payload()
        assert payload == {"role": "user", "content": "hi"}

    def test_usage_merge(self):
        merged = Usage(10, 5).merge(Usage(1, 2))
        assert merged.prompt_tokens == 11 and merged.completion_tokens == 7
        assert merged.total_tokens == 18


class TestOllamaClient:
    def test_default_endpoint_is_11435(self):
        assert OllamaClient().settings.base_url == "http://localhost:11435"

    def test_streaming_chat(self):
        lines = [
            json.dumps({"message": {"content": "Hel"}, "done": False}),
            json.dumps({"message": {"content": "lo"}, "done": False}),
            json.dumps(
                {
                    "message": {"content": ""},
                    "done": True,
                    "prompt_eval_count": 11,
                    "eval_count": 7,
                    "total_duration": 2_000_000,
                }
            ),
        ]
        client = make_client([FakeResponse(200, lines=lines)])
        deltas = []
        text, usage = client.chat([ChatMessage.user("hi")], on_delta=deltas.append)
        assert text == "Hello"
        assert deltas == ["Hel", "lo"]
        assert usage.prompt_tokens == 11 and usage.completion_tokens == 7
        assert usage.estimated is False
        assert usage.total_duration_ms == 2

    def test_stream_without_final_chunk_estimates_usage(self):
        lines = [json.dumps({"message": {"content": "partial"}, "done": False})]
        client = make_client([FakeResponse(200, lines=lines)])
        text, usage = client.chat([ChatMessage.user("hi")])
        assert text == "partial"
        assert usage.estimated is True
        assert usage.completion_tokens > 0

    def test_non_streaming_chat(self):
        payload = {"message": {"content": "answer"}, "done": True, "prompt_eval_count": 3, "eval_count": 2}
        client = make_client([FakeResponse(200, payload=payload)], stream=False)
        text, usage = client.chat([ChatMessage.user("hi")])
        assert text == "answer" and usage.total_tokens == 5

    def test_error_inside_stream(self):
        lines = [json.dumps({"error": "model not found"})]
        client = make_client([FakeResponse(200, lines=lines)])
        with pytest.raises(OllamaError, match="model not found"):
            client.chat([ChatMessage.user("hi")])

    def test_retries_then_succeeds(self):
        good = FakeResponse(200, lines=[json.dumps({"message": {"content": "ok"}, "done": True})])
        client = make_client([requests.ConnectionError("refused"), requests.ConnectionError("refused"), good])
        text, _usage = client.chat([ChatMessage.user("hi")])
        assert text == "ok"

    def test_unavailable_after_retries(self):
        client = make_client([requests.ConnectionError("refused")] * 12)
        with pytest.raises(OllamaUnavailableError):
            client.chat([ChatMessage.user("hi")])

    def test_http_error_is_raised(self):
        client = make_client([FakeResponse(404, text="model missing")])
        with pytest.raises(OllamaError, match="404"):
            client.chat([ChatMessage.user("hi")])

    def test_retryable_status_is_retried(self):
        good = FakeResponse(200, lines=[json.dumps({"message": {"content": "ok"}, "done": True})])
        client = make_client([FakeResponse(503, text="busy"), good])
        text, _ = client.chat([ChatMessage.user("hi")])
        assert text == "ok"

    def test_cancellation(self):
        cancel = threading.Event()
        cancel.set()
        client = make_client([FakeResponse(200, lines=[])])
        with pytest.raises(Exception):
            client.chat([ChatMessage.user("hi")], cancel=cancel)

    def test_health_ok(self):
        tags = FakeResponse(200, payload={"models": [{"name": "qwen3.5-9b-abliterated"}]})
        version = FakeResponse(200, payload={"version": "0.5.0"})
        client = make_client([tags, version])
        health = client.health()
        assert health["ok"] is True
        assert health["models"] == ["qwen3.5-9b-abliterated"]
        assert health["version"] == "0.5.0"

    def test_health_failure_is_soft(self):
        client = make_client([requests.ConnectionError("nope")] * 12)
        health = client.health()
        assert health["ok"] is False and health["models"] == []

    def test_resolve_model_falls_back(self):
        tags = FakeResponse(200, payload={"models": [{"name": "qwen2.5-coder:7b"}]})
        version = FakeResponse(200, payload={"version": "0.5.0"})
        client = make_client([tags, version])
        assert client.resolve_model("qwen3.5-9b-abliterated") == "qwen2.5-coder:7b"

    def test_payload_contains_options_and_keep_alive(self):
        client = make_client([FakeResponse(200, lines=[json.dumps({"message": {"content": ""}, "done": True})])])
        client.chat([ChatMessage.user("hi")])
        sent = client._session.calls[0]["json"]  # noqa: SLF001
        assert sent["stream"] is True
        assert sent["keep_alive"] == "15m"
        assert sent["options"]["num_ctx"] == 16384
        assert client._session.calls[0]["url"].endswith("/api/chat")  # noqa: SLF001
