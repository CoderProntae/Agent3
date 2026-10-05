"""Tests for reasoning ("thinking") discovery and transport.

Ollama exposes per-model reasoning metadata in ``/api/show`` as a top level
``thinking`` object - ``{"values": [...], "default": ...}`` - where values may
be booleans, named levels, or both. The point of these tests is that Agent3
never *invents* a level: whatever the UI offers and whatever lands in the
request body has to come from that response.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from agent3.core.config import OllamaSettings
from agent3.llm.messages import ChatMessage, StreamEvent
from agent3.llm.ollama_client import (
    FAMILY_THINK_LEVELS,
    KNOWN_THINK_LEVELS,
    THINK_AUTO,
    THINK_OFF,
    THINK_ON,
    OllamaClient,
    ThinkingSupport,
)

GPT_OSS = {
    "capabilities": ["completion", "tools", "thinking"],
    "thinking": {"values": ["low", "medium", "high"], "default": "medium"},
}
QWEN3 = {
    "capabilities": ["completion", "tools", "thinking"],
    "thinking": {"values": [True, False], "default": True},
}
DEEPSEEK = {
    "capabilities": ["completion", "thinking"],
    "thinking": {"values": [False, "low", "high", "max"], "default": "low"},
}
PLAIN = {"capabilities": ["completion", "tools"]}
NO_METADATA = {"capabilities": ["completion", "thinking"]}


class TestThinkingSupport:
    def test_level_only_model_cannot_be_disabled(self):
        support = ThinkingSupport.from_show("gpt-oss:20b", GPT_OSS)
        assert support.supported
        assert support.levels == ("low", "medium", "high")
        assert support.can_disable is False
        assert support.forced is True

    def test_boolean_model_can_be_toggled(self):
        support = ThinkingSupport.from_show("qwen3:8b", QWEN3)
        assert support.supported and support.can_disable
        assert support.levels == ()
        assert support.forced is False

    def test_mixed_values_expose_both(self):
        support = ThinkingSupport.from_show("deepseek-v3.1", DEEPSEEK)
        assert support.can_disable
        assert support.levels == ("low", "high", "max")

    def test_model_without_the_capability(self):
        support = ThinkingSupport.from_show("llama3.1:8b", PLAIN)
        assert not support.supported
        assert support.choices() == [(THINK_AUTO, "Auto")]
        assert support.resolve("high") is None

    def test_capability_without_metadata_falls_back_to_on_off(self):
        support = ThinkingSupport.from_show("mystery:latest", NO_METADATA)
        assert support.supported
        assert support.known is False
        assert support.resolve(THINK_ON) is True

    def test_family_table_only_applies_without_metadata(self):
        support = ThinkingSupport.from_show("gpt-oss:120b", {"capabilities": ["completion"]})
        assert support.levels == FAMILY_THINK_LEVELS["gpt-oss"]
        assert support.known is False

    def test_server_metadata_beats_the_family_table(self):
        support = ThinkingSupport.from_show(
            "gpt-oss:20b",
            {"capabilities": ["thinking"], "thinking": {"values": ["low", "xhigh"]}},
        )
        assert support.levels == ("low", "xhigh")  # verbatim, not the hard-coded trio

    # ------------------------------------------------------------- choices
    def test_choices_for_a_level_model_omit_off(self):
        choices = dict(ThinkingSupport.from_show("gpt-oss:20b", GPT_OSS).choices())
        assert THINK_OFF not in choices
        assert choices["low"] == "Low" and choices["high"] == "High"

    def test_choices_for_a_boolean_model(self):
        choices = dict(ThinkingSupport.from_show("qwen3:8b", QWEN3).choices())
        assert set(choices) == {THINK_AUTO, THINK_OFF, THINK_ON}

    def test_no_invented_levels_anywhere(self):
        support = ThinkingSupport.from_show("qwen3:8b", QWEN3)
        offered = {spec for spec, _ in support.choices()}
        assert offered.isdisjoint({"low", "medium", "high", "max", "einstein"})

    # ------------------------------------------------------------ resolve
    def test_auto_omits_the_field(self):
        assert ThinkingSupport.from_show("gpt-oss:20b", GPT_OSS).resolve(THINK_AUTO) is None

    def test_exact_level_is_forwarded(self):
        assert ThinkingSupport.from_show("gpt-oss:20b", GPT_OSS).resolve("high") == "high"

    def test_unsupported_level_is_dropped(self):
        assert ThinkingSupport.from_show("qwen3:8b", QWEN3).resolve("high") is None

    def test_off_is_dropped_when_it_cannot_be_disabled(self):
        assert ThinkingSupport.from_show("gpt-oss:20b", GPT_OSS).resolve(THINK_OFF) is None

    def test_off_is_honoured_when_supported(self):
        assert ThinkingSupport.from_show("qwen3:8b", QWEN3).resolve(THINK_OFF) is False

    def test_on_uses_the_default_level_for_level_models(self):
        assert ThinkingSupport.from_show("gpt-oss:20b", GPT_OSS).resolve(THINK_ON) == "medium"

    def test_accepts_matches_resolve(self):
        support = ThinkingSupport.from_show("deepseek-v3.1", DEEPSEEK)
        assert support.accepts("max") and not support.accepts("medium")

    def test_case_is_ignored(self):
        assert ThinkingSupport.from_show("gpt-oss:20b", GPT_OSS).resolve("HIGH") == "high"

    def test_describe_is_human_readable(self):
        text = ThinkingSupport.from_show("gpt-oss:20b", GPT_OSS).describe()
        assert "low, medium, high" in text and "always on" in text

    def test_known_levels_are_the_documented_set(self):
        assert KNOWN_THINK_LEVELS == ("low", "medium", "high", "max")


# ========================================================== transport
class _FakeResponse:
    def __init__(self, payload: Any, lines: List[str] | None = None) -> None:
        self._payload = payload
        self._lines = lines or []

    def json(self) -> Any:
        return self._payload

    def iter_lines(self, decode_unicode: bool = False):
        return iter(self._lines)

    def close(self) -> None:
        pass


class _RecordingClient(OllamaClient):
    """Client whose HTTP layer is replaced by a scripted responder."""

    def __init__(self, show: Dict[str, Any], stream_lines: List[str] | None = None) -> None:
        super().__init__(OllamaSettings(stream=bool(stream_lines)))
        self._show = show
        self._stream_lines = stream_lines or []
        self.requests: List[Dict[str, Any]] = []

    def _request(self, method, route, *, payload=None, stream=False, cancel=None, timeout=None):
        self.requests.append({"route": route, "payload": payload})
        if route == "/api/show":
            return _FakeResponse(self._show)
        if stream:
            return _FakeResponse({}, self._stream_lines)
        return _FakeResponse(
            {"message": {"content": "hello", "thinking": "because"}, "done": True}
        )

    @property
    def chat_payload(self) -> Dict[str, Any]:
        for entry in reversed(self.requests):
            if entry["route"] == "/api/chat":
                return entry["payload"]
        raise AssertionError("no chat request was made")


class TestClientThinking:
    def test_show_is_cached(self):
        client = _RecordingClient(GPT_OSS)
        client.thinking_support("gpt-oss:20b")
        client.thinking_support("gpt-oss:20b")
        assert sum(1 for r in client.requests if r["route"] == "/api/show") == 1

    def test_capability_cache_can_be_cleared(self):
        client = _RecordingClient(GPT_OSS)
        client.thinking_support("gpt-oss:20b")
        client.clear_capability_cache()
        client.thinking_support("gpt-oss:20b")
        assert sum(1 for r in client.requests if r["route"] == "/api/show") == 2

    def test_model_capabilities_are_exposed(self):
        client = _RecordingClient(GPT_OSS)
        assert "tools" in client.model_capabilities("gpt-oss:20b")
        assert client.supports_tools("gpt-oss:20b")

    def test_supported_level_reaches_the_wire(self):
        client = _RecordingClient(GPT_OSS)
        client.chat([ChatMessage.user("hi")], model="gpt-oss:20b", think="high")
        assert client.chat_payload["think"] == "high"

    def test_unsupported_level_is_not_sent(self):
        client = _RecordingClient(QWEN3)
        client.chat([ChatMessage.user("hi")], model="qwen3:8b", think="high")
        assert "think" not in client.chat_payload

    def test_auto_sends_nothing(self):
        client = _RecordingClient(GPT_OSS)
        client.chat([ChatMessage.user("hi")], model="gpt-oss:20b", think=THINK_AUTO)
        assert "think" not in client.chat_payload

    def test_off_is_sent_as_false(self):
        client = _RecordingClient(QWEN3)
        client.chat([ChatMessage.user("hi")], model="qwen3:8b", think=THINK_OFF)
        assert client.chat_payload["think"] is False

    def test_settings_supply_the_default(self):
        client = _RecordingClient(QWEN3)
        client.settings.think = THINK_ON
        client.chat([ChatMessage.user("hi")], model="qwen3:8b")
        assert client.chat_payload["think"] is True

    def test_non_streaming_splits_answer_and_reasoning(self):
        client = _RecordingClient(QWEN3)
        text, _usage = client.chat([ChatMessage.user("hi")], model="qwen3:8b")
        assert text == "hello"
        assert client.last_thinking == "because"

    def test_streaming_keeps_reasoning_out_of_the_text(self):
        lines = [
            json.dumps({"message": {"thinking": "step one "}}),
            json.dumps({"message": {"thinking": "step two"}}),
            json.dumps({"message": {"content": "ans"}}),
            json.dumps({"message": {"content": "wer"}, "done": True, "eval_count": 3,
                        "prompt_eval_count": 7}),
        ]
        client = _RecordingClient(QWEN3, stream_lines=lines)
        thoughts: List[str] = []
        answer: List[str] = []
        text, usage = client.chat(
            [ChatMessage.user("hi")],
            model="qwen3:8b",
            on_delta=answer.append,
            on_thinking=thoughts.append,
        )
        assert text == "answer"
        assert "".join(thoughts) == "step one step two"
        assert "".join(answer) == "answer"
        assert client.last_thinking == "step one step two"
        assert usage.completion_tokens == 3

    def test_stream_events_carry_thinking(self):
        lines = [json.dumps({"message": {"thinking": "t"}, "done": True})]
        client = _RecordingClient(QWEN3, stream_lines=lines)
        events = list(client.chat_stream([ChatMessage.user("hi")], model="qwen3:8b"))
        assert events[0].thinking == "t"
        assert events[0].has_payload

    def test_stream_event_without_payload(self):
        assert StreamEvent(done=True).has_payload is False

    def test_resolve_think_helper(self):
        client = _RecordingClient(GPT_OSS)
        assert client.resolve_think("low", "gpt-oss:20b") == "low"


class TestSettingsIntegration:
    def test_default_is_auto(self):
        assert OllamaSettings().think == THINK_AUTO

    def test_show_thinking_defaults_on(self):
        assert OllamaSettings().show_thinking is True

    def test_think_survives_a_config_round_trip(self):
        from agent3.core.config import AppConfig

        config = AppConfig.from_dict({"ollama": {"think": "high", "show_thinking": False}})
        assert config.ollama.think == "high"
        assert config.ollama.show_thinking is False
