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
    KNOWN_THINK_LEVELS,
    SOURCE_METADATA,
    SOURCE_TEMPLATE,
    THINK_AUTO,
    THINK_OFF,
    THINK_ON,
    OllamaClient,
    ThinkingSupport,
)
from agent3.llm.reasoning import (
    SOFT_SWITCH_OFF,
    ReasoningFilter,
    parse_template_controls,
    prompt_controls,
    split_inline_reasoning,
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

#: Trimmed but faithful excerpt of the Qwen3.x chat template carried by many
#: community GGUF repacks. The level names, the "undefined means on" default
#: and the instruction sentences are reproduced exactly as they ship, because
#: discovery reads precisely these lines.
QWEN35_TEMPLATE = """
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort value') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, and consider edge cases before answering.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
{%- if '/no_think' in last_user_message %}{%- set enable_thinking = false %}{%- endif %}
{{- '<|im_start|>assistant\\n' }}
"""

#: A plain on/off template: a switch, no effort levels.
TOGGLE_TEMPLATE = """
{%- if enable_thinking %}{{- '<think>' }}{%- endif %}
{{- '<|im_start|>assistant' }}
"""

TEMPLATE_MODEL = {"capabilities": ["completion"], "template": QWEN35_TEMPLATE}


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

    def test_a_famous_name_without_evidence_gets_nothing(self):
        """Classification follows the server, never the model's name."""
        support = ThinkingSupport.from_show("gpt-oss:120b", {"capabilities": ["completion"]})
        assert support.supported is False
        assert support.levels == ()
        assert support.source == "none"

    def test_an_unknown_name_with_evidence_gets_everything(self):
        """...and an obscure community repack is judged on its template."""
        support = ThinkingSupport.from_show(
            "bernquant/Qwen3.5-9B-The-Defiant-Fable-Uncensored-NEO-IMATRIX-GGUF",
            {"capabilities": ["completion"], "template": QWEN35_TEMPLATE},
        )
        assert support.supported
        assert support.levels == ("xhigh", "medium", "low")

    def test_server_metadata_beats_every_other_source(self):
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

    def __init__(
        self,
        show: Dict[str, Any],
        stream_lines: List[str] | None = None,
        *,
        reply: str = "hello",
        thinking: str = "because",
        stream: bool | None = None,
    ) -> None:
        super().__init__(
            OllamaSettings(stream=bool(stream_lines) if stream is None else stream)
        )
        self._show = show
        self._stream_lines = stream_lines or []
        self._reply = reply
        self._thinking = thinking
        self.requests: List[Dict[str, Any]] = []

    def _request(self, method, route, *, payload=None, stream=False, cancel=None, timeout=None):
        self.requests.append({"route": route, "payload": payload})
        if route == "/api/show":
            return _FakeResponse(self._show)
        if stream:
            return _FakeResponse({}, self._stream_lines)
        return _FakeResponse(
            {
                "message": {"content": self._reply, "thinking": self._thinking},
                "done": True,
            }
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


class TestTemplateDiscovery:
    """Reasoning support is read out of the model's own chat template."""

    def test_levels_come_from_the_templates_validation_tuple(self):
        controls = parse_template_controls(QWEN35_TEMPLATE)
        assert controls.has_effort
        assert controls.levels == ("xhigh", "medium", "low")
        assert controls.default == "xhigh"

    def test_thinking_is_on_when_the_switch_is_undefined(self):
        controls = parse_template_controls(QWEN35_TEMPLATE)
        assert controls.has_switch
        assert controls.thinking_is_default is True

    def test_soft_switch_is_detected(self):
        assert parse_template_controls(QWEN35_TEMPLATE).soft_switch is True
        assert parse_template_controls(TOGGLE_TEMPLATE).soft_switch is False

    def test_instruction_sentences_are_lifted_verbatim(self):
        controls = parse_template_controls(QWEN35_TEMPLATE)
        assert controls.instructions["low"].startswith("Reasoning effort is set to low.")
        assert "edge cases" in controls.instructions["xhigh"]
        # medium carries no sentence in the template, so we must not invent one
        assert "medium" not in controls.instructions

    def test_toggle_template_offers_on_off_only(self):
        support = ThinkingSupport.from_show("toggler", {"template": TOGGLE_TEMPLATE})
        assert support.supported and support.can_disable
        assert support.levels == ()
        assert dict(support.choices()) == {"auto": "Auto", "off": "Off", "on": "On"}

    def test_template_levels_reach_the_selector(self):
        support = ThinkingSupport.from_show("repack", TEMPLATE_MODEL)
        assert support.source == "template"
        assert dict(support.choices()) == {
            "auto": "Auto",
            "off": "Off",
            "xhigh": "X-High",
            "medium": "Medium",
            "low": "Low",
        }

    def test_an_empty_template_proves_nothing(self):
        support = ThinkingSupport.from_show("bare", {"template": "{{ messages }}"})
        assert support.supported is False
        assert support.source == "none"

    def test_template_beats_a_bare_capability_flag(self):
        support = ThinkingSupport.from_show(
            "repack", {"capabilities": ["thinking"], "template": QWEN35_TEMPLATE}
        )
        assert support.source == "template"
        assert support.levels == ("xhigh", "medium", "low")

    def test_metadata_beats_the_template(self):
        support = ThinkingSupport.from_show(
            "repack",
            {"template": QWEN35_TEMPLATE, "thinking": {"values": [True, False]}},
        )
        assert support.source == "metadata"
        assert support.levels == ()

    def test_native_models_need_no_prompt_help(self):
        assert ThinkingSupport.from_show("gpt-oss:20b", GPT_OSS).native is True
        assert ThinkingSupport.from_show("gpt-oss:20b", GPT_OSS).needs_prompt_enforcement is False

    def test_template_models_do_need_prompt_help(self):
        support = ThinkingSupport.from_show("repack", TEMPLATE_MODEL)
        assert support.native is False
        assert support.needs_prompt_enforcement is True


class TestPromptEnforcement:
    """What we say in the prompt when the transport cannot be trusted."""

    def _controls(self, spec, data=TEMPLATE_MODEL):
        support = ThinkingSupport.from_show("repack", data)
        enabled, level = support.intent(spec)
        return prompt_controls(
            enabled=enabled,
            level=level,
            controls=support.controls,
            native=support.native,
        )

    def test_off_sends_the_soft_switch_and_an_instruction(self):
        controls = self._controls(THINK_OFF)
        assert controls.suffix == SOFT_SWITCH_OFF
        assert "no <think> block" in controls.system

    def test_a_level_reuses_the_templates_own_sentence(self):
        controls = self._controls("low")
        assert controls.system.startswith("Reasoning effort is set to low.")
        assert controls.suffix == ""

    def test_a_level_without_a_sentence_gets_a_neutral_one(self):
        controls = self._controls("medium")
        assert "medium" in controls.system

    def test_on_resolves_to_the_templates_default_level(self):
        controls = self._controls(THINK_ON)
        assert "xhigh" in controls.system

    def test_auto_stays_silent(self):
        assert not self._controls(THINK_AUTO)

    def test_native_models_are_left_alone(self):
        support = ThinkingSupport.from_show("gpt-oss:20b", GPT_OSS)
        enabled, level = support.intent("high")
        controls = prompt_controls(
            enabled=enabled, level=level, controls=support.controls, native=True
        )
        assert not controls

    def test_instructions_are_woven_into_the_conversation(self):
        client = _RecordingClient(TEMPLATE_MODEL)
        client.chat([ChatMessage.system("Be terse."), ChatMessage.user("hi")], think=THINK_OFF)
        messages = client.chat_payload["messages"]
        assert messages[0]["role"] == "system"
        assert "no <think> block" in messages[0]["content"]
        assert "Be terse." in messages[0]["content"]
        assert messages[-1]["content"].endswith(SOFT_SWITCH_OFF)

    def test_a_system_turn_is_created_when_there_is_none(self):
        client = _RecordingClient(TEMPLATE_MODEL)
        client.chat([ChatMessage.user("hi")], think="low")
        messages = client.chat_payload["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith("Reasoning effort is set to low.")

    def test_enforcement_can_be_switched_off(self):
        client = _RecordingClient(TEMPLATE_MODEL)
        client.settings.enforce_think_in_prompt = False
        client.chat([ChatMessage.user("hi")], think=THINK_OFF)
        assert client.chat_payload["messages"] == [{"role": "user", "content": "hi"}]
        assert client.chat_payload["think"] is False


class TestReasoningFilter:
    """Inline <think> blocks never reach the answer."""

    def test_a_whole_block_is_removed(self):
        assert split_inline_reasoning("<think>plan</think>done") == ("done", "plan")

    def test_text_without_tags_is_untouched(self):
        assert split_inline_reasoning("just an answer") == ("just an answer", "")

    def test_a_tag_split_across_chunks_is_still_caught(self):
        stream = ReasoningFilter()
        answer, reasoning = "", ""
        for chunk in ["Hel", "lo <thi", "nk>secret ", "plan</th", "ink> world"]:
            a, r = stream.feed(chunk)
            answer += a
            reasoning += r
        a, r = stream.flush()
        assert answer + a == "Hello  world"
        assert reasoning + r == "secret plan"

    def test_an_unterminated_block_is_not_leaked_as_answer(self):
        stream = ReasoningFilter()
        answer, reasoning = stream.feed("<think>still going")
        tail_answer, tail_reasoning = stream.flush()
        assert answer + tail_answer == ""
        assert reasoning + tail_reasoning == "still going"

    def test_alternative_delimiters(self):
        assert split_inline_reasoning("◁think▷hm◁/think▷ok") == ("ok", "hm")
        assert split_inline_reasoning("<reasoning>hm</reasoning>ok") == ("ok", "hm")

    def test_several_blocks_in_one_reply(self):
        answer, reasoning = split_inline_reasoning("<think>a</think>X<think>b</think>Y")
        assert answer == "XY"
        assert reasoning == "ab"

    def test_a_lone_angle_bracket_is_not_swallowed(self):
        assert split_inline_reasoning("if a < b then") == ("if a < b then", "")

    def test_the_filter_reports_that_it_fired(self):
        stream = ReasoningFilter()
        stream.feed("plain text")
        assert stream.saw_inline is False
        stream.feed("<think>x</think>")
        assert stream.saw_inline is True


class TestLeakedReasoning:
    """The client notices when a model reasons despite being told not to."""

    def test_inline_trace_is_routed_to_the_thinking_channel(self):
        client = _RecordingClient(
            TEMPLATE_MODEL, reply="<think>scheming</think>Hello", thinking="", stream=False
        )
        text, _ = client.chat([ChatMessage.user("hi")], think=THINK_OFF)
        assert text == "Hello"
        assert client.last_thinking == "scheming"
        assert client.last_reasoning_leaked is True
        assert client.last_think_ignored is True

    def test_a_quiet_model_sets_no_flags(self):
        client = _RecordingClient(TEMPLATE_MODEL, reply="Hello", thinking="", stream=False)
        client.chat([ChatMessage.user("hi")], think=THINK_OFF)
        assert client.last_reasoning_leaked is False
        assert client.last_think_ignored is False

    def test_stripping_can_be_disabled(self):
        client = _RecordingClient(
            TEMPLATE_MODEL, reply="<think>x</think>Hi", thinking="", stream=False
        )
        client.settings.strip_inline_reasoning = False
        text, _ = client.chat([ChatMessage.user("hi")], think=THINK_AUTO)
        assert text == "<think>x</think>Hi"

    def test_json_mode_is_never_filtered(self):
        client = _RecordingClient(
            TEMPLATE_MODEL, reply='{"a": "<think>"}', thinking="", stream=False
        )
        text, _ = client.chat([ChatMessage.user("hi")], format_json=True)
        assert text == '{"a": "<think>"}'


class TestLoopWarning:
    """The agent loop surfaces an ignored "thinking off" setting once."""

    class _Stub:
        last_thinking = "x"
        last_think_ignored = True

    def _loop(self):
        from agent3.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.client = self._Stub()
        loop._think_warning_sent = False
        seen: List[str] = []

        class _CB:
            @staticmethod
            def emit(name, *args):
                if name == "on_status":
                    seen.append(args[0])

        loop.callbacks = _CB()
        return loop, seen

    def test_the_notice_is_emitted(self):
        loop, seen = self._loop()
        loop._warn_if_think_ignored()
        assert seen and "keeps reasoning" in seen[0]

    def test_the_notice_is_not_repeated(self):
        loop, seen = self._loop()
        loop._warn_if_think_ignored()
        loop._warn_if_think_ignored()
        assert len(seen) == 1

    def test_a_well_behaved_model_triggers_nothing(self):
        loop, seen = self._loop()
        loop.client.last_think_ignored = False
        loop._warn_if_think_ignored()
        assert seen == []
