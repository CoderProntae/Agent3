"""Discovering and enforcing a model's *real* reasoning behaviour.

Two problems this module exists to solve, both of them observed in the wild
with community GGUF builds served by Ollama:

1. **Discovery by name is wrong.** ``gpt-oss`` is not the only model with
   named effort levels, and a repack called ``Qwen3.5-9B-...-GGUF`` may well
   expose ``low``/``medium``/``xhigh`` through its chat template. The only
   honest sources are the server's ``/api/show`` ``thinking`` metadata and,
   failing that, the model's own chat template - which literally contains the
   list of values it validates against::

       {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
       {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
           {{- raise_exception('Unexpected reasoning effort ...') }}

   :func:`parse_template_controls` reads exactly that, so the levels offered
   in the UI are the levels *this file on this machine* accepts.

2. **"Thinking off" is frequently ignored.** Many templates start with
   ``{%- if enable_thinking is undefined or enable_thinking is true %}``,
   i.e. thinking is on unless something explicitly turns it off - and the
   ``think`` request field does not always reach the template. The model then
   reasons anyway, and the trace arrives inline in ``message.content`` wrapped
   in ``<think>`` tags instead of in the separate ``message.thinking`` field.

   Two defences, applied together:

   * :func:`prompt_controls` reproduces, in the prompt, what the template
     would have done: Qwen's documented ``/no_think`` soft switch to turn
     reasoning off, and the template's *own* effort instruction text to set a
     level. Nothing is invented - the instruction strings are extracted from
     the template when it carries them.
   * :class:`ReasoningFilter` strips inline ``<think>`` blocks out of the
     answer stream and routes them to the reasoning channel. The answer stays
     clean, the tool-call parser never sees the model's scratchpad, and the
     application can tell the user that the model ignored the switch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from agent3.core.logging_setup import get_logger

logger = get_logger(__name__)

#: Inline reasoning delimiters emitted by models whose server does not split
#: the trace into its own field. Ordered longest-first so that overlapping
#: prefixes cannot shadow each other.
REASONING_TAGS: Tuple[Tuple[str, str], ...] = (
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
    ("<reflection>", "</reflection>"),
    ("<|begin_of_thought|>", "<|end_of_thought|>"),
    ("<think>", "</think>"),
    ("◁think▷", "◁/think▷"),
)

#: Template identifiers that switch reasoning on and off.
_ENABLE_VARS = ("enable_thinking", "add_thinking", ".Thinking", "thinking_enabled")

#: Template identifiers that select an effort level.
_EFFORT_VARS = ("reasoning_effort", "thinking_effort", ".ThinkLevel", ".ThinkingLevel")

#: Soft switches documented by the Qwen family and honoured by their template.
SOFT_SWITCH_OFF = "/no_think"
SOFT_SWITCH_ON = "/think"

_QUOTED = re.compile(r"""['"]([A-Za-z][A-Za-z0-9_-]{1,14})['"]""")


#: Display names for level tokens that do not capitalise nicely.
_LEVEL_LABELS = {
    "xhigh": "X-High",
    "xlow": "X-Low",
    "none": "None",
    "minimal": "Minimal",
}


def level_label(level: str) -> str:
    """Human readable name for a raw effort token from a chat template."""
    key = str(level).strip().lower()
    return _LEVEL_LABELS.get(key, key.replace("_", " ").replace("-", " ").title())


@dataclass
class TemplateControls:
    """What a chat template really reads, as found in its source text."""

    #: Template exposes an on/off knob (``enable_thinking`` and friends).
    has_switch: bool = False
    #: Template exposes a named effort knob (``reasoning_effort`` and friends).
    has_effort: bool = False
    #: Effort values the template validates against, in template order.
    levels: Tuple[str, ...] = ()
    #: Value used when the caller says nothing.
    default: Optional[str] = None
    #: Reasoning is on unless explicitly disabled (``is undefined or is true``).
    thinking_is_default: bool = False
    #: Template honours the ``/no_think`` soft switch.
    soft_switch: bool = False
    #: Per level, the instruction the template injects into the system turn.
    instructions: Dict[str, str] = field(default_factory=dict)

    @property
    def found_anything(self) -> bool:
        return self.has_switch or self.has_effort or self.soft_switch


def _find_variable(text: str, names: Sequence[str]) -> str:
    for name in names:
        if name in text:
            return name
    return ""


def _collect_levels(text: str, variable: str) -> Tuple[List[str], Optional[str]]:
    """Pull the accepted effort values out of the template's own validation.

    Templates express this in one of two shapes::

        reasoning_effort|default('xhigh')
        resolved_reasoning_effort not in ('xhigh', 'medium', 'low')

    so we follow the variable (including a ``resolved_``-style alias) and keep
    every quoted literal that it is compared against, in order of appearance.
    """
    if not variable:
        return [], None

    # The template usually copies the variable into a local alias first.
    aliases = {variable.lstrip(".")}
    for match in re.finditer(
        r"set\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^\n]*" + re.escape(variable.lstrip(".")), text
    ):
        aliases.add(match.group(1))

    default: Optional[str] = None
    default_match = re.search(
        re.escape(variable.lstrip(".")) + r"\s*\|\s*default\(\s*['\"]([A-Za-z0-9_-]+)['\"]",
        text,
    )
    if default_match:
        default = default_match.group(1)

    levels: List[str] = []
    for line in text.splitlines():
        if not any(alias in line for alias in aliases):
            continue
        if not any(token in line for token in (" in ", "==", "!=", "default(")):
            continue
        for candidate in _QUOTED.findall(line):
            lowered = candidate.lower()
            if lowered in {alias.lower() for alias in aliases}:
                continue
            if lowered in ("true", "false", "none", "undefined", "default"):
                continue
            if lowered not in levels:
                levels.append(lowered)
    if default and default not in levels:
        levels.insert(0, default)
    return levels, default


def _collect_instructions(text: str, levels: Sequence[str]) -> Dict[str, str]:
    """Extract the sentence the template injects for each effort level.

    Qwen's template carries them verbatim::

        {%- if resolved_reasoning_effort == 'low' %}
            {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your ...' %}

    Reproducing *that exact string* in a system message is what makes the
    level work even when the server never renders it.
    """
    instructions: Dict[str, str] = {}
    for level in levels:
        pattern = re.compile(
            r"==\s*['\"]" + re.escape(level) + r"['\"].{0,200}?"
            r"=\s*'((?:[^'\\]|\\.){20,600})'",
            re.DOTALL,
        )
        match = pattern.search(text)
        if match:
            sentence = match.group(1).replace("\\'", "'").strip()
            if sentence and " " in sentence:
                instructions[level] = sentence
    return instructions


def parse_template_controls(template: str) -> TemplateControls:
    """Read a chat template and report the reasoning knobs it actually has."""
    controls = TemplateControls()
    if not template or not isinstance(template, str):
        return controls

    switch_var = _find_variable(template, _ENABLE_VARS)
    effort_var = _find_variable(template, _EFFORT_VARS)
    controls.has_switch = bool(switch_var)
    controls.has_effort = bool(effort_var)
    controls.soft_switch = "no_think" in template or "nothink" in template.replace("-", "_")

    if switch_var:
        # "enable_thinking is undefined or enable_thinking is true" means the
        # model reasons unless the caller actively says otherwise.
        escaped = re.escape(switch_var.lstrip("."))
        controls.thinking_is_default = bool(
            re.search(escaped + r"\s+is\s+undefined", template)
            or re.search(r"default\(\s*true\s*\)", template)
        )

    levels, default = _collect_levels(template, effort_var)
    controls.levels = tuple(levels)
    controls.default = default
    if levels:
        controls.instructions = _collect_instructions(template, levels)
    return controls


# --------------------------------------------------------------- enforcement
@dataclass
class PromptControls:
    """Prompt-level instructions that enforce a reasoning setting.

    Used when the transport cannot be trusted to deliver the request field -
    the common case for community GGUF repacks, where ``think`` never reaches
    the chat template and the model reasons at its default effort regardless.
    """

    #: Appended to the last user message (``/no_think`` and friends).
    suffix: str = ""
    #: Added as an extra system message.
    system: str = ""

    def __bool__(self) -> bool:
        return bool(self.suffix or self.system)


#: Fallback wording used when the template does not carry its own sentence.
_GENERIC_EFFORT = (
    "Reasoning effort is set to {level}. "
    "Match the depth of your private reasoning to that level before answering."
)

_HARD_OFF = (
    "Do not produce a reasoning preamble. Answer directly, with no <think> block "
    "and no step-by-step narration of your own deliberation."
)


def prompt_controls(
    *,
    enabled: Optional[bool],
    level: str,
    controls: TemplateControls,
    native: bool,
) -> PromptControls:
    """Work out what has to be said in the prompt to make the setting stick.

    *native* is ``True`` when the server advertised proper ``thinking``
    metadata, which means it honours the ``think`` request field itself and
    nothing extra is needed. Everything else is a best effort that mirrors
    what the template would have done on its own.
    """
    result = PromptControls()
    if native or enabled is None:
        return result

    if enabled is False:
        # Turning reasoning off is the case that silently fails most often,
        # so it gets both the soft switch and an explicit instruction.
        if controls.soft_switch:
            result.suffix = SOFT_SWITCH_OFF
        if controls.thinking_is_default or controls.has_switch or controls.soft_switch:
            result.system = _HARD_OFF
        return result

    if level and controls.has_effort:
        sentence = controls.instructions.get(level.lower())
        result.system = sentence or _GENERIC_EFFORT.format(level=level)
    elif controls.soft_switch:
        result.suffix = SOFT_SWITCH_ON
    return result


# ------------------------------------------------------------------ filter
def _longest_partial_tag(text: str, tags: Sequence[str]) -> int:
    """Length of the trailing run that might still grow into one of *tags*."""
    longest = 0
    for tag in tags:
        limit = min(len(tag) - 1, len(text))
        for size in range(limit, 0, -1):
            if text.endswith(tag[:size]):
                longest = max(longest, size)
                break
    return longest


class ReasoningFilter:
    """Split a token stream into answer text and inline reasoning text.

    Streaming-safe: a tag arriving split across two chunks is held back until
    it can be resolved, so ``<thi`` + ``nk>`` is recognised as one tag and
    never leaks into the answer.
    """

    def __init__(self, tags: Sequence[Tuple[str, str]] = REASONING_TAGS) -> None:
        self._tags = tuple(tags)
        self._open = tuple(pair[0] for pair in self._tags)
        self._close_for: Dict[str, str] = {pair[0]: pair[1] for pair in self._tags}
        self._pending = ""
        self._inside: Optional[str] = None
        #: True once an inline reasoning block has been seen.
        self.saw_inline = False

    @property
    def inside(self) -> bool:
        return self._inside is not None

    def feed(self, chunk: str) -> Tuple[str, str]:
        """Consume *chunk*; return ``(answer_delta, reasoning_delta)``."""
        if not chunk:
            return "", ""
        self._pending += chunk
        answer: List[str] = []
        reasoning: List[str] = []

        while True:
            if self._inside is None:
                index, tag = self._earliest(self._pending, self._open)
                if tag is not None:
                    answer.append(self._pending[:index])
                    self._pending = self._pending[index + len(tag) :]
                    self._inside = tag
                    self.saw_inline = True
                    continue
                hold = _longest_partial_tag(self._pending, self._open)
                if hold:
                    answer.append(self._pending[: len(self._pending) - hold])
                    self._pending = self._pending[len(self._pending) - hold :]
                else:
                    answer.append(self._pending)
                    self._pending = ""
                break

            close = self._close_for[self._inside]
            index = self._pending.find(close)
            if index >= 0:
                reasoning.append(self._pending[:index])
                self._pending = self._pending[index + len(close) :]
                self._inside = None
                continue
            hold = _longest_partial_tag(self._pending, (close,))
            if hold:
                reasoning.append(self._pending[: len(self._pending) - hold])
                self._pending = self._pending[len(self._pending) - hold :]
            else:
                reasoning.append(self._pending)
                self._pending = ""
            break

        return "".join(answer), "".join(reasoning)

    def flush(self) -> Tuple[str, str]:
        """Release whatever is still buffered at the end of the stream."""
        rest, self._pending = self._pending, ""
        if self._inside is not None:
            self._inside = None
            return "", rest
        return rest, ""

    @staticmethod
    def _earliest(text: str, tags: Sequence[str]) -> Tuple[int, Optional[str]]:
        best_index = -1
        best_tag: Optional[str] = None
        for tag in tags:
            index = text.find(tag)
            if index >= 0 and (best_index < 0 or index < best_index):
                best_index, best_tag = index, tag
        return best_index, best_tag


def split_inline_reasoning(text: str) -> Tuple[str, str]:
    """One-shot version of :class:`ReasoningFilter` for non-streamed replies."""
    stream = ReasoningFilter()
    answer, reasoning = stream.feed(text)
    tail_answer, tail_reasoning = stream.flush()
    return answer + tail_answer, reasoning + tail_reasoning
