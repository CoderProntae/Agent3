"""Model-agnostic token estimation.

Ollama reports authoritative counts (``prompt_eval_count`` /``eval_count``)
once a request finishes, but the quota engine has to *pre-authorise* a call
before a single byte is sent.  The heuristic below is deliberately simple,
dependency free and conservative (it slightly over-estimates), which is the
right bias for a rate limiter.

Calibration: for Latin-script source code and prose, BPE tokenizers average
~3.6 characters per token; whitespace-separated words average ~1.3 tokens.  We
blend both signals and add a small per-message overhead for chat templates.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from agent3.llm.messages import ChatMessage

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CHARS_PER_TOKEN = 3.6
_TOKENS_PER_WORD = 1.32
#: Chat templates wrap every message with role markers / separators.
MESSAGE_OVERHEAD_TOKENS = 4
#: Non-ASCII heavy text (CJK, Turkish, emoji) tokenises less efficiently.
_NON_ASCII_PENALTY = 0.35


def estimate_tokens(text: str) -> int:
    """Return a conservative token estimate for *text*."""
    if not text:
        return 0
    char_estimate = len(text) / _CHARS_PER_TOKEN
    word_estimate = len(_WORD_RE.findall(text)) * _TOKENS_PER_WORD
    blended = (char_estimate + word_estimate) / 2.0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    blended += non_ascii * _NON_ASCII_PENALTY
    return max(1, int(round(blended)))


def count_message_tokens(messages: Sequence[ChatMessage] | Iterable[ChatMessage]) -> int:
    """Estimate the prompt size of a whole conversation."""
    total = 0
    for message in messages:
        total += MESSAGE_OVERHEAD_TOKENS + estimate_tokens(message.content or "")
        if message.name:
            total += estimate_tokens(message.name)
    return total


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Trim *text* so that it fits into roughly *max_tokens* tokens.

    Used when a tool produces a huge stdout blob that would otherwise blow the
    context window.  The middle is removed, keeping head and tail which is
    where the useful signal (command + error) usually lives.
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    budget_chars = int(max_tokens * _CHARS_PER_TOKEN)
    head = budget_chars // 2
    tail = budget_chars - head
    omitted = len(text) - head - tail
    if omitted <= 0:  # pragma: no cover - defensive
        return text[:budget_chars]
    return f"{text[:head]}\n\n... [{omitted} characters truncated by Agent3] ...\n\n{text[-tail:]}"
