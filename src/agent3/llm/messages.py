"""Transport-neutral chat primitives shared by the client and the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    """Conversation roles understood by Ollama's ``/api/chat`` endpoint."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass
class ChatMessage:
    """A single conversational turn."""

    role: Role
    content: str
    name: Optional[str] = None
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Serialise to the wire format expected by Ollama."""
        payload: Dict[str, Any] = {"role": str(self.role), "content": self.content}
        if self.images:
            payload["images"] = list(self.images)
        return payload

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls(Role.USER, content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        return cls(Role.ASSISTANT, content)

    @classmethod
    def tool(cls, content: str, name: str = "tool") -> "ChatMessage":
        return cls(Role.TOOL, content, name=name)


@dataclass
class Usage:
    """Token / timing accounting for one completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ms: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    def merge(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_duration_ms=self.total_duration_ms + other.total_duration_ms,
            estimated=self.estimated or other.estimated,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "estimated": self.estimated,
        }


@dataclass
class StreamEvent:
    """One chunk of a streamed completion.

    ``delta`` carries the newly produced answer text; ``thinking`` carries the
    model's reasoning trace, which Ollama streams in a *separate*
    ``message.thinking`` field so it never contaminates the answer (and, for
    this agent, never gets parsed as a tool call). The final event has
    ``done=True`` and a populated :class:`Usage`.
    """

    delta: str = ""
    done: bool = False
    usage: Optional[Usage] = None
    model: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    thinking: str = ""

    @property
    def has_payload(self) -> bool:
        """True when the chunk carried any text at all."""
        return bool(self.delta or self.thinking)
