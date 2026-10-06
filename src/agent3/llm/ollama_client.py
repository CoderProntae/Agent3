"""Robust HTTP client for a local Ollama inference server.

Highlights
----------
* Targets ``http://localhost:11435`` by default (configurable, with automatic
  fallback endpoints such as ``127.0.0.1:11435`` and the stock ``11434``).
* Chunked / NDJSON **streaming** of ``/api/chat`` responses, surfaced as
  :class:`~agent3.llm.messages.StreamEvent` objects.
* Connection pooling with HTTP keep-alive, separate connect/read timeouts,
  exponential backoff with jitter, and retry only on *safe* failures
  (connection refused, read timeout, 5xx, 429).
* Cooperative cancellation through a :class:`threading.Event`.
* Authoritative token usage when the server reports it, estimated otherwise.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import requests
from requests.adapters import HTTPAdapter

from agent3.core.config import OllamaSettings
from agent3.core.logging_setup import get_logger
from agent3.llm.messages import ChatMessage, Role, StreamEvent, Usage
from agent3.llm.reasoning import (
    SOFT_SWITCH_OFF,
    PromptControls,
    level_label,
    ReasoningFilter,
    TemplateControls,
    parse_template_controls,
    prompt_controls,
    split_inline_reasoning,
)
from agent3.llm.tokenizer import count_message_tokens, estimate_tokens

logger = get_logger(__name__)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: ``think`` spec meaning "let the model decide" (the field is omitted).
THINK_AUTO = "auto"
#: ``think`` spec meaning "no reasoning trace" (sends ``think: false``).
THINK_OFF = "off"
#: ``think`` spec meaning "reason" (sends ``think: true``).
THINK_ON = "on"

#: Values Ollama's API layer accepts in the ``think`` field. Used only to
#: sanity-check a level that came from somewhere else - never as a menu.
KNOWN_THINK_LEVELS = ("low", "medium", "high", "max")

#: Where a model's reasoning capability was established, worst to best.
SOURCE_NONE = "none"
SOURCE_CAPABILITY = "capability"
SOURCE_TEMPLATE = "template"
SOURCE_METADATA = "metadata"


@dataclass(frozen=True)
class ThinkingSupport:
    """What a specific model really accepts in the ``think`` request field.

    Established from the server, never from the model's name, in this order:

    1. ``/api/show`` → ``thinking: {"values": [...], "default": ...}``.
       Authoritative: the server itself will honour the ``think`` field.
    2. The model's **chat template**, also returned by ``/api/show``. A GGUF
       repack usually carries its controls there - ``enable_thinking`` for
       on/off, ``reasoning_effort`` validated against a literal tuple such as
       ``('xhigh', 'medium', 'low')`` for the levels. These are the real
       values *this file* accepts, whatever the model is called.
    3. ``capabilities`` containing ``"thinking"`` → on/off only.
    4. Nothing → the model does not reason.

    Only case 1 is delivered natively by the server. In cases 2 and 3 the
    setting has to be reproduced in the prompt (see
    :mod:`agent3.llm.reasoning`), because the ``think`` field is frequently
    dropped before it reaches the template.
    """

    model: str = ""
    values: tuple[Any, ...] = ()
    default: Any = None
    capabilities: tuple[str, ...] = ()
    known: bool = False
    #: One of ``metadata`` / ``template`` / ``capability`` / ``none``.
    source: str = SOURCE_NONE
    #: Reasoning knobs found in the chat template (empty when unused).
    controls: TemplateControls = field(default_factory=TemplateControls)

    # -------------------------------------------------------------- shape
    @property
    def levels(self) -> tuple[str, ...]:
        """Named levels (``low``/``xhigh``/...) the model understands."""
        return tuple(str(v) for v in self.values if isinstance(v, str))

    @property
    def can_enable(self) -> bool:
        return bool(self.levels) or True in self.values

    @property
    def can_disable(self) -> bool:
        """True when the trace can be suppressed at all."""
        if self.source == SOURCE_METADATA:
            return False in self.values
        if self.source == SOURCE_TEMPLATE:
            # A template with an on/off knob can always be turned off, even
            # if the knob only works through the prompt.
            return self.controls.has_switch or self.controls.soft_switch
        return self.supported

    @property
    def supported(self) -> bool:
        if self.source == SOURCE_NONE:
            return False
        if self.known:
            return self.can_enable
        return True

    @property
    def forced(self) -> bool:
        """True when the model always reasons and cannot be turned off."""
        return self.supported and not self.can_disable

    @property
    def native(self) -> bool:
        """True when the server honours ``think`` on its own."""
        return self.source == SOURCE_METADATA

    @property
    def needs_prompt_enforcement(self) -> bool:
        """True when the setting has to be restated in the prompt."""
        return self.supported and not self.native

    @property
    def thinks_by_default(self) -> bool:
        """True when the model reasons unless actively told not to."""
        if self.source == SOURCE_METADATA:
            return self.default not in (None, False)
        return self.controls.thinking_is_default or bool(self.levels)

    def source_label(self) -> str:
        return {
            SOURCE_METADATA: "reported by the server",
            SOURCE_TEMPLATE: "read from the chat template",
            SOURCE_CAPABILITY: "capability flag only",
            SOURCE_NONE: "not supported",
        }.get(self.source, self.source)

    # ------------------------------------------------------------ choices
    def choices(self) -> List[tuple[str, str]]:
        """``(spec, label)`` pairs for a UI selector - real values only."""
        options: List[tuple[str, str]] = [(THINK_AUTO, "Auto")]
        if not self.supported:
            return options
        if self.can_disable:
            options.append((THINK_OFF, "Off"))
        if True in self.values or (not self.levels and self.supported):
            options.append((THINK_ON, "On"))
        for level in self.levels:
            options.append((level, level_label(level)))
        return options

    def accepts(self, spec: str) -> bool:
        key = str(spec or "").strip().lower()
        if key in ("", THINK_AUTO):
            return True
        if key == THINK_OFF:
            return self.can_disable
        if key == THINK_ON:
            return self.supported and (True in self.values or not self.levels)
        return key in {level.lower() for level in self.levels}

    # ------------------------------------------------------------ resolve
    def resolve(self, spec: str) -> Optional[Any]:
        """Translate a stored spec into the literal ``think`` field value."""
        key = str(spec or "").strip().lower()
        if key in ("", THINK_AUTO):
            return None
        if not self.supported:
            return None
        if key == THINK_OFF:
            return False if self.can_disable else None
        if key == THINK_ON:
            if True in self.values or not self.levels:
                return True
            if isinstance(self.default, str):
                return self.default
            return self.levels[len(self.levels) // 2]
        for level in self.levels:
            if level.lower() == key:
                # Levels that Ollama's own API refuses are still useful: they
                # are delivered through the prompt instead of the field.
                return level if level.lower() in KNOWN_THINK_LEVELS or self.native else level
        return None

    def intent(self, spec: str) -> tuple[Optional[bool], str]:
        """Interpret *spec* as ``(thinking_enabled, level)``.

        Used by the prompt-level enforcement path, which needs the user's
        intention rather than the wire value.
        """
        key = str(spec or "").strip().lower()
        if key in ("", THINK_AUTO):
            return None, ""
        if key == THINK_OFF:
            return False, ""
        if key == THINK_ON:
            # On a levels-only model "on" means "the level the template would
            # have picked for itself", so the instruction text still matches.
            if self.levels:
                if isinstance(self.default, str) and self.default in self.levels:
                    return True, self.default
                return True, self.levels[0]
            return True, ""
        for level in self.levels:
            if level.lower() == key:
                return True, level
        return None, ""

    def describe(self) -> str:
        if not self.supported:
            return "thinking not supported"
        parts = []
        if self.levels:
            parts.append("levels: " + ", ".join(self.levels))
        if True in self.values:
            parts.append("on/off")
        if self.forced:
            parts.append("always on")
        if self.default is not None:
            parts.append(f"default: {self.default}")
        parts.append(self.source_label())
        if self.needs_prompt_enforcement:
            parts.append("enforced through the prompt")
        return "; ".join(parts)

    # ------------------------------------------------------------- build
    @classmethod
    def from_show(cls, model: str, data: Dict[str, Any]) -> "ThinkingSupport":
        """Build from an ``/api/show`` response body."""
        capabilities = tuple(
            str(item) for item in (data.get("capabilities") or []) if isinstance(item, str)
        )
        template = str(data.get("template") or "")
        controls = parse_template_controls(template)

        # 1. Server-reported metadata wins outright.
        block = data.get("thinking")
        if isinstance(block, dict) and isinstance(block.get("values"), list):
            return cls(
                model=model,
                values=tuple(block["values"]),
                default=block.get("default"),
                capabilities=capabilities,
                known=True,
                source=SOURCE_METADATA,
                controls=controls,
            )

        # 2. The chat template is the next most trustworthy source: it is the
        #    code that will actually run for this exact file.
        if controls.found_anything:
            values: List[Any] = []
            if controls.has_switch or controls.soft_switch:
                values.append(False)
            if controls.levels:
                values.extend(controls.levels)
            else:
                values.append(True)
            return cls(
                model=model,
                values=tuple(values),
                default=controls.default if controls.levels else True,
                capabilities=capabilities or ("thinking",),
                known=False,
                source=SOURCE_TEMPLATE,
                controls=controls,
            )

        # 3. A bare capability flag: on/off, nothing finer.
        if "thinking" in capabilities:
            return cls(
                model=model,
                values=(True, False),
                capabilities=capabilities,
                known=False,
                source=SOURCE_CAPABILITY,
                controls=controls,
            )

        # 4. No evidence of reasoning at all.
        return cls(
            model=model,
            values=(),
            capabilities=capabilities,
            source=SOURCE_NONE,
            controls=controls,
        )


class OllamaError(RuntimeError):
    """Base class for every transport level failure."""


class OllamaUnavailableError(OllamaError):
    """The server could not be reached on any configured endpoint."""


class OllamaCancelled(OllamaError):
    """The caller cancelled the request through its cancel event."""


class OllamaClient:
    """Thread-safe-ish wrapper around the Ollama REST API.

    One client instance is meant to be used by one worker thread at a time;
    the UI creates a dedicated client for the agent worker.
    """

    def __init__(self, settings: Optional[OllamaSettings] = None) -> None:
        self.settings = settings or OllamaSettings()
        self._session = self._build_session()
        self._active_endpoint: Optional[str] = None
        self._lock = threading.Lock()
        self._show_cache: Dict[str, Dict[str, Any]] = {}
        self._thinking_cache: Dict[str, ThinkingSupport] = {}
        #: Reasoning trace produced by the most recent :meth:`chat` call.
        self.last_thinking: str = ""
        #: True when the last call produced reasoning that arrived inline in
        #: the answer (``<think>`` tags) rather than in its own field, which
        #: means the server ignored the ``think`` request field.
        self.last_reasoning_leaked: bool = False
        #: True when the last call asked for silence and reasoned anyway.
        self.last_think_ignored: bool = False

    # ------------------------------------------------------------ plumbing
    def _build_session(self) -> requests.Session:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/x-ndjson, application/json",
                "Connection": "keep-alive",
                "User-Agent": "Agent3/1.0 (+local-coding-agent)",
            }
        )
        if self.settings.extra_headers:
            session.headers.update(self.settings.extra_headers)
        return session

    def update_settings(self, settings: OllamaSettings) -> None:
        """Swap connection settings at runtime (settings dialog)."""
        with self._lock:
            self.settings = settings
            self._active_endpoint = None
            self._show_cache.clear()
            self._thinking_cache.clear()
            try:
                self._session.close()
            except Exception:  # pragma: no cover - defensive
                pass
            self._session = self._build_session()

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def timeout(self) -> tuple[float, float]:
        return (float(self.settings.connect_timeout), float(self.settings.read_timeout))

    @property
    def active_endpoint(self) -> str:
        return self._active_endpoint or self.settings.base_url

    def _endpoints(self) -> List[str]:
        if self._active_endpoint:
            others = [e for e in self.settings.endpoints() if e != self._active_endpoint]
            return [self._active_endpoint, *others]
        return self.settings.endpoints()

    @staticmethod
    def _sleep_backoff(attempt: int, base: float, cancel: Optional[threading.Event]) -> None:
        delay = min(base * (2 ** attempt), 20.0) * (0.75 + random.random() * 0.5)
        logger.debug("retrying in %.2fs (attempt %d)", delay, attempt + 1)
        if cancel is not None:
            if cancel.wait(delay):
                raise OllamaCancelled("request cancelled while backing off")
        else:
            time.sleep(delay)

    def _request(
        self,
        method: str,
        route: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        cancel: Optional[threading.Event] = None,
        timeout: Optional[tuple[float, float]] = None,
    ) -> requests.Response:
        """Perform a request with endpoint fallback and bounded retries."""
        attempts = max(1, int(self.settings.max_retries))
        last_error: Optional[BaseException] = None
        endpoints = self._endpoints()

        for attempt in range(attempts):
            if cancel is not None and cancel.is_set():
                raise OllamaCancelled("request cancelled before dispatch")
            for endpoint in endpoints:
                url = f"{endpoint}{route}"
                try:
                    response = self._session.request(
                        method,
                        url,
                        json=payload,
                        stream=stream,
                        timeout=timeout or self.timeout,
                    )
                except (requests.ConnectionError, requests.Timeout) as exc:
                    last_error = exc
                    logger.debug("transport failure on %s: %s", url, exc)
                    continue
                except requests.RequestException as exc:  # pragma: no cover - rare
                    last_error = exc
                    logger.debug("request error on %s: %s", url, exc)
                    continue

                if response.status_code in RETRYABLE_STATUS:
                    last_error = OllamaError(
                        f"{url} returned HTTP {response.status_code}: {response.text[:300]}"
                    )
                    response.close()
                    continue
                if response.status_code >= 400:
                    detail = response.text[:500]
                    response.close()
                    raise OllamaError(f"HTTP {response.status_code} from {url}: {detail}")

                self._active_endpoint = endpoint
                return response

            if attempt < attempts - 1:
                self._sleep_backoff(attempt, self.settings.retry_backoff, cancel)

        raise OllamaUnavailableError(
            "Ollama is not reachable on "
            + ", ".join(endpoints)
            + f" after {attempts} attempt(s). Last error: {last_error}"
        )

    # --------------------------------------------------------------- health
    def health(self, cancel: Optional[threading.Event] = None) -> Dict[str, Any]:
        """Ping the server and return ``{ok, endpoint, version, models}``."""
        try:
            response = self._request(
                "GET", "/api/tags", cancel=cancel, timeout=(self.settings.connect_timeout, 15.0)
            )
        except OllamaError as exc:
            return {"ok": False, "endpoint": self.settings.base_url, "error": str(exc), "models": []}
        try:
            data = response.json()
        except ValueError:
            data = {}
        finally:
            response.close()
        models = [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]
        return {
            "ok": True,
            "endpoint": self.active_endpoint,
            "models": [m for m in models if m],
            "version": self.version(cancel=cancel),
        }

    def version(self, cancel: Optional[threading.Event] = None) -> str:
        try:
            response = self._request(
                "GET", "/api/version", cancel=cancel, timeout=(self.settings.connect_timeout, 10.0)
            )
        except OllamaError:
            return ""
        try:
            return str(response.json().get("version", ""))
        except ValueError:  # pragma: no cover
            return ""
        finally:
            response.close()

    def list_models(self, cancel: Optional[threading.Event] = None) -> List[str]:
        """Return the model tags installed in the local Ollama instance."""
        return list(self.health(cancel=cancel).get("models", []))

    # ------------------------------------------------------- capabilities
    def show_model(
        self, model: Optional[str] = None, cancel: Optional[threading.Event] = None
    ) -> Dict[str, Any]:
        """Return the raw ``/api/show`` payload for *model* (cached).

        Never raises: an unreachable server simply yields ``{}`` so the UI can
        degrade to "unknown capabilities" instead of blocking.
        """
        name = (model or self.settings.model).strip()
        if not name:
            return {}
        with self._lock:
            cached = self._show_cache.get(name)
        if cached is not None:
            return cached
        try:
            response = self._request(
                "POST",
                "/api/show",
                payload={"model": name},
                cancel=cancel,
                timeout=(self.settings.connect_timeout, 20.0),
            )
        except OllamaError as exc:
            logger.debug("show %s failed: %s", name, exc)
            return {}
        try:
            data = response.json()
        except ValueError:  # pragma: no cover - malformed server
            data = {}
        finally:
            response.close()
        if not isinstance(data, dict):  # pragma: no cover - defensive
            data = {}
        with self._lock:
            self._show_cache[name] = data
        return data

    def thinking_support(
        self, model: Optional[str] = None, cancel: Optional[threading.Event] = None
    ) -> ThinkingSupport:
        """Discover which ``think`` values *model* really accepts.

        The values come from the server, not from a hard-coded table: Ollama
        exposes them in ``/api/show`` precisely so clients stop guessing.
        """
        name = (model or self.settings.model).strip()
        with self._lock:
            cached = self._thinking_cache.get(name)
        if cached is not None:
            return cached
        support = ThinkingSupport.from_show(name, self.show_model(name, cancel=cancel))
        with self._lock:
            self._thinking_cache[name] = support
        return support

    def model_capabilities(
        self, model: Optional[str] = None, cancel: Optional[threading.Event] = None
    ) -> List[str]:
        """``["completion", "tools", "thinking", ...]`` as reported by Ollama."""
        data = self.show_model(model, cancel=cancel)
        return [str(item) for item in (data.get("capabilities") or [])]

    def supports_tools(self, model: Optional[str] = None) -> bool:
        return "tools" in self.model_capabilities(model)

    def clear_capability_cache(self, model: Optional[str] = None) -> None:
        """Drop cached ``/api/show`` data (after a pull or a host change)."""
        with self._lock:
            if model is None:
                self._show_cache.clear()
                self._thinking_cache.clear()
            else:
                self._show_cache.pop(model, None)
                self._thinking_cache.pop(model, None)

    def resolve_think(self, spec: str, model: Optional[str] = None) -> Optional[Any]:
        """Translate a stored think spec into the literal request value."""
        return self.thinking_support(model).resolve(spec)

    def resolve_model(self, preferred: Optional[str] = None) -> str:
        """Pick *preferred* when installed, else the first configured fallback.

        Never raises: a missing model is reported by the chat call itself, and
        the UI shows a banner - the agent must stay usable offline.
        """
        target = preferred or self.settings.model
        try:
            installed = self.list_models()
        except OllamaError:  # pragma: no cover - health already swallows
            return target
        if not installed:
            return target
        names = {name.split(":")[0]: name for name in installed}
        if target in installed or target.split(":")[0] in names:
            return target
        for candidate in self.settings.fallback_models:
            if candidate in installed or candidate.split(":")[0] in names:
                logger.warning("model %r not installed - falling back to %r", target, candidate)
                return candidate
        return target

    # ------------------------------------------------------- reasoning plan
    def think_plan(
        self, spec: Optional[str], model: Optional[str] = None
    ) -> tuple[Optional[Any], PromptControls, ThinkingSupport]:
        """Work out how to deliver a reasoning setting for *model*.

        Returns the literal ``think`` request value (or ``None`` to omit the
        field), the prompt level instructions needed to make the setting
        stick, and the support record the decision was based on.

        The prompt level part exists because Ollama does not forward the
        ``think`` field into the chat template for every model: a template
        that reads ``enable_thinking``/``reasoning_effort`` never sees it and
        falls back to its own default, which is usually "reason, at full
        effort". Restating the instruction in the conversation is the only
        transport-independent way to be obeyed.
        """
        raw = self.settings.think if spec is None else spec
        key = str(raw or "").strip().lower()
        if key in ("", THINK_AUTO):
            # "Auto" means "say nothing at all", so there is no reason to
            # query the server about a model we are not going to configure.
            return None, PromptControls(), ThinkingSupport(model=model or "")

        support = self.thinking_support(model)
        wire = support.resolve(key)
        controls = PromptControls()
        if self.settings.enforce_think_in_prompt and support.needs_prompt_enforcement:
            enabled, level = support.intent(key)
            controls = prompt_controls(
                enabled=enabled,
                level=level,
                controls=support.controls,
                native=support.native,
            )
        return wire, controls, support

    @staticmethod
    def _apply_prompt_controls(
        messages: Sequence[ChatMessage], controls: PromptControls
    ) -> List[ChatMessage]:
        """Return *messages* with the reasoning instructions woven in.

        The effort sentence goes in front of the first system message - the
        same position the chat template would have rendered it - and the soft
        switch is appended to the final user turn, which is where Qwen style
        templates look for it.
        """
        prepared = list(messages)
        if not controls:
            return prepared

        if controls.system:
            for index, message in enumerate(prepared):
                if message.role == Role.SYSTEM:
                    prepared[index] = ChatMessage(
                        role=Role.SYSTEM,
                        content=f"{controls.system}\n\n{message.content}",
                        name=message.name,
                    )
                    break
            else:
                prepared.insert(0, ChatMessage(role=Role.SYSTEM, content=controls.system))

        if controls.suffix:
            for index in range(len(prepared) - 1, -1, -1):
                message = prepared[index]
                if message.role == Role.USER:
                    if controls.suffix not in message.content:
                        prepared[index] = ChatMessage(
                            role=Role.USER,
                            content=f"{message.content}\n\n{controls.suffix}",
                            name=message.name,
                        )
                    break
        return prepared

    # ----------------------------------------------------------------- chat
    def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        cancel: Optional[threading.Event] = None,
        format_json: bool = False,
        think: Optional[str] = None,
    ) -> Iterator[StreamEvent]:
        """Yield :class:`StreamEvent` objects for a streamed chat completion.

        *think* is a spec (``"auto"``/``"off"``/``"on"`` or a level name such
        as ``"high"``); it is validated against the model's real capabilities
        before being put on the wire, so an unsupported value is dropped
        instead of making the server reject the whole request.
        """
        model_name = model or self.settings.model
        think_value, controls, support = self.think_plan(think, model_name)
        prepared = self._apply_prompt_controls(messages, controls)
        wanted_silence = think_value is False or controls.suffix == SOFT_SWITCH_OFF
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [m.to_payload() for m in prepared],
            "stream": True,
            "keep_alive": self.settings.keep_alive,
            "options": {**self.settings.options(), **(options or {})},
        }
        if format_json:
            payload["format"] = "json"
        if think_value is not None:
            payload["think"] = think_value

        # Models whose server does not split the trace emit it inline in the
        # answer; the filter routes it to the reasoning channel so the tool
        # parser only ever sees real output.
        strip_inline = bool(self.settings.strip_inline_reasoning) and not format_json
        inline = ReasoningFilter() if strip_inline else None

        prompt_tokens_estimate = count_message_tokens(prepared)
        started = time.monotonic()
        response = self._request("POST", "/api/chat", payload=payload, stream=True, cancel=cancel)

        collected: List[str] = []
        thoughts: List[str] = []
        final_seen = False
        try:
            for line in response.iter_lines(decode_unicode=True):
                if cancel is not None and cancel.is_set():
                    raise OllamaCancelled("generation cancelled by user")
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("skipping non-JSON stream line: %.120s", line)
                    continue
                if "error" in chunk:
                    raise OllamaError(str(chunk["error"]))

                message = chunk.get("message") or {}
                delta = str(message.get("content", "") or "")
                reasoning = str(message.get("thinking", "") or "")
                if delta and inline is not None:
                    delta, extracted = inline.feed(delta)
                    if extracted:
                        reasoning += extracted
                        self.last_reasoning_leaked = True
                if delta:
                    collected.append(delta)
                if reasoning:
                    thoughts.append(reasoning)
                    if wanted_silence:
                        self.last_think_ignored = True
                done = bool(chunk.get("done"))
                usage: Optional[Usage] = None
                if done:
                    final_seen = True
                    if inline is not None:
                        tail, tail_reasoning = inline.flush()
                        if tail:
                            delta += tail
                            collected.append(tail)
                        if tail_reasoning:
                            reasoning += tail_reasoning
                            thoughts.append(tail_reasoning)
                    usage = self._usage_from_chunk(
                        chunk,
                        fallback_prompt=prompt_tokens_estimate,
                        fallback_completion="".join(collected),
                        elapsed_ms=int((time.monotonic() - started) * 1000),
                    )
                yield StreamEvent(
                    delta=delta,
                    done=done,
                    usage=usage,
                    model=str(chunk.get("model", model_name)),
                    raw=chunk,
                    thinking=reasoning,
                )
        finally:
            response.close()

        if not final_seen:
            # Server closed the stream without the terminating object.
            yield StreamEvent(
                delta="",
                done=True,
                usage=Usage(
                    prompt_tokens=prompt_tokens_estimate,
                    completion_tokens=estimate_tokens("".join(collected)),
                    total_duration_ms=int((time.monotonic() - started) * 1000),
                    estimated=True,
                ),
                model=model_name,
            )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        cancel: Optional[threading.Event] = None,
        on_delta: Optional[Any] = None,
        format_json: bool = False,
        think: Optional[str] = None,
        on_thinking: Optional[Any] = None,
    ) -> tuple[str, Usage]:
        """Run a completion and return ``(full_text, usage)``.

        When *on_delta* is given it is invoked with each text fragment, which
        is how the UI renders tokens as they arrive.  *on_thinking* receives
        the reasoning trace separately - it is deliberately kept out of the
        returned text so the tool-call parser never sees the model's
        scratchpad. The complete trace is also available afterwards through
        :attr:`last_thinking`. Honours the configured ``stream`` flag: non
        streaming mode performs a single blocking POST.
        """
        self.last_thinking = ""
        self.last_reasoning_leaked = False
        self.last_think_ignored = False
        if self.settings.stream:
            parts: List[str] = []
            thoughts: List[str] = []
            usage = Usage()
            for event in self.chat_stream(
                messages,
                model=model,
                options=options,
                cancel=cancel,
                format_json=format_json,
                think=think,
            ):
                if event.thinking:
                    thoughts.append(event.thinking)
                    if on_thinking is not None:
                        on_thinking(event.thinking)
                if event.delta:
                    parts.append(event.delta)
                    if on_delta is not None:
                        on_delta(event.delta)
                if event.done and event.usage is not None:
                    usage = event.usage
            self.last_thinking = "".join(thoughts)
            return "".join(parts), usage

        model_name = model or self.settings.model
        think_value, controls, support = self.think_plan(think, model_name)
        prepared = self._apply_prompt_controls(messages, controls)
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [m.to_payload() for m in prepared],
            "stream": False,
            "keep_alive": self.settings.keep_alive,
            "options": {**self.settings.options(), **(options or {})},
        }
        if format_json:
            payload["format"] = "json"
        if think_value is not None:
            payload["think"] = think_value
        started = time.monotonic()
        response = self._request("POST", "/api/chat", payload=payload, cancel=cancel)
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError(f"invalid JSON response from Ollama: {exc}") from exc
        finally:
            response.close()
        if "error" in data:
            raise OllamaError(str(data["error"]))
        message = data.get("message") or {}
        text = str(message.get("content", "") or "")
        self.last_thinking = str(message.get("thinking", "") or "")
        if self.settings.strip_inline_reasoning and not format_json:
            text, extracted = split_inline_reasoning(text)
            if extracted:
                self.last_reasoning_leaked = True
                self.last_thinking = f"{self.last_thinking}{extracted}"
        if self.last_thinking and think_value is False:
            self.last_think_ignored = True
        if on_thinking is not None and self.last_thinking:
            on_thinking(self.last_thinking)
        if on_delta is not None and text:
            on_delta(text)
        usage = self._usage_from_chunk(
            data,
            fallback_prompt=count_message_tokens(prepared),
            fallback_completion=text,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        return text, usage

    # ---------------------------------------------------------------- utils
    def _think_value(self, spec: Optional[str], model: str) -> Optional[Any]:
        """Resolve the ``think`` request field, or ``None`` to omit it."""
        try:
            return self.think_plan(spec, model)[0]
        except OllamaError:  # pragma: no cover - show() already swallows
            return None

    @staticmethod
    def _usage_from_chunk(
        chunk: Dict[str, Any], *, fallback_prompt: int, fallback_completion: str, elapsed_ms: int
    ) -> Usage:
        """Prefer server reported counters, fall back to the estimator."""
        prompt = chunk.get("prompt_eval_count")
        completion = chunk.get("eval_count")
        estimated = prompt is None or completion is None
        total_duration = chunk.get("total_duration")
        duration_ms = int(total_duration / 1_000_000) if isinstance(total_duration, (int, float)) else elapsed_ms
        return Usage(
            prompt_tokens=int(prompt) if isinstance(prompt, (int, float)) else int(fallback_prompt),
            completion_tokens=(
                int(completion)
                if isinstance(completion, (int, float))
                else estimate_tokens(fallback_completion)
            ),
            total_duration_ms=duration_ms,
            estimated=estimated,
        )

    def pull_model(
        self, name: str, cancel: Optional[threading.Event] = None
    ) -> Iterable[Dict[str, Any]]:
        """Stream progress dictionaries while Ollama downloads *name*."""
        response = self._request(
            "POST", "/api/pull", payload={"model": name, "stream": True}, stream=True, cancel=cancel
        )
        try:
            for line in response.iter_lines(decode_unicode=True):
                if cancel is not None and cancel.is_set():
                    raise OllamaCancelled("pull cancelled")
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:  # pragma: no cover
                    continue
        finally:
            response.close()
