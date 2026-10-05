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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import requests
from requests.adapters import HTTPAdapter

from agent3.core.config import OllamaSettings
from agent3.core.logging_setup import get_logger
from agent3.llm.messages import ChatMessage, StreamEvent, Usage
from agent3.llm.tokenizer import count_message_tokens, estimate_tokens

logger = get_logger(__name__)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


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

    # ----------------------------------------------------------------- chat
    def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        cancel: Optional[threading.Event] = None,
        format_json: bool = False,
    ) -> Iterator[StreamEvent]:
        """Yield :class:`StreamEvent` objects for a streamed chat completion."""
        model_name = model or self.settings.model
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [m.to_payload() for m in messages],
            "stream": True,
            "keep_alive": self.settings.keep_alive,
            "options": {**self.settings.options(), **(options or {})},
        }
        if format_json:
            payload["format"] = "json"

        prompt_tokens_estimate = count_message_tokens(messages)
        started = time.monotonic()
        response = self._request("POST", "/api/chat", payload=payload, stream=True, cancel=cancel)

        collected: List[str] = []
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

                delta = str((chunk.get("message") or {}).get("content", "") or "")
                if delta:
                    collected.append(delta)
                done = bool(chunk.get("done"))
                usage: Optional[Usage] = None
                if done:
                    final_seen = True
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
    ) -> tuple[str, Usage]:
        """Run a completion and return ``(full_text, usage)``.

        When *on_delta* is given it is invoked with each text fragment, which
        is how the UI renders tokens as they arrive.  Honours the configured
        ``stream`` flag: non streaming mode performs a single blocking POST.
        """
        if self.settings.stream:
            parts: List[str] = []
            usage = Usage()
            for event in self.chat_stream(
                messages, model=model, options=options, cancel=cancel, format_json=format_json
            ):
                if event.delta:
                    parts.append(event.delta)
                    if on_delta is not None:
                        on_delta(event.delta)
                if event.done and event.usage is not None:
                    usage = event.usage
            return "".join(parts), usage

        model_name = model or self.settings.model
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [m.to_payload() for m in messages],
            "stream": False,
            "keep_alive": self.settings.keep_alive,
            "options": {**self.settings.options(), **(options or {})},
        }
        if format_json:
            payload["format"] = "json"
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
        text = str((data.get("message") or {}).get("content", "") or "")
        if on_delta is not None and text:
            on_delta(text)
        usage = self._usage_from_chunk(
            data,
            fallback_prompt=count_message_tokens(messages),
            fallback_completion=text,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        return text, usage

    # ---------------------------------------------------------------- utils
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
