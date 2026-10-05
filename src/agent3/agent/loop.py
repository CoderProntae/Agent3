"""The autonomous reason -> act -> observe loop with self-correction.

The loop is pure Python and completely UI agnostic: every piece of progress is
reported through :class:`AgentCallbacks`, which the Qt layer binds to signals.
That also makes the whole thing unit-testable with a fake LLM client.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence

from agent3.agent.prompts import (
    FAILED_VERIFY_HINT,
    LOOP_HINT,
    NO_TOOL_HINT,
    RETRY_HINT,
    VERIFY_HINT,
    build_observation,
    build_system_prompt,
    build_user_request,
)
from agent3.agent.tools import ToolCall, ToolContext, ToolRegistry, ToolResult, parse_tool_calls, strip_tool_calls
from agent3.core.config import AgentSettings
from agent3.core.logging_setup import get_logger, log_exception
from agent3.limits.manager import UsageManager
from agent3.llm.messages import ChatMessage, Role
from agent3.llm.ollama_client import OllamaCancelled, OllamaClient, OllamaError
from agent3.llm.tokenizer import count_message_tokens, truncate_to_tokens

logger = get_logger(__name__)


class AgentStopReason(str, Enum):
    """Why a run ended."""

    FINISHED = "finished"
    NO_TOOL_CALL = "no_tool_call"
    MAX_ITERATIONS = "max_iterations"
    QUOTA_BLOCKED = "quota_blocked"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class AgentCallbacks:
    """Observer hooks; every field is optional."""

    on_status: Optional[Callable[[str], None]] = None
    on_assistant_delta: Optional[Callable[[str], None]] = None
    on_assistant_message: Optional[Callable[[str], None]] = None
    on_tool_start: Optional[Callable[[ToolCall], None]] = None
    on_tool_result: Optional[Callable[[ToolCall, ToolResult], None]] = None
    on_usage: Optional[Callable[[Dict[str, int]], None]] = None
    on_error: Optional[Callable[[str], None]] = None

    def emit(self, hook: str, *args) -> None:
        """Invoke a hook, swallowing UI exceptions so the loop never dies."""
        callback = getattr(self, hook, None)
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:  # pragma: no cover - defensive around UI code
            logger.debug("callback %s raised", hook, exc_info=True)


@dataclass
class AgentRunResult:
    """Everything that happened during one run."""

    stop_reason: AgentStopReason
    final_message: str = ""
    iterations: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0
    error: str = ""
    transcript: List[ChatMessage] = field(default_factory=list)
    #: Workspace-relative paths written during the run.
    changed_files: List[str] = field(default_factory=list)
    #: True when a command succeeded after the final file modification.
    verified: bool = False
    #: The command used as proof, when there is one.
    verification_command: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def ok(self) -> bool:
        return self.stop_reason in (AgentStopReason.FINISHED, AgentStopReason.NO_TOOL_CALL)


class AgentLoop:
    """Drives the model, the tools and the quota engine."""

    def __init__(
        self,
        client: OllamaClient,
        registry: ToolRegistry,
        context: ToolContext,
        usage: UsageManager,
        settings: Optional[AgentSettings] = None,
        *,
        model: Optional[str] = None,
        callbacks: Optional[AgentCallbacks] = None,
        extra_instructions: str = "",
    ) -> None:
        self.client = client
        self.registry = registry
        self.context = context
        self.usage = usage
        self.settings = settings or AgentSettings()
        self.model = model or client.settings.model
        self.callbacks = callbacks or AgentCallbacks()
        self.extra_instructions = extra_instructions
        self.history: List[ChatMessage] = []
        self._cancel = threading.Event()

    # ------------------------------------------------------------- control
    def cancel(self) -> None:
        """Request cooperative cancellation of the current run."""
        self._cancel.set()
        try:
            self.context.runner.terminate()
        except Exception:  # pragma: no cover
            logger.debug("terminating runner failed", exc_info=True)

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel

    def reset(self) -> None:
        """Forget the conversation (new session)."""
        self.history.clear()
        self._cancel.clear()

    def load_history(self, messages: Sequence[ChatMessage]) -> None:
        """Restore a persisted conversation."""
        self.history = [m for m in messages if m.role is not Role.SYSTEM]

    # -------------------------------------------------------------- prompt
    def system_prompt(self) -> str:
        fs = self.context.fs
        try:
            tree = fs.tree_text(max_depth=3, max_entries=300)
        except Exception:  # pragma: no cover - unreadable workspace
            tree = "(workspace tree unavailable)"
        git_state = "not a git repository"
        try:
            if self.context.git.is_repo():
                branch = self.context.git.current_branch() or "detached"
                dirty = "dirty" if self.context.git.is_dirty() else "clean"
                git_state = f"yes (branch: {branch}, {dirty})"
        except Exception:  # pragma: no cover
            pass
        return build_system_prompt(
            workspace=str(fs.root),
            tools=self.registry.describe(),
            tree=tree,
            git_state=git_state,
            extra_instructions=self.extra_instructions,
        )

    def _window(self) -> List[ChatMessage]:
        """System prompt + a bounded slice of the conversation."""
        window = max(6, int(self.settings.history_window))
        tail = self.history[-window:]
        return [ChatMessage.system(self.system_prompt()), *tail]

    # ----------------------------------------------------------------- run
    def run(self, user_message: str, *, context_hint: str = "") -> AgentRunResult:
        """Execute one autonomous run for *user_message*."""
        self._cancel.clear()
        started = time.monotonic()
        result = AgentRunResult(stop_reason=AgentStopReason.ERROR)
        self.usage.begin_run()

        self.history.append(
            ChatMessage.user(build_user_request(user_message, context=context_hint or None))
        )

        max_iterations = max(1, int(self.settings.max_iterations))
        repeated: Dict[str, int] = {}
        consecutive_failures = 0

        # --- "definition of done" bookkeeping -----------------------------
        #: Tools whose success leaves the workspace in an unverified state.
        mutating = {"write_file", "edit_file", "patch_file", "delete_file", "rename_file"}
        changed: List[str] = []
        unverified = False          # files changed, nothing run since
        verified_with = ""          # the command that proved the work
        last_failed_command = ""
        last_failed_exit = 0
        nudges_left = max(0, int(getattr(self.settings, "verification_nudges", 1)))
        require_verification = bool(getattr(self.settings, "require_verification", True))

        try:
            for iteration in range(1, max_iterations + 1):
                if self._cancel.is_set():
                    result.stop_reason = AgentStopReason.CANCELLED
                    break
                result.iterations = iteration

                messages = self._window()
                decision = self.usage.check_request(count_message_tokens(messages))
                if not decision.allowed:
                    if decision.code == "cooldown" and decision.retry_after_seconds > 0:
                        self.callbacks.emit("on_status", f"Rate limited - waiting {decision.retry_after_seconds:.1f}s")
                        if self._cancel.wait(decision.retry_after_seconds):
                            result.stop_reason = AgentStopReason.CANCELLED
                            break
                        decision = self.usage.check_request(count_message_tokens(messages))
                    if not decision.allowed:
                        result.stop_reason = AgentStopReason.QUOTA_BLOCKED
                        result.error = decision.reason
                        self.callbacks.emit("on_error", decision.reason)
                        break
                if decision.reason:
                    self.callbacks.emit("on_status", decision.reason)

                self.callbacks.emit("on_status", f"Thinking (step {iteration}/{max_iterations})")
                try:
                    text, usage = self.client.chat(
                        messages,
                        model=self.model,
                        cancel=self._cancel,
                        on_delta=lambda delta: self.callbacks.emit("on_assistant_delta", delta),
                    )
                except OllamaCancelled:
                    result.stop_reason = AgentStopReason.CANCELLED
                    break
                except OllamaError as exc:
                    trace = log_exception(logger, exc, "LLM call failed")
                    self.usage.record_error(str(exc), where="llm")
                    consecutive_failures += 1
                    self.callbacks.emit("on_error", f"Ollama error: {exc}")
                    if consecutive_failures > max(1, int(self.settings.self_correction_retries)):
                        result.stop_reason = AgentStopReason.ERROR
                        result.error = str(exc)
                        break
                    self.history.append(
                        ChatMessage.user(
                            f"The inference server returned an error:\n{trace[-800:]}\n\n{RETRY_HINT}"
                        )
                    )
                    continue

                result.prompt_tokens += usage.prompt_tokens
                result.completion_tokens += usage.completion_tokens
                self.usage.record_request(
                    model=self.model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    duration_ms=usage.total_duration_ms,
                    meta={"iteration": iteration},
                )
                self.callbacks.emit(
                    "on_usage",
                    {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                )

                self.history.append(ChatMessage.assistant(text))
                prose = strip_tool_calls(text)
                if prose:
                    self.callbacks.emit("on_assistant_message", prose)

                calls = parse_tool_calls(text, self.registry.names())
                if not calls:
                    # No action: treat a non-trivial answer as the final reply.
                    if prose and iteration > 1:
                        result.stop_reason = AgentStopReason.NO_TOOL_CALL
                        result.final_message = prose
                        break
                    self.history.append(ChatMessage.user(NO_TOOL_HINT))
                    consecutive_failures += 1
                    if consecutive_failures > max(2, int(self.settings.self_correction_retries) + 1):
                        result.stop_reason = AgentStopReason.NO_TOOL_CALL
                        result.final_message = prose or text.strip()
                        break
                    continue

                call = calls[0]
                if call.name == "finish":
                    summary = str(call.args.get("summary") or call.args.get("message") or prose or "Task complete.")

                    # Definition of done: code that was never executed is not
                    # finished work. Push the model back to its test command
                    # instead of accepting an optimistic summary.
                    if require_verification and unverified and nudges_left > 0:
                        nudges_left -= 1
                        hint = (
                            FAILED_VERIFY_HINT.format(
                                command=last_failed_command, exit_code=last_failed_exit
                            )
                            if last_failed_command
                            else VERIFY_HINT.format(changed=len(set(changed)))
                        )
                        self.callbacks.emit("on_tool_start", call)
                        self.callbacks.emit(
                            "on_tool_result",
                            call,
                            ToolResult(
                                False,
                                error="finish refused: the changes have not been verified yet",
                                title="finish (blocked)",
                            ),
                        )
                        self.callbacks.emit("on_status", "Verification required before finishing")
                        self.history.append(ChatMessage.user(hint))
                        consecutive_failures = 0
                        continue

                    if require_verification and unverified:
                        summary += (
                            "\n\n> Warning: Agent3 could not confirm these changes - "
                            "no verification command succeeded after the last edit."
                        )
                    result.stop_reason = AgentStopReason.FINISHED
                    result.final_message = summary
                    self.callbacks.emit("on_tool_start", call)
                    self.callbacks.emit("on_tool_result", call, ToolResult(True, output=summary, title="finish"))
                    break

                signature = call.signature()
                repeated[signature] = repeated.get(signature, 0) + 1

                quota = self.usage.check_tool_call()
                if not quota.allowed:
                    result.stop_reason = AgentStopReason.QUOTA_BLOCKED
                    result.error = quota.reason
                    self.callbacks.emit("on_error", quota.reason)
                    break

                self.callbacks.emit("on_status", f"Running tool: {call.name}")
                self.callbacks.emit("on_tool_start", call)
                tool_result = self.registry.execute(call, self.context)
                result.tool_calls += 1
                self.usage.record_tool_call(
                    call.name, duration_ms=tool_result.duration_ms, ok=tool_result.ok
                )
                self.callbacks.emit("on_tool_result", call, tool_result)

                if tool_result.ok:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    self.usage.record_error(tool_result.error, where=f"tool:{call.name}")

                # Track what still needs proving.
                if tool_result.ok and call.name in mutating:
                    touched = str(tool_result.data.get("path") or call.args.get("path") or "")
                    if touched:
                        changed.append(touched)
                    unverified = True
                    verified_with = ""
                elif call.name == "run_command":
                    if tool_result.ok:
                        unverified = False
                        verified_with = str(call.args.get("command") or "")
                        last_failed_command = ""
                        last_failed_exit = 0
                    else:
                        last_failed_command = str(call.args.get("command") or "")
                        last_failed_exit = int(tool_result.data.get("exit_code") or 1)

                observation = truncate_to_tokens(
                    tool_result.observation(self.settings.max_output_chars), 6000
                )
                follow_up = build_observation(
                    call.name, observation, iteration=iteration, max_iterations=max_iterations
                )
                if not tool_result.ok:
                    follow_up += f"\n\n{RETRY_HINT}"
                if repeated[signature] >= 3:
                    follow_up += f"\n\n{LOOP_HINT}"
                self.history.append(ChatMessage.user(follow_up))
            else:
                result.stop_reason = AgentStopReason.MAX_ITERATIONS
                result.error = f"stopped after {max_iterations} steps without calling finish"

        except Exception as exc:  # pragma: no cover - safety net
            trace = log_exception(logger, exc, "agent loop crashed")
            result.stop_reason = AgentStopReason.ERROR
            result.error = trace[-2000:]
            self.usage.record_error(str(exc), where="loop")
            self.callbacks.emit("on_error", str(exc))

        result.duration_ms = int((time.monotonic() - started) * 1000)
        result.transcript = list(self.history)
        result.changed_files = sorted(set(changed))
        result.verified = bool(changed) and not unverified
        result.verification_command = verified_with
        if not result.final_message:
            result.final_message = self._fallback_message(result)
        self.callbacks.emit("on_status", f"Done ({result.stop_reason.value})")
        logger.info(
            "run finished: %s in %d steps / %d tool calls / %d tokens",
            result.stop_reason.value,
            result.iterations,
            result.tool_calls,
            result.total_tokens,
        )
        return result

    @staticmethod
    def _fallback_message(result: AgentRunResult) -> str:
        mapping = {
            AgentStopReason.CANCELLED: "Run cancelled by the user.",
            AgentStopReason.QUOTA_BLOCKED: f"Blocked by usage policy: {result.error}",
            AgentStopReason.MAX_ITERATIONS: (
                "Reached the maximum number of steps before finishing. "
                "Review the actions above and send a follow-up instruction."
            ),
            AgentStopReason.ERROR: f"The run failed: {result.error}",
        }
        return mapping.get(result.stop_reason, "")
