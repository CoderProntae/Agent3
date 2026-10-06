"""Background QThread workers so the GUI never blocks.

Two workers exist:

``AgentWorker``
    Runs one :class:`~agent3.agent.loop.AgentLoop` turn and re-emits every
    callback as a Qt signal (signals are thread-safe, direct UI calls are not).
``CommandWorker``
    Runs a single shell command for the embedded terminal.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QThread, Signal

from agent3.agent.loop import AgentCallbacks, AgentLoop, AgentRunResult
from agent3.agent.tools import ToolCall, ToolResult
from agent3.core.logging_setup import get_logger, log_exception
from agent3.workspace.terminal import CommandResult, CommandRunner

logger = get_logger(__name__)


class AgentWorker(QThread):
    """Executes one agent run off the UI thread."""

    status_changed = Signal(str)
    assistant_delta = Signal(str)
    thinking_delta = Signal(str)
    assistant_message = Signal(str)
    tasks_changed = Signal(object)          # TaskList
    processes_changed = Signal()
    tool_started = Signal(object)          # ToolCall
    tool_finished = Signal(object, object)  # ToolCall, ToolResult
    command_output = Signal(str, str)       # stream, text
    file_changed = Signal(str, str)         # path, unified diff
    error_raised = Signal(str)
    finished_run = Signal(object)           # AgentRunResult

    def __init__(self, loop: AgentLoop, message: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._loop = loop
        self._message = message
        self._result: Optional[AgentRunResult] = None

    @property
    def result(self) -> Optional[AgentRunResult]:
        return self._result

    def cancel(self) -> None:
        """Ask the loop to stop at the next safe point."""
        self._loop.cancel()

    def run(self) -> None:  # noqa: D102 - QThread entry point
        self._loop.callbacks = AgentCallbacks(
            on_status=self.status_changed.emit,
            on_assistant_delta=self.assistant_delta.emit,
            on_thinking_delta=self.thinking_delta.emit,
            on_assistant_message=self.assistant_message.emit,
            on_tasks_changed=self.tasks_changed.emit,
            on_tool_start=self._emit_tool_start,
            on_tool_result=self._emit_tool_result,
            on_error=self.error_raised.emit,
        )
        self._loop.context.on_command_output = self.command_output.emit
        self._loop.context.on_file_changed = self.file_changed.emit
        self._loop.context.on_tasks_changed = self.tasks_changed.emit
        self._loop.context.on_processes_changed = self.processes_changed.emit
        try:
            self._result = self._loop.run(self._message)
        except Exception as exc:  # pragma: no cover - the loop already guards
            trace = log_exception(logger, exc, "agent worker crashed")
            self.error_raised.emit(trace[-1500:])
            from agent3.agent.loop import AgentStopReason  # local import avoids a cycle

            self._result = AgentRunResult(stop_reason=AgentStopReason.ERROR, error=str(exc))
        finally:
            self._loop.context.on_command_output = None
            self._loop.context.on_file_changed = None
            self._loop.context.on_processes_changed = None
        self.finished_run.emit(self._result)

    def _emit_tool_start(self, call: ToolCall) -> None:
        self.tool_started.emit(call)

    def _emit_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        self.tool_finished.emit(call, result)


class CommandWorker(QThread):
    """Runs one shell command for the embedded terminal."""

    output = Signal(str, str)       # stream, text
    completed = Signal(object)      # CommandResult

    def __init__(
        self,
        runner: CommandRunner,
        command: str,
        *,
        timeout: Optional[float] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._command = command
        self._timeout = timeout

    def cancel(self) -> None:
        self._runner.terminate()

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            result = self._runner.run(
                self._command, timeout=self._timeout, on_output=self.output.emit
            )
        except Exception as exc:  # pragma: no cover - runner is defensive already
            trace = log_exception(logger, exc, "command worker crashed")
            result = CommandResult(
                command=self._command, cwd=str(self._runner.cwd), exit_code=1, stderr=trace, error=str(exc)
            )
        self.completed.emit(result)


class ModelCapabilityWorker(QThread):
    """Asks Ollama what a model can do, off the UI thread.

    ``/api/show`` is a real HTTP round trip and the answer decides which
    reasoning controls the composer strip may offer, so it must never run on
    the GUI thread: a cold model listing can take seconds.
    """

    completed = Signal(str, object)  # model, ThinkingSupport

    def __init__(self, client, model: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._client = client
        self._model = model

    def run(self) -> None:  # noqa: D102
        from agent3.llm.ollama_client import ThinkingSupport

        try:
            support = self._client.thinking_support(self._model)
        except Exception as exc:  # pragma: no cover - the client swallows already
            logger.debug("capability probe for %s failed: %s", self._model, exc)
            support = ThinkingSupport(model=self._model)
        self.completed.emit(self._model, support)


class HealthWorker(QThread):
    """Pings the Ollama server without freezing the settings dialog."""

    completed = Signal(dict)

    def __init__(self, client, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._client = client

    def run(self) -> None:  # noqa: D102
        payload: Dict[str, Any]
        try:
            payload = self._client.health()
        except Exception as exc:  # pragma: no cover - health swallows already
            payload = {"ok": False, "error": str(exc), "models": []}
        self.completed.emit(payload)
