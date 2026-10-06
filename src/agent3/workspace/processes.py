"""Long-lived background processes (dev servers, watchers, test runners).

``run_command`` is synchronous: it waits for the command to exit, which makes
it useless for ``npm run dev`` or ``uvicorn app:app --reload``. The agent needs
to *start* a server, keep working while it runs, read its log to see whether it
came up, and shut it down when it is done.

:class:`ProcessManager` provides exactly that triple. Every process:

* inherits the non-interactive environment and the deny-list of
  :class:`~agent3.workspace.terminal.CommandRunner` (the same safety rules);
* is spawned in its own session/process group so the whole tree can be killed;
* streams stdout and stderr into a bounded ring buffer, so a chatty server
  cannot exhaust memory;
* is terminated when the manager shuts down - no orphan servers survive
  closing the workspace or the application.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence

from agent3.core.logging_setup import get_logger
from agent3.workspace.terminal import (
    INTERACTIVE_HINT,
    CommandBlockedError,
    CommandRunner,
)

logger = get_logger(__name__)

#: Lines of stdout/stderr kept per process.
DEFAULT_LOG_LINES = 2000

#: Hard ceiling on concurrently running background processes.
DEFAULT_MAX_PROCESSES = 8


class ProcessError(RuntimeError):
    """Raised for an unknown process id or a refused start request."""


@dataclass
class LogLine:
    """One captured output line."""

    stream: str  # stdout | stderr
    text: str
    at: float = field(default_factory=time.time)


@dataclass
class ProcessInfo:
    """Serialisable snapshot of a background process."""

    process_id: str
    command: str
    cwd: str
    pid: int
    running: bool
    exit_code: Optional[int]
    started_at: float
    stopped_at: Optional[float]
    log_lines: int

    @property
    def uptime_seconds(self) -> float:
        end = self.stopped_at if self.stopped_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def to_dict(self) -> Dict[str, object]:
        return {
            "process_id": self.process_id,
            "command": self.command,
            "cwd": self.cwd,
            "pid": self.pid,
            "running": self.running,
            "exit_code": self.exit_code,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "log_lines": self.log_lines,
        }

    def summary(self) -> str:
        state = "running" if self.running else f"exited ({self.exit_code})"
        return (
            f"{self.process_id}  {state}  pid={self.pid}  "
            f"up={self.uptime_seconds:.0f}s  logs={self.log_lines}\n"
            f"  $ {self.command}"
        )


class BackgroundProcess:
    """A single supervised child process."""

    def __init__(
        self,
        process_id: str,
        command: str,
        cwd: Path,
        popen: subprocess.Popen,
        *,
        max_log_lines: int = DEFAULT_LOG_LINES,
    ) -> None:
        self.process_id = process_id
        self.command = command
        self.cwd = cwd
        self._popen = popen
        self._log: Deque[LogLine] = deque(maxlen=max_log_lines)
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.stopped_at: Optional[float] = None
        self._threads = [
            threading.Thread(target=self._pump, args=(popen.stdout, "stdout"), daemon=True),
            threading.Thread(target=self._pump, args=(popen.stderr, "stderr"), daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    # ------------------------------------------------------------- capture
    def _pump(self, stream, name: str) -> None:
        if stream is None:  # pragma: no cover - defensive
            return
        try:
            for raw in iter(stream.readline, b""):
                text = raw.decode("utf-8", errors="replace").rstrip("\n")
                with self._lock:
                    self._log.append(LogLine(name, text))
        except (ValueError, OSError):  # pragma: no cover - stream closed
            pass
        finally:
            try:
                stream.close()
            except Exception:  # pragma: no cover
                pass

    # -------------------------------------------------------------- state
    @property
    def pid(self) -> int:
        return self._popen.pid

    def poll(self) -> Optional[int]:
        code = self._popen.poll()
        if code is not None and self.stopped_at is None:
            self.stopped_at = time.time()
        return code

    @property
    def running(self) -> bool:
        return self.poll() is None

    def info(self) -> ProcessInfo:
        with self._lock:
            lines = len(self._log)
        return ProcessInfo(
            process_id=self.process_id,
            command=self.command,
            cwd=str(self.cwd),
            pid=self.pid,
            running=self.running,
            exit_code=self.poll(),
            started_at=self.started_at,
            stopped_at=self.stopped_at,
            log_lines=lines,
        )

    # --------------------------------------------------------------- logs
    def logs(self, lines: int = 100, *, stream: str = "all", since: int = 0) -> List[LogLine]:
        """Return the last *lines* captured rows, optionally filtered."""
        with self._lock:
            rows = list(self._log)
        if stream in ("stdout", "stderr"):
            rows = [row for row in rows if row.stream == stream]
        if since > 0:
            rows = rows[since:]
        if lines > 0:
            rows = rows[-lines:]
        return rows

    def wait_for_log(self, needle: str, timeout: float = 20.0) -> bool:
        """Block until *needle* appears in the output (or *timeout*)."""
        deadline = time.monotonic() + max(0.0, timeout)
        lowered = needle.lower()
        while time.monotonic() < deadline:
            with self._lock:
                found = any(lowered in row.text.lower() for row in self._log)
            if found:
                return True
            if not self.running:
                return False
            time.sleep(0.2)
        return False

    # ----------------------------------------------------------- lifecycle
    def stop(self, timeout: float = 8.0) -> int:
        """Terminate the whole process tree; returns the exit code."""
        code = self.poll()
        if code is not None:
            return code
        try:
            if sys.platform.startswith("win"):
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.pid)],
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
            else:
                os.killpg(os.getpgid(self.pid), signal.SIGTERM)
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            logger.debug("terminate %s failed: %s", self.process_id, exc)
        try:
            self._popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
            try:
                if not sys.platform.startswith("win"):
                    os.killpg(os.getpgid(self.pid), signal.SIGKILL)
                else:
                    self._popen.kill()
            except OSError:
                pass
            try:
                self._popen.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        for thread in self._threads:
            thread.join(timeout=2.0)
        self.stopped_at = time.time()
        return self.poll() if self.poll() is not None else -1


class ProcessManager:
    """Owns every background process started inside one workspace."""

    def __init__(
        self,
        runner: CommandRunner,
        *,
        max_processes: int = DEFAULT_MAX_PROCESSES,
        max_log_lines: int = DEFAULT_LOG_LINES,
    ) -> None:
        self._runner = runner
        self.max_processes = int(max_processes)
        self.max_log_lines = int(max_log_lines)
        self._processes: Dict[str, BackgroundProcess] = {}
        self._counter = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------- startup
    def start(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        wait_for: str = "",
        wait_timeout: float = 15.0,
    ) -> BackgroundProcess:
        """Spawn *command* in the background and return its handle.

        The same safety checks as the synchronous runner apply: destructive
        commands are blocked and commands that need a human are refused.
        """
        text = command.strip()
        if not text:
            raise ProcessError("the command must not be empty")
        self._runner.validate(text)  # raises CommandBlockedError
        advice = self._runner.check_interactive(text)
        if advice is not None:
            raise ProcessError(
                f"'{text}' expects interactive input.\n{INTERACTIVE_HINT}\n"
                f"Suggested non-interactive form: {advice}"
            )

        self.reap()
        with self._lock:
            live = [p for p in self._processes.values() if p.running]
            if len(live) >= self.max_processes:
                raise ProcessError(
                    f"too many background processes ({len(live)}/{self.max_processes}) - "
                    "stop one with stop_process before starting another"
                )

        workdir = self._runner._resolve_cwd(cwd)  # noqa: SLF001 - same package contract
        popen_kwargs: Dict[str, object] = {
            "cwd": str(workdir),
            "env": self._runner._build_env(),  # noqa: SLF001
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": subprocess.DEVNULL,
            "shell": True,
            "bufsize": 0,
        }
        if sys.platform.startswith("win"):
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["executable"] = os.environ.get("SHELL", "/bin/sh")
            popen_kwargs["start_new_session"] = True

        try:
            popen = subprocess.Popen(text, **popen_kwargs)  # type: ignore[arg-type]
        except OSError as exc:
            raise ProcessError(f"could not start the process: {exc}") from exc

        with self._lock:
            self._counter += 1
            process_id = f"proc-{self._counter}"
            process = BackgroundProcess(
                process_id, text, workdir, popen, max_log_lines=self.max_log_lines
            )
            self._processes[process_id] = process

        logger.info("started background process %s: %s", process_id, text)
        if wait_for:
            process.wait_for_log(wait_for, timeout=wait_timeout)
        else:
            # Give a failing command a moment to die so the first log read
            # already shows the error instead of an empty buffer.
            time.sleep(min(1.0, max(0.0, wait_timeout)) if wait_timeout else 0.0)
        return process

    # ------------------------------------------------------------ lookup
    def get(self, process_id: str) -> BackgroundProcess:
        with self._lock:
            process = self._processes.get(process_id)
        if process is None:
            known = ", ".join(sorted(self._processes)) or "none"
            raise ProcessError(f"unknown process id {process_id!r} (known: {known})")
        return process

    def list(self, *, running_only: bool = False) -> List[ProcessInfo]:
        with self._lock:
            processes = list(self._processes.values())
        infos = [p.info() for p in processes]
        if running_only:
            infos = [info for info in infos if info.running]
        return infos

    @property
    def running_count(self) -> int:
        return sum(1 for info in self.list() if info.running)

    # --------------------------------------------------------- teardown
    def stop(self, process_id: str, timeout: float = 8.0) -> ProcessInfo:
        process = self.get(process_id)
        process.stop(timeout=timeout)
        logger.info("stopped background process %s", process_id)
        return process.info()

    def stop_all(self, timeout: float = 5.0) -> List[str]:
        """Terminate every live process; returns the ids that were running."""
        stopped: List[str] = []
        for info in self.list(running_only=True):
            try:
                self.stop(info.process_id, timeout=timeout)
                stopped.append(info.process_id)
            except ProcessError:  # pragma: no cover - race with natural exit
                continue
        return stopped

    def reap(self) -> None:
        """Refresh exit codes of finished processes."""
        with self._lock:
            processes = list(self._processes.values())
        for process in processes:
            process.poll()

    def close(self) -> None:
        self.stop_all()
        with self._lock:
            self._processes.clear()

    def __enter__(self) -> "ProcessManager":  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc_info: object) -> None:  # pragma: no cover
        self.close()


def format_log_lines(rows: Sequence[LogLine], *, with_stream: bool = True) -> str:
    """Render captured rows for the model / the UI."""
    if not rows:
        return "(no output captured yet)"
    if not with_stream:
        return "\n".join(row.text for row in rows)
    return "\n".join(
        f"[{row.stream}] {row.text}" if row.stream == "stderr" else row.text for row in rows
    )
