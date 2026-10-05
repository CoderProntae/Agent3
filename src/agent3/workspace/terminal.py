"""Headless shell execution with capture, timeouts and a safety deny-list.

The agent uses this to run ``pytest``, ``pip install``, ``npm run build``,
``git status`` and friends.  Output is streamed line by line so the embedded
terminal console stays live, and the final :class:`CommandResult` (stdout,
stderr, exit code, duration) is what the agent reasons about when it has to
repair its own mistakes.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from agent3.core.logging_setup import get_logger

logger = get_logger(__name__)

OutputCallback = Callable[[str, str], None]  # (stream_name, text)

DEFAULT_BLOCKED_PATTERNS: tuple[str, ...] = (
    r"rm\s+-rf\s+(/|~|\*)\s*$",
    r"rm\s+-rf\s+/\s",
    r":\(\)\s*\{\s*:\|:&\s*\};:",
    r"\bmkfs(\.\w+)?\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\bformat\s+[a-zA-Z]:",
    r"del\s+/[sf]\s+/q\s+[a-zA-Z]:\\",
    r">\s*/dev/sd[a-z]",
    r"\bdd\b.*\bof=/dev/",
    r"chmod\s+-R\s+777\s+/\s*$",
    r"curl[^|]*\|\s*(ba)?sh",
    r"wget[^|]*\|\s*(ba)?sh",
)


class CommandBlockedError(PermissionError):
    """Raised when a command matches the destructive-command deny-list."""


@dataclass
class CommandResult:
    """Everything the agent needs to judge a command execution."""

    command: str
    cwd: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False
    blocked: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.blocked

    @property
    def combined_output(self) -> str:
        parts: List[str] = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append(f"[stderr]\n{self.stderr.rstrip()}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, object]:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "blocked": self.blocked,
            "ok": self.ok,
            "error": self.error,
        }

    def summary(self, max_chars: int = 4000) -> str:
        """Compact text block handed back to the model."""
        status = (
            "BLOCKED"
            if self.blocked
            else "TIMEOUT"
            if self.timed_out
            else "OK"
            if self.ok
            else f"FAILED (exit {self.exit_code})"
        )
        body = self.combined_output or "(no output)"
        if len(body) > max_chars:
            keep = max_chars // 2
            body = f"{body[:keep]}\n... [output truncated] ...\n{body[-keep:]}"
        return f"$ {self.command}\nstatus: {status}  |  {self.duration_ms} ms\n{body}"


class CommandRunner:
    """Execute shell commands confined to the workspace directory."""

    def __init__(
        self,
        cwd: Path | str,
        *,
        blocked_patterns: Sequence[str] = DEFAULT_BLOCKED_PATTERNS,
        default_timeout: float = 240.0,
        env_overrides: Optional[Dict[str, str]] = None,
        max_output_chars: int = 200_000,
    ) -> None:
        self._cwd = Path(cwd).expanduser().resolve()
        self._patterns = [re.compile(p, re.IGNORECASE) for p in blocked_patterns]
        self.default_timeout = float(default_timeout)
        self.env_overrides = dict(env_overrides or {})
        self.max_output_chars = int(max_output_chars)
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- config
    @property
    def cwd(self) -> Path:
        return self._cwd

    def set_cwd(self, cwd: Path | str) -> None:
        self._cwd = Path(cwd).expanduser().resolve()

    def set_blocked_patterns(self, patterns: Sequence[str]) -> None:
        self._patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def validate(self, command: str) -> None:
        """Raise :class:`CommandBlockedError` for destructive commands."""
        text = command.strip()
        if not text:
            raise CommandBlockedError("empty command")
        for pattern in self._patterns:
            if pattern.search(text):
                raise CommandBlockedError(
                    f"command blocked by Agent3 safety policy (pattern: {pattern.pattern})"
                )

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Keep tool output machine readable.
        env.setdefault("NO_COLOR", "1")
        env.setdefault("TERM", "dumb")
        env.setdefault("GIT_PAGER", "cat")
        env.setdefault("PAGER", "cat")
        env.update(self.env_overrides)
        return env

    def _resolve_cwd(self, cwd: Optional[str]) -> Path:
        if not cwd:
            return self._cwd
        candidate = (self._cwd / cwd).resolve() if not Path(cwd).is_absolute() else Path(cwd).resolve()
        if candidate != self._cwd and self._cwd not in candidate.parents:
            raise CommandBlockedError(f"working directory escapes the workspace: {cwd}")
        return candidate

    # ---------------------------------------------------------------- run
    def run(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        on_output: Optional[OutputCallback] = None,
        cancel: Optional[threading.Event] = None,
    ) -> CommandResult:
        """Run *command* and return the captured :class:`CommandResult`."""
        started = time.monotonic()
        try:
            self.validate(command)
            workdir = self._resolve_cwd(cwd)
        except CommandBlockedError as exc:
            logger.warning("blocked command: %s (%s)", command, exc)
            return CommandResult(
                command=command,
                cwd=str(cwd or self._cwd),
                exit_code=126,
                stderr=str(exc),
                blocked=True,
                error=str(exc),
            )

        limit = float(timeout or self.default_timeout)
        popen_kwargs: Dict[str, object] = {
            "cwd": str(workdir),
            "env": self._build_env(),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": subprocess.DEVNULL,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if sys.platform.startswith("win"):
            popen_kwargs["shell"] = True
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["shell"] = True
            popen_kwargs["executable"] = os.environ.get("SHELL", "/bin/sh")
            popen_kwargs["start_new_session"] = True

        try:
            process = subprocess.Popen(command, **popen_kwargs)  # type: ignore[arg-type]
        except OSError as exc:
            return CommandResult(
                command=command,
                cwd=str(workdir),
                exit_code=127,
                stderr=str(exc),
                error=str(exc),
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        with self._lock:
            self._process = process

        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []

        def pump(stream, sink: List[str], name: str) -> None:
            try:
                for line in iter(stream.readline, ""):
                    sink.append(line)
                    if on_output is not None:
                        try:
                            on_output(name, line)
                        except Exception:  # pragma: no cover - UI callback must never kill us
                            logger.debug("output callback raised", exc_info=True)
            except (ValueError, OSError):  # pragma: no cover - stream closed early
                pass
            finally:
                try:
                    stream.close()
                except Exception:  # pragma: no cover
                    pass

        threads = [
            threading.Thread(target=pump, args=(process.stdout, stdout_chunks, "stdout"), daemon=True),
            threading.Thread(target=pump, args=(process.stderr, stderr_chunks, "stderr"), daemon=True),
        ]
        for thread in threads:
            thread.start()

        timed_out = False
        deadline = time.monotonic() + limit
        while True:
            try:
                process.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    self.terminate(process)
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    self.terminate(process)
                    break

        for thread in threads:
            thread.join(timeout=3.0)
        with self._lock:
            self._process = None

        stdout = "".join(stdout_chunks)[: self.max_output_chars]
        stderr = "".join(stderr_chunks)[: self.max_output_chars]
        exit_code = process.returncode if process.returncode is not None else -1
        result = CommandResult(
            command=command,
            cwd=str(workdir),
            exit_code=124 if timed_out else exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=timed_out,
            error=f"timed out after {limit:.0f}s" if timed_out else "",
        )
        logger.info(
            "command %r finished exit=%s in %dms", command, result.exit_code, result.duration_ms
        )
        return result

    def run_argv(self, argv: Sequence[str], **kwargs) -> CommandResult:
        """Convenience wrapper that quotes *argv* into a shell string."""
        return self.run(" ".join(shlex.quote(str(a)) for a in argv), **kwargs)

    # ---------------------------------------------------------- lifecycle
    def terminate(self, process: Optional[subprocess.Popen] = None) -> None:
        """Terminate the (current) child process tree."""
        target = process
        if target is None:
            with self._lock:
                target = self._process
        if target is None or target.poll() is not None:
            return
        try:
            if sys.platform.startswith("win"):
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(target.pid)],
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
            else:
                os.killpg(os.getpgid(target.pid), signal.SIGTERM)
                try:
                    target.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(target.pid), signal.SIGKILL)
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            logger.debug("terminate failed: %s", exc)
            try:
                target.kill()
            except Exception:
                pass

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None
