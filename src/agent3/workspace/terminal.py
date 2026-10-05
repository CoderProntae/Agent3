"""Headless shell execution with capture, timeouts and a safety deny-list.

The agent uses this to run ``pytest``, ``pip install``, ``npm run build``,
``git status`` and friends.  Output is streamed line by line so the embedded
terminal console stays live, and the final :class:`CommandResult` (stdout,
stderr, exit code, duration) is what the agent reasons about when it has to
repair its own mistakes.
"""

from __future__ import annotations

import codecs
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


#: Commands that will sit there waiting for a human.  Each entry is
#: ``(pattern, non-interactive advice)``.  The check runs *before* the process
#: is spawned, so the agent loses a few milliseconds instead of a 240 s
#: timeout, and it is told exactly which flag to add.
INTERACTIVE_COMMAND_RULES: tuple[tuple[str, str], ...] = (
    (r"^\s*npm\s+init\s*(?!.*(-y|--yes))", "npm init -y"),
    (r"^\s*yarn\s+init\s*(?!.*(-y|--yes|-2))", "yarn init -y"),
    (r"^\s*pnpm\s+init\s*(?!.*(-y|--yes))", "pnpm init"),
    (r"^\s*npx\s+(?!.*(--yes|-y))", "npx --yes <package>"),
    (r"^\s*(apt|apt-get|yum|dnf|zypper)\s+(install|remove|upgrade|purge)\b(?!.*(-y|--yes))", "apt-get install -y <package>"),
    (r"^\s*pacman\s+-S\b(?!.*--noconfirm)", "pacman -S --noconfirm <package>"),
    (r"^\s*(pip|pip3|python\s+-m\s+pip)\s+uninstall\b(?!.*(-y|--yes))", "pip uninstall -y <package>"),
    (r"^\s*conda\s+(install|remove|update|create)\b(?!.*(-y|--yes))", "conda install -y <package>"),
    (r"^\s*git\s+commit\b(?!.*(-m|--message|-F|--file|--no-edit|--amend\s+--no-edit))", 'git commit -m "message"'),
    (r"^\s*git\s+(rebase|add)\s+(-i|--interactive)\b", "git rebase --onto / git add <paths>"),
    (r"^\s*(vi|vim|nvim|nano|emacs|pico|joe)\b", "read_file / write_file instead of an editor"),
    (r"^\s*(less|more|man|top|htop|watch|tail\s+-f|journalctl\s+-f)\b", "cat / tail -n 200 (never a pager or follow mode)"),
    (r"^\s*(ssh|sftp|telnet|ftp|mysql|psql|sqlite3|mongo|redis-cli)\s*$", "pass the query as an argument, e.g. psql -c \"SELECT 1\""),
    (r"^\s*(python|python3|node|irb|php\s+-a|R)\s*$", 'python -c "..." or run a script file'),
    (r"^\s*(docker|podman)\s+(run|exec)\b.*\s-[a-z]*i[a-z]*t?\b", "docker run --rm (no -it)"),
    (r"^\s*ssh-keygen\b(?!.*-N)", 'ssh-keygen -N "" -f <path>'),
    (r"^\s*(rails|django-admin|php\s+artisan)\s+.*(--interactive|-i)\b", "the non-interactive form of the generator"),
    (r"^\s*read\s+(-p|[A-Za-z_])", "do not read from stdin inside an agent command"),
    (r"^\s*(gh|aws|gcloud|az)\s+(auth\s+)?(login|configure)\b(?!.*(--with-token|--no-browser))", "export the credential as an environment variable"),
)

#: Tail-of-output patterns that mean "a prompt is on screen, nobody will type".
PROMPT_PATTERNS: tuple[str, ...] = (
    r"\((y|yes)/(n|no)\)\s*[:?]?\s*$",
    r"\[(y|Y)/(n|N)\]\s*[:?]?\s*$",
    r"\[(yes|no)\]\s*[:?]?\s*$",
    r"\?\s*\(.*\)\s*$",
    r"(?i)\bpassword\s*:\s*$",
    r"(?i)\bpassphrase[^:]*:\s*$",
    r"(?i)\busername\s*:\s*$",
    r"(?i)\bcontinue\b[^\n]{0,40}\?\s*$",
    r"(?i)\bproceed\b[^\n]{0,40}\?\s*$",
    r"(?i)\boverwrite\b[^\n]{0,40}\?\s*$",
    r"(?i)press\s+(any\s+key|enter|return)",
    r"(?i)are\s+you\s+sure",
    r"(?i)^\s*enter\s+[^\n]{0,60}:\s*$",
    r"(?i)^\s*select\s+[^\n]{0,60}:\s*$",
    r"(?i)^\s*choose\s+[^\n]{0,60}:\s*$",
    r"^\s*[>?]{1,3}\s*$",
    r"^\s*\.\.\.\s*$",
    r"(?i)^\s*ok\s*\?\s*$",
)

#: Advice appended to every "this command wanted a human" result.
INTERACTIVE_HINT = (
    "This command was stopped because it is waiting for interactive input, "
    "which is impossible inside the agent: nobody can type an answer. "
    "Re-run it in non-interactive mode - add the flag that accepts the "
    "defaults (for example `npm init -y`, `apt-get install -y`, "
    "`git commit -m \"...\"`, `pip uninstall -y`), pipe the answer in "
    "(`yes | <command>`), or set the matching environment variable "
    "(`DEBIAN_FRONTEND=noninteractive`, `CI=1`, `GIT_TERMINAL_PROMPT=0`)."
)


class CommandNeedsInputError(RuntimeError):
    """Raised when a command obviously requires a human at the keyboard."""


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
    interactive: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.exit_code == 0
            and not self.timed_out
            and not self.blocked
            and not self.interactive
        )

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
            "interactive": self.interactive,
            "ok": self.ok,
            "error": self.error,
        }

    def summary(self, max_chars: int = 4000) -> str:
        """Compact text block handed back to the model."""
        status = (
            "BLOCKED"
            if self.blocked
            else "NEEDS INTERACTIVE INPUT"
            if self.interactive
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
        detect_interactive: bool = True,
        interactive_idle_seconds: float = 15.0,
    ) -> None:
        self._cwd = Path(cwd).expanduser().resolve()
        self._patterns = [re.compile(p, re.IGNORECASE) for p in blocked_patterns]
        self.default_timeout = float(default_timeout)
        self.env_overrides = dict(env_overrides or {})
        self.max_output_chars = int(max_output_chars)
        self.detect_interactive = bool(detect_interactive)
        self.interactive_idle_seconds = float(interactive_idle_seconds)
        self._interactive_rules = [
            (re.compile(pattern, re.IGNORECASE), advice)
            for pattern, advice in INTERACTIVE_COMMAND_RULES
        ]
        self._prompt_patterns = [re.compile(p, re.MULTILINE) for p in PROMPT_PATTERNS]
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

    def check_interactive(self, command: str) -> Optional[str]:
        """Return the non-interactive advice when *command* needs a human.

        Only the first command of a pipeline/chain is inspected for the
        "bare REPL" rules, but every segment is checked against the flag
        rules, so ``cd app && npm init`` is caught as well.
        """
        if not self.detect_interactive:
            return None
        text = command.strip()
        if not text:
            return None
        if re.search(r"(^|\s)(yes|printf|echo)\s[^|]*\|", text):
            return None  # the model is already piping an answer in
        segments = [seg.strip() for seg in re.split(r"&&|\|\||;|\n", text) if seg.strip()]
        for segment in segments or [text]:
            for pattern, advice in self._interactive_rules:
                if pattern.search(segment):
                    return advice
        return None

    def looks_like_prompt(self, tail: str) -> bool:
        """True when the tail of the output is an unanswered question."""
        if not tail.strip():
            return False
        last = tail.rstrip("\n")
        last = last.splitlines()[-1] if last.splitlines() else ""
        candidate = last.strip()
        if not candidate or len(candidate) > 200:
            return False
        return any(pattern.search(candidate) for pattern in self._prompt_patterns)

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Keep tool output machine readable.
        env.setdefault("NO_COLOR", "1")
        env.setdefault("TERM", "dumb")
        env.setdefault("GIT_PAGER", "cat")
        env.setdefault("PAGER", "cat")
        # Tell well behaved tools up front that there is no human present.
        env.setdefault("CI", "1")
        env.setdefault("DEBIAN_FRONTEND", "noninteractive")
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        env.setdefault("GIT_EDITOR", "true")
        env.setdefault("EDITOR", "true")
        env.setdefault("VISUAL", "true")
        env.setdefault("GIT_MERGE_AUTOEDIT", "no")
        env.setdefault("NPM_CONFIG_YES", "true")
        env.setdefault("PIP_NO_INPUT", "1")
        env.setdefault("COMPOSER_NO_INTERACTION", "1")
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

        advice = self.check_interactive(command)
        if advice is not None:
            logger.info("refused interactive command: %s", command)
            message = (
                f"'{command.strip()}' expects interactive input.\n{INTERACTIVE_HINT}\n"
                f"Suggested non-interactive form: {advice}"
            )
            return CommandResult(
                command=command,
                cwd=str(cwd or self._cwd),
                exit_code=125,
                stderr=message,
                interactive=True,
                error=message,
                duration_ms=int((time.monotonic() - started) * 1000),
            )

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
            # Binary + unbuffered on purpose: a prompt such as "Continue? [y/N] "
            # carries NO trailing newline, so a line oriented reader would block
            # on it forever and the stall detector below would never see the
            # question. Reading raw chunks makes partial lines visible.
            "text": False,
            "bufsize": 0,
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
        # Index 0 holds the not-yet-terminated tail of each stream, which is
        # exactly where an unanswered prompt lives.
        stdout_partial: List[str] = [""]
        stderr_partial: List[str] = [""]

        def pump(stream, sink: List[str], partial: List[str], name: str) -> None:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            buffer = ""
            try:
                while True:
                    data = stream.read(4096)
                    if not data:
                        break
                    buffer += decoder.decode(data)
                    while True:
                        index = buffer.find("\n")
                        if index < 0:
                            break
                        line, buffer = buffer[: index + 1], buffer[index + 1 :]
                        sink.append(line)
                        if on_output is not None:
                            try:
                                on_output(name, line)
                            except Exception:  # pragma: no cover - UI callback must never kill us
                                logger.debug("output callback raised", exc_info=True)
                    partial[0] = buffer
            except (ValueError, OSError):  # pragma: no cover - stream closed early
                pass
            finally:
                buffer += decoder.decode(b"", final=True)
                if buffer:
                    sink.append(buffer)
                    if on_output is not None:
                        try:
                            on_output(name, buffer)
                        except Exception:  # pragma: no cover
                            logger.debug("output callback raised", exc_info=True)
                partial[0] = ""
                try:
                    stream.close()
                except Exception:  # pragma: no cover
                    pass

        threads = [
            threading.Thread(
                target=pump, args=(process.stdout, stdout_chunks, stdout_partial, "stdout"), daemon=True
            ),
            threading.Thread(
                target=pump, args=(process.stderr, stderr_chunks, stderr_partial, "stderr"), daemon=True
            ),
        ]
        for thread in threads:
            thread.start()

        timed_out = False
        needs_input = False
        deadline = time.monotonic() + limit
        idle_limit = max(2.0, self.interactive_idle_seconds)
        last_len = 0
        last_change = time.monotonic()
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
                # A process that printed a question and then went quiet is
                # waiting for an answer that will never come. Kill it early
                # instead of burning the whole timeout, but only when the
                # tail really looks like a prompt - a silent compiler must
                # not be mistaken for one.
                if not self.detect_interactive:
                    continue
                current_len = (
                    len(stdout_chunks)
                    + len(stderr_chunks)
                    + len(stdout_partial[0])
                    + len(stderr_partial[0])
                )
                if current_len != last_len:
                    last_len = current_len
                    last_change = time.monotonic()
                    continue
                if time.monotonic() - last_change < idle_limit:
                    continue
                tail = (
                    "".join(stdout_chunks[-2:])
                    + stdout_partial[0]
                    + "\n"
                    + "".join(stderr_chunks[-2:])
                    + stderr_partial[0]
                )
                if self.looks_like_prompt(tail):
                    needs_input = True
                    logger.info("terminating command waiting for input: %s", command)
                    self.terminate(process)
                    break
                last_change = time.monotonic()

        for thread in threads:
            thread.join(timeout=3.0)
        with self._lock:
            self._process = None

        stdout = "".join(stdout_chunks)[: self.max_output_chars]
        stderr = "".join(stderr_chunks)[: self.max_output_chars]
        exit_code = process.returncode if process.returncode is not None else -1
        if needs_input:
            error = (
                "the command stopped and waited for input"
                f" (last prompt: {stdout.strip().splitlines()[-1].strip() if stdout.strip() else stderr.strip().splitlines()[-1].strip() if stderr.strip() else '?'})"
                f"\n{INTERACTIVE_HINT}"
            )
        elif timed_out:
            error = f"timed out after {limit:.0f}s"
        else:
            error = ""
        result = CommandResult(
            command=command,
            cwd=str(workdir),
            exit_code=125 if needs_input else 124 if timed_out else exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=timed_out,
            interactive=needs_input,
            error=error,
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
