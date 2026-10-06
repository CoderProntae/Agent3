"""Tool registry, JSON call protocol and the executable tool implementations.

Protocol
--------
The model answers with normal markdown prose and, when it wants to act, a
single fenced JSON block::

    ```json
    {"tool": "write_file", "args": {"path": "src/app.py", "content": "..."}}
    ```

:func:`parse_tool_calls` is deliberately forgiving - it accepts fenced blocks
with or without a language tag, bare JSON objects, a ``{"tool_calls": [...]}`
wrapper and Ollama's native ``function``/``name``/``parameters`` shape - because
small local models are inconsistent.  Everything that cannot be parsed is
treated as plain prose, never as a silent failure.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent3.core.logging_setup import get_logger, log_exception
from agent3.workspace.command_paths import explain_failure, preflight
from agent3.workspace.diffing import (
    EditError,
    PatchError,
    apply_search_replace,
    apply_unified_patch,
    diff_stats,
    make_unified_diff,
    patch_target_path,
)
from agent3.agent.tasks import TaskError, TaskList
from agent3.workspace.outline import build_outline
from agent3.workspace.fs import WorkspaceFS, WorkspaceSecurityError
from agent3.workspace.git_ops import GitRepo
from agent3.workspace.processes import ProcessError, ProcessManager, format_log_lines
from agent3.workspace.snapshots import SnapshotStore
from agent3.workspace.syntax_check import check_file, check_source
from agent3.workspace.terminal import CommandRunner

logger = get_logger(__name__)

_FENCE_RE = re.compile(r"```(?:json|JSON|tool|action)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class ToolCall:
    """A parsed request to execute one tool."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    raw: str = ""

    def signature(self) -> str:
        """Stable identity used to detect the model looping on itself."""
        try:
            payload = json.dumps(self.args, sort_keys=True)[:400]
        except (TypeError, ValueError):  # pragma: no cover
            payload = str(self.args)[:400]
        return f"{self.name}:{payload}"


@dataclass
class ToolResult:
    """Normalised outcome of a tool execution."""

    ok: bool
    output: str = ""
    error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    diff: str = ""
    title: str = ""
    duration_ms: int = 0

    def observation(self, max_chars: int = 8000) -> str:
        """Text appended to the conversation as the tool's observation."""
        body = self.output if self.ok else (self.error or "unknown error")
        if self.diff:
            body = f"{body}\n\n--- diff ---\n{self.diff}" if body else self.diff
        if len(body) > max_chars:
            keep = max_chars // 2
            body = f"{body[:keep]}\n... [truncated] ...\n{body[-keep:]}"
        status = "SUCCESS" if self.ok else "ERROR"
        return f"[{status}] {body}".strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "data": self.data,
            "diff": self.diff,
            "title": self.title,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ToolContext:
    """Everything a tool needs to touch the outside world."""

    fs: WorkspaceFS
    runner: CommandRunner
    git: GitRepo
    command_timeout: float = 240.0
    max_output_chars: int = 20_000
    #: Optional live-output sink used by the embedded terminal.
    on_command_output: Optional[Callable[[str, str], None]] = None
    #: Invoked right after a file changed so the editor can refresh.
    on_file_changed: Optional[Callable[[str, str], None]] = None
    #: Background servers / watchers started by the agent.
    processes: Optional[ProcessManager] = None
    #: Undo buffer holding the pre-mutation content of every touched file.
    snapshots: Optional[SnapshotStore] = None
    #: The plan the model maintains while it works.
    tasks: Optional[TaskList] = None
    #: Parse/lint every written file and append the verdict to the result.
    auto_syntax_check: bool = True
    #: Allow eslint/tsc/ruff to run when they are installed.
    run_external_linters: bool = True
    #: Notified whenever the task list changed, so the sidebar can redraw.
    on_tasks_changed: Optional[Callable[[TaskList], None]] = None
    #: Notified whenever a background process started or stopped.
    on_processes_changed: Optional[Callable[[], None]] = None

    def process_manager(self) -> ProcessManager:
        """Lazily create the process manager (tests build bare contexts)."""
        if self.processes is None:
            self.processes = ProcessManager(self.runner)
        return self.processes

    def snapshot_store(self) -> SnapshotStore:
        if self.snapshots is None:
            self.snapshots = SnapshotStore()
        return self.snapshots

    def task_list(self) -> TaskList:
        if self.tasks is None:
            self.tasks = TaskList()
        return self.tasks


@dataclass
class ToolSpec:
    """Declarative description used both for dispatch and for the prompt."""

    name: str
    description: str
    parameters: Dict[str, str]
    handler: Callable[[ToolContext, Dict[str, Any]], ToolResult]
    required: List[str] = field(default_factory=list)
    destructive: bool = False
    #: Alternative names accepted from the model (never shown in the prompt,
    #: because offering two spellings of one tool invites it to use both).
    aliases: List[str] = field(default_factory=list)

    def prompt_line(self) -> str:
        params = ", ".join(
            f"{key}{'*' if key in self.required else ''}: {desc}"
            for key, desc in self.parameters.items()
        )
        return f"- {self.name}({params})\n    {self.description}"


# --------------------------------------------------------------------- utils
def _arg(args: Dict[str, Any], *names: str, default: Any = None) -> Any:
    """Fetch the first present key out of several aliases."""
    for name in names:
        if name in args and args[name] is not None:
            return args[name]
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


# ------------------------------------------------- snapshots & syntax glue
def _snapshot(ctx: ToolContext, path: str, tool: str) -> None:
    """Record the pre-mutation state of *path* so undo_file_change works."""
    store = ctx.snapshot_store()
    try:
        if ctx.fs.exists(path) and not ctx.fs.is_dir(path):
            content: Optional[str] = ctx.fs.read_text(path)
        else:
            content = None
    except (OSError, UnicodeDecodeError, WorkspaceSecurityError):
        return  # binary or unreadable: nothing sensible to restore
    store.record(path, content, tool)


def _attach_syntax(ctx: ToolContext, result: ToolResult, path: str) -> ToolResult:
    """Check the file we just wrote and fold the verdict into *result*.

    A syntax error does **not** undo the write - the model needs to see the
    broken file to repair it - but it does flip the result to a failure so the
    verification gate refuses to let the run finish on top of it.
    """
    if not result.ok or not ctx.auto_syntax_check:
        return result
    try:
        check = check_file(
            path,
            root=ctx.fs.root,
            run_linters=ctx.run_external_linters,
        )
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        logger.debug("syntax check of %s failed: %s", path, exc)
        return result
    if not check.checked:
        return result
    result.data["syntax"] = check.to_dict()
    if check.ok:
        if check.warnings:
            result.output = f"{result.output}\n{check.report()}"
        else:
            result.output = f"{result.output}  [{check.summary()}]"
        return result
    report = check.report()
    result.ok = False
    result.error = (
        f"{result.output}\n{report}\n"
        "The file was written but it does not parse. Fix it now - do not move on."
    )
    result.output = result.error
    return result


# ----------------------------------------------------------------- handlers
def _tool_list_files(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = str(_arg(args, "path", "dir", "directory", default="") or "")
    depth = _as_int(_arg(args, "depth", "max_depth", default=3), 3)
    text = ctx.fs.tree_text(path, max_depth=max(1, min(depth, 8)))
    return ToolResult(True, output=text, title=f"list_files {path or '.'}")


def _tool_read_file(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = str(_arg(args, "path", "file", "filename", default=""))
    if not path:
        return ToolResult(False, error="'path' is required")
    start = _arg(args, "start_line", "start")
    end = _arg(args, "end_line", "end")
    if start or end:
        body = ctx.fs.read_lines(path, _as_int(start, 1), _as_int(end, 0) or None)
    else:
        body = ctx.fs.read_text(path)
    return ToolResult(
        True,
        output=f"{path}:\n{body}",
        data={"path": path, "content": body},
        title=f"read_file {path}",
    )


def _tool_write_file(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = str(_arg(args, "path", "file", "filename", default=""))
    content = _arg(args, "content", "text", "body", default=None)
    if not path:
        return ToolResult(False, error="'path' is required")
    if content is None:
        return ToolResult(False, error="'content' is required (use an empty string to clear a file)")
    content = str(content)
    existed = ctx.fs.exists(path) and not ctx.fs.is_dir(path)
    previous = ctx.fs.read_text(path) if existed else ""
    _snapshot(ctx, path, "write_file")
    ctx.fs.write_text(path, content)
    unified = make_unified_diff(previous, content, path)
    stats = diff_stats(unified)
    if ctx.on_file_changed:
        ctx.on_file_changed(path, unified)
    verb = "updated" if existed else "created"
    result = ToolResult(
        True,
        output=f"{verb} {path} (+{stats.added} -{stats.removed}, {len(content.splitlines())} lines)",
        data={"path": path, "created": not existed, "added": stats.added, "removed": stats.removed},
        diff=unified,
        title=f"write_file {path}",
    )
    return _attach_syntax(ctx, result, path)


def _tool_edit_file(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = str(_arg(args, "path", "file", default=""))
    search = _arg(args, "search", "old", "old_text", "find", default=None)
    replace = _arg(args, "replace", "new", "new_text", default=None)
    if not path:
        return ToolResult(False, error="'path' is required")
    if search is None or replace is None:
        return ToolResult(False, error="'search' and 'replace' are both required")
    count = _as_int(_arg(args, "count", default=1), 1)
    previous = ctx.fs.read_text(path)
    try:
        updated = apply_search_replace(previous, str(search), str(replace), count=count)
    except EditError as exc:
        return ToolResult(False, error=str(exc), title=f"edit_file {path}")
    if updated == previous:
        return ToolResult(False, error="the edit produced no change", title=f"edit_file {path}")
    _snapshot(ctx, path, "edit_file")
    ctx.fs.write_text(path, updated)
    unified = make_unified_diff(previous, updated, path)
    stats = diff_stats(unified)
    if ctx.on_file_changed:
        ctx.on_file_changed(path, unified)
    result = ToolResult(
        True,
        output=f"edited {path} (+{stats.added} -{stats.removed})",
        data={"path": path, "added": stats.added, "removed": stats.removed},
        diff=unified,
        title=f"edit_file {path}",
    )
    return _attach_syntax(ctx, result, path)


def _tool_view_outline(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    """Structural map of a file: classes, functions, signatures, docstrings."""
    path = str(_arg(args, "path", "file", "filename", default=""))
    if not path:
        return ToolResult(False, error="'path' is required")
    if ctx.fs.is_dir(path):
        return ToolResult(
            False,
            error=f"{path} is a directory - use list_files for directories",
            title=f"view_outline {path}",
        )
    source = ctx.fs.read_text(path)
    outline = build_outline(path, source)
    include_imports = _as_bool(_arg(args, "imports", "include_imports", default=True), True)
    body = outline.render(include_imports=include_imports)
    if outline.is_empty and outline.error:
        # Not a failure: the model simply has to fall back to read_file.
        body += "\n(hint: call read_file for this file type)"
    return ToolResult(
        True,
        output=body,
        data=outline.to_dict(),
        title=f"view_outline {path}",
    )


def _tool_patch_file(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    """Apply a multi-hunk unified diff in a single, atomic step."""
    patch = _arg(args, "patch", "diff", "unified_diff", "content", default=None)
    if patch is None or not str(patch).strip():
        return ToolResult(False, error="'patch' is required and must contain a unified diff")
    patch = str(patch)
    path = str(_arg(args, "path", "file", "filename", default="") or patch_target_path(patch))
    if not path:
        return ToolResult(
            False,
            error="'path' is required (or add a '+++ b/<path>' header to the patch)",
        )
    # Resolve first so an escape attempt reports the real reason (the sandbox)
    # instead of the misleading "file does not exist".
    ctx.fs.resolve(path)
    if not ctx.fs.exists(path):
        return ToolResult(
            False,
            error=f"{path} does not exist - use write_file to create it",
            title=f"patch_file {path}",
        )

    previous = ctx.fs.read_text(path)
    try:
        patched = apply_unified_patch(previous, patch)
    except PatchError as exc:
        return ToolResult(False, error=str(exc), title=f"patch_file {path}")
    if patched.content == previous:
        return ToolResult(
            False,
            error="the patch produced no change - the file already matches the desired state",
            title=f"patch_file {path}",
        )

    _snapshot(ctx, path, "patch_file")
    ctx.fs.write_text(path, patched.content)
    unified = make_unified_diff(previous, patched.content, path)
    stats = diff_stats(unified)
    if ctx.on_file_changed:
        ctx.on_file_changed(path, unified)

    notes = []
    if patched.shifted:
        notes.append(f"hunks shifted by {patched.offsets} line(s)")
    if patched.fuzzy:
        notes.append(f"{patched.fuzzy} hunk(s) matched ignoring whitespace")
    suffix = f" ({'; '.join(notes)})" if notes else ""
    result = ToolResult(
        True,
        output=(
            f"patched {path}: {patched.hunks_applied} hunk(s) applied, "
            f"+{stats.added} -{stats.removed}{suffix}"
        ),
        data={
            "path": path,
            "hunks": patched.hunks_applied,
            "added": stats.added,
            "removed": stats.removed,
            "offsets": patched.offsets,
            "fuzzy": patched.fuzzy,
        },
        diff=unified,
        title=f"patch_file {path}",
    )
    return _attach_syntax(ctx, result, path)


def _tool_delete_file(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = str(_arg(args, "path", "file", default=""))
    if not path:
        return ToolResult(False, error="'path' is required")
    recursive = _as_bool(_arg(args, "recursive", default=False))
    _snapshot(ctx, path, "delete_file")
    ctx.fs.delete(path, recursive=recursive)
    if ctx.on_file_changed:
        ctx.on_file_changed(path, "")
    return ToolResult(True, output=f"deleted {path}", title=f"delete_file {path}")


def _tool_rename_file(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    source = str(_arg(args, "source", "src", "from", "path", default=""))
    destination = str(_arg(args, "destination", "dst", "to", "new_path", default=""))
    if not source or not destination:
        return ToolResult(False, error="'source' and 'destination' are required")
    _snapshot(ctx, source, "rename_file")
    _snapshot(ctx, destination, "rename_file")
    ctx.fs.rename(source, destination, overwrite=_as_bool(_arg(args, "overwrite", default=False)))
    if ctx.on_file_changed:
        ctx.on_file_changed(destination, "")
    return ToolResult(True, output=f"renamed {source} -> {destination}", title="rename_file")


def _tool_make_directory(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = str(_arg(args, "path", "dir", default=""))
    if not path:
        return ToolResult(False, error="'path' is required")
    ctx.fs.mkdir(path)
    return ToolResult(True, output=f"created directory {path}", title=f"make_directory {path}")


def _tool_search_code(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    query = str(_arg(args, "query", "pattern", "text", default=""))
    if not query:
        return ToolResult(False, error="'query' is required")
    glob = str(_arg(args, "glob", "file_glob", default="*") or "*")
    regex = _as_bool(_arg(args, "regex", default=False))
    hits = ctx.fs.search_text(query, glob=glob, regex=regex, max_results=120)
    if not hits:
        return ToolResult(True, output=f"no matches for {query!r}", title="search_code")
    body = "\n".join(f"{h['path']}:{h['line']}: {h['text']}" for h in hits)
    return ToolResult(
        True, output=f"{len(hits)} match(es):\n{body}", data={"hits": hits}, title="search_code"
    )


def _tool_run_command(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    command = str(_arg(args, "command", "cmd", "shell", default="")).strip()
    if not command:
        return ToolResult(False, error="'command' is required")
    timeout = float(_as_int(_arg(args, "timeout", default=ctx.command_timeout), int(ctx.command_timeout)))
    cwd = _arg(args, "cwd", "directory", default=None)

    # Commands that cannot possibly work are rejected before they burn a
    # shell round-trip - see agent3.workspace.command_paths for why this is
    # limited to the two cases that are certain.
    audit = preflight(command, ctx.fs)
    if audit.blocked:
        return ToolResult(
            False,
            error=audit.message(),
            output=audit.message(),
            title=f"run_command {command[:60]}",
        )

    result = ctx.runner.run(
        command,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        on_output=ctx.on_command_output,
    )
    summary = result.summary(ctx.max_output_chars)
    if getattr(result, "interactive", False):
        # Surface the remediation advice as the error so the loop feeds it
        # straight back to the model instead of a bare "exit 125".
        summary = f"{summary}\n\n{result.error}"
    elif not result.ok:
        # A failed command gets its paths checked against the real workspace,
        # so "No such file or directory" becomes an actionable sentence
        # instead of an invitation to guess again.
        explanation = explain_failure(command, ctx.fs)
        if explanation:
            summary = f"{summary}\n\n{explanation.message()}"
    return ToolResult(
        ok=result.ok,
        output=summary,
        error="" if result.ok else summary,
        data=result.to_dict(),
        title=f"run_command {command[:60]}",
    )


def _tool_git(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    action = str(_arg(args, "action", "subcommand", "command", default="status")).strip().lower()
    repo = ctx.git

    if action in {"status", "st"}:
        return ToolResult(True, output=repo.status_text(), title="git status")
    if action == "init":
        result = repo.init()
        return ToolResult(result.ok, output=result.output, error="" if result.ok else result.output, title="git init")
    if action == "add":
        paths = _arg(args, "paths", "path", "files", default=".")
        result = repo.add(paths if isinstance(paths, (list, tuple)) else str(paths))
        return ToolResult(result.ok, output=result.output or "staged", error=result.stderr, title="git add")
    if action == "commit":
        message = str(_arg(args, "message", "msg", default="")).strip()
        if not message:
            return ToolResult(False, error="'message' is required for a commit")
        result = repo.commit(message, stage_all=_as_bool(_arg(args, "stage_all", "all", default=True), True))
        return ToolResult(result.ok, output=result.output, error="" if result.ok else result.output, title="git commit")
    if action in {"diff", "changes"}:
        text = repo.diff(_arg(args, "path", default=None), staged=_as_bool(_arg(args, "staged", default=False)))
        return ToolResult(True, output=text or "(no differences)", diff=text, title="git diff")
    if action == "log":
        commits = repo.log(_as_int(_arg(args, "limit", default=15), 15))
        body = "\n".join(f"{c['short']} {c['date']} {c['author']}: {c['subject']}" for c in commits)
        return ToolResult(True, output=body or "(no commits yet)", data={"commits": commits}, title="git log")
    if action in {"branch", "checkout"}:
        name = str(_arg(args, "name", "branch", default="")).strip()
        if not name:
            return ToolResult(True, output="\n".join(repo.branches()), title="git branch")
        result = repo.create_branch(name) if action == "branch" else repo.checkout(name)
        return ToolResult(result.ok, output=result.output, error="" if result.ok else result.output, title=f"git {action}")
    if action == "push":
        result = repo.push(
            str(_arg(args, "remote", default="origin")),
            _arg(args, "branch", default=None),
            set_upstream=_as_bool(_arg(args, "set_upstream", default=True), True),
        )
        return ToolResult(result.ok, output=result.output, error="" if result.ok else result.output, title="git push")
    if action == "pull":
        result = repo.pull(str(_arg(args, "remote", default="origin")), _arg(args, "branch", default=None))
        return ToolResult(result.ok, output=result.output, error="" if result.ok else result.output, title="git pull")

    raw_args = _arg(args, "args", default=None)
    if isinstance(raw_args, list):
        result = repo.run(*[str(a) for a in raw_args])
        return ToolResult(result.ok, output=result.output, error="" if result.ok else result.output, title="git")
    return ToolResult(False, error=f"unsupported git action: {action!r}")


# --------------------------------------------------- background processes
def _tool_start_process(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    """Launch a long-running command (dev server, watcher) in the background."""
    command = str(_arg(args, "command", "cmd", "shell", default="")).strip()
    if not command:
        return ToolResult(False, error="'command' is required")
    cwd = _arg(args, "cwd", "directory", default=None)
    wait_for = str(_arg(args, "wait_for", "ready_when", default="") or "")
    wait_timeout = float(_as_int(_arg(args, "wait_timeout", "timeout", default=15), 15))

    audit = preflight(command, ctx.fs)
    if audit.blocked:
        return ToolResult(False, error=audit.message(), title="start_process")

    manager = ctx.process_manager()
    try:
        process = manager.start(
            command,
            cwd=str(cwd) if cwd else None,
            wait_for=wait_for,
            wait_timeout=max(0.0, min(wait_timeout, 120.0)),
        )
    except ProcessError as exc:
        return ToolResult(False, error=str(exc), title="start_process")
    if ctx.on_processes_changed:
        ctx.on_processes_changed()

    info = process.info()
    tail = format_log_lines(process.logs(40))
    if info.running:
        head = (
            f"started {info.process_id} (pid {info.pid}): {command}\n"
            f"It keeps running in the background. Read its output with "
            f"get_process_logs('{info.process_id}') and shut it down with "
            f"stop_process('{info.process_id}') when you are done."
        )
        ok = True
    else:
        head = (
            f"{info.process_id} exited immediately with code {info.exit_code}. "
            "This command is not a long-running process, or it failed to start."
        )
        ok = info.exit_code == 0
    body = f"{head}\n--- first output ---\n{tail}"
    return ToolResult(
        ok,
        output=body,
        error="" if ok else body,
        data=info.to_dict(),
        title=f"start_process {command[:50]}",
    )


def _tool_get_process_logs(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    """Read the captured output of a background process (or list them all)."""
    manager = ctx.process_manager()
    process_id = str(_arg(args, "process_id", "id", "pid", "name", default="") or "").strip()
    if not process_id:
        infos = manager.list()
        if not infos:
            return ToolResult(
                True,
                output="no background processes have been started",
                data={"processes": []},
                title="get_process_logs",
            )
        body = "\n".join(info.summary() for info in infos)
        return ToolResult(
            True,
            output=f"{len(infos)} background process(es):\n{body}",
            data={"processes": [info.to_dict() for info in infos]},
            title="get_process_logs",
        )

    lines = _as_int(_arg(args, "lines", "tail", "n", "count", default=100), 100)
    stream = str(_arg(args, "stream", default="all") or "all").lower()
    wait_for = str(_arg(args, "wait_for", default="") or "")
    try:
        process = manager.get(process_id)
    except ProcessError as exc:
        return ToolResult(False, error=str(exc), title="get_process_logs")
    if wait_for:
        process.wait_for_log(
            wait_for, timeout=float(_as_int(_arg(args, "wait_timeout", default=20), 20))
        )
    rows = process.logs(max(1, min(lines, 1000)), stream=stream)
    info = process.info()
    state = "running" if info.running else f"exited with code {info.exit_code}"
    body = (
        f"{info.process_id} ({state}, {info.log_lines} line(s) captured)\n"
        f"$ {info.command}\n--- last {len(rows)} line(s) ---\n"
        f"{format_log_lines(rows)}"
    )
    return ToolResult(
        True,
        output=body,
        data={**info.to_dict(), "returned_lines": len(rows)},
        title=f"get_process_logs {info.process_id}",
    )


def _tool_stop_process(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    """Terminate a background process tree."""
    manager = ctx.process_manager()
    process_id = str(_arg(args, "process_id", "id", "pid", "name", default="") or "").strip()
    if not process_id or process_id.lower() == "all":
        stopped = manager.stop_all()
        if ctx.on_processes_changed:
            ctx.on_processes_changed()
        if not stopped:
            return ToolResult(True, output="no background process was running", title="stop_process")
        return ToolResult(
            True,
            output=f"stopped {len(stopped)} process(es): {', '.join(stopped)}",
            data={"stopped": stopped},
            title="stop_process",
        )
    try:
        info = manager.stop(process_id)
    except ProcessError as exc:
        return ToolResult(False, error=str(exc), title="stop_process")
    if ctx.on_processes_changed:
        ctx.on_processes_changed()
    tail = format_log_lines(manager.get(process_id).logs(20))
    return ToolResult(
        True,
        output=(
            f"stopped {info.process_id} after {info.uptime_seconds:.0f}s "
            f"(exit code {info.exit_code})\n--- final output ---\n{tail}"
        ),
        data=info.to_dict(),
        title=f"stop_process {info.process_id}",
    )


# ------------------------------------------------------------- task list
def _tool_manage_tasks(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    """Keep a visible plan: add steps, flip their status, list the state."""
    tasks = ctx.task_list()
    action = str(_arg(args, "action", "op", "command", default="list")).strip().lower()
    raw = _arg(args, "tasks", "task", "items", "todos", default=None)

    payloads: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        payloads = [raw]
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                payloads.append(dict(item))
            elif isinstance(item, str) and item.strip():
                payloads.append({"description": item.strip()})
    elif isinstance(raw, str) and raw.strip():
        payloads = [{"description": raw.strip()}]

    # Allow the flat shape {"action":"update","id":"x","status":"completed"}.
    if not payloads and (args.get("id") or args.get("description") or args.get("status")):
        payloads = [
            {
                "id": args.get("id", ""),
                "description": args.get("description", ""),
                "status": args.get("status"),
                "note": args.get("note"),
            }
        ]

    try:
        if action in {"list", "show", "get", "status", ""}:
            pass
        elif action in {"add", "create", "append", "plan"}:
            if not payloads:
                return ToolResult(False, error="'tasks' is required for action=add")
            for payload in payloads:
                tasks.upsert(payload)
        elif action in {"update", "set_status", "progress", "complete", "done"}:
            if not payloads:
                return ToolResult(False, error="'tasks' is required for action=update")
            for payload in payloads:
                task_id = str(payload.get("id") or "").strip()
                if not task_id:
                    return ToolResult(
                        False,
                        error="each task in action=update needs an 'id' (call action=list to see them)",
                    )
                if action in {"complete", "done"} and not payload.get("status"):
                    payload["status"] = "completed"
                tasks.require(task_id)
                tasks.upsert(payload)
        elif action in {"set", "replace", "reset"}:
            tasks.replace(payloads)
        elif action in {"remove", "delete", "drop"}:
            if not payloads:
                return ToolResult(False, error="'tasks' is required for action=remove")
            for payload in payloads:
                tasks.remove(str(payload.get("id") or ""))
        elif action in {"clear", "empty"}:
            tasks.clear()
        else:
            return ToolResult(
                False,
                error=f"unknown action {action!r} - use add, update, list, set, remove or clear",
            )
    except TaskError as exc:
        return ToolResult(False, error=str(exc), title="manage_tasks")

    if ctx.on_tasks_changed:
        ctx.on_tasks_changed(tasks)
    done, total = tasks.progress()
    return ToolResult(
        True,
        output=tasks.render(),
        data={"tasks": tasks.to_list(), "done": done, "total": total, "action": action},
        title=f"manage_tasks {action} ({done}/{total})",
    )


# ---------------------------------------------------------- syntax / undo
def _tool_check_syntax(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    """Parse / lint one file (or an inline snippet) without executing it."""
    path = str(_arg(args, "path", "file", "filename", default="") or "")
    content = _arg(args, "content", "code", "source", default=None)
    run_linters = _as_bool(_arg(args, "lint", "linters", default=ctx.run_external_linters), True)

    if not path and content is None:
        return ToolResult(False, error="'path' is required (or 'content' plus a 'path' to name it)")

    if content is not None:
        check = check_source(
            path or "snippet.py",
            str(content),
            root=ctx.fs.root,
            absolute_path=None,
            run_linters=False,
        )
    else:
        ctx.fs.resolve(path)
        if not ctx.fs.exists(path):
            return ToolResult(False, error=f"{path} does not exist", title=f"check_syntax {path}")
        if ctx.fs.is_dir(path):
            return ToolResult(
                False,
                error=f"{path} is a directory - check one file at a time",
                title=f"check_syntax {path}",
            )
        check = check_file(path, root=ctx.fs.root, run_linters=run_linters)

    report = check.report(max_items=25)
    if not check.checked:
        return ToolResult(
            True,
            output=f"{path}: {check.skipped_reason}",
            data=check.to_dict(),
            title=f"check_syntax {path}",
        )
    return ToolResult(
        check.ok,
        output=report,
        error="" if check.ok else report,
        data=check.to_dict(),
        title=f"check_syntax {path}",
    )


def _tool_undo_file_change(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    """Restore the state a file had before the last tool touched it."""
    store = ctx.snapshot_store()
    path = str(_arg(args, "path", "file", "filename", default="") or "").strip()
    if _as_bool(_arg(args, "list", "history", default=False)):
        entries = store.history(path or None)
        if not entries:
            return ToolResult(True, output="the undo history is empty", title="undo_file_change")
        body = "\n".join(f"{i + 1}. {e.describe()}" for i, e in enumerate(entries))
        return ToolResult(
            True,
            output=f"undo history (newest first):\n{body}",
            data={"history": [e.to_dict() for e in entries]},
            title="undo_file_change",
        )

    snapshot = store.pop(path or None)
    if snapshot is None:
        scope = f" for {path}" if path else ""
        return ToolResult(
            False,
            error=f"nothing to undo{scope} - no file change has been recorded yet",
            title="undo_file_change",
        )

    target = snapshot.path
    try:
        current = ctx.fs.read_text(target) if ctx.fs.exists(target) else ""
        if snapshot.content is None:
            if ctx.fs.exists(target):
                ctx.fs.delete(target, recursive=False)
            unified = make_unified_diff(current, "", target)
            message = f"undid {snapshot.tool}: removed {target} (it did not exist before)"
        else:
            ctx.fs.write_text(target, snapshot.content)
            unified = make_unified_diff(current, snapshot.content, target)
            stats = diff_stats(unified)
            message = (
                f"undid {snapshot.tool}: restored {target} "
                f"(+{stats.added} -{stats.removed}), {len(store)} undo step(s) left"
            )
    except (OSError, WorkspaceSecurityError) as exc:
        return ToolResult(False, error=f"could not restore {target}: {exc}", title="undo_file_change")

    if ctx.on_file_changed:
        ctx.on_file_changed(target, unified)
    return ToolResult(
        True,
        output=message,
        data={"path": target, "tool": snapshot.tool, "remaining": len(store)},
        diff=unified,
        title=f"undo_file_change {target}",
    )


def _tool_project_overview(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    summary = ctx.fs.summarize()
    tree = ctx.fs.tree_text(max_depth=3, max_entries=250)
    git_summary = ctx.git.summary()
    body = (
        f"root: {summary['root']}\n"
        f"files: {summary['files']} ({summary['bytes']} bytes)\n"
        f"extensions: {json.dumps(summary['extensions'])}\n"
        f"git: {json.dumps(git_summary, default=str)[:1500]}\n\n"
        f"{tree}"
    )
    return ToolResult(True, output=body, data={"summary": summary}, title="project_overview")


def _tool_finish(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    summary = str(_arg(args, "summary", "message", "result", default="Task complete."))
    return ToolResult(True, output=summary, data={"final": True}, title="finish")


# ----------------------------------------------------------------- registry
class ToolRegistry:
    """Holds every :class:`ToolSpec` and dispatches calls."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._aliases: Dict[str, str] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register(
            ToolSpec(
                "project_overview",
                "Summarise the workspace: file tree, file-type stats and git state. Call this first when you do not know the project yet.",
                {},
                _tool_project_overview,
            )
        )
        self.register(
            ToolSpec(
                "list_files",
                "List the workspace tree starting at 'path'.",
                {"path": "relative directory (default: workspace root)", "depth": "levels to descend (default 3)"},
                _tool_list_files,
            )
        )
        self.register(
            ToolSpec(
                "read_file",
                "Read a text file. Optionally restrict to a line range.",
                {"path": "relative file path", "start_line": "first line (1-based)", "end_line": "last line"},
                _tool_read_file,
                required=["path"],
            )
        )
        self.register(
            ToolSpec(
                "view_outline",
                "Map a source file WITHOUT reading it: classes, functions, methods, signatures, "
                "docstrings and line numbers. Use this before read_file on anything large, then "
                "read only the line range you actually need.",
                {
                    "path": "relative file path",
                    "imports": "false to hide the import list (default true)",
                },
                _tool_view_outline,
                required=["path"],
            )
        )
        self.register(
            ToolSpec(
                "write_file",
                "Create a file or replace its entire content. Always send the complete file.",
                {"path": "relative file path", "content": "the full new file content"},
                _tool_write_file,
                required=["path", "content"],
            )
        )
        self.register(
            ToolSpec(
                "edit_file",
                "Surgically replace an exact snippet inside a file. Prefer this over write_file for small changes.",
                {
                    "path": "relative file path",
                    "search": "exact snippet to find (include enough context to be unique)",
                    "replace": "replacement snippet",
                    "count": "number of occurrences, 0 = all (default 1)",
                },
                _tool_edit_file,
                required=["path", "search", "replace"],
            )
        )
        self.register(
            ToolSpec(
                "patch_file",
                "Apply a multi-hunk unified diff to ONE file in a single atomic step. Use this "
                "instead of several consecutive edit_file calls when a file changes in two or "
                "more places: hunks are located by their context, so later hunks still land "
                "after earlier ones shifted the line numbers.",
                {
                    "path": "relative file path (optional if the patch has a '+++ b/<path>' header)",
                    "patch": "the unified diff: '@@ -old,count +new,count @@' hunks with ' ' context, '-' removed and '+' added lines",
                },
                _tool_patch_file,
                required=["patch"],
            )
        )
        self.register(
            ToolSpec(
                "delete_file",
                "Delete a file (set recursive=true for a directory).",
                {"path": "relative path", "recursive": "true to delete a directory tree"},
                _tool_delete_file,
                required=["path"],
                destructive=True,
            )
        )
        self.register(
            ToolSpec(
                "rename_file",
                "Rename or move a file inside the workspace.",
                {"source": "current path", "destination": "new path", "overwrite": "true to replace an existing file"},
                _tool_rename_file,
                required=["source", "destination"],
                destructive=True,
            )
        )
        self.register(
            ToolSpec(
                "make_directory",
                "Create a directory (including parents).",
                {"path": "relative directory path"},
                _tool_make_directory,
                required=["path"],
            )
        )
        self.register(
            ToolSpec(
                "search_code",
                "Search the workspace for a string or regular expression.",
                {"query": "text or regex", "glob": "filename filter, e.g. *.py", "regex": "true to treat query as regex"},
                _tool_search_code,
                required=["query"],
            )
        )
        self.register(
            ToolSpec(
                "run_command",
                "Run a shell command inside the workspace (tests, builds, installs). Read the output before continuing.",
                {"command": "the shell command", "cwd": "relative working directory", "timeout": "seconds"},
                _tool_run_command,
                required=["command"],
                destructive=True,
            )
        )
        self.register(
            ToolSpec(
                "git",
                "Version control: action = status|init|add|commit|diff|log|branch|checkout|push|pull.",
                {"action": "git action", "message": "commit message", "paths": "paths for add", "name": "branch name"},
                _tool_git,
                required=["action"],
            )
        )
        self.register(
            ToolSpec(
                "start_process",
                "Start a long-running command in the BACKGROUND and keep working: dev servers, "
                "watchers, 'npm run dev', 'uvicorn app:app'. Never use run_command for these - it "
                "blocks until the command exits, which a server never does. Returns a process_id.",
                {
                    "command": "the shell command to run in the background",
                    "cwd": "relative working directory",
                    "wait_for": "text to wait for in the output before returning, e.g. 'Listening on'",
                    "wait_timeout": "seconds to wait for that text (default 15)",
                },
                _tool_start_process,
                required=["command"],
                destructive=True,
            )
        )
        self.register(
            ToolSpec(
                "get_process_logs",
                "Read the output a background process produced so far. Call it without a "
                "process_id to list every background process and its state.",
                {
                    "process_id": "id returned by start_process (omit to list all processes)",
                    "lines": "how many trailing lines to return (default 100)",
                    "stream": "all | stdout | stderr",
                    "wait_for": "block until this text appears in the output",
                },
                _tool_get_process_logs,
            )
        )
        self.register(
            ToolSpec(
                "stop_process",
                "Terminate a background process and its children. Pass process_id='all' to stop "
                "everything. Always stop what you started before you finish.",
                {"process_id": "id returned by start_process, or 'all'"},
                _tool_stop_process,
                destructive=True,
            )
        )
        self.register(
            ToolSpec(
                "manage_tasks",
                "Maintain your visible plan. action=add with a list of tasks writes the plan down, "
                "action=update flips a task to in_progress or completed, action=list shows it. "
                "Plan before you touch anything non-trivial, mark each step done as you finish it, "
                "and never finish while a task is still pending.",
                {
                    "action": "add | update | list | set | remove | clear",
                    "tasks": "list of {id, description, status} - status is pending|in_progress|completed",
                },
                _tool_manage_tasks,
                aliases=["todo_list", "todo", "task_list", "update_plan", "manage_todo"],
            )
        )
        self.register(
            ToolSpec(
                "check_syntax",
                "Parse and lint a file without running it (Python, JSON, TOML, YAML, XML, INI, "
                "JavaScript, TypeScript). Written files are checked automatically; call this "
                "explicitly to re-verify a file you repaired or to validate a snippet.",
                {
                    "path": "relative file path",
                    "content": "optional snippet to check instead of the file on disk",
                    "lint": "false to skip external linters (eslint/tsc/ruff)",
                },
                _tool_check_syntax,
            )
        )
        self.register(
            ToolSpec(
                "undo_file_change",
                "Roll a file back to the state it had before the last tool modified it. Use this "
                "when an edit went wrong instead of trying to repair the damage by hand. Pass "
                "list=true to see the undo history.",
                {
                    "path": "file to roll back (omit to undo the most recent change anywhere)",
                    "list": "true to show the undo history without changing anything",
                },
                _tool_undo_file_change,
                aliases=["rollback_file", "revert_file", "undo"],
                destructive=True,
            )
        )
        self.register(
            ToolSpec(
                "finish",
                "Finish the task. Provide a short summary of what you changed and how you verified it.",
                {"summary": "final answer for the user"},
                _tool_finish,
                required=["summary"],
            )
        )

    # ------------------------------------------------------------------ api
    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec
        for alias in spec.aliases:
            self._aliases[alias] = spec.name

    def get(self, name: str) -> Optional[ToolSpec]:
        """Look a tool up by its name or by one of its accepted aliases."""
        spec = self._tools.get(name)
        if spec is not None:
            return spec
        key = str(name or "").strip().lower()
        target = self._aliases.get(key) or self._aliases.get(name)
        return self._tools.get(target) if target else self._tools.get(key)

    def names(self) -> List[str]:
        return list(self._tools)

    def specs(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def describe(self) -> str:
        """Render the tool catalogue for the system prompt."""
        return "\n".join(spec.prompt_line() for spec in self._tools.values())

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """Dispatch *call*, converting every exception into a ToolResult."""
        spec = self.get(call.name)
        started = time.monotonic()
        if spec is None:
            known = ", ".join(sorted(self._tools))
            return ToolResult(
                False,
                error=f"unknown tool {call.name!r}. Available tools: {known}",
                title=call.name,
            )
        missing = [key for key in spec.required if not str(call.args.get(key, "")).strip()]
        if missing and not (spec.name == "write_file" and "content" in call.args):
            return ToolResult(
                False,
                error=f"missing required argument(s) for {spec.name}: {', '.join(missing)}",
                title=spec.name,
            )
        try:
            result = spec.handler(ctx, call.args)
        except WorkspaceSecurityError as exc:
            result = ToolResult(False, error=f"security violation: {exc}", title=spec.name)
        except FileNotFoundError as exc:
            result = ToolResult(False, error=f"file not found: {exc}", title=spec.name)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result = ToolResult(False, error=f"{type(exc).__name__}: {exc}", title=spec.name)
        except Exception as exc:  # pragma: no cover - last line of defence
            trace = log_exception(logger, exc, f"tool {spec.name} crashed")
            result = ToolResult(False, error=f"internal tool error:\n{trace[-1500:]}", title=spec.name)
        result.duration_ms = int((time.monotonic() - started) * 1000)
        if not result.title:
            result.title = spec.name
        return result


# -------------------------------------------------------------- parsing
def _coerce_call(payload: Any) -> Optional[ToolCall]:
    """Turn one parsed JSON object into a :class:`ToolCall` if possible."""
    if not isinstance(payload, dict):
        return None
    # Native Ollama / OpenAI style: {"function": {"name": ..., "arguments": {...}}}
    if "function" in payload and isinstance(payload["function"], dict):
        function = payload["function"]
        name = str(function.get("name", "")).strip()
        args = function.get("arguments", {})
    else:
        name = str(payload.get("tool") or payload.get("name") or payload.get("action") or "").strip()
        args = (
            payload.get("args")
            or payload.get("arguments")
            or payload.get("parameters")
            or payload.get("input")
            or {}
        )
    if not name:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"value": args}
    if not isinstance(args, dict):
        args = {"value": args}
    # Some models inline the arguments next to "tool".
    if not args:
        args = {
            k: v
            for k, v in payload.items()
            if k not in {"tool", "name", "action", "thought", "reasoning", "function"}
        }
    thought = str(payload.get("thought") or payload.get("reasoning") or "")
    return ToolCall(name=name, args=args, thought=thought, raw=json.dumps(payload)[:4000])


def _iter_json_candidates(text: str) -> List[str]:
    """Yield JSON-looking substrings: fenced blocks first, then brace scan."""
    candidates = [block.strip() for block in _FENCE_RE.findall(text)]
    if candidates:
        return candidates
    # Brace matching scan for a bare top level object.
    depth = 0
    start = -1
    in_string = False
    escape = False
    found: List[str] = []
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    found.append(text[start : index + 1])
    return found


def parse_tool_calls(text: str, known_tools: Optional[List[str]] = None) -> List[ToolCall]:
    """Extract every tool call contained in an assistant message."""
    if not text or "{" not in text:
        return []
    calls: List[ToolCall] = []
    seen: set[str] = set()
    for candidate in _iter_json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        items: List[Any]
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
            items = payload["tool_calls"]
        else:
            items = [payload]
        for item in items:
            call = _coerce_call(item)
            if call is None:
                continue
            if known_tools is not None and call.name not in known_tools:
                continue
            signature = call.signature()
            if signature in seen:
                continue
            seen.add(signature)
            calls.append(call)
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove fenced tool blocks so the chat bubble shows only prose."""
    cleaned = _FENCE_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
