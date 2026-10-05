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
from agent3.workspace.diffing import EditError, apply_search_replace, diff_stats, make_unified_diff
from agent3.workspace.fs import WorkspaceFS, WorkspaceSecurityError
from agent3.workspace.git_ops import GitRepo
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


@dataclass
class ToolSpec:
    """Declarative description used both for dispatch and for the prompt."""

    name: str
    description: str
    parameters: Dict[str, str]
    handler: Callable[[ToolContext, Dict[str, Any]], ToolResult]
    required: List[str] = field(default_factory=list)
    destructive: bool = False

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
    ctx.fs.write_text(path, content)
    unified = make_unified_diff(previous, content, path)
    stats = diff_stats(unified)
    if ctx.on_file_changed:
        ctx.on_file_changed(path, unified)
    verb = "updated" if existed else "created"
    return ToolResult(
        True,
        output=f"{verb} {path} (+{stats.added} -{stats.removed}, {len(content.splitlines())} lines)",
        data={"path": path, "created": not existed, "added": stats.added, "removed": stats.removed},
        diff=unified,
        title=f"write_file {path}",
    )


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
    ctx.fs.write_text(path, updated)
    unified = make_unified_diff(previous, updated, path)
    stats = diff_stats(unified)
    if ctx.on_file_changed:
        ctx.on_file_changed(path, unified)
    return ToolResult(
        True,
        output=f"edited {path} (+{stats.added} -{stats.removed})",
        data={"path": path, "added": stats.added, "removed": stats.removed},
        diff=unified,
        title=f"edit_file {path}",
    )


def _tool_delete_file(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = str(_arg(args, "path", "file", default=""))
    if not path:
        return ToolResult(False, error="'path' is required")
    recursive = _as_bool(_arg(args, "recursive", default=False))
    ctx.fs.delete(path, recursive=recursive)
    if ctx.on_file_changed:
        ctx.on_file_changed(path, "")
    return ToolResult(True, output=f"deleted {path}", title=f"delete_file {path}")


def _tool_rename_file(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    source = str(_arg(args, "source", "src", "from", "path", default=""))
    destination = str(_arg(args, "destination", "dst", "to", "new_path", default=""))
    if not source or not destination:
        return ToolResult(False, error="'source' and 'destination' are required")
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
    result = ctx.runner.run(
        command,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        on_output=ctx.on_command_output,
    )
    return ToolResult(
        ok=result.ok,
        output=result.summary(ctx.max_output_chars),
        error="" if result.ok else result.summary(ctx.max_output_chars),
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

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

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
