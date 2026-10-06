"""Catch the shell mistakes small models make over and over.

A 9B model driving a Windows shell fails in a very narrow set of ways, and it
fails *repeatedly*, because a bare ``No such file or directory`` tells it
nothing it can act on. A real transcript::

    $ python projede Duz/snake_game.py       # unquoted path with a space
    $ python projede\\Duz\\snake_game.py       # invented a directory level
    $ mv "a" "b"                             # mv does not exist on Windows
    $ cd projede\\Duz & python snake_game.py  # cd failed, '&' ran it anyway
    ... eleven attempts, none of them correct

Every one of those is diagnosable from the workspace itself before or right
after the command runs, which is what this module does:

* :func:`preflight` refuses, *without executing*, the commands that cannot
  possibly work - a ``cd`` into a directory that does not exist, or a
  Unix-only binary on Windows - and names the correct alternative.
* :func:`explain_failure` runs after a command has already failed and turns
  the generic error into "that path does not exist, this one does", including
  the quoting advice when the real path contains a space.

The split matters: refusing up front is only safe when the command is
*certainly* wrong. A path that does not exist yet is often the whole point
(``mkdir``, ``git clone``, a compiler output), so missing paths are only
reported after the shell has actually complained.
"""

from __future__ import annotations

import difflib
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from agent3.workspace.fs import WorkspaceFS

#: Commands that exist on POSIX shells and simply are not there on Windows,
#: mapped to the thing the agent should do instead. Reaching for these is the
#: single most common way a run wastes a step on a Windows workspace.
UNIX_ONLY_ON_WINDOWS = {
    "mv": "use the `rename_file` tool",
    "cp": "use the `read_file` + `write_file` tools, or `copy` in cmd",
    "rm": "use the `delete_file` tool",
    "ls": "use the `list_files` tool",
    "cat": "use the `read_file` tool",
    "touch": "use the `write_file` tool",
    "grep": "use the `search_code` tool",
    "sed": "use the `edit_file` or `patch_file` tool",
    "which": "use `where` on Windows",
    "pwd": "use `cd` with no arguments on Windows",
}

#: Extensions that make a bare token (no separator) obviously a path.
_PATH_EXTENSIONS = {
    ".py", ".pyw", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json",
    ".md", ".txt", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".xml", ".html",
    ".htm", ".css", ".scss", ".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp",
    ".hpp", ".cs", ".rb", ".php", ".sh", ".ps1", ".bat", ".sql", ".csv",
}

_URL = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


@dataclass
class PathSuggestion:
    """One wrong path token and the real path it most likely meant."""

    token: str
    suggestion: str = ""
    needs_quoting: bool = False

    def describe(self) -> str:
        if self.suggestion and self.needs_quoting:
            return (
                f'"{self.token}" does not exist. The real path is '
                f'"{self.suggestion}" - it contains a space, so it must be '
                f"quoted in a shell command."
            )
        if self.suggestion:
            return f'"{self.token}" does not exist. Did you mean "{self.suggestion}"?'
        return f'"{self.token}" does not exist in this workspace.'


@dataclass
class CommandAudit:
    """Outcome of inspecting one shell command."""

    blocked: bool = False
    reason: str = ""
    suggestions: List[PathSuggestion] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.reason or self.suggestions)

    def message(self) -> str:
        parts: List[str] = []
        if self.reason:
            parts.append(self.reason)
        if self.suggestions:
            parts.append("\n".join(f"- {item.describe()}" for item in self.suggestions))
        return "\n\n".join(parts)


# --------------------------------------------------------------- tokenising
def split_command(command: str) -> List[str]:
    """Split *command* into tokens while honouring quotes.

    ``posix=False`` keeps Windows backslashes intact (``shlex`` in POSIX mode
    would eat them as escapes) at the cost of leaving the quote characters on
    the token, which :func:`_clean_token` strips.
    """
    try:
        return shlex.split(command, posix=False)
    except ValueError:  # unbalanced quote - fall back to whitespace
        return command.split()


def _clean_token(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        token = token[1:-1]
    return token


def looks_like_path(token: str) -> bool:
    """True when *token* is plausibly a filesystem path rather than a flag."""
    if not token or token.startswith("-"):
        return False
    if _URL.match(token):
        return False
    if "=" in token.split(os.sep)[0] and "/" not in token and "\\" not in token:
        return False  # KEY=value
    if "/" in token or "\\" in token:
        return True
    _, extension = os.path.splitext(token)
    return extension.lower() in _PATH_EXTENSIONS


def normalise(token: str) -> str:
    """Workspace-relative POSIX form of *token* (``a\\b`` -> ``a/b``)."""
    text = _clean_token(token).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


# ------------------------------------------------------------- suggestions
def _candidate_paths(fs: WorkspaceFS, limit: int = 4000) -> List[str]:
    """Every workspace-relative file and directory path, cheaply."""
    paths: List[str] = []
    try:
        for index, file_path in enumerate(fs.walk_files()):
            if index >= limit:
                break
            relative = fs.relative(file_path)
            paths.append(relative)
            parent = os.path.dirname(relative)
            while parent and parent not in paths:
                paths.append(parent)
                parent = os.path.dirname(parent)
    except Exception:  # pragma: no cover - unreadable workspace
        return paths
    return paths


def closest_path(target: str, candidates: Sequence[str]) -> str:
    """Best real path for *target*, or ``""`` when nothing is close enough.

    Two passes, because the model's mistakes cluster into two shapes:

    1. it dropped or mangled a directory level, so the *basename* is right
       and the prefix is wrong (``Duz/snake_game.py`` for
       ``projede Duz/snake_game.py``);
    2. it mistyped the whole thing, where a plain similarity ratio wins.
    """
    if not target or not candidates:
        return ""
    wanted = normalise(target).lower()
    base = os.path.basename(wanted)

    if base:
        same_name = [c for c in candidates if os.path.basename(c).lower() == base]
        if len(same_name) == 1:
            return same_name[0]
        if same_name:
            # Several files share the name: prefer the one whose full path
            # looks most like what was asked for.
            return max(
                same_name,
                key=lambda c: difflib.SequenceMatcher(None, wanted, c.lower()).ratio(),
            )

    matches = difflib.get_close_matches(
        wanted, [c.lower() for c in candidates], n=1, cutoff=0.72
    )
    if not matches:
        return ""
    lowered = matches[0]
    for candidate in candidates:
        if candidate.lower() == lowered:
            return candidate
    return ""


# ----------------------------------------------------------------- preflight
def preflight(command: str, fs: WorkspaceFS, *, windows: Optional[bool] = None) -> CommandAudit:
    """Reject a command that is guaranteed to fail, before it is executed.

    Only two cases qualify, because a false positive here costs the user a
    working command: a Unix-only binary on Windows, and ``cd`` into a
    directory that is not there.
    """
    audit = CommandAudit()
    text = command.strip()
    if not text:
        return audit
    is_windows = os.name == "nt" if windows is None else windows

    tokens = split_command(text)
    if not tokens:
        return audit

    head = _clean_token(tokens[0]).lower()
    head = os.path.basename(head)
    if head.endswith(".exe"):
        head = head[:-4]

    if is_windows and head in UNIX_ONLY_ON_WINDOWS:
        audit.blocked = True
        audit.reason = (
            f"`{head}` is not a command on Windows, so this would fail no matter "
            f"how it is written - {UNIX_ONLY_ON_WINDOWS[head]}. Nothing was executed."
        )
        return audit

    if head == "cd" and len(tokens) > 1:
        target = _clean_token(tokens[1])
        if target not in ("", ".", "..") and not _URL.match(target):
            relative = normalise(target)
            if relative and not fs.exists(relative):
                suggestion = PathSuggestion(
                    token=target,
                    suggestion=closest_path(relative, _candidate_paths(fs)),
                )
                suggestion.needs_quoting = " " in suggestion.suggestion
                audit.blocked = True
                audit.suggestions.append(suggestion)
                audit.reason = (
                    "That directory does not exist, so the `cd` would fail - and "
                    "a shell keeps going after a failed `cd`, running the rest of "
                    "the line in the wrong place. Nothing was executed.\n"
                    "Do not chain `cd`: pass the directory as the `cwd` argument "
                    'instead, e.g. {"tool": "run_command", "args": {"command": '
                    '"python app.py", "cwd": "src"}}.'
                )
    return audit


# ------------------------------------------------------------ post-mortem
def explain_failure(command: str, fs: WorkspaceFS, *, windows: Optional[bool] = None) -> CommandAudit:
    """Explain a command that already failed, in workspace terms.

    Runs only on a non-zero exit, so a path that the command was supposed to
    *create* has had its chance and a missing path is now genuinely news.
    """
    audit = CommandAudit()
    text = command.strip()
    if not text:
        return audit
    is_windows = os.name == "nt" if windows is None else windows

    tokens = split_command(text)
    chained_cd = any(_clean_token(token).lower() == "cd" for token in tokens)
    if chained_cd and len(tokens) > 2:
        # A `cd` in the line makes every later path relative to a directory we
        # did not pick, so path guesses would be noise. The chaining itself is
        # the bug worth reporting.
        audit.reason = (
            "Do not chain `cd` with the real command. `cd` does not persist "
            "between run_command calls"
            + (
                ", and on Windows `&` runs the next command even when the `cd` "
                "failed, so you end up executing in the wrong directory"
                if is_windows and "&" in text
                else ""
            )
            + ". Pass the folder as the `cwd` argument instead: "
            '{"tool": "run_command", "args": {"command": "python snake_game.py", '
            '"cwd": "projede Duz"}}.'
        )
        return audit

    candidates: Optional[List[str]] = None
    seen: set[str] = set()
    for raw in tokens:
        token = _clean_token(raw)
        if not looks_like_path(token) or token in seen:
            continue
        seen.add(token)
        relative = normalise(token)
        if not relative or fs.exists(relative):
            continue
        if candidates is None:
            candidates = _candidate_paths(fs)
        best = closest_path(relative, candidates)
        if not best:
            continue
        audit.suggestions.append(
            PathSuggestion(token=token, suggestion=best, needs_quoting=" " in best)
        )

    if audit.suggestions:
        audit.reason = "The paths in that command do not match this workspace:"
        if any(item.needs_quoting for item in audit.suggestions):
            audit.reason += (
                "\n(A path containing a space must be quoted, otherwise the shell "
                "splits it into two arguments.)"
            )
    return audit
