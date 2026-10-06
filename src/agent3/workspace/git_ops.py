"""Git integration built on the ``git`` CLI.

Shelling out to the real binary (instead of a pure-python implementation)
keeps behaviour identical to what the developer sees in their own terminal,
supports every hook/credential-helper the user already has configured, and
adds zero wheels to the frozen executable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from agent3.core.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 120.0


@dataclass
class GitResult:
    """Outcome of one git invocation."""

    args: List[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def output(self) -> str:
        return (self.stdout or self.stderr).strip()

    def to_dict(self) -> Dict[str, object]:
        return {
            "command": "git " + " ".join(self.args),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "ok": self.ok,
        }


@dataclass
class FileStatus:
    """One entry of ``git status --porcelain``."""

    path: str
    index_status: str
    worktree_status: str

    @property
    def staged(self) -> bool:
        return self.index_status not in (" ", "?", "")

    @property
    def untracked(self) -> bool:
        return self.index_status == "?" and self.worktree_status == "?"

    @property
    def label(self) -> str:
        if self.untracked:
            return "untracked"
        mapping = {"M": "modified", "A": "added", "D": "deleted", "R": "renamed", "C": "copied"}
        code = self.index_status.strip() or self.worktree_status.strip()
        return mapping.get(code, code or "unknown")


class GitRepo:
    """Thin, defensive wrapper around a git working copy."""

    def __init__(self, root: Path | str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._root = Path(root).expanduser().resolve()
        self.timeout = float(timeout)

    # -------------------------------------------------------------- basics
    @property
    def root(self) -> Path:
        return self._root

    def set_root(self, root: Path | str) -> None:
        self._root = Path(root).expanduser().resolve()

    @staticmethod
    def git_available() -> bool:
        return shutil.which("git") is not None

    def run(self, *args: str, timeout: Optional[float] = None) -> GitResult:
        """Execute ``git <args>`` inside the workspace."""
        if not self.git_available():
            return GitResult(list(args), 127, stderr="git executable not found in PATH")
        env = os.environ.copy()
        env.update({"GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"})
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(self._root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout or self.timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return GitResult(list(args), 124, stderr=f"git {' '.join(args)} timed out")
        except OSError as exc:  # pragma: no cover - defensive
            return GitResult(list(args), 127, stderr=str(exc))
        return GitResult(list(args), completed.returncode, completed.stdout, completed.stderr)

    # ------------------------------------------------------------- queries
    def is_repo(self) -> bool:
        if not (self._root / ".git").exists():
            return self.run("rev-parse", "--is-inside-work-tree").stdout.strip() == "true"
        return True

    def current_branch(self) -> str:
        result = self.run("rev-parse", "--abbrev-ref", "HEAD")
        return result.stdout.strip() if result.ok else ""

    def status(self) -> List[FileStatus]:
        """Parse ``git status --porcelain`` into structured entries."""
        result = self.run("status", "--porcelain=v1", "--untracked-files=all")
        entries: List[FileStatus] = []
        if not result.ok:
            return entries
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            index_status, worktree_status, path = line[0], line[1], line[3:]
            if " -> " in path:  # rename
                path = path.split(" -> ", 1)[1]
            entries.append(FileStatus(path.strip().strip('"'), index_status, worktree_status))
        return entries

    def status_text(self) -> str:
        result = self.run("status", "--short", "--branch")
        return result.output or "(clean working tree)"

    def diff(self, path: Optional[str] = None, *, staged: bool = False, context: int = 3) -> str:
        args = ["diff", f"--unified={int(context)}", "--no-color"]
        if staged:
            args.append("--cached")
        if path:
            args.extend(["--", path])
        result = self.run(*args)
        return result.stdout if result.ok else result.stderr

    def log(self, limit: int = 20, *, path: Optional[str] = None) -> List[Dict[str, str]]:
        """Recent commits as dictionaries (hash/author/date/subject)."""
        fmt = "%H%x1f%an%x1f%ad%x1f%s"
        args = ["log", f"-{int(limit)}", f"--pretty=format:{fmt}", "--date=short"]
        if path:
            args.extend(["--", path])
        result = self.run(*args)
        commits: List[Dict[str, str]] = []
        if not result.ok:
            return commits
        for line in result.stdout.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 4:
                commits.append(
                    {"hash": parts[0], "short": parts[0][:8], "author": parts[1], "date": parts[2], "subject": parts[3]}
                )
        return commits

    def show_file(self, path: str, revision: str = "HEAD") -> str:
        result = self.run("show", f"{revision}:{path}")
        return result.stdout if result.ok else ""

    def branches(self) -> List[str]:
        result = self.run("branch", "--list", "--format=%(refname:short)")
        return [b.strip() for b in result.stdout.splitlines() if b.strip()] if result.ok else []

    def remotes(self) -> Dict[str, str]:
        result = self.run("remote", "-v")
        remotes: Dict[str, str] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                remotes.setdefault(parts[0], parts[1])
        return remotes

    def is_dirty(self) -> bool:
        return bool(self.status())

    # ------------------------------------------------------------ mutation
    def init(self, *, default_branch: str = "main") -> GitResult:
        if self.is_repo():
            return GitResult(["init"], 0, stdout="repository already initialised")
        result = self.run("init", "-b", default_branch)
        if not result.ok:  # older git without -b
            result = self.run("init")
            if result.ok:
                self.run("checkout", "-b", default_branch)
        self._ensure_identity()
        return result

    def _ensure_identity(self) -> None:
        """Make commits possible on a machine without global git identity."""
        if not self.run("config", "user.email").output:
            self.run("config", "user.email", "agent3@localhost")
        if not self.run("config", "user.name").output:
            self.run("config", "user.name", "Agent3")

    def add(self, paths: Sequence[str] | str = ".") -> GitResult:
        targets = [paths] if isinstance(paths, str) else list(paths)
        return self.run("add", "--", *targets)

    def commit(self, message: str, *, stage_all: bool = False, allow_empty: bool = False) -> GitResult:
        if stage_all:
            self.add(".")
        self._ensure_identity()
        args = ["commit", "-m", message]
        if allow_empty:
            args.append("--allow-empty")
        return self.run(*args)

    def create_branch(self, name: str, *, checkout: bool = True) -> GitResult:
        return self.run("checkout", "-b", name) if checkout else self.run("branch", name)

    def checkout(self, ref: str) -> GitResult:
        return self.run("checkout", ref)

    def reset_file(self, path: str) -> GitResult:
        return self.run("checkout", "--", path)

    def stash(self, message: str = "agent3 autostash") -> GitResult:
        return self.run("stash", "push", "-u", "-m", message)

    def pull(self, remote: str = "origin", branch: Optional[str] = None) -> GitResult:
        args = ["pull", "--ff-only", remote]
        if branch:
            args.append(branch)
        return self.run(*args, timeout=max(self.timeout, 180.0))

    def push(self, remote: str = "origin", branch: Optional[str] = None, *, set_upstream: bool = False) -> GitResult:
        args = ["push"]
        if set_upstream:
            args.append("--set-upstream")
        args.append(remote)
        args.append(branch or self.current_branch())
        return self.run(*args, timeout=max(self.timeout, 300.0))

    def set_remote(self, name: str, url: str) -> GitResult:
        if name in self.remotes():
            return self.run("remote", "set-url", name, url)
        return self.run("remote", "add", name, url)

    def summary(self) -> Dict[str, object]:
        """Compact repository state used to prime the agent's context."""
        if not self.is_repo():
            return {"repo": False}
        changes = self.status()
        return {
            "repo": True,
            "branch": self.current_branch(),
            "dirty": bool(changes),
            "changed_files": [f"{c.label}: {c.path}" for c in changes[:40]],
            "remotes": self.remotes(),
            "recent_commits": [f"{c['short']} {c['subject']}" for c in self.log(5)],
        }
