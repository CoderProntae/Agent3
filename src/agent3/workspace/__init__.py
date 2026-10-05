"""Sandboxed workspace primitives: filesystem, diffing, shell and git."""

from __future__ import annotations

from agent3.workspace.diffing import DiffStats, apply_search_replace, diff_stats, make_unified_diff
from agent3.workspace.fs import FileEntry, WorkspaceFS, WorkspaceSecurityError
from agent3.workspace.git_ops import GitRepo, GitResult
from agent3.workspace.terminal import CommandBlockedError, CommandResult, CommandRunner

__all__ = [
    "CommandBlockedError",
    "CommandResult",
    "CommandRunner",
    "DiffStats",
    "FileEntry",
    "GitRepo",
    "GitResult",
    "WorkspaceFS",
    "WorkspaceSecurityError",
    "apply_search_replace",
    "diff_stats",
    "make_unified_diff",
]
