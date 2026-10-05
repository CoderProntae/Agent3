"""Shared pytest fixtures.

Every test runs against an isolated ``AGENT3_HOME`` so the developer's real
configuration, quota policy and usage database are never touched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point every Agent3 data path at a throwaway directory."""
    home = tmp_path / "agent3-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AGENT3_HOME", str(home))

    from agent3.core import paths as paths_module

    paths_module.reset_app_paths_cache()
    yield home
    paths_module.reset_app_paths_cache()


@pytest.fixture
def workspace(tmp_path):
    """An empty workspace directory with a couple of seed files."""
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text(
        "def greet(name):\n    return f'hello {name}'\n", encoding="utf-8"
    )
    (root / "README.md").write_text("# Demo project\n", encoding="utf-8")
    return root


@pytest.fixture
def fs(workspace):
    from agent3.workspace.fs import WorkspaceFS

    return WorkspaceFS(workspace)


@pytest.fixture
def runner(workspace):
    from agent3.workspace.terminal import CommandRunner

    return CommandRunner(workspace, default_timeout=30.0)


@pytest.fixture
def git_repo(workspace):
    from agent3.workspace.git_ops import GitRepo

    return GitRepo(workspace)


@pytest.fixture
def tool_context(fs, runner, git_repo):
    from agent3.agent.tools import ToolContext

    return ToolContext(fs=fs, runner=runner, git=git_repo, command_timeout=30.0)


@pytest.fixture
def usage_manager():
    from agent3.limits.manager import UsageManager

    manager = UsageManager()
    yield manager
    manager.close()


def requires_git() -> bool:
    from agent3.workspace.git_ops import GitRepo

    return GitRepo.git_available()


skip_without_git = pytest.mark.skipif(not requires_git(), reason="git is not installed")
skip_on_windows = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell only")
