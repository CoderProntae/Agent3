"""Tests for the sandboxed filesystem, diffing, terminal and git wrappers."""

from __future__ import annotations

import sys

import pytest

from agent3.workspace.diffing import (
    EditError,
    apply_search_replace,
    diff_stats,
    make_unified_diff,
    parse_unified_diff,
    side_by_side,
)
from agent3.workspace.fs import WorkspaceFS, WorkspaceSecurityError
from agent3.workspace.terminal import CommandBlockedError, CommandRunner

from conftest import skip_without_git


class TestWorkspaceSandbox:
    def test_resolve_relative(self, fs):
        assert fs.resolve("src/app.py").is_file()

    def test_rejects_parent_traversal(self, fs):
        with pytest.raises(WorkspaceSecurityError):
            fs.resolve("../../etc/passwd")

    def test_rejects_absolute_outside(self, fs):
        with pytest.raises(WorkspaceSecurityError):
            fs.resolve("/etc/passwd")

    def test_accepts_absolute_inside(self, fs):
        assert fs.resolve(str(fs.root / "README.md")).name == "README.md"

    def test_refuses_to_delete_root(self, fs):
        with pytest.raises(WorkspaceSecurityError):
            fs.delete(".", recursive=True)


class TestWorkspaceFiles:
    def test_read_write_round_trip(self, fs):
        fs.write_text("notes/todo.md", "- first\n")
        assert fs.read_text("notes/todo.md") == "- first\n"

    def test_write_creates_parent_directories(self, fs):
        fs.write_text("a/b/c/deep.txt", "x")
        assert fs.exists("a/b/c/deep.txt")

    def test_read_lines_range(self, fs):
        fs.write_text("nums.txt", "\n".join(str(i) for i in range(1, 11)) + "\n")
        chunk = fs.read_lines("nums.txt", 3, 5)
        assert chunk.splitlines()[0].strip().startswith("3 |")
        assert len(chunk.splitlines()) == 3

    def test_rename_and_delete(self, fs):
        fs.write_text("one.txt", "1")
        fs.rename("one.txt", "two.txt")
        assert not fs.exists("one.txt") and fs.exists("two.txt")
        fs.delete("two.txt")
        assert not fs.exists("two.txt")

    def test_delete_directory_requires_recursive(self, fs):
        fs.mkdir("folder")
        fs.write_text("folder/x.txt", "x")
        with pytest.raises(Exception):
            fs.delete("folder")
        fs.delete("folder", recursive=True)
        assert not fs.exists("folder")

    def test_tree_text_contains_files(self, fs):
        text = fs.tree_text()
        assert "src" in text and "app.py" in text

    def test_ignores_noise_directories(self, fs):
        fs.mkdir("node_modules/pkg")
        fs.write_text("node_modules/pkg/index.js", "x")
        assert "node_modules" not in fs.tree_text()

    def test_binary_detection(self, fs):
        (fs.root / "blob.bin").write_bytes(b"\x00\x01\x02binary")
        assert fs.is_binary("blob.bin") is True
        assert fs.is_binary("README.md") is False

    def test_search_text(self, fs):
        hits = fs.search_text("greet")
        assert hits and hits[0]["path"] == "src/app.py"
        assert hits[0]["line"] == 1

    def test_search_regex(self, fs):
        hits = fs.search_text(r"def\s+\w+", regex=True, glob="*.py")
        assert len(hits) == 1

    def test_find_files(self, fs):
        assert "src/app.py" in fs.find_files("*.py")

    def test_truncates_huge_reads(self, workspace):
        small = WorkspaceFS(workspace, max_read_bytes=20)
        small.write_text("big.txt", "x" * 500)
        assert "truncated" in small.read_text("big.txt")

    def test_summarize(self, fs):
        summary = fs.summarize()
        assert summary["files"] >= 2
        assert ".py" in summary["extensions"]


class TestDiffing:
    def test_unified_diff_and_stats(self):
        unified = make_unified_diff("a\nb\nc\n", "a\nB\nc\n", "f.txt")
        stats = diff_stats(unified)
        assert stats.added == 1 and stats.removed == 1 and stats.hunks == 1

    def test_parse_unified_diff_rows(self):
        rows = parse_unified_diff(make_unified_diff("a\n", "b\n", "f.txt"))
        kinds = {row.kind for row in rows}
        assert {"added", "removed", "header", "hunk"} <= kinds

    def test_side_by_side(self):
        rows = side_by_side("a\nb\n", "a\nc\n")
        assert rows[0][0] == "equal"
        assert rows[1][0] == "replace"

    def test_search_replace_exact(self):
        assert apply_search_replace("x = 1\ny = 2\n", "y = 2", "y = 3") == "x = 1\ny = 3\n"

    def test_search_replace_ambiguous_raises(self):
        with pytest.raises(EditError):
            apply_search_replace("a\na\n", "a", "b")

    def test_search_replace_all_occurrences(self):
        assert apply_search_replace("a\na\n", "a", "b", count=0) == "b\nb\n"

    def test_search_replace_missing_raises(self):
        with pytest.raises(EditError):
            apply_search_replace("hello\n", "goodbye", "x")

    def test_search_replace_whitespace_tolerant(self):
        content = "def f():\n        return 1\n"
        updated = apply_search_replace(content, "def f():\nreturn 1", "def f():\n    return 2")
        assert "return 2" in updated


class TestCommandRunner:
    def test_runs_and_captures_stdout(self, runner):
        result = runner.run(f'"{sys.executable}" -c "print(6*7)"')
        assert result.ok and "42" in result.stdout

    def test_captures_exit_code_and_stderr(self, runner):
        result = runner.run(f'"{sys.executable}" -c "import sys; sys.stderr.write(\'boom\'); sys.exit(3)"')
        assert result.exit_code == 3
        assert "boom" in result.stderr
        assert result.ok is False

    def test_timeout(self, runner):
        result = runner.run(f'"{sys.executable}" -c "import time; time.sleep(10)"', timeout=1.0)
        assert result.timed_out is True
        assert result.exit_code == 124

    def test_blocked_command(self, runner):
        result = runner.run("rm -rf /")
        assert result.blocked is True
        assert result.exit_code == 126

    def test_validate_raises(self, runner):
        with pytest.raises(CommandBlockedError):
            runner.validate("mkfs.ext4 /dev/sda")

    def test_cwd_escape_blocked(self, runner):
        result = runner.run("echo hi", cwd="../..")
        assert result.blocked is True

    def test_runs_inside_workspace(self, runner, workspace):
        result = runner.run(f'"{sys.executable}" -c "import os; print(os.getcwd())"')
        assert str(workspace.resolve()) in result.stdout

    def test_streaming_callback(self, runner):
        chunks = []
        runner.run(
            f'"{sys.executable}" -c "print(\'line1\'); print(\'line2\')"',
            on_output=lambda stream, text: chunks.append((stream, text)),
        )
        assert any("line1" in text for _stream, text in chunks)

    def test_summary_formatting(self, runner):
        summary = runner.run(f'"{sys.executable}" -c "print(1)"').summary()
        assert "status: OK" in summary

    def test_custom_blocked_patterns(self, workspace):
        custom = CommandRunner(workspace, blocked_patterns=[r"\bnpm\b"])
        assert custom.run("npm install").blocked is True
        assert custom.run("echo safe").blocked is False


@skip_without_git
class TestGitRepo:
    def test_init_and_commit(self, git_repo, fs):
        assert git_repo.is_repo() is False
        git_repo.init(default_branch="main")
        assert git_repo.is_repo() is True
        fs.write_text("file.txt", "hello\n")
        assert git_repo.is_dirty() is True
        result = git_repo.commit("initial commit", stage_all=True)
        assert result.ok, result.stderr
        assert git_repo.is_dirty() is False
        assert git_repo.log(1)[0]["subject"] == "initial commit"

    def test_status_parsing(self, git_repo, fs):
        git_repo.init()
        fs.write_text("new.txt", "x")
        statuses = {entry.path: entry.label for entry in git_repo.status()}
        assert statuses.get("new.txt") == "untracked"

    def test_diff_after_change(self, git_repo, fs):
        git_repo.init()
        fs.write_text("code.py", "a = 1\n")
        git_repo.commit("add", stage_all=True)
        fs.write_text("code.py", "a = 2\n")
        assert "-a = 1" in git_repo.diff()

    def test_branch_management(self, git_repo, fs):
        git_repo.init()
        fs.write_text("a.txt", "a")
        git_repo.commit("init", stage_all=True)
        git_repo.create_branch("feature/x")
        assert git_repo.current_branch() == "feature/x"
        assert "feature/x" in git_repo.branches()

    def test_summary(self, git_repo, fs):
        git_repo.init()
        fs.write_text("a.txt", "a")
        git_repo.commit("init", stage_all=True)
        summary = git_repo.summary()
        assert summary["repo"] is True
        assert summary["dirty"] is False
