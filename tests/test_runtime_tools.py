"""Tests for background processes, snapshots, syntax checking and the plan.

These four subsystems share one job: making a long autonomous run survivable.
A server that blocks the loop, a bad edit that cannot be taken back, a file
that does not parse and a plan the model forgets are the four ways a run goes
silently wrong, so each one gets its own guard here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from agent3.agent.tasks import STATUSES, Task, TaskError, TaskList, normalise_status, slugify
from agent3.agent.tools import ToolCall, ToolContext, ToolRegistry
from agent3.workspace.fs import WorkspaceFS
from agent3.workspace.git_ops import GitRepo
from agent3.workspace.processes import ProcessError, ProcessManager, format_log_lines
from agent3.workspace.snapshots import Snapshot, SnapshotStore
from agent3.workspace.syntax_check import check_file, check_source
from agent3.workspace.terminal import CommandRunner


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspaceFS:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
    return WorkspaceFS(tmp_path)


@pytest.fixture()
def ctx(workspace: WorkspaceFS) -> ToolContext:
    runner = CommandRunner(workspace.root, default_timeout=30)
    return ToolContext(
        fs=workspace,
        runner=runner,
        git=GitRepo(workspace.root),
        processes=ProcessManager(runner, max_processes=3),
        snapshots=SnapshotStore(),
        tasks=TaskList(),
        run_external_linters=False,
    )


def run(registry: ToolRegistry, ctx: ToolContext, name: str, **args):
    return registry.execute(ToolCall(name=name, args=args), ctx)


# ============================================================ syntax check
class TestSyntaxChecker:
    def test_valid_python_passes(self):
        result = check_source("a.py", "x = 1\n")
        assert result.checked and result.ok
        assert "python-ast" in result.tools_used

    def test_broken_python_reports_the_line(self):
        result = check_source("a.py", "def f(:\n    pass\n")
        assert result.checked and not result.ok
        assert result.errors[0].line == 1
        assert "SYNTAX ERROR" in result.summary()

    def test_unclosed_bracket_is_caught(self):
        result = check_source("a.py", "value = (1, 2\n")
        assert not result.ok

    def test_json_errors_carry_a_position(self):
        result = check_source("data.json", '{"a": 1,}')
        assert not result.ok
        assert result.errors[0].line == 1
        assert result.errors[0].source == "json"

    def test_valid_json_passes(self):
        assert check_source("data.json", json.dumps({"a": [1, 2]})).ok

    def test_empty_json_is_an_error(self):
        assert not check_source("data.json", "   ").ok

    def test_toml_is_parsed(self):
        assert check_source("p.toml", 'name = "x"\n').ok
        assert not check_source("p.toml", "name = \n").ok

    def test_yaml_tabs_are_rejected(self):
        result = check_source("c.yaml", "root:\n\tchild: 1\n")
        assert not result.ok

    def test_xml_is_parsed(self):
        assert check_source("f.xml", "<a><b/></a>").ok
        assert not check_source("f.xml", "<a><b></a>").ok

    def test_unknown_extension_is_skipped_not_failed(self):
        result = check_source("notes.txt", "anything at all (((")
        assert not result.checked
        assert result.ok  # a skip is never a failure
        assert "no checker" in result.skipped_reason

    def test_nul_bytes_are_rejected(self):
        assert not check_source("a.py", "x = 1\x00").ok

    def test_check_file_reads_from_disk(self, tmp_path: Path):
        (tmp_path / "broken.py").write_text("def f(\n", encoding="utf-8")
        result = check_file("broken.py", root=tmp_path, run_linters=False)
        assert not result.ok
        assert "broken.py" in result.report()

    def test_missing_file_is_skipped(self, tmp_path: Path):
        result = check_file("nope.py", root=tmp_path, run_linters=False)
        assert not result.checked and result.ok

    def test_report_lists_every_diagnostic(self):
        result = check_source("a.py", "def f(:\n")
        assert result.report().count("error") >= 1


# ====================================================== automatic checking
class TestAutomaticSyntaxCheck:
    def test_write_file_appends_the_verdict(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "write_file", path="good.py", content="x = 1\n")
        assert result.ok
        assert "syntax OK" in result.output
        assert result.data["syntax"]["ok"] is True

    def test_write_file_fails_on_broken_syntax(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "write_file", path="bad.py", content="def f(:\n")
        assert not result.ok
        assert "SYNTAX ERROR" in result.error
        # The file is still written - the model must see what it produced.
        assert ctx.fs.exists("bad.py")

    def test_edit_file_is_checked_too(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(
            registry, ctx, "edit_file", path="pkg/mod.py", search="return 1", replace="return ("
        )
        assert not result.ok
        assert "SYNTAX ERROR" in result.error

    def test_non_code_files_are_untouched(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "write_file", path="README.md", content="# hi ((( \n")
        assert result.ok
        assert "SYNTAX" not in result.output

    def test_checking_can_be_disabled(self, ctx: ToolContext):
        ctx.auto_syntax_check = False
        registry = ToolRegistry()
        result = run(registry, ctx, "write_file", path="bad2.py", content="def f(:\n")
        assert result.ok
        assert "syntax" not in result.data

    def test_check_syntax_tool_on_a_file(self, ctx: ToolContext):
        registry = ToolRegistry()
        ctx.fs.write_text("x.py", "def f(:\n")
        result = run(registry, ctx, "check_syntax", path="x.py")
        assert not result.ok
        assert "line 1" in result.error

    def test_check_syntax_tool_on_a_snippet(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "check_syntax", path="snippet.py", content="y = [1,2]\n")
        assert result.ok

    def test_check_syntax_needs_an_existing_file(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "check_syntax", path="ghost.py")
        assert not result.ok and "does not exist" in result.error


# ================================================================ undo
class TestSnapshotStore:
    def test_records_and_pops_lifo(self):
        store = SnapshotStore()
        store.record("a.py", "one", "write_file")
        store.record("b.py", "two", "edit_file")
        assert len(store) == 2
        assert store.pop().path == "b.py"
        assert store.pop().path == "a.py"
        assert store.pop() is None

    def test_pop_can_target_a_path(self):
        store = SnapshotStore()
        store.record("a.py", "one", "write_file")
        store.record("b.py", "two", "write_file")
        assert store.pop("a.py").content == "one"
        assert len(store) == 1

    def test_absent_files_are_recorded_as_none(self):
        store = SnapshotStore()
        snapshot = store.record("new.py", None, "write_file")
        assert snapshot.existed is False
        assert "absent" in snapshot.describe()

    def test_entry_budget_is_enforced(self):
        store = SnapshotStore(max_entries=3)
        for index in range(10):
            store.record(f"f{index}.py", "x", "write_file")
        assert len(store) == 3
        assert store.peek().path == "f9.py"

    def test_byte_budget_evicts_old_entries(self):
        store = SnapshotStore(max_entries=100, max_bytes=100)
        store.record("big.py", "x" * 200, "write_file")
        store.record("small.py", "y", "write_file")
        assert len(store) == 1
        assert store.peek().path == "small.py"

    def test_history_is_newest_first(self):
        store = SnapshotStore()
        store.record("a.py", "1", "write_file")
        store.record("a.py", "2", "edit_file")
        history = store.history("a.py")
        assert [entry.content for entry in history] == ["2", "1"]


class TestUndoTool:
    def test_undo_restores_the_previous_content(self, ctx: ToolContext):
        registry = ToolRegistry()
        original = ctx.fs.read_text("pkg/mod.py")
        run(registry, ctx, "write_file", path="pkg/mod.py", content="def ok():\n    return 2\n")
        assert "return 2" in ctx.fs.read_text("pkg/mod.py")

        result = run(registry, ctx, "undo_file_change")
        assert result.ok
        assert ctx.fs.read_text("pkg/mod.py") == original
        assert "restored" in result.output

    def test_undo_deletes_a_file_that_did_not_exist(self, ctx: ToolContext):
        registry = ToolRegistry()
        run(registry, ctx, "write_file", path="fresh.py", content="x = 1\n")
        assert ctx.fs.exists("fresh.py")
        result = run(registry, ctx, "undo_file_change", path="fresh.py")
        assert result.ok
        assert not ctx.fs.exists("fresh.py")

    def test_undo_can_target_one_file(self, ctx: ToolContext):
        registry = ToolRegistry()
        run(registry, ctx, "write_file", path="a.py", content="a = 1\n")
        run(registry, ctx, "write_file", path="b.py", content="b = 1\n")
        run(registry, ctx, "undo_file_change", path="a.py")
        assert not ctx.fs.exists("a.py")
        assert ctx.fs.exists("b.py")

    def test_undo_is_repeatable(self, ctx: ToolContext):
        registry = ToolRegistry()
        original = ctx.fs.read_text("pkg/mod.py")
        run(registry, ctx, "write_file", path="pkg/mod.py", content="x = 1\n")
        run(registry, ctx, "write_file", path="pkg/mod.py", content="x = 2\n")
        run(registry, ctx, "undo_file_change")
        assert ctx.fs.read_text("pkg/mod.py") == "x = 1\n"
        run(registry, ctx, "undo_file_change")
        assert ctx.fs.read_text("pkg/mod.py") == original

    def test_undo_with_nothing_recorded_fails_clearly(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "undo_file_change")
        assert not result.ok and "nothing to undo" in result.error

    def test_undo_restores_a_deleted_file(self, ctx: ToolContext):
        registry = ToolRegistry()
        run(registry, ctx, "delete_file", path="pkg/mod.py")
        assert not ctx.fs.exists("pkg/mod.py")
        result = run(registry, ctx, "undo_file_change")
        assert result.ok
        assert "return 1" in ctx.fs.read_text("pkg/mod.py")

    def test_history_listing_changes_nothing(self, ctx: ToolContext):
        registry = ToolRegistry()
        run(registry, ctx, "write_file", path="c.py", content="c = 1\n")
        result = run(registry, ctx, "undo_file_change", list=True)
        assert result.ok and "undo history" in result.output
        assert ctx.fs.exists("c.py")

    def test_rollback_file_is_an_accepted_alias(self, ctx: ToolContext):
        registry = ToolRegistry()
        assert registry.get("rollback_file").name == "undo_file_change"


# ========================================================== process manager
@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell semantics")
class TestProcessManager:
    def test_start_capture_and_stop(self, ctx: ToolContext):
        manager = ctx.processes
        process = manager.start("for i in 1 2 3; do echo line-$i; sleep 0.2; done; sleep 30")
        assert process.wait_for_log("line-3", timeout=10)
        assert process.running
        rows = process.logs(10)
        assert any("line-1" in row.text for row in rows)

        info = manager.stop(process.process_id)
        assert not info.running

    def test_exiting_command_is_reaped(self, ctx: ToolContext):
        manager = ctx.processes
        process = manager.start("echo hello", wait_timeout=2)
        time.sleep(0.4)
        assert process.poll() is not None
        assert any("hello" in row.text for row in process.logs(10))

    def test_stderr_is_tagged(self, ctx: ToolContext):
        manager = ctx.processes
        process = manager.start("echo oops 1>&2; sleep 5", wait_for="oops", wait_timeout=5)
        rows = process.logs(10, stream="stderr")
        assert rows and rows[0].text.strip() == "oops"
        assert "[stderr]" in format_log_lines(rows)
        manager.stop(process.process_id)

    def test_unknown_id_is_reported(self, ctx: ToolContext):
        with pytest.raises(ProcessError):
            ctx.processes.get("proc-999")

    def test_concurrency_cap(self, ctx: ToolContext):
        manager = ctx.processes
        for _ in range(3):
            manager.start("sleep 20", wait_timeout=0)
        with pytest.raises(ProcessError) as excinfo:
            manager.start("sleep 20", wait_timeout=0)
        assert "too many background processes" in str(excinfo.value)
        manager.stop_all()

    def test_interactive_commands_are_refused(self, ctx: ToolContext):
        with pytest.raises(ProcessError) as excinfo:
            ctx.processes.start("npm init")
        assert "interactive" in str(excinfo.value)

    def test_blocked_commands_are_refused(self, ctx: ToolContext):
        with pytest.raises(Exception):
            ctx.processes.start("rm -rf /")

    def test_ring_buffer_is_bounded(self, ctx: ToolContext):
        manager = ProcessManager(ctx.runner, max_log_lines=5)
        process = manager.start("for i in $(seq 1 40); do echo row-$i; done; sleep 5", wait_timeout=2)
        time.sleep(0.6)
        assert len(process.logs(100)) <= 5
        manager.stop_all()

    def test_stop_all_reports_what_it_killed(self, ctx: ToolContext):
        ctx.processes.start("sleep 20", wait_timeout=0)
        ctx.processes.start("sleep 20", wait_timeout=0)
        stopped = ctx.processes.stop_all()
        assert len(stopped) == 2
        assert ctx.processes.running_count == 0


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell semantics")
class TestProcessTools:
    def test_start_then_read_then_stop(self, ctx: ToolContext):
        registry = ToolRegistry()
        started = run(
            registry,
            ctx,
            "start_process",
            command="for i in 1 2 3; do echo tick-$i; sleep 0.2; done; sleep 30",
            wait_for="tick-1",
            wait_timeout=8,
        )
        assert started.ok
        process_id = started.data["process_id"]
        assert started.data["running"] is True

        logs = run(registry, ctx, "get_process_logs", process_id=process_id, wait_for="tick-3")
        assert logs.ok and "tick-3" in logs.output

        stopped = run(registry, ctx, "stop_process", process_id=process_id)
        assert stopped.ok and "stopped" in stopped.output
        assert ctx.processes.running_count == 0

    def test_logs_without_an_id_lists_processes(self, ctx: ToolContext):
        registry = ToolRegistry()
        run(registry, ctx, "start_process", command="sleep 20", wait_timeout=0)
        listing = run(registry, ctx, "get_process_logs")
        assert listing.ok
        assert "background process" in listing.output
        assert len(listing.data["processes"]) == 1
        run(registry, ctx, "stop_process", process_id="all")

    def test_immediate_exit_is_reported_as_such(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "start_process", command="echo done", wait_timeout=2)
        assert "exited immediately" in result.output

    def test_failing_command_is_not_ok(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "start_process", command="exit 3", wait_timeout=2)
        assert not result.ok

    def test_unknown_process_id(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "get_process_logs", process_id="proc-42")
        assert not result.ok and "unknown process id" in result.error

    def test_stop_all_when_nothing_runs(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "stop_process", process_id="all")
        assert result.ok and "no background process" in result.output

    def test_interactive_command_is_refused_by_the_tool(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "start_process", command="vim notes.txt")
        assert not result.ok and "interactive" in result.error


# ================================================================== tasks
class TestTaskList:
    def test_add_derives_a_readable_id(self):
        tasks = TaskList()
        task = tasks.add("Write the migration script")
        assert task.id == "write-the-migration-script"
        assert task.status == "pending"

    def test_duplicate_ids_are_disambiguated(self):
        tasks = TaskList()
        tasks.add("Fix bug", task_id="fix")
        second = tasks.add("Fix other bug", task_id="fix")
        assert second.id == "fix-2"

    def test_update_changes_status(self):
        tasks = TaskList()
        tasks.add("Step one", task_id="one")
        tasks.update("one", status="in_progress")
        assert tasks.get("one").status == "in_progress"
        tasks.update("one", status="done")  # alias
        assert tasks.get("one").status == "completed"

    def test_unknown_id_raises(self):
        with pytest.raises(TaskError):
            TaskList().require("ghost")

    def test_invalid_status_raises(self):
        with pytest.raises(TaskError):
            normalise_status("almost")

    def test_status_aliases(self):
        assert normalise_status("WIP") == "in_progress"
        assert normalise_status("Done") == "completed"
        assert normalise_status("in-progress") == "in_progress"

    def test_current_prefers_in_progress(self):
        tasks = TaskList()
        tasks.add("a", task_id="a")
        tasks.add("b", task_id="b", status="in_progress")
        assert tasks.current.id == "b"

    def test_current_falls_back_to_the_first_pending(self):
        tasks = TaskList()
        tasks.add("a", task_id="a", status="completed")
        tasks.add("b", task_id="b")
        assert tasks.current.id == "b"

    def test_progress_and_all_done(self):
        tasks = TaskList()
        tasks.add("a", task_id="a", status="completed")
        tasks.add("b", task_id="b", status="cancelled")
        assert tasks.progress() == (2, 2)
        assert tasks.all_done
        assert tasks.open_tasks == []

    def test_replace_wipes_the_plan(self):
        tasks = TaskList()
        tasks.add("old", task_id="old")
        tasks.replace([{"id": "new", "description": "new work"}])
        assert [t.id for t in tasks] == ["new"]

    def test_render_shows_checkboxes(self):
        tasks = TaskList()
        tasks.add("one", task_id="one", status="completed")
        tasks.add("two", task_id="two")
        rendered = tasks.render()
        assert "[x] one" in rendered and "[ ] two" in rendered
        assert "1/2 done" in rendered

    def test_empty_render_is_explicit(self):
        assert "empty" in TaskList().render()

    def test_markdown_strikes_completed_items(self):
        tasks = TaskList()
        tasks.add("done thing", task_id="d", status="completed")
        assert "~~done thing~~" in tasks.render_markdown()

    def test_slugify_handles_punctuation(self):
        assert slugify("Fix: the (broken) thing!") == "fix-the-broken-thing"

    def test_every_status_has_a_glyph(self):
        for status in STATUSES:
            assert Task(id="x", description="y", status=status).glyph


class TestTaskTool:
    def test_add_then_list(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(
            registry,
            ctx,
            "manage_tasks",
            action="add",
            tasks=[
                {"id": "read", "description": "Read the module"},
                {"id": "edit", "description": "Apply the fix"},
            ],
        )
        assert result.ok
        assert result.data["total"] == 2
        listed = run(registry, ctx, "manage_tasks", action="list")
        assert "Read the module" in listed.output

    def test_update_marks_progress(self, ctx: ToolContext):
        registry = ToolRegistry()
        run(registry, ctx, "manage_tasks", action="add", tasks=[{"id": "a", "description": "A"}])
        result = run(
            registry,
            ctx,
            "manage_tasks",
            action="update",
            tasks=[{"id": "a", "status": "completed"}],
        )
        assert result.ok
        assert ctx.tasks.get("a").status == "completed"
        assert result.data["done"] == 1

    def test_update_without_id_is_rejected(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(
            registry, ctx, "manage_tasks", action="update", tasks=[{"status": "completed"}]
        )
        assert not result.ok and "needs an 'id'" in result.error

    def test_unknown_id_is_rejected(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(
            registry, ctx, "manage_tasks", action="update", tasks=[{"id": "nope", "status": "done"}]
        )
        assert not result.ok and "unknown task id" in result.error

    def test_plain_strings_become_tasks(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "manage_tasks", action="add", tasks=["first step", "second"])
        assert result.ok and result.data["total"] == 2

    def test_flat_arguments_are_accepted(self, ctx: ToolContext):
        registry = ToolRegistry()
        run(registry, ctx, "manage_tasks", action="add", description="Only step")
        assert len(ctx.tasks) == 1

    def test_unknown_action_is_rejected(self, ctx: ToolContext):
        registry = ToolRegistry()
        result = run(registry, ctx, "manage_tasks", action="teleport")
        assert not result.ok and "unknown action" in result.error

    def test_callback_fires_on_change(self, ctx: ToolContext):
        seen = []
        ctx.on_tasks_changed = seen.append
        registry = ToolRegistry()
        run(registry, ctx, "manage_tasks", action="add", tasks=["x"])
        assert len(seen) == 1

    def test_todo_list_alias(self, ctx: ToolContext):
        registry = ToolRegistry()
        assert registry.get("todo_list").name == "manage_tasks"
        result = registry.execute(
            ToolCall(name="todo_list", args={"action": "add", "tasks": ["via alias"]}), ctx
        )
        assert result.ok and len(ctx.tasks) == 1


# ============================================================== catalogue
def test_every_new_tool_is_registered():
    registry = ToolRegistry()
    for name in (
        "start_process",
        "get_process_logs",
        "stop_process",
        "manage_tasks",
        "check_syntax",
        "undo_file_change",
    ):
        assert registry.get(name) is not None, name
    assert len(registry.names()) == 20


def test_prompt_catalogue_hides_aliases():
    described = ToolRegistry().describe()
    assert "manage_tasks(" in described
    assert "todo_list(" not in described
