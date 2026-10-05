"""Tests for tool parsing, tool execution, the agent loop and sessions."""

from __future__ import annotations

import json
import sys
from typing import List, Optional, Sequence

import pytest

from agent3.agent.loop import AgentCallbacks, AgentLoop, AgentStopReason
from agent3.agent.session import SessionStore
from agent3.agent.tools import ToolCall, ToolRegistry, parse_tool_calls, strip_tool_calls
from agent3.core.config import AgentSettings
from agent3.llm.messages import ChatMessage, Role, Usage

from conftest import skip_without_git


# --------------------------------------------------------------------- fakes
class FakeClient:
    """Replays a scripted list of assistant messages."""

    def __init__(self, responses: Sequence[str]):
        self._responses = list(responses)
        self.calls: List[List[ChatMessage]] = []
        self.settings = type("S", (), {"model": "fake-model"})()

    def chat(self, messages, *, model=None, options=None, cancel=None, on_delta=None, format_json=False):
        self.calls.append(list(messages))
        text = self._responses.pop(0) if self._responses else '```json\n{"tool":"finish","args":{"summary":"done"}}\n```'
        if on_delta:
            on_delta(text)
        return text, Usage(prompt_tokens=10, completion_tokens=5, total_duration_ms=50)


def tool_block(name: str, **args) -> str:
    return f"Doing {name}.\n```json\n{json.dumps({'tool': name, 'args': args})}\n```"


# -------------------------------------------------------------------- parsing
class TestToolParsing:
    def test_fenced_json_block(self):
        calls = parse_tool_calls(tool_block("read_file", path="a.py"))
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].args == {"path": "a.py"}

    def test_bare_json_object(self):
        calls = parse_tool_calls('I will act. {"tool": "list_files", "args": {"path": "src"}}')
        assert calls[0].name == "list_files"

    def test_openai_function_shape(self):
        text = '```json\n{"function": {"name": "read_file", "arguments": {"path": "x.py"}}}\n```'
        assert parse_tool_calls(text)[0].args["path"] == "x.py"

    def test_tool_calls_wrapper(self):
        text = '```json\n{"tool_calls": [{"tool": "git", "args": {"action": "status"}}]}\n```'
        assert parse_tool_calls(text)[0].name == "git"

    def test_string_arguments_are_parsed(self):
        text = '```json\n{"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\"}"}\n```'
        assert parse_tool_calls(text)[0].args["path"] == "a.py"

    def test_inline_arguments(self):
        text = '```json\n{"tool": "read_file", "path": "inline.py"}\n```'
        assert parse_tool_calls(text)[0].args["path"] == "inline.py"

    def test_unknown_tool_filtered(self):
        assert parse_tool_calls(tool_block("not_a_tool"), ToolRegistry().names()) == []

    def test_plain_prose_yields_nothing(self):
        assert parse_tool_calls("Just talking about {braces} but no json.") == []

    def test_duplicates_removed(self):
        text = tool_block("read_file", path="a.py") + "\n" + tool_block("read_file", path="a.py")
        assert len(parse_tool_calls(text)) == 1

    def test_strip_tool_calls(self):
        assert strip_tool_calls(tool_block("read_file", path="a.py")) == "Doing read_file."

    def test_signature_is_stable(self):
        a = ToolCall("x", {"b": 1, "a": 2})
        b = ToolCall("x", {"a": 2, "b": 1})
        assert a.signature() == b.signature()


# ------------------------------------------------------------------ execution
class TestToolExecution:
    def setup_method(self):
        self.registry = ToolRegistry()

    def run(self, ctx, name, **args):
        return self.registry.execute(ToolCall(name, args), ctx)

    def test_catalogue_is_documented(self):
        description = self.registry.describe()
        for name in ("read_file", "write_file", "edit_file", "run_command", "git", "finish"):
            assert name in description

    def test_write_file_creates_and_diffs(self, tool_context):
        result = self.run(tool_context, "write_file", path="new.py", content="print(1)\n")
        assert result.ok
        assert tool_context.fs.read_text("new.py") == "print(1)\n"
        assert "+print(1)" in result.diff

    def test_write_file_requires_content(self, tool_context):
        assert self.run(tool_context, "write_file", path="x.py").ok is False

    def test_read_file(self, tool_context):
        result = self.run(tool_context, "read_file", path="src/app.py")
        assert result.ok and "def greet" in result.output

    def test_read_missing_file(self, tool_context):
        result = self.run(tool_context, "read_file", path="nope.py")
        assert result.ok is False and "not found" in result.error.lower()

    def test_edit_file(self, tool_context):
        result = self.run(
            tool_context, "edit_file", path="src/app.py", search="hello", replace="hi"
        )
        assert result.ok
        assert "hi {name}" in tool_context.fs.read_text("src/app.py")

    def test_edit_file_missing_anchor(self, tool_context):
        result = self.run(tool_context, "edit_file", path="src/app.py", search="zzz", replace="x")
        assert result.ok is False and "not found" in result.error

    def test_sandbox_violation_is_reported(self, tool_context):
        result = self.run(tool_context, "write_file", path="../escape.txt", content="x")
        assert result.ok is False and "security violation" in result.error

    def test_list_files(self, tool_context):
        assert "app.py" in self.run(tool_context, "list_files").output

    def test_search_code(self, tool_context):
        assert "src/app.py" in self.run(tool_context, "search_code", query="greet").output

    def test_run_command(self, tool_context):
        result = self.run(tool_context, "run_command", command=f'"{sys.executable}" -c "print(99)"')
        assert result.ok and "99" in result.output

    def test_run_command_failure_is_not_ok(self, tool_context):
        result = self.run(tool_context, "run_command", command=f'"{sys.executable}" -c "exit(2)"')
        assert result.ok is False and "exit 2" in result.error

    def test_blocked_command(self, tool_context):
        result = self.run(tool_context, "run_command", command="rm -rf /")
        assert result.ok is False

    def test_delete_and_rename(self, tool_context):
        self.run(tool_context, "write_file", path="tmp.txt", content="x")
        assert self.run(tool_context, "rename_file", source="tmp.txt", destination="tmp2.txt").ok
        assert self.run(tool_context, "delete_file", path="tmp2.txt").ok
        assert tool_context.fs.exists("tmp2.txt") is False

    def test_make_directory(self, tool_context):
        assert self.run(tool_context, "make_directory", path="pkg/sub").ok
        assert tool_context.fs.is_dir("pkg/sub")

    def test_project_overview(self, tool_context):
        assert "files:" in self.run(tool_context, "project_overview").output

    def test_unknown_tool(self, tool_context):
        result = self.registry.execute(ToolCall("nope", {}), tool_context)
        assert result.ok is False and "unknown tool" in result.error

    def test_missing_required_argument(self, tool_context):
        result = self.run(tool_context, "read_file")
        assert result.ok is False and "missing required" in result.error

    def test_file_changed_callback(self, tool_context):
        seen = []
        tool_context.on_file_changed = lambda path, diff: seen.append(path)
        self.run(tool_context, "write_file", path="cb.txt", content="x")
        assert seen == ["cb.txt"]

    @skip_without_git
    def test_git_tool_flow(self, tool_context):
        assert self.run(tool_context, "git", action="init").ok
        self.run(tool_context, "write_file", path="g.txt", content="x")
        assert self.run(tool_context, "git", action="add", paths=".").ok
        assert self.run(tool_context, "git", action="commit", message="test commit").ok
        assert "test commit" in self.run(tool_context, "git", action="log").output

    def test_git_commit_requires_message(self, tool_context):
        assert self.run(tool_context, "git", action="commit").ok is False


# ----------------------------------------------------------------------- loop
class TestAgentLoop:
    def build(self, responses, usage_manager, tool_context, **settings):
        client = FakeClient(responses)
        loop = AgentLoop(
            client,
            ToolRegistry(),
            tool_context,
            usage_manager,
            AgentSettings(max_iterations=settings.pop("max_iterations", 8), **settings),
            model="fake-model",
        )
        return client, loop

    def test_single_write_then_finish(self, usage_manager, tool_context):
        _client, loop = self.build(
            [
                tool_block("write_file", path="out.py", content="print('x')\n"),
                tool_block("finish", summary="Created out.py"),
            ],
            usage_manager,
            tool_context,
        )
        result = loop.run("create out.py")
        assert result.stop_reason is AgentStopReason.FINISHED
        assert result.final_message == "Created out.py"
        assert result.tool_calls == 1
        assert tool_context.fs.read_text("out.py") == "print('x')\n"

    def test_usage_is_recorded(self, usage_manager, tool_context):
        _client, loop = self.build([tool_block("finish", summary="ok")], usage_manager, tool_context)
        result = loop.run("hello")
        assert result.prompt_tokens == 10 and result.completion_tokens == 5
        assert usage_manager.store.totals_for_day().requests == 1

    def test_self_correction_after_tool_error(self, usage_manager, tool_context):
        _client, loop = self.build(
            [
                tool_block("read_file", path="missing.py"),
                tool_block("write_file", path="missing.py", content="ok\n"),
                tool_block("finish", summary="recovered"),
            ],
            usage_manager,
            tool_context,
        )
        result = loop.run("read the file")
        assert result.stop_reason is AgentStopReason.FINISHED
        assert result.tool_calls == 2
        assert tool_context.fs.exists("missing.py")

    def test_retry_hint_is_injected(self, usage_manager, tool_context):
        client, loop = self.build(
            [tool_block("read_file", path="missing.py"), tool_block("finish", summary="done")],
            usage_manager,
            tool_context,
        )
        loop.run("go")
        second_prompt = client.calls[1][-1].content
        assert "ERROR" in second_prompt and "corrected tool call" in second_prompt

    def test_max_iterations_guard(self, usage_manager, tool_context):
        responses = [tool_block("list_files", path=".") for _ in range(20)]
        _client, loop = self.build(responses, usage_manager, tool_context, max_iterations=3)
        result = loop.run("loop forever")
        assert result.stop_reason is AgentStopReason.MAX_ITERATIONS
        assert result.iterations == 3

    def test_loop_hint_after_repeats(self, usage_manager, tool_context):
        responses = [tool_block("list_files", path=".") for _ in range(6)]
        client, loop = self.build(responses, usage_manager, tool_context, max_iterations=5)
        loop.run("repeat")
        assert any("fundamentally different approach" in m.content for m in client.calls[-1])

    def test_plain_answer_terminates(self, usage_manager, tool_context):
        _client, loop = self.build(
            [tool_block("list_files"), "The project only contains a README."],
            usage_manager,
            tool_context,
        )
        result = loop.run("what is in here?")
        assert result.stop_reason is AgentStopReason.NO_TOOL_CALL
        assert "README" in result.final_message

    def test_quota_block_stops_the_run(self, usage_manager, tool_context):
        policy = usage_manager.policies.load()
        policy.max_requests_per_day = 1
        usage_manager.policies.save(policy)
        usage_manager.record_request(model="m", prompt_tokens=1, completion_tokens=1, duration_ms=1)
        _client, loop = self.build([tool_block("finish", summary="x")], usage_manager, tool_context)
        result = loop.run("do something")
        assert result.stop_reason is AgentStopReason.QUOTA_BLOCKED
        assert "Daily request limit" in result.error

    def test_tool_call_quota_stops_the_run(self, usage_manager, tool_context):
        policy = usage_manager.policies.load()
        policy.max_tool_calls_per_run = 1
        usage_manager.policies.save(policy)
        _client, loop = self.build(
            [tool_block("list_files"), tool_block("list_files"), tool_block("finish", summary="x")],
            usage_manager,
            tool_context,
        )
        result = loop.run("go")
        assert result.stop_reason is AgentStopReason.QUOTA_BLOCKED

    def test_callbacks_fire(self, usage_manager, tool_context):
        events = {"status": 0, "tools": [], "messages": []}
        _client, loop = self.build(
            [tool_block("write_file", path="a.txt", content="a"), tool_block("finish", summary="ok")],
            usage_manager,
            tool_context,
        )
        loop.callbacks = AgentCallbacks(
            on_status=lambda _t: events.__setitem__("status", events["status"] + 1),
            on_tool_start=lambda call: events["tools"].append(call.name),
            on_assistant_message=lambda text: events["messages"].append(text),
        )
        loop.run("go")
        assert events["status"] > 0
        assert events["tools"] == ["write_file", "finish"]
        assert events["messages"]

    def test_cancellation(self, usage_manager, tool_context):
        _client, loop = self.build([tool_block("list_files")] * 5, usage_manager, tool_context)
        # Simulate the user pressing "Stop" while the first tool is running.
        loop.callbacks = AgentCallbacks(on_tool_start=lambda _call: loop.cancel())
        result = loop.run("go")
        assert result.stop_reason is AgentStopReason.CANCELLED
        assert result.tool_calls == 1

    def test_system_prompt_contains_workspace_and_tools(self, usage_manager, tool_context):
        _client, loop = self.build([], usage_manager, tool_context)
        prompt = loop.system_prompt()
        assert str(tool_context.fs.root) in prompt
        assert "write_file" in prompt and "finish" in prompt
        assert "app.py" in prompt


# ------------------------------------------------------------------- sessions
class TestSessionStore:
    def test_create_and_list(self):
        store = SessionStore()
        session = store.create(title="First", workspace="/tmp")
        assert [s.id for s in store.list()] == [session.id]
        store.close()

    def test_messages_round_trip(self):
        store = SessionStore()
        session = store.create()
        store.add_message(session.id, Role.USER, "hello")
        store.add_message(session.id, Role.ASSISTANT, "hi", {"tool": "none"})
        messages = store.messages(session.id)
        assert [m.role for m in messages] == [Role.USER, Role.ASSISTANT]
        assert messages[1].metadata == {"tool": "none"}
        store.close()

    def test_rename_and_delete(self):
        store = SessionStore()
        session = store.create()
        store.rename(session.id, "Renamed")
        assert store.get(session.id).title == "Renamed"
        store.delete(session.id)
        assert store.get(session.id) is None
        store.close()

    def test_autoname(self):
        store = SessionStore()
        session = store.create()
        title = store.autoname(session.id, "Build a REST API with FastAPI and tests")
        assert title.startswith("Build a REST API")
        store.close()

    def test_export_markdown(self):
        store = SessionStore()
        session = store.create(title="Export me")
        store.add_message(session.id, Role.USER, "question")
        markdown = store.export_markdown(session.id)
        assert "# Export me" in markdown and "question" in markdown
        store.close()
