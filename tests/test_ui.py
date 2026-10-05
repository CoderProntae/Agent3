"""Headless smoke tests for the Qt layer.

They run with ``QT_QPA_PLATFORM=offscreen`` (set automatically below) and are
skipped entirely when no Qt platform plugin can be initialised, so the suite
still passes on a bare CI container.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:  # PySide6 needs libGL/libxkbcommon which a bare container may lack.
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"PySide6 is not usable here: {exc}", allow_module_level=True)

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:  # pragma: no cover - no display backend at all
            pytest.skip(f"cannot start QApplication: {exc}")
    yield app
    app.processEvents()


class TestTheme:
    def test_stylesheet_is_generated(self, qapp):
        from agent3.ui.theme import apply_theme, build_stylesheet

        sheet = build_stylesheet()
        assert "QPushButton" in sheet and "QTreeView" in sheet
        apply_theme(qapp)
        assert qapp.styleSheet()

    def test_mono_font(self, qapp):
        from agent3.ui.theme import mono_font

        assert mono_font(12).pointSize() == 12


class TestSyntax:
    def test_language_detection(self):
        from agent3.ui.syntax import language_for

        assert language_for("a/b/main.py") == "python"
        assert language_for("index.tsx") == "js"
        assert language_for("data.json") == "json"
        assert language_for("notes.unknown") == "plain"

    def test_highlighter_runs(self, qapp):
        from PySide6.QtGui import QTextDocument

        from agent3.ui.syntax import CodeHighlighter

        document = QTextDocument()
        highlighter = CodeHighlighter(document, "python")
        document.setPlainText("def f():\n    return '''x'''  # comment\n")
        highlighter.rehighlight()
        highlighter.set_language("json")
        assert document.blockCount() == 3


class TestWidgets:
    def test_chat_view_flow(self, qapp):
        from agent3.ui.widgets.chat_view import ChatView

        chat = ChatView()
        chat.add_message("user", "**hello**")
        chat.begin_assistant_stream()
        chat.append_assistant_delta("partial ")
        chat.append_assistant_delta("answer")
        assert chat.is_streaming is True
        chat.end_assistant_stream("partial answer")
        assert chat.is_streaming is False
        card = chat.add_action_card("[AGENT] write_file", "path: x.py")
        card.set_status("success", "12 ms")
        card.set_details("some details")
        chat.add_notice("done", "success")
        chat.clear()

    def test_usage_panel_updates(self, qapp, usage_manager):
        from agent3.ui.widgets.usage_panel import UsagePanel

        usage_manager.record_request(model="m", prompt_tokens=10, completion_tokens=5, duration_ms=10)
        panel = UsagePanel()
        panel.update_snapshot(usage_manager.snapshot())

    def test_file_tree_lists_workspace(self, qapp, fs):
        from agent3.ui.widgets.file_tree import WorkspaceTree

        tree = WorkspaceTree()
        tree.set_workspace(fs)
        names = {tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())}
        assert {"src", "README.md"} <= names

    def test_editor_opens_and_saves(self, qapp, fs):
        from agent3.ui.widgets.editor_panel import EditorPanel

        panel = EditorPanel()
        panel.set_workspace(fs)
        editor = panel.open_file("src/app.py")
        assert editor is not None and "def greet" in editor.toPlainText()
        editor.setPlainText("changed = True\n")
        assert panel.has_unsaved() is True
        assert panel.save_current() is True
        assert fs.read_text("src/app.py") == "changed = True\n"
        assert panel.has_unsaved() is False

    def test_editor_rejects_binary(self, qapp, fs):
        from agent3.ui.widgets.editor_panel import EditorPanel

        (fs.root / "blob.bin").write_bytes(b"\x00\x01\x02")
        panel = EditorPanel()
        panel.set_workspace(fs)
        assert panel.open_file("blob.bin") is None

    def test_diff_view_renders(self, qapp):
        from agent3.ui.widgets.diff_view import DiffView
        from agent3.workspace.diffing import make_unified_diff

        view = DiffView()
        view.show_diff("f.py", make_unified_diff("a\n", "b\n", "f.py"), "a\n", "b\n")
        view.clear()

    def test_terminal_panel_runs_command(self, qapp, runner):
        from agent3.ui.widgets.terminal_panel import TerminalPanel

        panel = TerminalPanel()
        panel.set_runner(runner)
        panel.append_command("echo hi")
        panel.append_stream("stdout", "hi\n")
        panel.clear()

    def test_session_list(self, qapp):
        from agent3.agent.session import SessionStore
        from agent3.ui.widgets.session_list import SessionList

        store = SessionStore()
        first = store.create(title="One")
        store.create(title="Two")
        widget = SessionList()
        widget.set_sessions(store.list(), first.id)
        assert widget.current_session_id() == first.id
        store.close()

    def test_settings_dialog_builds(self, qapp):
        from agent3.core.config import ConfigManager
        from agent3.ui.widgets.settings_dialog import SettingsDialog

        manager = ConfigManager()
        dialog = SettingsDialog(manager)
        assert dialog._port.value() == 11435  # noqa: SLF001 - verifying the default endpoint
        dialog._port.setValue(11500)  # noqa: SLF001
        dialog._apply_values()  # noqa: SLF001
        assert ConfigManager().config.ollama.port == 11500


class TestMainWindow:
    def test_window_builds_and_mounts_workspace(self, qapp, workspace):
        from agent3.core.config import ConfigManager
        from agent3.ui.main_window import MainWindow

        manager = ConfigManager()
        window = MainWindow(manager)
        window.open_workspace(str(workspace))
        assert window.fs is not None and window.loop is not None
        assert window.tree.topLevelItemCount() >= 2
        window.editor.open_file("README.md")
        window.refresh_usage()
        window._describe_call  # attribute exists
        window.close()

    def test_usage_limit_editor_builds(self, qapp):
        from usage_limit_editor.main_window import UsageLimitEditorWindow

        window = UsageLimitEditorWindow()
        assert window._requests_per_day.value() == 500  # noqa: SLF001
        window._requests_per_day.setValue(250)  # noqa: SLF001
        window._save()  # noqa: SLF001
        from agent3.limits.policy import PolicyStore

        assert PolicyStore().load(force=True).max_requests_per_day == 250
        window.close()


class TestDiffRendering:
    """The action-card review view: green additions, red deletions, numbers."""

    SAMPLE = (
        "--- a/src/api.py\n"
        "+++ b/src/api.py\n"
        "@@ -1,4 +1,6 @@\n"
        " from fastapi import FastAPI\n"
        " \n"
        " app = FastAPI()\n"
        "+\n"
        '+@app.get("/health")\n'
        "-removed_line()\n"
    )

    def test_render_marks_additions_and_deletions(self):
        from agent3.ui.diff_render import render_diff_html
        from agent3.ui.theme import COLORS

        html = render_diff_html(self.SAMPLE)
        assert COLORS.added_bg in html and COLORS.added_fg in html
        assert COLORS.removed_bg in html and COLORS.removed_fg in html
        assert "/health" in html
        assert "<table" in html

    def test_headers_are_hidden_in_cards_by_default(self):
        from agent3.ui.diff_render import render_diff_html

        assert "a/src/api.py" not in render_diff_html(self.SAMPLE)
        assert "a/src/api.py" in render_diff_html(self.SAMPLE, show_headers=True)

    def test_line_numbers_are_present(self):
        from agent3.ui.diff_render import render_diff_html

        html = render_diff_html(self.SAMPLE)
        assert ">1<" in html and ">3<" in html

    def test_html_is_escaped(self):
        from agent3.ui.diff_render import render_diff_html

        html = render_diff_html('@@ -1 +1 @@\n+<script>alert("x")</script>\n')
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_large_diffs_are_truncated_with_a_notice(self):
        from agent3.ui.diff_render import render_diff_html

        big = "@@ -1,600 +1,600 @@\n" + "".join(f"+line {i}\n" for i in range(600))
        html = render_diff_html(big, max_rows=50)
        assert "more diff lines" in html

    def test_empty_diff_says_so(self):
        from agent3.ui.diff_render import render_diff_html

        assert "no textual changes" in render_diff_html("")

    def test_badge_counts(self):
        from agent3.ui.diff_render import diff_badge_text

        assert diff_badge_text(self.SAMPLE) == "+2 -1"
        assert diff_badge_text("") == ""

    def test_detail_kind_sniffing(self):
        from agent3.ui.diff_render import guess_detail_kind

        assert guess_detail_kind(self.SAMPLE) == "diff"
        assert guess_detail_kind("$ pytest -q\n1 passed") == "output"
        assert guess_detail_kind(None) == "output"

    def test_output_rendering_colours_failures_and_passes(self):
        from agent3.ui.diff_render import render_output_html
        from agent3.ui.theme import COLORS

        html = render_output_html("$ pytest -q\n3 passed\nERROR: boom\nwarning: old api")
        assert COLORS.success in html
        assert COLORS.danger in html
        assert COLORS.warning in html
        assert COLORS.accent in html


class TestActionCardDetails:
    def test_card_renders_a_diff_and_shows_its_badge(self, qapp):
        from agent3.ui.widgets.chat_view import ActionCard

        card = ActionCard("[AGENT] write_file", "path: a.py")
        card.set_diff("@@ -1,2 +1,2 @@\n-old\n+new\n context\n")
        assert card.badge_text() == "+1 -1"
        assert card._toggle.text() in {"Diff", "Hide diff"}
        card._toggle.setChecked(True)
        assert not card._details.isHidden()
        assert "new" in card._details.toHtml()

    def test_card_sniffs_a_diff_passed_as_details(self, qapp):
        from agent3.ui.widgets.chat_view import ActionCard

        card = ActionCard("[AGENT] patch_file")
        card.set_details("@@ -1 +1 @@\n-a\n+b\n")
        assert card.badge_text() == "+1 -1"
        assert card._toggle.text() == "Diff"

    def test_card_renders_command_output_as_console(self, qapp):
        from agent3.ui.widgets.chat_view import ActionCard

        card = ActionCard("[AGENT TERMINAL] run_command", "$ pytest -q")
        card.set_details("$ pytest -q\n12 passed in 0.4s", kind="output")
        assert card.badge_text() == ""
        assert card._toggle.text() == "Details"
        assert "12 passed" in card._details.toHtml()

    def test_empty_details_hide_the_toggle(self, qapp):
        from agent3.ui.widgets.chat_view import ActionCard

        card = ActionCard("[AGENT] list_files")
        card.set_details("")
        assert card._toggle.isHidden() is True

    def test_blocked_status_is_rendered(self, qapp):
        from agent3.ui.widgets.chat_view import ActionCard

        card = ActionCard("[AGENT TERMINAL] run_command", "$ npm init")
        card.set_status("blocked", "4 ms")
        assert card._icon.text() == "!"
        assert "blocked" not in card._subtitle.text()


# ==================================================== composer control strip
GPT_OSS_SHOW = {
    "capabilities": ["completion", "tools", "thinking"],
    "thinking": {"values": ["low", "medium", "high"], "default": "medium"},
}
QWEN_SHOW = {
    "capabilities": ["completion", "thinking"],
    "thinking": {"values": [True, False], "default": True},
}
PLAIN_SHOW = {"capabilities": ["completion"]}


def _support(model, payload):
    from agent3.llm.ollama_client import ThinkingSupport

    return ThinkingSupport.from_show(model, payload)


class TestComposerBar:
    def test_model_picker_lives_here(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        bar.set_models(["a:1", "b:2"], "b:2")
        assert bar.current_model() == "b:2"
        assert bar.available_models() == ["a:1", "b:2"]

    def test_model_change_is_emitted(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        seen = []
        bar.model_changed.connect(seen.append)
        bar.model_box.setEditText("qwen3:8b")
        assert seen and seen[-1] == "qwen3:8b"

    def test_programmatic_model_set_is_silent(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        seen = []
        bar.model_changed.connect(seen.append)
        bar.set_models(["x:1"], "x:1")
        assert seen == []

    def test_levels_come_from_the_server(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        bar.set_thinking_support(_support("gpt-oss:20b", GPT_OSS_SHOW))
        labels = [bar.level_box.itemText(i) for i in range(bar.level_box.count())]
        assert labels == ["Auto", "Low", "Medium", "High"]

    def test_level_only_model_cannot_switch_thinking_off(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        bar.set_thinking_support(_support("gpt-oss:20b", GPT_OSS_SHOW))
        assert bar.think_toggle.isChecked() is True
        assert bar.think_toggle.isEnabled() is False

    def test_boolean_model_hides_the_level_picker(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        bar.set_thinking_support(_support("qwen3:8b", QWEN_SHOW))
        assert bar.think_toggle.isEnabled() is True
        assert bar.level_box.isHidden() is True

    def test_model_without_reasoning_disables_everything(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        bar.set_thinking_support(_support("llama3.1:8b", PLAIN_SHOW))
        assert bar.think_toggle.isEnabled() is False
        assert bar.think_toggle.isChecked() is False
        assert bar.level_box.isHidden() is True
        assert bar.capability_label.text() == "no reasoning"

    def test_selecting_a_level_emits_that_exact_value(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        bar.set_thinking_support(_support("gpt-oss:20b", GPT_OSS_SHOW))
        seen = []
        bar.think_changed.connect(seen.append)
        bar.level_box.setCurrentIndex(bar.level_box.findData("low"))
        assert seen == ["low"]
        assert bar.think_spec() == "low"

    def test_toggling_off_emits_off(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        bar.set_thinking_support(_support("qwen3:8b", QWEN_SHOW), "on")
        seen = []
        bar.think_changed.connect(seen.append)
        bar.think_toggle.setChecked(False)
        assert seen == ["off"]

    def test_unsupported_spec_is_reset_when_the_model_changes(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        bar.set_thinking_support(_support("gpt-oss:20b", GPT_OSS_SHOW), "high")
        assert bar.think_spec() == "high"
        bar.set_thinking_support(_support("qwen3:8b", QWEN_SHOW))
        assert bar.think_spec() == "auto"  # qwen3 has no "high"

    def test_summary_text_describes_the_setting(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        bar.set_thinking_support(_support("gpt-oss:20b", GPT_OSS_SHOW), "medium")
        assert bar.thinking_summary() == "reasoning: medium"

    def test_connection_badge(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        bar.set_connection(True, "online · http://localhost:11435")
        assert "online" in bar.connection_label.text()
        bar.set_connection(False, "offline")
        assert "offline" in bar.connection_label.text()

    def test_busy_locks_the_controls(self, qapp):
        from agent3.ui.widgets.composer_bar import ComposerBar

        bar = ComposerBar()
        bar.set_thinking_support(_support("qwen3:8b", QWEN_SHOW))
        bar.set_busy(True)
        assert bar.model_box.isEnabled() is False
        bar.set_busy(False)
        assert bar.model_box.isEnabled() is True


# ================================================================ plan panel
class TestTaskPanel:
    def test_empty_state(self, qapp):
        from agent3.ui.widgets.task_panel import TaskPanel

        panel = TaskPanel()
        assert panel.list.isHidden() is True
        assert panel._empty.isHidden() is False

    def test_tasks_are_listed_with_progress(self, qapp):
        from agent3.agent.tasks import TaskList
        from agent3.ui.widgets.task_panel import TaskPanel

        tasks = TaskList()
        tasks.add("one", task_id="one", status="completed")
        tasks.add("two", task_id="two", status="in_progress")
        tasks.add("three", task_id="three")

        panel = TaskPanel()
        panel.set_tasks(list(tasks))
        assert panel.list.count() == 3
        assert panel._counter.text() == "1/3"
        assert panel._bar.value() == 33
        assert panel.task_ids() == ["one", "two", "three"]

    def test_glyphs_reflect_status(self, qapp):
        from agent3.agent.tasks import TaskList
        from agent3.ui.widgets.task_panel import TaskPanel

        tasks = TaskList()
        tasks.add("done", task_id="d", status="completed")
        tasks.add("doing", task_id="i", status="in_progress")
        panel = TaskPanel()
        panel.set_tasks(list(tasks))
        assert panel.list.item(0).text().startswith("✓")
        assert panel.list.item(1).text().startswith("◐")
        assert panel.list.item(1).font().bold() is True

    def test_clear_returns_to_empty(self, qapp):
        from agent3.agent.tasks import TaskList
        from agent3.ui.widgets.task_panel import TaskPanel

        tasks = TaskList()
        tasks.add("x", task_id="x")
        panel = TaskPanel()
        panel.set_tasks(list(tasks))
        panel.clear()
        assert panel.list.count() == 0
        assert panel._empty.isHidden() is False


# ========================================================== reasoning block
class TestThinkingBlock:
    def test_collapsed_by_default(self, qapp):
        from agent3.ui.widgets.chat_view import ThinkingBlock

        block = ThinkingBlock()
        assert block.toggle.isChecked() is False
        assert block.body.isHidden() is True

    def test_streaming_updates_the_word_count(self, qapp):
        from agent3.ui.widgets.chat_view import ThinkingBlock

        block = ThinkingBlock()
        block.append("one two ")
        block.append("three")
        assert block.text == "one two three"
        assert "3 word(s)" in block.toggle.text()

    def test_expanding_reveals_the_trace(self, qapp):
        from agent3.ui.widgets.chat_view import ThinkingBlock

        block = ThinkingBlock()
        block.append("reasoning text")
        block.toggle.setChecked(True)
        assert block.body.isHidden() is False
        assert "reasoning text" in block.body.toPlainText()

    def test_chat_view_streams_into_a_block(self, qapp):
        from agent3.ui.widgets.chat_view import ChatView, ThinkingBlock

        chat = ChatView()
        chat.append_thinking_delta("thinking out loud")
        assert chat.thinking_text == "thinking out loud"
        chat.end_thinking()
        assert chat.thinking_text == ""

    def test_empty_block_is_discarded(self, qapp):
        from agent3.ui.widgets.chat_view import ChatView

        chat = ChatView()
        chat.begin_thinking()
        before = chat._layout.count()
        chat.end_thinking()
        assert chat._layout.count() == before - 1

    def test_ending_the_answer_closes_the_reasoning(self, qapp):
        from agent3.ui.widgets.chat_view import ChatView

        chat = ChatView()
        chat.append_thinking_delta("because")
        chat.append_assistant_delta("answer")
        chat.end_assistant_stream()
        assert chat._thinking is None


# ============================================== terminal background badge
class TestProcessBadge:
    def test_hidden_when_nothing_runs(self, qapp):
        from agent3.ui.widgets.terminal_panel import TerminalPanel

        panel = TerminalPanel()
        panel.set_process_count(0)
        assert panel._process_badge.isHidden() is True
        assert panel.process_count_text() == ""

    def test_shows_the_count(self, qapp):
        from agent3.ui.widgets.terminal_panel import TerminalPanel

        panel = TerminalPanel()
        panel.set_process_count(3)
        assert panel._process_badge.isHidden() is False
        assert "3 background" in panel._process_badge.text()
