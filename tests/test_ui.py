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
