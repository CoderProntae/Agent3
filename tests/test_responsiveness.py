"""Regressions for the three things that made the app feel broken.

1. The system prompt was re-rendered on every step, so the inference server
   could never reuse its KV cache and spent ~20 s re-evaluating the whole
   conversation before emitting a single token.
2. Chat bubbles measured their height before the document had been laid out
   at its real width, so everything after the first line was clipped away.
3. The reasoning strip advertised its discovery source ("template",
   "capability") instead of the effort level the user wanted to change.
"""

from __future__ import annotations

import os

import pytest

from agent3.agent.prompts import (
    SYSTEM_STABLE,
    SYSTEM_VOLATILE,
    build_stable_prefix,
    build_system_prompt,
)

TOOLS = "- read_file: read a file\n- write_file: write a file"


class TestPromptCachePrefix:
    """The invariant half of the prompt must come first, byte for byte."""

    def _prompt(self, tree: str, git_state: str) -> str:
        return build_system_prompt(
            workspace="/w", tools=TOOLS, tree=tree, git_state=git_state
        )

    def test_the_stable_prefix_survives_a_changing_tree(self):
        a = self._prompt("src/\n  a.py", "yes (branch: main, clean)")
        b = self._prompt("src/\n  a.py\n  b.py", "yes (branch: main, dirty)")
        prefix = build_stable_prefix(TOOLS)
        assert a.startswith(prefix)
        assert b.startswith(prefix)

    def test_the_shared_prefix_is_most_of_the_prompt(self):
        a = self._prompt("src/", "clean")
        b = self._prompt("src/\nbuild/\ndist/", "dirty")
        shared = len(os.path.commonprefix([a, b]))
        # Before the fix the prompts diverged after the first paragraph.
        assert shared > 3000
        assert shared > 0.6 * min(len(a), len(b))

    def test_volatile_values_are_absent_from_the_stable_half(self):
        prefix = build_stable_prefix(TOOLS)
        for volatile in ("Workspace root:", "Project tree", "Git repository:"):
            assert volatile not in prefix

    def test_volatile_values_are_present_in_the_whole_prompt(self):
        prompt = self._prompt("src/\n  a.py", "yes (branch: main, clean)")
        assert "Workspace root: /w" in prompt
        assert "a.py" in prompt
        assert "branch: main" in prompt

    def test_the_rules_still_come_before_the_snapshot(self):
        prompt = self._prompt("src/", "clean")
        assert prompt.index("## Hard rules") < prompt.index("## Project tree")
        assert prompt.index("## Available tools") < prompt.index("## Environment")

    def test_the_prompt_carries_no_clock(self):
        """A minute-resolution timestamp would break the cache on its own."""
        prompt = self._prompt("src/", "clean")
        line = [l for l in prompt.splitlines() if l.startswith("- Date (local):")][0]
        assert ":" not in line.split("Date (local):")[1]

    def test_project_instructions_stay_in_the_stable_half(self):
        prompt = build_system_prompt(
            workspace="/w",
            tools=TOOLS,
            tree="src/",
            extra_instructions="Always use tabs.",
        )
        assert prompt.index("Always use tabs.") < prompt.index("## Environment")

    def test_template_is_the_two_halves_joined(self):
        from agent3.agent.prompts import SYSTEM_TEMPLATE

        assert SYSTEM_TEMPLATE == SYSTEM_STABLE + SYSTEM_VOLATILE


class TestFrozenSystemPrompt:
    """The workspace snapshot is taken once per run, not once per step."""

    def _loop(self, tmp_path):
        from agent3.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._system_prompt = None
        loop._renders = 0

        def render() -> str:
            loop._renders += 1
            return f"prompt #{loop._renders}"

        loop._render_system_prompt = render
        return loop

    def test_repeated_steps_reuse_one_render(self, tmp_path):
        loop = self._loop(tmp_path)
        first = loop.system_prompt()
        assert loop.system_prompt() == first
        assert loop.system_prompt() == first
        assert loop._renders == 1

    def test_refresh_takes_a_new_snapshot(self, tmp_path):
        loop = self._loop(tmp_path)
        first = loop.system_prompt()
        loop.refresh_system_prompt()
        assert loop.system_prompt() != first
        assert loop._renders == 2


try:
    from PySide6.QtWidgets import QApplication, QWidget
except ImportError:  # pragma: no cover - headless CI without Qt libs
    pytest.skip("PySide6 is unavailable", allow_module_level=True)

from agent3.llm.ollama_client import ThinkingSupport  # noqa: E402
from agent3.ui.widgets.chat_view import MessageBubble  # noqa: E402
from agent3.ui.widgets.composer_bar import ComposerBar  # noqa: E402

QWEN_TEMPLATE = """
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('bad') }}
    {%- endif %}
{%- endif %}
"""

LONG_MARKDOWN = (
    "A paragraph that is quite long and will certainly need to wrap over "
    "several lines once the bubble is constrained to a realistic width.\n\n"
    "| Column | Value |\n|---|---|\n| one | 1 |\n| two | 2 |\n\n"
    "```python\ndef f():\n    return 1\n```\n\n"
    "Closing paragraph after the code block.\n"
)


#: Qt destroys a widget as soon as its last Python reference goes away, so
#: every widget built by a test is parked here for the module's lifetime.
_KEEP: list = []


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class TestBubbleHeight:
    """Bubbles must be tall enough for everything they contain."""

    def _bubble(self, app, width: int = 600) -> MessageBubble:
        host = QWidget()
        host.resize(width, 900)
        bubble = MessageBubble("assistant", LONG_MARKDOWN, parent=host)
        bubble.resize(width, bubble.sizeHint().height())
        host.show()
        app.processEvents()
        _KEEP.append(host)
        return bubble

    def test_multi_block_content_is_not_clipped(self, app):
        bubble = self._bubble(app)
        body = bubble._body  # noqa: SLF001
        document_height = body.document().size().height()
        assert body.height() >= document_height - 1

    def test_the_bubble_is_taller_than_a_single_line(self, app):
        bubble = self._bubble(app)
        assert bubble._body.height() > 100  # noqa: SLF001

    def test_narrowing_the_bubble_makes_it_taller(self, app):
        wide = self._bubble(app, width=900)._body.height()  # noqa: SLF001
        narrow = self._bubble(app, width=380)._body.height()  # noqa: SLF001
        assert narrow > wide

    def test_the_document_tracks_the_viewport_width(self, app):
        bubble = self._bubble(app, width=640)
        body = bubble._body  # noqa: SLF001
        assert body.document().textWidth() == pytest.approx(
            body.viewport().width(), abs=4
        )


#: One very long unwrappable line inside a fenced block - exactly the shape a
#: tool call takes. Qt's markdown importer marks fenced blocks as
#: "non breakable", so without a fix the document grows *sideways* instead of
#: wrapping, and the measured height collapses to a single line.
WIDE_CODE = (
    "Calling the tool now:\n\n"
    "```json\n"
    '{"tool": "write_file", "args": {"path": "src/agent3/ui/widgets/chat_view.py", '
    '"content": "a very long single line of generated source code that never ends"}}\n'
    "```\n"
)


class TestWideCodeBlocksWrap:
    """A long tool call must wrap and be read, not scroll off the side."""

    def _bubble(self, app, width: int = 600) -> MessageBubble:
        host = QWidget()
        host.resize(width, 900)
        bubble = MessageBubble("assistant", WIDE_CODE, parent=host)
        bubble.resize(width, bubble.sizeHint().height())
        host.show()
        app.processEvents()
        _KEEP.append(host)
        return bubble

    def test_the_document_never_grows_wider_than_the_viewport(self, app):
        body = self._bubble(app)._body  # noqa: SLF001
        assert body.document().size().width() <= body.document().textWidth() + 1

    def test_a_wrapped_tool_call_is_more_than_one_line(self, app):
        body = self._bubble(app)._body  # noqa: SLF001
        line_height = body.fontMetrics().height()
        assert body.height() > 2 * line_height

    def test_there_is_no_horizontal_scrollbar(self, app):
        body = self._bubble(app)._body  # noqa: SLF001
        assert not body.horizontalScrollBar().isVisible()

    def test_the_body_font_is_not_microscopic(self, app):
        body = self._bubble(app)._body  # noqa: SLF001
        assert body.font().pointSizeF() >= 10.5


class TestComposerShowsEffortNotJargon:
    """The strip offers a level to change, not a word to decode."""

    def _bar(self, app, data) -> ComposerBar:
        bar = ComposerBar()
        bar.set_thinking_support(ThinkingSupport.from_show("m", data))
        _KEEP.append(bar)
        return bar

    def test_the_source_name_is_never_printed(self, app):
        for data in (
            {"capabilities": ["completion"], "template": QWEN_TEMPLATE},
            {"capabilities": ["completion", "thinking"]},
            {"thinking": {"values": ["low", "high"], "default": "low"}},
        ):
            caption = self._bar(app, data).capability_label.text()
            assert caption.lower() not in ("template", "server", "capability", "metadata")

    def test_a_model_with_levels_says_effort(self, app):
        bar = self._bar(app, {"template": QWEN_TEMPLATE})
        assert bar.capability_label.text() == "effort"

    def test_the_real_levels_are_offered(self, app):
        bar = self._bar(app, {"template": QWEN_TEMPLATE})
        values = [bar.level_box.itemData(i) for i in range(bar.level_box.count())]
        assert values == ["auto", "xhigh", "medium", "low"]

    def test_the_level_box_stays_usable_while_thinking_is_off(self, app):
        bar = self._bar(app, {"template": QWEN_TEMPLATE})
        bar.think_toggle.setChecked(False)
        assert bar.level_box.isEnabled()

    def test_picking_a_level_turns_reasoning_on(self, app):
        bar = self._bar(app, {"template": QWEN_TEMPLATE})
        bar.think_toggle.setChecked(False)
        bar.level_box.setCurrentIndex(bar.level_box.findData("low"))
        assert bar.think_toggle.isChecked()
        assert bar.think_spec() == "low"

    def test_provenance_moved_into_the_tooltip(self, app):
        bar = self._bar(app, {"template": QWEN_TEMPLATE})
        tooltip = bar.capability_label.toolTip()
        assert "chat template" in tooltip
        assert "xhigh" in tooltip

    def test_an_unsupported_model_still_explains_itself(self, app):
        bar = self._bar(app, {"capabilities": ["completion"]})
        assert bar.capability_label.text() == "no reasoning"
        assert bar.capability_label.toolTip()
