"""The markdown renderer that replaced ``QTextDocument.setMarkdown``.

Qt's own markdown importer ignores the document stylesheet, so every colour
and background the theme defines was being thrown away. These tests pin the
behaviour of the replacement: the structures a model actually emits, the
escaping rules that keep a chat transcript from executing as markup, and the
syntax colouring that makes a code block readable.
"""

from __future__ import annotations

import html
import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:  # the renderer pulls colours from the theme, which imports Qt
    from agent3.ui.markdown_render import (
        highlight_code_html,
        language_alias,
        markdown_to_html,
    )
    from agent3.ui.theme import COLORS
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"PySide6 is not usable here: {exc}", allow_module_level=True)


def text_of(markup: str) -> str:
    """Strip tags and entities so a test sees what the user actually reads."""
    return html.unescape(re.sub(r"<[^>]+>", "", markup))


class TestInline:
    def test_bold_italic_and_strike(self):
        out = markdown_to_html("**b** *i* ~~s~~ ***both***")
        assert "<b>b</b>" in out
        assert "<i>i</i>" in out
        assert "<s>s</s>" in out
        assert "<b><i>both</i></b>" in out

    def test_underscore_emphasis(self):
        out = markdown_to_html("__bold__ and _italic_")
        assert "<b>bold</b>" in out and "<i>italic</i>" in out

    def test_an_underscore_inside_a_word_is_not_emphasis(self):
        """`snake_case_name` must survive intact - models write it constantly."""
        out = markdown_to_html("call snake_case_name now")
        assert "<i>" not in out
        assert "snake_case_name" in text_of(out)

    def test_inline_code_gets_the_code_colour(self):
        out = markdown_to_html("run `pytest -q` please")
        assert COLORS.code_string in out
        assert "pytest -q" in text_of(out)

    def test_markup_inside_a_code_span_stays_literal(self):
        out = markdown_to_html("`**not bold**`")
        assert "<b>" not in out
        assert "**not bold**" in text_of(out)

    def test_links_become_anchors(self):
        out = markdown_to_html("see [the docs](https://example.com/a)")
        assert '<a href="https://example.com/a"' in out
        assert "the docs" in text_of(out)

    def test_a_bare_url_is_clickable(self):
        out = markdown_to_html("visit https://example.com/x for more")
        assert '<a href="https://example.com/x"' in out

    def test_images_degrade_to_a_labelled_link(self):
        out = markdown_to_html("![a cat](https://example.com/cat.png)")
        assert '<a href="https://example.com/cat.png"' in out
        assert "[a cat]" in text_of(out)

    def test_html_in_the_source_is_escaped(self):
        out = markdown_to_html("<script>alert(1)</script> & <b>raw</b>")
        assert "<script>" not in out
        assert "alert(1)" in text_of(out)

    def test_a_hard_break_becomes_a_br(self):
        assert "<br/>" in markdown_to_html("first  \nsecond")

    def test_empty_input_is_empty_output(self):
        assert markdown_to_html("") == ""


class TestBlocks:
    def test_headings_render_at_distinct_sizes(self):
        one = markdown_to_html("# Title")
        three = markdown_to_html("### Title")
        assert "Title" in text_of(one)
        assert one != three

    def test_a_horizontal_rule_is_drawn(self):
        out = markdown_to_html("above\n\n---\n\nbelow")
        assert COLORS.border in out
        assert "above" in text_of(out) and "below" in text_of(out)

    def test_bullets_and_numbers_keep_their_markers(self):
        bullets = markdown_to_html("- one\n- two")
        assert text_of(bullets).count("•") == 2
        numbers = markdown_to_html("1. one\n2. two")
        assert "1." in text_of(numbers) and "2." in text_of(numbers)

    def test_nested_lists_are_nested(self):
        out = markdown_to_html("1. outer\n   - inner\n   - inner two")
        assert out.count("<table") >= 2
        body = text_of(out)
        assert "outer" in body and "inner two" in body

    def test_task_lists_show_their_state(self):
        out = markdown_to_html("- [x] done\n- [ ] todo")
        body = text_of(out)
        assert "✓" in body and "○" in body
        assert COLORS.success in out

    def test_blockquotes_render_their_inner_markdown(self):
        out = markdown_to_html("> quoted **word**")
        assert "<b>word</b>" in out

    def test_tables_get_a_header_row_and_alignment(self):
        out = markdown_to_html("| A | B |\n|---|--:|\n| 1 | 2 |")
        assert "<th" in out and "<td" in out
        assert 'align="right"' in out
        assert COLORS.panel_alt in out

    def test_a_pipe_line_without_a_divider_is_just_a_paragraph(self):
        out = markdown_to_html("a | b | c")
        assert "<th" not in out
        assert "a | b | c" in text_of(out)

    def test_paragraphs_are_separated(self):
        out = markdown_to_html("first para\n\nsecond para")
        assert out.count("<div") >= 2

    def test_crlf_input_is_handled(self):
        out = markdown_to_html("# Title\r\n\r\nbody\r\n")
        assert "Title" in text_of(out) and "body" in text_of(out)


class TestFencedCode:
    SNIPPET = '```python\ndef f(x):\n    # note\n    return "y"\n```'

    def test_a_fenced_block_becomes_a_pre(self):
        out = markdown_to_html(self.SNIPPET)
        assert "<pre" in out

    def test_code_wraps_instead_of_widening_the_bubble(self):
        """The regression that made long tool calls render as a thin strip."""
        assert "white-space:pre-wrap" in markdown_to_html(self.SNIPPET)

    def test_keywords_strings_and_comments_are_coloured(self):
        out = markdown_to_html(self.SNIPPET)
        assert COLORS.code_keyword in out
        assert COLORS.code_string in out
        assert COLORS.code_comment in out

    def test_the_language_is_captioned(self):
        assert "python" in text_of(markdown_to_html(self.SNIPPET))

    def test_markdown_inside_code_stays_literal(self):
        out = markdown_to_html("```\n**not bold** and `tick`\n```")
        assert "<b>" not in out
        assert "**not bold**" in text_of(out)

    def test_html_inside_code_is_escaped(self):
        out = markdown_to_html("```html\n<div class='x'>\n```")
        assert "<div class='x'>" not in out
        assert "<div class='x'>" in text_of(out)

    def test_an_unclosed_fence_still_renders(self):
        out = markdown_to_html("```python\nprint(1)")
        assert "print(1)" in text_of(out)

    def test_tildes_work_as_a_fence(self):
        out = markdown_to_html("~~~\nplain\n~~~")
        assert "<pre" in out and "plain" in text_of(out)

    def test_a_tool_call_is_rendered_as_json(self):
        out = markdown_to_html('```json\n{"tool": "finish", "args": {"summary": "ok"}}\n```')
        assert COLORS.code_builtin in out  # the keys
        assert '"tool"' in text_of(out)


class TestHighlighter:
    def test_an_unknown_language_is_escaped_but_uncoloured(self):
        out = highlight_code_html("<x> & 'y'", "brainfuck")
        assert "<span" not in out
        assert "&lt;x&gt;" in out

    def test_a_hash_inside_a_string_is_not_a_comment(self):
        out = highlight_code_html('s = "# not a comment"', "python")
        assert out.count(COLORS.code_comment) == 0

    def test_diff_blocks_use_the_diff_colours(self):
        out = highlight_code_html("--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new", "diff")
        assert COLORS.added_fg in out and COLORS.removed_fg in out

    def test_numbers_are_coloured(self):
        assert COLORS.code_number in highlight_code_html("x = 42", "python")

    def test_language_aliases_collapse_onto_families(self):
        assert language_alias("ts") == "js"
        assert language_alias("Bash") == "shell"
        assert language_alias("") == "plain"
        assert language_alias("nonsense") == "plain"
