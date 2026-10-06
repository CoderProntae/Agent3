"""Tests for the unified-diff engine behind the ``patch_file`` tool."""

from __future__ import annotations

import pytest

from agent3.workspace.diffing import (
    PatchError,
    apply_unified_patch,
    make_unified_diff,
    parse_patch_hunks,
    patch_target_path,
)

ORIGINAL = "\n".join(f"line {i}" for i in range(1, 41)) + "\n"


def three_point_change(text: str) -> str:
    return (
        text.replace("line 3\n", "LINE THREE\n")
        .replace("line 20\n", "LINE TWENTY\n")
        .replace("line 38\n", "LINE THIRTY EIGHT\n")
    )


class TestParsing:
    def test_hunk_boundaries(self):
        patch = make_unified_diff(ORIGINAL, three_point_change(ORIGINAL), "f.txt")
        hunks = parse_patch_hunks(patch)
        assert len(hunks) == 3
        assert hunks[0].old_start == 1
        assert all(hunk.lines for hunk in hunks)

    def test_prose_around_the_patch_is_ignored(self):
        patch = make_unified_diff(ORIGINAL, three_point_change(ORIGINAL), "f.txt")
        noisy = f"Here is my patch, please apply it:\n\n{patch}\nThat should do it.\n"
        assert len(parse_patch_hunks(noisy)) == 3
        assert apply_unified_patch(ORIGINAL, noisy).content == three_point_change(ORIGINAL)

    def test_path_is_recovered_from_the_header(self):
        patch = make_unified_diff("a\n", "b\n", "src/deep/module.py")
        assert patch_target_path(patch) == "src/deep/module.py"
        assert patch_target_path("@@ -1 +1 @@\n-a\n+b\n") == ""

    def test_git_style_header(self):
        patch = (
            "diff --git a/src/app.py b/src/app.py\n"
            "index 83db48f..bf269f4 100644\n"
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old\n"
            "+new\n"
            " tail\n"
        )
        assert patch_target_path(patch) == "src/app.py"
        assert apply_unified_patch("old\ntail\n", patch).content == "new\ntail\n"

    def test_stripped_context_space_is_tolerated(self):
        """Markdown round-trips eat the leading space of blank context lines."""
        patch = "@@ -1,3 +1,3 @@\n a\n\n-b\n+B\n"
        assert apply_unified_patch("a\n\nb\n", patch).content == "a\n\nB\n"


class TestApplying:
    def test_round_trip(self):
        changed = three_point_change(ORIGINAL)
        patch = make_unified_diff(ORIGINAL, changed, "f.txt")
        result = apply_unified_patch(ORIGINAL, patch)
        assert result.content == changed
        assert result.hunks_applied == 3
        assert result.added == 3
        assert result.removed == 3
        assert result.fuzzy == 0
        assert not result.shifted

    def test_three_places_in_one_call(self):
        """The whole reason the tool exists: no state guessing between edits."""
        changed = three_point_change(ORIGINAL)
        result = apply_unified_patch(ORIGINAL, make_unified_diff(ORIGINAL, changed, "f.txt"))
        assert result.content.count("LINE ") == 3

    def test_hunks_follow_shifted_content(self):
        patch = make_unified_diff(ORIGINAL, three_point_change(ORIGINAL), "f.txt")
        shifted = "header 1\nheader 2\nheader 3\n" + ORIGINAL
        result = apply_unified_patch(shifted, patch)
        assert result.content == "header 1\nheader 2\nheader 3\n" + three_point_change(ORIGINAL)
        assert result.shifted
        assert set(result.offsets) == {3}

    def test_later_hunks_survive_earlier_line_count_changes(self):
        new = ORIGINAL.replace("line 3\n", "line 3\nline 3b\nline 3c\n").replace(
            "line 30\n", "LINE THIRTY\n"
        )
        patch = make_unified_diff(ORIGINAL, new, "f.txt")
        assert apply_unified_patch(ORIGINAL, patch).content == new

    def test_whitespace_insensitive_fallback(self):
        patch = make_unified_diff(ORIGINAL, three_point_change(ORIGINAL), "f.txt")
        reindented = ORIGINAL.replace("line 20\n", "line 20   \n")
        result = apply_unified_patch(reindented, patch)
        assert result.fuzzy >= 1
        assert "LINE TWENTY" in result.content

    def test_pure_insertion_hunk(self):
        patch = "@@ -0,0 +1,2 @@\n+first\n+second\n"
        assert apply_unified_patch("", patch).content == "first\nsecond\n"

    def test_deletion_only_hunk(self):
        patch = "@@ -1,3 +1,2 @@\n a\n-b\n c\n"
        assert apply_unified_patch("a\nb\nc\n", patch).content == "a\nc\n"

    def test_trailing_newline_is_preserved(self):
        assert apply_unified_patch("a\nb\n", "@@ -1,2 +1,2 @@\n a\n-b\n+B\n").content == "a\nB\n"

    def test_file_without_trailing_newline(self):
        result = apply_unified_patch("a\nb", "@@ -1,2 +1,2 @@\n a\n-b\n+B\n")
        assert result.content.startswith("a\nB")


class TestFailures:
    def test_missing_hunks_is_an_explicit_error(self):
        with pytest.raises(PatchError) as excinfo:
            apply_unified_patch("a\n", "please change a to b")
        assert "unified diff" in str(excinfo.value)

    def test_context_mismatch_names_the_hunk(self):
        patch = make_unified_diff(ORIGINAL, three_point_change(ORIGINAL), "f.txt")
        with pytest.raises(PatchError) as excinfo:
            apply_unified_patch("something else entirely\n", patch)
        message = str(excinfo.value)
        assert "hunk #1" in message
        assert "expected" in message
        assert "Re-read the file" in message

    def test_second_hunk_failure_is_reported_as_such(self):
        patch = "@@ -1,1 +1,1 @@\n-a\n+A\n@@ -5,1 +5,1 @@\n-nope\n+NOPE\n"
        with pytest.raises(PatchError) as excinfo:
            apply_unified_patch("a\nb\nc\n", patch)
        assert "hunk #2" in str(excinfo.value)

    def test_offset_search_is_bounded(self):
        patch = "@@ -1,1 +1,1 @@\n-needle\n+NEEDLE\n"
        far_away = "filler\n" * 500 + "needle\n"
        with pytest.raises(PatchError):
            apply_unified_patch(far_away, patch, max_offset=10)
        assert apply_unified_patch(far_away, patch, max_offset=600).content.endswith("NEEDLE\n")
