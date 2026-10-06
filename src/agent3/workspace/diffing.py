"""Unified diff generation plus the search/replace edit primitive.

The agent edits code in two ways:

``write_file``
    Full overwrite - simple, but wasteful for large files.
``edit_file``
    Anchored *search / replace*, implemented by :func:`apply_search_replace`,
    with a whitespace-tolerant fallback so that a model which re-indents its
    search block still lands the edit.
``patch_file``
    A real multi-hunk unified diff, applied by :func:`apply_unified_patch`.
    This is the right tool when several places in one file change at once:
    the model emits the patch in a single turn instead of guessing the file
    state between three consecutive ``edit_file`` calls.

Both paths produce a unified diff that the UI renders in the right panel.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class DiffStats:
    """Line level change statistics for one diff."""

    added: int = 0
    removed: int = 0
    hunks: int = 0

    @property
    def changed(self) -> int:
        return self.added + self.removed

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"+{self.added} -{self.removed}"


@dataclass
class DiffLine:
    """A single rendered diff row (used by the Qt diff viewer)."""

    kind: str  # 'context' | 'added' | 'removed' | 'header' | 'hunk'
    text: str
    old_lineno: Optional[int] = None
    new_lineno: Optional[int] = None


def make_unified_diff(
    old: str,
    new: str,
    path: str = "file",
    *,
    context: int = 3,
) -> str:
    """Return a unified diff between *old* and *new*."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context,
    )
    text = "".join(diff)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def diff_stats(unified: str) -> DiffStats:
    """Count additions / removals / hunks inside a unified diff."""
    stats = DiffStats()
    for line in unified.splitlines():
        if line.startswith("@@"):
            stats.hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            stats.added += 1
        elif line.startswith("-") and not line.startswith("---"):
            stats.removed += 1
    return stats


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified_diff(unified: str) -> List[DiffLine]:
    """Convert a unified diff into renderable :class:`DiffLine` rows."""
    rows: List[DiffLine] = []
    old_no = new_no = 0
    for raw in unified.splitlines():
        if raw.startswith(("--- ", "+++ ")):
            rows.append(DiffLine("header", raw))
            continue
        match = _HUNK_RE.match(raw)
        if match:
            old_no = int(match.group(1))
            new_no = int(match.group(3))
            rows.append(DiffLine("hunk", raw))
            continue
        if raw.startswith("+"):
            rows.append(DiffLine("added", raw[1:], None, new_no))
            new_no += 1
        elif raw.startswith("-"):
            rows.append(DiffLine("removed", raw[1:], old_no, None))
            old_no += 1
        elif raw.startswith("\\"):  # "\ No newline at end of file"
            rows.append(DiffLine("context", raw))
        else:
            rows.append(DiffLine("context", raw[1:] if raw.startswith(" ") else raw, old_no, new_no))
            old_no += 1
            new_no += 1
    return rows


def side_by_side(old: str, new: str) -> List[Tuple[str, str, str]]:
    """Pair up lines for a two-column viewer.

    Returns ``(kind, left, right)`` tuples where *kind* is one of
    ``equal``/``replace``/``delete``/``insert``.
    """
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    rows: List[Tuple[str, str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                rows.append(("equal", old_lines[i1 + offset], new_lines[j1 + offset]))
        elif tag == "replace":
            span = max(i2 - i1, j2 - j1)
            for offset in range(span):
                left = old_lines[i1 + offset] if i1 + offset < i2 else ""
                right = new_lines[j1 + offset] if j1 + offset < j2 else ""
                rows.append(("replace", left, right))
        elif tag == "delete":
            for offset in range(i1, i2):
                rows.append(("delete", old_lines[offset], ""))
        elif tag == "insert":
            for offset in range(j1, j2):
                rows.append(("insert", "", new_lines[offset]))
    return rows


class EditError(ValueError):
    """Raised when a search/replace edit cannot be applied safely."""


def _normalise(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines())


def apply_search_replace(
    content: str,
    search: str,
    replace: str,
    *,
    count: int = 1,
    allow_fuzzy: bool = True,
) -> str:
    """Return *content* with *search* replaced by *replace*.

    Raises :class:`EditError` when the anchor is missing or ambiguous, which
    the agent loop turns into a corrective observation for the model.
    """
    if not search:
        raise EditError("the search block must not be empty")

    occurrences = content.count(search)
    if occurrences == 1 or (occurrences > 1 and count != 1):
        return content.replace(search, replace, count if count > 0 else -1)
    if occurrences > 1 and count == 1:
        raise EditError(
            f"the search block matches {occurrences} times - add surrounding context "
            "to make it unique, or pass count=0 to replace all occurrences"
        )

    if not allow_fuzzy:
        raise EditError("the search block was not found in the file")

    # Fuzzy pass: match on whitespace-insensitive content, preserving the
    # original indentation of the first replaced line.
    content_lines = content.splitlines(keepends=True)
    search_lines = search.strip("\n").splitlines()
    if not search_lines:
        raise EditError("the search block was not found in the file")
    needle = _normalise(search)
    window = len(search_lines)
    for start in range(0, max(0, len(content_lines) - window + 1)):
        chunk = "".join(content_lines[start : start + window])
        if _normalise(chunk) != needle:
            continue
        original_indent = re.match(r"[ \t]*", content_lines[start]).group(0)
        replacement_lines = replace.splitlines()
        if replacement_lines:
            base_indent = re.match(r"[ \t]*", replacement_lines[0]).group(0)
            rebuilt: List[str] = []
            for line in replacement_lines:
                stripped = line[len(base_indent) :] if line.startswith(base_indent) else line.lstrip()
                rebuilt.append(f"{original_indent}{stripped}" if stripped else "")
            new_block = "\n".join(rebuilt)
        else:
            new_block = ""
        tail_newline = "\n" if chunk.endswith("\n") else ""
        return "".join(content_lines[:start]) + new_block + tail_newline + "".join(
            content_lines[start + window :]
        )

    raise EditError(
        "the search block was not found in the file - re-read the file and copy an exact snippet"
    )


@dataclass
class PatchHunk:
    """One ``@@`` section of a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str] = field(default_factory=list)
    header: str = ""

    @property
    def old_block(self) -> List[str]:
        """Lines the hunk expects to find (context + removals)."""
        return [line[1:] for line in self.lines if line[:1] in (" ", "-")]

    @property
    def new_block(self) -> List[str]:
        """Lines the hunk produces (context + additions)."""
        return [line[1:] for line in self.lines if line[:1] in (" ", "+")]

    @property
    def stats(self) -> Tuple[int, int]:
        added = sum(1 for line in self.lines if line.startswith("+"))
        removed = sum(1 for line in self.lines if line.startswith("-"))
        return added, removed


class PatchError(EditError):
    """Raised when a unified diff cannot be applied."""


@dataclass
class PatchResult:
    """Outcome of :func:`apply_unified_patch`."""

    content: str
    hunks_applied: int = 0
    offsets: List[int] = field(default_factory=list)
    fuzzy: int = 0
    added: int = 0
    removed: int = 0

    @property
    def shifted(self) -> bool:
        return any(offset != 0 for offset in self.offsets)


def parse_patch_hunks(patch: str) -> List[PatchHunk]:
    """Split a unified diff into :class:`PatchHunk` objects.

    ``--- a/x`` / ``+++ b/x`` headers, ``diff --git`` lines and ``index``
    lines are tolerated and ignored; everything outside a hunk is dropped so
    that a model wrapping the patch in prose still produces a valid result.
    """
    hunks: List[PatchHunk] = []
    current: Optional[PatchHunk] = None
    for raw in patch.splitlines():
        match = _HUNK_RE.match(raw)
        if match:
            current = PatchHunk(
                old_start=int(match.group(1)),
                old_count=int(match.group(2)) if match.group(2) is not None else 1,
                new_start=int(match.group(3)),
                new_count=int(match.group(4)) if match.group(4) is not None else 1,
                header=raw,
            )
            hunks.append(current)
            continue
        if current is None:
            continue
        if raw.startswith(("--- ", "+++ ", "diff --git", "index ", "similarity index", "rename ")):
            continue
        if raw.startswith("\\"):  # "\ No newline at end of file"
            continue
        if raw[:1] in (" ", "+", "-"):
            current.lines.append(raw)
        elif raw == "":
            # An empty line is ambiguous: it is either a context line whose
            # single leading space was stripped (markdown round-trips do this
            # constantly) or the blank line that separates the patch from the
            # surrounding prose. The hunk header tells us which: only treat it
            # as context while the hunk still owes lines.
            consumed_old = sum(1 for line in current.lines if line[:1] in (" ", "-"))
            consumed_new = sum(1 for line in current.lines if line[:1] in (" ", "+"))
            if consumed_old < current.old_count or consumed_new < current.new_count:
                current.lines.append(" ")
            else:
                current = None
        else:
            # Anything else ends the hunk: we are back in prose.
            current = None
    return [hunk for hunk in hunks if hunk.lines]


def patch_target_path(patch: str) -> str:
    """Best effort extraction of the file path from a diff header."""
    for raw in patch.splitlines():
        if raw.startswith("+++ "):
            candidate = raw[4:].strip().split("\t")[0]
            if candidate and candidate != "/dev/null":
                if candidate.startswith(("a/", "b/")):
                    candidate = candidate[2:]
                return candidate
        if raw.startswith("diff --git "):
            parts = raw.split()
            if len(parts) >= 4:
                candidate = parts[3]
                return candidate[2:] if candidate.startswith(("a/", "b/")) else candidate
    return ""


def _match_at(haystack: Sequence[str], needle: Sequence[str], index: int, *, loose: bool) -> bool:
    if index < 0 or index + len(needle) > len(haystack):
        return False
    for offset, expected in enumerate(needle):
        actual = haystack[index + offset]
        if actual == expected:
            continue
        if loose and actual.strip() == expected.strip():
            continue
        return False
    return True


def apply_unified_patch(
    content: str,
    patch: str,
    *,
    max_offset: int = 200,
) -> PatchResult:
    """Apply a unified diff to *content* and return the patched text.

    The implementation mirrors GNU ``patch`` closely enough for agent use:

    * hunks are located by their context, not by blind line numbers, so a
      patch still applies after earlier hunks changed the line count;
    * the search walks outwards from the declared position (up to
      *max_offset* lines) and only then falls back to a whitespace
      insensitive comparison;
    * a hunk that cannot be placed raises :class:`PatchError` naming the hunk
      and the context it expected, which the agent loop feeds back to the
      model as a corrective observation.
    """
    hunks = parse_patch_hunks(patch)
    if not hunks:
        raise PatchError(
            "no '@@ -old,+new @@' hunks found - send a real unified diff "
            "(you can generate one with `git diff`)"
        )

    keepends = content.splitlines(keepends=True)
    lines = content.splitlines()
    trailing_newline = content.endswith("\n") or not content
    result = PatchResult(content=content)
    drift = 0

    for number, hunk in enumerate(hunks, start=1):
        expected = hunk.old_block
        replacement = hunk.new_block
        start = hunk.old_start - 1 + drift

        if not expected:
            # Pure insertion hunk: place it at the declared position.
            index = max(0, min(start, len(lines)))
            lines[index:index] = replacement
            result.hunks_applied += 1
            result.offsets.append(0)
            drift += len(replacement)
            added, removed = hunk.stats
            result.added += added
            result.removed += removed
            continue

        index = -1
        used_fuzzy = False
        for loose in (False, True):
            if _match_at(lines, expected, start, loose=loose):
                index = start
            else:
                for distance in range(1, max_offset + 1):
                    if _match_at(lines, expected, start - distance, loose=loose):
                        index = start - distance
                        break
                    if _match_at(lines, expected, start + distance, loose=loose):
                        index = start + distance
                        break
            if index >= 0:
                used_fuzzy = loose
                break

        if index < 0:
            preview = "\n".join(expected[:6])
            raise PatchError(
                f"hunk #{number} ({hunk.header.strip()}) does not apply: the file does not "
                f"contain the expected context near line {hunk.old_start}.\n"
                f"--- expected ---\n{preview}\n"
                "Re-read the file and rebuild the patch from its current content."
            )

        lines[index : index + len(expected)] = replacement
        result.hunks_applied += 1
        result.offsets.append(index - start)
        if used_fuzzy:
            result.fuzzy += 1
        drift += len(replacement) - len(expected)
        added, removed = hunk.stats
        result.added += added
        result.removed += removed

    patched = "\n".join(lines)
    if trailing_newline and patched and not patched.endswith("\n"):
        patched += "\n"
    if not keepends and patched and not patched.endswith("\n"):
        patched += "\n"
    result.content = patched
    return result


def summarize_changes(path: str, old: str, new: str) -> str:
    """One-line human summary, e.g. ``src/app.py (+12 -3)``."""
    stats = diff_stats(make_unified_diff(old, new, path))
    if stats.changed == 0:
        return f"{path} (no changes)"
    return f"{path} (+{stats.added} -{stats.removed})"


def chunk_diff_for_prompt(unified: str, max_lines: int = 200) -> str:
    """Trim an oversized diff before feeding it back to the model."""
    lines: Sequence[str] = unified.splitlines()
    if len(lines) <= max_lines:
        return unified
    head = list(lines[: max_lines // 2])
    tail = list(lines[-max_lines // 2 :])
    omitted = len(lines) - len(head) - len(tail)
    return "\n".join([*head, f"... [{omitted} diff lines omitted] ...", *tail])
