"""Shared HTML renderers for diffs and command output.

The agent produces two kinds of detail that deserve more than grey monospace
text in the chat transcript:

* a unified diff, which should read like a review - old/new line numbers,
  green additions, red deletions, dimmed context;
* terminal output, where ``stderr`` and a non-zero exit code must stand out.

Both renderers emit self-contained HTML tables so they can be dropped into any
``QTextBrowser`` (the action cards and the right-hand diff viewer).
"""

from __future__ import annotations

import html
from typing import List, Optional

from agent3.ui.theme import COLORS
from agent3.workspace.diffing import DiffStats, diff_stats, parse_unified_diff

#: Rows rendered before the diff is cut off inside a chat action card.
CARD_DIFF_MAX_ROWS = 400


def _cell(text: str) -> str:
    """Escape a source line, keeping its indentation visible."""
    escaped = html.escape(text.rstrip("\n")).replace("\t", "&nbsp;" * 4)
    escaped = escaped.replace("  ", "&nbsp;&nbsp;")
    return escaped or "&nbsp;"


def render_diff_html(
    unified: str,
    *,
    max_rows: int = CARD_DIFF_MAX_ROWS,
    show_headers: bool = False,
    font_size: int = 9,
) -> str:
    """Render *unified* as a colourised, line-numbered HTML table.

    ``show_headers`` keeps the ``--- a/x`` / ``+++ b/x`` preamble; action
    cards hide it because the file name is already in the card title.
    """
    rows = parse_unified_diff(unified or "")
    if not show_headers:
        rows = [row for row in rows if row.kind != "header"]
    if not rows:
        return (
            f"<div style='color:{COLORS.text_faint};font-family:Consolas,monospace;"
            f"font-size:{font_size}pt;padding:4px'>(no textual changes)</div>"
        )

    truncated = len(rows) > max_rows
    visible = rows[:max_rows]

    parts: List[str] = [
        "<table cellspacing='0' cellpadding='0' "
        f"style='width:100%;font-family:\"Cascadia Code\",Consolas,monospace;font-size:{font_size}pt;'>"
    ]
    for row in visible:
        if row.kind == "added":
            style = f"background:{COLORS.added_bg};color:{COLORS.added_fg};"
            gutter = f"color:{COLORS.added_fg};background:{COLORS.added_bg};"
            marker = "+"
        elif row.kind == "removed":
            style = f"background:{COLORS.removed_bg};color:{COLORS.removed_fg};"
            gutter = f"color:{COLORS.removed_fg};background:{COLORS.removed_bg};"
            marker = "-"
        elif row.kind == "hunk":
            style = f"color:{COLORS.hunk_fg};background:{COLORS.panel_alt};"
            gutter = f"color:{COLORS.hunk_fg};background:{COLORS.panel_alt};"
            marker = "@"
        elif row.kind == "header":
            style = f"color:{COLORS.text_faint};"
            gutter = f"color:{COLORS.text_faint};"
            marker = ""
        else:
            style = f"color:{COLORS.text_dim};"
            gutter = f"color:{COLORS.text_faint};"
            marker = "&nbsp;"

        old_no = str(row.old_lineno) if row.old_lineno else "&nbsp;"
        new_no = str(row.new_lineno) if row.new_lineno else "&nbsp;"
        if row.kind in ("hunk", "header"):
            old_no = new_no = "&nbsp;"
        parts.append(
            f"<tr style='{style}'>"
            f"<td style='width:38px;text-align:right;padding:0 6px;{gutter}'>{old_no}</td>"
            f"<td style='width:38px;text-align:right;padding:0 6px;{gutter}'>{new_no}</td>"
            f"<td style='width:12px;text-align:center;{gutter}'>{marker}</td>"
            f"<td style='white-space:pre;padding-left:6px'>{_cell(row.text)}</td>"
            "</tr>"
        )
    parts.append("</table>")
    if truncated:
        parts.append(
            f"<div style='color:{COLORS.text_faint};font-family:Consolas,monospace;"
            f"font-size:{font_size}pt;padding:4px 6px'>"
            f"... {len(rows) - max_rows} more diff lines (open the file in the editor to see all)"
            "</div>"
        )
    return "".join(parts)


def render_diff_badge(stats: DiffStats) -> str:
    """Small ``+12 -3`` badge shown in an action card header."""
    chunks: List[str] = []
    if stats.added:
        chunks.append(f"<span style='color:{COLORS.added_fg}'>+{stats.added}</span>")
    if stats.removed:
        chunks.append(f"<span style='color:{COLORS.removed_fg}'>-{stats.removed}</span>")
    if not chunks:
        return ""
    return "&nbsp;".join(chunks)


def diff_badge_text(unified: str) -> str:
    """Plain-text ``+12 -3`` badge (for widgets that cannot render HTML)."""
    stats = diff_stats(unified or "")
    if not stats.changed:
        return ""
    return f"+{stats.added} -{stats.removed}"


def render_output_html(
    text: str,
    *,
    stream: str = "stdout",
    font_size: int = 9,
    max_lines: int = 400,
) -> str:
    """Render captured command output, highlighting errors and diagnostics."""
    lines = (text or "").splitlines()
    truncated = len(lines) > max_lines
    if truncated:
        head = lines[: max_lines // 2]
        tail = lines[-max_lines // 2 :]
        lines = head + [f"... {len(text.splitlines()) - len(head) - len(tail)} lines omitted ..."] + tail

    base = COLORS.danger if stream == "stderr" else COLORS.text_dim
    parts: List[str] = [
        "<div style='font-family:\"Cascadia Code\",Consolas,monospace;"
        f"font-size:{font_size}pt;color:{base};white-space:pre-wrap'>"
    ]
    for line in lines:
        lowered = line.lower()
        colour = base
        if line.startswith("$ "):
            colour = COLORS.accent
        elif "[stderr]" in lowered or "traceback" in lowered or lowered.startswith("error"):
            colour = COLORS.danger
        elif " failed" in lowered or "failed " in lowered or lowered.startswith("fail"):
            colour = COLORS.danger
        elif "passed" in lowered or lowered.startswith("ok") or "success" in lowered:
            colour = COLORS.success
        elif lowered.startswith("warning") or "deprecat" in lowered:
            colour = COLORS.warning
        parts.append(f"<span style='color:{colour}'>{html.escape(line) or '&nbsp;'}</span><br>")
    parts.append("</div>")
    return "".join(parts)


def guess_detail_kind(text: Optional[str]) -> str:
    """``diff`` when *text* looks like a unified diff, else ``output``."""
    if not text:
        return "output"
    for line in text.splitlines()[:40]:
        if line.startswith("@@") or line.startswith("--- a/") or line.startswith("+++ b/"):
            return "diff"
    return "output"
