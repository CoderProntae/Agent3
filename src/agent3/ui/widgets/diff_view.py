"""Coloured unified / side-by-side diff viewer."""

from __future__ import annotations

import html
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from agent3.ui.theme import COLORS, mono_font
from agent3.workspace.diffing import diff_stats, parse_unified_diff, side_by_side


EMPTY_TITLE = "No changes to display"


class DiffView(QWidget):
    """Renders a unified diff; can switch to a two column layout."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._unified = ""
        self._old_text = ""
        self._new_text = ""
        self._path = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        bar = QFrame()
        bar.setStyleSheet(
            f"QFrame {{ background-color: {COLORS.bg_alt};"
            f" border-bottom: 1px solid {COLORS.border_soft}; }}"
            f"QFrame QLabel {{ background: transparent; }}"
        )
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(12, 6, 10, 6)
        bar_layout.setSpacing(9)

        self._title = QLabel(EMPTY_TITLE)
        self._title.setStyleSheet(f"color: {COLORS.text_dim};")
        bar_layout.addWidget(self._title, 1)

        self._stats = QLabel("")
        self._stats.setStyleSheet(f"color: {COLORS.text_faint};")
        bar_layout.addWidget(self._stats, 0)

        self._mode = QComboBox()
        self._mode.addItems(["Inline", "Side by side"])
        self._mode.currentIndexChanged.connect(lambda _i: self._render())
        self._mode.setFixedWidth(130)
        bar_layout.addWidget(self._mode, 0)
        layout.addWidget(bar)

        self._browser = QTextBrowser()
        self._browser.setFont(mono_font(10))
        self._browser.setFrameShape(QFrame.Shape.NoFrame)
        self._browser.setLineWrapMode(QTextBrowser.LineWrapMode.NoWrap)
        self._browser.setStyleSheet(
            f"background-color: {COLORS.bg}; color: {COLORS.text}; padding: 6px 2px;"
        )
        layout.addWidget(self._browser, 1)
        self._show_empty_state()

    def _show_empty_state(self) -> None:
        """Explain the blank panel instead of leaving a black rectangle."""
        # Diffs are rendered without wrapping so columns line up; prose has
        # to wrap or it runs straight off the edge of the panel.
        self._browser.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)
        self._browser.setHtml(
            f'<div style="color:{COLORS.text_faint};padding:28px 22px;'
            f'line-height:165%;">'
            f'<div style="font-size:11pt;color:{COLORS.text_dim};">'
            f"Nothing to review yet</div>"
            f"<div>Every file the agent writes shows up here as a coloured "
            f"diff, and you can open any file from the explorer to read it "
            f"side by side.</div></div>"
        )

    # --------------------------------------------------------------- api
    def show_diff(self, path: str, unified: str, old_text: str = "", new_text: str = "") -> None:
        """Display *unified* (optionally with the raw texts for 2-column mode)."""
        self._path = path
        self._unified = unified or ""
        self._old_text = old_text
        self._new_text = new_text
        stats = diff_stats(self._unified)
        self._title.setText(path or "diff")
        self._stats.setText(f"+{stats.added}  -{stats.removed}")
        self._stats.setStyleSheet(f"color: {COLORS.added_fg};" if stats.added else f"color: {COLORS.text_faint};")
        self._render()

    def clear(self) -> None:
        self._unified = ""
        self._path = ""
        self._title.setText(EMPTY_TITLE)
        self._stats.setText("")
        self._show_empty_state()

    # ----------------------------------------------------------- render
    def _render(self) -> None:
        self._browser.setLineWrapMode(QTextBrowser.LineWrapMode.NoWrap)
        if not self._unified:
            self._browser.setHtml(
                f"<div style='color:{COLORS.text_faint};padding:16px'>No diff selected.</div>"
            )
            return
        if self._mode.currentIndex() == 1 and (self._old_text or self._new_text):
            self._browser.setHtml(self._render_side_by_side())
        else:
            self._browser.setHtml(self._render_inline())

    def _render_inline(self) -> str:
        rows = parse_unified_diff(self._unified)
        parts = [
            f"<body style='background:{COLORS.bg};'>",
            "<table cellspacing='0' cellpadding='0' style='width:100%;font-family:Consolas,monospace;font-size:10pt'>",
        ]
        for row in rows:
            text = html.escape(row.text).replace(" ", "&nbsp;") or "&nbsp;"
            if row.kind == "added":
                style = f"background:{COLORS.added_bg};color:{COLORS.added_fg};"
                marker = "+"
            elif row.kind == "removed":
                style = f"background:{COLORS.removed_bg};color:{COLORS.removed_fg};"
                marker = "-"
            elif row.kind == "hunk":
                style = f"color:{COLORS.hunk_fg};background:{COLORS.panel_alt};"
                marker = "@"
            elif row.kind == "header":
                style = f"color:{COLORS.text_faint};"
                marker = ""
            else:
                style = f"color:{COLORS.text_dim};"
                marker = " "
            old_no = str(row.old_lineno or "")
            new_no = str(row.new_lineno or "")
            parts.append(
                f"<tr style='{style}'>"
                f"<td style='width:46px;text-align:right;color:{COLORS.text_faint};padding-right:8px'>{old_no}</td>"
                f"<td style='width:46px;text-align:right;color:{COLORS.text_faint};padding-right:8px'>{new_no}</td>"
                f"<td style='width:14px;'>{marker}</td>"
                f"<td style='white-space:pre'>{text}</td>"
                "</tr>"
            )
        parts.append("</table></body>")
        return "".join(parts)

    def _render_side_by_side(self) -> str:
        rows = side_by_side(self._old_text, self._new_text)
        parts = [
            f"<body style='background:{COLORS.bg};'>",
            "<table cellspacing='0' cellpadding='2' style='width:100%;font-family:Consolas,monospace;font-size:10pt'>",
            f"<tr><th style='color:{COLORS.text_faint};text-align:left'>before</th>"
            f"<th style='color:{COLORS.text_faint};text-align:left'>after</th></tr>",
        ]
        for kind, left, right in rows:
            if kind == "equal":
                left_style = right_style = f"color:{COLORS.text_dim};"
            elif kind == "replace":
                left_style = f"background:{COLORS.removed_bg};color:{COLORS.removed_fg};"
                right_style = f"background:{COLORS.added_bg};color:{COLORS.added_fg};"
            elif kind == "delete":
                left_style = f"background:{COLORS.removed_bg};color:{COLORS.removed_fg};"
                right_style = ""
            else:
                left_style = ""
                right_style = f"background:{COLORS.added_bg};color:{COLORS.added_fg};"
            left_html = html.escape(left).replace(" ", "&nbsp;") or "&nbsp;"
            right_html = html.escape(right).replace(" ", "&nbsp;") or "&nbsp;"
            parts.append(
                f"<tr><td style='width:50%;white-space:pre;{left_style}'>{left_html}</td>"
                f"<td style='width:50%;white-space:pre;{right_style}'>{right_html}</td></tr>"
            )
        parts.append("</table></body>")
        return "".join(parts)
