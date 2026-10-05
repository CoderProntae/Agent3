"""Chat transcript with markdown bubbles and live agent action cards."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextOption
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from agent3.ui.diff_render import (
    diff_badge_text,
    guess_detail_kind,
    render_diff_badge,
    render_diff_html,
    render_output_html,
)
from agent3.ui.theme import COLORS, mono_font, role_colors
from agent3.workspace.diffing import diff_stats

#: Tallest an action-card detail pane may grow before it scrolls internally.
DETAILS_MAX_HEIGHT = 340

MARKDOWN_CSS = f"""
    body {{ color: {COLORS.text}; }}
    p {{ margin: 2px 0 8px 0; line-height: 145%; }}
    h1, h2, h3, h4 {{ color: {COLORS.text}; margin: 10px 0 6px 0; }}
    code {{
        background-color: {COLORS.bg_alt};
        color: #a5d6ff;
        border: 1px solid {COLORS.border};
        padding: 1px 4px;
        border-radius: 3px;
        font-family: 'Cascadia Code','Consolas',monospace;
    }}
    pre {{
        background-color: {COLORS.bg};
        border: 1px solid {COLORS.border};
        border-radius: 6px;
        padding: 10px;
        font-family: 'Cascadia Code','Consolas',monospace;
        color: #d1d9e6;
    }}
    a {{ color: {COLORS.accent}; }}
    ul, ol {{ margin-left: 18px; }}
    blockquote {{ border-left: 3px solid {COLORS.border}; margin-left: 4px; padding-left: 10px; color: {COLORS.text_dim}; }}
    table {{ border-collapse: collapse; }}
    td, th {{ border: 1px solid {COLORS.border}; padding: 4px 8px; }}
"""


class _AutoTextBrowser(QTextBrowser):
    """A QTextBrowser that grows to fit its content (no inner scrollbar)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setStyleSheet("background: transparent; border: none;")
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.document().setDefaultStyleSheet(MARKDOWN_CSS)
        self.document().documentLayout().documentSizeChanged.connect(self._fit)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _fit(self) -> None:
        height = int(self.document().size().height()) + 8
        self.setFixedHeight(max(24, min(height, 20000)))

    def set_markdown(self, text: str) -> None:
        self.setMarkdown(text)
        self._fit()


class MessageBubble(QFrame):
    """One chat turn (user / assistant / system / error)."""

    def __init__(self, role: str, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._role = role
        self._buffer = text
        accent = role_colors().get(role, COLORS.text_dim)
        self.setStyleSheet(
            f"QFrame#Card {{ background-color: {COLORS.panel}; border: 1px solid {COLORS.border};"
            f" border-left: 3px solid {accent}; border-radius: 8px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(4)

        header = QLabel(
            {"user": "YOU", "assistant": "AGENT3", "system": "SYSTEM", "error": "ERROR"}.get(
                role, role.upper()
            )
        )
        header_font = QFont()
        header_font.setPointSize(8)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet(f"color: {accent}; letter-spacing: 1px;")
        layout.addWidget(header)

        self._body = _AutoTextBrowser()
        self._body.set_markdown(text)
        layout.addWidget(self._body)

    @property
    def text(self) -> str:
        return self._buffer

    def set_text(self, text: str) -> None:
        self._buffer = text
        self._body.set_markdown(text)

    def append_text(self, delta: str) -> None:
        self._buffer += delta
        self._body.set_markdown(self._buffer)


class ActionCard(QFrame):
    """Live card describing one agent tool invocation."""

    def __init__(self, title: str, subtitle: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._status = "running"
        self._details_text = ""
        self.setStyleSheet(
            f"QFrame#Card {{ background-color: {COLORS.bg_alt}; border: 1px solid {COLORS.border};"
            f" border-left: 3px solid {COLORS.warning}; border-radius: 8px; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(5)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._icon = QLabel("●")
        self._icon.setStyleSheet(f"color: {COLORS.warning}; font-size: 13px;")
        header.addWidget(self._icon, 0)

        self._title = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        self._title.setFont(title_font)
        self._title.setStyleSheet(f"color: {COLORS.text};")
        header.addWidget(self._title, 1)

        self._badge = QLabel("")
        self._badge.setTextFormat(Qt.TextFormat.RichText)
        self._badge.setVisible(False)
        badge_font = QFont()
        badge_font.setBold(True)
        self._badge.setFont(badge_font)
        header.addWidget(self._badge, 0)

        self._meta = QLabel("")
        self._meta.setStyleSheet(f"color: {COLORS.text_faint};")
        header.addWidget(self._meta, 0)

        self._toggle = QPushButton("Details")
        self._toggle.setObjectName("Ghost")
        self._toggle.setCheckable(True)
        self._toggle.setVisible(False)
        self._toggle.toggled.connect(self._on_toggle)
        header.addWidget(self._toggle, 0)
        outer.addLayout(header)

        self._subtitle = QLabel(subtitle)
        self._subtitle.setWordWrap(True)
        self._subtitle.setStyleSheet(f"color: {COLORS.text_dim};")
        self._subtitle.setVisible(bool(subtitle))
        outer.addWidget(self._subtitle)

        # Rich detail pane: a real diff review for file changes, colourised
        # console output for commands.
        self._details = QTextBrowser()
        self._details.setReadOnly(True)
        self._details.setFont(mono_font(9))
        self._details.setVisible(False)
        self._details.setMinimumHeight(28)
        self._details.setMaximumHeight(DETAILS_MAX_HEIGHT)
        self._details.document().documentLayout().documentSizeChanged.connect(
            lambda _size: self._fit_details()
        )
        self._details.setLineWrapMode(QTextBrowser.LineWrapMode.NoWrap)
        self._details.setFrameShape(QFrame.Shape.NoFrame)
        self._details.setStyleSheet(
            f"QTextBrowser {{ background-color: {COLORS.bg}; border: 1px solid {COLORS.border};"
            f" border-radius: 6px; color: {COLORS.text_dim}; }}"
        )
        outer.addWidget(self._details)

    # ------------------------------------------------------------ updates
    def _fit_details(self) -> None:
        """Shrink-wrap the detail pane around its content."""
        document = self._details.document()
        document.setTextWidth(max(120, self._details.viewport().width()))
        height = int(document.size().height()) + 12
        self._details.setFixedHeight(max(28, min(height, DETAILS_MAX_HEIGHT)))

    def set_subtitle(self, text: str) -> None:
        self._subtitle.setText(text)
        self._subtitle.setVisible(bool(text))

    def set_details(self, text: str, *, kind: str = "auto") -> None:
        """Fill the collapsible detail pane.

        ``kind`` is ``diff``, ``output`` or ``auto`` (sniff the content).
        A diff is rendered as a colourised review table with old/new line
        numbers; anything else is rendered as console output with errors in
        red and passing tests in green.
        """
        self._details_text = text or ""
        if not self._details_text:
            self._details.setHtml("")
            self._toggle.setVisible(False)
            return
        resolved = guess_detail_kind(self._details_text) if kind == "auto" else kind
        if resolved == "diff":
            self.set_diff(self._details_text)
            return
        self._details.setHtml(render_output_html(self._details_text[:60000]))
        self._toggle.setVisible(True)

    def set_diff(self, unified: str) -> None:
        """Render *unified* as a green/red review table and show the badge."""
        self._details_text = unified or ""
        if not self._details_text.strip():
            self._details.setHtml("")
            self._toggle.setVisible(False)
            return
        self._details.setHtml(render_diff_html(self._details_text[:200000]))
        stats = diff_stats(self._details_text)
        badge = render_diff_badge(stats)
        if badge:
            self._badge.setText(badge)
            self._badge.setVisible(True)
        self._toggle.setVisible(True)
        self._toggle.setText("Hide diff" if self._toggle.isChecked() else "Diff")

    @property
    def details_text(self) -> str:
        return self._details_text

    def badge_text(self) -> str:
        """Plain-text form of the +/- badge (used by tests)."""
        return diff_badge_text(self._details_text)

    def set_status(self, status: str, meta: str = "") -> None:
        """``running`` | ``success`` | ``error`` | ``blocked``."""
        self._status = status
        colors = {
            "running": COLORS.warning,
            "success": COLORS.success,
            "error": COLORS.danger,
            "blocked": COLORS.danger,
        }
        glyphs = {"running": "●", "success": "✓", "error": "✕", "blocked": "!"}
        color = colors.get(status, COLORS.text_dim)
        self._icon.setText(glyphs.get(status, "●"))
        self._icon.setStyleSheet(f"color: {color}; font-size: 13px;")
        self.setStyleSheet(
            f"QFrame#Card {{ background-color: {COLORS.bg_alt}; border: 1px solid {COLORS.border};"
            f" border-left: 3px solid {color}; border-radius: 8px; }}"
        )
        if meta:
            self._meta.setText(meta)

    def _on_toggle(self, checked: bool) -> None:
        self._details.setVisible(checked)
        is_diff = guess_detail_kind(self._details_text) == "diff"
        if checked:
            self._toggle.setText("Hide diff" if is_diff else "Hide")
        else:
            self._toggle.setText("Diff" if is_diff else "Details")


class ThinkingBlock(QFrame):
    """Collapsible panel holding the model's reasoning trace.

    Ollama streams reasoning in a separate ``message.thinking`` field, so it
    can be shown without polluting the answer. It is collapsed by default -
    it is context for the curious, not the result.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ThinkingBlock")
        self.setStyleSheet(
            f"""
            QFrame#ThinkingBlock {{
                background-color: {COLORS.panel};
                border: 1px dashed {COLORS.border};
                border-radius: 8px;
            }}
            """
        )
        self._text = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 8)
        layout.setSpacing(4)

        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setText("Reasoning")
        self.toggle.setStyleSheet(
            f"QToolButton {{ color: {COLORS.purple}; border: none; font-weight: 600; }}"
        )
        self.toggle.toggled.connect(self._on_toggle)
        layout.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignLeft)

        self.body = QPlainTextEdit()
        self.body.setReadOnly(True)
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setFont(mono_font(9))
        self.body.setVisible(False)
        self.body.setMaximumHeight(240)
        self.body.setStyleSheet(
            f"QPlainTextEdit {{ background-color: transparent; color: {COLORS.text_dim};"
            f" border: none; }}"
        )
        layout.addWidget(self.body)

    def append(self, delta: str) -> None:
        self._text += delta
        self.body.setPlainText(self._text)
        self.body.verticalScrollBar().setValue(self.body.verticalScrollBar().maximum())
        words = len(self._text.split())
        self.toggle.setText(f"Reasoning · {words} word(s)")

    @property
    def text(self) -> str:
        return self._text

    def _on_toggle(self, checked: bool) -> None:
        self.body.setVisible(checked)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )


class ChatView(QScrollArea):
    """Scrollable transcript of bubbles and action cards."""

    link_clicked = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(14, 14, 14, 14)
        self._layout.setSpacing(10)
        self._layout.addStretch(1)
        self.setWidget(self._container)

        self._streaming: Optional[MessageBubble] = None
        self._thinking: Optional[ThinkingBlock] = None

    # -------------------------------------------------------------- items
    def _insert(self, widget: QWidget) -> None:
        self._layout.insertWidget(self._layout.count() - 1, widget)
        self.scroll_to_bottom()

    def add_message(self, role: str, text: str) -> MessageBubble:
        bubble = MessageBubble(role, text)
        self._insert(bubble)
        return bubble

    def add_action_card(self, title: str, subtitle: str = "") -> ActionCard:
        card = ActionCard(title, subtitle)
        self._insert(card)
        return card

    def add_notice(self, text: str, kind: str = "system") -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        color = {"system": COLORS.text_faint, "error": COLORS.danger, "success": COLORS.success}.get(
            kind, COLORS.text_faint
        )
        label.setStyleSheet(f"color: {color}; padding: 2px 6px;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._insert(label)
        return label

    # ---------------------------------------------------------- streaming
    def begin_thinking(self) -> "ThinkingBlock":
        """Open a reasoning block for the step that is starting."""
        block = ThinkingBlock()
        self._insert(block)
        self._thinking = block
        return block

    def append_thinking_delta(self, delta: str) -> None:
        """Stream a reasoning fragment into the current block."""
        if not delta:
            return
        if self._thinking is None:
            self.begin_thinking()
        assert self._thinking is not None
        self._thinking.append(delta)
        self.scroll_to_bottom()

    def end_thinking(self) -> None:
        """Close the reasoning block; drop it when nothing was emitted."""
        if self._thinking is None:
            return
        if not self._thinking.text.strip():
            self._thinking.setParent(None)
            self._thinking.deleteLater()
        self._thinking = None

    @property
    def thinking_text(self) -> str:
        return self._thinking.text if self._thinking is not None else ""

    def begin_assistant_stream(self) -> MessageBubble:
        self._streaming = self.add_message("assistant", "")
        return self._streaming

    def append_assistant_delta(self, delta: str) -> None:
        if self._streaming is None:
            self.begin_assistant_stream()
        assert self._streaming is not None
        self._streaming.append_text(delta)
        self.scroll_to_bottom()

    def end_assistant_stream(self, final_text: Optional[str] = None) -> None:
        if self._streaming is None:
            return
        if final_text is not None:
            self._streaming.set_text(final_text)
        if not self._streaming.text.strip():
            self._streaming.setParent(None)
            self._streaming.deleteLater()
        self._streaming = None
        self.end_thinking()

    @property
    def is_streaming(self) -> bool:
        return self._streaming is not None

    # ------------------------------------------------------------- utils
    def clear(self) -> None:
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._streaming = None
        self._thinking = None

    def scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
