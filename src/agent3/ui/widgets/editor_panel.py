"""Tabbed code editor with line numbers, syntax highlighting and a diff tab."""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QTextFormat, QTextCursor
from PySide6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from agent3.core.logging_setup import get_logger
from agent3.ui.syntax import CodeHighlighter, language_for
from agent3.ui.theme import COLORS, mono_font
from agent3.ui.widgets.diff_view import DiffView
from agent3.workspace.fs import WorkspaceFS

logger = get_logger(__name__)


class _LineNumberArea(QWidget):
    """Gutter painted by :class:`CodeEditor`."""

    def __init__(self, editor: "CodeEditor") -> None:
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._editor.paint_line_numbers(event)


class CodeEditor(QPlainTextEdit):
    """Plain text editor with gutter, current-line highlight and highlighting."""

    dirty_changed = Signal(bool)

    def __init__(self, path: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.path = path
        self._dirty = False
        self._original = ""

        self.setFont(mono_font(11))
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {COLORS.bg}; color: {COLORS.text};"
            f" border: none; selection-background-color: {COLORS.selection}; }}"
        )

        self._gutter = _LineNumberArea(self)
        self.blockCountChanged.connect(lambda _n: self._update_gutter_width())
        self.updateRequest.connect(self._update_gutter)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self.textChanged.connect(self._on_text_changed)

        self._highlighter = CodeHighlighter(self.document(), language_for(path))
        self._update_gutter_width()
        self._highlight_current_line()

    # ------------------------------------------------------------ content
    def set_content(self, text: str, path: Optional[str] = None) -> None:
        if path is not None:
            self.path = path
            self._highlighter.set_language(language_for(path))
        self._original = text
        self.blockSignals(True)
        self.setPlainText(text)
        self.blockSignals(False)
        self._set_dirty(False)

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_saved(self) -> None:
        self._original = self.toPlainText()
        self._set_dirty(False)

    def original_text(self) -> str:
        return self._original

    def _on_text_changed(self) -> None:
        self._set_dirty(self.toPlainText() != self._original)

    def _set_dirty(self, value: bool) -> None:
        if value != self._dirty:
            self._dirty = value
            self.dirty_changed.emit(value)

    def goto_line(self, line: int) -> None:
        cursor = QTextCursor(self.document().findBlockByLineNumber(max(0, line - 1)))
        self.setTextCursor(cursor)
        self.centerCursor()

    # ------------------------------------------------------------- gutter
    def line_number_area_width(self) -> int:
        digits = max(3, len(str(max(1, self.blockCount()))))
        return 14 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_gutter_width(self) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_gutter(self, rect: QRect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_gutter_width()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        contents = self.contentsRect()
        self._gutter.setGeometry(
            QRect(contents.left(), contents.top(), self.line_number_area_width(), contents.height())
        )

    def paint_line_numbers(self, event) -> None:
        painter = QPainter(self._gutter)
        painter.fillRect(event.rect(), QColor(COLORS.bg_alt))
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        current = self.textCursor().blockNumber()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QColor(COLORS.text if number == current else COLORS.text_faint))
                painter.drawText(
                    0,
                    top,
                    self._gutter.width() - 7,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    str(number + 1),
                )
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            number += 1

    def _highlight_current_line(self) -> None:
        selections = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            selection.format.setBackground(QColor(COLORS.panel_alt))
            selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            selections.append(selection)
        self.setExtraSelections(selections)


class EditorPanel(QWidget):
    """Tab container for open files plus a pinned diff tab."""

    file_saved = Signal(str)
    dirty_changed = Signal(bool)

    DIFF_TAB_TITLE = "Diff"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._fs: Optional[WorkspaceFS] = None
        self._editors: Dict[str, CodeEditor] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.setDocumentMode(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)
        layout.addWidget(self._tabs, 1)

        self.diff_view = DiffView()
        self._tabs.addTab(self.diff_view, self.DIFF_TAB_TITLE)
        self._tabs.tabBar().setTabButton(0, self._tabs.tabBar().ButtonPosition.RightSide, None)

        self._placeholder = QLabel("Open a file from the explorer, or let the agent edit one.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(f"color: {COLORS.text_faint};")

    # ------------------------------------------------------------- setup
    def set_workspace(self, fs: Optional[WorkspaceFS]) -> None:
        self._fs = fs
        self.close_all()
        self.diff_view.clear()

    # -------------------------------------------------------------- files
    def open_file(self, relative: str, *, line: int = 0) -> Optional[CodeEditor]:
        """Open (or focus) *relative* in a tab."""
        if self._fs is None:
            return None
        if relative in self._editors:
            editor = self._editors[relative]
            self._tabs.setCurrentWidget(editor)
            if line:
                editor.goto_line(line)
            return editor
        try:
            if self._fs.is_binary(relative):
                self._show_message(f"{relative} is a binary file and cannot be displayed.")
                return None
            content = self._fs.read_text(relative)
        except (OSError, ValueError, PermissionError) as exc:
            self._show_message(f"Cannot open {relative}: {exc}")
            return None

        editor = CodeEditor(relative)
        editor.set_content(content, relative)
        editor.dirty_changed.connect(lambda dirty, path=relative: self._on_dirty(path, dirty))
        self._editors[relative] = editor
        index = self._tabs.addTab(editor, relative.split("/")[-1])
        self._tabs.setTabToolTip(index, relative)
        self._tabs.setCurrentIndex(index)
        if line:
            editor.goto_line(line)
        return editor

    def reload_file(self, relative: str) -> None:
        """Refresh an open tab after the agent changed the file on disk."""
        editor = self._editors.get(relative)
        if editor is None or self._fs is None:
            return
        try:
            editor.set_content(self._fs.read_text(relative), relative)
        except (OSError, ValueError) as exc:  # pragma: no cover - file vanished
            logger.debug("reload failed for %s: %s", relative, exc)

    def current_editor(self) -> Optional[CodeEditor]:
        widget = self._tabs.currentWidget()
        return widget if isinstance(widget, CodeEditor) else None

    def save_current(self) -> bool:
        editor = self.current_editor()
        if editor is None or self._fs is None:
            return False
        try:
            self._fs.write_text(editor.path, editor.toPlainText())
        except (OSError, PermissionError) as exc:
            self._show_message(f"Cannot save {editor.path}: {exc}")
            return False
        editor.mark_saved()
        self._refresh_tab_title(editor.path)
        self.file_saved.emit(editor.path)
        return True

    def save_all(self) -> int:
        saved = 0
        for path, editor in list(self._editors.items()):
            if editor.dirty and self._fs is not None:
                try:
                    self._fs.write_text(path, editor.toPlainText())
                    editor.mark_saved()
                    self._refresh_tab_title(path)
                    self.file_saved.emit(path)
                    saved += 1
                except (OSError, PermissionError) as exc:  # pragma: no cover
                    logger.warning("save failed for %s: %s", path, exc)
        return saved

    def has_unsaved(self) -> bool:
        return any(editor.dirty for editor in self._editors.values())

    def close_all(self) -> None:
        for index in reversed(range(self._tabs.count())):
            widget = self._tabs.widget(index)
            if isinstance(widget, CodeEditor):
                self._tabs.removeTab(index)
                widget.deleteLater()
        self._editors.clear()

    # --------------------------------------------------------------- diff
    def show_diff(self, path: str, unified: str, old_text: str = "", new_text: str = "") -> None:
        self.diff_view.show_diff(path, unified, old_text, new_text)
        self._tabs.setCurrentWidget(self.diff_view)
        self._tabs.setTabText(self._tabs.indexOf(self.diff_view), f"Diff · {path.split('/')[-1]}")

    def focus_diff(self) -> None:
        self._tabs.setCurrentWidget(self.diff_view)

    # ------------------------------------------------------------ private
    def _close_tab(self, index: int) -> None:
        widget = self._tabs.widget(index)
        if not isinstance(widget, CodeEditor):
            return
        self._editors.pop(widget.path, None)
        self._tabs.removeTab(index)
        widget.deleteLater()
        self.dirty_changed.emit(self.has_unsaved())

    def _on_dirty(self, path: str, _dirty: bool) -> None:
        self._refresh_tab_title(path)
        self.dirty_changed.emit(self.has_unsaved())

    def _refresh_tab_title(self, path: str) -> None:
        editor = self._editors.get(path)
        if editor is None:
            return
        index = self._tabs.indexOf(editor)
        if index >= 0:
            name = path.split("/")[-1]
            self._tabs.setTabText(index, f"● {name}" if editor.dirty else name)

    def _show_message(self, text: str) -> None:
        self._placeholder.setText(text)
        logger.info(text)
