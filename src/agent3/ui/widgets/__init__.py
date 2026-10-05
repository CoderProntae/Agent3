"""Reusable Qt widgets that compose the Agent3 workspace shell."""

from __future__ import annotations

from agent3.ui.widgets.chat_view import ActionCard, ChatView
from agent3.ui.widgets.diff_view import DiffView
from agent3.ui.widgets.editor_panel import CodeEditor, EditorPanel
from agent3.ui.widgets.file_tree import WorkspaceTree
from agent3.ui.widgets.session_list import SessionList
from agent3.ui.widgets.settings_dialog import SettingsDialog
from agent3.ui.widgets.terminal_panel import TerminalPanel
from agent3.ui.widgets.usage_panel import UsagePanel

__all__ = [
    "ActionCard",
    "ChatView",
    "CodeEditor",
    "DiffView",
    "EditorPanel",
    "SessionList",
    "SettingsDialog",
    "TerminalPanel",
    "UsagePanel",
    "WorkspaceTree",
]
