"""Dark "workspace" theme: colour tokens, Qt palette and the global stylesheet.

Everything visual is centralised here so the whole application can be
re-skinned (or given a light mode) from a single place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Palette:
    """Named colour tokens used across the UI."""

    bg: str = "#0d0f14"
    bg_alt: str = "#13161d"
    panel: str = "#171b24"
    panel_alt: str = "#1e2330"
    border: str = "#252b38"
    border_soft: str = "#1b202a"
    text: str = "#e6e9ef"
    text_dim: str = "#9aa4b5"
    text_faint: str = "#6b7484"
    accent: str = "#4f8cff"
    accent_dim: str = "#2f5fb8"
    success: str = "#3fb950"
    warning: str = "#d29922"
    danger: str = "#f85149"
    purple: str = "#a371f7"
    added_bg: str = "#13261a"
    added_fg: str = "#56d364"
    removed_bg: str = "#2b1416"
    removed_fg: str = "#f87171"
    hunk_fg: str = "#79c0ff"
    selection: str = "#264f78"

    # Syntax token colours, shared by the editor highlighter and the HTML
    # markdown renderer so a snippet looks identical in the chat and in a
    # tab.
    code_keyword: str = "#ff7b72"
    code_builtin: str = "#79c0ff"
    code_string: str = "#a5d6ff"
    code_number: str = "#f2cc60"
    code_comment: str = "#6b7484"
    code_function: str = "#d2a8ff"
    code_type: str = "#7ee787"


COLORS = Palette()

MONO_FONTS = "'Cascadia Code','JetBrains Mono','Fira Code','Consolas','DejaVu Sans Mono',monospace"
UI_FONTS = "'Segoe UI','Inter','Ubuntu','Helvetica Neue',sans-serif"


def role_colors() -> Dict[str, str]:
    """Accent colour per chat role / card type."""
    return {
        "user": COLORS.accent,
        "assistant": COLORS.purple,
        "tool": COLORS.warning,
        "system": COLORS.text_faint,
        "error": COLORS.danger,
        "success": COLORS.success,
    }


def build_stylesheet(accent: str = COLORS.accent, font_size: int = 10) -> str:
    """Return the global Qt stylesheet."""
    c = COLORS
    return f"""
    * {{
        font-family: {UI_FONTS};
        font-size: {font_size}pt;
        outline: 0;
    }}
    QWidget {{
        background-color: {c.bg};
        color: {c.text};
    }}
    QMainWindow, QDialog {{ background-color: {c.bg}; }}

    QToolTip {{
        background-color: {c.panel_alt};
        color: {c.text};
        border: 1px solid {c.border};
        padding: 4px 6px;
    }}

    /* ---------------------------------------------------------- frames */
    QFrame#Card, QFrame#Panel {{
        background-color: {c.panel};
        border: 1px solid {c.border_soft};
        border-radius: 10px;
    }}
    /* A QLabel inherits the window background, which paints a dark slab
       across any card it sits on. Every label inside a card has to be
       transparent for the card to read as one surface. */
    QFrame#Card QLabel, QFrame#Panel QLabel,
    QFrame#Card QCheckBox, QFrame#Panel QCheckBox {{ background: transparent; }}
    QFrame#Separator {{ background-color: {c.border}; max-height: 1px; }}

    /* A header band is the only divider between the stacked sidebar panes,
       now that splitter handles stay invisible until you reach for one.
       The band lives on the container so it spans the full width even when
       the row also holds a counter or a button. */
    QWidget#SectionHeader {{
        background-color: {c.bg_alt};
        border-top: 1px solid {c.border_soft};
        border-bottom: 1px solid {c.border_soft};
    }}
    QWidget#SectionHeader QLabel, QWidget#SectionHeader QPushButton {{
        background: transparent;
    }}
    QLabel#SectionTitle {{
        color: {c.text_faint};
        font-size: {max(8, font_size - 2)}pt;
        font-weight: 700;
        letter-spacing: 1.4px;
        padding: 0;
        background: transparent;
    }}
    QLabel#Hint {{ color: {c.text_faint}; }}
    QLabel#StatusBadge {{
        color: {c.text_dim};
        background-color: {c.panel_alt};
        border: 1px solid {c.border_soft};
        border-radius: 10px;
        padding: 3px 11px;
    }}

    /* --------------------------------------------------------- buttons */
    QPushButton {{
        background-color: {c.panel_alt};
        border: 1px solid {c.border};
        border-radius: 8px;
        padding: 7px 14px;
        color: {c.text};
    }}
    QPushButton:hover {{ background-color: #222836; border-color: {accent}; }}
    QPushButton:pressed {{ background-color: #1a1f29; }}
    QPushButton:disabled {{ color: {c.text_faint}; border-color: {c.border_soft}; }}
    QPushButton#Primary {{
        background-color: {accent};
        border: 1px solid {accent};
        color: #ffffff;
        font-weight: 600;
    }}
    QPushButton#Primary:hover {{ background-color: #6ba0ff; }}
    QPushButton#Primary:disabled {{ background-color: {c.accent_dim}; color: #cfd8e8; }}
    QPushButton#Danger {{ border-color: {c.danger}; color: {c.danger}; }}
    QPushButton#Danger:hover {{ background-color: #2a1618; }}
    QPushButton#Ghost {{ background: transparent; border: none; color: {c.text_dim}; padding: 4px 8px; }}
    QPushButton#Ghost:hover {{ color: {c.text}; background-color: {c.panel_alt}; }}

    /* ---------------------------------------------------------- inputs */
    QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {c.bg_alt};
        border: 1px solid {c.border};
        border-radius: 8px;
        padding: 8px 10px;
        selection-background-color: {c.selection};
        color: {c.text};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus,
    QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {accent}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background-color: {c.panel_alt};
        border: 1px solid {c.border};
        selection-background-color: {c.selection};
    }}
    QCheckBox, QRadioButton {{ spacing: 7px; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 15px; height: 15px;
        border: 1px solid {c.border};
        border-radius: 3px;
        background-color: {c.bg_alt};
    }}
    QCheckBox::indicator:checked {{ background-color: {accent}; border-color: {accent}; }}

    /* ------------------------------------------------------------ tree */
    QTreeView, QTreeWidget, QListView, QListWidget, QTableView {{
        background-color: {c.panel};
        border: none;
        alternate-background-color: {c.panel_alt};
        selection-background-color: {c.selection};
        selection-color: {c.text};
    }}
    QTreeView::item, QListWidget::item {{ padding: 5px 4px; border-radius: 6px; }}
    QTreeView::item:hover, QListWidget::item:hover {{ background-color: {c.panel_alt}; }}
    QHeaderView::section {{
        background-color: {c.panel_alt};
        color: {c.text_dim};
        border: none;
        border-bottom: 1px solid {c.border};
        padding: 5px;
    }}

    /* ------------------------------------------------------------ tabs */
    QTabWidget::pane {{ border: 1px solid {c.border_soft}; border-radius: 10px; top: -1px; }}
    QTabBar::tab {{
        background-color: {c.bg_alt};
        color: {c.text_dim};
        border: 1px solid {c.border};
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 7px 16px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{ background-color: {c.panel}; color: {c.text}; border-bottom: 2px solid {accent}; }}
    QTabBar::tab:hover {{ color: {c.text}; }}
    QTabBar::close-button {{ subcontrol-position: right; }}

    /* --------------------------------------------------------- splitter */
    /* The handle is breathing room you can drag, not a visible rule: it
       only picks up a colour when the pointer is on it. */
    QSplitter::handle {{ background-color: transparent; }}
    QSplitter::handle:horizontal {{ width: 8px; }}
    QSplitter::handle:vertical {{ height: 8px; }}
    QSplitter::handle:hover {{ background-color: {c.border}; }}
    QSplitter::handle:pressed {{ background-color: {accent}; }}

    /* ------------------------------------------------------- scrollbars */
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 3px 2px; }}
    QScrollBar::handle:vertical {{ background: #262d3b; border-radius: 4px; min-height: 36px; }}
    QScrollBar::handle:vertical:hover {{ background: #394256; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px 3px; }}
    QScrollBar::handle:horizontal {{ background: #262d3b; border-radius: 4px; min-width: 36px; }}
    QScrollBar::handle:horizontal:hover {{ background: #394256; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

    /* --------------------------------------------------------- progress */
    QProgressBar {{
        background-color: {c.bg_alt};
        border: 1px solid {c.border};
        border-radius: 5px;
        height: 10px;
        text-align: center;
        color: {c.text_dim};
    }}
    QProgressBar::chunk {{ background-color: {accent}; border-radius: 4px; }}

    /* ----------------------------------------------------------- menus */
    QMenuBar {{ background-color: {c.bg_alt}; border-bottom: 1px solid {c.border_soft}; padding: 2px 4px; }}
    QMenuBar::item {{ padding: 6px 12px; border-radius: 6px; background: transparent; }}
    QMenuBar::item:selected {{ background-color: {c.panel_alt}; }}
    QMenu {{ background-color: {c.panel_alt}; border: 1px solid {c.border}; padding: 5px; }}
    QMenu::item {{ padding: 6px 24px 6px 18px; border-radius: 4px; }}
    QMenu::item:selected {{ background-color: {c.selection}; }}
    QMenu::separator {{ height: 1px; background: {c.border}; margin: 4px 8px; }}

    /* --------------------------------------------------------- toolbar */
    QToolBar {{ background-color: {c.bg_alt}; border-bottom: 1px solid {c.border_soft}; spacing: 8px; padding: 7px 10px; }}
    QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 5px; padding: 5px 9px; }}
    QToolButton:hover {{ background-color: {c.panel_alt}; border-color: {c.border}; }}
    QStatusBar {{ background-color: {c.bg_alt}; border-top: 1px solid {c.border_soft}; color: {c.text_dim}; padding: 3px 6px; }}
    QStatusBar QLabel {{ background: transparent; }}
    QStatusBar::item {{ border: none; }}

    /* ------------------------------------------------------- scroll area */
    QScrollArea {{ border: none; background-color: {c.bg}; }}
    QGroupBox {{
        border: 1px solid {c.border};
        border-radius: 8px;
        margin-top: 14px;
        padding-top: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: {c.text_dim};
        font-weight: 600;
    }}
    """


def apply_theme(app: QApplication, accent: str = COLORS.accent, font_size: int = 10) -> None:
    """Apply the dark palette + stylesheet to *app*."""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS.bg))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS.text))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS.bg_alt))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS.panel_alt))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS.text))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS.panel_alt))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS.text))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS.selection))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(COLORS.text))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLORS.panel_alt))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(COLORS.text))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(COLORS.text_faint))
    app.setPalette(palette)
    app.setStyleSheet(build_stylesheet(accent, font_size))


def mono_font(size: int = 11) -> QFont:
    """Return the preferred monospace font."""
    font = QFont("Cascadia Code")
    font.setStyleHint(QFont.StyleHint.Monospace)
    for family in ("Cascadia Code", "JetBrains Mono", "Consolas", "DejaVu Sans Mono", "Courier New"):
        font.setFamily(family)
        break
    font.setPointSize(size)
    font.setFixedPitch(True)
    return font
