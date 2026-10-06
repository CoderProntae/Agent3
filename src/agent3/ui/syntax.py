"""Lightweight multi-language syntax highlighting for the editor panel.

A full tree-sitter grammar set would bloat the frozen executable, so this is a
regex based highlighter tuned for the languages an agent touches most:
Python, JavaScript/TypeScript, JSON, YAML, shell, C-family, Rust, Go, HTML/CSS
and Markdown.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QTextDocument

from agent3.ui.theme import COLORS

# --------------------------------------------------------------- keywords
PYTHON_KEYWORDS = (
    "False None True and as assert async await break class continue def del elif else except "
    "finally for from global if import in is lambda nonlocal not or pass raise return try while "
    "with yield match case self cls"
).split()

JS_KEYWORDS = (
    "abstract async await break case catch class const continue debugger default delete do else "
    "enum export extends false finally for from function get if implements import in instanceof "
    "interface let new null of package private protected public return set static super switch "
    "this throw true try type typeof undefined var void while yield as satisfies keyof readonly"
).split()

C_KEYWORDS = (
    "auto bool break case char class const constexpr continue default delete do double else enum "
    "explicit extern false float for friend goto if inline int long namespace new nullptr operator "
    "private protected public register return short signed sizeof static struct switch template "
    "this throw true try typedef typename union unsigned using virtual void volatile while var "
    "func package import interface map chan go defer range select type fn let mut impl trait pub "
    "crate match unsafe where async await move dyn static ref box"
).split()

SHELL_KEYWORDS = (
    "if then else elif fi for while until do done case esac function select in return exit export "
    "local readonly set unset shift source alias echo cd mkdir rm cp mv git python pip npm node"
).split()

LANGUAGE_BY_EXTENSION: Dict[str, str] = {
    ".py": "python", ".pyi": "python", ".pyw": "python",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "js", ".tsx": "js",
    ".json": "json", ".jsonc": "json",
    ".yml": "yaml", ".yaml": "yaml",
    ".toml": "toml", ".ini": "toml", ".cfg": "toml",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ps1": "shell", ".bat": "shell", ".cmd": "shell",
    ".c": "c", ".h": "c", ".cpp": "c", ".hpp": "c", ".cc": "c", ".cs": "c", ".java": "c",
    ".go": "c", ".rs": "c", ".kt": "c", ".swift": "c", ".php": "c", ".rb": "c",
    ".html": "markup", ".htm": "markup", ".xml": "markup", ".svg": "markup",
    ".css": "css", ".scss": "css", ".less": "css",
    ".md": "markdown", ".markdown": "markdown", ".rst": "markdown",
    ".sql": "sql",
}


def language_for(path: str) -> str:
    """Infer the highlighting language from a file name."""
    lowered = path.lower()
    for extension, language in LANGUAGE_BY_EXTENSION.items():
        if lowered.endswith(extension):
            return language
    if lowered.endswith(("dockerfile", "makefile")) or "/dockerfile" in lowered:
        return "shell"
    return "plain"


def _fmt(color: str, *, bold: bool = False, italic: bool = False) -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(color))
    if bold:
        fmt.setFontWeight(QFont.Weight.Bold)
    if italic:
        fmt.setFontItalic(True)
    return fmt


class CodeHighlighter(QSyntaxHighlighter):
    """Regex driven highlighter with multi-line string/comment support."""

    def __init__(self, document: QTextDocument, language: str = "python") -> None:
        super().__init__(document)
        self._language = language
        self._rules: List[Tuple[QRegularExpression, QTextCharFormat]] = []
        self._multiline: List[Tuple[QRegularExpression, QRegularExpression, QTextCharFormat]] = []
        self._build_rules()

    # ------------------------------------------------------------- setup
    def set_language(self, language: str) -> None:
        if language == self._language:
            return
        self._language = language
        self._build_rules()
        self.rehighlight()

    def _add(self, pattern: str, fmt: QTextCharFormat) -> None:
        self._rules.append((QRegularExpression(pattern), fmt))

    def _build_rules(self) -> None:
        self._rules = []
        self._multiline = []

        keyword_fmt = _fmt("#ff7b72", bold=True)
        builtin_fmt = _fmt("#79c0ff")
        string_fmt = _fmt("#a5d6ff")
        number_fmt = _fmt("#f2cc60")
        comment_fmt = _fmt(COLORS.text_faint, italic=True)
        decorator_fmt = _fmt("#d2a8ff")
        function_fmt = _fmt("#d2a8ff")
        type_fmt = _fmt("#7ee787")
        key_fmt = _fmt("#79c0ff")

        language = self._language

        def keywords(words: Sequence[str]) -> None:
            if words:
                self._add(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", keyword_fmt)

        if language == "python":
            keywords(PYTHON_KEYWORDS)
            self._add(r"\b(print|len|range|dict|list|set|tuple|str|int|float|bool|open|super|"
                      r"isinstance|enumerate|zip|map|filter|sorted|sum|min|max|abs|type|"
                      r"Exception|ValueError|TypeError|KeyError|OSError|RuntimeError)\b", builtin_fmt)
            self._add(r"\bclass\s+(\w+)", type_fmt)
            self._add(r"\bdef\s+(\w+)", function_fmt)
            self._add(r"^\s*@\w[\w.]*", decorator_fmt)
            self._add(r"#[^\n]*", comment_fmt)
            self._multiline.append(
                (QRegularExpression('"""'), QRegularExpression('"""'), string_fmt)
            )
            self._multiline.append((QRegularExpression("'''"), QRegularExpression("'''"), string_fmt))
        elif language == "js":
            keywords(JS_KEYWORDS)
            self._add(r"\b(console|document|window|Math|JSON|Promise|Array|Object|String|Number)\b", builtin_fmt)
            self._add(r"\bfunction\s+(\w+)", function_fmt)
            self._add(r"//[^\n]*", comment_fmt)
            self._add(r"`[^`]*`", string_fmt)
            self._multiline.append((QRegularExpression(r"/\*"), QRegularExpression(r"\*/"), comment_fmt))
        elif language == "c":
            keywords(C_KEYWORDS)
            self._add(r"^\s*#\s*\w+", decorator_fmt)
            self._add(r"//[^\n]*", comment_fmt)
            self._multiline.append((QRegularExpression(r"/\*"), QRegularExpression(r"\*/"), comment_fmt))
        elif language == "shell":
            keywords(SHELL_KEYWORDS)
            self._add(r"\$\{?\w+\}?", builtin_fmt)
            self._add(r"#[^\n]*", comment_fmt)
        elif language == "json":
            self._add(r'"(\\.|[^"\\])*"\s*:', key_fmt)
            self._add(r"\b(true|false|null)\b", keyword_fmt)
        elif language == "yaml":
            self._add(r"^\s*[-]?\s*[\w.\-]+\s*:", key_fmt)
            self._add(r"\b(true|false|null|yes|no|on|off)\b", keyword_fmt)
            self._add(r"#[^\n]*", comment_fmt)
        elif language == "toml":
            self._add(r"^\s*\[[^\]]+\]", type_fmt)
            self._add(r"^\s*[\w.\-]+\s*=", key_fmt)
            self._add(r"[#;][^\n]*", comment_fmt)
        elif language == "markup":
            self._add(r"</?[\w:.\-]+", keyword_fmt)
            self._add(r"\b[\w\-]+(?==)", key_fmt)
            self._multiline.append((QRegularExpression("<!--"), QRegularExpression("-->"), comment_fmt))
        elif language == "css":
            self._add(r"[.#]?[\w\-]+(?=\s*\{)", type_fmt)
            self._add(r"[\w\-]+(?=\s*:)", key_fmt)
            self._multiline.append((QRegularExpression(r"/\*"), QRegularExpression(r"\*/"), comment_fmt))
        elif language == "markdown":
            self._add(r"^#{1,6}\s.*$", _fmt(COLORS.accent, bold=True))
            self._add(r"\*\*[^*]+\*\*", _fmt(COLORS.text, bold=True))
            self._add(r"`[^`]+`", string_fmt)
            self._add(r"^\s*[-*+]\s", keyword_fmt)
            self._multiline.append((QRegularExpression("```"), QRegularExpression("```"), string_fmt))
        elif language == "sql":
            self._add(r"\b(SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|CREATE|TABLE|INDEX|JOIN|LEFT|"
                      r"RIGHT|INNER|GROUP|ORDER|BY|LIMIT|VALUES|SET|AND|OR|NOT|NULL|PRIMARY|KEY)\b",
                      keyword_fmt)
            self._add(r"--[^\n]*", comment_fmt)

        # Common rules applied to every language (strings & numbers last so
        # that they win over keyword matches inside literals).
        if language not in {"markdown", "plain"}:
            self._add(r"\b\d+(\.\d+)?([eE][+-]?\d+)?\b", number_fmt)
            self._add(r'"(\\.|[^"\\])*"', string_fmt)
            self._add(r"'(\\.|[^'\\])*'", string_fmt)

    # ---------------------------------------------------------- rendering
    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt API)
        for expression, fmt in self._rules:
            iterator = expression.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                group = 1 if match.lastCapturedIndex() >= 1 else 0
                start = match.capturedStart(group)
                length = match.capturedLength(group)
                if length > 0:
                    self.setFormat(start, length, fmt)

        self.setCurrentBlockState(0)
        for index, (start_expr, end_expr, fmt) in enumerate(self._multiline, start=1):
            if self._apply_multiline(text, start_expr, end_expr, fmt, index):
                break

    def _apply_multiline(
        self,
        text: str,
        start_expr: QRegularExpression,
        end_expr: QRegularExpression,
        fmt: QTextCharFormat,
        state: int,
    ) -> bool:
        start_index = 0
        if self.previousBlockState() != state:
            match = start_expr.match(text)
            start_index = match.capturedStart() if match.hasMatch() else -1

        while start_index >= 0:
            end_match = end_expr.match(text, start_index + 3)
            if end_match.hasMatch():
                end_index = end_match.capturedEnd()
                length = end_index - start_index
                self.setFormat(start_index, length, fmt)
                next_match = start_expr.match(text, end_index)
                start_index = next_match.capturedStart() if next_match.hasMatch() else -1
            else:
                self.setCurrentBlockState(state)
                self.setFormat(start_index, len(text) - start_index, fmt)
                return True
        return False
