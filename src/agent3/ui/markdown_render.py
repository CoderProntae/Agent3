"""Markdown to Qt-rich-text renderer for the chat transcript.

Why not ``QTextDocument.setMarkdown``?
-------------------------------------
Qt can parse markdown, but it builds the document *programmatically* and
never consults the document's default stylesheet. Everything the theme has
to say about inline code, tables, blockquotes and headings is therefore
discarded: the transcript comes out as undifferentiated grey text with
monospace fragments, which is exactly how Agent3 used to look.

Rendering to HTML and calling ``setHtml`` instead puts the stylesheet back in
charge, and - more importantly - lets us emit things Qt's markdown importer
simply does not produce: syntax-highlighted code blocks, header-shaded
tables, checkbox lists for the agent's plan, and language captions.

What is supported
-----------------
Blocks
    ATX headings, fenced code (``` and ~~~, with an info string), indented
    code, blockquotes (nested, with blocks inside), unordered/ordered lists
    (nested by indentation, with task-list checkboxes), pipe tables with
    alignment, horizontal rules, paragraphs.
Inline
    ``code spans``, **bold**, *italic*, ***both***, ~~strikethrough~~,
    [links](url), bare URLs, autolinks, images (rendered as a labelled
    link), hard line breaks.

Everything is HTML-escaped before any markup is produced, and the output is
restricted to the tag/attribute subset Qt's rich text engine actually
understands - no CSS that Qt would silently drop.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from agent3.ui.syntax import (
    C_KEYWORDS,
    JS_KEYWORDS,
    PYTHON_KEYWORDS,
    SHELL_KEYWORDS,
)
from agent3.ui.theme import COLORS, MONO_FONTS

__all__ = ["markdown_to_html", "highlight_code_html", "language_alias"]


# --------------------------------------------------------------- languages
#: Info-string aliases a model is likely to write, mapped onto the keyword
#: family that highlights them.
_LANGUAGE_ALIASES: Dict[str, str] = {
    "python": "python", "py": "python", "python3": "python",
    "js": "js", "javascript": "js", "node": "js", "jsx": "js",
    "ts": "js", "typescript": "js", "tsx": "js",
    "json": "json", "jsonc": "json", "json5": "json",
    "yaml": "yaml", "yml": "yaml",
    "toml": "toml", "ini": "toml", "cfg": "toml",
    "sh": "shell", "bash": "shell", "zsh": "shell", "shell": "shell",
    "console": "shell", "terminal": "shell", "cmd": "shell",
    "bat": "shell", "batch": "shell", "powershell": "shell", "ps1": "shell",
    "c": "c", "h": "c", "cpp": "c", "c++": "c", "cxx": "c", "hpp": "c",
    "cs": "c", "csharp": "c", "java": "c", "kotlin": "c", "kt": "c",
    "go": "c", "golang": "c", "rust": "c", "rs": "c", "swift": "c",
    "php": "c", "ruby": "c", "rb": "c", "scala": "c", "dart": "c",
    "sql": "sql",
    "html": "markup", "xml": "markup", "svg": "markup", "vue": "markup",
    "css": "css", "scss": "css", "less": "css",
    "diff": "diff", "patch": "diff",
    "text": "plain", "txt": "plain", "plain": "plain", "": "plain",
}

_KEYWORDS: Dict[str, Sequence[str]] = {
    "python": PYTHON_KEYWORDS,
    "js": JS_KEYWORDS,
    "c": C_KEYWORDS,
    "shell": SHELL_KEYWORDS,
    "sql": (
        "select from where insert into update delete create table drop alter add "
        "join left right inner outer on group by order having limit offset as and "
        "or not null primary key foreign references index view distinct union all "
        "values set default constraint unique check cascade"
    ).split(),
    "css": (
        "import media keyframes font-face supports charset namespace from to "
        "important and not only screen print all"
    ).split(),
}

_PYTHON_BUILTINS = {
    "print", "len", "range", "dict", "list", "set", "tuple", "str", "int",
    "float", "bool", "open", "super", "isinstance", "enumerate", "zip", "map",
    "filter", "sorted", "sum", "min", "max", "abs", "type", "Exception",
    "ValueError", "TypeError", "KeyError", "OSError", "RuntimeError", "repr",
    "any", "all", "hasattr", "getattr", "setattr", "format",
}

_JSON_LITERALS = {"true", "false", "null"}


def language_alias(info: str) -> str:
    """Normalise a fenced-block info string to a highlighting family."""
    token = (info or "").strip().lower().split()[0] if (info or "").strip() else ""
    token = token.split(":")[0].strip("{}.")
    return _LANGUAGE_ALIASES.get(token, "plain")


# --------------------------------------------------------- code highlighting
def _comment_pattern(language: str) -> str:
    if language in ("python", "shell", "yaml", "toml"):
        return r"#[^\n]*"
    if language == "sql":
        return r"--[^\n]*|/\*(?:.|\n)*?\*/"
    if language in ("js", "c", "css"):
        return r"//[^\n]*|/\*(?:.|\n)*?\*/"
    if language == "markup":
        return r"<!--(?:.|\n)*?-->"
    return r"(?!x)x"  # never matches


def _string_pattern(language: str) -> str:
    triple = r'"""(?:.|\n)*?"""' + r"|'''(?:.|\n)*?'''"
    double = r'"(?:\\.|[^"\\\n])*"'
    single = r"'(?:\\.|[^'\\\n])*'"
    backtick = r"`(?:\\.|[^`\\])*`"
    if language == "python":
        return f"{triple}|{double}|{single}"
    if language in ("js", "shell"):
        return f"{double}|{single}|{backtick}"
    if language == "json":
        return double
    return f"{double}|{single}"


def _span(text: str, color: str, *, bold: bool = False, italic: bool = False) -> str:
    style = f"color:{color};"
    if bold:
        style += "font-weight:600;"
    if italic:
        style += "font-style:italic;"
    return f'<span style="{style}">{text}</span>'


def highlight_code_html(code: str, language: str = "plain") -> str:
    """Return *code* as HTML with syntax colouring, already escaped.

    One ordered regex alternation walks the source once: whatever matches
    first wins, which is what keeps a ``#`` inside a string from turning the
    rest of the line into a comment. Everything the scanner does not claim is
    emitted as plain escaped text, so an unknown language still renders
    correctly - just without colour.
    """
    language = _LANGUAGE_ALIASES.get(language, language)
    if language == "diff":
        return _highlight_diff_html(code)
    if language not in _KEYWORDS and language not in ("json", "yaml", "toml", "markup"):
        return html.escape(code)

    keywords = set(_KEYWORDS.get(language, ()))
    scanner = re.compile(
        "(?P<comment>" + _comment_pattern(language) + ")"
        "|(?P<string>" + _string_pattern(language) + ")"
        r"|(?P<decorator>^[ \t]*@[\w.]+)"
        r"|(?P<number>\b\d+\.?\d*(?:[eE][+-]?\d+)?\b)"
        r"|(?P<name>[A-Za-z_][\w.]*)",
        re.MULTILINE,
    )

    out: List[str] = []
    position = 0
    for match in scanner.finditer(code):
        out.append(html.escape(code[position : match.start()]))
        position = match.end()
        text = html.escape(match.group(0))
        kind = match.lastgroup
        if kind == "comment":
            out.append(_span(text, COLORS.code_comment, italic=True))
        elif kind == "string":
            # A quoted word immediately followed by ':' is a mapping key, and
            # colouring it as a key is what makes JSON and YAML readable.
            after = code[match.end() : match.end() + 2].lstrip()
            colour = COLORS.code_builtin if after.startswith(":") else COLORS.code_string
            out.append(_span(text, colour))
        elif kind == "decorator":
            out.append(_span(text, COLORS.code_function))
        elif kind == "number":
            out.append(_span(text, COLORS.code_number))
        else:
            word = match.group(0)
            follows = code[match.end() : match.end() + 1]
            if word in keywords:
                out.append(_span(text, COLORS.code_keyword, bold=True))
            elif language == "json" and word in _JSON_LITERALS:
                out.append(_span(text, COLORS.code_keyword))
            elif language == "python" and word in _PYTHON_BUILTINS:
                out.append(_span(text, COLORS.code_builtin))
            elif follows == "(":
                out.append(_span(text, COLORS.code_function))
            elif word[:1].isupper() and language in ("python", "js", "c"):
                out.append(_span(text, COLORS.code_type))
            else:
                out.append(text)
    out.append(html.escape(code[position:]))
    return "".join(out)


def _highlight_diff_html(code: str) -> str:
    """Colour a unified diff the same way the diff viewer does."""
    lines: List[str] = []
    for line in code.split("\n"):
        escaped = html.escape(line)
        if line.startswith(("+++", "---")):
            lines.append(_span(escaped, COLORS.text_dim, bold=True))
        elif line.startswith("@@"):
            lines.append(_span(escaped, COLORS.hunk_fg, bold=True))
        elif line.startswith("+"):
            lines.append(_span(escaped, COLORS.added_fg))
        elif line.startswith("-"):
            lines.append(_span(escaped, COLORS.removed_fg))
        else:
            lines.append(escaped)
    return "\n".join(lines)


# ------------------------------------------------------------------ inline
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL)
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_AUTOLINK = re.compile(r"<((?:https?|ftp|mailto):[^>\s]+)>")
_BARE_URL = re.compile(r"(?<![\"'=>\w])(https?://[^\s<>\"')\]]+)")
_BOLD_ITALIC = re.compile(r"\*\*\*(?=\S)(.+?)(?<=\S)\*\*\*", re.DOTALL)
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|__(?=\S)(.+?)(?<=\S)__", re.DOTALL)
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])|(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")
_STRIKE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~", re.DOTALL)

#: Placeholder for an already-rendered fragment. Uses a private-use code
#: point so it can never collide with anything a model writes.
_SENTINEL = "\ue000"
_PLACEHOLDER = re.compile(_SENTINEL + r"(\d+)" + _SENTINEL)


def _inline(text: str) -> str:
    """Render inline markdown in *text*, escaping everything else."""
    protected: List[str] = []

    def protect(fragment: str) -> str:
        protected.append(fragment)
        return f"{_SENTINEL}{len(protected) - 1}{_SENTINEL}"

    # Code spans win over every other inline rule: `**x**` inside backticks
    # has to stay literal.
    def code_span(match: "re.Match[str]") -> str:
        body = match.group(2)
        if body.startswith(" ") and body.endswith(" ") and body.strip():
            body = body[1:-1]
        return protect(
            f'<span style="background-color:{COLORS.bg};color:{COLORS.code_string};'
            f'font-family:{MONO_FONTS};">&nbsp;{html.escape(body)}&nbsp;</span>'
        )

    text = _CODE_SPAN.sub(code_span, text)

    def image(match: "re.Match[str]") -> str:
        alt = html.escape(match.group(1) or "image")
        url = html.escape(match.group(2), quote=True)
        return protect(f'<a href="{url}" style="color:{COLORS.accent};">[{alt}]</a>')

    def link(match: "re.Match[str]") -> str:
        label = match.group(1)
        url = html.escape(match.group(2), quote=True)
        return protect(
            f'<a href="{url}" style="color:{COLORS.accent};">{_inline(label)}</a>'
        )

    def autolink(match: "re.Match[str]") -> str:
        url = html.escape(match.group(1), quote=True)
        return protect(f'<a href="{url}" style="color:{COLORS.accent};">{url}</a>')

    text = _IMAGE.sub(image, text)
    text = _LINK.sub(link, text)
    text = _AUTOLINK.sub(autolink, text)

    text = html.escape(text)

    def bare_url(match: "re.Match[str]") -> str:
        url = match.group(1)
        return f'<a href="{url}" style="color:{COLORS.accent};">{url}</a>'

    text = _BARE_URL.sub(bare_url, text)
    text = _BOLD_ITALIC.sub(r"<b><i>\1</i></b>", text)
    text = _BOLD.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", text)
    text = _ITALIC.sub(lambda m: f"<i>{m.group(1) or m.group(2)}</i>", text)
    text = _STRIKE.sub(r"<s>\1</s>", text)
    # Markdown hard break: two trailing spaces, or a trailing backslash.
    text = re.sub(r"(?:  +|\\)\n", "<br/>", text)
    text = text.replace("\n", " ")

    return _PLACEHOLDER.sub(lambda m: protected[int(m.group(1))], text)


# ------------------------------------------------------------------ blocks
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([^\s`]*)\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_RULE = re.compile(r"^\s{0,3}(?:-\s*-\s*-[-\s]*|\*\s*\*\s*\*[\*\s]*|_\s*_\s*_[_\s]*)$")
_QUOTE = re.compile(r"^\s{0,3}>\s?(.*)$")
_BULLET = re.compile(r"^(\s*)([-*+])\s+(.*)$")
_ORDERED = re.compile(r"^(\s*)(\d{1,9})[.)]\s+(.*)$")
_TASK = re.compile(r"^\[([ xX])\]\s+(.*)$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")

_HEADING_SIZES = {1: 17, 2: 15, 3: 13, 4: 12, 5: 11, 6: 11}


@dataclass
class _ListItem:
    content: List[str]
    checked: Optional[bool] = None


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and not stripped.startswith("|--")


def _split_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def _alignments(divider: str) -> List[str]:
    result = []
    for cell in _split_row(divider):
        left, right = cell.startswith(":"), cell.endswith(":")
        result.append("center" if left and right else "right" if right else "left")
    return result


def _render_table(rows: List[str], divider: str) -> str:
    align = _alignments(divider)
    header = _split_row(rows[0])
    body = [_split_row(row) for row in rows[1:]]

    def cells(values: List[str], tag: str, extra: str) -> str:
        out = []
        for index, value in enumerate(values):
            how = align[index] if index < len(align) else "left"
            out.append(
                f'<{tag} align="{how}" style="{extra}">{_inline(value)}</{tag}>'
            )
        return "".join(out)

    head_style = (
        f"background-color:{COLORS.panel_alt};color:{COLORS.text};"
        f"font-weight:600;padding:6px 10px;"
    )
    cell_style = f"color:{COLORS.text};padding:6px 10px;"
    html_rows = [f"<tr>{cells(header, 'th', head_style)}</tr>"]
    for row in body:
        html_rows.append(f"<tr>{cells(row, 'td', cell_style)}</tr>")
    return (
        f'<table border="1" cellspacing="0" cellpadding="6" width="100%" '
        f'style="border-color:{COLORS.border};margin:6px 0 10px 0;">'
        + "".join(html_rows)
        + "</table>"
    )


def _render_code(code: str, info: str) -> str:
    language = language_alias(info)
    body = highlight_code_html(code, language)
    caption = ""
    label = (info or "").strip()
    if label:
        caption = (
            f'<div style="color:{COLORS.text_faint};font-family:{MONO_FONTS};'
            f'font-size:8pt;margin:6px 0 0 0;">{html.escape(label)}</div>'
        )
    # `white-space: pre-wrap` is what Qt needs to wrap a long line instead of
    # widening the whole document; _AutoTextBrowser additionally clears the
    # non-breakable flag as a belt-and-braces measure.
    return (
        caption
        + f'<pre style="background-color:{COLORS.bg};color:{COLORS.text};'
        f'font-family:{MONO_FONTS};white-space:pre-wrap;margin:2px 0 10px 0;">'
        + body
        + "</pre>"
    )


def _render_blocks(lines: Sequence[str]) -> str:
    """Turn a list of markdown lines into HTML (recursively for nesting)."""
    out: List[str] = []
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]

        if not line.strip():
            index += 1
            continue

        # ------------------------------------------------- fenced code
        fence = _FENCE.match(line)
        if fence:
            marker, info = fence.group(2), fence.group(3)
            indent = len(fence.group(1))
            index += 1
            body: List[str] = []
            closer = marker[0] * 3
            while index < total:
                candidate = lines[index]
                if candidate.strip().startswith(closer) and not candidate.strip().strip(
                    marker[0]
                ):
                    index += 1
                    break
                body.append(candidate[indent:] if candidate[:indent].isspace() else candidate)
                index += 1
            out.append(_render_code("\n".join(body), info))
            continue

        # ---------------------------------------------------- heading
        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            size = _HEADING_SIZES[level]
            colour = COLORS.text if level <= 2 else COLORS.text
            top = 14 if out else 2
            rule = (
                f"border-bottom:1px solid {COLORS.border};padding-bottom:4px;"
                if level <= 2
                else ""
            )
            out.append(
                f'<div style="color:{colour};font-size:{size}pt;font-weight:600;'
                f'margin:{top}px 0 6px 0;{rule}">{_inline(heading.group(2))}</div>'
            )
            index += 1
            continue

        # ------------------------------------------------------- rule
        if _RULE.match(line):
            # An empty div with a border collapses to nothing in Qt, so the
            # rule is drawn as a one-row table instead.
            out.append(
                f'<table border="0" cellspacing="0" cellpadding="0" width="100%" '
                f'style="margin:12px 0 14px 0;"><tr>'
                f'<td style="background-color:{COLORS.border};font-size:1pt;"> </td>'
                f"</tr></table>"
            )
            index += 1
            continue

        # -------------------------------------------------- blockquote
        if _QUOTE.match(line):
            quoted: List[str] = []
            while index < total and (_QUOTE.match(lines[index]) or lines[index].strip()):
                match = _QUOTE.match(lines[index])
                if match is None:
                    if not quoted:
                        break
                    quoted.append(lines[index].strip())  # lazy continuation
                else:
                    quoted.append(match.group(1))
                index += 1
            inner = _render_blocks(quoted)
            out.append(
                f'<table border="0" cellspacing="0" cellpadding="0" '
                f'style="margin:4px 0 10px 0;"><tr>'
                f'<td style="background-color:{COLORS.accent_dim};width:3px;"> </td>'
                f'<td style="padding-left:12px;color:{COLORS.text_dim};">{inner}</td>'
                f"</tr></table>"
            )
            continue

        # -------------------------------------------------------- list
        bullet = _BULLET.match(line)
        ordered = _ORDERED.match(line)
        if bullet or ordered:
            index, rendered = _render_list(lines, index)
            out.append(rendered)
            continue

        # ------------------------------------------------------- table
        if _is_table_row(line) and index + 1 < total and _TABLE_DIVIDER.match(lines[index + 1]):
            divider = lines[index + 1]
            rows = [line]
            index += 2
            while index < total and _is_table_row(lines[index]) and lines[index].strip():
                rows.append(lines[index])
                index += 1
            out.append(_render_table(rows, divider))
            continue

        # --------------------------------------------------- paragraph
        paragraph: List[str] = []
        while index < total and lines[index].strip():
            candidate = lines[index]
            starts_table = (
                _is_table_row(candidate)
                and index + 1 < total
                and _TABLE_DIVIDER.match(lines[index + 1]) is not None
            )
            if (
                _FENCE.match(candidate)
                or _HEADING.match(candidate)
                or _RULE.match(candidate)
                or _QUOTE.match(candidate)
                or (starts_table and paragraph)
            ):
                break
            if paragraph and (_BULLET.match(candidate) or _ORDERED.match(candidate)):
                break
            # Only the leading indent is insignificant: two trailing spaces
            # are markdown's hard line break and must survive to _inline.
            paragraph.append(candidate.lstrip())
            index += 1
        if paragraph:
            out.append(
                f'<div style="margin:0 0 9px 0;line-height:150%;">'
                f"{_inline(chr(10).join(paragraph).rstrip())}</div>"
            )
        else:  # pragma: no cover - defensive, keeps the loop advancing
            index += 1

    return "".join(out)


def _render_list(lines: Sequence[str], start: int) -> Tuple[int, str]:
    """Render one list (and anything nested in it) starting at *start*."""
    first = _BULLET.match(lines[start]) or _ORDERED.match(lines[start])
    assert first is not None
    base_indent = len(first.group(1))
    is_ordered = _ORDERED.match(lines[start]) is not None

    items: List[_ListItem] = []
    index = start
    total = len(lines)

    while index < total:
        line = lines[index]
        if not line.strip():
            # A blank line ends the list unless the next line continues it.
            following = index + 1
            while following < total and not lines[following].strip():
                following += 1
            if following >= total:
                index = following
                break
            nxt = _BULLET.match(lines[following]) or _ORDERED.match(lines[following])
            if nxt is None or len(nxt.group(1)) < base_indent:
                break
            index = following
            continue

        match = _BULLET.match(line) or _ORDERED.match(line)
        if match is None:
            if not items:
                break
            stripped = line.strip()
            if len(line) - len(line.lstrip()) <= base_indent and not stripped.startswith(
                (" ", "\t")
            ):
                # A non-indented paragraph after a list item ends the list.
                break
            items[-1].content.append(stripped)
            index += 1
            continue

        indent = len(match.group(1))
        if indent < base_indent:
            break
        if indent > base_indent:
            if not items:
                break
            consumed, nested = _render_list(lines, index)
            items[-1].content.append(f"{_SENTINEL}HTML{_SENTINEL}{nested}")
            index = consumed
            continue

        body = match.group(3) if match.re is _BULLET else match.group(3)
        task = _TASK.match(body)
        checked: Optional[bool] = None
        if task:
            checked = task.group(1).lower() == "x"
            body = task.group(2)
        items.append(_ListItem(content=[body], checked=checked))
        index += 1

    rendered: List[str] = []
    for position, item in enumerate(items, start=1):
        pieces: List[str] = []
        text_lines: List[str] = []
        for chunk in item.content:
            if chunk.startswith(f"{_SENTINEL}HTML{_SENTINEL}"):
                if text_lines:
                    pieces.append(_inline(" ".join(text_lines)))
                    text_lines = []
                pieces.append(chunk[len(f"{_SENTINEL}HTML{_SENTINEL}") :])
            else:
                text_lines.append(chunk)
        if text_lines:
            pieces.append(_inline(" ".join(text_lines)))
        body_html = "".join(pieces)

        if item.checked is None:
            marker = f"{position}." if is_ordered else "•"
            marker_colour = COLORS.text_dim if is_ordered else COLORS.accent
        else:
            marker = "✓" if item.checked else "○"
            marker_colour = COLORS.success if item.checked else COLORS.text_faint
            if item.checked:
                body_html = f'<span style="color:{COLORS.text_dim};">{body_html}</span>'

        rendered.append(
            f"<tr>"
            f'<td valign="top" style="color:{marker_colour};padding:0 8px 4px 0;">'
            f"{marker}</td>"
            f'<td valign="top" style="padding:0 0 4px 0;line-height:145%;">'
            f"{body_html}</td>"
            f"</tr>"
        )

    table = (
        f'<table border="0" cellspacing="0" cellpadding="0" '
        f'style="margin:2px 0 8px 14px;">' + "".join(rendered) + "</table>"
    )
    return index, table


def markdown_to_html(text: str) -> str:
    """Render *text* as HTML suitable for ``QTextDocument.setHtml``."""
    if not text:
        return ""
    normalised = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    return _render_blocks(normalised.split("\n"))
