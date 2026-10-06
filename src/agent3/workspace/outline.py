"""Cheap structural outline of a source file.

``read_file`` is the blunt instrument: it spends thousands of context tokens to
tell the model something it often only needs one line of ("which methods does
``SnapshotStore`` have?").  :func:`build_outline` answers that question for a
fraction of the cost.

Python is parsed with the real :mod:`ast`, so the result is exact: classes,
functions, methods, decorators, signatures, docstring summaries, assignments at
module level.  Every other supported language is scanned with conservative
regular expressions - good enough to navigate, and never wrong in a way that
hurts, because the model can always fall back to ``read_file``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: Extension -> language key used by the regex scanners.
LANGUAGE_BY_SUFFIX: Dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".sh": "shell",
    ".bash": "shell",
    ".sql": "sql",
    ".md": "markdown",
}


@dataclass
class Symbol:
    """One named entity found in a source file."""

    kind: str  # class | function | method | property | const | section | ...
    name: str
    line: int
    signature: str = ""
    doc: str = ""
    depth: int = 0
    decorators: List[str] = field(default_factory=list)
    end_line: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "line": self.line,
            "end_line": self.end_line,
            "signature": self.signature,
            "doc": self.doc,
            "depth": self.depth,
            "decorators": list(self.decorators),
        }

    def render(self) -> str:
        """One rendered outline row."""
        indent = "  " * self.depth
        head = f"{self.line:>5}: {indent}{self.kind} {self.name}{self.signature}"
        decorators = [d for d in self.decorators if d not in {"property", "staticmethod", "classmethod"}]
        if decorators:
            head += f"   @{' @'.join(decorators)}"
        if self.doc:
            head += f"\n{' ' * 7}{indent}| {self.doc}"
        return head


@dataclass
class Outline:
    """The structural summary of a single file."""

    path: str
    language: str
    total_lines: int
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.symbols and not self.imports

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "language": self.language,
            "total_lines": self.total_lines,
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": list(self.imports),
            "error": self.error,
        }

    def render(self, *, include_imports: bool = True, max_symbols: int = 400) -> str:
        """Human/model readable outline block."""
        lines = [f"{self.path} - {self.language}, {self.total_lines} lines"]
        if self.error:
            lines.append(f"! {self.error}")
        if include_imports and self.imports:
            shown = self.imports[:40]
            lines.append(f"imports ({len(self.imports)}): {', '.join(shown)}")
            if len(self.imports) > len(shown):
                lines.append(f"  ... {len(self.imports) - len(shown)} more")
        if not self.symbols:
            lines.append("(no top level definitions found - use read_file for the body)")
            return "\n".join(lines)
        lines.append("")
        for symbol in self.symbols[:max_symbols]:
            lines.append(symbol.render())
        if len(self.symbols) > max_symbols:
            lines.append(f"... {len(self.symbols) - max_symbols} more definitions")
        return "\n".join(lines)


# ------------------------------------------------------------------- python
def _first_doc_line(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node)  # type: ignore[arg-type]
    except TypeError:  # pragma: no cover - non docstring-able node
        return ""
    if not doc:
        return ""
    first = doc.strip().splitlines()[0].strip()
    return first[:160]


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def _python_signature(node: ast.AST) -> str:
    """Render ``(a, b=1, *args, **kwargs) -> int`` without the body."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    args = node.args
    parts: List[str] = []

    def render_arg(arg: ast.arg, default: Optional[ast.expr] = None) -> str:
        text = arg.arg
        if arg.annotation is not None:
            text += f": {_unparse(arg.annotation)}"
        if default is not None:
            text += f"={_unparse(default)}"
        return text

    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    defaults: List[Optional[ast.expr]] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    for arg, default in zip(positional, defaults):
        parts.append(render_arg(arg, default))
        if getattr(args, "posonlyargs", None) and arg is args.posonlyargs[-1]:
            parts.append("/")
    if args.vararg is not None:
        parts.append(f"*{render_arg(args.vararg)}")
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(render_arg(arg, default))
    if args.kwarg is not None:
        parts.append(f"**{render_arg(args.kwarg)}")

    rendered = f"({', '.join(parts)})"
    if node.returns is not None:
        rendered += f" -> {_unparse(node.returns)}"
    return rendered


def _unparse(node: ast.AST) -> str:
    try:
        text = ast.unparse(node)
    except Exception:  # pragma: no cover - exotic nodes
        return "..."
    return text if len(text) <= 60 else text[:57] + "..."


def _python_outline(path: str, source: str) -> Outline:
    total = len(source.splitlines())
    outline = Outline(path=path, language="python", total_lines=total)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        outline.error = f"syntax error at line {exc.lineno}: {exc.msg}"
        return outline

    module_doc = _first_doc_line(tree)
    if module_doc:
        outline.symbols.append(Symbol("module", path, 1, doc=module_doc))

    def walk(nodes: Sequence[ast.stmt], depth: int, in_class: bool) -> None:
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and depth == 0:
                if isinstance(node, ast.Import):
                    outline.imports.extend(alias.name for alias in node.names)
                else:
                    module = node.module or "."
                    outline.imports.extend(f"{module}.{a.name}" for a in node.names)
                continue
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(_unparse(b) for b in node.bases)
                outline.symbols.append(
                    Symbol(
                        "class",
                        node.name,
                        node.lineno,
                        signature=f"({bases})" if bases else "",
                        doc=_first_doc_line(node),
                        depth=depth,
                        decorators=[d for d in map(_decorator_name, node.decorator_list) if d],
                        end_line=getattr(node, "end_lineno", None),
                    )
                )
                walk(node.body, depth + 1, True)
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = [d for d in map(_decorator_name, node.decorator_list) if d]
                kind = "method" if in_class else "function"
                if "property" in decorators:
                    kind = "property"
                elif "staticmethod" in decorators:
                    kind = "staticmethod"
                elif "classmethod" in decorators:
                    kind = "classmethod"
                if isinstance(node, ast.AsyncFunctionDef):
                    kind = f"async {kind}"
                outline.symbols.append(
                    Symbol(
                        kind,
                        node.name,
                        node.lineno,
                        signature=_python_signature(node),
                        doc=_first_doc_line(node),
                        depth=depth,
                        decorators=decorators,
                        end_line=getattr(node, "end_lineno", None),
                    )
                )
                # Nested functions are rarely useful in an outline; skip bodies.
                continue
            if depth == 0 and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)
                )
                for target in targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        outline.symbols.append(
                            Symbol("const", target.id, node.lineno, depth=depth)
                        )

    walk(tree.body, 0, False)
    return outline


# -------------------------------------------------------------------- regex
#: ``language -> [(kind, compiled pattern)]``; group ``name`` holds the symbol.
_REGEX_RULES: Dict[str, List[Tuple[str, "re.Pattern[str]"]]] = {
    "javascript": [
        ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)\s*(?P<sig>\([^)]*\))")),
        ("const", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?P<sig>\([^)]*\))\s*=>")),
        ("const", re.compile(r"^\s*export\s+(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=")),
        ("method", re.compile(r"^\s{2,}(?:static\s+)?(?:async\s+)?(?P<name>[A-Za-z_$][\w$]*)\s*(?P<sig>\([^)]*\))\s*\{")),
    ],
    "go": [
        ("func", re.compile(r"^func\s+(?:\((?P<recv>[^)]*)\)\s*)?(?P<name>[A-Za-z_]\w*)\s*(?P<sig>\([^)]*\))")),
        ("type", re.compile(r"^type\s+(?P<name>[A-Za-z_]\w*)\s+(?P<sig>struct|interface|\w+)")),
        ("const", re.compile(r"^(?:const|var)\s+(?P<name>[A-Za-z_]\w*)")),
    ],
    "rust": [
        ("fn", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(?P<name>\w+)\s*(?P<sig>\([^)]*\))")),
        ("struct", re.compile(r"^\s*(?:pub\s+)?struct\s+(?P<name>\w+)")),
        ("enum", re.compile(r"^\s*(?:pub\s+)?enum\s+(?P<name>\w+)")),
        ("trait", re.compile(r"^\s*(?:pub\s+)?trait\s+(?P<name>\w+)")),
        ("impl", re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?P<name>[\w:<>, ]+)")),
    ],
    "java": [
        ("class", re.compile(r"^\s*(?:public|private|protected|final|abstract|static|\s)*(?:class|interface|enum|record)\s+(?P<name>\w+)")),
        ("method", re.compile(r"^\s+(?:public|private|protected|static|final|synchronized|abstract|native|\s)+[\w<>\[\],.?\s]+\s+(?P<name>\w+)\s*(?P<sig>\([^)]*\))\s*(?:throws[\w,.\s]*)?\{")),
    ],
    "csharp": [
        ("class", re.compile(r"^\s*(?:public|private|protected|internal|sealed|static|abstract|partial|\s)*(?:class|interface|struct|enum|record)\s+(?P<name>\w+)")),
        ("method", re.compile(r"^\s+(?:public|private|protected|internal|static|virtual|override|async|sealed|\s)+[\w<>\[\],.?]+\s+(?P<name>\w+)\s*(?P<sig>\([^)]*\))")),
    ],
    "kotlin": [
        ("class", re.compile(r"^\s*(?:open|data|sealed|abstract|internal|private|\s)*class\s+(?P<name>\w+)")),
        ("fun", re.compile(r"^\s*(?:override|private|public|internal|suspend|inline|\s)*fun\s+(?P<name>\w+)\s*(?P<sig>\([^)]*\))")),
    ],
    "swift": [
        ("class", re.compile(r"^\s*(?:public|private|internal|final|open|\s)*(?:class|struct|enum|protocol|extension)\s+(?P<name>\w+)")),
        ("func", re.compile(r"^\s*(?:public|private|internal|static|override|mutating|\s)*func\s+(?P<name>\w+)\s*(?P<sig>\([^)]*\))")),
    ],
    "ruby": [
        ("class", re.compile(r"^\s*class\s+(?P<name>[\w:]+)")),
        ("module", re.compile(r"^\s*module\s+(?P<name>[\w:]+)")),
        ("def", re.compile(r"^\s*def\s+(?P<name>[\w.?!=]+)\s*(?P<sig>\([^)]*\))?")),
    ],
    "php": [
        ("class", re.compile(r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait)\s+(?P<name>\w+)")),
        ("function", re.compile(r"^\s*(?:public|private|protected|static|abstract|final|\s)*function\s+(?P<name>\w+)\s*(?P<sig>\([^)]*\))")),
    ],
    "c": [
        ("struct", re.compile(r"^\s*(?:typedef\s+)?struct\s+(?P<name>\w+)")),
        ("function", re.compile(r"^[A-Za-z_][\w\s\*]*\s\*?(?P<name>\w+)\s*(?P<sig>\([^;]*\))\s*\{")),
    ],
    "shell": [
        ("function", re.compile(r"^\s*(?:function\s+)?(?P<name>[\w\-]+)\s*\(\)\s*\{")),
    ],
    "sql": [
        ("table", re.compile(r"^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>[\w.\"`]+)", re.IGNORECASE)),
        ("view", re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(?P<name>[\w.\"`]+)", re.IGNORECASE)),
        ("index", re.compile(r"^\s*CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>[\w.\"`]+)", re.IGNORECASE)),
    ],
    "markdown": [
        ("section", re.compile(r"^(?P<hashes>#{1,6})\s+(?P<name>.+?)\s*$")),
    ],
}
_REGEX_RULES["typescript"] = _REGEX_RULES["javascript"] + [
    ("interface", re.compile(r"^\s*(?:export\s+)?interface\s+(?P<name>[A-Za-z_$][\w$]*)")),
    ("type", re.compile(r"^\s*(?:export\s+)?type\s+(?P<name>[A-Za-z_$][\w$]*)\s*=")),
    ("enum", re.compile(r"^\s*(?:export\s+)?(?:const\s+)?enum\s+(?P<name>[A-Za-z_$][\w$]*)")),
]
_REGEX_RULES["cpp"] = _REGEX_RULES["c"] + [
    ("class", re.compile(r"^\s*(?:template\s*<[^>]*>\s*)?class\s+(?P<name>\w+)")),
    ("namespace", re.compile(r"^\s*namespace\s+(?P<name>\w+)")),
]

_IMPORT_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "javascript": re.compile(r"""^\s*(?:import\s.*?from\s+['"](?P<mod>[^'"]+)['"]|(?:const|let|var)\s.*?=\s*require\(['"](?P<mod2>[^'"]+)['"]\))"""),
    "go": re.compile(r"""^\s*(?:import\s+)?[\w.]*\s*"(?P<mod>[^"]+)"\s*$"""),
    "rust": re.compile(r"^\s*(?:pub\s+)?use\s+(?P<mod>[\w:{}, *]+);"),
    "java": re.compile(r"^\s*import\s+(?:static\s+)?(?P<mod>[\w.*]+);"),
    "csharp": re.compile(r"^\s*using\s+(?P<mod>[\w.]+);"),
    "php": re.compile(r"^\s*use\s+(?P<mod>[\w\\]+);"),
    "ruby": re.compile(r"""^\s*require(?:_relative)?\s+['"](?P<mod>[^'"]+)['"]"""),
    "c": re.compile(r"^\s*#include\s+[<\"](?P<mod>[^>\"]+)[>\"]"),
}
_IMPORT_PATTERNS["typescript"] = _IMPORT_PATTERNS["javascript"]
_IMPORT_PATTERNS["cpp"] = _IMPORT_PATTERNS["c"]

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _regex_outline(path: str, source: str, language: str) -> Outline:
    total = len(source.splitlines())
    outline = Outline(path=path, language=language, total_lines=total)
    rules = _REGEX_RULES.get(language, [])
    import_re = _IMPORT_PATTERNS.get(language)
    # Strip block comments so commented-out code is not reported as structure.
    cleaned = _BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), source)

    seen: set[Tuple[str, int]] = set()
    for number, line in enumerate(cleaned.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#!", "*")):
            if language != "python" and not stripped.startswith("#!"):
                pass
            if not stripped or stripped.startswith(("//", "*")):
                continue
        if import_re is not None:
            match = import_re.match(line)
            if match:
                module = match.groupdict().get("mod") or match.groupdict().get("mod2")
                if module:
                    outline.imports.append(module.strip())
                    continue
        for kind, pattern in rules:
            match = pattern.match(line)
            if not match:
                continue
            name = (match.groupdict().get("name") or "").strip()
            if not name or (name, number) in seen:
                continue
            seen.add((name, number))
            signature = (match.groupdict().get("sig") or "").strip()
            if len(signature) > 110:
                signature = signature[:107] + "...)"
            depth = 0
            if language == "markdown":
                depth = len(match.group("hashes")) - 1
            else:
                indent = len(line) - len(line.lstrip())
                depth = min(indent // 2, 3) if kind in {"method", "fn", "func", "def"} else 0
            receiver = (match.groupdict().get("recv") or "").strip()
            if receiver:
                name = f"({receiver}) {name}"
            outline.symbols.append(
                Symbol(kind, name, number, signature=signature, depth=depth)
            )
            break
    return outline


# --------------------------------------------------------------------- api
def detect_language(path: str) -> str:
    """Return the outline language key for *path* (``""`` when unsupported)."""
    lowered = path.lower()
    for suffix, language in LANGUAGE_BY_SUFFIX.items():
        if lowered.endswith(suffix):
            return language
    return ""


def build_outline(path: str, source: str) -> Outline:
    """Return the structural :class:`Outline` of *source*."""
    language = detect_language(path)
    if language == "python":
        return _python_outline(path, source)
    if language in _REGEX_RULES:
        return _regex_outline(path, source, language)
    return Outline(
        path=path,
        language=language or "text",
        total_lines=len(source.splitlines()),
        error="no structural parser for this file type - use read_file instead",
    )


def outline_text(path: str, source: str, *, include_imports: bool = True) -> str:
    """Convenience wrapper returning the rendered outline."""
    return build_outline(path, source).render(include_imports=include_imports)
