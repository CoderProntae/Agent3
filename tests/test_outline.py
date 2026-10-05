"""Tests for the structural outline used by the ``view_outline`` tool."""

from __future__ import annotations

from agent3.workspace.outline import (
    LANGUAGE_BY_SUFFIX,
    build_outline,
    detect_language,
    outline_text,
)

PYTHON_SAMPLE = '''"""Module docstring that explains everything."""

from __future__ import annotations

import os
from typing import Optional

VERSION = "1.2.3"
_PRIVATE = 1


class Repository:
    """Stores widgets."""

    table = "widgets"

    def __init__(self, dsn: str, *, echo: bool = False) -> None:
        """Open the connection."""
        self.dsn = dsn

    @property
    def is_open(self) -> bool:
        return True

    @staticmethod
    def helper(value: int = 3) -> str:
        return str(value)

    @classmethod
    def from_env(cls) -> "Repository":
        return cls("")

    async def fetch_all(self, limit: Optional[int] = None) -> list:
        """Return every widget."""
        return []


def top_level(a, b=2, *args, key: str = "x", **kwargs) -> dict:
    """Do the thing."""

    def nested() -> None:
        pass

    return {}


async def fetch(url: str) -> bytes:
    return b""
'''


class TestPythonOutline:
    def test_language_detection(self):
        assert detect_language("a/b/c.py") == "python"
        assert detect_language("app.TSX".lower()) == "typescript"
        assert detect_language("notes.txt") == ""

    def test_module_docstring_and_imports(self):
        outline = build_outline("repo.py", PYTHON_SAMPLE)
        assert outline.language == "python"
        assert outline.total_lines == len(PYTHON_SAMPLE.splitlines())
        assert "Module docstring that explains everything." in outline.symbols[0].doc
        assert "os" in outline.imports
        assert "typing.Optional" in outline.imports

    def test_classes_and_methods(self):
        outline = build_outline("repo.py", PYTHON_SAMPLE)
        names = {s.name: s for s in outline.symbols}
        assert "Repository" in names
        assert names["Repository"].kind == "class"
        assert names["Repository"].doc == "Stores widgets."
        assert names["__init__"].kind == "method"
        assert names["__init__"].depth == 1
        assert names["is_open"].kind == "property"
        assert names["helper"].kind == "staticmethod"
        assert names["from_env"].kind == "classmethod"
        assert names["fetch_all"].kind == "async method"
        assert names["top_level"].kind == "function"
        assert names["fetch"].kind == "async function"

    def test_signatures_are_rendered(self):
        outline = build_outline("repo.py", PYTHON_SAMPLE)
        signatures = {s.name: s.signature for s in outline.symbols}
        assert signatures["__init__"] == "(self, dsn: str, *, echo: bool=False) -> None"
        assert signatures["top_level"] == "(a, b=2, *args, key: str='x', **kwargs) -> dict"
        assert signatures["fetch_all"] == "(self, limit: Optional[int]=None) -> list"

    def test_public_constants_only(self):
        outline = build_outline("repo.py", PYTHON_SAMPLE)
        consts = {s.name for s in outline.symbols if s.kind == "const"}
        assert "VERSION" in consts
        assert "_PRIVATE" not in consts

    def test_nested_functions_are_skipped(self):
        outline = build_outline("repo.py", PYTHON_SAMPLE)
        assert "nested" not in {s.name for s in outline.symbols}

    def test_line_numbers_point_at_the_definition(self):
        outline = build_outline("repo.py", PYTHON_SAMPLE)
        lines = PYTHON_SAMPLE.splitlines()
        for symbol in outline.symbols:
            if symbol.kind in {"class", "function", "method", "async function", "async method"}:
                assert symbol.name in lines[symbol.line - 1]

    def test_syntax_error_is_reported_not_raised(self):
        outline = build_outline("broken.py", "def oops(:\n    pass\n")
        assert "syntax error" in outline.error
        assert outline.symbols == []

    def test_render_is_compact(self):
        rendered = outline_text("repo.py", PYTHON_SAMPLE)
        assert "class Repository" in rendered
        assert "Stores widgets." in rendered
        # The whole point: far smaller than the file itself.
        assert len(rendered) < len(PYTHON_SAMPLE)


class TestRegexOutlines:
    def test_javascript(self):
        source = (
            "import React from 'react';\n"
            "const helper = (a, b) => a + b;\n"
            "export function render(props) {\n"
            "  return null;\n"
            "}\n"
            "export default class Widget {\n"
            "  constructor(props) {}\n"
            "  async load(id) {}\n"
            "}\n"
        )
        outline = build_outline("app.js", source)
        kinds = {(s.kind, s.name) for s in outline.symbols}
        assert ("function", "render") in kinds
        assert ("class", "Widget") in kinds
        assert ("const", "helper") in kinds
        assert "react" in outline.imports

    def test_typescript_interfaces_and_types(self):
        source = (
            "export interface User { id: number }\n"
            "export type Id = string;\n"
            "export enum Role { Admin }\n"
        )
        outline = build_outline("model.ts", source)
        kinds = {(s.kind, s.name) for s in outline.symbols}
        assert ("interface", "User") in kinds
        assert ("type", "Id") in kinds
        assert ("enum", "Role") in kinds

    def test_go_with_receiver(self):
        source = (
            'package main\n\nimport "fmt"\n\n'
            "type Server struct {\n}\n\n"
            "func (s *Server) Start(addr string) error {\n    return nil\n}\n\n"
            "func main() {\n}\n"
        )
        outline = build_outline("main.go", source)
        names = [s.name for s in outline.symbols]
        assert "(s *Server) Start" in names
        assert "main" in names
        assert "Server" in names

    def test_rust_and_java(self):
        rust = build_outline("lib.rs", "pub struct A;\npub fn run(x: u8) -> u8 { x }\n")
        assert {"A", "run"} <= {s.name for s in rust.symbols}
        java = build_outline(
            "A.java",
            "package x;\nimport java.util.List;\npublic class A {\n"
            "    public void run(int a) throws Exception {\n    }\n}\n",
        )
        assert "A" in {s.name for s in java.symbols}
        assert "java.util.List" in java.imports

    def test_markdown_sections_are_nested(self):
        outline = build_outline("README.md", "# Title\n\ntext\n\n## Install\n\n### Linux\n")
        rows = [(s.name, s.depth) for s in outline.symbols]
        assert ("Title", 0) in rows
        assert ("Install", 1) in rows
        assert ("Linux", 2) in rows

    def test_commented_out_code_is_ignored(self):
        source = "// function ghost() {}\n/* class Ghost {} */\nfunction real() {}\n"
        outline = build_outline("a.js", source)
        names = {s.name for s in outline.symbols}
        assert names == {"real"}

    def test_unsupported_type_explains_itself(self):
        outline = build_outline("data.bin", "\x00\x01")
        assert "read_file" in outline.error
        assert outline.symbols == []

    def test_every_mapped_suffix_parses_without_raising(self):
        for suffix in LANGUAGE_BY_SUFFIX:
            build_outline(f"sample{suffix}", "x = 1\nclass A {}\nfunc b() {}\n")
