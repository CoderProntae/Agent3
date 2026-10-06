"""Guards for the PyInstaller build definition.

A frozen build cannot be exercised from the unit test suite, so these tests
protect the two mistakes that are only visible *after* shipping a binary:

* excluding a module the application actually imports at runtime
  (this is how ``ModuleNotFoundError: No module named 'shiboken6.Shiboken'``
  escaped into a release build), and
* pointing the spec files at resources that do not exist.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))

import common  # noqa: E402  (path juggling above is intentional)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SPEC_ARG = str(PROJECT_ROOT / "packaging")


def _source_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def _imported_modules() -> set[str]:
    """Every module name imported anywhere under ``src/``."""
    found: set[str] = set()
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module)
                for alias in node.names:
                    found.add(f"{node.module}.{alias.name}")
    return found


def test_source_tree_is_not_empty() -> None:
    assert len(_source_files()) > 20


# --------------------------------------------------------------- exclusions
def test_shiboken_is_never_excluded() -> None:
    """``shiboken6`` is the PySide6 binding layer - excluding it breaks the exe."""
    for name in common.EXCLUDED_MODULES:
        assert not name.startswith("shiboken"), (
            f"{name!r} must not be excluded: PySide6 imports shiboken6.Shiboken "
            "during 'import PySide6' and the frozen binary will not start."
        )


def test_excluded_modules_are_not_imported_by_the_application() -> None:
    """No module the code imports may appear in the PyInstaller exclude list."""
    imported = _imported_modules()
    offenders = []
    for excluded in common.EXCLUDED_MODULES:
        for name in imported:
            if name == excluded or name.startswith(excluded + "."):
                offenders.append((excluded, name))
    assert not offenders, f"excluded modules are imported at runtime: {offenders}"


def test_pyside_modules_used_in_code_are_not_excluded() -> None:
    """Every ``PySide6.Qt*`` module referenced in the source must survive."""
    used = {name for name in _imported_modules() if name.startswith("PySide6.")}
    excluded = set(common.EXCLUDED_MODULES)
    assert used, "expected the UI code to import PySide6 modules"
    assert not (used & excluded)


def test_runtime_dependencies_are_not_excluded() -> None:
    for required in ("requests", "cryptography", "sqlite3", "shiboken6", "PySide6"):
        assert required not in common.EXCLUDED_MODULES


# ------------------------------------------------------------ hidden imports
@pytest.mark.parametrize("module", common.HIDDEN_IMPORTS)
def test_hidden_imports_are_importable(module: str) -> None:
    """A typo in ``HIDDEN_IMPORTS`` silently does nothing - catch it here."""
    try:
        importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        if module.startswith(("PySide6", "shiboken6")):
            pytest.skip(f"Qt runtime libraries unavailable: {exc}")
        raise


def test_qt_modules_used_by_the_ui_are_declared_hidden() -> None:
    used = {name.split(".")[0] + "." + name.split(".")[1] for name in _imported_modules() if name.startswith("PySide6.")}
    missing = used - set(common.HIDDEN_IMPORTS)
    assert not missing, f"add these to HIDDEN_IMPORTS: {sorted(missing)}"


# ----------------------------------------------------------------- resources
def test_resource_datas_points_at_an_existing_folder() -> None:
    datas = common.resource_datas(SPEC_ARG)
    assert datas, "the resources folder must be bundled"
    source, target = datas[0]
    assert Path(source).is_dir()
    assert target == "resources"
    assert (Path(source) / "agent3.ico").exists()
    assert (Path(source) / "agent3.png").exists()


def test_icon_path_resolves() -> None:
    icon = common.icon_path(SPEC_ARG)
    assert icon is not None
    assert Path(icon).exists()


def test_project_root_accepts_file_and_directory() -> None:
    from_dir = common.project_root(SPEC_ARG)
    from_file = common.project_root(str(PROJECT_ROOT / "packaging" / "Agent3.spec"))
    assert from_dir == from_file == PROJECT_ROOT


def test_spec_files_exist_and_are_valid_python() -> None:
    for name in ("Agent3.spec",):
        spec = PROJECT_ROOT / "packaging" / name
        assert spec.exists(), f"missing {name}"
        ast.parse(spec.read_text(encoding="utf-8"), filename=str(spec))


def test_specs_reference_the_shared_helpers() -> None:
    for name in ("Agent3.spec",):
        text = (PROJECT_ROOT / "packaging" / name).read_text(encoding="utf-8")
        assert "EXCLUDED_MODULES" in text
        assert "HIDDEN_IMPORTS" in text
