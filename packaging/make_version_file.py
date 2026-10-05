"""Generate the Windows VERSIONINFO resource consumed by the ``.spec`` files.

Run from the repository root::

    python packaging/make_version_file.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "src" / "agent3" / "__init__.py"
TARGET = Path(__file__).resolve().parent / "version_info.txt"

TEMPLATE = """# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({major}, {minor}, {patch}, 0),
    prodvers=({major}, {minor}, {patch}, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [StringStruct('CompanyName', 'Agent3 Contributors'),
         StringStruct('FileDescription', 'Agent3 - local autonomous AI coding workspace'),
         StringStruct('FileVersion', '{version}'),
         StringStruct('InternalName', 'Agent3'),
         StringStruct('LegalCopyright', 'MIT License'),
         StringStruct('OriginalFilename', 'Agent3.exe'),
         StringStruct('ProductName', 'Agent3'),
         StringStruct('ProductVersion', '{version}')])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def read_version() -> str:
    """Extract ``__version__`` from the package without importing it."""
    match = re.search(r'__version__\s*=\s*"([^"]+)"', INIT.read_text(encoding="utf-8"))
    if not match:  # pragma: no cover - the constant is always present
        raise SystemExit("could not determine __version__ from agent3/__init__.py")
    return match.group(1)


def main() -> int:
    version = read_version()
    parts = (version.split(".") + ["0", "0", "0"])[:3]
    major, minor, patch = (int(re.sub(r"\D", "", p) or 0) for p in parts)
    TARGET.write_text(
        TEMPLATE.format(major=major, minor=minor, patch=patch, version=version), encoding="utf-8"
    )
    print(f"wrote {TARGET} for version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
