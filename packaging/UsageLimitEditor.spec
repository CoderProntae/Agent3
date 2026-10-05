# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone administrator utility.

Build with::

    pyinstaller packaging/UsageLimitEditor.spec --noconfirm --clean
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(SPECPATH).resolve()))

from common import (  # noqa: E402  (SPECPATH is injected by PyInstaller)
    EXCLUDED_MODULES,
    HIDDEN_IMPORTS,
    console_mode,
    icon_path,
    resource_datas,
    source_root,
    version_file,
)

SRC = source_root(SPECPATH)
ENTRY = SRC / "usage_limit_editor" / "__main__.py"

block_cipher = None

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC)],
    binaries=[],
    datas=resource_datas(SPECPATH),
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDED_MODULES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="UsageLimitEditor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=console_mode(),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path(SPECPATH),
    version=version_file(SPECPATH),
)
