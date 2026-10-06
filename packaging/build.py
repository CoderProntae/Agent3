"""Build the application executable locally (CI runs the very same steps).

Usage::

    python packaging/build.py              # build Agent3.exe
    python packaging/build.py --clean      # wipe build/ and dist/ first
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging"
DIST = ROOT / "dist"
BUILD = ROOT / "build"

TARGETS = {
    "agent": ("Agent3", PACKAGING / "Agent3.spec"),
}


def run(command: list[str]) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit(f"command failed with exit code {completed.returncode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze Agent3 with PyInstaller")
    parser.add_argument("--only", choices=sorted(TARGETS), help="build a single target")
    parser.add_argument("--clean", action="store_true", help="remove build/ and dist/ first")
    args = parser.parse_args(argv)

    if args.clean:
        for folder in (DIST, BUILD):
            if folder.exists():
                shutil.rmtree(folder)
                print(f"removed {folder}")

    if sys.platform.startswith("win"):
        run([sys.executable, str(PACKAGING / "make_version_file.py")])

    selected = [args.only] if args.only else list(TARGETS)
    for key in selected:
        name, spec = TARGETS[key]
        print(f"\n=== building {name} ({platform.system()} {platform.machine()}) ===")
        run([sys.executable, "-m", "PyInstaller", str(spec), "--noconfirm", "--clean"])

    print("\nArtifacts:")
    if DIST.exists():
        for item in sorted(DIST.iterdir()):
            size = item.stat().st_size / (1024 * 1024) if item.is_file() else 0
            print(f"  {item.name}  ({size:.1f} MiB)" if size else f"  {item.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
