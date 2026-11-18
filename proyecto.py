"""Entrada única: python proyecto.py --help."""

from pathlib import Path
import os
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from cli import main

    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
