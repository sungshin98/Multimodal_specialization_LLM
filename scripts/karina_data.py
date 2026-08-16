"""Convenience wrapper: python scripts/karina_data.py ..."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from datasets.karina.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
