"""Thin CLI wrapper for chunk building."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingest import run

if __name__ == "__main__":
    sys.exit(run())
