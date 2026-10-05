"""Discover prepared ASELMDB shards consistently for training and data tools."""

from pathlib import Path
from typing import List


def discover_aselmdb_files(path: Path) -> List[Path]:
    """Return sorted recursive shards, or the explicitly supplied single file."""
    path = Path(path)
    if path.is_dir():
        db_files = sorted(path.rglob("*.aselmdb"))
    elif path.is_file():
        db_files = [path]
    else:
        raise FileNotFoundError(f"LMDB path not found: {path}")

    if not db_files:
        raise FileNotFoundError(f"No .aselmdb files found under: {path}")
    return db_files
