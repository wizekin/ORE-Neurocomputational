"""Public entry point for indexing Point 06 definitions in the shared ORE SQLite logger."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def register_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index all Point 06 classes, functions, and methods in the logger annotations table."""
    from .pipeline import register_annotations as register_package_annotations
    return register_package_annotations(database_path, repository_root)