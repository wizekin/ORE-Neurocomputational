"""Point 03 entry point for AST annotation registration in the shared ORE SQLite catalogue."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def register_annotations(
    database_path: str | Path, repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every Point 03 class and callable using the repository-wide AST catalogue."""
    from .pipeline import register_annotations as register_package_annotations
    return register_package_annotations(database_path, repository_root)
