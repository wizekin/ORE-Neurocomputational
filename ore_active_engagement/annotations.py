"""Point 01 entry point for registering package definitions in the ORE SQLite logger."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every Point 01 class and callable using the shared AST annotation catalogue."""
    from .pipeline import register_annotations as register_package_annotations

    return register_package_annotations(database_path, repository_root)
