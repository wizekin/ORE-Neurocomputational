"""Point 17 SQLite annotation-catalogue convenience entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index Point 17 classes and callables in the ORE SQLite symbol catalogue."""
    from .pipeline import register_annotations as register_package_annotations

    return register_package_annotations(database_path, repository_root)
