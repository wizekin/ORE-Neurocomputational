"""Public entry point for registering Point 07 definitions in the shared ORE logger."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def register_annotations(database_path: str | Path | None = None,
                         repository_root: str | Path | None = None) -> dict[str, Any]:
    """Index every Point 07 class, function, and method in the logger SQLite annotations table."""
    from .pipeline import register_annotations as register_package_annotations

    return register_package_annotations(database_path, repository_root)
