"""Explicit helper for indexing Point 07 classes and functions in the ORE SQLite catalogue."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index package docstrings and signatures for every class and callable in the Point 07 package."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    database = LoggerDatabase(database_path)
    return sync_repository(database, root, include_paths={"ore_prototype_exemplar"})
