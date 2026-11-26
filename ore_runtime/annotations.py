"""Public entry point for indexing ore_runtime definitions in the shared ORE logger."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def register_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every runtime module, class, function and method in ore_logger.annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_log" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    report = sync_repository(database, root, include_paths={"ore_runtime"})
    return {**report, "database_path": str(target), "scope": "ore_runtime", "annotation_table": "annotations"}
