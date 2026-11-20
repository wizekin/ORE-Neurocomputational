"""Register all memory-integration definitions in ORE's shared logger SQLite catalogue."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def register_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every package class, function, and method in ore_logger.annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    configured = database_path or os.environ.get("ORE_LOGGER_DB")
    target = Path(configured).expanduser().resolve() if configured else root / "ore_logs" / "ore_logger.sqlite3"
    catalogue = sync_repository(
        LoggerDatabase(target),
        root,
        include_paths={"ore_memory_integration"},
    )
    return {
        **catalogue,
        "database_path": str(target),
        "scope": "ore_memory_integration",
        "annotation_table": "annotations",
    }
