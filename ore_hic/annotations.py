"""Explicit setup helper for persisting HIC source annotations in ORE SQLite."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every class and callable docstring in the HIC package into SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    database = LoggerDatabase(database_path)
    return sync_repository(database, root, include_paths={"ore_hic"})
