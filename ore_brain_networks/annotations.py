"""Integration with ORE's AST-backed SQLite annotation catalogue."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def register_bnbea_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index every ORE class and callable docstring into the standard SQLite annotation database.

    The default database is ``ore_logs/ore_logger.sqlite3`` under the repository root, matching
    ORE's execution logger. A path may be supplied for isolated tests or alternate workspaces.
    """
    from ore_logger import LoggerDatabase, sync_repository

    if repository_root:
        root = Path(repository_root).resolve()
    else:
        candidates = (Path(__file__).resolve().parents[1], Path.cwd().resolve())
        root = next(
            (item for item in candidates if (item / "ore_logger").is_dir() and (item / "pyproject.toml").is_file()),
            Path(__file__).resolve().parents[1],
        )
    target = Path(
        database_path
        or os.environ.get("ORE_LOGGER_DB")
        or root / "ore_logs" / "ore_logger.sqlite3"
    ).expanduser().resolve()
    database = LoggerDatabase(target)
    report = sync_repository(database, root)
    return {
        **report,
        "database_path": str(target),
        "module": "ore_brain_networks",
        "annotation_source": "docstring",
        "storage": "ore_logger.annotations SQLite table",
    }
