"""Register VWFA Python definitions in ORE's repository-wide SQLite annotation catalogue."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def register_vwfa_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Synchronize class, function, method, and module docstrings into ``ore_logger.annotations``.

    The default database matches the shared ORE execution logger at ``ore_logs/ore_logger.sqlite3``.
    ``database_path`` supports isolated tests and alternate checkouts without changing global settings.
    """
    from ore_logger import LoggerDatabase, sync_repository

    if repository_root is not None:
        root = Path(repository_root).expanduser().resolve()
    else:
        candidates = (Path(__file__).resolve().parents[1], Path.cwd().resolve())
        root = next(
            (item for item in candidates if (item / "ore_logger").is_dir()
             and (item / "pyproject.toml").is_file()),
            Path(__file__).resolve().parents[1],
        )
    target = Path(
        database_path or os.environ.get("ORE_LOGGER_DB") or root / "ore_logs" / "ore_logger.sqlite3"
    ).expanduser().resolve()
    database = LoggerDatabase(target)
    report = sync_repository(database, root)
    return {
        **report,
        "database_path": str(target),
        "module": "ore_vwfa",
        "annotation_source": "python_docstring",
        "storage": "ore_logger.annotations SQLite table",
    }
