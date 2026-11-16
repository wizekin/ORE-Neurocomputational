"""Register sensory-perception classes and callable docstrings in ORE's shared SQLite catalogue."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def register_sensory_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Synchronize every repository Python definition into ``ore_logger.annotations``.

    The default target is ``ore_logs/ore_logger.sqlite3``. The repository-wide AST catalog stores
    class/function/method docstrings, qualified names, source locations, signatures and source hashes.
    A custom root or path makes isolated validation possible without modifying the live logger.
    """
    from ore_logger import LoggerDatabase, sync_repository

    if repository_root is not None:
        root = Path(repository_root).expanduser().resolve()
    else:
        candidates = (Path(__file__).resolve().parents[1], Path.cwd().resolve())
        root = next(
            (candidate for candidate in candidates
             if (candidate / "ore_logger").is_dir() and (candidate / "pyproject.toml").is_file()),
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
        "module": "ore_sensory_perception",
        "annotation_source": "python_docstring",
        "storage": "ore_logger.annotations SQLite table",
    }
