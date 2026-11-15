"""Registration of visual-orthography docstrings in ORE's shared SQLite annotation catalogue."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def register_visual_orthography_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index every repository class and callable docstring in the existing ORE SQLite database.

    The default target is ore_logs/ore_logger.sqlite3. This function uses the repository's
    AST-backed ore_logger catalogue and does not create a separate annotation database.
    """
    from ore_logger import LoggerDatabase, sync_repository

    if repository_root is not None:
        root = Path(repository_root).expanduser().resolve()
    else:
        candidates = (Path(__file__).resolve().parents[1], Path.cwd().resolve())
        root = next(
            (
                item for item in candidates
                if (item / "ore_logger").is_dir() and (item / "pyproject.toml").is_file()
            ),
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
        "module": "ore_visual_orthography",
        "annotation_source": "docstring",
        "storage": "ore_logger.annotations SQLite table",
    }
