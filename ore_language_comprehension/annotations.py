"""Register source-ordered language-comprehension definitions in ORE's shared SQLite annotation catalogue."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def register_point01_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index every module, class, function and method docstring for Point 1 and its sensory dependency.

    The existing repository-wide annotation table is reused. The include-path scope ensures this
    package refresh does not mark annotations from unrelated ORE packages as stale or delete them.
    """
    from ore_logger import LoggerDatabase, sync_repository

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    target = Path(
        database_path
        or os.environ.get("ORE_LOGGER_DB")
        or root / "ore_logs" / "ore_logger.sqlite3"
    ).expanduser().resolve()
    database = LoggerDatabase(target)
    report = sync_repository(
        database,
        root,
        include_paths={"ore_language_comprehension", "ore_sensory_perception"},
    )
    return {
        **report,
        "database_path": str(target),
        "modules": ["ore_language_comprehension", "ore_sensory_perception"],
        "annotation_source": "python_docstring_and_ast_signature",
        "storage": "ore_logger.annotations SQLite table",
    }
