"""Register every new pathway symbol in the shared ORE SQLite annotation catalogue."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def register_visual_language_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index every class, function, nested function and method in this package into SQLite.

    The existing ore_logger.annotations table stores qualified names, source locations,
    signatures, docstrings and source hashes. The scan is deliberately scoped to the new
    package; existing sensory, language and working-memory packages retain their own
    annotation registrations and are not redundantly rewritten.
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
    report = sync_repository(database, root, include_paths={"ore_text_localization"})
    return {
        **report,
        "database_path": str(target),
        "modules": ["ore_text_localization"],
        "annotation_source": "python_docstring_and_ast_signature",
        "storage": "ore_logger.annotations SQLite table",
        "definition_coverage": "classes, functions, nested functions and methods",
    }


__all__ = ["register_visual_language_annotations"]
