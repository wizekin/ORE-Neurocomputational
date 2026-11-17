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



def register_point02_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index Point 2, the lexical engine and lexical inventory contracts in the shared SQLite catalogue.

    The repository logger stores class, function and method annotations in the existing annotations
    table. Scoped synchronization refreshes only the language-comprehension, lexical-access and
    lexical-context packages; it does not create a separate Point 2 database.
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
        include_paths={
            "ore_language_comprehension",
            "ore_lexical_access",
            "ore_lexical_context",
        },
    )
    return {
        **report,
        "database_path": str(target),
        "modules": [
            "ore_language_comprehension",
            "ore_lexical_access",
            "ore_lexical_context",
        ],
        "annotation_source": "python_docstring_and_ast_signature",
        "storage": "ore_logger.annotations SQLite table",
    }


def register_point03_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index Point 3 and the shared syntax engine classes/functions in ORE's SQLite annotation table.

    Scoped synchronization refreshes only ore_language_comprehension and ore_syntax. Point 1 and
    Point 2 annotations remain in the same repository-wide annotations table; no new database is
    created for the syntactic-parsing stage.
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
        include_paths={"ore_language_comprehension", "ore_syntax"},
    )
    return {
        **report,
        "database_path": str(target),
        "modules": ["ore_language_comprehension", "ore_syntax"],
        "annotation_source": "python_docstring_and_ast_signature",
        "storage": "ore_logger.annotations SQLite table",
    }


def register_point04_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index Point 4, semantic composition, lexical semantics, and syntax definitions in shared SQLite.

    Scoped synchronization refreshes the language-comprehension, semantic-composition,
    semantic-concept, and syntax packages within the existing repository-wide annotations table.
    It does not create a separate Point 4 database.
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
    modules = [
        "ore_language_comprehension",
        "ore_semantic_composition",
        "ore_semantic",
        "ore_syntax",
    ]
    report = sync_repository(
        database,
        root,
        include_paths=set(modules),
    )
    return {
        **report,
        "database_path": str(target),
        "modules": modules,
        "annotation_source": "python_docstring_and_ast_signature",
        "storage": "ore_logger.annotations SQLite table",
    }


def register_point05_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index Point 5 and its discourse dependencies in the shared ORE SQLite annotation table.

    The synchronization refreshes the language-comprehension and discourse packages in the
    existing repository-wide annotations database; it does not create a separate database.
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
    modules = ["ore_language_comprehension", "ore_discourse"]
    report = sync_repository(database, root, include_paths=set(modules))
    return {
        **report,
        "database_path": str(target),
        "modules": modules,
        "annotation_source": "python_docstring_and_ast_signature",
        "storage": "ore_logger.annotations SQLite table",
    }


def register_point06_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index Point 6 and its working-memory dependencies in the shared ORE SQLite catalogue.

    The scoped synchronization refreshes the language-comprehension and working-memory
    packages in the existing annotations database; it does not create a separate database.
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
    modules = ["ore_language_comprehension", "ore_working_memory"]
    report = sync_repository(database, root, include_paths=set(modules))
    return {
        **report,
        "database_path": str(target),
        "modules": modules,
        "annotation_source": "python_docstring_and_ast_signature",
        "storage": "ore_logger.annotations SQLite table",
    }


def register_point07_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index Point 7 memory/decision definitions in the existing ORE SQLite annotation table.

    Point 7 has its own optional memory database at runtime, but its source annotations
    remain in the shared repository-wide ore_logger.annotations catalogue.
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
    modules = ["ore_language_comprehension"]
    report = sync_repository(database, root, include_paths=set(modules))
    return {
        **report,
        "database_path": str(target),
        "modules": modules,
        "annotation_source": "python_docstring_and_ast_signature",
        "storage": "ore_logger.annotations SQLite table",
    }
