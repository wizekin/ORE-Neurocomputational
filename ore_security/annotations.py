"""Public entry point for registering Point 20 definitions in the shared ORE logger."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def register_annotations(database_path: str | Path | None = None,
                         repository_root: str | Path | None = None) -> dict[str, Any]:
    """Index every ore_security class, function, and method in the logger SQLite catalog."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    target = (Path(database_path).expanduser().resolve() if database_path
              else root / "ore_logs" / "ore_logger.sqlite3")
    catalogue = sync_repository(LoggerDatabase(target), root, include_paths={"ore_security"})
    return {**catalogue, "database_path": str(target),
            "scope": "ore_security", "annotation_table": "annotations"}


def record_pipeline_run_to_sqlite(result: Any, database_path: str | Path,
                                  repository_root: str | Path | None = None) -> dict[str, Any]:
    """Persist one protected-operation result as an ore_logger run with linked events."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(database, root, include_paths={"ore_security"})
    run_id = database.create_run(
        script_path="ore_security/pipeline",
        module_name="ore_security.pipeline",
        entry_kind="point20_trust_layer",
        working_directory=str(root),
    )
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    database.record_event(
        run_id=run_id,
        sequence_no=1,
        event_kind="protected_operation",
        source_path="ore_security/pipeline.py",
        metadata=payload,
    )
    database.finish_run(run_id, "completed")
    return {"run_id": run_id, "annotation_catalogue": catalogue,
            "database_summary": database.summary(), "events_written": 1}
