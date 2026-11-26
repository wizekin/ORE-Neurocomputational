"""Register ore_science symbol annotations and persist result traces in the shared logger."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ore_logger import LoggerDatabase, sync_repository
from ore_logger.recorder import SQLiteRunRecorder

from .contracts import ScienceResult

__all__ = ["index_science_annotations", "record_science_result"]


def index_science_annotations(
    repository_root: str | Path,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index every package module, class, and callable into the shared ORE annotations table."""
    root = Path(repository_root).expanduser().resolve()
    target = Path(database_path) if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    report = sync_repository(database, root)
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT symbol_kind, COUNT(*) AS count
            FROM annotations
            WHERE source_path LIKE 'ore_science/%'
            GROUP BY symbol_kind
            """
        ).fetchall()
    counts = {str(row["symbol_kind"]): int(row["count"]) for row in rows}
    return {
        **report,
        "database_path": str(database.path),
        "science_annotations": counts,
        "science_symbol_count": int(sum(counts.values())),
    }


def record_science_result(
    result: ScienceResult,
    repository_root: str | Path,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Store each scientific operation and stage as a run/event using the shared ORE SQLite schema."""
    root = Path(repository_root).expanduser().resolve()
    target = Path(database_path) if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    started = datetime.now(timezone.utc)
    recorder = SQLiteRunRecorder(
        database_path=target,
        repository_root=root,
        run_id=uuid.uuid4().hex,
        component="ore_science",
        callable_name="ScientificReasoner.process",
        started_at_utc=started.isoformat(timespec="milliseconds"),
        entry_kind="scientific_reasoning",
    )
    for row in result.operation_trace:
        recorder.log_operation({
            "step": row["operation_id"],
            "operation": "{0} {1}".format(row["operation_id"], row["operation"]),
            "phase": row["phase"],
            "status": row["functional_status"],
            "explanation": row["explanation"],
            "biological_status": "not_applicable",
        })
    recorder.log_component_traces(result.stage_trace)
    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
    metadata = recorder.finish(
        "completed" if result.diagnostics.get("valid") else "completed_with_warnings",
        elapsed_ms,
        output_type="ScienceResult",
    )
    metadata["stage_count"] = len(result.stage_trace)
    metadata["operation_count"] = len(result.operation_trace)
    metadata["diagnostics_valid"] = bool(result.diagnostics.get("valid"))
    metadata["outcome"] = result.outcome.value
    metadata["database_summary"] = recorder.database.summary()
    return metadata
