"""Register CSC symbol annotations and persist CSC stage traces in the shared ORE SQLite logger."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid
from typing import Any

from ore_logger import LoggerDatabase, sync_repository
from ore_logger.recorder import SQLiteRunRecorder

from .contracts import CSCResult


def index_csc_annotations(
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
            WHERE source_path LIKE 'ore_csc/%'
            GROUP BY symbol_kind
            """
        ).fetchall()
    counts = {str(row["symbol_kind"]): int(row["count"]) for row in rows}
    return {
        **report,
        "database_path": str(database.path),
        "csc_annotations": counts,
        "csc_symbol_count": int(sum(counts.values())),
    }


def record_csc_result(
    result: CSCResult,
    repository_root: str | Path,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Store each CSC operation and stage as a run/event using the shared ORE SQLite schema."""
    root = Path(repository_root).expanduser().resolve()
    target = Path(database_path) if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    started = datetime.now(timezone.utc)
    recorder = SQLiteRunRecorder(
        database_path=target,
        repository_root=root,
        run_id=uuid.uuid4().hex,
        component="ore_csc",
        callable_name="CSCProcessor.process",
        started_at_utc=started.isoformat(timespec="milliseconds"),
        entry_kind="controlled_semantic_cognition",
    )
    for row in result.operation_trace:
        recorder.log_operation({
            "step": row["operation_id"],
            "operation": "{0} {1}".format(row["operation_id"], row["operation"]),
            "phase": row["stage_id"],
            "status": row["functional_status"],
            "explanation": (
                "{0} Mathematical operation: {1} Biological reference: {2}"
            ).format(
                row["functional_responsibility"],
                row["mathematical_operation"],
                row["biological_reference"],
            ),
            "biological_status": row["biological_status"],
        })
    recorder.log_component_traces(result.stage_trace)
    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
    metadata = recorder.finish(
        "completed" if result.diagnostics.get("valid") else "completed_with_warnings",
        elapsed_ms,
        output_type="CSCResult",
    )
    metadata["stage_count"] = len(result.stage_trace)
    metadata["operation_count"] = len(result.operation_trace)
    metadata["diagnostics_valid"] = bool(result.diagnostics.get("valid"))
    metadata["database_summary"] = recorder.database.summary()
    return metadata
