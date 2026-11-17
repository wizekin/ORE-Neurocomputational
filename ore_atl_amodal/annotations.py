"""SQLite annotation indexing and execution recording for the amodal ATL pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid
from typing import Any

from ore_logger import LoggerDatabase, sync_repository
from ore_logger.recorder import SQLiteRunRecorder

from .contracts import AmodalATLResult


def index_atl_amodal_annotations(
    repository_root: str | Path,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index every class, function, and module in ore_atl_amodal into the repository's shared SQLite annotation catalogue."""
    root = Path(repository_root).expanduser().resolve()
    target = Path(database_path) if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    report = sync_repository(database, root)
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT symbol_kind, COUNT(*) AS count
            FROM annotations
            WHERE source_path LIKE 'ore_atl_amodal/%'
            GROUP BY symbol_kind
            """
        ).fetchall()
    counts = {str(row["symbol_kind"]): int(row["count"]) for row in rows}
    return {
        **report,
        "database_path": str(database.path),
        "atl_amodal_annotations": counts,
        "atl_amodal_symbol_count": int(sum(counts.values())),
    }


def record_atl_amodal_result(
    result: AmodalATLResult,
    repository_root: str | Path,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist every ATL sub-operation and stage as SQLite events linked to AST-indexed class and function annotations."""
    root = Path(repository_root).expanduser().resolve()
    target = Path(database_path) if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    started = datetime.now(timezone.utc)
    run_id = uuid.uuid4().hex
    recorder = SQLiteRunRecorder(
        database_path=target,
        repository_root=root,
        run_id=run_id,
        component="ore_atl_amodal",
        callable_name="AmodalATLProcessor.process",
        started_at_utc=started.isoformat(timespec="milliseconds"),
        entry_kind="atl_amodal_pipeline",
    )
    for row in result.operation_trace:
        recorder.log_operation({
            "step": row["operation_id"],
            "operation": f"{row['operation_id']} {row['operation']}",
            "phase": row["stage_id"],
            "status": row["functional_status"],
            "explanation": (
                f"{row['functional_responsibility']} "
                f"Mathematical operation: {row['mathematical_operation']} "
                f"Biological reference: {row['biological_reference']}"
            ),
            "biological_status": row["biological_status"],
        })
    recorder.log_component_traces(list(result.stage_trace))
    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
    metadata = recorder.finish(
        "completed" if result.diagnostics.get("valid") else "completed_with_warnings",
        elapsed_ms,
        output_type="AmodalATLResult",
    )
    metadata["stage_count"] = len(result.stage_trace)
    metadata["operation_count"] = len(result.operation_trace)
    metadata["diagnostics_valid"] = bool(result.diagnostics.get("valid"))
    metadata["database_summary"] = recorder.database.summary()
    return metadata
