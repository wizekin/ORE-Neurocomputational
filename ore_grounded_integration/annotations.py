"""Register grounded-semantic symbols and persist complete stage traces in ORE SQLite."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import uuid
from typing import Any

from ore_logger import LoggerDatabase, sync_repository

from .contracts import GroundedSemanticResult


def _repository_root(repository_root: str | Path | None = None) -> Path:
    """Resolve the repository root from an explicit path or the installed package location."""
    if repository_root is not None:
        root = Path(repository_root).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[1]
    if not (root / "ore_logger").is_dir():
        raise FileNotFoundError(f"ORE logger package was not found under repository root {root}")
    return root


def register_grounded_annotations(
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index every class and function in the repository into the shared SQLite annotation catalogue.

    The default database is ore_logs/ore_logger.sqlite3 (overridable with ORE_LOGGER_DB).
    New grounded-semantic Python symbols are indexed from their docstrings with source locations,
    signatures and source hashes. No database file is committed to Git.
    """
    from .registry import registry_summary

    root = _repository_root(repository_root)
    target = Path(
        database_path or os.environ.get("ORE_LOGGER_DB")
        or root / "ore_logs" / "ore_logger.sqlite3"
    ).expanduser().resolve()
    database = LoggerDatabase(target)
    report = sync_repository(database, root)
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT symbol_kind, COUNT(*) AS count
               FROM annotations
               WHERE source_path LIKE 'ore_grounded_integration/%'
               GROUP BY symbol_kind"""
        ).fetchall()
    counts = {str(row["symbol_kind"]): int(row["count"]) for row in rows}
    return {
        **report,
        **registry_summary(),
        "database_path": str(database.path),
        "module": "ore_grounded_integration",
        "annotation_counts": counts,
        "symbol_count": sum(counts.values()),
        "storage": "ore_logger.annotations SQLite table",
        "annotation_source": "python_docstring_or_source_analysis",
    }


def record_grounded_result(
    result: GroundedSemanticResult,
    repository_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist every registered stage operation and stage trace as events in the ORE SQLite logger.

    Function/class definitions are synchronized first. Each mathematical operation receives a
    reusable synthetic annotation; each stage receives a component-trace event. The run stores
    metadata and outputs, not raw image/audio payloads.
    """
    from .registry import registry_summary
    from ore_logger.recorder import SQLiteRunRecorder

    root = _repository_root(repository_root)
    target = Path(
        database_path or os.environ.get("ORE_LOGGER_DB")
        or root / "ore_logs" / "ore_logger.sqlite3"
    ).expanduser().resolve()
    started = datetime.now(timezone.utc)
    recorder = SQLiteRunRecorder(
        database_path=target,
        repository_root=root,
        run_id=uuid.uuid4().hex,
        component="ore_grounded_integration",
        callable_name="GroundedSemanticIntegrationProcessor.process",
        started_at_utc=started.isoformat(timespec="milliseconds"),
        entry_kind="grounded_semantic_pipeline",
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
            "biological_status": row.get("biological_status", "reference_only"),
        })
    recorder.log_component_traces(list(result.stage_trace))
    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
    metadata = recorder.finish(
        "completed" if result.diagnostics.get("valid") else "completed_with_warnings",
        elapsed_ms,
        output_type="GroundedSemanticResult",
    )
    metadata.update(registry_summary())
    metadata["stage_trace_count"] = len(result.stage_trace)
    metadata["operation_trace_count"] = len(result.operation_trace)
    metadata["diagnostics_valid"] = bool(result.diagnostics.get("valid"))
    metadata["database_summary"] = recorder.database.summary()
    return metadata
