"""AST symbol indexing and 21-stage operation tracing for ORE SQLite."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import uuid
from typing import Any

from ore_logger import LoggerDatabase, sync_repository
from ore_logger.recorder import SQLiteRunRecorder

from .contracts import DistributedSemanticResult
from .registry import STAGE_REGISTRY


def _symbol_identifier(source_path: str, kind: str, qualified_name: str) -> str:
    """Build a deterministic SQLite annotation identifier for a stage or operation."""
    raw = f"{source_path}|{kind}|{qualified_name}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def index_distributed_annotations(
    repository_root: str | Path,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index every package module, class, and function and persist stage/operation annotations.

    ORE's AST catalogue supplies symbol-level annotations from Python docstrings.
    This function additionally registers the source algorithm's stage and
    sub-operation descriptions, including their mathematical and biological maps.
    """
    root = Path(repository_root).expanduser().resolve()
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    report = sync_repository(database, root)
    source_path = "ore_distributed_only/registry.py"
    for stage in STAGE_REGISTRY:
        stage_annotation = (
            f"{stage.objective} Biological interpretation: {stage.biological_interpretation}"
        )
        database.register_annotation(
            symbol_id=_symbol_identifier(source_path, "stage", stage.stage_id),
            module_name="ore_distributed_only",
            qualified_name=stage.stage_id,
            symbol_kind="stage",
            source_path=source_path,
            annotation=stage_annotation,
            annotation_source="operation_catalog",
            source_hash=hashlib.sha256(stage_annotation.encode("utf-8")).hexdigest(),
        )
        for operation in stage.operations:
            qualified_name = f"{stage.stage_id}.{operation.name}"
            annotation = (
                f"{operation.functional_responsibility} "
                f"Mathematical operation: {operation.equation} "
                f"Biological reference: {operation.biological_reference}"
            )
            database.register_annotation(
                symbol_id=_symbol_identifier(source_path, "operation", qualified_name),
                module_name="ore_distributed_only",
                qualified_name=qualified_name,
                symbol_kind="operation",
                source_path=source_path,
                annotation=annotation,
                annotation_source="operation_catalog",
                source_hash=hashlib.sha256(annotation.encode("utf-8")).hexdigest(),
            )
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT symbol_kind, COUNT(*) AS count
            FROM annotations
            WHERE source_path LIKE 'ore_distributed_only/%'
            GROUP BY symbol_kind
            """
        ).fetchall()
    counts = {str(row["symbol_kind"]): int(row["count"]) for row in rows}
    return {
        **report,
        "database_path": str(database.path),
        "distributed_annotations": counts,
        "distributed_symbol_count": int(sum(counts.values())),
        "stage_count": len(STAGE_REGISTRY),
        "operation_count": sum(len(stage.operations) for stage in STAGE_REGISTRY),
    }


def record_distributed_result(
    result: DistributedSemanticResult,
    repository_root: str | Path,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist class/function annotations and every stage/sub-operation execution to SQLite."""
    root = Path(repository_root).expanduser().resolve()
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    started = datetime.now(timezone.utc)
    run_id = uuid.uuid4().hex
    recorder = SQLiteRunRecorder(
        database_path=target,
        repository_root=root,
        run_id=run_id,
        component="ore_distributed_only",
        callable_name="DistributedOnlyProcessor.process",
        started_at_utc=started.isoformat(timespec="milliseconds"),
        entry_kind="distributed_only_semantic_pipeline",
    )
    for row in result.operation_trace:
        recorder.log_operation({
            "step": row["operation_id"],
            "operation": row["operation_id"],
            "phase": row["stage_id"],
            "status": row["functional_status"],
            "explanation": (
                f"{row['functional_responsibility']} "
                f"Mathematical operation: {row['mathematical_operation']} "
                f"Biological reference: {row['biological_reference']}"
            ),
            "biological_status": row["biological_status"],
            "output": row.get("output"),
        })
    recorder.log_component_traces(result.stage_trace)
    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
    metadata = recorder.finish(
        "completed" if result.diagnostics.get("valid") else "completed_with_warnings",
        elapsed_ms,
        output_type="DistributedSemanticResult",
    )
    metadata["stage_count"] = len(result.stage_trace)
    metadata["operation_count"] = len(result.operation_trace)
    metadata["diagnostics_valid"] = bool(result.diagnostics.get("valid"))
    metadata["database_summary"] = recorder.database.summary()
    return metadata
