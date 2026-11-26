"""ore_logger SQLite integration for HACOL (Phase G logging).

Follows the repository's established pattern (ore_confirmation_bias.pipeline):
annotations are synchronized with ``sync_repository(..., include_paths={
"ore_hacol"})``, a run is created, one event is recorded per executed
integrated step with metadata-first payloads (module/class/function, task,
dimensions, algorithm/backend, precision, device, duration, memory, transfers,
error, status, provenance), and the run is finished. When ore_logger is not
importable (standalone deployment of the zip), events fall back to an
in-memory list and the report records that fact.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from .contracts import OptimizationResult, ProvenanceRecord


def _default_paths(database_path: str | Path | None, repository_root: str | Path | None) -> tuple[Path, Path]:
    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root
        else Path(__file__).resolve().parents[1]
    )
    target = (
        Path(database_path).expanduser().resolve()
        if database_path
        else root / "ore_logs" / "ore_logger.sqlite3"
    )
    return root, target


def register_hacol_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register every HACOL class, function, and method in the shared catalogue."""
    root, target = _default_paths(database_path, repository_root)
    try:
        from ore_logger.catalog import sync_repository
        from ore_logger.database import LoggerDatabase
    except ImportError:
        return {
            "registered": False,
            "database_path": str(target),
            "scope": "ore_hacol",
            "note": "ore_logger not importable; annotations unavailable in standalone mode",
        }
    catalogue = sync_repository(LoggerDatabase(target), root, include_paths={"ore_hacol"})
    return {
        **catalogue,
        "database_path": str(target),
        "scope": "ore_hacol",
        "annotation_table": "annotations",
    }


def record_result_to_sqlite(
    result: OptimizationResult,
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the executed step trace and result envelope into shared SQLite."""
    if not isinstance(result, OptimizationResult):
        raise TypeError("result must be an OptimizationResult")
    root, target = _default_paths(database_path, repository_root)
    envelope = result.to_envelope()
    provenance: ProvenanceRecord | None = result.provenance
    common_metadata = {
        "task_id": result.task_id,
        "operation_id": result.operation_id,
        "operation_kind": result.operation_kind,
        "algorithm_identity": result.algorithm_identity,
        "backend": result.backend,
        "dtype": result.precision,
        "device": result.hardware_identity,
        "duration_seconds": result.elapsed_seconds,
        "memory_peak_bytes": result.peak_memory_bytes,
        "transfer_cost_bytes": result.transfer_cost_bytes,
        "error_max_abs": result.measured_error.max_abs if result.measured_error else None,
        "error_max_rel": result.measured_error.max_rel if result.measured_error else None,
        "validation_status": result.validation_status.value,
        "provenance": provenance.to_dict() if provenance else None,
        "config": envelope.get("configuration"),
    }
    try:
        from ore_logger.catalog import sync_repository
        from ore_logger.database import LoggerDatabase
    except ImportError:
        fallback_events = [
            {"step": step.get("step"), "name": step.get("name"), "status": step.get("status")}
            for step in result.step_trace
        ]
        return {
            "run_id": None,
            "database_path": str(target),
            "event_count": len(fallback_events),
            "stored_in_sqlite": False,
            "note": "ore_logger not importable; events retained in memory only",
            "events": fallback_events,
        }
    database = LoggerDatabase(target)
    catalogue = sync_repository(database, root, include_paths={"ore_hacol"})
    run_id = database.create_run(
        script_path="ore_hacol/pipeline",
        module_name="ore_hacol.pipeline",
        entry_kind="hacol_point21",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, step in enumerate(result.step_trace, start=1):
            metadata = {**common_metadata, "step": step.get("step"), "name": step.get("name"),
                        "phase": step.get("phase"), "detail": step.get("detail", "")[:500]}
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="hacol_step",
                source_path="ore_hacol/pipeline.py",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata=metadata,
            )
        database.record_event(
            run_id=run_id,
            sequence_no=len(result.step_trace) + 1,
            event_kind="hacol_result",
            source_path="ore_hacol/pipeline.py",
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            metadata={
                **common_metadata,
                "envelope": {
                    key: value
                    for key, value in envelope.items()
                    if isinstance(value, (str, int, float, bool, list, dict, type(None)))
                },
            },
        )
        database.finish_run(
            run_id,
            "completed" if result.validation_status.value in ("passed", "reference_only", "fallback") else result.validation_status.value,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    except Exception as exc:  # pragma: no cover - database failures propagate after logging
        database.finish_run(
            run_id,
            "failed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            error_summary=f"{type(exc).__name__}: {exc}",
        )
        raise
    return {
        "run_id": run_id,
        "annotation_catalogue": catalogue,
        "database_path": str(target),
        "database_summary": database.summary(),
        "stored_in_sqlite": True,
        "event_count": len(result.step_trace) + 1,
    }


def record_events_in_memory(
    result: OptimizationResult,
) -> dict[str, Any]:
    """Metadata-first in-memory event list (no SQLite dependency)."""
    return {
        "run_id": None,
        "stored_in_sqlite": False,
        "event_count": len(result.step_trace),
        "note": "in-memory fallback; ore_logger integration requires the ORE repository",
    }


class HACOLLogger:
    """Optional logger facade retaining metadata-first events in memory."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record_operation(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Append one metadata-only operation record (never raw payloads)."""
        entry = dict(metadata)
        entry["recorded_at_utc"] = time.time()
        self.events.append(entry)
        return entry
