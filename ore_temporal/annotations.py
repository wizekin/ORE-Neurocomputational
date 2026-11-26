"""Structured ore_logger/SQLite annotations for the Point 13 temporal layer.

This module fills the specification's ``ore_temporal/logging/`` role through the
repository's established ``annotations.py`` convention: ``register_annotations`` indexes
the package in the shared SQLite catalogue, ``record_result_to_sqlite`` persists one
operation event per pipeline step, and ``TemporalAnnotationRecorder`` provides
engine-time structured logging that keeps secrets and raw payloads out of the database.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

__all__ = ("TemporalAnnotationRecorder", "record_result_to_sqlite", "register_annotations")


def register_annotations(database_path: str | Path | None = None,
                         repository_root: str | Path | None = None) -> dict[str, Any]:
    """Index every Point 13 package module, class, function, and method in SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    target = (Path(database_path).expanduser().resolve() if database_path
              else root / "ore_logs" / "ore_logger.sqlite3")
    catalogue = sync_repository(LoggerDatabase(target), root,
                                include_paths={"ore_temporal"})
    return {**catalogue, "database_path": str(target),
            "scope": "ore_temporal", "annotation_table": "annotations"}


def record_result_to_sqlite(result: Any, database_path: str | Path | None = None,
                            repository_root: str | Path | None = None) -> dict[str, Any]:
    """Persist one operation event per pipeline step linked to its annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    from .contracts import TemporalResult

    if not isinstance(result, TemporalResult):
        raise TypeError("result must be a TemporalResult")
    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    target = (Path(database_path).expanduser().resolve() if database_path
              else root / "ore_logs" / "ore_logger.sqlite3")
    database = LoggerDatabase(target)
    catalogue = sync_repository(database, root, include_paths={"ore_temporal"})
    run_id = database.create_run(
        script_path="ore_temporal/point13",
        module_name="ore_temporal.pipeline",
        entry_kind="temporal_intelligence",
        working_directory=str(root),
    )
    class_row = database.get_annotation(
        "ore_temporal.pipeline", "TemporalEngine", "class", "ore_temporal/pipeline.py")
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id, sequence_no=sequence, event_kind="operation",
                symbol_id=None,
                class_symbol_id=class_row["symbol_id"] if class_row else None,
                source_path="ore_temporal/pipeline.py",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata={
                    "operation_id": item.get("operation_id"),
                    "operation": item.get("operation"),
                    "step": item.get("step"),
                    "phase": item.get("phase"),
                    "status": item.get("status"),
                    "detail": item.get("detail"),
                })
        database.finish_run(run_id, "completed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3))
    except Exception as exc:
        database.finish_run(run_id, "failed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                            error_summary=type(exc).__name__ + ": " + str(exc))
        raise
    return {"run_id": run_id, "annotation_catalogue": catalogue,
            "database_summary": database.summary(),
            "events_written": len(result.operation_trace),
            "class_annotation_linked": class_row is not None}


_FORBIDDEN_METADATA_KEYS = frozenset({
    "secret", "secrets", "password", "token", "api_key", "credentials", "authorization",
})


class TemporalAnnotationRecorder:
    """Engine-time structured event recorder with payload minimization (Step 50)."""

    def __init__(self, database_path: str | Path | None = None,
                 repository_root: str | Path | None = None,
                 display_label: str = "ore_temporal") -> None:
        """Configure the SQLite target; events are no-ops when the database is absent."""
        self.database_path = database_path
        self.repository_root = repository_root
        self.display_label = display_label
        self._database: Any = None
        self._run_id: str | None = None
        self._sequence = 0
        self.events: list[dict[str, Any]] = []

    def __enter__(self) -> "TemporalAnnotationRecorder":
        """Open a logger database and create a run when possible."""
        try:
            from ore_logger.database import LoggerDatabase

            root = (Path(self.repository_root).expanduser().resolve()
                    if self.repository_root else Path(__file__).resolve().parents[1])
            target = (Path(self.database_path).expanduser().resolve()
                      if self.database_path
                      else root / "ore_logs" / "ore_logger.sqlite3")
            self._database = LoggerDatabase(target)
            self._run_id = self._database.create_run(
                script_path="ore_temporal/point13",
                module_name="ore_temporal.pipeline",
                entry_kind="temporal_intelligence",
                working_directory=str(root),
            )
        except Exception:
            self._database = None
            self._run_id = None
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Finish the run with completed/failed status; never propagate logging errors."""
        if self._database is not None and self._run_id is not None:
            try:
                self._database.finish_run(
                    self._run_id, "failed" if exc_type else "completed",
                    error_summary=(f"{type(exc).__name__}: {exc}" if exc else None))
            except Exception:
                pass
        return False

    def record(self, operation: str, *, task_id: str = "", model: str = "",
               status: str = "completed", metadata: Mapping[str, Any] | None = None,
               duration_ms: float = 0.0, exception_category: str = "",
               provenance: str = "ore_temporal.point13") -> dict[str, Any]:
        """Record one structured event; secrets and raw payloads are stripped."""
        self._sequence += 1
        safe_metadata = {key: value for key, value in (metadata or {}).items()
                         if str(key).lower() not in _FORBIDDEN_METADATA_KEYS
                         and str(key).lower() not in ("raw_stream", "raw_payload")}
        event = {
            "sequence_no": self._sequence,
            "event_kind": "operation",
            "operation": operation,
            "module": "ore_temporal",
            "class": "TemporalEngine",
            "task_id": task_id,
            "model": model,
            "status": status,
            "duration_ms": round(float(duration_ms), 3),
            "exception_category": exception_category,
            "provenance": provenance,
            "metadata": safe_metadata,
        }
        self.events.append(event)
        if self._database is not None and self._run_id is not None:
            try:
                self._database.record_event(
                    run_id=self._run_id, sequence_no=self._sequence,
                    event_kind="operation", elapsed_ms=round(float(duration_ms), 3),
                    metadata=safe_metadata)
            except Exception:
                pass
        return event
