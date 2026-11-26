"""SQLite persistence for RLMS audit events (Point 14, Step 54).

The audit store is deliberately simple and self-contained: one append-only table
with the sanitized event schema, plus query helpers that link events to tasks and
models.  Raw sensitive payloads are structurally impossible to store because
:class:`~ore_learning.audit.events.AuditEvent` sanitizes at construction time.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
import time
from typing import Any, Sequence

from .events import AUDIT_FORBIDDEN_KEYS, AuditEvent, sanitize_audit_metadata

__all__ = ("AuditStore", "connect_audit_store")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ore_learning_audit_events (
    event_id TEXT PRIMARY KEY,
    sequence_no INTEGER NOT NULL,
    event_kind TEXT NOT NULL,
    component TEXT NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    task_id TEXT NOT NULL DEFAULT '',
    model_id TEXT NOT NULL DEFAULT '',
    model_version TEXT NOT NULL DEFAULT '',
    duration_ms REAL NOT NULL DEFAULT 0.0,
    exception_category TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    correction_of TEXT NOT NULL DEFAULT '',
    recorded_at REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS ore_learning_audit_model
    ON ore_learning_audit_events (model_id, model_version, sequence_no);
CREATE INDEX IF NOT EXISTS ore_learning_audit_task
    ON ore_learning_audit_events (task_id, sequence_no);
"""


class AuditStore:
    """Append-only SQLite audit log with task and model linkage (Step 54)."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        """Open (or create) the audit database."""
        if database_path is None:
            database_path = Path("ore_logs") / "ore_learning_lifecycle.sqlite3"
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()
        self._sequence = int(self._count_rows()) + 1
        self.buffered: list[AuditEvent] = []

    def _count_rows(self) -> int:
        """Count stored events to continue the sequence numbering."""
        row = self._connection.execute(
            "SELECT COUNT(*) AS n FROM ore_learning_audit_events").fetchone()
        return int(row["n"])

    def append(self, event: AuditEvent) -> AuditEvent:
        """Persist one sanitized audit event; append-only, corrections link back."""
        event.sequence = self._sequence
        self._sequence += 1
        if not event.recorded_at:
            event.recorded_at = time.time()
        self._connection.execute(
            "INSERT OR REPLACE INTO ore_learning_audit_events "
            "(event_id, sequence_no, event_kind, component, operation, status, task_id, "
            " model_id, model_version, duration_ms, exception_category, metadata_json, "
            " correction_of, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (event.event_id, event.sequence, event.event_kind, event.component,
             event.operation, event.status, event.task_id, event.model_id,
             event.model_version, float(event.duration_ms), event.exception_category,
             _dump_json(event.metadata), event.correction_of, event.recorded_at))
        self._connection.commit()
        self.buffered.append(event)
        return event

    def record(self, component: str, operation: str, event_kind: str, *,
               status: str = "completed", task_id: str = "", model_id: str = "",
               model_version: str = "", duration_ms: float = 0.0,
               exception_category: str = "",
               metadata: dict[str, Any] | None = None,
               correction_of: str = "") -> AuditEvent:
        """Build and persist one audit event in a single call."""
        event = AuditEvent(
            event_id=f"aud-{self._sequence:06d}-{operation}", component=component,
            operation=operation, event_kind=event_kind, status=status, task_id=task_id,
            model_id=model_id, model_version=model_version,
            duration_ms=duration_ms, exception_category=exception_category,
            metadata=metadata, correction_of=correction_of)
        return self.append(event)

    def correct(self, original_event_id: str, component: str, operation: str, *,
                metadata: dict[str, Any] | None = None, **kwargs: Any) -> AuditEvent:
        """Corrections become new linked events; the original record is untouched."""
        kwargs.setdefault("event_kind", "correction")
        kwargs["metadata"] = {**(metadata or {}), "correction_of": original_event_id}
        return self.record(component, operation, correction_of=original_event_id,
                           **kwargs)

    def query(self, *, model_id: str | None = None, task_id: str | None = None,
              event_kind: str | None = None,
              limit: int = 500) -> list[dict[str, Any]]:
        """Query persisted events linked to a model, task, and/or kind."""
        clauses, parameters = [], []
        if model_id is not None:
            clauses.append("model_id = ?")
            parameters.append(model_id)
        if task_id is not None:
            clauses.append("task_id = ?")
            parameters.append(task_id)
        if event_kind is not None:
            clauses.append("event_kind = ?")
            parameters.append(event_kind)
        statement = "SELECT * FROM ore_learning_audit_events"
        if clauses:
            statement += " WHERE " + " AND ".join(clauses)
        statement += " ORDER BY sequence_no ASC LIMIT ?"
        parameters.append(int(limit))
        rows = self._connection.execute(statement, parameters).fetchall()
        return [_row_to_dict(row) for row in rows]

    def assert_no_forbidden_keys(self) -> list[str]:
        """Verify no stored metadata carries forbidden keys (acceptance test)."""
        rows = self._connection.execute(
            "SELECT event_id, metadata_json FROM ore_learning_audit_events").fetchall()
        violations = []
        for row in rows:
            import json as _json
            metadata = _json.loads(row["metadata_json"])
            leaked = sorted(set(metadata) & AUDIT_FORBIDDEN_KEYS)
            if leaked:
                violations.append(f"{row['event_id']}: {leaked}")
        return violations

    def summary(self) -> dict[str, Any]:
        """Summarize the audit database."""
        total = self._count_rows()
        kinds = self._connection.execute(
            "SELECT event_kind, COUNT(*) AS n FROM ore_learning_audit_events "
            "GROUP BY event_kind").fetchall()
        return {"database_path": str(self.path), "events": total,
                "by_kind": {row["event_kind"]: int(row["n"]) for row in kinds}}

    def close(self) -> None:
        """Close the underlying connection."""
        self._connection.close()

    def __enter__(self) -> "AuditStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Context manager exit closes the store."""
        self.close()
        return False


def _dump_json(metadata: dict[str, Any]) -> str:
    """Serialize metadata deterministically."""
    import json as _json

    return _json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert one row to a plain dictionary."""
    import json as _json

    return {"event_id": row["event_id"], "sequence": int(row["sequence_no"]),
            "event_kind": row["event_kind"], "component": row["component"],
            "operation": row["operation"], "status": row["status"],
            "task_id": row["task_id"], "model_id": row["model_id"],
            "model_version": row["model_version"],
            "duration_ms": row["duration_ms"],
            "exception_category": row["exception_category"],
            "metadata": _json.loads(row["metadata_json"]),
            "correction_of": row["correction_of"], "recorded_at": row["recorded_at"]}


def connect_audit_store(database_path: str | Path | None = None) -> AuditStore:
    """Open an audit store; returns None-safe helper for optional wiring."""
    return AuditStore(database_path)
