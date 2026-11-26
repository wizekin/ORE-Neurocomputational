"""SQLite persistence for the agentic layer (specification Section 4).

The adapter adds only the tables the existing repository does not provide:
the ore_logger schema (annotations, runs, events) stays untouched.  All
agentic tables store JSON payloads with content digests so checkpoints and
workflow state can be verified after restart.  Writes are transactional.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
from typing import Any, Iterator, Mapping, Sequence
from pathlib import Path

from ..contracts import content_digest, utc_now

__all__ = ("AgenticDatabase", "SCHEMA_VERSION")

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agentic_workflows (
    workflow_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    idempotency_key TEXT,
    objective TEXT NOT NULL,
    status TEXT NOT NULL,
    plan_version TEXT NOT NULL,
    plan_hash TEXT NOT NULL,
    workspace_revision INTEGER NOT NULL DEFAULT 0,
    result_digest TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_agentic_idempotency
    ON agentic_workflows(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE TABLE IF NOT EXISTS agentic_tasks (
    workflow_id TEXT NOT NULL REFERENCES agentic_workflows(workflow_id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    state TEXT NOT NULL,
    capability TEXT NOT NULL,
    worker_id TEXT DEFAULT '',
    side_effect_class TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (workflow_id, task_id)
);
CREATE TABLE IF NOT EXISTS agentic_task_results (
    workflow_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (workflow_id, task_id, worker_id)
);
CREATE TABLE IF NOT EXISTS agentic_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    plan_hash TEXT NOT NULL,
    digest TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (workflow_id, sequence)
);
CREATE TABLE IF NOT EXISTS agentic_side_effects (
    idempotency_key TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    result_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agentic_outbox (
    entry_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    state TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agentic_episodes (
    episode_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    task_id TEXT DEFAULT '',
    kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    outcome TEXT DEFAULT '',
    retention_class TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agentic_procedures (
    procedure_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL,
    body_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    promoted_at TEXT
);
CREATE TABLE IF NOT EXISTS agentic_procedure_tests (
    test_id TEXT PRIMARY KEY,
    procedure_id TEXT NOT NULL REFERENCES agentic_procedures(procedure_id) ON DELETE CASCADE,
    passed INTEGER NOT NULL,
    detail TEXT DEFAULT '',
    executed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agentic_memory_links (
    link_id TEXT PRIMARY KEY,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agentic_semantic_proposals (
    proposal_id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    lifecycle TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    origin_workflow_id TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agentic_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agentic_spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    attributes_json TEXT NOT NULL,
    workflow_id TEXT DEFAULT '',
    task_id TEXT DEFAULT '',
    parent_span_id TEXT DEFAULT '',
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agentic_a2a_records (
    message_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    delegation_id TEXT NOT NULL,
    validated INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agentic_meta (
    setting_key TEXT PRIMARY KEY,
    setting_value TEXT NOT NULL
);
"""


class AgenticDatabase:
    """Transactional SQLite persistence for workflow and memory state."""

    def __init__(self, path: str | Path) -> None:
        """Open (and if needed create) the agentic database in WAL mode."""
        self._path = Path(path)
        if self._path != Path(":memory:"):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self._path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        with self.transaction():
            self._connection.executescript(_SCHEMA)
            self._connection.execute(
                "INSERT OR IGNORE INTO agentic_meta (setting_key, setting_value) "
                "VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    @property
    def path(self) -> str:
        """Return the database path."""
        return str(self._path)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside one transaction (all-or-nothing)."""
        try:
            yield self._connection
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Execute one statement outside an explicit transaction."""
        cursor = self._connection.execute(sql, parameters)
        self._connection.commit()
        return cursor

    def query(self, sql: str, parameters: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Run a query and return rows as dictionaries."""
        rows = self._connection.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close the connection."""
        self._connection.close()

    # ------------------------------------------------------------ helpers

    @staticmethod
    def dumps(value: Any) -> str:
        """Serialize a payload deterministically for storage."""
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def loads(raw: str) -> Any:
        """Deserialize a stored payload."""
        return json.loads(raw)

    def record_audit(self, kind: str, payload: Mapping[str, Any]) -> None:
        """Persist one audit entry (Steps 34, 43)."""
        self.execute(
            "INSERT INTO agentic_audit (kind, payload_json, created_at) "
            "VALUES (?, ?, ?)", (kind, self.dumps(dict(payload)), utc_now()))

    def recent_audit(self, kind: str | None = None, limit: int = 100
                     ) -> list[dict[str, Any]]:
        """Return recent audit entries."""
        if kind is None:
            return self.query(
                "SELECT * FROM agentic_audit ORDER BY audit_id DESC LIMIT ?",
                (limit,))
        return self.query(
            "SELECT * FROM agentic_audit WHERE kind = ? ORDER BY audit_id DESC LIMIT ?",
            (kind, limit))

    def save_span(self, span: Mapping[str, Any]) -> None:
        """Persist one traced span (Step 43)."""
        self.execute(
            "INSERT OR REPLACE INTO agentic_spans (span_id, trace_id, operation, "
            "status, duration_ms, attributes_json, workflow_id, task_id, "
            "parent_span_id, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (span["span_id"], span["trace_id"], span["operation"], span["status"],
             float(span.get("duration_ms", 0.0)), self.dumps(span.get("attributes", {})),
             span.get("workflow_id", ""), span.get("task_id", ""),
             span.get("parent_span_id", ""), span.get("recorded_at", utc_now())))

    def spans_for_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """Return the spans of one trace."""
        return self.query(
            "SELECT * FROM agentic_spans WHERE trace_id = ? ORDER BY recorded_at",
            (trace_id,))

    def summary(self) -> dict[str, Any]:
        """Return table row counts for diagnostics."""
        tables = ("agentic_workflows", "agentic_tasks", "agentic_task_results",
                  "agentic_checkpoints", "agentic_side_effects", "agentic_outbox",
                  "agentic_episodes", "agentic_procedures", "agentic_procedure_tests",
                  "agentic_memory_links", "agentic_semantic_proposals",
                  "agentic_audit", "agentic_spans", "agentic_a2a_records")
        counts: dict[str, int] = {}
        for table in tables:
            row = self._connection.execute(
                f"SELECT COUNT(*) AS n FROM {table}").fetchone()  # noqa: S608
            counts[table] = int(row["n"])
        return {"schema_version": SCHEMA_VERSION, "tables": counts,
                "digest_of_counts": content_digest(counts)}
