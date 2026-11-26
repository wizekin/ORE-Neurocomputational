"""WorkflowStore: create, transition, checkpoint and recover (spec Section 4).

The store implements the specification's illustrative WorkflowStore contract
(``create(spec)``, ``transition(workflow_id, event)``, ``checkpoint``, 
``load_checkpoint``) on top of :class:`AgenticDatabase` with transactional
state changes and integrity-verified checkpoints.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..contracts import TaskResult, WorkflowStatus, content_digest, utc_now
from ..workflow.checkpoints import CheckpointData, CheckpointStore
from .database import AgenticDatabase

__all__ = ("WorkflowStore", "WorkflowEvent")


class WorkflowEvent:
    """Namespaced workflow event identifiers for transitions (Step 17)."""

    DISPATCH = "dispatch"
    TASK_SUCCEEDED = "task_succeeded"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    TASK_COMPENSATED = "task_compensated"
    CANCELLED = "cancelled"
    CLOSED = "closed"


class WorkflowStore(CheckpointStore):
    """Transactional workflow persistence (Steps 8, 17, 20-21)."""

    def __init__(self, database: AgenticDatabase) -> None:
        """Bind the store to the agentic database."""
        self._db = database

    # ------------------------------------------------------------------ create

    def create(self, spec: Mapping[str, Any], *, idempotency_key: str | None = None
               ) -> str:
        """Persist an immutable plan version and hash (Step 8).

        Raises ``ValueError`` when the idempotency key is already registered,
        preventing duplicate work at the storage boundary (Step 4).
        """
        workflow_id = str(spec["workflow_id"])
        plan_hash = str(spec["plan_hash"])
        now = utc_now()
        try:
            with self._db.transaction() as connection:
                connection.execute(
                    "INSERT INTO agentic_workflows (workflow_id, request_id, "
                    "request_digest, idempotency_key, objective, status, "
                    "plan_version, plan_hash, workspace_revision, created_at, "
                    "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (workflow_id, str(spec.get("request_id", "")),
                     str(spec.get("request_digest", "")), idempotency_key,
                     str(spec.get("objective", "")),
                     WorkflowStatus.COMPILED.value, str(spec.get("plan_version", "1")),
                     plan_hash, int(spec.get("workspace_revision", 0)), now, now))
                for task_id, task in dict(spec.get("tasks", {})).items():
                    connection.execute(
                        "INSERT OR REPLACE INTO agentic_tasks (workflow_id, task_id, "
                        "state, capability, worker_id, side_effect_class, attempts, "
                        "updated_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                        (workflow_id, task_id, "pending", str(task.get("capability", "")),
                         "", str(task.get("side_effect_class", "read_only")), now))
        except Exception as exc:
            if "UNIQUE" in str(exc) or "idx_agentic_idempotency" in str(exc):
                raise ValueError(
                    f"idempotency key {idempotency_key!r} already exists") from exc
            raise
        return workflow_id

    # ------------------------------------------------------------------ reads

    def get(self, workflow_id: str) -> dict[str, Any] | None:
        """Return one workflow row."""
        rows = self._db.query(
            "SELECT * FROM agentic_workflows WHERE workflow_id = ?", (workflow_id,))
        return rows[0] if rows else None

    def find_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        """Find a workflow by idempotency key (Step 4)."""
        rows = self._db.query(
            "SELECT * FROM agentic_workflows WHERE idempotency_key = ?", (key,))
        return rows[0] if rows else None

    def list_workflows(self, status: str | None = None) -> list[dict[str, Any]]:
        """List workflows, optionally filtered by status."""
        if status is None:
            return self._db.query(
                "SELECT * FROM agentic_workflows ORDER BY created_at")
        return self._db.query(
            "SELECT * FROM agentic_workflows WHERE status = ? ORDER BY created_at",
            (status,))

    def tasks(self, workflow_id: str) -> list[dict[str, Any]]:
        """Return the task rows of one workflow."""
        return self._db.query(
            "SELECT * FROM agentic_tasks WHERE workflow_id = ? ORDER BY task_id",
            (workflow_id,))

    # ----------------------------------------------------------- transitions

    def transition(self, workflow_id: str, event: str,
                   *, new_status: str | None = None,
                   task_id: str | None = None,
                   task_state: str | None = None,
                   worker_id: str | None = None,
                   attempts: int | None = None) -> None:
        """Apply one transactional state transition (Step 17)."""
        now = utc_now()
        with self._db.transaction() as connection:
            if new_status is not None:
                connection.execute(
                    "UPDATE agentic_workflows SET status = ?, updated_at = ? "
                    "WHERE workflow_id = ?", (new_status, now, workflow_id))
            if task_id is not None and task_state is not None:
                connection.execute(
                    "UPDATE agentic_tasks SET state = ?, updated_at = ?, "
                    "worker_id = COALESCE(?, worker_id), "
                    "attempts = COALESCE(?, attempts) WHERE workflow_id = ? "
                    "AND task_id = ?",
                    (task_state, now, worker_id, attempts, workflow_id, task_id))
            connection.execute(
                "INSERT INTO agentic_audit (kind, payload_json, created_at) "
                "VALUES (?, ?, ?)",
                ("workflow_event", self._db.dumps({
                    "workflow_id": workflow_id, "event": event, "task_id": task_id,
                    "task_state": task_state, "new_status": new_status}), now))

    def record_result(self, result: TaskResult, workflow_id: str) -> None:
        """Persist one task result with provenance (Step 13)."""
        self._db.execute(
            "INSERT OR REPLACE INTO agentic_task_results (workflow_id, task_id, "
            "worker_id, status, payload_json, provenance_json, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (workflow_id, result.task_id, result.worker_id, result.status.value,
             self._db.dumps(dict(result.payload)),
             self._db.dumps(result.provenance.to_dict()
                            if result.provenance else {}), utc_now()))

    def results(self, workflow_id: str) -> list[dict[str, Any]]:
        """Return the recorded results of one workflow."""
        return self._db.query(
            "SELECT * FROM agentic_task_results WHERE workflow_id = ? "
            "ORDER BY recorded_at, task_id", (workflow_id,))

    # ------------------------------------------------------------- checkpoints

    def save_checkpoint(self, data: CheckpointData) -> str:
        """Persist one checkpoint transactionally (Step 20)."""
        sequence = len(self.list_checkpoints(data.workflow_id)) + 1
        self._db.execute(
            "INSERT INTO agentic_checkpoints (checkpoint_id, workflow_id, sequence, "
            "plan_hash, digest, snapshot_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (data.checkpoint_id, data.workflow_id, sequence, data.plan_hash,
             data.compute_digest(), self._db.dumps(data.to_dict()), data.created_at))
        return data.checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> CheckpointData | None:
        """Load and integrity-verify one checkpoint (Step 21)."""
        rows = self._db.query(
            "SELECT * FROM agentic_checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,))
        if not rows:
            return None
        row = rows[0]
        data = CheckpointData.from_dict(self._db.loads(row["snapshot_json"]))
        if data.compute_digest() != row["digest"]:
            raise ValueError(
                f"checkpoint {checkpoint_id} failed integrity verification")
        return data

    def list_checkpoints(self, workflow_id: str) -> tuple[str, ...]:
        """List checkpoint IDs in sequence order (Step 21)."""
        rows = self._db.query(
            "SELECT checkpoint_id FROM agentic_checkpoints WHERE workflow_id = ? "
            "ORDER BY sequence", (workflow_id,))
        return tuple(row["checkpoint_id"] for row in rows)

    def prune_checkpoints(self, workflow_id: str, keep: int) -> None:
        """Keep only the most recent ``keep`` checkpoints (Step 20)."""
        rows = self.list_checkpoints(workflow_id)
        stale = rows[:-keep] if len(rows) > keep else ()
        for checkpoint_id in stale:
            self._db.execute(
                "DELETE FROM agentic_checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,))

    # ------------------------------------------------------------- side ledger

    def record_side_effect(self, idempotency_key: str, operation: str,
                           status: str, result_digest: str) -> None:
        """Persist one side-effect receipt (Step 23)."""
        self._db.execute(
            "INSERT OR IGNORE INTO agentic_side_effects (idempotency_key, operation, "
            "status, result_digest, created_at) VALUES (?, ?, ?, ?, ?)",
            (idempotency_key, operation, status, result_digest, utc_now()))

    def side_effect(self, idempotency_key: str) -> dict[str, Any] | None:
        """Return one side-effect receipt (Step 23)."""
        rows = self._db.query(
            "SELECT * FROM agentic_side_effects WHERE idempotency_key = ?",
            (idempotency_key,))
        return rows[0] if rows else None

    def mark_workflow_result(self, workflow_id: str,
                             result: Mapping[str, Any]) -> None:
        """Persist the final result digest at close (Step 50)."""
        self._db.execute(
            "UPDATE agentic_workflows SET result_digest = ?, updated_at = ? "
            "WHERE workflow_id = ?",
            (content_digest(dict(result)), utc_now(), workflow_id))
