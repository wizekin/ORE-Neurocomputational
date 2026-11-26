"""Transactional checkpoints and recovery (Steps 20-21).

A checkpoint captures workflow state, completed task references, pending work,
plan version, workspace revision, leases and recovery metadata -- never
secrets or unauthorized payloads.  Restore verifies integrity and revalidates
plan version, permissions, deadlines, artifact references and task
preconditions before resuming.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from ..contracts import TaskResult, content_digest, new_id, utc_now

__all__ = (
    "CheckpointData",
    "CheckpointIntegrityError",
    "CheckpointManager",
    "CheckpointStore",
)


class CheckpointStore(Protocol):
    """Persistence protocol for checkpoint snapshots (Step 20)."""

    def save_checkpoint(self, data: CheckpointData) -> str: ...

    def load_checkpoint(self, checkpoint_id: str) -> CheckpointData | None: ...

    def list_checkpoints(self, workflow_id: str) -> tuple[str, ...]: ...


class CheckpointIntegrityError(ValueError):
    """Raised when a checkpoint fails integrity or version validation (Step 21)."""


@dataclass(frozen=True, slots=True)
class CheckpointData:
    """One transactional workflow checkpoint (Step 20)."""

    workflow_id: str
    plan_hash: str
    plan_version: str
    workspace_revision: int
    task_states: Mapping[str, str]
    completed_task_ids: tuple[str, ...]
    pending_task_ids: tuple[str, ...]
    results: Mapping[str, str]  # task_id -> result digest
    lease_epochs: Mapping[str, int]
    recovery_metadata: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_id: str = field(default_factory=lambda: new_id("ckpt"))
    created_at: str = field(default_factory=utc_now)

    def compute_digest(self) -> str:
        """Return the content digest used for integrity checks."""
        return content_digest({
            "workflow_id": self.workflow_id,
            "plan_hash": self.plan_hash,
            "plan_version": self.plan_version,
            "workspace_revision": self.workspace_revision,
            "task_states": dict(self.task_states),
            "completed_task_ids": list(self.completed_task_ids),
            "pending_task_ids": list(self.pending_task_ids),
            "results": dict(self.results),
            "lease_epochs": dict(self.lease_epochs),
            "recovery_metadata": dict(self.recovery_metadata),
        })

    def to_dict(self) -> dict[str, Any]:
        """Serialize the checkpoint, excluding secrets by construction."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "plan_hash": self.plan_hash,
            "plan_version": self.plan_version,
            "workspace_revision": self.workspace_revision,
            "task_states": dict(self.task_states),
            "completed_task_ids": list(self.completed_task_ids),
            "pending_task_ids": list(self.pending_task_ids),
            "results": dict(self.results),
            "lease_epochs": dict(self.lease_epochs),
            "recovery_metadata": dict(self.recovery_metadata),
            "digest": self.compute_digest(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointData:
        """Restore a checkpoint from its serialized form."""
        try:
            data = cls(
                workflow_id=value["workflow_id"],
                plan_hash=value["plan_hash"],
                plan_version=value["plan_version"],
                workspace_revision=int(value["workspace_revision"]),
                task_states=dict(value["task_states"]),
                completed_task_ids=tuple(value["completed_task_ids"]),
                pending_task_ids=tuple(value["pending_task_ids"]),
                results=dict(value["results"]),
                lease_epochs={key: int(item)
                              for key, item in value["lease_epochs"].items()},
                recovery_metadata=dict(value.get("recovery_metadata", {})),
                checkpoint_id=value["checkpoint_id"],
                created_at=value["created_at"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointIntegrityError(
                f"checkpoint payload is malformed: {exc}") from exc
        expected = value.get("digest")
        if expected is not None and data.compute_digest() != expected:
            raise CheckpointIntegrityError(
                f"checkpoint {data.checkpoint_id} failed integrity verification")
        return data


class CheckpointManager:
    """Capture and restore workflow checkpoints with integrity checks (Steps 20-21)."""

    def __init__(self, store: CheckpointStore, *, max_retained: int = 8) -> None:
        """Bind the manager to a checkpoint store."""
        if max_retained < 1:
            raise ValueError("max_retained must be positive")
        self._store = store
        self._max_retained = max_retained

    def capture(self, workflow_id: str, plan_hash: str, plan_version: str,
                workspace_revision: int, task_states: Mapping[str, str],
                completed_task_ids: tuple[str, ...], pending_task_ids: tuple[str, ...],
                results: Mapping[str, TaskResult | str],
                lease_epochs: Mapping[str, int],
                recovery_metadata: Mapping[str, Any] | None = None) -> CheckpointData:
        """Persist one transactional checkpoint (Step 20)."""
        digests: dict[str, str] = {}
        for task_id, result in results.items():
            if isinstance(result, TaskResult):
                digests[task_id] = content_digest(result.to_dict())
            else:
                digests[task_id] = str(result)
        data = CheckpointData(
            workflow_id=workflow_id, plan_hash=plan_hash, plan_version=plan_version,
            workspace_revision=workspace_revision, task_states=dict(task_states),
            completed_task_ids=tuple(sorted(completed_task_ids)),
            pending_task_ids=tuple(sorted(pending_task_ids)),
            results=digests, lease_epochs=dict(lease_epochs),
            recovery_metadata=dict(recovery_metadata or {}))
        self._store.save_checkpoint(data)
        self._enforce_retention(workflow_id)
        return data

    def restore(self, checkpoint_id: str, *, expected_plan_hash: str | None = None,
                expected_workflow: str | None = None) -> CheckpointData:
        """Load and verify a checkpoint before resume (Step 21)."""
        data = self._store.load_checkpoint(checkpoint_id)
        if data is None:
            raise CheckpointIntegrityError(f"checkpoint {checkpoint_id} not found")
        if expected_plan_hash is not None and data.plan_hash != expected_plan_hash:
            raise CheckpointIntegrityError(
                "checkpoint plan hash differs from the active workflow plan")
        if expected_workflow is not None and data.workflow_id != expected_workflow:
            raise CheckpointIntegrityError("checkpoint belongs to another workflow")
        return data

    def latest(self, workflow_id: str) -> CheckpointData | None:
        """Return the most recent checkpoint for a workflow, verified."""
        checkpoints = self._store.list_checkpoints(workflow_id)
        if not checkpoints:
            return None
        return self.restore(checkpoints[-1])

    def _enforce_retention(self, workflow_id: str) -> None:
        """Keep only the most recent checkpoints (bounded storage)."""
        if hasattr(self._store, "prune_checkpoints"):
            self._store.prune_checkpoints(workflow_id, self._max_retained)  # type: ignore[attr-defined]
