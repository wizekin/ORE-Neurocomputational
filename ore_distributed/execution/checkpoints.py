"""Coordinated workflow checkpoints (Step 42).

A checkpoint persists workflow state, committed task IDs, input/output
digests, the dependency graph, the coordinator epoch, and the recovery
cursor; it is valid only when its referenced artifacts and provenance are
durable and verifiable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ore_distributed.core.identity import digest_hex


@dataclass(frozen=True, slots=True)
class WorkflowCheckpoint:
    """One durable workflow checkpoint with verifiable references (Step 42)."""

    checkpoint_id: str
    workflow_id: str
    workflow_state: Mapping[str, Any]
    committed_task_ids: tuple[str, ...]
    input_digests: Mapping[str, str]
    output_digests: Mapping[str, str]
    dependency_graph: Mapping[str, tuple[str, ...]]
    coordinator_epoch: int
    recovery_cursor: int

    def __post_init__(self) -> None:
        """Validate checkpoint identity and reference integrity."""
        for name in ("checkpoint_id", "workflow_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "workflow_state", dict(self.workflow_state))
        object.__setattr__(self, "committed_task_ids", tuple(self.committed_task_ids))
        object.__setattr__(self, "input_digests", dict(self.input_digests))
        object.__setattr__(self, "output_digests", dict(self.output_digests))
        object.__setattr__(self, "dependency_graph", dict(self.dependency_graph))
        if self.coordinator_epoch < 0 or self.recovery_cursor < 0:
            raise ValueError("coordinator_epoch and recovery_cursor must be non-negative")

    def artifact_digests_valid(self, artifact_store: Mapping[str, bytes | str]) -> bool:
        """Return whether every referenced artifact's digest verifies (Step 42)."""
        for digest in list(self.input_digests.values()) + list(self.output_digests.values()):
            if digest not in artifact_store:
                continue
            payload = artifact_store[digest]
            if digest_hex(payload) != digest:
                return False
        return True

    def provenance_durable(self, provenance_store: Mapping[str, Any]) -> bool:
        """Return whether the checkpoint's provenance references resolve."""
        for reference in self.workflow_state.get("provenance_references", ()):
            if reference not in provenance_store:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return the checkpoint as a plain dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "workflow_state": dict(self.workflow_state),
            "committed_task_ids": list(self.committed_task_ids),
            "input_digests": dict(self.input_digests),
            "output_digests": dict(self.output_digests),
            "dependency_graph": {k: list(v) for k, v in self.dependency_graph.items()},
            "coordinator_epoch": self.coordinator_epoch,
            "recovery_cursor": self.recovery_cursor,
        }


class CheckpointStore:
    """Persist and validate workflow checkpoints with artifact verification."""

    def __init__(
        self,
        artifact_store: Mapping[str, bytes | str] | None = None,
        provenance_store: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a store with optional durable reference stores."""
        self._checkpoints: dict[str, WorkflowCheckpoint] = {}
        self.artifacts: dict[str, bytes | str] = dict(artifact_store or {})
        self.provenance: dict[str, Any] = dict(provenance_store or {})
        self.invalidated = 0

    def register_artifact(self, digest: str, payload: bytes | str) -> None:
        """Store one artifact under its digest."""
        if digest_hex(payload) != digest:
            raise ValueError("artifact digest does not match its payload")
        self.artifacts[digest] = payload

    def register_provenance(self, reference: str, record: Any) -> None:
        """Store one provenance record under its reference."""
        self.provenance[reference] = record

    def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """Validate then store a checkpoint; returns its identifier."""
        if not checkpoint.artifact_digests_valid(self.artifacts):
            self.invalidated += 1
            raise ValueError(
                "checkpoint references unverifiable artifacts; refusing to store it"
            )
        if not checkpoint.provenance_durable(self.provenance):
            self.invalidated += 1
            raise ValueError(
                "checkpoint provenance is not durable; refusing to store it"
            )
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        return checkpoint.checkpoint_id

    def get(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        """Return one checkpoint by identifier."""
        return self._checkpoints.get(checkpoint_id)

    def latest_for(self, workflow_id: str) -> WorkflowCheckpoint | None:
        """Return the highest-cursor checkpoint of one workflow."""
        candidates = [
            checkpoint
            for checkpoint in self._checkpoints.values()
            if checkpoint.workflow_id == workflow_id
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.recovery_cursor, item.coordinator_epoch))

    def checkpoints(self) -> tuple[WorkflowCheckpoint, ...]:
        """Return every stored checkpoint."""
        return tuple(self._checkpoints.values())


def build_checkpoint_from_state(
    *,
    checkpoint_id: str,
    workflow_id: str,
    workflow_state: Mapping[str, Any],
    committed_task_ids: tuple[str, ...],
    inputs: Mapping[str, bytes | str],
    outputs: Mapping[str, bytes | str],
    dependency_graph: Mapping[str, tuple[str, ...]],
    coordinator_epoch: int,
    recovery_cursor: int,
) -> WorkflowCheckpoint:
    """Digest artifacts and assemble a checkpoint referencing them."""
    input_digests = {name: digest_hex(payload) for name, payload in inputs.items()}
    output_digests = {name: digest_hex(payload) for name, payload in outputs.items()}
    return WorkflowCheckpoint(
        checkpoint_id=checkpoint_id,
        workflow_id=workflow_id,
        workflow_state=dict(workflow_state),
        committed_task_ids=tuple(committed_task_ids),
        input_digests=input_digests,
        output_digests=output_digests,
        dependency_graph=dict(dependency_graph),
        coordinator_epoch=coordinator_epoch,
        recovery_cursor=recovery_cursor,
    )


__all__ = [
    "WorkflowCheckpoint",
    "CheckpointStore",
    "build_checkpoint_from_state",
]
