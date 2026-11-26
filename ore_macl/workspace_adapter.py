"""Workspace artifact adapter: versioned reads, scoped snapshots, and proposals.

MACL is not a second blackboard. Agents read consistent versioned snapshots
through this adapter (restricted to their permitted fields), and every proposal
is published through the existing ``FindingPublisher`` validation pipeline so
the state manager remains the sole authority for acceptance, conflict, deferral,
and duplication decisions (spec Steps 24-30).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ore_workspace import (
    Finding,
    Provenance,
    SharedWorkspace,
    UpdateDecision,
    ValidationResult,
    WorkspaceSnapshot,
)
from ore_workspace.models import FindingTarget

from .contracts import AgentDescriptor, AgentTask, ArtifactRef
from .errors import OutOfScopeAccess, StaleRevisionError


def _content_hash(value: Any) -> str:
    """Fingerprint referenced evidence compactly for artifact references."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ScopedSnapshot:
    """Expose only the workspace fields an agent is permitted to observe (Step 4)."""

    snapshot: WorkspaceSnapshot
    agent_id: str
    permitted_fields: frozenset[str]

    @property
    def revision(self) -> int:
        """Return the revision this snapshot was taken at."""
        return self.snapshot.revision

    def read(self, field: str) -> Any:
        """Read one workspace field or raise for out-of-scope access."""
        if field not in self.permitted_fields:
            raise OutOfScopeAccess(
                f"agent {self.agent_id!r} attempted to read out-of-scope field {field!r}"
            )
        if field not in self.snapshot.state:
            return None
        value = self.snapshot.state[field]
        return json.loads(json.dumps(value, default=str)) if isinstance(value, (dict, list)) else value

    def available_fields(self) -> frozenset[str]:
        """Return the fields this agent may read."""
        return frozenset(self.permitted_fields)


class WorkspaceArtifactAdapter:
    """Mediate every MACL interaction with the shared cognitive workspace (Step 24)."""

    def __init__(self, workspace: SharedWorkspace) -> None:
        """Bind the adapter to one authoritative shared workspace."""
        self.workspace = workspace

    @property
    def revision(self) -> int:
        """Return the current authoritative workspace revision."""
        return self.workspace.revision

    def snapshot_for(self, agent: AgentDescriptor, revision: int | None = None) -> ScopedSnapshot:
        """Return a detached, revision-bound snapshot scoped to the agent's permissions."""
        snapshot = self.workspace.snapshot(revision)
        return ScopedSnapshot(
            snapshot=snapshot,
            agent_id=agent.agent_id,
            permitted_fields=frozenset(agent.permitted_workspace_fields),
        )

    def task_snapshot(self, task: AgentTask, agent: AgentDescriptor) -> ScopedSnapshot:
        """Return the exact task-bound revision, scoped to the intersection of permissions."""
        snapshot = self.workspace.snapshot(task.workspace_revision)
        permitted = frozenset(task.permitted_workspace_fields) & frozenset(agent.permitted_workspace_fields)
        return ScopedSnapshot(
            snapshot=snapshot,
            agent_id=agent.agent_id,
            permitted_fields=permitted,
        )

    def propose(
        self,
        agent: AgentDescriptor,
        task: AgentTask,
        *,
        field: str,
        payload: Mapping[str, Any],
        key: str | None = None,
        finding_type: str = "agent_proposal",
        evidence: tuple[Mapping[str, Any], ...] = (),
        rules_applied: tuple[str, ...] = (),
        confidence: float | None = None,
        base_revision: int | None = None,
    ) -> ValidationResult:
        """Publish one proposal through the existing controlled pipeline (Step 28).

        Proposals are hypotheses, not facts: acceptance remains the workspace state
        manager's decision, and this method never mutates state directly.
        """
        if field not in agent.writable_workspace_fields:
            raise OutOfScopeAccess(
                f"agent {agent.agent_id!r} is not permitted to propose writes to {field!r}"
            )
        revision = self.workspace.revision if base_revision is None else base_revision
        provenance = Provenance(
            source=agent.agent_id,
            rules_applied=tuple(rules_applied) or ("macl_agent_proposal",),
            evidence=tuple(dict(item) for item in evidence),
            notes=f"macl task {task.task_id} goal {task.goal_id}",
        )
        finding = Finding(
            workspace_id=self.workspace.workspace_id,
            producer_module=agent.agent_id,
            finding_type=finding_type,
            target=FindingTarget(field=field, key=key),
            payload=dict(payload),
            provenance=provenance,
            base_workspace_revision=revision,
            confidence=confidence,
        )
        return self.workspace.publish(finding)

    def check_revision_currency(self, task: AgentTask) -> tuple[bool, int]:
        """Return whether the task's base revision is still current (Step 27)."""
        current = self.workspace.revision
        return task.workspace_revision == current, current

    def artifact_reference(self, *, artifact_id: str, location: str, payload: Any) -> ArtifactRef:
        """Build a compact artifact reference instead of embedding payloads (Step 25)."""
        return ArtifactRef(
            artifact_id=artifact_id,
            payload_hash=_content_hash(payload),
            location=location,
        )

    def resolve_artifact(self, reference: ArtifactRef) -> bool:
        """Check that a referenced artifact exists in authoritative storage (Step 45)."""
        state = self.workspace.snapshot().state
        if reference.artifact_id == "current_input" and state.get("current_input"):
            return True
        for record in state.get("artifact_references", ()):  # coordinator-published references
            if isinstance(record, Mapping) and record.get("artifact_id") == reference.artifact_id:
                return True
        for field in ("lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses",
                      "retrieved_memories", "discourse_context"):
            items = state.get(field)
            if isinstance(items, Mapping):
                for records in items.values():
                    if any(
                        isinstance(record, Mapping) and record.get("finding_id") == reference.artifact_id
                        for record in (records if isinstance(records, list) else ())
                    ):
                        return True
            elif isinstance(items, list) and any(
                isinstance(record, Mapping) and record.get("finding_id") == reference.artifact_id
                for record in items
            ):
                return True
        for finding_id in self.workspace.findings:
            if finding_id == reference.artifact_id:
                return True
        return False

    def validate_evidence_lineage(
        self, references: tuple[ArtifactRef, ...]
    ) -> tuple[str, ...]:
        """Return errors for evidence references that do not resolve (Step 45)."""
        errors = []
        for reference in references:
            if not self.resolve_artifact(reference):
                errors.append(
                    f"unresolvable evidence reference: {reference.artifact_id} at {reference.location}"
                )
        return tuple(errors)

    def current_input(self) -> str:
        """Return the workspace's current input text."""
        return str(self.workspace.snapshot().state.get("current_input", ""))

    def publication_stats(self) -> dict[str, Any]:
        """Summarize workspace publication decisions for observability."""
        state = self.workspace.snapshot().state
        return {
            "workspace_id": self.workspace.workspace_id,
            "revision": self.workspace.revision,
            "accepted_updates": len(state.get("accepted_updates", ())),
            "pending_updates": len(state.get("pending_updates", ())),
            "rejected_updates": len(state.get("rejected_updates", ())),
            "conflicts": len(self.workspace.conflicts.list(unresolved_only=True)),
        }


def is_stale_decision(result: ValidationResult, task: AgentTask, current_revision: int) -> bool:
    """Detect a deferral caused by stale workspace state (Step 27)."""
    return (
        result.decision is UpdateDecision.DEFERRED
        and task.workspace_revision < current_revision
    )


def raise_if_stale(task: AgentTask, current_revision: int) -> None:
    """Raise for results derived from an obsolete revision that must be recomputed."""
    if task.workspace_revision < current_revision:
        raise StaleRevisionError(
            f"task {task.task_id} is bound to revision {task.workspace_revision} "
            f"but the workspace is at {current_revision}"
        )
