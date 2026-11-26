"""Phase III, Steps 11-15 - shared cognitive state and event exchange.

Step 11 creates one isolated task-scoped workspace per cognitive task.  Step 12
publishes findings through controlled state management (schema validation,
permission checks, optimistic revision preconditions, deduplication and atomic
commits).  Step 13 preserves contradictory hypotheses as conflict records.
Step 14 activates subscribers using bounded event processing with the
relevance pattern ``R_m = sum_j a_mj * Delta_j``.  Step 15 records state
transitions and execution traces with monotonic-clock durations.

The bridge deliberately *reuses* ``ore_workspace`` rather than duplicating it:
WorkspaceManager, WorkspaceStateManager, PermissionPolicy, SubscriptionRegistry
and the event bus remain the single owners of shared state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from ore_workspace import (
    ConflictRecord,
    DeliveryResult,
    Finding,
    FindingTarget,
    ModuleActivationResult,
    Provenance,
    WorkspaceManager,
)
from ore_workspace.models import WorkspaceState
from ore_workspace.workspace import SharedWorkspace

from .core import new_id, sha256_digest, utc_now_iso

WORKSPACE_BRIDGE_SCHEMA = "ore.point24.workspace_bridge.v1"


class CognitiveWorkspaceFactory:
    """Step 11 - create one isolated workspace per cognitive task.

    Each workspace has a unique identity, an initial revision, the task goal,
    module permissions and subscriptions.  Concurrent tasks stay isolated:
    a workspace never leaks state into another, and modules read detached
    snapshots rather than mutating shared state directly.
    """

    def __init__(
        self,
        *,
        repository_root: str | None = None,
        logger_database_path: str | None = None,
        enable_logging: bool = False,
    ) -> None:
        self._manager = WorkspaceManager(
            repository_root=repository_root,
            logger_database_path=logger_database_path,
            enable_logging=enable_logging,
        )

    @property
    def manager(self) -> WorkspaceManager:
        return self._manager

    def create_task_workspace(
        self,
        *,
        task_id: str,
        goal: str,
        input_text: str = "",
        permissions: Any | None = None,
        subscriptions: Any | None = None,
    ) -> SharedWorkspace:
        """Create the task-scoped shared cognitive workspace (Step 11)."""
        if not str(task_id).strip():
            raise ValueError("task_id must be non-empty")
        return self._manager.create_workspace(
            current_input=input_text,
            current_goal=goal,
            workspace_id=f"workspace_{task_id}",
            permissions=permissions,
            subscriptions=subscriptions,
        )

    def close(self) -> None:
        self._manager.close()


def build_finding(
    workspace: SharedWorkspace,
    *,
    producer_module: str,
    finding_type: str,
    field: str,
    payload: Mapping[str, Any],
    provenance: Provenance | None = None,
    dependencies: Iterable[str] = (),
    confidence: float | None = None,
    base_revision: int | None = None,
    key: str | None = None,
) -> Finding:
    """Step 12 - build a typed finding against the current workspace revision."""
    return Finding(
        workspace_id=workspace.workspace_id,
        producer_module=producer_module,
        finding_type=finding_type,
        target=FindingTarget(field=field, key=key),
        payload=dict(payload),
        provenance=provenance or Provenance(source=producer_module),
        base_workspace_revision=(
            workspace.revision if base_revision is None else base_revision
        ),
        dependencies=tuple(dependencies),
        confidence=confidence,
    )


@dataclass(frozen=True, slots=True)
class PublicationOutcome:
    """The explicit Step 12 decision for one submitted finding."""

    decision: str
    finding_id: str
    workspace_revision: int
    reason: str
    conflict_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "finding_id": self.finding_id,
            "workspace_revision": self.workspace_revision,
            "reason": self.reason,
            "conflict_ids": list(self.conflict_ids),
        }


def publish_finding(workspace: SharedWorkspace, finding: Finding) -> PublicationOutcome:
    """Submit a finding through the controlled publication path.

    The workspace state manager validates the schema, checks producer
    permissions, compares the base revision (optimistic concurrency),
    deduplicates identical submissions by stable fingerprint and commits
    accepted changes atomically.  Unauthorized or malformed findings never
    enter authoritative workspace state.
    """
    before = workspace.revision
    result = workspace.publish(finding)
    return PublicationOutcome(
        decision=str(result.decision.value if hasattr(result.decision, "value") else result.decision),
        finding_id=finding.finding_id,
        workspace_revision=workspace.revision,
        reason=str(getattr(result, "message", "") or ""),
        conflict_ids=tuple(
            conflict.conflict_id for conflict in workspace.conflicts.records()
        ) if hasattr(workspace.conflicts, "records") else (),
    )


def inspect_conflicts(workspace: SharedWorkspace) -> tuple[dict[str, Any], ...]:
    """Step 13 - return every conflict record with both alternatives retained.

    Absence of agreement is an abstention or unresolved conflict, never proof
    that the most popular hypothesis is correct; resolution happens through a
    new event, and earlier alternatives are never erased.
    """
    records = []
    registry = workspace.conflicts
    listing = getattr(registry, "list", None)
    if callable(listing):
        conflicts = listing(workspace.workspace_id)
    else:  # pragma: no cover - defensive for alternative registries
        conflicts = getattr(registry, "records", lambda: [])()
    for conflict in conflicts:
        records.append({
            "conflict_id": conflict.conflict_id,
            "conflict_type": conflict.conflict_type,
            "status": conflict.status,
            "conflicting_findings": list(conflict.conflicting_findings),
            "affected_fields": list(conflict.affected_fields),
            "preferred_finding_id": conflict.preferred_finding_id,
            "resolution_candidates": list(conflict.resolution_candidates),
        })
    return tuple(records)


def register_bounded_subscriber(
    workspace: SharedWorkspace,
    module_id: str,
    callback: Callable[[Any], None],
    *,
    fields: Iterable[str],
    threshold: float = 0.0,
    sensitivity: Mapping[str, float] | None = None,
) -> None:
    """Step 14 - register a subscriber with explicit field relevance weights.

    Activation happens only when ``R_m = sum_j a_mj * Delta_j >= threshold_m``
    and all non-score constraints pass; the workspace event bus isolates
    subscriber exceptions and deduplicates repeated activation.
    """
    workspace.register_consumer(
        module_id, callback, fields, threshold=threshold,
        sensitivity=dict(sensitivity) if sensitivity else None,
    )


def bounded_activation_report(workspace: SharedWorkspace) -> dict[str, Any]:
    """Return the bounded activation evidence: deliveries and queued modules."""
    deliveries = list(getattr(workspace, "delivery_results", []))
    return {
        "schema": WORKSPACE_BRIDGE_SCHEMA,
        "workspace_id": workspace.workspace_id,
        "workspace_revision": workspace.revision,
        "delivery_results": [
            {
                "module_id": item.module_id,
                "event_id": item.event_id,
                "delivered": item.delivered,
                "error": item.error,
                "timed_out": item.timed_out,
            }
            for item in deliveries
        ],
        "events": [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "changed_fields": list(event.changed_fields),
                "workspace_revision": event.workspace_revision,
            }
            for event in getattr(workspace, "events", [])
        ],
    }


class TraceRecorder:
    """Step 15 - record state transitions and execution traces.

    Durations are computed from monotonic clocks; payloads stay out of general
    event metadata (only fingerprints and references are recorded).  The
    recorder is an observer: it is never the message bus or the source of
    cognitive truth.
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._clocks: dict[str, float] = {}
        self._event_keys: set[tuple[str, str]] = set()

    def start(self, operation_id: str, *, module_id: str, task_id: str, run_id: str) -> None:
        self._clocks[operation_id] = time.monotonic()

    def record(
        self,
        operation_id: str,
        *,
        module_id: str,
        task_id: str,
        run_id: str,
        method: str,
        status: str,
        metadata: Mapping[str, Any] | None = None,
        artifact_references: Iterable[str] = (),
        input_fingerprint: str | None = None,
        output_fingerprint: str | None = None,
        warning: str | None = None,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        """Record one traced invocation with a monotonic duration."""
        started = self._clocks.pop(operation_id, None)
        duration_ms = (
            round((time.monotonic() - started) * 1000.0, 3) if started is not None else None
        )
        entry = {
            "trace_id": new_id("trace"),
            "task_id": task_id,
            "run_id": run_id,
            "module_id": module_id,
            "annotation": f"{module_id}.{method}",
            "method": method,
            "status": status,
            "duration_ms": duration_ms,
            "metadata": dict(metadata or {}),
            "artifact_references": list(artifact_references),
            "input_fingerprint": input_fingerprint,
            "output_fingerprint": output_fingerprint,
            "warning": warning,
            "error_type": error_type,
            "recorded_at": utc_now_iso(),
        }
        self._entries.append(entry)
        return dict(entry)

    def entries(self, *, module_id: str | None = None) -> tuple[dict[str, Any], ...]:
        if module_id is None:
            return tuple(self._entries)
        return tuple(item for item in self._entries if item["module_id"] == module_id)

    def latency_statistics(self, module_id: str) -> dict[str, float] | None:
        """Measured latency distribution from recorded runs, never estimates."""
        durations = [
            item["duration_ms"] for item in self._entries
            if item["module_id"] == module_id and item["duration_ms"] is not None
        ]
        if not durations:
            return None
        ordered = sorted(durations)
        return {
            "count": len(ordered),
            "mean_ms": sum(ordered) / len(ordered),
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
            "p50_ms": ordered[len(ordered) // 2],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "schema": WORKSPACE_BRIDGE_SCHEMA,
            "entry_count": len(self._entries),
            "modules": sorted({item["module_id"] for item in self._entries}),
            "digest": sha256_digest({"entries": self._entries}),
        }


__all__ = [
    "WORKSPACE_BRIDGE_SCHEMA",
    "CognitiveWorkspaceFactory",
    "PublicationOutcome",
    "TraceRecorder",
    "bounded_activation_report",
    "build_finding",
    "inspect_conflicts",
    "publish_finding",
    "register_bounded_subscriber",
]
