"""Shared cognitive workspace connector over typed events (Step 46).

The workspace publishes typed events rather than exposing its entire internal
memory as a replicated database: coordinator decisions and task ownership use
the authoritative consensus path; selected findings use mergeable CRDT paths;
agent scratch state stays local; every cross-node item retains origin and
confidence metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ore_workspace.models import WorkspaceEvent

from ore_distributed.replication.replica import CrdtReplica, ObservationStatus


#: Workspace event types routed to the authoritative consensus path.
AUTHORITATIVE_EVENT_TYPES = frozenset(
    {"coordinator_decision", "task_ownership", "workflow_transition", "policy_commit"}
)

#: Workspace event types routed to the mergeable collaborative path.
MERGEABLE_EVENT_TYPES = frozenset(
    {"finding_published", "annotation_added", "observation_contributed"}
)

#: Workspace event types that must stay local by default.
LOCAL_EVENT_TYPES = frozenset(
    {"scratch_update", "cache_refresh", "intermediate_computation"}
)


@dataclass(frozen=True, slots=True)
class RoutedEvent:
    """One workspace event classified onto a distribution path (Step 46)."""

    event_id: str
    event_type: str
    path: str  # authoritative | mergeable | local
    origin: str
    confidence: float

    def __post_init__(self) -> None:
        """Validate the routing record."""
        if not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if self.path not in {"authoritative", "mergeable", "local"}:
            raise ValueError("path must be authoritative, mergeable, or local")
        if not self.origin.strip():
            raise ValueError("origin must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


class WorkspaceDistributedConnector:
    """Route typed workspace events onto the right distribution path (Step 46)."""

    def __init__(
        self,
        node_id: str,
        replica: CrdtReplica | None = None,
    ) -> None:
        """Create a connector for one node with its mergeable replica."""
        if not node_id.strip():
            raise ValueError("node_id must be non-empty")
        self.node_id = node_id
        self.replica = replica or CrdtReplica(node_id)
        self.routed: list[RoutedEvent] = []
        self.authoritative_queue: list[WorkspaceEvent] = []
        self._scratch: dict[str, Any] = {}

    def route_event(self, event: WorkspaceEvent, confidence: float = 1.0) -> RoutedEvent:
        """Classify one workspace event onto a distribution path."""
        event_type = str(getattr(event, "event_type", "")).strip()
        if not event_type:
            raise ValueError("workspace event carries no event_type")
        if event_type in AUTHORITATIVE_EVENT_TYPES:
            path = "authoritative"
            self.authoritative_queue.append(event)
        elif event_type in MERGEABLE_EVENT_TYPES:
            path = "mergeable"
            for finding_id in event.finding_ids:
                self.replica.apply_local(
                    content=str(finding_id),
                    kind="workspace_finding",
                    context={"event_type": event_type},
                )
        elif event_type in LOCAL_EVENT_TYPES:
            path = "local"
        else:
            # Unknown event types default to local visibility only, never to
            # replication: routing is explicit, not guessed (Step 46).
            path = "local"
        routed = RoutedEvent(
            event_id=str(event.event_id),
            event_type=event_type,
            path=path,
            origin=str(event.producer_module),
            confidence=float(confidence),
        )
        self.routed.append(routed)
        return routed

    def publish_findings(self, findings: Mapping[str, float]) -> tuple[str, ...]:
        """Publish selected findings on the mergeable path with confidence metadata.

        Returns the event identifiers assigned to each finding; every item
        retains origin and confidence metadata for downstream consumers.
        """
        event_ids: list[str] = []
        for finding, confidence in findings.items():
            observation = self.replica.apply_local(
                content=str(finding),
                kind="selected_finding",
                context={"confidence": float(confidence), "origin": self.node_id},
            )
            event_ids.append(observation.event_id)
        return tuple(event_ids)

    def local_scratch(self, key: str, value: Any) -> None:
        """Record agent-local scratch state that is never replicated."""
        self._scratch[key] = value
        self.routed.append(
            RoutedEvent(
                event_id=f"scratch:{key}",
                event_type="scratch_update",
                path="local",
                origin=self.node_id,
                confidence=0.0,
            )
        )

    def pending_authoritative(self) -> tuple[WorkspaceEvent, ...]:
        """Return the workspace events queued for the authoritative path."""
        return tuple(self.authoritative_queue)

    def mergeable_status(self) -> dict[str, ObservationStatus]:
        """Return the dissemination status of every published finding."""
        return {
            event_id: status
            for event_id, status in self.replica._statuses.items()  # noqa: SLF001
        }


__all__ = [
    "AUTHORITATIVE_EVENT_TYPES",
    "MERGEABLE_EVENT_TYPES",
    "LOCAL_EVENT_TYPES",
    "RoutedEvent",
    "WorkspaceDistributedConnector",
]
