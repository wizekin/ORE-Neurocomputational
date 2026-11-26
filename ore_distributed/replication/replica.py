"""CRDT replica behavior: local acceptance, dedup, and dissemination (Steps 25, 27).

Local acceptance means "recorded locally," never "globally visible" or
"semantically verified" — those statuses are exposed separately. Every
contribution carries a stable event ID and origin: identical text from two
agents is two independent observations, while repeated delivery of one event
is deduplicated by ID, never by text hash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import Any, Iterable, Mapping

from ore_distributed.replication.crdts import ORSet
from ore_distributed.time.hlc import HLCTimestamp, HybridLogicalClock


class ObservationStatus(str, Enum):
    """Separate replication statuses from semantic verification statuses (Step 27)."""

    RECORDED_LOCALLY = "recorded_locally"
    QUEUED_FOR_DISSEMINATION = "queued_for_dissemination"
    DISSEMINATED = "disseminated"
    RECONCILED = "reconciled"


@dataclass(frozen=True, slots=True)
class DistributedObservation:
    """One independently authored observation with stable identity (Step 25)."""

    event_id: str
    origin: str
    content: str
    hlc: HLCTimestamp
    kind: str = "observation"
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate observation identity and origin."""
        for name in ("event_id", "origin", "content", "kind"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.hlc, HLCTimestamp):
            raise ValueError("hlc must be an HLCTimestamp")
        object.__setattr__(self, "context", dict(self.context))

    @property
    def content_digest_collision_risk(self) -> bool:
        """Duplicate content from another origin is still distinct (Step 25)."""
        return False

    def to_dict(self) -> dict[str, Any]:
        """Return the observation as a plain dictionary."""
        return {
            "event_id": self.event_id,
            "origin": self.origin,
            "content": self.content,
            "hlc": self.hlc.to_dict(),
            "kind": self.kind,
            "context": dict(self.context),
        }


class CrdtReplica:
    """One node's replica of a mergeable collaborative record.

    The replica applies valid local updates immediately, assigns unique event
    identities, queues dissemination, and deduplicates repeated delivery by
    event ID. Dissemination status is tracked separately from semantic
    verification, which stays outside the CRDT layer.
    """

    def __init__(
        self,
        replica_id: str,
        clock: HybridLogicalClock | None = None,
    ) -> None:
        """Create an empty replica with its own clock."""
        if not replica_id.strip():
            raise ValueError("replica_id must be non-empty")
        self.replica_id = replica_id
        self.clock = clock or HybridLogicalClock(replica_id)
        self.contributions = ORSet(node_id=replica_id)
        self._observations: dict[str, DistributedObservation] = {}
        self._statuses: dict[str, ObservationStatus] = {}
        self._seen_event_ids: set[str] = set()
        self.dissemination_queue: list[str] = []
        self.duplicates_dropped = 0
        self.applied_local = 0
        self.applied_remote = 0

    def new_event_id(self) -> str:
        """Return a fresh, stable event identifier."""
        return f"obs_{uuid.uuid4().hex}"

    def apply_local(
        self,
        content: str,
        *,
        kind: str = "observation",
        context: Mapping[str, Any] | None = None,
        event_id: str | None = None,
    ) -> DistributedObservation:
        """Accept a local update: recorded locally, queued for dissemination (Step 27)."""
        event_id = event_id or self.new_event_id()
        observation = DistributedObservation(
            event_id=event_id,
            origin=self.replica_id,
            content=content,
            hlc=self.clock.tick(),
            kind=kind,
            context=dict(context or {}),
        )
        self._observations[event_id] = observation
        self.contributions.add(event_id)
        self._statuses[event_id] = ObservationStatus.RECORDED_LOCALLY
        self._seen_event_ids.add(event_id)
        self.dissemination_queue.append(event_id)
        self.applied_local += 1
        return observation

    def apply_remote(self, observation: DistributedObservation) -> str:
        """Absorb a remote observation; returns 'applied' or 'duplicate'."""
        if not isinstance(observation, DistributedObservation):
            raise TypeError("apply_remote expects a DistributedObservation")
        if observation.event_id in self._seen_event_ids:
            # Repeated delivery of one event is deduplicated by ID (Step 25).
            self.duplicates_dropped += 1
            return "duplicate"
        self.clock.merge(observation.hlc)
        self._observations[observation.event_id] = observation
        self.contributions.add(observation.event_id)
        self._statuses[observation.event_id] = ObservationStatus.RECONCILED
        self._seen_event_ids.add(observation.event_id)
        self.applied_remote += 1
        return "applied"

    def observation(self, event_id: str) -> DistributedObservation | None:
        """Return one observation by event identifier."""
        return self._observations.get(event_id)

    def status_of(self, event_id: str) -> ObservationStatus | None:
        """Return the dissemination status of one observation."""
        return self._statuses.get(event_id)

    def mark_disseminated(self, event_ids: Iterable[str]) -> int:
        """Mark observations as disseminated after gossip delivery."""
        marked = 0
        for event_id in event_ids:
            if event_id in self._statuses:
                self._statuses[event_id] = ObservationStatus.DISSEMINATED
                if event_id in self.dissemination_queue:
                    self.dissemination_queue.remove(event_id)
                marked += 1
        return marked

    def pending_dissemination(self) -> tuple[DistributedObservation, ...]:
        """Return observations still awaiting dissemination."""
        return tuple(
            self._observations[event_id]
            for event_id in self.dissemination_queue
            if event_id in self._observations
        )

    def observations(self) -> tuple[DistributedObservation, ...]:
        """Return every held observation ordered by event identifier."""
        return tuple(self._observations[key] for key in sorted(self._observations))

    def observations_by_content_hash(self) -> dict[str, list[DistributedObservation]]:
        """Group observations by content hash to expose duplicate-content cases.

        Identical text from two agents remains two independent observations
        with distinct event IDs; this grouping exists so callers can inspect
        that property rather than collapse records (Step 25).
        """
        import hashlib

        groups: dict[str, list[DistributedObservation]] = {}
        for observation in self.observations():
            digest = hashlib.sha256(observation.content.encode("utf-8")).hexdigest()
            groups.setdefault(digest, []).append(observation)
        return groups

    def replica_state(self) -> dict[str, Any]:
        """Return the CRDT state summary exchanged during reconciliation."""
        return {
            "replica_id": self.replica_id,
            "contributions": self.contributions.state(),
            "observation_count": len(self._observations),
            "seen_event_ids": sorted(self._seen_event_ids),
        }


def identical_text_two_origins(
    replica: CrdtReplica, other: CrdtReplica, text: str
) -> tuple[str, str]:
    """Record the same text on two replicas and return both event IDs (Step 25 check)."""
    first = replica.apply_local(text)
    second = other.apply_local(text)
    assert first.event_id != second.event_id
    return first.event_id, second.event_id


__all__ = [
    "ObservationStatus",
    "DistributedObservation",
    "CrdtReplica",
    "identical_text_two_origins",
]
