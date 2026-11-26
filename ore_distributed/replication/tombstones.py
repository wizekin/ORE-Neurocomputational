"""Tombstone bounding and causal-stability garbage collection (Step 29).

Deletion metadata is removed only after safe causal-stability conditions hold:
an offline replica that has not seen a deletion must never be able to
resurrect a deleted value from an old tombstone it never observed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ore_distributed.replication.crdts import ORSet


@dataclass(frozen=True, slots=True)
class TombstoneSafetyReport:
    """Explain one tombstone-collection decision for audit purposes."""

    tag: str
    stable: bool
    replicas_observing: int
    replicas_known: int


class TombstoneCollector:
    """Collect OR-set tombstones only when causally stable (Step 29)."""

    def __init__(self, known_replicas: Iterable[str]) -> None:
        """Track deletion observations across a fixed set of replica identifiers."""
        self.known_replicas = frozenset(
            item.strip() for item in known_replicas if item.strip()
        )
        if not self.known_replicas:
            raise ValueError("at least one known replica is required")
        self._observed: dict[str, set[str]] = {}

    def record_observation(self, tag: str, replica_id: str) -> None:
        """Record that one replica has observed one deletion tombstone."""
        if replica_id not in self.known_replicas:
            raise ValueError(f"unknown replica {replica_id!r}")
        if not tag.strip():
            raise ValueError("tag must be non-empty")
        self._observed.setdefault(tag, set()).add(replica_id)

    def record_delivery(self, tag: str, recipients: Iterable[str]) -> None:
        """Record that a tombstone reached specific replicas."""
        for recipient in recipients:
            self.record_observation(tag, recipient)

    def observations_for(self, tag: str) -> frozenset[str]:
        """Return the replicas known to have observed one tombstone."""
        return frozenset(self._observed.get(tag, ()))

    def is_causally_stable(self, tag: str) -> bool:
        """Return whether every known replica has observed the tombstone."""
        return self.known_replicas <= self.observations_for(tag)

    def safety_report(self, tag: str) -> TombstoneSafetyReport:
        """Return the stability explanation for one tombstone."""
        observing = self.observations_for(tag)
        return TombstoneSafetyReport(
            tag=tag,
            stable=self.known_replicas <= observing,
            replicas_observing=len(observing),
            replicas_known=len(self.known_replicas),
        )

    def safe_to_collect(self, tags: Iterable[str]) -> tuple[str, ...]:
        """Return the subset of tags collectable without resurrection risk."""
        return tuple(
            tag
            for tag in tags
            if tag.strip() and self.known_replicas <= self.observations_for(tag)
        )

    def collect(self, target: ORSet, tags: Iterable[str] | None = None) -> int:
        """Collect stable tombstones from an OR-set; returns the number dropped."""
        if not isinstance(target, ORSet):
            raise TypeError("collect expects an ORSet")
        candidates = tuple(self._observed.keys()) if tags is None else tuple(tags)
        stable = set(self.safe_to_collect(candidates))
        if not stable:
            return 0
        return target.collect_tombstones(stable)

    def resurrected_risk(self, tag: str, replica_id: str) -> bool:
        """Return whether collecting a tag would risk resurrection for a replica."""
        return replica_id in self.known_replicas and replica_id not in self.observations_for(tag)

    def stale_tags(self) -> tuple[str, ...]:
        """Return every tracked tombstone tag ordered alphabetically."""
        return tuple(sorted(self._observed))


def deletion_survives_offline_replica(
    collector: TombstoneCollector,
    offline_replica: str,
    tags: Iterable[str],
) -> tuple[bool, str]:
    """Check the Step 29 invariant: no resurrection from an old offline replica.

    A tombstone is only safe to collect when the offline replica has observed
    the deletion; otherwise collection stays blocked so the replica cannot
    resurrect the deleted value from stale state.
    """
    for tag in tags:
        if collector.resurrected_risk(tag, offline_replica):
            return False, f"replica {offline_replica!r} has not observed deletion {tag!r}"
    return True, "all replicas observed the deletions; collection is safe"


__all__ = [
    "TombstoneSafetyReport",
    "TombstoneCollector",
    "deletion_survives_offline_replica",
]
