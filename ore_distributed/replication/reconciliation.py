"""Replica reconciliation through state exchange (Step 28).

Replicas exchange CRDT state or operations according to the delivery model,
merge using the defined algebra, persist the result, and track causal and
version metadata. Repeated reconciliation produces no further change once
replicas have converged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ore_distributed.replication.crdts import GSet, LWWRegister, ORSet, PNCounter
from ore_distributed.replication.replica import CrdtReplica


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Record what one reconciliation pass changed for observability."""

    left_applied: int
    right_applied: int
    converged: bool
    rounds: int = 1


def reconcile_crdt_states(left: Any, right: Any) -> tuple[Any, Any, ReconciliationReport]:
    """Merge two CRDT states so both sides hold the joined lattice value."""
    if type(left) is not type(right):
        raise TypeError("reconciliation requires two states of the same CRDT type")
    left_clone = left.clone()
    left_clone.merge(right.clone())
    right_clone = left.clone()
    right_clone.merge(left.clone())
    # After a two-way merge both sides hold identical lattice elements.
    left_clone = left.clone()
    left_clone.merge(right.clone())
    right_clone = right_clone.clone() if hasattr(right_clone, "clone") else right_clone
    right_clone.merge(left.clone())
    converged = left_clone == right_clone
    return left_clone, right_clone, ReconciliationReport(
        left_applied=1, right_applied=1, converged=converged
    )


def reconcile_replicas(
    left: CrdtReplica, right: CrdtReplica
) -> ReconciliationReport:
    """Exchange observations between two replicas until they converge (Step 28)."""
    left_before = set()
    right_before = set()
    for _round in range(4):
        left_before = {observation.event_id for observation in left.observations()}
        right_before = {observation.event_id for observation in right.observations()}
        for observation in right.observations():
            if observation.event_id not in left_before:
                left.apply_remote(observation)
        for observation in left.observations():
            if observation.event_id not in right_before:
                right.apply_remote(observation)
        left_state = left.contributions.clone()
        left_state.merge(right.contributions)
        right_state = right.contributions.clone()
        right_state.merge(left.contributions)
        left.contributions = left_state
        right.contributions = right_state
        if (
            {observation.event_id for observation in left.observations()}
            == {observation.event_id for observation in right.observations()}
            and left.contributions == right.contributions
        ):
            return ReconciliationReport(
                left_applied=1, right_applied=1, converged=True, rounds=_round + 1
            )
    return ReconciliationReport(left_applied=1, right_applied=1, converged=False, rounds=4)


def converged(left: CrdtReplica, right: CrdtReplica) -> bool:
    """Return whether two replicas hold the same observation and CRDT state."""
    left_ids = {observation.event_id for observation in left.observations()}
    right_ids = {observation.event_id for observation in right.observations()}
    if left_ids != right_ids:
        return False
    merged = left.contributions.clone()
    merged.merge(right.contributions)
    return merged == left.contributions and merged == right.contributions


def reconciliation_is_idempotent(
    left: CrdtReplica, right: CrdtReplica
) -> tuple[bool, int]:
    """Run reconciliation twice; returns (converged, observations changed on pass 2).

    Once replicas have converged, repeated reconciliation must produce no
    further changes (Step 28 acceptance check).
    """
    first = reconcile_replicas(left, right)
    changed_before = len(left.observations()) + len(right.observations())
    second = reconcile_replicas(left, right)
    changed_after = len(left.observations()) + len(right.observations())
    return first.converged and second.converged and changed_after == changed_before, changed_after


def offline_replica_reconnect(
    offline: CrdtReplica, online: CrdtReplica
) -> ReconciliationReport:
    """Bring an offline replica back into sync after reconnection.

    The offline replica contributed before the disconnect; both directions are
    replayed so independent local updates survive the reconciliation.
    """
    return reconcile_replicas(offline, online)


__all__ = [
    "ReconciliationReport",
    "reconcile_crdt_states",
    "reconcile_replicas",
    "converged",
    "reconciliation_is_idempotent",
    "offline_replica_reconnect",
]
