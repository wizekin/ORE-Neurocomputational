"""Explicit network-partition handling policy (Step 44).

During a partition, the consensus majority may continue authoritative
operations; minority nodes must not independently commit conflicting
authoritative state; merge-safe CRDT updates may be accepted locally with
visibility marked provisional until reconciliation; degraded-mode status is
exposed to the cognitive coordinator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ore_distributed.consensus.log import quorum_size
from ore_distributed.consensus.raft import RaftCluster, RaftRole


@dataclass(frozen=True, slots=True)
class PartitionPolicy:
    """The documented availability policy during partitions (Step 44)."""

    allow_authoritative_majority: bool = True
    block_authoritative_minority: bool = True
    allow_provisional_crdt_local: bool = True
    mark_crdt_provisional_until_reconciled: bool = True

    def __post_init__(self) -> None:
        """Keep the safety-critical settings fixed on."""
        if not (self.allow_authoritative_majority and self.block_authoritative_minority):
            raise ValueError(
                "the majority-continues/minority-blocked policy is a safety invariant"
            )


class PartitionTracker:
    """Track one active partition and its per-side consequences (Step 44)."""

    def __init__(self, policy: PartitionPolicy | None = None) -> None:
        """Create a tracker with the standing policy."""
        self.policy = policy or PartitionPolicy()
        self._active: list[tuple[frozenset[str], frozenset[str]]] = []
        self.events: list[str] = []

    def start_partition(self, group_a: Iterable[str], group_b: Iterable[str]) -> None:
        """Record a partition between two node groups."""
        pair = (frozenset(group_a), frozenset(group_b))
        self._active.append(pair)
        self.events.append(
            f"partition started: {sorted(pair[0])} | {sorted(pair[1])}"
        )

    def end_partition(self) -> None:
        """Heal the currently tracked partitions."""
        if self._active:
            self.events.append("partition healed")
        self._active.clear()

    def is_minority(self, group: Sequence[str], all_nodes: Sequence[str]) -> bool:
        """Return whether a group is a minority of the full node set."""
        total = len(set(all_nodes))
        return len(set(group)) < quorum_size(total)

    def authoritative_side(self, cluster: RaftCluster) -> str | None:
        """Return the side that holds the current leader, if any."""
        leader = cluster.leader()
        return leader.node_id if leader else None

    def minority_may_commit_authoritative(self, cluster: RaftCluster, node_id: str) -> bool:
        """Return whether a node may commit authoritative state right now.

        Minority nodes must not independently commit conflicting authoritative
        state; only the leader's side (or a majority quorum) may.
        """
        node = cluster.nodes.get(node_id)
        if node is None:
            return False
        leader = cluster.leader()
        if leader is None:
            # No leader at all: nobody may commit authoritative state.
            return False
        # A partitioned minority can never be the recognized leader's side.
        return node.node_id == leader.node_id and node.role is RaftRole.LEADER

    def crdt_updates_during_partition(
        self,
        provisional: bool,
    ) -> dict[str, object]:
        """Describe the CRDT write policy while partitioned (Step 44)."""
        return {
            "accept_locally": self.policy.allow_provisional_crdt_local,
            "visibility": (
                "provisional_until_reconciled"
                if self.policy.mark_crdt_provisional_until_reconciled
                else "visible"
            ),
            "provisional": provisional,
        }

    def degraded_status(self, cluster: RaftCluster) -> dict[str, object]:
        """Expose degraded-mode status to the cognitive coordinator (Step 44, 48)."""
        leader = cluster.leader()
        quorum = quorum_size(len(cluster.node_ids))
        available_voters = len(cluster.node_ids)
        return {
            "mode": "normal" if leader is not None else "degraded",
            "leader": leader.node_id if leader else None,
            "quorum_required": quorum,
            "voters_available": available_voters,
            "authoritative_writes": leader is not None,
            "safe_read_only_fallback": leader is None,
        }


def majority_continues_policy() -> PartitionPolicy:
    """Return the documented availability policy for acceptance checks."""
    return PartitionPolicy()


__all__ = [
    "PartitionPolicy",
    "PartitionTracker",
    "majority_continues_policy",
]
