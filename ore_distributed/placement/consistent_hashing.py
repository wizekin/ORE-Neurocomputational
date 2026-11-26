"""Consistent hashing with virtual nodes and incremental rebalancing (Step 38).

Keys and node identifiers map onto a hash ring; virtual nodes reduce placement
imbalance, the replication factor and failure-domain constraints are explicit,
and node join or leave moves only the expected subset of keys instead of
migrating the entire knowledge base.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping


def _ring_hash(value: str) -> int:
    """Return the deterministic 64-bit ring position of an identifier."""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class RingNode:
    """One virtual node position on the ring."""

    position: int
    node_id: str
    vnode: int


@dataclass(frozen=True, slots=True)
class RingPlacement:
    """The owners of one key under the current ring."""

    key: str
    owners: tuple[str, ...]
    positions: tuple[int, ...]


class ConsistentHashRing:
    """A consistent-hash ring with virtual nodes and explicit replication."""

    def __init__(
        self,
        nodes: Iterable[str] | None = None,
        *,
        virtual_nodes: int = 64,
        replication_factor: int = 3,
        failure_domains: Mapping[str, str] | None = None,
    ) -> None:
        """Create a ring with per-node failure-domain constraints."""
        if virtual_nodes < 1:
            raise ValueError("virtual_nodes must be positive")
        if replication_factor < 1:
            raise ValueError("replication_factor must be positive")
        self.virtual_nodes = int(virtual_nodes)
        self.replication_factor = int(replication_factor)
        self._failure_domains: dict[str, str] = dict(failure_domains or {})
        self._ring: list[RingNode] = []
        self._members: set[str] = set()
        for node in nodes or ():
            self.add_node(node)

    # ------------------------------------------------------------------ membership

    def add_node(self, node_id: str) -> None:
        """Add one node with its virtual nodes; returns nothing."""
        if not node_id.strip():
            raise ValueError("node_id must be non-empty")
        if node_id in self._members:
            return
        self._members.add(node_id)
        for vnode in range(self.virtual_nodes):
            self._ring.append(
                RingNode(
                    position=_ring_hash(f"{node_id}#v{vnode}"),
                    node_id=node_id,
                    vnode=vnode,
                )
            )
        self._ring.sort(key=lambda item: (item.position, item.node_id))

    def remove_node(self, node_id: str) -> bool:
        """Remove one node's virtual nodes; returns whether it was present."""
        if node_id not in self._members:
            return False
        self._members.discard(node_id)
        self._ring = [item for item in self._ring if item.node_id != node_id]
        return True

    def members(self) -> tuple[str, ...]:
        """Return the current physical nodes sorted by identifier."""
        return tuple(sorted(self._members))

    def __len__(self) -> int:
        """Return the number of physical nodes on the ring."""
        return len(self._members)

    # ------------------------------------------------------------------ placement

    def owners_of(self, key: str) -> RingPlacement:
        """Return the ordered owners of a key under domain constraints (Step 38)."""
        if not self._ring:
            raise ValueError("the ring has no nodes")
        if not key.strip():
            raise ValueError("key must be non-empty")
        position = _ring_hash(key)
        owners: list[str] = []
        positions: list[int] = []
        domains_seen: set[str] = set()
        wrapped = self._ring[self._binary_search(position):] + self._ring
        for ring_node in wrapped:
            if ring_node.node_id in owners:
                continue
            domain = self._failure_domains.get(ring_node.node_id, "default")
            if domain in domains_seen and len(
                {n for n in owners if self._failure_domains.get(n, "default") == domain}
            ) >= 1 and len(self._members) > len(owners) + 1:
                # Prefer spreading replicas across failure domains when possible.
                continue
            owners.append(ring_node.node_id)
            positions.append(ring_node.position)
            domains_seen.add(domain)
            if len(owners) >= self.replication_factor:
                break
        # Fall back to plain next-N when domain spread is impossible.
        if len(owners) < self.replication_factor:
            owners = []
            positions = []
            for ring_node in wrapped:
                if ring_node.node_id in owners:
                    continue
                owners.append(ring_node.node_id)
                positions.append(ring_node.position)
                if len(owners) >= self.replication_factor:
                    break
        return RingPlacement(key=key, owners=tuple(owners), positions=tuple(positions))

    def _binary_search(self, position: int) -> int:
        """Return the first ring index at or after a position."""
        low, high = 0, len(self._ring)
        while low < high:
            middle = (low + high) // 2
            if self._ring[middle].position < position:
                low = middle + 1
            else:
                high = middle
        return low

    def placement_balance(self) -> dict[str, int]:
        """Return vnode counts per physical node for balance checks."""
        counts: dict[str, int] = {node: 0 for node in self._members}
        for ring_node in self._ring:
            counts[ring_node.node_id] += 1
        return counts

    def keys_that_move(
        self, previous: Mapping[str, tuple[str, ...]], key: str
    ) -> bool:
        """Return whether a key's owners changed after a membership change."""
        return tuple(previous.get(key, ())) != self.owners_of(key).owners


def rebalance_plan(
    ring: ConsistentHashRing,
    previous_owners: Mapping[str, tuple[str, ...]],
    keys: Iterable[str],
) -> tuple[str, ...]:
    """Return only the keys whose placement changed during churn (Step 38).

    Incremental rebalancing moves the expected subset of keys; the caller
    compares this set against the full key space to prove no wholesale
    migration happened.
    """
    return tuple(
        key
        for key in keys
        if tuple(previous_owners.get(key, ())) != ring.owners_of(key).owners
    )


def shard_balance(owners_per_key: Mapping[str, tuple[str, ...]]) -> dict[str, int]:
    """Return primary-plus-replica load per node for balance observability."""
    counts: dict[str, int] = {}
    for owners in owners_per_key.values():
        for owner in owners:
            counts[owner] = counts.get(owner, 0) + 1
    return counts


__all__ = [
    "RingNode",
    "RingPlacement",
    "ConsistentHashRing",
    "rebalance_plan",
    "shard_balance",
]
