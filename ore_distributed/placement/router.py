"""Semantic and operational shard routing by data class (Step 39).

Distinct partition keys are used for episodic records, semantic indexes, task
queues, and artifacts; authoritative control metadata stays in the consensus
state machine; USG queries can resolve shard ownership and retrieve
provenance alongside data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from ore_distributed.placement.consistent_hashing import ConsistentHashRing, RingPlacement


class DataClass(str, Enum):
    """The placement classes of distributed ORE data (Step 39)."""

    EPISODIC_RECORD = "episodic_record"
    SEMANTIC_INDEX = "semantic_index"
    TASK_QUEUE = "task_queue"
    ARTIFACT = "artifact"


#: Data classes routed through the hash ring; control metadata is not.
_RING_CLASSES = frozenset(
    {DataClass.EPISODIC_RECORD, DataClass.SEMANTIC_INDEX, DataClass.TASK_QUEUE, DataClass.ARTIFACT}
)

#: Prefix namespaces keep classes out of each other's key space.
_CLASS_NAMESPACES: Mapping[DataClass, str] = {
    DataClass.EPISODIC_RECORD: "episodic",
    DataClass.SEMANTIC_INDEX: "semantic",
    DataClass.TASK_QUEUE: "queue",
    DataClass.ARTIFACT: "artifact",
}


@dataclass(frozen=True, slots=True)
class ShardPlacement:
    """One routed data item with its owners and provenance reference."""

    data_class: DataClass
    item_id: str
    partition_key: str
    owners: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the placement as a plain dictionary."""
        return {
            "data_class": self.data_class.value,
            "item_id": self.item_id,
            "partition_key": self.partition_key,
            "owners": list(self.owners),
        }


class ShardRouter:
    """Route data by semantic and operational ownership (Step 39)."""

    def __init__(
        self,
        ring: ConsistentHashRing | None = None,
        *,
        control_plane: str = "consensus_state_machine",
    ) -> None:
        """Create a router over one ring with an explicit control plane."""
        self.ring = ring or ConsistentHashRing()
        self.control_plane = control_plane

    def partition_key_for(self, data_class: DataClass, item_id: str) -> str:
        """Compose the namespaced partition key of one item."""
        if not isinstance(data_class, DataClass):
            raise TypeError("data_class must be a DataClass value")
        if not item_id.strip():
            raise ValueError("item_id must be non-empty")
        namespace = _CLASS_NAMESPACES[data_class]
        return f"{namespace}:{item_id.strip()}"

    def route(
        self,
        data_class: DataClass,
        item_id: str,
        *,
        provenance_reference: str = "",
    ) -> ShardPlacement:
        """Return the shard placement of one item (Step 39)."""
        if data_class not in _RING_CLASSES:
            raise ValueError(
                f"data class {data_class.value!r} does not use hash-ring placement; "
                "authoritative control metadata belongs to the consensus state machine"
            )
        partition_key = self.partition_key_for(data_class, item_id)
        placement: RingPlacement = self.ring.owners_of(partition_key)
        return ShardPlacement(
            data_class=data_class,
            item_id=item_id.strip(),
            partition_key=partition_key,
            owners=placement.owners,
        )

    def route_control(self, item_id: str) -> str:
        """Return the authoritative store for control metadata (Step 39)."""
        if not item_id.strip():
            raise ValueError("item_id must be non-empty")
        return self.control_plane

    def resolve_usg_query(
        self,
        semantic_item_ids: Sequence[str],
    ) -> tuple[ShardPlacement, ...]:
        """Resolve shard ownership for a USG query (Step 39)."""
        return tuple(
            self.route(DataClass.SEMANTIC_INDEX, item_id)
            for item_id in semantic_item_ids
        )

    def retrieve_with_provenance(
        self,
        placement: ShardPlacement,
        provenance_store: Mapping[str, Any],
    ) -> tuple[dict[str, Any], Any]:
        """Return data and its provenance reference for one placement."""
        provenance_key = placement.partition_key + ":provenance"
        return (
            placement.to_dict(),
            provenance_store.get(provenance_key, None),
        )


def distinct_key_spaces(router: ShardRouter, item_id: str) -> tuple[str, ...]:
    """Return the four partition keys of one item across data classes.

    Different data classes must never share a partition key, keeping task
    queues, episodic records, semantic indexes, and artifacts separable.
    """
    return tuple(
        router.partition_key_for(data_class, item_id)
        for data_class in (
            DataClass.EPISODIC_RECORD,
            DataClass.SEMANTIC_INDEX,
            DataClass.TASK_QUEUE,
            DataClass.ARTIFACT,
        )
    )


__all__ = [
    "DataClass",
    "ShardPlacement",
    "ShardRouter",
    "distinct_key_spaces",
]
