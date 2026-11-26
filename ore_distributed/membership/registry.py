"""Node registry and capability descriptors (Steps 7 and 9).

The registry distinguishes discovered, authenticated, eligible, voting, draining
and removed nodes. Capability descriptors route work but remain claims that may
require validation or attestation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping

from ore_distributed.core.deployment import TransportEndpoint


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def utc_now() -> str:
    """Return the current ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class NodeHealth(str, Enum):
    """Lifecycle states of a node from discovery to removal (Step 7)."""

    DISCOVERED = "discovered"
    AUTHENTICATED = "authenticated"
    ELIGIBLE = "eligible"
    VOTING = "voting"
    DRAINING = "draining"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Advertised capabilities of one node (Step 9).

    Claims include supported ORE modules, USG schema versions, hardware resources,
    and permitted workload classes; the registry stores them as claims that may
    require validation, never as verified facts.
    """

    node_id: str
    ore_modules: frozenset[str] = frozenset()
    usg_schema_versions: frozenset[str] = frozenset()
    hardware: Mapping[str, float] = field(default_factory=dict)
    workload_classes: frozenset[str] = frozenset()
    software_version: str = "0.1.0"
    attestation: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize capability claims."""
        object.__setattr__(self, "node_id", _non_empty("node_id", self.node_id))
        object.__setattr__(self, "ore_modules", frozenset(self.ore_modules))
        object.__setattr__(self, "usg_schema_versions", frozenset(self.usg_schema_versions))
        object.__setattr__(self, "workload_classes", frozenset(self.workload_classes))
        hardware = {str(k): float(v) for k, v in dict(self.hardware).items()}
        if any(not (v >= 0.0) or v != v for v in hardware.values()):
            raise ValueError("hardware claims must be finite non-negative numbers")
        object.__setattr__(self, "hardware", hardware)
        object.__setattr__(self, "software_version", _non_empty("software_version", self.software_version))
        if self.attestation is not None:
            object.__setattr__(self, "attestation", _non_empty("attestation", self.attestation))

    def to_dict(self) -> dict[str, object]:
        """Return the descriptor as a plain dictionary."""
        return {
            "node_id": self.node_id,
            "ore_modules": sorted(self.ore_modules),
            "usg_schema_versions": sorted(self.usg_schema_versions),
            "hardware": dict(self.hardware),
            "workload_classes": sorted(self.workload_classes),
            "software_version": self.software_version,
            "attestation": self.attestation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CapabilityDescriptor":
        """Rebuild a descriptor from its dictionary form."""
        return cls(
            node_id=str(payload["node_id"]),
            ore_modules=frozenset(str(m) for m in payload.get("ore_modules", ())),
            usg_schema_versions=frozenset(str(v) for v in payload.get("usg_schema_versions", ())),
            hardware=dict(payload.get("hardware", {})),
            workload_classes=frozenset(str(w) for w in payload.get("workload_classes", ())),
            software_version=str(payload.get("software_version", "0.1.0")),
            attestation=(str(payload["attestation"]) if payload.get("attestation") else None),
        )


@dataclass(slots=True)
class NodeRecord:
    """Registry entry combining identity, endpoints, health, and capabilities."""

    node_id: str
    endpoints: tuple[TransportEndpoint, ...] = ()
    capabilities: CapabilityDescriptor | None = None
    health: NodeHealth = NodeHealth.DISCOVERED
    last_heard: str = ""
    authorization: str = "pending"
    failure_domain: str = "default"

    def __post_init__(self) -> None:
        """Validate node identity and default fields."""
        self.node_id = _non_empty("node_id", self.node_id)
        self.endpoints = tuple(self.endpoints)
        self.last_heard = self.last_heard or utc_now()

    def is_voter(self) -> bool:
        """Return whether this node participates in the voting configuration."""
        return self.health is NodeHealth.VOTING

    def to_dict(self) -> dict[str, object]:
        """Return the record as a plain dictionary."""
        return {
            "node_id": self.node_id,
            "endpoints": [item.to_dict() for item in self.endpoints],
            "capabilities": self.capabilities.to_dict() if self.capabilities else None,
            "health": self.health.value,
            "last_heard": self.last_heard,
            "authorization": self.authorization,
            "failure_domain": self.failure_domain,
        }


class NodeRegistry:
    """Membership view distinguishing discovery from trust (Steps 7 and 12)."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._records: dict[str, NodeRecord] = {}

    def register(self, record: NodeRecord) -> NodeRecord:
        """Insert or refresh a record; discovery alone grants no voting rights."""
        if not isinstance(record, NodeRecord):
            raise TypeError("register expects a NodeRecord")
        existing = self._records.get(record.node_id)
        if existing is not None and existing.is_voter():
            # A re-registration attempt cannot silently demote a voter (Step 12).
            if record.health is not NodeHealth.VOTING:
                raise ValueError(
                    f"voter {record.node_id} may only change state through an "
                    "authorized membership change"
                )
        self._records[record.node_id] = record
        return record

    def get(self, node_id: str) -> NodeRecord | None:
        """Return one record or None."""
        return self._records.get(node_id)

    def require(self, node_id: str) -> NodeRecord:
        """Return one record or raise a membership error."""
        record = self._records.get(node_id)
        if record is None:
            raise KeyError(f"unknown node {node_id!r}")
        return record

    def remove(self, node_id: str) -> bool:
        """Mark a node removed from the registry view (not a config change)."""
        record = self._records.get(node_id)
        if record is None:
            return False
        record.health = NodeHealth.REMOVED
        return True

    def by_health(self, health: NodeHealth) -> tuple[NodeRecord, ...]:
        """Return all records in a given health state ordered by node id."""
        return tuple(
            self._records[node] for node in sorted(self._records)
            if self._records[node].health is health
        )

    def voters(self) -> tuple[NodeRecord, ...]:
        """Return the current voting members ordered by node id."""
        return self.by_health(NodeHealth.VOTING)

    def authenticated(self) -> tuple[NodeRecord, ...]:
        """Return authenticated or better records ordered by node id."""
        order = {
            NodeHealth.DISCOVERED: 0,
            NodeHealth.AUTHENTICATED: 1,
            NodeHealth.ELIGIBLE: 2,
            NodeHealth.VOTING: 3,
            NodeHealth.DRAINING: 1,
            NodeHealth.REMOVED: -1,
        }
        return tuple(
            self._records[node] for node in sorted(self._records)
            if order[self._records[node].health] >= 1
        )

    def touch(self, node_id: str) -> None:
        """Refresh last-heard time for heartbeats and gossip contacts."""
        record = self._records.get(node_id)
        if record is not None:
            record.last_heard = utc_now()

    def publish_capabilities(self, descriptor: CapabilityDescriptor) -> None:
        """Store a capability claim for a node, creating a discovered record (Step 9)."""
        record = self._records.get(descriptor.node_id)
        if record is None:
            record = NodeRecord(node_id=descriptor.node_id, health=NodeHealth.DISCOVERED)
            self._records[descriptor.node_id] = record
        record.capabilities = descriptor
        record.last_heard = utc_now()

    def eligible_for(self, workload_class: str) -> tuple[NodeRecord, ...]:
        """Return authenticated nodes advertising a workload class (routing input)."""
        workload_class = _non_empty("workload_class", workload_class)
        return tuple(
            record for record in self.authenticated()
            if record.capabilities is not None
            and workload_class in record.capabilities.workload_classes
        )

    def node_ids(self) -> tuple[str, ...]:
        """Return all known node identifiers ordered alphabetically."""
        return tuple(sorted(self._records))

    def size(self) -> int:
        """Return the number of records including removed ones."""
        return len(self._records)


def records_from_specs(
    specs: Iterable[tuple[str, NodeHealth, Iterable[str]]],
) -> tuple[NodeRecord, ...]:
    """Build registry records from (node_id, health, roles) tuples for tests."""
    return tuple(NodeRecord(node_id=node, health=state) for node, state, _ in specs)


__all__ = [
    "NodeHealth",
    "CapabilityDescriptor",
    "NodeRecord",
    "NodeRegistry",
    "utc_now",
    "records_from_specs",
]
