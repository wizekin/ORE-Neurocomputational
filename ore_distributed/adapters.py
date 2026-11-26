"""Proposed package boundary and coordination interfaces (Step 45).

``ore_distributed`` keeps protocol implementations behind interfaces --
NodeRegistry, ConsensusStateStore, ReplicatedLog, CRDTReplica, GossipTransport,
EventClock, ProvenanceJournal, ShardRouter, and RecoveryManager -- so local
deployments can use in-memory adapters and single-process installations never
require a consensus cluster.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from ore_distributed.core.deployment import DeploymentMode, TopologyDescriptor
from ore_distributed.core.state_class import StateSchemaRegistry, default_state_schema_registry


@runtime_checkable
class NodeRegistryProtocol(Protocol):
    """The registry interface over membership state (Step 45)."""

    def register(self, record: Any) -> Any: ...

    def get(self, node_id: str) -> Any | None: ...

    def voters(self) -> tuple[Any, ...]: ...


@runtime_checkable
class ConsensusStateStoreProtocol(Protocol):
    """The authoritative state store interface (Step 45)."""

    def apply(self, command: Any, log_index: int) -> str: ...

    def snapshot(self) -> Mapping[str, Any]: ...


@runtime_checkable
class ReplicatedLogProtocol(Protocol):
    """The replicated log interface (Step 45)."""

    def append(self, term: int, command: Any) -> Any: ...

    def truncate_suffix(self, from_index: int) -> int: ...


@runtime_checkable
class CRDTReplicaProtocol(Protocol):
    """The mergeable replica interface (Step 45)."""

    def apply_local(self, content: str) -> Any: ...

    def apply_remote(self, observation: Any) -> str: ...


@runtime_checkable
class GossipTransportProtocol(Protocol):
    """The gossip transport interface (Step 45)."""

    def digest(self) -> Any: ...

    def exchange(self, peer_digest: Any, peer_entries: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class EventClockProtocol(Protocol):
    """The event clock interface (Step 45)."""

    def tick(self) -> Any: ...

    def merge(self, remote: Any) -> Any: ...


@runtime_checkable
class ProvenanceJournalProtocol(Protocol):
    """The provenance journal interface (Step 45)."""

    def accept(self, envelope: Any, payload: Mapping[str, object] | None) -> str: ...

    def audit_trail(self) -> tuple[Any, ...]: ...


@runtime_checkable
class ShardRouterProtocol(Protocol):
    """The shard router interface (Step 45)."""

    def route(self, data_class: Any, item_id: str) -> Any: ...


@runtime_checkable
class RecoveryManagerProtocol(Protocol):
    """The recovery manager interface (Step 45)."""

    def recover_node(self, node_id: str) -> Any: ...


class LocalDeployment:
    """Wire the coordination layer for a single-process deployment (Step 45).

    A single-process deployment keeps local coordination primitives and does
    not instantiate a consensus cluster merely because modules share a
    workspace; every interface below is an in-memory adapter.
    """

    def __init__(
        self,
        node_id: str = "local-node",
        cluster_id: str = "local-cluster",
    ) -> None:
        """Assemble the local in-memory adapter set."""
        if not node_id.strip() or not cluster_id.strip():
            raise ValueError("node_id and cluster_id must be non-empty")
        self.node_id = node_id
        self.cluster_id = cluster_id
        self.topology = TopologyDescriptor(
            node_id=node_id,
            cluster_id=cluster_id,
            mode=DeploymentMode.SINGLE_PROCESS,
        )
        self.state_schemas: StateSchemaRegistry = default_state_schema_registry()
        self._local_events: list[Mapping[str, Any]] = []
        self._local_clock_position = 0

    def describe(self) -> dict[str, Any]:
        """Return the deployment descriptor as a dictionary."""
        return self.topology.to_dict()

    def is_consensus_free(self) -> bool:
        """Return whether this deployment runs without a consensus cluster."""
        return self.topology.mode is DeploymentMode.SINGLE_PROCESS

    def publish_local_event(self, event: Mapping[str, Any]) -> int:
        """Record a local event in the in-memory adapter (never replicated)."""
        self._local_events.append(dict(event))
        self._local_clock_position += 1
        return self._local_clock_position

    def local_events(self) -> tuple[Mapping[str, Any], ...]:
        """Return the locally recorded events."""
        return tuple(self._local_events)


def verify_protocol_implementations() -> dict[str, bool]:
    """Check that every in-memory adapter satisfies its Step 45 interface."""
    from ore_distributed.membership.registry import NodeRegistry
    from ore_distributed.consensus.log import ReplicatedLog
    from ore_distributed.consensus.state_machine import AuthoritativeStateMachine
    from ore_distributed.replication.replica import CrdtReplica
    from ore_distributed.gossip.dissemination import GossipDissemination
    from ore_distributed.time.hlc import HybridLogicalClock
    from ore_distributed.provenance.journal import ProvenanceJournal
    from ore_distributed.placement.router import ShardRouter
    from ore_distributed.recovery.recovery import RecoveryManager

    checks = {
        "NodeRegistry": issubclass(NodeRegistry, NodeRegistryProtocol),
        "ConsensusStateStore": issubclass(
            AuthoritativeStateMachine, ConsensusStateStoreProtocol
        ),
        "ReplicatedLog": issubclass(ReplicatedLog, ReplicatedLogProtocol),
        "CRDTReplica": issubclass(CrdtReplica, CRDTReplicaProtocol),
        "GossipTransport": issubclass(GossipDissemination, GossipTransportProtocol),
        "EventClock": issubclass(HybridLogicalClock, EventClockProtocol),
        "ProvenanceJournal": issubclass(ProvenanceJournal, ProvenanceJournalProtocol),
        "ShardRouter": issubclass(ShardRouter, ShardRouterProtocol),
        "RecoveryManager": issubclass(RecoveryManager, RecoveryManagerProtocol),
    }
    return checks


__all__ = [
    "NodeRegistryProtocol",
    "ConsensusStateStoreProtocol",
    "ReplicatedLogProtocol",
    "CRDTReplicaProtocol",
    "GossipTransportProtocol",
    "EventClockProtocol",
    "ProvenanceJournalProtocol",
    "ShardRouterProtocol",
    "RecoveryManagerProtocol",
    "LocalDeployment",
    "verify_protocol_implementations",
]
