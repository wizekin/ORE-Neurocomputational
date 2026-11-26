"""Distributed coordination pipeline (Phases A-G as a runtime assembly).

The pipeline classifies deployment topology, builds the state classification,
and wires the coordination subsystems a deployment actually needs: a
single-process node gets in-memory adapters and never instantiates a consensus
cluster, while multi-process and cluster deployments activate the full
authenticated transport, consensus, CRDT, gossip, provenance, placement,
execution, recovery, and integration stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ore_distributed.adapters import LocalDeployment, verify_protocol_implementations
from ore_distributed.core.deployment import (
    DeploymentMode,
    TopologyDescriptor,
    deployment_allows_consensus,
)
from ore_distributed.core.state_class import (
    StateSchemaRegistry,
    default_state_schema_registry,
)
from ore_distributed.registry import (
    ORE_DISTRIBUTED_CONTRACT,
    ORE_DISTRIBUTED_VERSION,
    OPERATION_REGISTRY,
    operation_count,
    phases,
)


@dataclass(slots=True)
class CoordinationCapabilities:
    """Which coordination subsystems a deployment instantiated."""

    topology: DeploymentMode
    consensus_active: bool
    crdt_replication_active: bool
    gossip_active: bool
    provenance_active: bool
    placement_active: bool
    execution_active: bool
    recovery_active: bool

    def to_dict(self) -> dict[str, object]:
        """Return the capability set as a plain dictionary."""
        return {
            "topology": self.topology.value,
            "consensus_active": self.consensus_active,
            "crdt_replication_active": self.crdt_replication_active,
            "gossip_active": self.gossip_active,
            "provenance_active": self.provenance_active,
            "placement_active": self.placement_active,
            "execution_active": self.execution_active,
            "recovery_active": self.recovery_active,
        }


class DistributedCoordinationPipeline:
    """Assemble the Point 17 coordination layer for one deployment."""

    def __init__(self, topology: TopologyDescriptor) -> None:
        """Classify the deployment and prepare the assembly plan."""
        if not isinstance(topology, TopologyDescriptor):
            raise TypeError("pipeline expects a TopologyDescriptor")
        self.topology = topology
        self.state_schemas: StateSchemaRegistry = default_state_schema_registry()
        self._local: LocalDeployment | None = None
        self.capabilities = self._plan_capabilities()

    def _plan_capabilities(self) -> CoordinationCapabilities:
        """Decide which subsystems the deployment mode may activate (Step 1)."""
        multi = deployment_allows_consensus(self.topology.mode)
        cluster = self.topology.mode is DeploymentMode.MULTI_MACHINE_CLUSTER
        return CoordinationCapabilities(
            topology=self.topology.mode,
            consensus_active=multi,
            crdt_replication_active=multi,
            gossip_active=multi,
            provenance_active=True,
            placement_active=cluster,
            execution_active=True,
            recovery_active=multi,
        )

    def classify_schema(self, schema_id: str) -> Mapping[str, Any]:
        """Return the classification of one state schema (Step 2)."""
        spec = self.state_schemas.require(schema_id)
        return spec.to_dict()

    def build_local_deployment(self) -> LocalDeployment:
        """Build the single-process adapter set (no consensus cluster)."""
        if self.topology.mode is not DeploymentMode.SINGLE_PROCESS:
            raise RuntimeError(
                "local adapters are for single-process deployments; the topology "
                f"is {self.topology.mode.value}"
            )
        self._local = LocalDeployment(
            node_id=self.topology.node_id,
            cluster_id=self.topology.cluster_id,
        )
        return self._local

    def local_deployment(self) -> LocalDeployment | None:
        """Return the local deployment if one was built."""
        return self._local

    def describe(self) -> dict[str, Any]:
        """Return the full assembly description of this deployment."""
        return {
            "contract": ORE_DISTRIBUTED_CONTRACT,
            "version": ORE_DISTRIBUTED_VERSION,
            "topology": self.topology.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "state_classifications": self.state_schemas.as_dicts(),
            "operation_count": operation_count(),
            "phases": list(phases()),
        }

    @staticmethod
    def registry_summary() -> dict[str, Any]:
        """Return the 48-operation registry summary."""
        return {
            "operation_count": operation_count(),
            "phases": list(phases()),
            "operations": [
                {
                    "operation_id": spec.operation_id,
                    "phase": spec.phase,
                    "step": spec.source_step,
                    "name": spec.name,
                }
                for spec in OPERATION_REGISTRY
            ],
        }

    @staticmethod
    def interface_checks() -> dict[str, bool]:
        """Return the Step 45 interface conformance results."""
        return verify_protocol_implementations()


def build_pipeline(
    node_id: str,
    cluster_id: str,
    process_count: int = 1,
    machine_count: int | None = 1,
) -> DistributedCoordinationPipeline:
    """Detect the deployment mode and build its coordination pipeline (Step 1)."""
    from ore_distributed.core.deployment import detect_deployment_mode

    mode = detect_deployment_mode(
        process_count=process_count,
        machine_count=machine_count,
        cluster_id=cluster_id,
    )
    topology = TopologyDescriptor(
        node_id=node_id,
        cluster_id=cluster_id,
        mode=mode,
    )
    return DistributedCoordinationPipeline(topology)


__all__ = [
    "CoordinationCapabilities",
    "DistributedCoordinationPipeline",
    "build_pipeline",
]
