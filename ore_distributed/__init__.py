"""ORE Point 17: distributed systems, consensus, and coordination algorithms.

A distributed coordination layer for ORE-Neurocomputational with USG semantic
knowledge and the CROS resource-management layer: Raft for authoritative
replicated operational state, selective CRDTs for mergeable collaborative
records, Hybrid Logical Clocks with explicit causal metadata for event
histories, gossip for scalable dissemination and failure suspicion, and
consistent hashing when distributed placement is justified.
"""
from ore_distributed.registry import (
    ORE_DISTRIBUTED_CONTRACT,
    ORE_DISTRIBUTED_VERSION,
    OPERATION_REGISTRY,
    OperationSpec,
    operation_count,
    phases,
    registry_as_dicts,
)
from ore_distributed.pipeline import (
    CoordinationCapabilities,
    DistributedCoordinationPipeline,
    build_pipeline,
)
from ore_distributed.adapters import (
    LocalDeployment,
    verify_protocol_implementations,
)

__version__ = ORE_DISTRIBUTED_VERSION

__all__ = [
    "ORE_DISTRIBUTED_CONTRACT",
    "ORE_DISTRIBUTED_VERSION",
    "OPERATION_REGISTRY",
    "OperationSpec",
    "operation_count",
    "phases",
    "registry_as_dicts",
    "CoordinationCapabilities",
    "DistributedCoordinationPipeline",
    "build_pipeline",
    "LocalDeployment",
    "verify_protocol_implementations",
    "__version__",
]
