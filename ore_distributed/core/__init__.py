"""Phase A foundation: deployment topology, state classification, and threat models.

Implements Steps 1-4 of the Point 17 distributed coordination algorithm: deployment
mode detection, per-schema state classification, consistency requirements, and the
failure/threat model that selects the tolerated fault class.
"""
from ore_distributed.core.deployment import (
    DeploymentMode,
    TopologyDescriptor,
    detect_deployment_mode,
    deployment_allows_consensus,
)
from ore_distributed.core.state_class import (
    ConsistencyModel,
    FailureModel,
    StateClass,
    StateClassificationError,
    StateObjectSpec,
    StateSchemaRegistry,
    ThreatModel,
    recommended_protocol_family,
)

__all__ = [
    "DeploymentMode",
    "TopologyDescriptor",
    "detect_deployment_mode",
    "deployment_allows_consensus",
    "ConsistencyModel",
    "FailureModel",
    "StateClass",
    "StateClassificationError",
    "StateObjectSpec",
    "StateSchemaRegistry",
    "ThreatModel",
    "recommended_protocol_family",
]
