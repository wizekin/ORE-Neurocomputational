"""ORE 21-stage Hub-and-Spoke semantic integration package."""

from .annotations import index_hub_spoke_annotations, record_hub_spoke_result
from .biology import POPULATION_NAMES, HubSpokeLIFSimulator
from .contracts import (
    MODALITIES,
    HubAndSpokeConfig,
    HubAndSpokeRequest,
    HubAndSpokeResult,
    LearningExperience,
)
from .pipeline import HubAndSpokeProcessor, request_from_semantic_request
from .point08 import (
    POINT08_STAGE_REGISTRY,
    BiologicalEligibilityPlasticity,
    Point08Config,
    Point08DistributedSemanticProcessor,
    Point08Result,
    Point08StageSpec,
    TemporalAssociationMemory,
    index_point08_annotations,
    record_point08_result,
)
from .registry import ALL_OPERATIONS, STAGE_REGISTRY, OperationSpec, StageSpec, registry_as_dicts

__all__ = [
    "ALL_OPERATIONS",
    "MODALITIES",
    "POPULATION_NAMES",
    "POINT08_STAGE_REGISTRY",
    "BiologicalEligibilityPlasticity",
    "HubAndSpokeConfig",
    "HubAndSpokeProcessor",
    "HubAndSpokeRequest",
    "HubAndSpokeResult",
    "HubSpokeLIFSimulator",
    "Point08Config",
    "Point08DistributedSemanticProcessor",
    "Point08Result",
    "Point08StageSpec",
    "LearningExperience",
    "OperationSpec",
    "STAGE_REGISTRY",
    "StageSpec",
    "TemporalAssociationMemory",
    "index_hub_spoke_annotations",
    "index_point08_annotations",
    "record_hub_spoke_result",
    "record_point08_result",
    "registry_as_dicts",
    "request_from_semantic_request",
]
