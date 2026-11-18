"""ORE Point 05 novelty-modulated hippocampal memory consolidation."""
from .annotations import register_annotations
from .biology import (
    BiologicalConsolidationNetwork,
    HippocampalPlasticityModel,
    NeuromodulatoryPopulationModel,
)
from .contracts import (
    CompetingTrace,
    ConsolidationConfig,
    ConsolidationRequest,
    ConsolidationResult,
    InternalState,
    MemoryTrace,
    PostLearningEvent,
    SleepEvidence,
)
from .pipeline import NoveltyModulatedHippocampalConsolidationProcessor
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "MemoryTrace",
    "PostLearningEvent",
    "InternalState",
    "SleepEvidence",
    "CompetingTrace",
    "ConsolidationConfig",
    "ConsolidationRequest",
    "ConsolidationResult",
    "BiologicalConsolidationNetwork",
    "HippocampalPlasticityModel",
    "NeuromodulatoryPopulationModel",
    "NoveltyModulatedHippocampalConsolidationProcessor",
    "OPERATION_REGISTRY",
    "OperationSpec",
    "register_annotations",
    "registry_as_dicts",
]
