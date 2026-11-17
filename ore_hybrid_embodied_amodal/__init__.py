"""ORE hybrid embodied–amodal concept formation and processing."""

from .annotations import index_hybrid_annotations, record_hybrid_result
from .biology import MODALITY_POPULATIONS, POPULATION_NAMES, HybridLIFSimulator
from .contracts import (
    MODALITIES,
    HybridConcept,
    HybridConfig,
    HybridLearningExample,
    HybridRequest,
    HybridResult,
)
from .pipeline import HybridEmbodiedAmodalProcessor
from .registry import ALL_OPERATIONS, STAGE_REGISTRY, OperationSpec, StageSpec, get_stage, registry_as_dicts

__all__ = [
    "ALL_OPERATIONS",
    "MODALITIES",
    "MODALITY_POPULATIONS",
    "POPULATION_NAMES",
    "STAGE_REGISTRY",
    "HybridConcept",
    "HybridConfig",
    "HybridEmbodiedAmodalProcessor",
    "HybridLearningExample",
    "HybridLIFSimulator",
    "HybridRequest",
    "HybridResult",
    "OperationSpec",
    "StageSpec",
    "get_stage",
    "index_hybrid_annotations",
    "record_hybrid_result",
    "registry_as_dicts",
]
