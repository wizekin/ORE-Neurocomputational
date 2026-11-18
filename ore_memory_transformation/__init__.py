"""ORE Point 04 episodic-to-semantic transformation and gist extraction."""
from .annotations import register_annotations
from .contracts import (
    EpisodeInput,
    MemoryTransformationConfig,
    MemoryTransformationRequest,
    MemoryTransformationResult,
    RelationEvidence,
)
from .pipeline import EpisodicToSemanticTransformationProcessor
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "EpisodeInput",
    "RelationEvidence",
    "MemoryTransformationConfig",
    "MemoryTransformationRequest",
    "MemoryTransformationResult",
    "EpisodicToSemanticTransformationProcessor",
    "OPERATION_REGISTRY",
    "OperationSpec",
    "registry_as_dicts",
    "register_annotations",
]
