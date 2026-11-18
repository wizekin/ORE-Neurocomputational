"""ORE Point 02 gradual cortical learning from repeated experiences."""
from .annotations import register_annotations
from .contracts import CorticalLearningConfig, CorticalLearningResult, EncounterInput
from .pipeline import GradualCorticalLearningProcessor
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "CorticalLearningConfig",
    "CorticalLearningResult",
    "EncounterInput",
    "GradualCorticalLearningProcessor",
    "OPERATION_REGISTRY",
    "OperationSpec",
    "register_annotations",
    "registry_as_dicts",
]
