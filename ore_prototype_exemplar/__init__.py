"""ORE Point 07 prototype-based and exemplar-based concept representations."""
from .annotations import register_annotations
from .biology import (
    BiologicalPrototypeExemplarSystem,
    DistributedCorticalPrototypeNetwork,
    HippocampalEpisodicNetwork,
    NeuromodulatoryFeedbackGate,
    PrefrontalParietalControlNetwork,
    SensoryAssociationNetwork,
)
from .contracts import (
    CategoryState,
    EpisodeInput,
    ExemplarTrace,
    LearningRequest,
    PrototypeExemplarConfig,
    PrototypeExemplarResult,
    PrototypeState,
)
from .pipeline import PrototypeExemplarProcessor
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "EpisodeInput", "PrototypeExemplarConfig", "LearningRequest", "PrototypeState",
    "ExemplarTrace", "CategoryState", "PrototypeExemplarResult",
    "SensoryAssociationNetwork", "HippocampalEpisodicNetwork",
    "DistributedCorticalPrototypeNetwork", "PrefrontalParietalControlNetwork",
    "NeuromodulatoryFeedbackGate", "BiologicalPrototypeExemplarSystem",
    "PrototypeExemplarProcessor", "OperationSpec", "OPERATION_REGISTRY",
    "registry_as_dicts", "register_annotations",
]
