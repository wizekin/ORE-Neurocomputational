"""ORE Point 06 commonality detection and revisable concept formation."""
from .annotations import register_annotations
from .biology import (
    AttentionControlNetwork,
    CorticostriatalFeedbackNetwork,
    BiologicalCommonalityNetwork,
    DistributedCorticalConceptNetwork,
    HippocampalEpisodeNetwork,
)
from .contracts import (
    ConceptCandidate,
    ConceptLearningConfig,
    ConceptLearningRequest,
    ConceptLearningResult,
    EpisodeInput,
)
from .pipeline import CommonalityDetectionConceptFormationProcessor
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "EpisodeInput",
    "ConceptLearningConfig",
    "ConceptLearningRequest",
    "ConceptCandidate",
    "ConceptLearningResult",
    "AttentionControlNetwork",
    "CorticostriatalFeedbackNetwork",
    "HippocampalEpisodeNetwork",
    "DistributedCorticalConceptNetwork",
    "BiologicalCommonalityNetwork",
    "CommonalityDetectionConceptFormationProcessor",
    "OPERATION_REGISTRY",
    "OperationSpec",
    "register_annotations",
    "registry_as_dicts",
]
