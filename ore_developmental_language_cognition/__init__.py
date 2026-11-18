"""ORE Point 16: lifespan development of language, concepts, memory, and executive cognition."""
from .annotations import register_annotations
from .biology import DevelopmentalBiologicalNetwork, DevelopmentalNetworkActivity
from .cognition import (
    calibrate_developmental_support, integrate_hippocampal_prefrontal_control,
    maintain_working_memory, refine_developing_circuits, refine_multi_step_plan,
    update_executive_control,
)
from .contracts import (
    DevelopmentalConfig, DevelopmentalExperience, DevelopmentalLearnerState,
    DevelopmentalProcessingResult,
)
from .functional import (
    accumulate_statistical_regularities, build_provisional_phonetic_categories,
    clamp, cosine_similarity, entropy, establish_broad_sensitivity, euclidean,
    extract_acoustic_features, feature_overlap, infer_word_boundaries, sigmoid, softmax, vectorize,
)
from .grounded_learning import (
    accelerate_vocabulary_learning, build_grounded_concepts, map_word_forms_to_meanings,
    stabilize_developmental_memory,
)
from .pipeline import DevelopmentalLanguageCognitionProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, DevelopmentalOperationSpec, registry_as_dicts

__all__ = [
    "DevelopmentalConfig", "DevelopmentalExperience", "DevelopmentalLearnerState",
    "DevelopmentalProcessingResult", "DevelopmentalBiologicalNetwork",
    "DevelopmentalNetworkActivity", "DevelopmentalLanguageCognitionProcessor",
    "DevelopmentalOperationSpec", "OPERATION_REGISTRY", "registry_as_dicts",
    "register_annotations", "record_result_to_sqlite",
    "clamp", "sigmoid", "softmax", "entropy", "vectorize", "euclidean",
    "cosine_similarity", "feature_overlap", "extract_acoustic_features",
    "establish_broad_sensitivity", "build_provisional_phonetic_categories",
    "accumulate_statistical_regularities", "infer_word_boundaries",
    "map_word_forms_to_meanings", "accelerate_vocabulary_learning",
    "build_grounded_concepts", "stabilize_developmental_memory",
    "maintain_working_memory", "update_executive_control",
    "refine_developing_circuits", "integrate_hippocampal_prefrontal_control",
    "refine_multi_step_plan", "calibrate_developmental_support",
]
