"""ORE Point 17: adult individual differences in learning and comprehension."""
from .annotations import register_annotations
from .biology import AdultComprehensionBiologicalNetwork, AdultNetworkActivity
from .cognition import (
    adapt_processing_support, calibrate_decision_and_confidence,
    estimate_interactivity_and_load, estimate_processing_schedule,
    monitor_interference_and_hypotheses, retrieve_gist_detail_source_memory,
)
from .contracts import (
    AdultComprehensionExperience, AdultComprehensionResult,
    AdultDifferencesConfig, AdultLearnerState,
)
from .functional import (
    clamp, entropy, establish_task_goal, estimate_current_resources,
    jaccard_similarity, normalize_probabilities, segment_input, sigmoid, softmax, token_set,
)
from .knowledge import (
    convert_familiar_details_to_chunks, incorporate_cultural_and_lived_experience,
    integrate_knowledge_with_provenance, retrieve_schema_candidates,
    update_knowledge_without_false_assimilation, weight_language_cues,
)
from .pipeline import AdultIndividualDifferencesProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, AdultOperationSpec, registry_as_dicts

__all__ = [
    "AdultComprehensionExperience", "AdultComprehensionResult", "AdultDifferencesConfig",
    "AdultLearnerState", "AdultComprehensionBiologicalNetwork", "AdultNetworkActivity",
    "AdultIndividualDifferencesProcessor", "AdultOperationSpec", "OPERATION_REGISTRY",
    "registry_as_dicts", "register_annotations", "record_result_to_sqlite",
    "clamp", "sigmoid", "softmax", "normalize_probabilities", "entropy",
    "token_set", "jaccard_similarity", "establish_task_goal", "estimate_current_resources",
    "segment_input", "retrieve_schema_candidates", "convert_familiar_details_to_chunks",
    "weight_language_cues", "incorporate_cultural_and_lived_experience",
    "integrate_knowledge_with_provenance", "update_knowledge_without_false_assimilation",
    "estimate_interactivity_and_load", "estimate_processing_schedule",
    "monitor_interference_and_hypotheses", "adapt_processing_support",
    "retrieve_gist_detail_source_memory", "calibrate_decision_and_confidence",
]
