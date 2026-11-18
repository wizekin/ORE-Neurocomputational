"""ORE Point 15: language-specific networks and domain-general reasoning."""
from .annotations import register_annotations
from .biology import BiologicalLanguageReasoningNetwork, NetworkActivity
from .contracts import (
    LanguageEvent, LanguageLearnerState, LanguageProcessingConfig, LanguageProcessingResult,
)
from .functional import (
    activate_language_representations, apply_deliberate_reasoning, build_incremental_structure,
    clamp, compose_sentence_meaning, convert_to_language_units,
    detect_mismatch_and_revise, encode_sensory_signal, estimate_processing_demand,
    infer_social_pragmatics, integrate_interacting_systems, recruit_domain_general_control,
    retrieve_conceptual_knowledge, retrieve_lexical_knowledge, select_task_output,
    shannon_entropy, sigmoid, softmax, tokenize, update_learned_representations,
)
from .pipeline import LanguageDomainReasoningProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "LanguageEvent", "LanguageLearnerState", "LanguageProcessingConfig",
    "LanguageProcessingResult", "NetworkActivity", "BiologicalLanguageReasoningNetwork",
    "LanguageDomainReasoningProcessor", "OPERATION_REGISTRY", "OperationSpec",
    "registry_as_dicts", "register_annotations", "record_result_to_sqlite",
    "clamp", "tokenize", "sigmoid", "softmax", "shannon_entropy",
    "encode_sensory_signal", "convert_to_language_units", "activate_language_representations",
    "retrieve_lexical_knowledge", "build_incremental_structure", "compose_sentence_meaning",
    "retrieve_conceptual_knowledge", "estimate_processing_demand", "recruit_domain_general_control",
    "apply_deliberate_reasoning", "infer_social_pragmatics", "integrate_interacting_systems",
    "detect_mismatch_and_revise", "select_task_output", "update_learned_representations",
]
