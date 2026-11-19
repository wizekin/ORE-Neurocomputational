"""ORE Point 01: active engagement and elaborative learning algorithms."""
from .annotations import register_annotations
from .biology import (
    BiologicalActiveEngagementNetwork,
    FrontoparietalAttentionNetwork,
    HippocampalEpisodeNetwork,
    NeuralPopulation,
    PrefrontalControlNetwork,
    SynapticPlasticityNetwork,
    TemporalLanguageNetwork,
)
from .contracts import (
    ActiveEngagementConfig,
    ActiveEngagementRequest,
    ActiveEngagementResult,
    ActiveEngagementState,
    LearningSegment,
)
from .functional import (
    allocate_attention,
    bind_episode,
    clamp,
    construct_imagery,
    cosine_similarity,
    decide_closure,
    elaborate_explanation,
    evaluate_explanation,
    extract_propositions,
    generate_candidate_answers,
    generate_questions,
    monitor_comprehension,
    prepare_teaching_output,
    reconstruct_explanation,
    repair_explanation,
    segment_material,
    select_target,
    shannon_entropy,
    sigmoid,
    softmax,
    tokenize,
    vectorize_texts,
)
from .pipeline import ActiveEngagementProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "ActiveEngagementConfig", "ActiveEngagementRequest", "ActiveEngagementResult",
    "ActiveEngagementState", "LearningSegment", "NeuralPopulation",
    "TemporalLanguageNetwork", "FrontoparietalAttentionNetwork",
    "PrefrontalControlNetwork", "HippocampalEpisodeNetwork",
    "SynapticPlasticityNetwork", "BiologicalActiveEngagementNetwork",
    "OperationSpec", "OPERATION_REGISTRY", "registry_as_dicts",
    "ActiveEngagementProcessor", "register_annotations", "record_result_to_sqlite",
    "tokenize", "clamp", "sigmoid", "softmax", "shannon_entropy",
    "cosine_similarity", "vectorize_texts", "segment_material",
    "extract_propositions", "select_target", "generate_questions",
    "allocate_attention", "generate_candidate_answers", "reconstruct_explanation",
    "elaborate_explanation", "evaluate_explanation", "repair_explanation",
    "prepare_teaching_output", "construct_imagery", "bind_episode",
    "monitor_comprehension", "decide_closure",
]
