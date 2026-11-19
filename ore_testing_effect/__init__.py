"""ORE Point 04: retrieval practice, corrective feedback, cue routes, and delayed retention."""
from .annotations import register_annotations
from .biology import (
    AnteriorCingulateErrorMonitor,
    BiologicalTestingEffectNetwork,
    CorticalAssociativePlasticity,
    HippocampalPatternCompletion,
    NeuralPopulation,
    NeuromodulatoryLearningGate,
    PrefrontalRetrievalControl,
    vectorize_text,
)
from .contracts import (
    Feedback,
    LearningTarget,
    RetrievalAttempt,
    TargetMemoryState,
    TestingEffectRequest,
    TestingEffectResult,
    TestingEffectState,
    normalize_utc,
)
from .functional import (
    assess_attempt,
    assess_cue_quality,
    assess_test_quality,
    calculate_future_priority,
    choose_learning_action,
    classify_errors,
    clamp,
    cosine_similarity,
    delayed_retention_assessment,
    detect_test_failures,
    lexical_alignment,
    score_order,
    schedule_handoff,
    sigmoid,
    softmax,
    tokenize,
    update_accessibility,
    update_cue_diversity,
    update_knowledge_graph,
    weighted_coverage,
)
from .pipeline import TestingEffectProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "LearningTarget", "RetrievalAttempt", "Feedback", "TestingEffectRequest",
    "TestingEffectState", "TargetMemoryState", "TestingEffectResult", "normalize_utc",
    "NeuralPopulation", "HippocampalPatternCompletion", "PrefrontalRetrievalControl",
    "AnteriorCingulateErrorMonitor", "NeuromodulatoryLearningGate",
    "CorticalAssociativePlasticity", "BiologicalTestingEffectNetwork", "vectorize_text",
    "OperationSpec", "OPERATION_REGISTRY", "registry_as_dicts",
    "TestingEffectProcessor", "register_annotations", "record_result_to_sqlite",
    "clamp", "sigmoid", "softmax", "tokenize", "cosine_similarity",
    "lexical_alignment", "weighted_coverage", "score_order", "assess_cue_quality",
    "assess_test_quality", "assess_attempt", "classify_errors", "update_accessibility",
    "choose_learning_action", "detect_test_failures", "update_knowledge_graph",
    "delayed_retention_assessment", "calculate_future_priority", "update_cue_diversity",
    "schedule_handoff",
]
