"""ORE Point 10 schema-guided inference and generalization package."""

from .annotations import register_annotations
from .biology import (
    AttentionConflictNetwork,
    BiologicalSchemaInferenceNetwork,
    HippocampalEpisodicComparisonNetwork,
    MedialPrefrontalPriorControlNetwork,
    OrbitofrontalActionSelectionNetwork,
    TemporalSemanticActivationNetwork,
)
from .contracts import (
    ActionDecision,
    ActionOption,
    CandidateScore,
    CaseEvidence,
    EvidenceItem,
    FeedbackResult,
    InferenceConfig,
    InferenceResult,
    InferredProperty,
    QueryOption,
    QueryRecommendation,
    RelationEvidence,
    SchemaProfile,
)
from .functional import (
    context_compatibility,
    entropy,
    event_transition_compatibility,
    expected_information_gain,
    feature_log_likelihood_ratio,
    goal_compatibility,
    merge_case_evidence,
    predict_missing_properties,
    predict_missing_relations,
    predict_next_events,
    rank_schema_profiles,
    relation_compatibility,
    select_action,
    select_hierarchical_category,
    select_information_query,
    softmax,
)
from .pipeline import SchemaGuidedInferenceProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "EvidenceItem", "RelationEvidence", "CaseEvidence", "SchemaProfile",
    "InferenceConfig", "CandidateScore", "InferredProperty", "QueryOption",
    "QueryRecommendation", "ActionOption", "ActionDecision", "FeedbackResult",
    "InferenceResult", "HippocampalEpisodicComparisonNetwork",
    "TemporalSemanticActivationNetwork", "MedialPrefrontalPriorControlNetwork",
    "AttentionConflictNetwork", "OrbitofrontalActionSelectionNetwork",
    "BiologicalSchemaInferenceNetwork", "SchemaGuidedInferenceProcessor",
    "OPERATION_REGISTRY", "OperationSpec", "registry_as_dicts",
    "register_annotations", "record_result_to_sqlite", "softmax", "entropy",
    "feature_log_likelihood_ratio", "relation_compatibility",
    "context_compatibility", "goal_compatibility", "event_transition_compatibility",
    "rank_schema_profiles", "select_hierarchical_category", "predict_missing_properties",
    "predict_missing_relations", "predict_next_events", "expected_information_gain",
    "select_information_query", "select_action", "merge_case_evidence",
]
