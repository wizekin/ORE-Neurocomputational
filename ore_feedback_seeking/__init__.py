"""ORE Point 07 feedback-seeking, error diagnosis, correction, and biological-reference package."""
from .annotations import register_annotations
from .biology import DistributedFeedbackNetwork, NeuralPopulation
from .contracts import (
    FeedbackConfig, FeedbackEvidence, FeedbackRequest, FeedbackResult,
    FeedbackSource, FeedbackState, PerformanceContext, ReasoningNode,
    TargetCriterion, TransferProbe,
)
from .functional import (
    ERROR_CATEGORIES, FUNCTIONAL_STEPS, brier_score, clamp, entropy,
    normalize_distribution, sigmoid, token_similarity, weighted_completeness,
    step_01_define_target, step_02_generate_interpretation,
    step_03_monitor_comprehension, step_04_estimate_confidence,
    step_05_select_feedback_route, step_06_obtain_feedback,
    step_07_verify_feedback, step_08_classify_error,
    step_09_separate_performance, step_10_apply_targeted_correction,
    step_11_integrate_knowledge, step_12_explain_correction,
    step_13_test_retrieval, step_14_calibrate_confidence,
    step_15_control_feedback_load, step_16_test_transfer,
    step_17_schedule_delayed_check, step_18_update_monitoring_strategy,
)
from .pipeline import Point07FeedbackProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "TargetCriterion", "ReasoningNode", "FeedbackSource", "FeedbackEvidence",
    "PerformanceContext", "TransferProbe", "FeedbackRequest", "FeedbackState",
    "FeedbackConfig", "FeedbackResult", "NeuralPopulation",
    "DistributedFeedbackNetwork", "Point07FeedbackProcessor", "OperationSpec",
    "OPERATION_REGISTRY", "registry_as_dicts", "register_annotations",
    "record_result_to_sqlite", "ERROR_CATEGORIES", "FUNCTIONAL_STEPS",
    "clamp", "sigmoid", "entropy", "brier_score", "normalize_distribution",
    "token_similarity", "weighted_completeness",
    "step_01_define_target", "step_02_generate_interpretation",
    "step_03_monitor_comprehension", "step_04_estimate_confidence",
    "step_05_select_feedback_route", "step_06_obtain_feedback",
    "step_07_verify_feedback", "step_08_classify_error",
    "step_09_separate_performance", "step_10_apply_targeted_correction",
    "step_11_integrate_knowledge", "step_12_explain_correction",
    "step_13_test_retrieval", "step_14_calibrate_confidence",
    "step_15_control_feedback_load", "step_16_test_transfer",
    "step_17_schedule_delayed_check", "step_18_update_monitoring_strategy",
]
