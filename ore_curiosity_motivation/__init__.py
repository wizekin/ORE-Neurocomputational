"""ORE Point 08 curiosity, motivation, exploration, and memory-consolidation implementation."""
from .annotations import register_annotations
from .biology import DistributedMotivationNetwork, NeuralPopulation
from .contracts import (
    LearningAction, LearningOutcome, LearningRequest, LearningResult, LearningState,
    LearningUnit, MotivationConfig, RetrievalCheck,
)
from .functional import (
    FUNCTIONAL_STEPS, clamp, sigmoid, shannon_entropy, softmax,
    step_01_parse_learning_units, step_02_detect_novelty_and_information_gaps,
    step_03_estimate_information_value, step_04_appraise_learnability_and_safety,
    step_05_estimate_curiosity, step_06_separate_motivation_sources,
    step_07_estimate_effort_and_opportunity_cost, step_08_set_goal_and_progress,
    step_09_generate_questions_and_actions, step_10_allocate_attention,
    step_11_explore_and_sample_evidence, step_12_update_expectations,
    step_13_encode_target_information, step_14_encode_incidental_information,
    step_15_appraise_emotion, step_16_integrate_motivation_and_emotion,
    step_17_detect_disengagement, step_18_adapt_challenge_and_explanation,
    step_19_reinforce_progress_and_competence, step_20_preserve_autonomy_and_agency,
    step_21_prioritize_memory_retention, step_22_support_consolidation,
    step_23_test_durable_understanding, step_24_update_future_motivation,
)
from .pipeline import Point08MotivatedLearningProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "LearningUnit", "LearningAction", "LearningOutcome", "RetrievalCheck",
    "LearningRequest", "LearningState", "MotivationConfig", "LearningResult",
    "NeuralPopulation", "DistributedMotivationNetwork", "Point08MotivatedLearningProcessor",
    "OperationSpec", "OPERATION_REGISTRY", "registry_as_dicts", "register_annotations",
    "record_result_to_sqlite", "FUNCTIONAL_STEPS", "clamp", "sigmoid", "softmax",
    "shannon_entropy",
    "step_01_parse_learning_units", "step_02_detect_novelty_and_information_gaps",
    "step_03_estimate_information_value", "step_04_appraise_learnability_and_safety",
    "step_05_estimate_curiosity", "step_06_separate_motivation_sources",
    "step_07_estimate_effort_and_opportunity_cost", "step_08_set_goal_and_progress",
    "step_09_generate_questions_and_actions", "step_10_allocate_attention",
    "step_11_explore_and_sample_evidence", "step_12_update_expectations",
    "step_13_encode_target_information", "step_14_encode_incidental_information",
    "step_15_appraise_emotion", "step_16_integrate_motivation_and_emotion",
    "step_17_detect_disengagement", "step_18_adapt_challenge_and_explanation",
    "step_19_reinforce_progress_and_competence", "step_20_preserve_autonomy_and_agency",
    "step_21_prioritize_memory_retention", "step_22_support_consolidation",
    "step_23_test_durable_understanding", "step_24_update_future_motivation",
]
