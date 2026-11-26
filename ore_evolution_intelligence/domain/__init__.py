"""Domain layer: strategy candidates, genotypes, behavior descriptors, evaluation records."""
from .actions import (
    ACTION_VOCABULARY, DEFER_REASONS, ESCALATION_MODES, ORE_MODULES,
    PROHIBITED_ACTIONS, PROHIBITED_ACTION_PREFIXES, ROUTE_QUEUES,
    ActionCall, ParameterSpec, is_approved_action, validate_action,
)
from .behavior_descriptor import (
    DEFAULT_CALIBRATION_SAMPLES, DESCRIPTOR_DIMENSIONS, BehaviorDescriptor,
    DescriptorCalibration, DescriptorDimension, default_calibration,
)
from .evaluation_record import (
    EVALUATOR_VERSION, OUTCOME_CREDIT, OUTCOME_KINDS, AggregateMetrics,
    EvaluationRecord, GateDecision, HardConstraints, NormalizationScales,
    ScalarFitness, TaskOutcomeRecord, t95,
)
from .genome import (
    POLICY_TABLE_CONTEXT_DIMENSIONS, ParameterDefinition, ParameterGenome,
    PolicyTableGenome, canonical_json, content_digest, default_search_space,
    table_context_key,
)
from .strategy import (
    LIFECYCLE_STATES, PRODUCTION_ELIGIBLE_STATES, LineageRecord,
    StrategyCandidate,
)

__all__ = [
    # actions
    "ACTION_VOCABULARY", "DEFER_REASONS", "ESCALATION_MODES", "ORE_MODULES",
    "PROHIBITED_ACTIONS", "PROHIBITED_ACTION_PREFIXES", "ROUTE_QUEUES",
    "ActionCall", "ParameterSpec", "is_approved_action", "validate_action",
    # behavior descriptors
    "DEFAULT_CALIBRATION_SAMPLES", "DESCRIPTOR_DIMENSIONS", "BehaviorDescriptor",
    "DescriptorCalibration", "DescriptorDimension", "default_calibration",
    # evaluation records
    "EVALUATOR_VERSION", "OUTCOME_CREDIT", "OUTCOME_KINDS", "AggregateMetrics",
    "EvaluationRecord", "GateDecision", "HardConstraints", "NormalizationScales",
    "ScalarFitness", "TaskOutcomeRecord", "t95",
    # genomes
    "POLICY_TABLE_CONTEXT_DIMENSIONS", "ParameterDefinition", "ParameterGenome",
    "PolicyTableGenome", "canonical_json", "content_digest", "default_search_space",
    "table_context_key",
    # strategies
    "LIFECYCLE_STATES", "PRODUCTION_ELIGIBLE_STATES", "LineageRecord",
    "StrategyCandidate",
]
