"""ORE Point 04 (ranked 13): Cognitive World Model and predictive control."""
from .annotations import register_annotations
from .baselines import (
    DEFAULT_OPERATION_CLASS_BASELINES,
    BetaBinomialEstimator,
    ExponentialMovingEstimator,
    OperationClassBaseline,
    RobustEstimator,
    baseline_for,
    evidence_coverage_proxy,
    expected_information_gain,
    shannon_entropy,
)
from .contracts import (
    MISSING,
    AdvisoryMode,
    ArtifactLifecycle,
    ArtifactTypeEffect,
    CognitiveOperation,
    CognitiveStateFeatures,
    CWMBudget,
    CWMConfig,
    CWMError,
    ExecutionStatus,
    FallbackDecision,
    FallbackTrigger,
    FeatureProvenance,
    FeatureValue,
    InformationGainEstimate,
    ObservedOutcome,
    OperationContext,
    OperationPrecondition,
    OperationType,
    OutcomeClass,
    OutcomeDimensions,
    PredictedOutcome,
    PredictionLineage,
    PredictionRecord,
    PredictionUncertainty,
    ResourceEnvelope,
    ResourceState,
    RiskAssessment,
    SafetyClass,
    StaleRevisionError,
)
from .features import FeatureExtractor
from .learning import (
    CalibrationMonitor,
    CalibrationReport,
    DriftMonitor,
    ExecutionLedger,
    ModelVersion,
    ModelVersionRegistry,
    PredictionErrorReport,
    evaluate_chronological,
)
from .observability import SQLiteCWMObserver, SQLitePredictionStore
from .planner import (
    CWMPlan,
    MPCObjectiveWeights,
    MPCPlanner,
    PredictedSequence,
    PredictedStep,
    stop_action,
    verification_action,
)
from .policy import (
    ConfidenceGate,
    ConstraintVerdict,
    FallbackPolicy,
    HardConstraintSet,
    fallback_trigger_for,
)
from .predictor import PredictedTransition, StatisticalTransitionModel, classify_operation
from .registry import OPERATION_REGISTRY, CWMOperationSpec, registry_as_dicts
from .service import (
    CognitiveWorldModel,
    CWMRunResult,
    CWMStepRecord,
    ShadowReport,
    record_result_to_sqlite,
)

__all__ = [
    # contracts
    "AdvisoryMode", "ArtifactLifecycle", "ArtifactTypeEffect", "CWMConfig", "CWMBudget",
    "CWMError", "CognitiveOperation", "CognitiveStateFeatures", "ExecutionStatus",
    "FeatureProvenance", "FeatureValue", "FallbackDecision", "FallbackTrigger",
    "InformationGainEstimate", "MISSING", "ObservedOutcome", "OperationContext",
    "OperationPrecondition", "OperationType", "OutcomeClass", "OutcomeDimensions",
    "PredictedOutcome", "PredictionLineage", "PredictionRecord", "PredictionUncertainty",
    "ResourceEnvelope", "ResourceState", "RiskAssessment", "SafetyClass",
    "StaleRevisionError",
    # features
    "FeatureExtractor",
    # baselines
    "BetaBinomialEstimator", "DEFAULT_OPERATION_CLASS_BASELINES",
    "ExponentialMovingEstimator", "OperationClassBaseline", "RobustEstimator",
    "baseline_for", "evidence_coverage_proxy", "expected_information_gain",
    "shannon_entropy",
    # learning
    "CalibrationMonitor", "CalibrationReport", "DriftMonitor", "ExecutionLedger",
    "ModelVersion", "ModelVersionRegistry", "PredictionErrorReport",
    "evaluate_chronological",
    # predictor
    "PredictedTransition", "StatisticalTransitionModel", "classify_operation",
    # planner
    "CWMPlan", "MPCObjectiveWeights", "MPCPlanner", "PredictedSequence", "PredictedStep",
    "verification_action", "stop_action",
    # policy
    "ConfidenceGate", "ConstraintVerdict", "FallbackPolicy", "HardConstraintSet",
    "fallback_trigger_for",
    # observability
    "SQLiteCWMObserver", "SQLitePredictionStore", "register_annotations",
    # registry
    "CWMOperationSpec", "OPERATION_REGISTRY", "registry_as_dicts",
    # service
    "CognitiveWorldModel", "CWMRunResult", "CWMStepRecord", "ShadowReport",
    "record_result_to_sqlite",
]
