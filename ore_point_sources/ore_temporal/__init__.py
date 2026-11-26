"""ORE Temporal and Event-Driven Intelligence Layer (TEDIL) -- Point 13.

A software-first implementation of the Point 13 specification: continuous-time neural
networks (LNN, CfC, Neural ODE), selective state-space models, spiking neural networks,
deterministic baselines, event and anomaly detection, conformal uncertainty, portable
hardware-independent graphs with a CPU reference backend, shared-workspace publication,
USG mapping, and governed model lifecycle -- integrated with the ORE-Neurocomputational
repository contracts.
"""
from __future__ import annotations

from .annotations import TemporalAnnotationRecorder, record_result_to_sqlite, register_annotations
from .biology import FAMILY_BIOLOGICAL_REFERENCES, MembranePopulation, family_biological_reference
from .config import SOFTWARE_VERSIONS, TemporalConfig, config_hash
from .contracts import (
    Actionability,
    AdmissionDecision,
    BackpressurePolicy,
    BaselineComparison,
    CancellationSignal,
    ClassificationResult,
    EpistemicStatus,
    Estimate,
    EventSeverity,
    ExecutionRecord,
    MissingDataPolicy,
    ModelFamily,
    ModelIdentity,
    ObservationChannel,
    QualityFlag,
    ResultEnvelope,
    ResultType,
    SelectionDecision,
    TaskStatus,
    TemporalObservation,
    TemporalProblemClass,
    TemporalRequest,
    TemporalResult,
    TimestampBasis,
    TypedEventRecord,
    UncertaintyReport,
    ValidationStatus,
)
from .functional import (
    coefficient_of_variation,
    compute_delta_times,
    cusum_update,
    ensemble_mean_variance,
    event_density,
    event_sparsity,
    exponential_relaxation,
    gap_quantiles,
    lagged_correlation,
    lif_advance,
    lif_derivative,
    robust_zscore,
    residual,
    rolling_mean,
    rk4_step,
    euler_step,
    ssm_step,
)
from .models import (
    BaselineKind,
    ClosedFormContinuousNetwork,
    LiquidNeuralNetwork,
    ModelFamilyError,
    ModelStateRecord,
    ModelStateStore,
    ModelStepOutput,
    NeuralODEModel,
    SelectiveStateSpaceModel,
    SimpleKalmanBaseline,
    SpikingNeuralNetwork,
    StateCompatibilityError,
    build_baseline,
    build_model,
    family_available,
    list_available_families,
    run_model_over_window,
)
from .monitoring import (
    ConformalCalibrator,
    DriftMonitor,
    DriftStatus,
    HealthReport,
    ensemble_uncertainty,
    false_alarms_per_unit_time,
    interval_coverage,
    mean_absolute_error,
    numerical_health_check,
    precision_recall_f1,
    root_mean_squared_error,
)
from .pipeline import CalibrationData, TemporalEngine, chronological_split
from .registry import (
    OPERATION_REGISTRY,
    LifecycleState,
    ModelArtifactRecord,
    OperationSpec,
    PromotionDecision,
    TemporalModelRegistry,
    evaluate_promotion,
    registry_as_dicts,
)
from .representations import (
    EventBatch,
    EventSummary,
    EvidenceBundle,
    EvidenceRef,
    GraphNode,
    PortableGraph,
    ProvenanceRecord,
    SUPPORTED_GRAPH_OPERATIONS,
    TemporalSample,
    TemporalWindow,
    UnsupportedGraphOperationError,
)
from .solvers import (
    SolverDiagnostics,
    SolverFailure,
    integrate_interval,
    solve_divergence_check,
)
from .selection import (
    ModelSelectionPolicy,
    ResourceAdmission,
    SelectionProfile,
    build_selection_profile,
)
from .events import (
    ActionabilityGate,
    AnomalyDetector,
    AnomalyHypothesis,
    CUSUMChangeDetector,
    ChangeDetector,
    RollingResidualDetector,
)
from .ingestion import (
    IngestionReport,
    MissingDataHandler,
    ReorderBuffer,
    SamplingProfile,
    TimestampValidator,
    UnitConversionError,
    UnitRegistry,
    VersionedNormalizer,
    apply_ordering_and_deltas,
    build_temporal_window,
    convert_units,
    deduplicate_observations,
    profile_sampling,
)
from .backends import (
    CPUReferenceBackend,
    LavaBackendAdapter,
    OptionalDependencyError,
    export_model_graph,
    benchmark_identical_workload,
    negotiate_capabilities,
)
from .workspace import (
    UNKNOWN_USG_ID,
    USGConceptMapper,
    WorkspaceBridge,
    build_coordinator_notification,
    build_downstream_task_hints,
)
from .registration import (
    ORE_TEMPORAL_MODULE_ID,
    ORE_TEMPORAL_CAPABILITIES,
    build_ore_temporal_manifest,
    register_ore_temporal_contracts,
    register_ore_temporal_module,
)

__all__ = (
    # contracts
    "Actionability", "AdmissionDecision", "BackpressurePolicy", "BaselineComparison",
    "CancellationSignal", "ClassificationResult", "EpistemicStatus", "Estimate",
    "EventSeverity", "ExecutionRecord", "MissingDataPolicy", "ModelFamily",
    "ModelIdentity", "ObservationChannel", "QualityFlag", "ResultEnvelope", "ResultType",
    "SelectionDecision", "TaskStatus", "TemporalObservation", "TemporalProblemClass",
    "TemporalRequest", "TemporalResult", "TimestampBasis", "TypedEventRecord",
    "UncertaintyReport", "ValidationStatus",
    # config and functional math
    "SOFTWARE_VERSIONS", "TemporalConfig", "config_hash",
    "coefficient_of_variation", "compute_delta_times", "cusum_update",
    "ensemble_mean_variance", "event_density", "event_sparsity", "exponential_relaxation",
    "gap_quantiles", "lagged_correlation", "lif_advance", "lif_derivative",
    "robust_zscore", "residual", "rolling_mean", "rk4_step", "euler_step", "ssm_step",
    # biology
    "FAMILY_BIOLOGICAL_REFERENCES", "MembranePopulation", "family_biological_reference",
    # models
    "BaselineKind", "ClosedFormContinuousNetwork", "LiquidNeuralNetwork",
    "ModelFamilyError", "ModelStateRecord", "ModelStateStore", "ModelStepOutput",
    "NeuralODEModel", "SelectiveStateSpaceModel", "SimpleKalmanBaseline",
    "SpikingNeuralNetwork", "StateCompatibilityError", "build_baseline", "build_model",
    "family_available", "list_available_families", "run_model_over_window",
    # solvers
    "SolverDiagnostics", "SolverFailure", "integrate_interval", "solve_divergence_check",
    # ingestion
    "IngestionReport", "MissingDataHandler", "ReorderBuffer", "SamplingProfile",
    "TimestampValidator", "UnitConversionError", "UnitRegistry", "VersionedNormalizer",
    "apply_ordering_and_deltas", "build_temporal_window", "convert_units",
    "deduplicate_observations", "profile_sampling",
    # representations
    "EventBatch", "EventSummary", "EvidenceBundle", "EvidenceRef", "GraphNode",
    "PortableGraph", "ProvenanceRecord", "SUPPORTED_GRAPH_OPERATIONS", "TemporalSample",
    "TemporalWindow", "UnsupportedGraphOperationError",
    # events
    "ActionabilityGate", "AnomalyDetector", "AnomalyHypothesis", "CUSUMChangeDetector",
    "ChangeDetector", "RollingResidualDetector",
    # selection
    "ModelSelectionPolicy", "ResourceAdmission", "SelectionProfile",
    "build_selection_profile",
    # monitoring
    "ConformalCalibrator", "DriftMonitor", "DriftStatus", "HealthReport",
    "ensemble_uncertainty", "false_alarms_per_unit_time", "interval_coverage",
    "mean_absolute_error", "numerical_health_check", "precision_recall_f1",
    "root_mean_squared_error",
    # backends
    "CPUReferenceBackend", "LavaBackendAdapter", "OptionalDependencyError",
    "export_model_graph", "benchmark_identical_workload", "negotiate_capabilities",
    # workspace
    "UNKNOWN_USG_ID", "USGConceptMapper", "WorkspaceBridge",
    "build_coordinator_notification", "build_downstream_task_hints",
    # registry and governance
    "OPERATION_REGISTRY", "LifecycleState", "ModelArtifactRecord", "OperationSpec",
    "PromotionDecision", "TemporalModelRegistry", "evaluate_promotion",
    "registry_as_dicts",
    # logging
    "TemporalAnnotationRecorder", "record_result_to_sqlite", "register_annotations",
    # pipeline
    "CalibrationData", "TemporalEngine", "chronological_split",
    # central registration
    "ORE_TEMPORAL_MODULE_ID", "ORE_TEMPORAL_CAPABILITIES", "build_ore_temporal_manifest",
    "register_ore_temporal_contracts", "register_ore_temporal_module",
)
