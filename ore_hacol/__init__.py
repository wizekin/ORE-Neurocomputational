"""ORE Point 21 (Most Intelligent Systems sequence): Hardware-Aware
Computational Optimization Layer (HACOL).

Implements the integrated algorithm of "ORE Integrated Algorithm -- Point 21:
Hardware-Aware Numerical Computing and Quantum Algorithms": a measured
optimization loop over ORE's actual numerical workloads with a trusted
reference preserved for every operation, mixed-precision and IO-aware
attention paths, randomized linear algebra with declared error budgets,
structured transforms, classical-first combinatorial optimization, quantum
algorithms kept strictly as research records, CROS admission, Blackboard
publication of versioned envelopes, and ore_logger SQLite integration.
"""
from .contracts import (
    ENVELOPE_VERSION,
    MODULE_VERSION,
    SPEC_SOURCE,
    AdmissionDecision,
    ContractAssessment,
    ConsequenceLevel,
    CorrectnessContract,
    CostEstimate,
    ErrorReport,
    HardwareProfile,
    OperationCancelled,
    OperationKind,
    OptimizationPlan,
    OptimizationRequest,
    OptimizationResult,
    ProvenanceRecord,
    RequestValidationError,
    ResourceAsk,
    SensitivityProfile,
    SUPPORTED_DTYPES,
    ValidationStatus,
    ValueEstimate,
)
from .attention import (
    ALL_MASKED_POLICIES,
    AttentionRecord,
    OnlineSoftmaxState,
    attention_eligibility,
    online_softmax_attention,
    reference_attention,
    select_tile_dimensions,
    stable_softmax,
    tiled_attention,
)
from .randomized_linalg import (
    RandomizedApproxMetadata,
    build_test_matrix,
    error_estimate_probe,
    low_rank_opportunity,
    randomized_range,
    randomized_svd,
    rank_deficiency_report,
    reconstruction_error,
    sketched_least_squares,
)
from .structured import (
    AmortizationReport,
    ButterflyFactors,
    StructureReport,
    ToeplitzOperator,
    VerificationReport,
    butterfly_certificate,
    detect_structure,
    measure_amortization,
    verify_structured_apply,
)
from .precision import (
    DTYPE_LIMITS,
    PrecisionDecision,
    RangeReport,
    check_range,
    choose_precision,
    iterative_refinement,
    mixed_precision_matmul,
    reject_unsafe_precision,
    to_bfloat16,
    verify_high_precision,
)
from .optimization import (
    ORTOOLS_AVAILABLE,
    AssignmentProblem,
    compare_solvers,
    cp_sat_assignment,
    decode_assignment,
    encode_assignment_qubo,
    exact_qubo_solver,
    greedy_assignment,
    hungarian_assignment,
    ising_energy,
    ising_from_qubo,
    qubo_energy,
    simulated_annealing,
    tabu_search,
    verify_penalty_preserves_optimum,
)
from .quantum_research import (
    GroverRecord,
    QECOverheadRecord,
    ShorRecord,
    grover_query_bound,
    grover_record,
    logical_error_rate_per_cycle,
    qec_overhead_record,
    quantum_research_gate,
    research_catalog,
    shor_record,
    surface_code_physical_qubits,
)
from .capabilities import (
    capability_summary,
    discover_hardware_profile,
    measure_bandwidth_bytes_per_second,
    measure_compute_rate_flops_per_second,
)
from .profiling import (
    FIXTURE_FACTORIES,
    LatencyStats,
    ProfileReport,
    attention_fixture,
    causal_mask,
    padding_mask,
    classify_bottleneck,
    estimate_optimization_value,
    least_squares_fixture,
    low_rank_matrix,
    measure_latency,
    profile_function,
    qubo_assignment_fixture,
    structured_transform_fixture,
    synthetic_bandwidth_bound_workload,
    synthetic_compute_bound_workload,
)
from .planning import POLICY_TABLE, Candidate, candidates_for, choose_plan
from .validation import (
    RegressionRecord,
    adversarial_attention_fixture,
    adversarial_least_squares,
    adversarial_overflow_matrix,
    adversarial_rank_deficient_matrix,
    check_attention_properties,
    compare_arrays,
    determinism_check,
    downstream_decision_stability,
    load_regression_records,
    run_adversarial_suite,
    save_regression_records,
)
from .cros import CROSGate, ExternalCROSAdapter, LocalCROSGate, default_cros_gate
from .backends import BackendSpec, discover_backends, reference_cpu_backend, select_backends
from .workspace import build_blackboard_envelope, publish_result, publish_to_blackboard
from .logging_adapter import (
    HACOLLogger,
    record_result_to_sqlite,
    register_hacol_annotations,
)
from .registry import (
    ALWAYS_STEPS,
    HACOLStepSpec,
    OPERATION_REGISTRY,
    PHASE_G_ACTIONS,
    registry_as_dicts,
    step_by_number,
    steps_for_phase,
)
from .biology import (
    BIOLOGY_REGISTRY,
    HACOLBiologyNetwork,
    biology_reference_for_phase,
)
from .pipeline import (
    CancellationToken,
    FeatureFlags,
    HardwareAwareOptimizer,
)

__all__ = [
    "AdmissionDecision", "ContractAssessment", "ConsequenceLevel",
    "CorrectnessContract", "CostEstimate", "ErrorReport", "HardwareProfile",
    "OperationCancelled", "OperationKind", "OptimizationPlan",
    "OptimizationRequest", "OptimizationResult", "ProvenanceRecord",
    "RequestValidationError", "ResourceAsk", "SensitivityProfile",
    "SUPPORTED_DTYPES", "ValidationStatus", "ValueEstimate",
    "ALL_MASKED_POLICIES", "AttentionRecord", "OnlineSoftmaxState",
    "attention_eligibility", "causal_mask", "online_softmax_attention",
    "padding_mask", "reference_attention", "select_tile_dimensions",
    "stable_softmax", "tiled_attention",
    "RandomizedApproxMetadata", "build_test_matrix", "error_estimate_probe",
    "low_rank_opportunity", "randomized_range", "randomized_svd",
    "rank_deficiency_report", "reconstruction_error", "sketched_least_squares",
    "AmortizationReport", "ButterflyFactors", "StructureReport",
    "ToeplitzOperator", "VerificationReport", "butterfly_certificate",
    "detect_structure", "measure_amortization", "verify_structured_apply",
    "DTYPE_LIMITS", "PrecisionDecision", "RangeReport", "check_range",
    "choose_precision", "iterative_refinement", "mixed_precision_matmul",
    "reject_unsafe_precision", "to_bfloat16", "verify_high_precision",
    "ORTOOLS_AVAILABLE", "AssignmentProblem", "compare_solvers",
    "cp_sat_assignment", "decode_assignment", "encode_assignment_qubo",
    "exact_qubo_solver", "greedy_assignment", "hungarian_assignment",
    "ising_energy", "ising_from_qubo", "qubo_energy", "simulated_annealing",
    "tabu_search", "verify_penalty_preserves_optimum",
    "GroverRecord", "QECOverheadRecord", "ShorRecord", "grover_query_bound",
    "grover_record", "logical_error_rate_per_cycle", "qec_overhead_record",
    "quantum_research_gate", "research_catalog", "shor_record",
    "surface_code_physical_qubits",
    "capability_summary", "discover_hardware_profile",
    "measure_bandwidth_bytes_per_second", "measure_compute_rate_flops_per_second",
    "FIXTURE_FACTORIES", "LatencyStats", "ProfileReport", "attention_fixture",
    "classify_bottleneck", "estimate_optimization_value", "least_squares_fixture",
    "low_rank_matrix", "measure_latency", "profile_function",
    "qubo_assignment_fixture", "structured_transform_fixture",
    "synthetic_bandwidth_bound_workload", "synthetic_compute_bound_workload",
    "POLICY_TABLE", "Candidate", "candidates_for", "choose_plan", "causal_mask", "padding_mask",
    "RegressionRecord", "adversarial_attention_fixture",
    "adversarial_least_squares", "adversarial_overflow_matrix",
    "adversarial_rank_deficient_matrix", "check_attention_properties",
    "compare_arrays", "determinism_check", "downstream_decision_stability",
    "load_regression_records", "run_adversarial_suite",
    "save_regression_records",
    "CROSGate", "ExternalCROSAdapter", "LocalCROSGate", "default_cros_gate",
    "BackendSpec", "discover_backends", "reference_cpu_backend",
    "select_backends",
    "build_blackboard_envelope", "publish_result", "publish_to_blackboard",
    "HACOLLogger", "record_result_to_sqlite", "register_hacol_annotations",
    "ALWAYS_STEPS", "HACOLStepSpec", "OPERATION_REGISTRY",
    "PHASE_G_ACTIONS", "registry_as_dicts", "step_by_number",
    "steps_for_phase",
    "BIOLOGY_REGISTRY", "HACOLBiologyNetwork", "biology_reference_for_phase",
    "CancellationToken", "FeatureFlags", "HardwareAwareOptimizer",
    "ENVELOPE_VERSION", "MODULE_VERSION", "SPEC_SOURCE",
]
