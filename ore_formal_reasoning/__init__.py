"""Formal Reasoning and Constraint Engine (FRCE) for ORE-Neurocomputational.

This package implements the 52-step integrated algorithm of ``ORE Integrated
Algorithm — Point 5: Automated Reasoning, Theorem Proving, SAT/SMT Solvers and
Constraint-Based Intelligence`` on top of the existing ``ore_workspace``,
``ore_usg`` and ``ore_logger`` components. One
``FormalReasoningCoordinator.run(request)`` call performs the full algorithm:
request intake with fact/assumption separation, typed formalization into a
versioned logical IR with per-domain semantic profiles, stratified Datalog
inference with derivation provenance and non-destructive contradiction
handling, Z3-backed SAT/SMT consistency and entailment-by-counterexample with
named assumptions and independently validated models, OR-Tools CP-SAT module
selection with honest optimality reporting, machine-checked proofs via a
trusted natural-deduction kernel with Lean export, unified typed result
envelopes, read-only USG governance with explicit promotion, SQLite audit
records, bounded solver workers, and capability-gated release. See
``docs/FORMAL_REASONING_IMPLEMENTATION.md`` for the 52-step mapping and
acceptance-test coverage.
"""
from .contracts import (
    AssuranceLevel,
    EvidenceTag,
    FormalResult,
    FormalizationRequest,
    NamedAssumption,
    NON_SUCCESS_STATUSES,
    PermittedTheory,
    ResourceBudget,
    ResultKind,
    ResultStatus,
    SolverIdentity,
    SolverStatus,
    SourceRef,
    TaskClass,
    TemporalScope,
    TheoryProfile,
    canonical_json,
    sha256_json,
    sha256_text,
)
from .consistency import (
    CardinalityConstraint,
    ConsistencyChecker,
    ConsistencyReport,
    ConflictKind,
    ConflictRecord,
    ExclusivityGroup,
    FunctionalConstraint,
)
from .coordinator import (
    FormalReasoningCoordinator,
    FormalReasoningRun,
    FormalReasoningService,
    FrcePolicy,
)
from .cpsat import (
    CpsatOptimizer,
    ModuleCandidate,
    OptimizationOutcome,
    OptimizationRouter,
    ReferenceOptimizer,
    SelectionProblem,
    validate_selection,
)
from .datalog import (
    DatalogCache,
    DatalogEngine,
    DatalogFact,
    DatalogLimits,
    DatalogResult,
    Derivation,
    DerivedFact,
    build_dependency_graph,
    facts_from_ir,
    rules_from_ir,
)
from .envelope import (
    CertificateReplay,
    EnvelopeCheck,
    EnvelopeValidation,
    result_fingerprint,
    validate_result_envelope,
)
from .evaluation import (
    BenchmarkReport,
    CaseResult,
    assert_unknown_never_coerced,
    classify_failure_for_improvement,
    coerce_to_truth_value,
    run_benchmark_suite,
)
from .exceptions import (
    AmbiguousEntityError,
    FormalReasoningError,
    GovernanceViolationError,
    InsufficientFormalizationError,
    InvalidFormalizationError,
    MalformedSolverOutputError,
    NonStratifiableProgramError,
    ProofCheckFailureError,
    SolverBudgetExceededError,
    SolverCrashError,
    UnitMismatchError,
    UnsafeRuleError,
    UnsupportedTheoryError,
)
from .governance import (
    KnowledgeWriteGovernance,
    PromotionDecision,
    PromotionPolicy,
    PromotionProposal,
    SQLiteAuditLog,
    WorkspacePublicationAdapter,
)
from .intake import (
    ClarificationRequest,
    CompletenessReport,
    EvidenceSeparation,
    IntakeReport,
    ProvenanceLedger,
    TaskClassification,
    run_intake,
)
from .ir import (
    AndF,
    ArithExpr,
    ArithOp,
    Atom,
    BVLit,
    BvExpr,
    BvOp,
    BodyLiteral,
    BoolLit,
    BoolTermF,
    CardinalityF,
    Compare,
    CompareOp,
    Const,
    DomainSemantics,
    ExistsF,
    FormalIR,
    FormalIRBuilder,
    ForallF,
    FuncApp,
    FunctionDecl,
    IffF,
    ImpliesF,
    NotF,
    NumLit,
    OrF,
    PredicateDecl,
    RuleDecl,
    SortDecl,
    SortKind,
    Statement,
    StatementKind,
    StrExpr,
    StrLit,
    StrOp,
    ToReal,
    UnitConversion,
    UnitDecl,
    UnitTable,
    Var,
    VarDecl,
)
from .policy import (
    BinaryAllowlist,
    BoundedWorker,
    CapabilityFlags,
    FileAccessPolicy,
    FailureClass,
    NetworkPolicy,
    RecoveryPolicy,
    WorkerOutcome,
    classify_failure,
)
from .proofs import (
    KernelVerdict,
    LeanChecker,
    LeanExporter,
    Premise,
    ProofArtifact,
    ProofCandidate,
    ProofCheckOutcome,
    ProofObligation,
    ProofSearcher,
    ProofVerifier,
    PropAnd,
    PropBottom,
    PropImplies,
    PropOr,
    PropVar,
    TrustedKernel,
    proposition_text,
)
from .registry import (
    FRCE_OPERATION_IDS,
    FRCE_OPERATIONS,
    OperationSpec,
    operation_by_id,
    operation_by_step,
    operations_by_phase,
)
from .smt import (
    IndependentEvaluator,
    SmtAdapter,
    SmtOutcome,
    TheoryNeed,
    Z3_AVAILABLE,
    Z3_VERSION,
    detect_theories,
)
from .usg_adapter import (
    AmbiguousResolution,
    KnowledgeRetriever,
    ResolvedEntity,
    RetrievalRecord,
    TypedKnowledgeRecord,
    USGEntityResolver,
    USGReadOnlyGuard,
    USGToIRMapper,
)
from .version import (
    FORMAL_REASONING_BENCHMARK_VERSION,
    FORMAL_REASONING_IR_VERSION,
    FORMAL_REASONING_PACKAGE_VERSION,
    FORMAL_REASONING_RESULT_VERSION,
    FORMAL_REASONING_SCHEMA_VERSION,
    FRCE_PRODUCER,
)

__all__ = [
    "FORMAL_REASONING_PACKAGE_VERSION",
    "FORMAL_REASONING_SCHEMA_VERSION",
    "FORMAL_REASONING_IR_VERSION",
    "FORMAL_REASONING_RESULT_VERSION",
    "FORMAL_REASONING_BENCHMARK_VERSION",
    "FRCE_PRODUCER",
    # contracts
    "AssuranceLevel", "EvidenceTag", "FormalResult", "FormalizationRequest",
    "NamedAssumption", "NON_SUCCESS_STATUSES", "PermittedTheory", "ResourceBudget",
    "ResultKind", "ResultStatus", "SolverIdentity", "SolverStatus", "SourceRef",
    "TaskClass", "TemporalScope", "TheoryProfile", "canonical_json", "sha256_json",
    "sha256_text",
    # consistency
    "CardinalityConstraint", "ConsistencyChecker", "ConsistencyReport",
    "ConflictKind", "ConflictRecord", "ExclusivityGroup", "FunctionalConstraint",
    # coordinator
    "FormalReasoningCoordinator", "FormalReasoningRun", "FormalReasoningService",
    "FrcePolicy",
    # cpsat
    "CpsatOptimizer", "ModuleCandidate", "OptimizationOutcome", "OptimizationRouter",
    "ReferenceOptimizer", "SelectionProblem", "validate_selection",
    # datalog
    "DatalogCache", "DatalogEngine", "DatalogFact", "DatalogLimits", "DatalogResult",
    "Derivation", "DerivedFact", "build_dependency_graph", "facts_from_ir",
    "rules_from_ir",
    # envelope
    "CertificateReplay", "EnvelopeCheck", "EnvelopeValidation", "result_fingerprint",
    "validate_result_envelope",
    # evaluation
    "BenchmarkReport", "CaseResult", "assert_unknown_never_coerced",
    "classify_failure_for_improvement", "coerce_to_truth_value", "run_benchmark_suite",
    # exceptions
    "AmbiguousEntityError", "FormalReasoningError", "GovernanceViolationError",
    "InsufficientFormalizationError", "InvalidFormalizationError",
    "MalformedSolverOutputError", "NonStratifiableProgramError",
    "ProofCheckFailureError", "SolverBudgetExceededError", "SolverCrashError",
    "UnitMismatchError", "UnsafeRuleError", "UnsupportedTheoryError",
    # governance
    "KnowledgeWriteGovernance", "PromotionDecision", "PromotionPolicy",
    "PromotionProposal", "SQLiteAuditLog", "WorkspacePublicationAdapter",
    # intake
    "ClarificationRequest", "CompletenessReport", "EvidenceSeparation",
    "IntakeReport", "ProvenanceLedger", "TaskClassification", "run_intake",
    # ir
    "AndF", "ArithExpr", "ArithOp", "Atom", "BVLit", "BvExpr", "BvOp",
    "BodyLiteral", "BoolLit", "BoolTermF", "CardinalityF", "Compare", "CompareOp",
    "Const", "DomainSemantics", "ExistsF", "FormalIR", "FormalIRBuilder", "ForallF",
    "FuncApp", "FunctionDecl", "IffF", "ImpliesF", "NotF", "NumLit", "OrF",
    "PredicateDecl", "RuleDecl", "SortDecl", "SortKind", "Statement",
    "StatementKind", "StrExpr", "StrLit", "StrOp", "ToReal", "UnitConversion",
    "UnitDecl", "UnitTable", "Var", "VarDecl",
    # policy
    "BinaryAllowlist", "BoundedWorker", "CapabilityFlags", "FileAccessPolicy",
    "FailureClass", "NetworkPolicy", "RecoveryPolicy", "WorkerOutcome",
    "classify_failure",
    # proofs
    "KernelVerdict", "LeanChecker", "LeanExporter", "Premise", "ProofArtifact",
    "ProofCandidate", "ProofCheckOutcome", "ProofObligation", "ProofSearcher",
    "ProofVerifier", "PropAnd", "PropBottom", "PropImplies", "PropOr", "PropVar",
    "TrustedKernel", "proposition_text",
    # registry
    "FRCE_OPERATION_IDS", "FRCE_OPERATIONS", "OperationSpec", "operation_by_id",
    "operation_by_step", "operations_by_phase",
    # smt
    "IndependentEvaluator", "SmtAdapter", "SmtOutcome", "TheoryNeed",
    "Z3_AVAILABLE", "Z3_VERSION", "detect_theories",
    # usg adapter
    "AmbiguousResolution", "KnowledgeRetriever", "ResolvedEntity", "RetrievalRecord",
    "TypedKnowledgeRecord", "USGEntityResolver", "USGReadOnlyGuard", "USGToIRMapper",
]
