"""Neuro-Symbolic Reasoning Layer for ORE-Neurocomputational (Point 3).

This package implements the 45-step integrated neuro-symbolic reasoning
algorithm of ``ORE —Neurocomputational Point 03 Neuro-Symbolic Reasoning
Algorithm.txt`` on top of the existing ``ore_workspace``, ``ore_contracts``,
``ore_usg`` and ``ore_logger`` components. One
``NeuroSymbolicReasoningCoordinator.run(request)`` call performs the full
algorithm: request binding over one workspace snapshot revision, read-only USG
semantic resolution with preserved ambiguity, safe-profile rule validation,
typed unification with the strict firing gate, bounded forward/backward
chaining with cycle detection, content-addressed proof DAGs with an
independent verifier, hard/soft constraint reasoning with preserved
alternatives, dependency-aware probabilistic evidence with decomposed
confidence, validated learned-model proposals that never become facts,
conflict preservation, gated publication through the atomic workspace path,
independent review requests, and checkpointed observability. See
``docs/NEUROSYMBOLIC_REASONING_IMPLEMENTATION.md`` for the 45-step mapping and
acceptance-test coverage.
"""
from .ambiguity_resolver import (
    AmbiguityGroup,
    AmbiguityResolver,
    RankingWeights,
    ScoredCandidate,
)
from .config import NSRLAblationSwitches, NSRLPolicy, default_policy
from .constraint_engine import (
    ConstraintEngine,
    ConstraintReport,
    HardConstraint,
    SoftConstraint,
)
from .context import BoundedContext, build_bounded_context
from .contracts import (
    NEUROSYMBOLIC_PACKAGE_VERSION,
    NEUROSYMBOLIC_SCHEMA_VERSION,
    NSRL_PRODUCER,
    CandidateInterpretation,
    CandidateStatus,
    ConfidenceVector,
    EpistemicStatus,
    EvidenceItem,
    InferenceMode,
    InferenceRun,
    Literal,
    LogicalTerm,
    LogicalVariable,
    ModelProposal,
    NSRLConflictRecord,
    ProofClass,
    ProofNode,
    PublicationState,
    ReasoningFinding,
    RuleClass,
    RuleDefinition,
    SemanticFact,
    VerificationLevel,
    VerificationStatus,
    canonical_json,
    sha256_json,
    sha256_text,
)
from .conflict_manager import ConflictManager
from .coordinator import NeuroSymbolicReasoningCoordinator
from .evaluation import (
    ABLATION_MECHANISMS,
    AblationReport,
    BenchmarkReport,
    ReasoningRunMetrics,
    evaluate_corpus,
    run_ablation,
    run_reasoning,
)
from .inference_queue import BoundedInferenceQueue, QueueEntry
from .learned_candidate_adapter import (
    AdaptationLedger,
    LearnedCandidateAdapter,
    ModelAdapter,
    ModelDescriptor,
    ModelInvocationError,
    RejectionRecord,
)
from .observability import CheckpointRecord, NSRLObservability, RunMetrics
from .probabilistic_adapter import (
    EvidenceSupport,
    IndependenceDeclaration,
    ProbabilisticEvidenceAdapter,
    ProbabilisticSemanticsError,
    differentiable_consistency,
)
from .proof_builder import LineageRecord, ProofBuildReport, ProofBuilder
from .proof_model import GraphIssue, ProofDAG
from .proof_verifier import IndependentProofVerifier, VerificationReport
from .registry import (
    NSRL_OPERATIONS,
    NSRL_OPERATION_IDS,
    OperationSpec,
    PHASE_STEP_RANGES,
    operation_by_id,
    operation_by_step,
)
from .request_models import ReasoningRequest, ReasoningRunResult
from .resource_policy import (
    ContinuationDecision,
    ResourceBudgets,
    ResourcePolicy,
    ResourceSnapshot,
)
from .rule_validation import (
    RuleProfile,
    RuleValidationIssue,
    RuleValidationReport,
    RuleValidator,
    build_predicate_arity_table,
    select_relevant_rules,
    validate_ruleset,
)
from .semantic_adapter import AdapterReport, SemanticReference, USGSemanticAdapter
from .symbolic_engine import (
    CycleEvent,
    Derivation,
    EngineBudgets,
    EngineReport,
    FactTable,
    SymbolicEngine,
)
from .trace import NSRLTraceBuilder
from .unification import Substitution, TypeTable, UnificationDiagnostic, unify_literals
from .workspace_publisher import (
    PublicationGates,
    PublicationReport,
    NSRLWorkspacePublisher,
)

__all__ = [
    "ABLATION_MECHANISMS",
    "AdaptationLedger",
    "AdapterReport",
    "AmbiguityGroup",
    "AmbiguityResolver",
    "AblationReport",
    "BenchmarkReport",
    "BoundedContext",
    "BoundedInferenceQueue",
    "CandidateInterpretation",
    "CandidateStatus",
    "CheckpointRecord",
    "ConfidenceVector",
    "ConflictManager",
    "ConstraintEngine",
    "ConstraintReport",
    "ContinuationDecision",
    "CycleEvent",
    "Derivation",
    "EngineBudgets",
    "EngineReport",
    "EpistemicStatus",
    "EvidenceItem",
    "EvidenceSupport",
    "FactTable",
    "GraphIssue",
    "HardConstraint",
    "IndependenceDeclaration",
    "IndependentProofVerifier",
    "InferenceMode",
    "InferenceRun",
    "LearnedCandidateAdapter",
    "LineageRecord",
    "Literal",
    "LogicalTerm",
    "LogicalVariable",
    "ModelAdapter",
    "ModelDescriptor",
    "ModelInvocationError",
    "ModelProposal",
    "NEUROSYMBOLIC_PACKAGE_VERSION",
    "NEUROSYMBOLIC_SCHEMA_VERSION",
    "NSRLAblationSwitches",
    "NSRLConflictRecord",
    "NSRLObservability",
    "NSRLPolicy",
    "NSRL_PRODUCER",
    "NSRLTraceBuilder",
    "NSRLWorkspacePublisher",
    "NSRL_OPERATIONS",
    "NSRL_OPERATION_IDS",
    "OperationSpec",
    "PHASE_STEP_RANGES",
    "ProofBuildReport",
    "ProofBuilder",
    "ProofClass",
    "ProofDAG",
    "ProofNode",
    "PublicationGates",
    "PublicationReport",
    "PublicationState",
    "QueueEntry",
    "RankingWeights",
    "ReasoningFinding",
    "ReasoningRequest",
    "ReasoningRunMetrics",
    "ReasoningRunResult",
    "RejectionRecord",
    "ResourceBudgets",
    "ResourcePolicy",
    "ResourceSnapshot",
    "RuleClass",
    "RuleDefinition",
    "RuleProfile",
    "RuleValidationIssue",
    "RuleValidationReport",
    "RuleValidator",
    "RunMetrics",
    "ScoredCandidate",
    "SemanticFact",
    "SemanticReference",
    "SoftConstraint",
    "Substitution",
    "SymbolicEngine",
    "TypeTable",
    "UnificationDiagnostic",
    "USGSemanticAdapter",
    "VerificationLevel",
    "VerificationReport",
    "VerificationStatus",
    "build_bounded_context",
    "build_predicate_arity_table",
    "canonical_json",
    "default_policy",
    "differentiable_consistency",
    "evaluate_corpus",
    "operation_by_id",
    "operation_by_step",
    "run_ablation",
    "run_reasoning",
    "select_relevant_rules",
    "sha256_json",
    "sha256_text",
    "unify_literals",
    "validate_ruleset",
]
