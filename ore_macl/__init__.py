"""ORE Point 11 multi-agent cognitive coordination and selective swarm optimization."""

from .aco import (
    ACOConfig,
    ACOResult,
    AntColonyOptimizer,
    OperationGraph,
    dependency_feasible,
)
from .agents import (
    AcceptDecision,
    BaseBoundedAgent,
    CriticAgent,
    EvidenceVerificationAgent,
    ExecutionContext,
    LexicalAnalysisAgent,
    LogicalClock,
    SemanticInterpretationAgent,
    SyntacticAnalysisAgent,
    TaskExecutor,
    WallClock,
    build_default_agents,
    default_specialist_descriptors,
    deterministic_lexical_hypotheses,
    deterministic_semantic_hypothesis,
    deterministic_syntactic_hypothesis,
)
from .annotations import register_annotations
from .biology import (
    STAGE_DRIVES,
    FunctionalPopulation,
    MultiAgentCoordinationNetwork,
)
from .contracts import (
    AdjudicationDecision,
    AgentDescriptor,
    AgentLifecycleState,
    AgentMessage,
    AgentResult,
    AgentTask,
    AgentTaskKind,
    ArtifactRef,
    AutonomyLevel,
    DeliveryOutcome,
    MessageKind,
    PeerNegotiationOutcome,
    ProvenanceTrail,
    ResourceEnvelope,
    RetryPolicy,
    SchemaSpec,
    SuccessCriteria,
    TaskOutcome,
    TaskTopology,
)
from .decomposition import (
    TaskDAG,
    TaskDecomposer,
    identify_required_roles,
    separate_independent_work,
)
from .errors import (
    BudgetExhausted,
    CyclicDependencyError,
    DeadlineExceeded,
    FabricatedProvenanceError,
    MACLError,
    MARLProductionControlError,
    OutOfScopeAccess,
    PermanentAgentError,
    PolicyViolation,
    QuarantineTriggered,
    StaleRevisionError,
    TaskBlockedError,
    TransientAgentError,
)
from .ledger import LedgerEntry, PerformanceLedger, sufficient_sample_warning
from .matching import (
    AgentTaskMatcher,
    AssignmentSolver,
    EligibilityGate,
    EligibilityReport,
    MatchingConfig,
    TopologySelector,
    assign_owners,
)
from .messages import AgentMessageBus, DeliveryReceipt, make_message
from .peer import DelegationChain, PeerCoordinator, reconcile_partition
from .pipeline import (
    MACLRunConfig,
    MACLRunResult,
    record_result_to_sqlite,
    run_isolated_swarm_experiments,
    run_multi_agent_coordination,
)
from .pso import (
    ParticleSwarmOptimizer,
    PSOConfig,
    PSOResult,
    resource_policy_objective,
)
from .registry import (
    STAGE_REGISTRY,
    AgentRegistry,
    MACLStageSpec,
    assignable_states,
    classify_component,
    registry_as_dicts,
    stage_for_step,
    valid_transitions,
)
from .reliability import (
    BetaBinomialEstimate,
    ReliabilityRecord,
    reliability_report,
    score_reliability,
    uncertainty_penalty,
)
from .supervisor import (
    CoordinatorGateway,
    GoalRequest,
    GoalRunReport,
    MACLConfig,
    MACLSupervisor,
)

__all__ = [
    # Contracts
    "AdjudicationDecision", "AgentDescriptor", "AgentLifecycleState", "AgentMessage",
    "AgentResult", "AgentTask", "AgentTaskKind", "ArtifactRef", "AutonomyLevel",
    "DeliveryOutcome", "MessageKind", "PeerNegotiationOutcome", "ProvenanceTrail",
    "ResourceEnvelope", "RetryPolicy", "SchemaSpec", "SuccessCriteria",
    "TaskOutcome", "TaskTopology",
    # Registry and lifecycle
    "AgentRegistry", "MACLStageSpec", "STAGE_REGISTRY", "assignable_states",
    "classify_component", "registry_as_dicts", "stage_for_step", "valid_transitions",
    # Agents and execution
    "AcceptDecision", "BaseBoundedAgent", "CriticAgent",
    "ExecutionContext", "EvidenceVerificationAgent", "LexicalAnalysisAgent",
    "LogicalClock", "SemanticInterpretationAgent", "SyntacticAnalysisAgent",
    "TaskExecutor", "WallClock", "build_default_agents",
    "default_specialist_descriptors", "deterministic_lexical_hypotheses",
    "deterministic_semantic_hypothesis", "deterministic_syntactic_hypothesis",
    # Decomposition and matching
    "TaskDAG", "TaskDecomposer", "identify_required_roles", "separate_independent_work",
    "AgentTaskMatcher", "AssignmentSolver", "EligibilityGate", "EligibilityReport",
    "MatchingConfig", "TopologySelector", "assign_owners",
    # Messages and peer coordination
    "AgentMessageBus", "DeliveryReceipt", "make_message",
    "DelegationChain", "PeerCoordinator", "reconcile_partition",
    # Performance ledger
    "LedgerEntry", "PerformanceLedger", "sufficient_sample_warning",
    # Swarm optimization (isolated, offline only)
    "ACOConfig", "ACOResult", "AntColonyOptimizer", "OperationGraph",
    "dependency_feasible", "PSOConfig", "PSOResult", "ParticleSwarmOptimizer",
    "resource_policy_objective",
    # Reliability and biology
    "BetaBinomialEstimate", "ReliabilityRecord", "reliability_report",
    "score_reliability", "uncertainty_penalty",
    "STAGE_DRIVES", "FunctionalPopulation", "MultiAgentCoordinationNetwork",
    # Supervisor and pipeline
    "CoordinatorGateway", "GoalRequest", "GoalRunReport", "MACLConfig",
    "MACLSupervisor", "MACLRunConfig", "MACLRunResult",
    "run_multi_agent_coordination", "run_isolated_swarm_experiments",
    "record_result_to_sqlite", "register_annotations",
    # Errors
    "BudgetExhausted", "CyclicDependencyError", "DeadlineExceeded",
    "FabricatedProvenanceError", "MACLError", "MARLProductionControlError",
    "OutOfScopeAccess", "PermanentAgentError", "PolicyViolation",
    "QuarantineTriggered", "StaleRevisionError", "TaskBlockedError",
    "TransientAgentError",
]
