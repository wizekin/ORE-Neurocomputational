"""ORE Point 9 - Evolutionary Intelligence, Genetic Programming and Quality-Diversity
(integrated unit).

An isolated Adaptive Cognitive Strategy Evolution Layer (ORE Point 9): it discovers,
evaluates, preserves and selects improved cognitive strategies. Evolution produces
candidates, not authority — no evolved program, policy or USG rule becomes active
merely because it scores well on its training benchmark.

Implemented pipeline (section 11):

Bounded Evolutionary Optimization -> Sandboxed Evaluation -> MAP-Elites Archive ->
Independent Verification -> Versioned Validated Strategy Registry -> Contextual
Coordinator Selection -> Live Monitoring and Rollback.

Design invariants:

- Genotypes are data interpreted by a capability-enforcing sandbox; unrestricted Python
  execution is structurally impossible (steps 8, 33).
- Hard correctness and safety conditions are gates, never small penalties (step 21).
- The held-out test set is sealed and only opened post-search (step 10).
- The baseline is an immutable, replayable rollback target (step 3).
- Strict lifecycle states: GENERATED, SANDBOX_TESTED, VALIDATED, APPROVED, CANARY,
  ACTIVE, DEPRECATED, REJECTED; production only selects approved strategies (s. 11).
- Evolution failures never interrupt ordinary ORE tasks (acceptance test 36).
"""
from .domain import (
    ACTION_VOCABULARY, LIFECYCLE_STATES, ORE_MODULES, PRODUCTION_ELIGIBLE_STATES,
    PROHIBITED_ACTIONS, ActionCall, AggregateMetrics, BehaviorDescriptor,
    DESCRIPTOR_DIMENSIONS, DescriptorCalibration, EvaluationRecord, GateDecision,
    HardConstraints, LineageRecord, NormalizationScales, OUTCOME_KINDS,
    ParameterDefinition, ParameterGenome, PolicyTableGenome, ScalarFitness,
    StrategyCandidate, TaskOutcomeRecord, canonical_json, content_digest,
    default_calibration, default_search_space, t95,
)
from .evaluation import (
    BenchmarkManifest, BudgetExhausted, CoordinatorSimulator, EvaluationBudget,
    GRANTED_CAPABILITIES, HeldoutComparison, HeldoutValidator, SandboxLimits,
    SandboxedStrategyExecutor, ScoringConfig, SeededRandom, StrategyEvaluator,
    TaskCase, ValidationEnvelope, build_benchmark, build_task_suite, split_tasks,
)
from .search import (
    CMAESAdapter, CMAESResult, DEFAULT_GRAMMAR, EvolutionSearchReport,
    EvolutionStrategyOptimizer, GeneticOptimizer, GeneticSearchConfig,
    NoveltyArchive, NoveltyGuidedSelector, ParetoArchive,
    PolicyIRGenome, TypedGrammar, crossover_genomes, dominates, mutate_genome,
    pareto_front_is_consistent, random_genome, subtree_crossover, subtree_mutation,
)
from .archive import (
    ArchiveEntry, ArchiveMetrics, ArchiveUpdate, CellIndex, MapElitesArchive,
    MapElitesSearch, QDSearchConfig, QDSearchReport, archive_metrics,
)
from .verification import (
    ApprovalRecord, CanaryPlan, PromotionPolicy, RegressionGate, RegressionPolicy,
    RejectionRecord, SemanticRuleCheck, StaticPolicyCheck,
)
from .registry import (
    ArtifactStore, RegistryRecord, RollbackTarget, StrategyArtifact,
    StrategyManifest, ValidatedStrategyRegistry, Versioning,
)
from .observability import (
    EVENT_KINDS, EvolutionEventRecorder, LineageTracker, SqliteEvolutionRepository,
)
from .orchestration import (
    AdmissionController, EvolutionBudgetSpec, EvolutionObjective,
    EvolutionRunReport, ResourceBudget, RolloutController,
    StrategyEvolutionCoordinator, WorkspaceEvolutionPublisher,
    evolution_workspace_permissions,
)

__version__ = "1.0.0"

__all__ = [
    # domain
    "ACTION_VOCABULARY", "LIFECYCLE_STATES", "ORE_MODULES",
    "PRODUCTION_ELIGIBLE_STATES", "PROHIBITED_ACTIONS", "ActionCall",
    "AggregateMetrics", "BehaviorDescriptor", "DESCRIPTOR_DIMENSIONS",
    "DescriptorCalibration", "EvaluationRecord", "GateDecision", "HardConstraints",
    "LineageRecord", "NormalizationScales", "OUTCOME_KINDS", "ParameterDefinition",
    "ParameterGenome", "PolicyTableGenome", "ScalarFitness", "StrategyCandidate",
    "TaskOutcomeRecord", "canonical_json", "content_digest", "default_calibration",
    "default_search_space", "t95",
    # evaluation
    "BenchmarkManifest", "BudgetExhausted", "CoordinatorSimulator",
    "EvaluationBudget", "GRANTED_CAPABILITIES", "HeldoutComparison",
    "HeldoutValidator", "SandboxLimits", "SandboxedStrategyExecutor", "ScoringConfig",
    "SeededRandom", "StrategyEvaluator", "TaskCase", "ValidationEnvelope",
    "build_benchmark", "build_task_suite", "split_tasks",
    # search
    "CMAESAdapter", "CMAESResult", "DEFAULT_GRAMMAR", "EvolutionSearchReport",
    "EvolutionStrategyOptimizer", "GeneticOptimizer", "GeneticSearchConfig",
    "NoveltyArchive", "NoveltyGuidedSelector",
    "ParetoArchive", "PolicyIRGenome", "TypedGrammar", "crossover_genomes", "dominates",
    "mutate_genome", "pareto_front_is_consistent", "random_genome",
    "subtree_crossover", "subtree_mutation",
    # archive
    "ArchiveEntry", "ArchiveMetrics", "ArchiveUpdate", "CellIndex",
    "MapElitesArchive", "MapElitesSearch", "QDSearchConfig", "QDSearchReport",
    "archive_metrics",
    # verification
    "ApprovalRecord", "CanaryPlan", "PromotionPolicy", "RegressionGate",
    "RegressionPolicy", "RejectionRecord", "SemanticRuleCheck", "StaticPolicyCheck",
    # registry
    "ArtifactStore", "RegistryRecord", "RollbackTarget", "StrategyArtifact",
    "StrategyManifest", "ValidatedStrategyRegistry", "Versioning",
    # observability
    "EVENT_KINDS", "EvolutionEventRecorder", "LineageTracker",
    "SqliteEvolutionRepository",
    # orchestration
    "AdmissionController", "EvolutionBudgetSpec", "EvolutionObjective",
    "EvolutionRunReport", "ResourceBudget", "RolloutController",
    "StrategyEvolutionCoordinator", "WorkspaceEvolutionPublisher",
    "evolution_workspace_permissions",
    "__version__",
]
