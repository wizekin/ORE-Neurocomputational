"""ore_rl package — reinforcement learning, hierarchical RL, and adaptive policy optimization.

Implements the "ORE Integrated Algorithm — Point 10" specification: a
subordinate, constraint-respecting learning layer that improves ORE's
decisions about which cognitive operation to execute, in what order, at
what depth, and when to stop.

Design boundaries (Point 10, section 1.2):
    * The Main Brain Coordinator (``CognitiveTaskCoordinator``) retains
      final authority; policies only propose.
    * AdaptiveResourceCoordinator / resource planning keeps hard budget
      enforcement outside the learned objective.
    * USG knowledge is consulted read-only; the policy can never rewrite it.
    * Formal verification gates (e.g. STOP_SUCCESS requires a verified
      conclusion) are deterministic, never learned.
    * Deployment starts in shadow mode; promotion requires measured evidence.

The package is numpy/stdlib only, matching the repository dependencies.
"""
from __future__ import annotations

from .contracts import (
    OUTCOME_STATUSES,
    RL_SCHEMA_VERSION,
    ActionOutcome,
    ActionSpecRecord,
    ActionType,
    OREAction,
    OREState,
    OREReward,
    ORETransition,
    OutcomeStatus,
    ParameterSpec,
    PolicyDecision,
    PolicyAuthority,
    RankedProposal,
    ResourceEstimate,
    RiskLevel,
    UnknownSchemaVersionError,
    canonical_json,
    require_schema_version,
    stable_hash,
)
from .state_encoder import (
    ENCODER_FEATURE_GROUPS,
    StateEncoder,
    StateEncoderVersion,
    StateSpaceSpec,
)
from .action_catalog import ActionCatalog, CatalogAction
from .safety import (
    AuthorityPolicy,
    ConstraintKind,
    DecisionValidation,
    FeasibleActionSet,
    HardConstraintGate,
    ResourceBudgetState,
    SafetyLedger,
    SafetyViolation,
)
from .reward import (
    AttributionMethod,
    CostModel,
    EpisodeAudit,
    EvidenceQualityEvaluator,
    EvaluatorIdentity,
    QualityScore,
    RewardHackingGuard,
    RewardSpec,
    RewardWeights,
    TaskQualityEvaluator,
    TerminalOutcome,
    attribute_episode_rewards,
    calibrate_reward_scale,
    compute_step_reward,
    compute_terminal_reward,
    recompute_total,
    sensitivity_analysis,
)
from .trajectory_store import (
    BanditTuple,
    DatasetManifest,
    SplitDefinition,
    SplitRule,
    TransitionTuple,
    TrajectoryStore,
)
from .environment import (
    ENVIRONMENT_VERSION,
    EnvironmentBudget,
    OREEnvironment,
    TaskInstance,
    TaskSampler,
)
from .modules import (
    ALL_CAPABILITIES,
    MODULE_IDS,
    build_example_backends,
    build_module_manifests,
    register_rl_contracts,
    register_rl_example_modules,
)
from .policies.base import (
    DeterministicBaselinePolicy,
    PolicyMetadata,
    PolicyService,
    PolicyServiceError,
    ProgressBoundaryPolicy,
    RandomEligiblePolicy,
    timed_proposal,
)
from .policies.bandit import (
    EmpiricalBandit,
    LinUCBPolicy,
    ThompsonSamplingPolicy,
    UCBPolicy,
)
from .policies.dqn import (
    DQNConfig,
    DQNPolicy,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from .policies.ppo import PPOConfig, PPOPolicy, compute_gae, masked_softmax
from .policies.options import (
    CognitiveOption,
    HierarchicalOptionPolicy,
    OptionContract,
    OptionLibrary,
    OptionLoopDetector,
    OptionRuntime,
)
from .policies.offline import (
    BehaviorCloningPolicy,
    CQLPolicy,
    IQLPolicy,
    OfflineTrainer,
)
from .constrained import (
    ConstrainedPolicyWrapper,
    LagrangianOptimizer,
    ResourceConstraint,
    ResourceConstraintSet,
)
from .world_model import EmpiricalWorldModel, PredictedOutcome
from .evaluation import (
    BenchmarkSuite,
    ComparisonReport,
    EpisodeRecord,
    EvaluationReport,
    MetricSnapshot,
    PromotionThresholds,
    bootstrap_improvement_ci,
    compare_policies,
    evaluate_policy,
)
from .registry import (
    PolicyRecord,
    PolicyRegistry,
    PolicyStatus,
    PromotionApproval,
)
from .controller_adapter import (
    DecisionBoundary,
    DecisionBoundaryPolicy,
    DecisionEvent,
    LimitedAuthorityController,
    ModulePreferenceBridge,
    RLPolicyService,
    ShadowDecisionRecord,
    ShadowModeController,
    StopDecisionBridge,
)

__version__ = "1.0.0"

__all__ = [
    "RL_SCHEMA_VERSION",
    "OUTCOME_STATUSES",
    "ActionOutcome",
    "ActionSpecRecord",
    "ActionType",
    "OREAction",
    "OREState",
    "OREReward",
    "ORETransition",
    "OutcomeStatus",
    "ParameterSpec",
    "PolicyDecision",
    "PolicyAuthority",
    "RankedProposal",
    "ResourceEstimate",
    "RiskLevel",
    "UnknownSchemaVersionError",
    "canonical_json",
    "require_schema_version",
    "stable_hash",
    "ENCODER_FEATURE_GROUPS",
    "StateEncoder",
    "StateEncoderVersion",
    "StateSpaceSpec",
    "ActionCatalog",
    "CatalogAction",
    "AuthorityPolicy",
    "ConstraintKind",
    "DecisionValidation",
    "FeasibleActionSet",
    "HardConstraintGate",
    "ResourceBudgetState",
    "SafetyLedger",
    "SafetyViolation",
    "AttributionMethod",
    "CostModel",
    "EpisodeAudit",
    "EvidenceQualityEvaluator",
    "EvaluatorIdentity",
    "QualityScore",
    "RewardHackingGuard",
    "RewardSpec",
    "RewardWeights",
    "TaskQualityEvaluator",
    "TerminalOutcome",
    "attribute_episode_rewards",
    "calibrate_reward_scale",
    "compute_step_reward",
    "compute_terminal_reward",
    "recompute_total",
    "sensitivity_analysis",
    "BanditTuple",
    "DatasetManifest",
    "SplitDefinition",
    "SplitRule",
    "TransitionTuple",
    "TrajectoryStore",
    "ENVIRONMENT_VERSION",
    "EnvironmentBudget",
    "OREEnvironment",
    "TaskInstance",
    "TaskSampler",
    "ALL_CAPABILITIES",
    "MODULE_IDS",
    "build_example_backends",
    "build_module_manifests",
    "register_rl_contracts",
    "register_rl_example_modules",
    "DeterministicBaselinePolicy",
    "PolicyMetadata",
    "PolicyService",
    "PolicyServiceError",
    "ProgressBoundaryPolicy",
    "RandomEligiblePolicy",
    "timed_proposal",
    "EmpiricalBandit",
    "LinUCBPolicy",
    "ThompsonSamplingPolicy",
    "UCBPolicy",
    "DQNConfig",
    "DQNPolicy",
    "PrioritizedReplayBuffer",
    "ReplayBuffer",
    "PPOConfig",
    "PPOPolicy",
    "compute_gae",
    "masked_softmax",
    "CognitiveOption",
    "HierarchicalOptionPolicy",
    "OptionContract",
    "OptionLibrary",
    "OptionLoopDetector",
    "OptionRuntime",
    "BehaviorCloningPolicy",
    "CQLPolicy",
    "IQLPolicy",
    "OfflineTrainer",
    "ConstrainedPolicyWrapper",
    "LagrangianOptimizer",
    "ResourceConstraint",
    "ResourceConstraintSet",
    "EmpiricalWorldModel",
    "PredictedOutcome",
    "BenchmarkSuite",
    "ComparisonReport",
    "EpisodeRecord",
    "EvaluationReport",
    "MetricSnapshot",
    "PromotionThresholds",
    "bootstrap_improvement_ci",
    "compare_policies",
    "evaluate_policy",
    "PolicyRecord",
    "PolicyRegistry",
    "PolicyStatus",
    "PromotionApproval",
    "DecisionBoundary",
    "DecisionBoundaryPolicy",
    "DecisionEvent",
    "LimitedAuthorityController",
    "ModulePreferenceBridge",
    "RLPolicyService",
    "ShadowDecisionRecord",
    "ShadowModeController",
    "StopDecisionBridge",
]
