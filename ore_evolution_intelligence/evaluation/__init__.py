"""Evaluation layer: versioned benchmarks, sandboxed execution, independent evaluation."""
from .benchmark_manifest import (
    DIFFICULTY_TIERS, RESOURCE_STATES, SPLIT_POLICY, SPLIT_RATIONALE, TASK_FAMILIES,
    BenchmarkManifest, ScoringConfig, TaskCase, build_benchmark, build_task_suite,
    runtime_environment, split_tasks,
)
from .evaluator import (
    BudgetExhausted, EvaluationBudget, HARDCODING_MIN_TASKS, SeededRandom,
    StrategyEvaluator, task_stream_seed,
)
from .heldout_validation import (
    DEFAULT_ENVELOPE_TOLERANCES, DEFAULT_PRACTICAL_THRESHOLD, DriftReport,
    HeldoutComparison, HeldoutValidator, ValidationEnvelope,
)
from .sandbox import ExecutionTrace, GRANTED_CAPABILITIES, SandboxLimits, SandboxedStrategyExecutor
from .task_simulators import (
    ESCALATION_COMPLEXITY_GATE, FAMILY_RELEVANCE, MODULE_PROFILES,
    CoordinatorSimulator, SimulationResult,
)

__all__ = [
    # benchmarks
    "DIFFICULTY_TIERS", "RESOURCE_STATES", "SPLIT_POLICY", "SPLIT_RATIONALE",
    "TASK_FAMILIES", "BenchmarkManifest", "ScoringConfig", "TaskCase",
    "build_benchmark", "build_task_suite", "runtime_environment", "split_tasks",
    # evaluator
    "BudgetExhausted", "EvaluationBudget", "HARDCODING_MIN_TASKS", "SeededRandom",
    "StrategyEvaluator", "task_stream_seed",
    # held-out validation
    "DEFAULT_ENVELOPE_TOLERANCES", "DEFAULT_PRACTICAL_THRESHOLD", "DriftReport",
    "HeldoutComparison", "HeldoutValidator", "ValidationEnvelope",
    # sandbox
    "ExecutionTrace", "GRANTED_CAPABILITIES", "SandboxLimits", "SandboxedStrategyExecutor",
    # simulator
    "ESCALATION_COMPLEXITY_GATE", "FAMILY_RELEVANCE", "MODULE_PROFILES",
    "CoordinatorSimulator", "SimulationResult",
]
