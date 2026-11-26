"""Bounded coordination policy for ORE blackboard cognitive coordination (Algorithm 1).

Every budget below is a hard cap. The defaults reuse the existing workspace
scheduler's bounds (8 workers, 8 rounds, 64 activations, 10-second timeouts) so the
blackboard coordinator never widens any limit that ``ore_workspace.ModuleScheduler``
already enforces; task profiles may choose lower limits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CycleBudgets:
    """Hard caps for loops, parallelism, retries, and resource consumption (Steps 15, 24, 26)."""

    max_rounds: int = 8
    max_workers: int = 8
    max_activations: int = 64
    max_module_calls: int = 64
    max_replans: int = 2
    max_retries: int = 3
    max_time_seconds: float | None = 60.0
    max_memory_mb: float | None = None
    default_timeout_seconds: float = 10.0
    event_delivery_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        """Reject non-positive or unbounded budgets."""
        positive_ints = (
            self.max_rounds, self.max_workers, self.max_activations,
            self.max_module_calls, self.max_replans, self.max_retries,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("integer budgets must be positive")
        if self.default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        if self.event_delivery_timeout_seconds <= 0:
            raise ValueError("event_delivery_timeout_seconds must be positive")
        if self.max_time_seconds is not None and (
            not math.isfinite(self.max_time_seconds) or self.max_time_seconds <= 0
        ):
            raise ValueError("max_time_seconds must be finite and positive when supplied")
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive when supplied")


@dataclass(frozen=True, slots=True)
class ComparisonCoefficients:
    """Task-profile coefficients for the proposed module comparison score (Step 14).

    ``U_m = alpha*R_m + beta*IG_m + gamma*G_m + delta*Q_m - lambda*L_m
    - mu*M_m - nu*K_m``. These are task-profile parameters, not universal
    constants, and the resulting score is explicitly a heuristic: it is not a
    calibrated probability or a validated information-theoretic optimum.
    """

    alpha: float = 0.4
    beta: float = 0.2
    gamma: float = 0.2
    delta: float = 0.1
    lam: float = 0.05
    mu: float = 0.025
    nu: float = 0.05

    def __post_init__(self) -> None:
        """Require finite, non-negative coefficients."""
        values = (
            self.alpha, self.beta, self.gamma, self.delta,
            self.lam, self.mu, self.nu,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("comparison coefficients must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class StagnationPolicy:
    """Proposed stagnation test parameters (Step 24).

    This is a proposed engineering heuristic for stopping unproductive cycles;
    it never overrides critical goal requirements.
    """

    epsilon: float = 1e-6
    patience_rounds: int = 2


@dataclass(frozen=True, slots=True)
class AblationSwitches:
    """Integration mechanisms that evaluation harnesses may disable one at a time (Step 30).

    These switches exist for ablation measurement only. The default configuration
    enables every integration link.
    """

    selective_subscriptions: bool = True
    resource_aware_choice: bool = True
    usg_retrieval: bool = True
    hypothesis_conflict_handling: bool = True
    dependency_deferral: bool = True
    provenance_checks: bool = True
    bounded_parallelism: bool = True

    def disabled_names(self) -> tuple[str, ...]:
        """Return the names of the mechanisms currently switched off."""
        return tuple(
            name for name in (
                "selective_subscriptions", "resource_aware_choice", "usg_retrieval",
                "hypothesis_conflict_handling", "dependency_deferral", "provenance_checks",
                "bounded_parallelism",
            ) if not getattr(self, name)
        )


@dataclass(frozen=True, slots=True)
class BlackboardPolicy:
    """Complete coordination policy for one blackboard task run."""

    budgets: CycleBudgets = field(default_factory=CycleBudgets)
    coefficients: ComparisonCoefficients = field(default_factory=ComparisonCoefficients)
    stagnation: StagnationPolicy = field(default_factory=StagnationPolicy)
    ablations: AblationSwitches = field(default_factory=AblationSwitches)
    goal_threshold: float = 0.75
    goal_weights: Mapping[str, float] | None = None
    critical_requirements: Mapping[str, tuple[str, ...]] | None = None
    allow_uncertain_completion: bool = True
    deterministic_seed: int = 0
    retry_backoff_initial_seconds: float = 0.0
    retry_backoff_cap_seconds: float = 1.0
    retry_backoff_jitter_seconds: float = 0.0
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_seconds: float = 60.0
    reliability_prior_alpha: float = 2.0
    reliability_prior_beta: float = 2.0
    cost_moving_average_alpha: float = 0.2
    enable_logging: bool = False
    enable_resource_history: bool = False
    resource_history_path: str | None = None
    logger_database_path: str | None = None
    declared_module_estimates: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate policy bounds and normalized priorities."""
        if not 0.0 <= self.goal_threshold <= 1.0:
            raise ValueError("goal_threshold must be in [0, 1]")
        if self.goal_weights is not None and any(
            value < 0 for value in self.goal_weights.values()
        ):
            raise ValueError("goal weights cannot be negative")
        if self.deterministic_seed < 0:
            raise ValueError("deterministic_seed cannot be negative")
        if self.retry_backoff_initial_seconds < 0 or self.retry_backoff_cap_seconds <= 0:
            raise ValueError("backoff initial must be non-negative and cap positive")
        if self.retry_backoff_jitter_seconds < 0:
            raise ValueError("backoff jitter must be non-negative")
        if self.circuit_breaker_threshold <= 0:
            raise ValueError("circuit_breaker_threshold must be positive")
        if self.circuit_breaker_cooldown_seconds <= 0:
            raise ValueError("circuit_breaker_cooldown_seconds must be positive")
        if self.reliability_prior_alpha <= 0 or self.reliability_prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        if not 0.0 < self.cost_moving_average_alpha <= 1.0:
            raise ValueError("cost moving-average alpha must be in (0, 1]")


def default_policy() -> BlackboardPolicy:
    """Return the default coordination policy instance."""
    return BlackboardPolicy()


__all__ = [
    "AblationSwitches",
    "BlackboardPolicy",
    "ComparisonCoefficients",
    "CycleBudgets",
    "StagnationPolicy",
    "default_policy",
]
