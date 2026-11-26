"""Coordination policy for the cognitive control layer (Point 02 §5, §8).

All coefficients below are task-profile configuration, not universal constants:
retrieval weights (alpha/beta/gamma/delta/lambda), utility weights (w_g, w_i,
w_c, w_t, w_r, w_f), stagnation windows, ambiguity margins, and procedure
promotion gates. Defaults are conservative so tests remain deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RetrievalCoefficients:
    """Weights for the retrieval score ``R_i`` (Point 02 §5.3)."""

    alpha: float = 0.2  # base-level activation B_i
    beta: float = 0.4  # cue association S_i
    gamma: float = 0.25  # task-specific relevance Q_i
    delta: float = 0.1  # provenance/quality support P_i
    lam: float = 0.05  # retrieval cost C_i

    def __post_init__(self) -> None:
        """Validate that retrieval coefficients are finite."""
        for name in ("alpha", "beta", "gamma", "delta", "lam"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    """Weights for the normalized operator score ``U(o)`` (Point 02 §5.4)."""

    w_g: float = 0.4  # expected goal progress
    w_i: float = 0.25  # information gain
    w_c: float = 0.15  # confidence / quality
    w_t: float = 0.15  # time cost
    w_r: float = 0.1  # risk cost
    w_f: float = 0.05  # resource footprint

    def __post_init__(self) -> None:
        """Validate that utility weights are finite."""
        for name in ("w_g", "w_i", "w_c", "w_t", "w_r", "w_f"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class CycleBudgets:
    """Bounded cycle, action, retrieval, retry, and deadline budgets (§8)."""

    max_cycles: int = 12
    max_operator_applications: int = 24
    max_retrievals: int = 16
    retrieval_budget: float = 8.0
    max_retries: int = 3
    max_children: int = 4
    max_depth: int = 2
    child_operation_budget: int = 8
    stagnation_window: int = 2
    stagnation_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        """Validate that every budget is a positive integer or finite scalar."""
        for name in (
            "max_cycles", "max_operator_applications", "max_retrievals",
            "max_retries", "max_children", "max_depth", "child_operation_budget",
            "stagnation_window",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("retrieval_budget", "stagnation_epsilon"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class AmbiguityPolicy:
    """Shortlist and winner-separation thresholds (Point 02 Step 16)."""

    margin: float = 0.1
    shortlist_size: int = 3
    require_winner_margin: bool = True

    def __post_init__(self) -> None:
        """Validate the ambiguity policy."""
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("ambiguity margin must be finite and non-negative")
        if self.shortlist_size < 1:
            raise ValueError("shortlist_size must be >= 1")


@dataclass(frozen=True, slots=True)
class ProcedurePolicy:
    """Procedure-learning promotion gates (Point 02 Phase H, §7)."""

    min_repeatable_runs: int = 2
    held_out_success_rate: float = 0.7
    held_out_examples: int = 3
    counterexample_checks: int = 2
    max_resource_regression: float = 1.5
    monitoring_window: int = 4
    disable_success_rate: float = 0.4
    high_risk_requires_manual_approval: bool = True

    def __post_init__(self) -> None:
        """Validate the procedure promotion gates."""
        for name in ("min_repeatable_runs", "held_out_examples", "counterexample_checks"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("held_out_success_rate", "disable_success_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("max_resource_regression",):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(f"{name} must be >= 1")
        if not isinstance(self.monitoring_window, int) or self.monitoring_window < 1:
            raise ValueError("monitoring_window must be a positive integer")


@dataclass(frozen=True, slots=True)
class RetrievalDepthPolicy:
    """Marginal-gain-versus-cost stopping policy (Point 02 Step 17)."""

    cache_gain: float = 0.5
    cache_cost: float = 0.1
    index_gain: float = 0.8
    index_cost: float = 0.3
    resource_gain: float = 1.0
    resource_cost: float = 0.9

    def levels(self) -> tuple[tuple[str, float, float], ...]:
        """Return the (level, expected gain, cost) ladder in escalation order."""
        return (
            ("local_cache", self.cache_gain, self.cache_cost),
            ("structured_index", self.index_gain, self.index_cost),
            ("full_resource", self.resource_gain, self.resource_cost),
        )

    def __post_init__(self) -> None:
        """Validate that depth gains and costs are finite and non-negative."""
        for name in (
            "cache_gain", "cache_cost", "index_gain", "index_cost",
            "resource_gain", "resource_cost",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass
class ControlPolicy:
    """Full coordination policy for one control-layer deployment."""

    budgets: CycleBudgets = field(default_factory=CycleBudgets)
    retrieval: RetrievalCoefficients = field(default_factory=RetrievalCoefficients)
    utility: UtilityWeights = field(default_factory=UtilityWeights)
    ambiguity: AmbiguityPolicy = field(default_factory=AmbiguityPolicy)
    procedures: ProcedurePolicy = field(default_factory=ProcedurePolicy)
    retrieval_depth: RetrievalDepthPolicy = field(default_factory=RetrievalDepthPolicy)
    activation_decay: float = 0.5
    activation_epsilon: float = 1e-3
    goal_threshold: float = 0.75
    eu_epsilon: float = 0.05
    deterministic_selection: bool = True
    enable_resource_history: bool = True
    resource_history_path: str | Path | None = None
    reliability_prior_alpha: float = 2.0
    reliability_prior_beta: float = 2.0
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_seconds: float = 60.0
    cost_moving_average_alpha: float = 0.3
    declared_operator_estimates: Mapping[str, Any] = field(default_factory=dict)
    metaconfidence_threshold: float = 0.35
    stale_evidence_revision_gap: int = 3

    def __post_init__(self) -> None:
        """Validate scalar policy parameters."""
        for name in (
            "activation_decay", "activation_epsilon", "goal_threshold", "eu_epsilon",
            "reliability_prior_alpha", "reliability_prior_beta",
            "circuit_breaker_cooldown_seconds", "cost_moving_average_alpha",
            "metaconfidence_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.activation_decay > 1.0:
            raise ValueError("activation_decay must be bounded (0, 1]")
        if not isinstance(self.circuit_breaker_threshold, int) or (
            self.circuit_breaker_threshold < 1
        ):
            raise ValueError("circuit_breaker_threshold must be a positive integer")
        if not isinstance(self.stale_evidence_revision_gap, int) or (
            self.stale_evidence_revision_gap < 1
        ):
            raise ValueError("stale_evidence_revision_gap must be a positive integer")


def default_policy() -> ControlPolicy:
    """Return the conservative default control policy."""
    return ControlPolicy()


__all__ = [
    "AmbiguityPolicy",
    "ControlPolicy",
    "CycleBudgets",
    "ProcedurePolicy",
    "RetrievalCoefficients",
    "RetrievalDepthPolicy",
    "UtilityWeights",
    "default_policy",
]
