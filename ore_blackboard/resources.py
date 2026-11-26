"""Adaptive resource coordination for blackboard module selection (Steps 14, 26, 31).

This module reuses the existing ``ore_contracts`` resource machinery verbatim:
SQLite-persisted reliability history with Beta(2, 2)-prior reliability estimates,
exponential moving averages for measured latency/memory/CPU costs, and circuit
breaker state. Information gain is heuristic, as documented by the repository,
and every score this module derives is labelled as such.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from ore_contracts.resource_coordination import (
    ModuleReliabilityTracker,
    ModuleResourceEstimator,
    SQLiteResourceHistory,
)
from ore_contracts.resource_profiles import ModuleResourceEstimate

from .config import BlackboardPolicy, ComparisonCoefficients
from .contracts import sha256_json

HEURISTIC_INFORMATION_GAIN = "heuristic_information_gain"


@dataclass(frozen=True, slots=True)
class ComparisonScore:
    """Report the proposed module comparison score with its labelled terms (Step 14)."""

    module_id: str
    score: float
    relevance: float
    heuristic_information_gain: float
    goal_contribution: float
    reliability: float
    latency_cost: float
    memory_cost: float
    risk_cost: float
    coefficients: ComparisonCoefficients
    heuristic: bool = True

    @property
    def definition(self) -> str:
        """Expose the score definition so rankings remain interpretable."""
        return (
            "U_m = alpha*R_m + beta*IG_m + gamma*G_m + delta*Q_m"
            " - lambda*L_m - mu*M_m - nu*K_m; heuristic task-profile score,"
            " not a calibrated probability or validated information-theoretic optimum"
        )


def compute_comparison_score(
    module_id: str,
    *,
    relevance: float,
    heuristic_information_gain: float,
    goal_contribution: float,
    reliability: float,
    latency_cost: float,
    memory_cost: float,
    risk_cost: float,
    coefficients: ComparisonCoefficients,
) -> ComparisonScore:
    """Compute the proposed comparison score ``U_m`` (Step 14).

    All inputs must be finite; costs are normalized to ``[0, 1]`` by the caller.
    The result is used only when a task profile needs a comparison beyond what
    the existing coordinator policy already provides; mandatory capability
    requirements can never be skipped because an optional module scores higher.
    """
    values = (relevance, heuristic_information_gain, goal_contribution, reliability,
              latency_cost, memory_cost, risk_cost)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("comparison score inputs must be finite")
    score = (
        coefficients.alpha * relevance
        + coefficients.beta * heuristic_information_gain
        + coefficients.gamma * goal_contribution
        + coefficients.delta * reliability
        - coefficients.lam * latency_cost
        - coefficients.mu * memory_cost
        - coefficients.nu * risk_cost
    )
    return ComparisonScore(
        module_id=module_id,
        score=score,
        relevance=relevance,
        heuristic_information_gain=heuristic_information_gain,
        goal_contribution=goal_contribution,
        reliability=reliability,
        latency_cost=latency_cost,
        memory_cost=memory_cost,
        risk_cost=risk_cost,
        coefficients=coefficients,
    )


def normalized_cost(estimate: ModuleResourceEstimate, budget_ms: float) -> float:
    """Normalize a measured cost estimate into ``[0, 1]`` against a latency budget."""
    if budget_ms <= 0:
        raise ValueError("budget must be positive")
    return min(1.0, max(0.0, estimate.expected_latency_ms / budget_ms))


def exponential_backoff(
    attempt: int, *, initial_seconds: float, cap_seconds: float, jitter_seconds: float,
) -> float:
    """Return ``t_n = min(t_cap, t_0 * 2^n) + jitter`` for transient retries (Step 26)."""
    if attempt < 0:
        raise ValueError("attempt cannot be negative")
    base = min(cap_seconds, initial_seconds * (2 ** attempt))
    return base + jitter_seconds


class BlackboardResourceCoordinator:
    """Own reliability history, cost estimates, and circuit-breaker gating (Steps 14, 26)."""

    def __init__(self, policy: BlackboardPolicy, *, task_family: str) -> None:
        """Bind the resource history stores to the task family and policy."""
        self.policy = policy
        self.task_family = task_family
        self._temporary_directory = None
        if policy.resource_history_path is not None:
            database_path: str | Path = Path(policy.resource_history_path)
        elif policy.enable_resource_history:
            import tempfile

            self._temporary_directory = tempfile.TemporaryDirectory(prefix="ore-blackboard-")
            database_path = Path(self._temporary_directory.name) / "resource_history.sqlite3"
        else:
            import tempfile

            self._temporary_directory = tempfile.TemporaryDirectory(prefix="ore-blackboard-")
            database_path = Path(self._temporary_directory.name) / "resource_history.sqlite3"
        self.history = SQLiteResourceHistory(database_path)
        self.reliability = ModuleReliabilityTracker(
            self.history,
            prior_alpha=policy.reliability_prior_alpha,
            prior_beta=policy.reliability_prior_beta,
            circuit_breaker_threshold=policy.circuit_breaker_threshold,
            circuit_breaker_cooldown_seconds=policy.circuit_breaker_cooldown_seconds,
        )
        self.estimator = ModuleResourceEstimator(
            self.history,
            self.reliability,
            policy.declared_module_estimates,
            moving_average_alpha=policy.cost_moving_average_alpha,
        )

    def circuit_open(self, module_id: str) -> bool:
        """Return whether the module's circuit breaker is currently open (Step 26)."""
        return self.reliability.is_circuit_open(self.task_family, module_id)

    def reliability_probability(self, module_id: str) -> float:
        """Return the Beta-posterior mean reliability estimate for one module."""
        return self.reliability.probability(self.task_family, module_id)

    def estimate(self, module_id: str, expected_information_gain: float) -> ModuleResourceEstimate:
        """Return a resource estimate with its declared provenance label (Step 14)."""
        return self.estimator.estimate(self.task_family, module_id, expected_information_gain)

    def record_outcome(self, module_id: str, success: bool) -> None:
        """Record one success or failure outcome into the shared reliability history."""
        self.reliability.record_outcome(self.task_family, module_id, success)

    def record_measurement(
        self, module_id: str, *, latency_ms: float, memory_mb: float | None = None,
        cpu_cost: float | None = None,
    ) -> None:
        """Record measured execution costs with an exponential moving average (Step 31)."""
        self.estimator.record_measurement(
            self.task_family, module_id, latency_ms=latency_ms,
            memory_mb=memory_mb, cpu_cost=cpu_cost,
        )

    def close(self) -> None:
        """Release the resource-history database without deleting task workspace state."""
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def profile_reference(self) -> str:
        """Return a stable reference to this coordinator's policy provenance."""
        return quote(
            sha256_json({
                "task_family": self.task_family,
                "circuit_breaker_threshold": self.policy.circuit_breaker_threshold,
                "circuit_breaker_cooldown_seconds": self.policy.circuit_breaker_cooldown_seconds,
                "reliability_prior": [
                    self.policy.reliability_prior_alpha, self.policy.reliability_prior_beta,
                ],
            }),
            safe="",
        )


__all__ = [
    "BlackboardResourceCoordinator",
    "ComparisonScore",
    "HEURISTIC_INFORMATION_GAIN",
    "compute_comparison_score",
    "exponential_backoff",
    "normalized_cost",
]
