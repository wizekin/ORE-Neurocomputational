"""Resource budgets and the Section 6.7 continuation utility (resource policy).

``U_continue = E[ΔI + ΔG] / (C_cpu + C_mem + C_latency + C_risk)`` decides
whether further inference is worth its measured cost. Continuation is allowed
only when the utility clears the policy threshold *and* all hard budgets still
have headroom (Section 6.7; Section 2: bound rule fires, recursion, depth,
candidate count, CPU, memory and wall-clock time).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ResourceBudgets:
    """Hard resource caps for one inference run (Section 2)."""

    max_rule_fires: int = 256
    max_depth: int = 8
    max_candidates: int = 4096
    max_model_calls: int = 16
    wall_clock_seconds: float = 30.0
    max_memory_mb: float = 512.0

    def __post_init__(self) -> None:
        """Reject non-positive caps."""
        positive = (
            self.max_rule_fires, self.max_depth, self.max_candidates,
            self.max_model_calls,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("integer resource budgets must be positive")
        if self.wall_clock_seconds <= 0 or self.max_memory_mb <= 0:
            raise ValueError("time and memory budgets must be positive")

    def to_dict(self) -> dict[str, float]:
        """Return the JSON-compatible record."""
        return {
            "max_rule_fires": self.max_rule_fires,
            "max_depth": self.max_depth,
            "max_candidates": self.max_candidates,
            "max_model_calls": self.max_model_calls,
            "wall_clock_seconds": self.wall_clock_seconds,
            "max_memory_mb": self.max_memory_mb,
        }


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Measured consumption so far in the current run (used for budget checks)."""

    rule_fires: int = 0
    max_depth_reached: int = 0
    candidates: int = 0
    model_calls: int = 0
    elapsed_seconds: float = 0.0
    memory_mb: float = 0.0

    def __post_init__(self) -> None:
        """Reject negative measurements."""
        values = (
            self.rule_fires, self.max_depth_reached, self.candidates,
            self.model_calls, self.elapsed_seconds, self.memory_mb,
        )
        if any(value < 0 for value in values):
            raise ValueError("resource snapshot values cannot be negative")

    def to_dict(self) -> dict[str, float]:
        """Return the JSON-compatible record."""
        return {
            "rule_fires": self.rule_fires,
            "max_depth_reached": self.max_depth_reached,
            "candidates": self.candidates,
            "model_calls": self.model_calls,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "memory_mb": round(self.memory_mb, 3),
        }


@dataclass(frozen=True, slots=True)
class ContinuationDecision:
    """Outcome of one Section 6.7 continuation evaluation."""

    utility: float
    threshold: float
    decision: str  # continue | stop_utility | stop_budget
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "utility": round(self.utility, 6),
            "threshold": self.threshold,
            "decision": self.decision,
            "reason": self.reason,
        }


class ResourcePolicy:
    """Evaluate continuation utility under hard budgets (Section 6.7)."""

    def __init__(
        self,
        budgets: ResourceBudgets | None = None,
        *,
        continuation_threshold: float = 1.0,
        risk_weight: float = 0.5,
    ) -> None:
        """Bind budgets, the utility threshold and the risk cost weight."""
        self.budgets = budgets or ResourceBudgets()
        if not math.isfinite(continuation_threshold) or continuation_threshold < 0:
            raise ValueError("continuation_threshold must be finite and non-negative")
        self.continuation_threshold = continuation_threshold
        if not math.isfinite(risk_weight) or risk_weight < 0:
            raise ValueError("risk_weight must be finite and non-negative")
        self.risk_weight = risk_weight

    def budgets_exhausted(self, snapshot: ResourceSnapshot) -> tuple[bool, str]:
        """Return whether any hard budget is exhausted (Section 2)."""
        if snapshot.rule_fires >= self.budgets.max_rule_fires:
            return True, "rule_fire_budget"
        if snapshot.max_depth_reached >= self.budgets.max_depth:
            return True, "depth_budget"
        if snapshot.candidates >= self.budgets.max_candidates:
            return True, "candidate_budget"
        if snapshot.model_calls >= self.budgets.max_model_calls:
            return True, "model_call_budget"
        if snapshot.elapsed_seconds >= self.budgets.wall_clock_seconds:
            return True, "wall_clock_budget"
        if snapshot.memory_mb >= self.budgets.max_memory_mb:
            return True, "memory_budget"
        return False, ""

    def continuation_utility(
        self,
        expected_information_gain: float,
        expected_goal_progress: float,
        cpu_cost: float,
        memory_cost: float,
        latency_cost: float,
        risk_cost: float,
    ) -> float:
        """Compute ``U_continue`` (Section 6.7) with a guard against zero cost."""
        values = (
            expected_information_gain, expected_goal_progress, cpu_cost,
            memory_cost, latency_cost, risk_cost,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("utility inputs must be finite and non-negative")
        denominator = cpu_cost + memory_cost + latency_cost + self.risk_weight * risk_cost
        if denominator <= 0:
            return 0.0
        return (expected_information_gain + expected_goal_progress) / denominator

    def evaluate_continuation(
        self,
        snapshot: ResourceSnapshot,
        expected_information_gain: float,
        expected_goal_progress: float,
        cpu_cost: float,
        memory_cost: float,
        latency_cost: float,
        risk_cost: float,
    ) -> ContinuationDecision:
        """Decide whether to continue inference (Section 6.7).

        Hard budgets are checked first: continuation is never allowed once a
        cap is reached, regardless of expected benefit.
        """
        exhausted, budget_name = self.budgets_exhausted(snapshot)
        if exhausted:
            return ContinuationDecision(
                utility=0.0,
                threshold=self.continuation_threshold,
                decision="stop_budget",
                reason=f"hard budget exhausted: {budget_name}",
            )
        utility = self.continuation_utility(
            expected_information_gain, expected_goal_progress,
            cpu_cost, memory_cost, latency_cost, risk_cost,
        )
        if utility >= self.continuation_threshold:
            return ContinuationDecision(
                utility=utility,
                threshold=self.continuation_threshold,
                decision="continue",
                reason="expected benefit clears the policy threshold",
            )
        return ContinuationDecision(
            utility=utility,
            threshold=self.continuation_threshold,
            decision="stop_utility",
            reason="expected benefit below the policy threshold",
        )


__all__ = [
    "ContinuationDecision",
    "ResourceBudgets",
    "ResourcePolicy",
    "ResourceSnapshot",
]
