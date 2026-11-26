"""Multi-dimensional operation costs, goal progress, and documented search heuristics.

Cost model formula (specification section 5):

    c(o) = w_t*T(o) + w_m*M(o) + w_r*R(o) + w_u*U(o) + w_f*F(o)

where T and M are normalized against explicit reference scales before weighting, the
normalized weights are versioned, and every heuristic declares whether it is
admissible. Optimality claims are only permitted with admissible heuristics.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import (
    CostVector,
    CostWeights,
    GoalSpec,
    OperatorSpec,
    PlanState,
    SuccessCondition,
)

HEURISTIC_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class GoalProgress:
    """Report weighted goal progress with critical predicates as hard gates."""

    score: float
    critical_satisfied: bool
    unsatisfied_conditions: tuple[str, ...]
    critical_unsatisfied: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate progress bounds."""
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("goal progress score must be in [0, 1]")

    @property
    def achieved(self) -> bool:
        """Indicate whether every critical gate and a non-trivial score are satisfied."""
        return self.critical_satisfied and self.score >= 1.0

    def to_dict(self) -> dict[str, Any]:
        """Return the progress record."""
        return {
            "score": self.score,
            "critical_satisfied": self.critical_satisfied,
            "unsatisfied_conditions": list(self.unsatisfied_conditions),
            "critical_unsatisfied": list(self.critical_unsatisfied),
        }


class CostModel:
    """Compute scalar operation costs and weighted goal progress from measured telemetry."""

    def __init__(
        self,
        weights: CostWeights | None = None,
        *,
        time_scale_ms: float = 1000.0,
        memory_scale_mb: float = 512.0,
    ) -> None:
        """Configure versioned weights and the normalization scales for time and memory."""
        self.weights = weights or CostWeights()
        self.time_scale_ms = float(time_scale_ms)
        self.memory_scale_mb = float(memory_scale_mb)
        if self.time_scale_ms <= 0 or self.memory_scale_mb <= 0:
            raise ValueError("normalization scales must be positive")

    def normalized_components(self, vector: CostVector) -> dict[str, float]:
        """Normalize every cost component into the unit range before weighting."""
        return {
            "time": min(1.0, vector.time_ms / self.time_scale_ms),
            "memory": min(1.0, vector.memory_mb / self.memory_scale_mb),
            "risk": vector.risk,
            "uncertainty": vector.uncertainty,
            "failure_recovery": vector.failure_recovery,
        }

    def operator_cost(self, operator: OperatorSpec) -> float:
        """Return the weighted scalar cost c(o) for one operator."""
        return self.vector_cost(operator.estimated_cost)

    def vector_cost(self, vector: CostVector) -> float:
        """Return the weighted scalar cost for an arbitrary measured cost vector."""
        parts = self.normalized_components(vector)
        weights = self.weights
        return (
            weights.time * parts["time"]
            + weights.memory * parts["memory"]
            + weights.risk * parts["risk"]
            + weights.uncertainty * parts["uncertainty"]
            + weights.failure_recovery * parts["failure_recovery"]
        )

    def plan_cost(self, operators: Iterable[OperatorSpec]) -> float:
        """Sum the weighted cost of every operator in a plan."""
        return sum(self.operator_cost(operator) for operator in operators)

    def goal_progress(self, goal: GoalSpec, facts: frozenset[str] | set[str]) -> GoalProgress:
        """Compute P(G) = sum(w_i*s_i)/sum(w_i) with critical predicates as hard gates."""
        satisfied_weights = 0.0
        total_weight = 0.0
        unsatisfied: list[str] = []
        critical_unsatisfied: list[str] = []
        for condition in goal.success_conditions:
            total_weight += condition.weight
            achieved = condition.predicate.satisfied_by(facts)
            if achieved:
                satisfied_weights += condition.weight
            else:
                unsatisfied.append(condition.predicate.to_atom())
                if condition.critical:
                    critical_unsatisfied.append(condition.predicate.to_atom())
        score = satisfied_weights / total_weight if total_weight > 0 else 0.0
        return GoalProgress(
            score=score,
            critical_satisfied=not critical_unsatisfied,
            unsatisfied_conditions=tuple(sorted(unsatisfied)),
            critical_unsatisfied=tuple(sorted(critical_unsatisfied)),
        )

    def calibrate(
        self,
        feedback_history: Sequence[Any],
        *,
        next_version: str = "1.1.0",
    ) -> CostModel:
        """Rebuild the model with scales derived from measured execution history."""
        times: list[float] = []
        memories: list[float] = []
        for feedback in feedback_history:
            actual = getattr(feedback, "actual_cost", None)
            if isinstance(actual, CostVector):
                if actual.time_ms > 0:
                    times.append(actual.time_ms)
                if actual.memory_mb > 0:
                    memories.append(actual.memory_mb)
        time_scale = max(times) if times else self.time_scale_ms
        memory_scale = max(memories) if memories else self.memory_scale_mb
        return CostModel(
            replace(self.weights, version=next_version),
            time_scale_ms=max(time_scale, 1.0),
            memory_scale_mb=max(memory_scale, 1.0),
        )

    def describe(self) -> dict[str, Any]:
        """Return the documented weight and scale configuration."""
        return {
            "weights": self.weights.to_dict(),
            "time_scale_ms": self.time_scale_ms,
            "memory_scale_mb": self.memory_scale_mb,
        }


class RemainingCostHeuristic:
    """Estimate remaining plan cost from unmet success conditions and provider costs."""

    admissible: bool
    consistent: bool
    version: str
    description: str

    def estimate(
        self,
        state: PlanState,
        goal: GoalSpec,
        operators: Mapping[str, OperatorSpec],
        cost_model: CostModel,
    ) -> float:
        """Return h(n) >= 0 for the given search state."""
        raise NotImplementedError


class ZeroHeuristic(RemainingCostHeuristic):
    """Use h(n) = 0, which is admissible and makes A* behave like uniform-cost search."""

    def __init__(self) -> None:
        """Declare an admissible, versioned zero heuristic."""
        self.admissible = True
        self.consistent = True
        self.version = HEURISTIC_VERSION
        self.description = "h(n)=0; admissible and consistent; A* degrades to uniform-cost search"

    def estimate(
        self,
        state: PlanState,
        goal: GoalSpec,
        operators: Mapping[str, OperatorSpec],
        cost_model: CostModel,
    ) -> float:
        """Return zero for every state."""
        return 0.0


class UnmetConditionHeuristic(RemainingCostHeuristic):
    """Sum the cheapest provider cost for every unmet success condition.

    Admissibility argument: each unmet condition requires at least one more operator
    before the goal test can pass, and operator costs are non-negative, so charging
    the minimum eligible provider cost per condition never overestimates the true
    remaining cost when distinct conditions need distinct operators. It may
    underestimate when one operator satisfies several conditions at once, which is
    safe for admissibility.
    """

    def __init__(self) -> None:
        """Declare an admissible, versioned unmet-condition heuristic."""
        self.admissible = True
        self.consistent = False
        self.version = HEURISTIC_VERSION
        self.description = (
            "sum of minimum provider costs over unmet success conditions; admissible "
            "under non-negative operator costs but not guaranteed consistent, so "
            "optimality requires exhaustion; documented in ore_planning.cost"
        )

    def estimate(
        self,
        state: PlanState,
        goal: GoalSpec,
        operators: Mapping[str, OperatorSpec],
        cost_model: CostModel,
    ) -> float:
        """Charge the cheapest provider cost for each unmet success condition."""
        unmet = [
            condition for condition in goal.success_conditions
            if not condition.predicate.satisfied_by(state.achieved_facts)
        ]
        if not unmet:
            return 0.0
        total = 0.0
        for condition in unmet:
            providers = [
                cost_model.operator_cost(operator)
                for operator in operators.values()
                if operator.provides(condition.predicate.to_atom())
            ]
            if providers:
                total += min(providers)
        return total


class ScaledHeuristic(RemainingCostHeuristic):
    """Scale a base heuristic by a constant factor, which breaks admissibility above 1.0.

    A scale greater than one produces a greedy-leaning, best-effort weighted search:
    the planner must not claim optimality while using this heuristic.
    """

    def __init__(self, base: RemainingCostHeuristic, scale: float) -> None:
        """Configure the base heuristic and its documented, finite scale factor."""
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be a positive finite number")
        self.base = base
        self.scale = scale
        self.admissible = bool(base.admissible and scale <= 1.0)
        self.consistent = bool(base.consistent and scale <= 1.0)
        self.version = f"{base.version}+scaled:{scale:g}"
        self.description = (
            f"{base.description}; scaled by {scale:g}; "
            + ("admissible" if self.admissible else "inadmissible: best-effort search only")
        )

    def estimate(
        self,
        state: PlanState,
        goal: GoalSpec,
        operators: Mapping[str, OperatorSpec],
        cost_model: CostModel,
    ) -> float:
        """Return the scaled base estimate."""
        return self.scale * self.base.estimate(state, goal, operators, cost_model)


def goal_progress_from_conditions(
    conditions: Sequence[SuccessCondition],
    facts: frozenset[str] | set[str],
) -> GoalProgress:
    """Compute weighted progress for an arbitrary condition collection."""
    satisfied = 0.0
    total = 0.0
    unsatisfied: list[str] = []
    critical_unsatisfied: list[str] = []
    for condition in conditions:
        total += condition.weight
        if condition.predicate.satisfied_by(facts):
            satisfied += condition.weight
        else:
            unsatisfied.append(condition.predicate.to_atom())
            if condition.critical:
                critical_unsatisfied.append(condition.predicate.to_atom())
    return GoalProgress(
        score=satisfied / total if total > 0 else 0.0,
        critical_satisfied=not critical_unsatisfied,
        unsatisfied_conditions=tuple(sorted(unsatisfied)),
        critical_unsatisfied=tuple(sorted(critical_unsatisfied)),
    )


__all__ = [
    "HEURISTIC_VERSION",
    "CostModel",
    "GoalProgress",
    "RemainingCostHeuristic",
    "ScaledHeuristic",
    "UnmetConditionHeuristic",
    "ZeroHeuristic",
    "goal_progress_from_conditions",
]
