"""Constrained policy optimization and runtime resource governance (Step 50).

Optimize expected return subject to expected cumulative cost budgets:

    max_pi J_R(pi)  subject to  J_{C_i}(pi) <= B_i

Lagrangian multipliers tune the *soft* resource trade-off inside the learned
objective. They never replace capability checks, formal validation, or
execution limits: hard, non-negotiable requirements remain deterministic
gates in ``ore_rl.safety`` and the resource coordinator boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .contracts import OREState, PolicyAuthority, PolicyDecision
from .policies.base import PolicyService


@dataclass(frozen=True, slots=True)
class ResourceConstraint:
    """One budgeted resource cost signal (Step 50)."""

    constraint_id: str
    budget: float
    description: str = ""


@dataclass(slots=True)
class ResourceConstraintSet:
    """The active constraint set with measured cumulative usage."""

    constraints: tuple[ResourceConstraint, ...]
    usage: dict[str, float] = field(default_factory=dict)

    def record(self, constraint_id: str, cost: float) -> None:
        self.usage[constraint_id] = self.usage.get(constraint_id, 0.0) + float(cost)

    def violations(self) -> tuple[str, ...]:
        exceeded: list[str] = []
        for constraint in self.constraints:
            if self.usage.get(constraint.constraint_id, 0.0) > constraint.budget:
                exceeded.append(constraint.constraint_id)
        return tuple(exceeded)

    def slack(self, constraint_id: str) -> float:
        constraint = next(
            (item for item in self.constraints if item.constraint_id == constraint_id), None
        )
        if constraint is None:
            return 0.0
        return constraint.budget - self.usage.get(constraint_id, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraints": [
                {
                    "constraint_id": item.constraint_id,
                    "budget": item.budget,
                    "usage": self.usage.get(item.constraint_id, 0.0),
                    "description": item.description,
                }
                for item in self.constraints
            ],
            "violations": list(self.violations()),
        }


class LagrangianOptimizer:
    """Dual ascent on soft constraint multipliers (Step 50).

    lambda_i <- max(0, lambda_i + eta * (J_Ci - B_i)); the effective reward
    passed to the learner is r - sum_i lambda_i * c_i. Multipliers are bounded
    and persisted so experiments are reproducible.
    """

    def __init__(
        self,
        constraint_set: ResourceConstraintSet,
        *,
        learning_rate: float = 0.05,
        max_multiplier: float = 10.0,
    ) -> None:
        self.constraint_set = constraint_set
        self.learning_rate = float(learning_rate)
        self.max_multiplier = float(max_multiplier)
        self.multipliers: dict[str, float] = {
            constraint.constraint_id: 0.0 for constraint in constraint_set.constraints
        }

    def shaped_reward(self, base_reward: float, costs: Mapping[str, float]) -> float:
        """Effective training reward: base minus multiplier-weighted costs."""
        penalty = sum(
            self.multipliers.get(constraint_id, 0.0) * float(cost)
            for constraint_id, cost in costs.items()
        )
        return base_reward - penalty

    def update(self, costs: Mapping[str, float]) -> dict[str, float]:
        """One dual-ascent step from realized costs (in expectation over episodes)."""
        for constraint in self.constraint_set.constraints:
            realized = self.constraint_set.usage.get(constraint.constraint_id, 0.0)
            violation = realized - constraint.budget
            updated = self.multipliers[constraint.constraint_id] + self.learning_rate * violation
            self.multipliers[constraint.constraint_id] = max(
                0.0, min(self.max_multiplier, updated)
            )
        return dict(self.multipliers)

    def state(self) -> dict[str, float]:
        return dict(self.multipliers)

    def load_state(self, multipliers: Mapping[str, float]) -> None:
        for key, value in multipliers.items():
            if key in self.multipliers:
                self.multipliers[key] = max(0.0, min(self.max_multiplier, float(value)))


class ConstrainedPolicyWrapper:
    """A policy decorator that applies constraint shaping before learning.

    Proposals are delegated unchanged — hard feasibility is enforced by the
    safety gate, never by the wrapper. The wrapper exposes
    ``observe_costs`` so a training loop can feed realized costs into the
    Lagrangian optimizer and obtain shaped rewards for policy updates.
    """

    def __init__(self, inner: PolicyService, optimizer: LagrangianOptimizer) -> None:
        self.inner = inner
        self.optimizer = optimizer
        inner_family = inner.metadata.algorithm_family
        self.metadata = type(inner.metadata)(
            policy_id=f"constrained::{inner.metadata.policy_id}",
            policy_version=inner.metadata.policy_version,
            algorithm_family=f"constrained/{inner_family}",
            observation_schema_version=inner.metadata.observation_schema_version,
            action_schema_version=inner.metadata.action_schema_version,
            training_dataset_fingerprint=inner.metadata.training_dataset_fingerprint,
            evaluation_fingerprint=inner.metadata.evaluation_fingerprint,
            resource_requirements=dict(inner.metadata.resource_requirements),
            uncertainty_behavior=inner.metadata.uncertainty_behavior,
            execution_authority=PolicyAuthority.PROPOSAL,
            description=f"constrained wrapper around {inner.metadata.policy_id}",
        )

    def propose_actions(
        self,
        state: OREState,
        eligible_actions: Sequence[str],
        budget: Mapping[str, float] | None = None,
    ) -> PolicyDecision:
        decision = self.inner.propose_actions(state, eligible_actions, budget)
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=decision.ranked,
            chosen_action_id=decision.chosen_action_id,
            propensity=decision.propensity,
            uncertainty=decision.uncertainty,
            authority=decision.authority,
            wall_time_ms=decision.wall_time_ms,
            fallback_used=decision.fallback_used,
            rationale=decision.rationale,
        )

    def shaped_reward(self, base_reward: float, costs: Mapping[str, float]) -> float:
        return self.optimizer.shaped_reward(base_reward, costs)

    def update_multipliers(self, costs: Mapping[str, float]) -> dict[str, float]:
        return self.optimizer.update(costs)
