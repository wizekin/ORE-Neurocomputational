"""Resource budgets and CROS-style admission control (step 12; section 5).

CROS / AdaptiveResourceCoordinator enforces experiment budgets and protects
production workload (integration map): evolution requests run at *low priority* and
are deferred whenever the remaining evaluation budget cannot keep a reserve for
production work. Budget exhaustion stops the run cleanly (step 28; acceptance test
22).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..evaluation.evaluator import BudgetExhausted, EvaluationBudget

#: Evolution requests always run at low scheduling priority (production protection).
EVOLUTION_PRIORITY = "low"

#: Fraction of the budget that must remain reserved for production work.
PRODUCTION_RESERVE_FRACTION = 0.05


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Admission decision for one experiment request."""

    allowed: bool
    priority: str = EVOLUTION_PRIORITY
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed, "priority": self.priority, "reason": self.reason,
        }


class ResourceBudget:
    """Whole-run resource budget: evaluations, generations, population (step 12)."""

    def __init__(
        self,
        *,
        total_evaluations: int,
        generation_cap: int = 12,
        population_cap: int = 24,
        label: str = "evolution",
    ) -> None:
        if total_evaluations < 1:
            raise ValueError("total_evaluations must be positive")
        if generation_cap < 1 or population_cap < 2:
            raise ValueError("generation and population caps must be positive")
        self.label = label
        self.evaluation_budget = EvaluationBudget(total_evaluations)
        self.generation_cap = generation_cap
        self.population_cap = population_cap
        self.granted_at_utc = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds",
        )

    @property
    def total_evaluations(self) -> int:
        return self.evaluation_budget.total_evaluations

    @property
    def consumed_evaluations(self) -> int:
        return self.evaluation_budget.consumed

    def check_generation(self, generation: int) -> None:
        if generation > self.generation_cap:
            raise BudgetExhausted(
                f"generation cap exceeded ({generation} > {self.generation_cap})"
            )

    def check_population(self, size: int) -> None:
        if size > self.population_cap:
            raise BudgetExhausted(
                f"population cap exceeded ({size} > {self.population_cap})"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "evaluation_budget": self.evaluation_budget.as_dict(),
            "generation_cap": self.generation_cap,
            "population_cap": self.population_cap,
            "granted_at_utc": self.granted_at_utc,
        }


class AdmissionController:
    """CROS-style admission for evolution experiment requests (step 12)."""

    def __init__(
        self,
        *,
        reserve_fraction: float = PRODUCTION_RESERVE_FRACTION,
    ) -> None:
        if not 0.0 <= reserve_fraction <= 1.0:
            raise ValueError("reserve fraction must be within [0, 1]")
        self.reserve_fraction = reserve_fraction
        self.decisions: list[dict[str, Any]] = []

    def decide(self, budget: ResourceBudget) -> AdmissionDecision:
        """Allow a low-priority evolution request only with a protected reserve."""
        remaining = budget.evaluation_budget.remaining
        reserve = max(int(budget.total_evaluations * self.reserve_fraction), 1)
        if remaining <= reserve:
            decision = AdmissionDecision(
                allowed=False,
                reason=(
                    f"admission deferred: remaining evaluations ({remaining}) cannot "
                    f"disturb the production reserve ({reserve})"
                ),
            )
        else:
            decision = AdmissionDecision(allowed=True, reason="budget reserve intact")
        self.decisions.append({
            "allowed": decision.allowed,
            "priority": decision.priority,
            "reason": decision.reason,
            "remaining": remaining,
            "decided_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds",
            ),
        })
        return decision


__all__ = [
    "AdmissionController", "AdmissionDecision", "EVOLUTION_PRIORITY",
    "PRODUCTION_RESERVE_FRACTION", "ResourceBudget",
]
