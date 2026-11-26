"""Phase VII, Step 32 - deterministic admission and resource budgets.

Set per-task CPU, memory, time and worker budgets, reserve minimum capacity
for critical system operations, apply deterministic admission rules, and
reject, defer or degrade optional work when budgets are insufficient.  A task
is admissible only when its estimated requirement fits the available budget
``r_hat_{t,j} <= B_{t,j}`` for every constrained dimension; a learned score
never overrides hard CPU, memory, time or safety ceilings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .core import new_id, sha256_digest, utc_now_iso

ADMISSION_SCHEMA = "ore.point24.admission.v1"

ADMITTED = "admitted"
DEFERRED = "deferred"
DEGRADED = "degraded"
REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """A deterministic multi-dimensional resource budget."""

    max_workers: int
    wall_clock_seconds: float | None = None
    max_memory_mb: float | None = None
    reserved_workers: int = 0
    reserved_wall_clock_seconds: float = 0.0
    reserved_memory_mb: float = 0.0

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least one")
        if self.reserved_workers < 0 or self.reserved_workers >= self.max_workers:
            raise ValueError("reserved_workers must be in [0, max_workers)")
        if self.wall_clock_seconds is not None and self.wall_clock_seconds <= 0:
            raise ValueError("wall_clock_seconds must be positive")
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")

    def available(self) -> dict[str, float]:
        """Budget minus the reserved minimum for critical system operations."""
        return {
            "workers": float(self.max_workers - self.reserved_workers),
            "wall_clock_seconds": float(
                (self.wall_clock_seconds or 0.0) - self.reserved_wall_clock_seconds
            ) if self.wall_clock_seconds is not None else float("inf"),
            "memory_mb": float(
                (self.max_memory_mb or 0.0) - self.reserved_memory_mb
            ) if self.max_memory_mb is not None else float("inf"),
        }


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """One deterministic admission decision with its recorded reason."""

    task_id: str
    decision: str
    reasons: tuple[str, ...]
    utilization: Mapping[str, float]
    decided_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "utilization": dict(self.utilization),
            "decided_at": self.decided_at,
        }


@dataclass(slots=True)
class ResourceAllocation:
    """Mutable current allocation state against one budget."""

    budget: ResourceBudget
    active_workers: int = 0
    active_wall_clock_seconds: float = 0.0
    active_memory_mb: float = 0.0
    admitted_tasks: list[str] = field(default_factory=list)
    deferred_tasks: list[str] = field(default_factory=list)
    rejected_tasks: list[str] = field(default_factory=list)

    def utilization(self) -> dict[str, float]:
        available = self.budget.available()
        return {
            "workers": self.active_workers / max(available["workers"], 1.0),
            "wall_clock_seconds": (
                self.active_wall_clock_seconds / available["wall_clock_seconds"]
                if available["wall_clock_seconds"] != float("inf") else 0.0
            ),
            "memory_mb": (
                self.active_memory_mb / available["memory_mb"]
                if available["memory_mb"] != float("inf") else 0.0
            ),
        }


class DeterministicAdmissionController:
    """Apply deterministic admission rules under hard budgets (Step 32).

    Admission is a pure function of the declared requirement, the current
    allocation and the budget; every decision records its reason, and a
    known-good fallback (defer optional work, keep critical capacity) is the
    default behaviour when budgets are insufficient.
    """

    def __init__(self, budget: ResourceBudget) -> None:
        self.state = ResourceAllocation(budget=budget)
        self.fallback_policy = "defer_optional_keep_critical"

    def request(
        self,
        task_id: str,
        *,
        workers: int = 1,
        wall_clock_seconds: float = 0.0,
        memory_mb: float = 0.0,
        optional: bool = True,
        estimate_uncertainty_margin: float = 0.1,
    ) -> AdmissionDecision:
        """Evaluate one admission request deterministically."""
        budget = self.state.budget
        available = budget.available()
        margin = 1.0 + max(0.0, estimate_uncertainty_margin)
        required = {
            "workers": float(workers),
            "wall_clock_seconds": wall_clock_seconds * margin,
            "memory_mb": memory_mb * margin,
        }
        reasons: list[str] = []
        fits = True
        if self.state.active_workers + required["workers"] > available["workers"]:
            fits = False
            reasons.append(
                f"worker budget exceeded: {self.state.active_workers + required['workers']:.0f} "
                f"> {available['workers']:.0f}"
            )
        if required["wall_clock_seconds"] > 0 and (
            self.state.active_wall_clock_seconds + required["wall_clock_seconds"]
            > available["wall_clock_seconds"]
        ):
            fits = False
            reasons.append(
                "wall-clock budget exceeded with uncertainty margin applied"
            )
        if required["memory_mb"] > 0 and (
            self.state.active_memory_mb + required["memory_mb"] > available["memory_mb"]
        ):
            fits = False
            reasons.append("memory budget exceeded with uncertainty margin applied")

        if fits:
            decision = ADMITTED
            reasons = ["requirement fits available budget"]
            self.state.active_workers += int(required["workers"])
            self.state.active_wall_clock_seconds += required["wall_clock_seconds"]
            self.state.active_memory_mb += required["memory_mb"]
            self.state.admitted_tasks.append(task_id)
        elif optional:
            decision = DEFERRED
            reasons.append("optional work deferred by the known-good fallback policy")
            self.state.deferred_tasks.append(task_id)
        else:
            decision = REJECTED
            reasons.append("mandatory work cannot fit; rejected with an explicit reason")
            self.state.rejected_tasks.append(task_id)
        return AdmissionDecision(
            task_id=task_id,
            decision=decision,
            reasons=tuple(reasons),
            utilization=self.state.utilization(),
        )

    def release(self, task_id: str, *, workers: int = 1, wall_clock_seconds: float = 0.0, memory_mb: float = 0.0) -> None:
        if task_id in self.state.admitted_tasks:
            self.state.admitted_tasks.remove(task_id)
            self.state.active_workers = max(0, self.state.active_workers - workers)
            self.state.active_wall_clock_seconds = max(
                0.0, self.state.active_wall_clock_seconds - wall_clock_seconds
            )
            self.state.active_memory_mb = max(0.0, self.state.active_memory_mb - memory_mb)

    def summary(self) -> dict[str, Any]:
        return {
            "schema": ADMISSION_SCHEMA,
            "fallback_policy": self.fallback_policy,
            "budget": {
                "max_workers": self.state.budget.max_workers,
                "reserved_workers": self.state.budget.reserved_workers,
                "wall_clock_seconds": self.state.budget.wall_clock_seconds,
                "max_memory_mb": self.state.budget.max_memory_mb,
            },
            "utilization": self.state.utilization(),
            "admitted": list(self.state.admitted_tasks),
            "deferred": list(self.state.deferred_tasks),
            "rejected": list(self.state.rejected_tasks),
            "note": "a learned score never overrides these hard ceilings",
        }


__all__ = [
    "ADMISSION_SCHEMA",
    "ADMITTED",
    "DEGRADED",
    "DEFERRED",
    "REJECTED",
    "AdmissionDecision",
    "DeterministicAdmissionController",
    "ResourceAllocation",
    "ResourceBudget",
]
