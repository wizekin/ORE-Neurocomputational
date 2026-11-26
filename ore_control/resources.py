"""Resource reservation and adaptive history for the control layer (Step 30).

This module reuses the repository's existing ``ore_contracts`` resource
machinery verbatim - SQLite-persisted reliability history with Beta(2, 2)
priors, exponential moving averages for measured costs, and circuit-breaker
state - and adds the reservation gate the control layer needs before dispatch.
Reservation failure leads to defer/re-rank, never to bypassing the
coordinator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ore_contracts.resource_coordination import (
    ModuleReliabilityTracker,
    ModuleResourceEstimator,
    SQLiteResourceHistory,
)
from ore_contracts.resource_profiles import ModuleResourceEstimate

from .config import ControlPolicy

CONTROL_TASK_FAMILY = "cognitive_control"


@dataclass(frozen=True, slots=True)
class ReservationDecision:
    """One reservation request verdict (Step 30)."""

    operator_id: str
    granted: bool
    reason: str
    estimated_time_seconds: float
    estimated_memory_mb: float
    circuit_open: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable reservation record."""
        return {
            "operator_id": self.operator_id,
            "granted": self.granted,
            "reason": self.reason,
            "estimated_time_seconds": float(self.estimated_time_seconds),
            "estimated_memory_mb": float(self.estimated_memory_mb),
            "circuit_open": bool(self.circuit_open),
        }


class ControlResourceCoordinator:
    """Own resource history, estimates, reservations, and circuit breakers."""

    def __init__(self, policy: ControlPolicy) -> None:
        """Bind the upstream resource machinery to the control policy."""
        self.policy = policy
        self._temporary_directory = None
        if policy.resource_history_path is not None:
            database_path: str | Path = Path(policy.resource_history_path)
        else:
            import tempfile

            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="ore-control-",
            )
            database_path = (
                Path(self._temporary_directory.name) / "resource_history.sqlite3"
            )
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
            policy.declared_operator_estimates,
            moving_average_alpha=policy.cost_moving_average_alpha,
        )
        self.reservations: list[ReservationDecision] = []

    # ------------------------------------------------------------------ API

    def reserve(
        self,
        operator_id: str,
        *,
        estimated_time_seconds: float,
        estimated_memory_mb: float,
        remaining_time_seconds: float,
        remaining_memory_mb: float,
    ) -> ReservationDecision:
        """Reserve resources for one operator dispatch (Step 30).

        The reservation is denied when the circuit breaker is open, when the
        estimate exceeds the remaining time budget, or when the memory
        footprint exceeds the remaining memory budget. Denial never bypasses
        the coordinator: the caller re-ranks or defers.
        """
        circuit_open = self.reliability.is_circuit_open(
            CONTROL_TASK_FAMILY, operator_id,
        )
        if circuit_open:
            decision = ReservationDecision(
                operator_id, False, "circuit_breaker_open",
                estimated_time_seconds, estimated_memory_mb, True,
            )
            self.reservations.append(decision)
            return decision
        if estimated_time_seconds > remaining_time_seconds:
            decision = ReservationDecision(
                operator_id, False, "time_budget_exceeded",
                estimated_time_seconds, estimated_memory_mb, False,
            )
            self.reservations.append(decision)
            return decision
        if estimated_memory_mb > remaining_memory_mb:
            decision = ReservationDecision(
                operator_id, False, "memory_budget_exceeded",
                estimated_time_seconds, estimated_memory_mb, False,
            )
            self.reservations.append(decision)
            return decision
        decision = ReservationDecision(
            operator_id, True, "granted",
            estimated_time_seconds, estimated_memory_mb, False,
        )
        self.reservations.append(decision)
        return decision

    def estimate(self, operator_id: str) -> ModuleResourceEstimate:
        """Return the EMA-updated resource estimate for one operator."""
        return self.estimator.estimate(
            CONTROL_TASK_FAMILY, operator_id, 0.5,
        )

    def reliability_probability(self, operator_id: str) -> float:
        """Return the Beta-posterior mean reliability estimate."""
        return self.reliability.probability(CONTROL_TASK_FAMILY, operator_id)

    def record_outcome(self, operator_id: str, success: bool) -> None:
        """Record one operator outcome into the shared reliability history."""
        self.reliability.record_outcome(CONTROL_TASK_FAMILY, operator_id, success)

    def record_measurement(
        self, operator_id: str, *, latency_ms: float,
        memory_mb: float | None = None, cpu_cost: float | None = None,
    ) -> None:
        """Record measured costs with the exponential moving average."""
        self.estimator.record_measurement(
            CONTROL_TASK_FAMILY, operator_id, latency_ms=latency_ms,
            memory_mb=memory_mb, cpu_cost=cpu_cost,
        )

    def measured_latency_ms(self, operator_id: str) -> float | None:
        """Return the smoothed measured latency for one operator, if recorded."""
        estimate = self.estimator.estimate(CONTROL_TASK_FAMILY, operator_id, 0.5)
        latency = getattr(estimate, "expected_latency_ms", None)
        if latency is None or not math.isfinite(float(latency)):
            return None
        return float(latency)

    def close(self) -> None:
        """Release the resource-history database without touching task state."""
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None


__all__ = [
    "CONTROL_TASK_FAMILY",
    "ControlResourceCoordinator",
    "ReservationDecision",
]
