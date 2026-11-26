"""Bounded numerical integration with evaluation budgets and divergence detection.

Implements the specification's solver requirements: explicit Euler and fourth-order
Runge-Kutta steps, bounded step sizes, exposed tolerances and function-evaluation counts,
and explicit failure when states explode or leave the finite regime.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from ..functional import euler_step, rk4_step

__all__ = ("SolverDiagnostics", "SolverFailure", "SolverMethod", "integrate_interval",
           "solve_divergence_check")

SolverMethod = Callable[[np.ndarray, float, Callable[[np.ndarray, float], np.ndarray], float],
                        np.ndarray]


class SolverFailure(RuntimeError):
    """Raised when integration exceeds its budget, diverges, or leaves the finite regime."""


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    """Expose solver behaviour: steps, evaluations, convergence, and failure reasons."""

    method: str
    steps: int
    evaluations: int
    max_evaluations: int
    converged: bool
    final_step_size: float
    failure_reason: str = ""

    @property
    def budget_exhausted(self) -> bool:
        """Report whether the evaluation budget was exhausted."""
        return self.evaluations >= self.max_evaluations

    def to_dict(self) -> dict[str, object]:
        """Serialize the solver diagnostics."""
        return {
            "method": self.method,
            "steps": self.steps,
            "evaluations": self.evaluations,
            "max_evaluations": self.max_evaluations,
            "converged": self.converged,
            "final_step_size": self.final_step_size,
            "failure_reason": self.failure_reason,
            "budget_exhausted": self.budget_exhausted,
        }


class _EvaluationCounter:
    """Count vector-field evaluations and refuse to exceed the configured budget."""

    def __init__(self, vector_field: Callable[[np.ndarray, float], np.ndarray],
                 max_evaluations: int) -> None:
        """Wrap a vector field with an evaluation budget."""
        self._vector_field = vector_field
        self.max_evaluations = max_evaluations
        self.evaluations = 0

    def __call__(self, state: np.ndarray, t: float) -> np.ndarray:
        """Evaluate the vector field or raise when the budget is exhausted."""
        if self.evaluations >= self.max_evaluations:
            raise SolverFailure(
                f"solver evaluation budget exhausted at {self.evaluations} evaluations")
        self.evaluations += 1
        return np.asarray(self._vector_field(state, t), dtype=float)


def solve_divergence_check(state: np.ndarray, state_norm_bound: float) -> None:
    """Raise SolverFailure when the state is non-finite or exceeds the norm bound."""
    array = np.asarray(state, dtype=float)
    if not np.all(np.isfinite(array)):
        raise SolverFailure("solver produced a non-finite state; output invalidated")
    norm = float(np.linalg.norm(array)) if array.size else 0.0
    if norm > state_norm_bound:
        raise SolverFailure(
            f"solver state norm {norm} exceeded the configured bound {state_norm_bound}")


def integrate_interval(vector_field: Callable[[np.ndarray, float], np.ndarray],
                       x0: np.ndarray, t0: float, t1: float, *,
                       method: str = "euler", max_step: float = 0.1,
                       max_evaluations: int = 10_000, state_norm_bound: float = 1e6
                       ) -> tuple[np.ndarray, SolverDiagnostics]:
    """Integrate dx/dt = f(x, t) from t0 to t1 with bounded steps and evaluations.

    Step sizes never exceed max_step; the evaluation budget is enforced exactly; any
    non-finite or out-of-bound state raises SolverFailure so callers can fall back.
    """
    if method not in ("euler", "rk4"):
        raise ValueError("method must be 'euler' or 'rk4'")
    if t1 < t0:
        raise ValueError("integration interval must not be reversed")
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    counter = _EvaluationCounter(vector_field, max_evaluations)
    state = np.asarray(x0, dtype=float).copy()
    solve_divergence_check(state, state_norm_bound)
    total = t1 - t0
    if total == 0.0:
        return (state, SolverDiagnostics(method, 0, 0, max_evaluations, True, 0.0))
    steps_taken = 0
    final_step = 0.0
    current = t0
    while current < t1:
        step = min(max_step, t1 - current)
        if step <= 0.0:
            break
        if method == "euler":
            state = euler_step(state, step, lambda x: counter(x, current))
        else:
            state = rk4_step(state, step, counter, current)
        steps_taken += 1
        final_step = step
        current += step
        solve_divergence_check(state, state_norm_bound)
    diagnostics = SolverDiagnostics(
        method=method,
        steps=steps_taken,
        evaluations=counter.evaluations,
        max_evaluations=max_evaluations,
        converged=math.isclose(current, t1, rel_tol=1e-9, abs_tol=1e-12) or current >= t1,
        final_step_size=final_step,
    )
    return (state, diagnostics)
