"""QUBO formulation and validation (Point 14, Steps 44-45).

A combinatorial problem is admissible only when it can be expressed with binary
decisions and quadratic interactions (selected scheduling, assignment, or graph
problems).  The QUBO objective

    min_{x in {0,1}^n} x^T Q x     with penalties     E(x) = f(x) + rho sum_j g_j(x)^2

is built from the declared objective and linear-equality constraints; the mapping is
verified exhaustively on small instances before any solver is trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import exhaustive_qubo, qubo_energy

__all__ = ("CombinatorialProblem", "OptimizationPolicyError", "QuboFormulation",
           "formulate_qubo", "verify_small_instance")


class OptimizationPolicyError(ValueError):
    """Raised when the baseline-before-Ising ordering or the policy boundary is violated."""


@dataclass(frozen=True, slots=True)
class CombinatorialProblem:
    """Step 44 -- a combinatorial problem over binary decisions.

    ``objective_matrix`` is Q_f for f(x) = x^T Q_f x (linear terms on the diagonal);
    ``constraints`` are linear equalities (a_j, b_j) meaning a_j . x = b_j; each
    contributes penalty rho * (a_j . x - b_j)^2.
    """

    name: str
    variable_names: tuple[str, ...]
    objective_matrix: np.ndarray
    constraints: tuple[tuple[np.ndarray, float], ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the problem shape (Step 44 admissibility)."""
        matrix = np.asarray(self.objective_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
            raise ValueError("objective_matrix must be square with at least 2 variables")
        if len(self.variable_names) != matrix.shape[0]:
            raise ValueError("one variable name per row/column is required")
        for row, bound in self.constraints:
            if np.asarray(row).shape != (matrix.shape[0],):
                raise ValueError("constraint rows must match the variable count")

    @property
    def n_variables(self) -> int:
        """Number of binary decisions."""
        return int(np.asarray(self.objective_matrix).shape[0])

    def objective_value(self, bits: np.ndarray) -> float:
        """f(x) evaluated directly from the declared objective."""
        return qubo_energy(np.asarray(bits, dtype=float),
                           np.asarray(self.objective_matrix, dtype=float))

    def constraint_violations(self, bits: np.ndarray) -> list[float]:
        """g_j(x) for every linear equality constraint."""
        x = np.asarray(bits, dtype=float)
        return [float(np.dot(np.asarray(row, dtype=float), x) - bound)
                for row, bound in self.constraints]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"name": self.name, "variables": list(self.variable_names),
                "n_variables": self.n_variables,
                "n_constraints": len(self.constraints),
                "description": self.description}


@dataclass(frozen=True, slots=True)
class QuboFormulation:
    """Step 45 -- the penalized QUBO objective with its construction record."""

    problem: CombinatorialProblem
    qubo_matrix: np.ndarray
    penalty_rho: float
    constant_offset: float
    construction: tuple[str, ...]

    def energy(self, bits: np.ndarray) -> float:
        """E(x) = x^T Q x (the constant offset does not affect the argmin)."""
        return qubo_energy(np.asarray(bits, dtype=float), self.qubo_matrix)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"problem": self.problem.to_dict(),
                "penalty_rho": self.penalty_rho,
                "constant_offset": self.constant_offset,
                "construction": list(self.construction),
                "note": "penalty terms discourage violations without destroying useful "
                        "numerical distinctions"}


def formulate_qubo(problem: CombinatorialProblem,
                   config: LearningConfig | None = None) -> QuboFormulation:
    """Step 45 -- build the penalized QUBO matrix E(x) = f(x) + rho sum g_j(x)^2.

    Each linear equality a.x = b contributes x^T (a a^T) x - 2b diag(a) x plus the
    constant b^2, which is recorded separately because it cannot change the argmin.
    """
    effective = config or LearningConfig()
    n = problem.n_variables
    q_total = np.asarray(problem.objective_matrix, dtype=float).copy()
    offset = 0.0
    construction: list[str] = ["objective: x^T Q_f x"]
    rho = effective.qubo_penalty_rho
    for index, (row, bound) in enumerate(problem.constraints):
        a = np.asarray(row, dtype=float)
        q_total = q_total + rho * (np.outer(a, a) - 2.0 * bound * np.diag(a))
        offset += rho * bound * bound
        construction.append(f"penalty {index}: rho*({a.tolist()} . x - {bound})^2")
    construction.append(f"constant offset {offset:.6g} recorded, not encoded")
    return QuboFormulation(problem=problem, qubo_matrix=q_total, penalty_rho=rho,
                           constant_offset=offset,
                           construction=tuple(construction))


def verify_small_instance(formulation: QuboFormulation,
                          config: LearningConfig | None = None) -> dict[str, Any]:
    """Step 45 -- exhaustively verify the mapping where enumeration is possible.

    For small instances the exhaustive QUBO optimum must match the best *feasible*
    solution of the original problem (objective value equal, constraints satisfied);
    acceptance criterion: "QUBO mapping reproduces known optima on small exhaustive
    cases".
    """
    effective = config or LearningConfig()
    problem = formulation.problem
    n = problem.n_variables
    if n > effective.qubo_exhaustive_verify_max_variables:
        return {"verified": None,
                "reason": "instance too large for exhaustive verification",
                "n_variables": n}
    best_bits, best_energy = exhaustive_qubo(formulation.qubo_matrix)
    violations = problem.constraint_violations(best_bits)
    feasible = all(abs(value) <= 1e-9 for value in violations)
    # exhaustive best feasible solution of the original problem
    best_feasible_objective = math_inf = float("inf")
    best_feasible_bits = None
    for mask in range(1 << n):
        bits = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        if all(abs(value) <= 1e-9 for value in problem.constraint_violations(bits)):
            objective = problem.objective_value(bits)
            if objective < best_feasible_objective:
                best_feasible_objective, best_feasible_bits = objective, bits
    reproduced = (feasible and best_feasible_bits is not None
                  and np.array_equal(best_bits, best_feasible_bits))
    return {"verified": reproduced, "feasible": feasible,
            "qubo_optimum_bits": best_bits.astype(int).tolist(),
            "qubo_optimum_energy": round(best_energy + formulation.constant_offset, 9),
            "violations": [round(value, 9) for value in violations],
            "best_feasible_objective": (round(best_feasible_objective, 9)
                                        if best_feasible_bits is not None else None),
            "objective_at_qubo_optimum": round(problem.objective_value(best_bits), 9),
            "n_variables": n,
            "note": "small instances are exhaustively checked before solvers are trusted"}


def math_inf_marker() -> float:  # pragma: no cover - documentation helper
    """The exhaustive search initializes with positive infinity."""
    return float("inf")
