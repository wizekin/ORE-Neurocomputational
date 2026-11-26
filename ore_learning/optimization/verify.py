"""Independent verification of optimization outputs (Point 14, Step 47).

Returned bit configurations are converted into domain assignments and then checked
independently for constraint satisfaction and objective value; no optimization
backend may directly alter CROS scheduling or coordinator policy without the normal
approval path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .qubo import CombinatorialProblem, OptimizationPolicyError, QuboFormulation

__all__ = ("OptimizationVerification", "apply_to_cros_or_policy",
           "decode_assignment", "independent_feasibility_check")


@dataclass(slots=True)
class OptimizationVerification:
    """The independent verification record for one candidate solution."""

    solver: str
    assignment: dict[str, int]
    feasible: bool
    objective_value: float
    violated_constraints: list[int]
    recomputed_energy: float
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"solver": self.solver, "assignment": dict(self.assignment),
                "feasible": self.feasible,
                "objective_value": round(self.objective_value, 9),
                "violated_constraints": list(self.violated_constraints),
                "recomputed_energy": round(self.recomputed_energy, 9),
                "accepted": self.accepted,
                "note": "candidate solution verified independently before any use"}


def decode_assignment(bits: Sequence[int],
                      problem: CombinatorialProblem) -> dict[str, int]:
    """Convert a bit configuration into a named domain assignment."""
    if len(bits) != problem.n_variables:
        raise ValueError("bit vector length must match the variable count")
    return {name: int(bit) for name, bit in zip(problem.variable_names, bits)}


def independent_feasibility_check(solver_name: str, bits: Sequence[int],
                                  problem: CombinatorialProblem,
                                  formulation: QuboFormulation) -> OptimizationVerification:
    """Step 47 -- independently check constraints and objective of returned bits.

    The check recomputes both the original objective and the penalized energy from
    the declared problem, flags violated constraints, and accepts the candidate only
    when feasible.  A separate guard refuses any direct policy application.
    """
    x = np.asarray(bits, dtype=float)
    if x.shape != (problem.n_variables,):
        raise ValueError("bit vector length must match the variable count")
    if np.any((x != 0.0) & (x != 1.0)):
        raise OptimizationPolicyError("returned configuration is not binary")
    violations = [index for index, value in
                  enumerate(problem.constraint_violations(x)) if abs(value) > 1e-9]
    feasible = not violations
    verification = OptimizationVerification(
        solver=solver_name, assignment=decode_assignment(bits, problem),
        feasible=feasible, objective_value=float(problem.objective_value(x)),
        violated_constraints=violations,
        recomputed_energy=float(formulation.energy(x) + formulation.constant_offset),
        accepted=feasible)
    return verification


def apply_to_cros_or_policy(verification: OptimizationVerification) -> None:
    """Refuse direct backend access to CROS scheduling or coordinator policy (Step 47)."""
    raise OptimizationPolicyError(
        "no optimization backend may directly alter CROS scheduling or coordinator "
        "policy; candidate solutions go through the normal approval path")
