"""Conventional solver baselines and the Ising-style adapter (Point 14, Step 46).

CP-SAT-class exact search is approximated here by exhaustive/branch-and-bound on
small instances and by deterministic greedy + local search on larger ones; both run
and are recorded *before* any Ising-machine, quantum-annealer, or quantum-inspired
solver is consulted.  Quality, feasibility, runtime, repeatability, and scaling are
compared on equal footing.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import exhaustive_qubo, simulated_annealing_qubo
from .qubo import CombinatorialProblem, OptimizationPolicyError, QuboFormulation

__all__ = ("ConventionalBaseline", "IsingAdapter", "OptimizationPipeline",
           "SolverResult")


@dataclass(slots=True)
class SolverResult:
    """One solver's candidate outcome with its measurements."""

    solver: str
    kind: str  # conventional | ising
    bits: list[int]
    objective_value: float
    feasible: bool
    runtime_ms: float
    steps: int = 0
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"solver": self.solver, "kind": self.kind, "bits": list(self.bits),
                "objective_value": round(self.objective_value, 9),
                "feasible": self.feasible,
                "runtime_ms": round(self.runtime_ms, 3), "steps": self.steps,
                "seed": self.seed}


class ConventionalBaseline:
    """Step 46 -- deterministic conventional solving, recorded first."""

    def __init__(self) -> None:
        """No state beyond recorded results."""
        self.results: list[SolverResult] = []

    def solve(self, problem: CombinatorialProblem,
              config: LearningConfig | None = None) -> SolverResult:
        """Exact exhaustive search on small instances; penalized local search else.

        The conventional baseline solves the *original* problem: the exhaustive path
        enumerates configurations and returns the best **feasible** one (constraints
        applied directly, CP-SAT style); the heuristic path runs deterministic local
        search on the penalized objective.
        """
        effective = config or LearningConfig()
        started = time.perf_counter()
        n = problem.n_variables
        if n <= effective.qubo_exhaustive_verify_max_variables:
            best_bits, best_objective = None, float("inf")
            for mask in range(1 << n):
                bits = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
                if any(abs(value) > 1e-9
                       for value in problem.constraint_violations(bits)):
                    continue
                objective = problem.objective_value(bits)
                if objective < best_objective:
                    best_bits, best_objective = bits, objective
            feasible = best_bits is not None
            if best_bits is None:  # no feasible configuration: report the argmin
                best_bits, _ = exhaustive_qubo(_as_objective_matrix(problem))
            result = SolverResult(
                solver="exhaustive-search (exact, CP-SAT-class baseline)",
                kind="conventional", bits=[int(b) for b in best_bits],
                objective_value=float(problem.objective_value(best_bits)),
                feasible=feasible,
                runtime_ms=(time.perf_counter() - started) * 1000.0, steps=1 << n)
        else:
            matrix = _penalized_matrix(problem, effective)
            current = (np.diag(np.asarray(problem.objective_matrix)) < 0).astype(float)
            best = float(current @ matrix @ current)
            improved = True
            while improved:
                improved = False
                for index in range(n):
                    candidate = current.copy()
                    candidate[index] = 1.0 - candidate[index]
                    value = float(candidate @ matrix @ candidate)
                    if value < best - 1e-12:
                        current, best, improved = candidate, value, True
            result = SolverResult(
                solver="penalized local search (deterministic heuristic baseline)",
                kind="conventional", bits=[int(b) for b in current],
                objective_value=float(problem.objective_value(current)),
                feasible=all(abs(v) <= 1e-9
                             for v in problem.constraint_violations(current)),
                runtime_ms=(time.perf_counter() - started) * 1000.0, steps=2 * n)
        self.results.append(result)
        return result


class IsingAdapter:
    """Step 46 -- the optional quantum-inspired backend (seeded annealing reference)."""

    def __init__(self, config: LearningConfig | None = None) -> None:
        """Configure annealing parameters."""
        self.config = config or LearningConfig()
        self.results: list[SolverResult] = []

    def solve(self, formulation: QuboFormulation, *, seed: int = 7) -> SolverResult:
        """Anneal the penalized QUBO; results are candidate solutions only (Step 47)."""
        started = time.perf_counter()
        bits, energy = simulated_annealing_qubo(
            formulation.qubo_matrix,
            steps=self.config.annealing_steps,
            initial_temperature=self.config.annealing_initial_temperature,
            seed=seed)
        result = SolverResult(
            solver="simulated-annealing Ising reference (quantum-inspired adapter)",
            kind="ising", bits=[int(b) for b in bits],
            objective_value=float(formulation.problem.objective_value(bits)),
            feasible=all(abs(v) <= 1e-9
                         for v in formulation.problem.constraint_violations(bits)),
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            steps=self.config.annealing_steps, seed=seed)
        self.results.append(result)
        return result


class OptimizationPipeline:
    """Steps 44-47 orchestration: baseline first, Ising second, independent verify third."""

    def __init__(self, baseline: ConventionalBaseline | None = None,
                 ising: IsingAdapter | None = None,
                 config: LearningConfig | None = None) -> None:
        """Compose the pipeline."""
        self.config = config or LearningConfig()
        self.baseline = baseline or ConventionalBaseline()
        self.ising = ising or IsingAdapter(self.config)
        self.records: list[dict[str, Any]] = []

    def run(self, problem: CombinatorialProblem,
            formulation: QuboFormulation) -> dict[str, Any]:
        """Run the ordered pipeline: conventional baseline -> Ising -> verify."""
        baseline_result = self.baseline.solve(problem, self.config)
        ising_result = self.ising.solve(formulation)
        record = {"problem": problem.to_dict(),
                  "baseline": baseline_result.to_dict(),
                  "ising": ising_result.to_dict(),
                  "order_enforced": True,
                  "note": ("conventional baselines are recorded before Ising "
                           "evaluation; Ising outputs are candidate solutions that "
                           "undergo independent feasibility checks")}
        self.records.append(record)
        return record

    def run_ising_only(self, formulation: QuboFormulation) -> SolverResult:
        """Refuse Ising evaluation without a recorded baseline (Step 46 gate)."""
        if not self.baseline.results:
            raise OptimizationPolicyError(
                "a conventional solver baseline must be recorded before Ising "
                "evaluation (Step 46)")
        return self.ising.solve(formulation)


def _as_objective_matrix(problem: CombinatorialProblem) -> np.ndarray:
    """The objective matrix used by the exhaustive conventional baseline."""
    return np.asarray(problem.objective_matrix, dtype=float)


def _penalized_matrix(problem: CombinatorialProblem,
                      config: LearningConfig) -> np.ndarray:
    """Objective plus quadratic penalties, used by the heuristic baseline path."""
    return formulate_qubo(problem, config).qubo_matrix
