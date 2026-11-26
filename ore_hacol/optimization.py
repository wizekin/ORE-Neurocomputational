"""Classical optimization and QUBO/Ising adapters (integrated steps 43-45).

Combinatorial problems are formulated classically first with variables,
objective, constraints, and feasibility (step 43); a QUBO/Ising encoding is
built only as a candidate and its correspondence with the original feasible
solutions is verified (step 44); classical solvers (Hungarian baseline,
simulated annealing, tabu search, exact enumeration, optional CP-SAT via
OR-Tools when installed) are compared with recorded feasibility, optimality
gap, runtime, and scaling before any experimental QUBO promotion (step 45).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.optimize

#: OR-Tools CP-SAT is optional; its absence never blocks the classical baseline.
try:  # pragma: no cover - depends on environment
    from ortools.sat.python import cp_model as _cp_model  # type: ignore

    ORTOOLS_AVAILABLE = True
except Exception:  # pragma: no cover
    ORTOOLS_AVAILABLE = False


@dataclass(frozen=True)
class AssignmentProblem:
    """Rectangular assignment problem: each task needs exactly one resource,
    each resource serves at most one task (classical formulation, step 43)."""

    cost: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.cost, dtype=np.float64)
        if matrix.ndim != 2 or min(matrix.shape) < 1:
            raise ValueError("cost must be a non-empty 2-D array")
        object.__setattr__(self, "cost", matrix)

    @property
    def tasks(self) -> int:
        return int(self.cost.shape[0])

    @property
    def resources(self) -> int:
        return int(self.cost.shape[1])

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "assignment",
            "tasks": self.tasks,
            "resources": self.resources,
            "constraints": ["each task exactly one resource", "each resource at most one task"],
        }


def encode_assignment_qubo(problem: AssignmentProblem, penalty: float) -> dict[str, Any]:
    """Encode the assignment problem as min z^T Q z + c^T z (step 44).

    Task-exactly-one uses a squared penalty (double counts both unassigned and
    multi-assigned); resource-at-most-one penalizes pairwise double booking.
    Penalty strength must be scaled above the objective spread so penalties do
    not distort the intended optimum; feasibility is checked after solving.
    """
    cost = problem.cost
    tasks, resources = problem.tasks, problem.resources
    size = tasks * resources
    Q = np.zeros((size, size), dtype=np.float64)
    c = np.zeros(size, dtype=np.float64)

    def idx(task: int, resource: int) -> int:
        return task * resources + resource

    for task in range(tasks):
        for resource in range(resources):
            c[idx(task, resource)] += cost[task, resource]
        for r1 in range(resources):
            Q[idx(task, r1), idx(task, r1)] += -penalty
            for r2 in range(r1 + 1, resources):
                Q[idx(task, r1), idx(task, r2)] += penalty
                Q[idx(task, r2), idx(task, r1)] += penalty
    for resource in range(resources):
        for t1 in range(tasks):
            for t2 in range(t1 + 1, tasks):
                Q[idx(t1, resource), idx(t2, resource)] += penalty / 2.0
                Q[idx(t2, resource), idx(t1, resource)] += penalty / 2.0
    return {
        "Q": Q,
        "c": c,
        "penalty": float(penalty),
        "variable_map": {
            f"x_{t}_{r}": idx(t, r) for t in range(tasks) for r in range(resources)
        },
        "notes": [
            "feasibility must be checked after solving; encoding does not enforce it",
            "penalty must exceed the objective spread to avoid distorting the optimum",
        ],
    }


def qubo_energy(Q: np.ndarray, c: np.ndarray, z: np.ndarray) -> float:
    """Evaluate E(z) = z^T Q z + c^T z."""
    variables = np.asarray(z, dtype=np.float64).ravel()
    return float(variables @ (np.asarray(Q) @ variables) + np.asarray(c) @ variables)


def decode_assignment(
    z: np.ndarray, tasks: int, resources: int
) -> dict[str, Any]:
    """Decode a binary vector into an assignment and check feasibility."""
    variables = np.asarray(z).astype(int).ravel()
    expected = tasks * resources
    if variables.shape[0] != expected:
        raise ValueError("z length does not match tasks x resources")
    counts = variables.reshape(tasks, resources)
    assignment: list[int] = []
    for task in range(tasks):
        chosen = np.flatnonzero(counts[task] == 1)
        assignment.append(int(chosen[0]) if chosen.size == 1 else -1)
    resource_load = counts.sum(axis=0)
    feasible = all(slot >= 0 for slot in assignment) and bool(
        np.all(resource_load <= 1)
    )
    return {
        "assignment": assignment,
        "task_assignments": counts.tolist(),
        "resource_load": resource_load.tolist(),
        "feasible": feasible,
    }


def ising_from_qubo(Q: np.ndarray, c: np.ndarray) -> dict[str, Any]:
    """Map QUBO to Ising via z = (s + 1) / 2, s in {-1, +1} (step 44).

    E(z) = z^T Q z + c^T z expands to s^T J s + h^T s + constant with
    J = Q / 4, h = (Q 1 + c) / 2 and constant = (1^T Q 1 + 2 c^T 1) / 4.
    """
    matrix = np.asarray(Q, dtype=np.float64)
    linear = np.asarray(c, dtype=np.float64)
    size = matrix.shape[0]
    ones = np.ones(size)
    J = matrix / 4.0
    h = (matrix @ ones + linear) / 2.0
    constant = float((ones @ matrix @ ones + 2.0 * linear @ ones) / 4.0)
    return {"J": J, "h": h, "constant": constant}


def ising_energy(J: np.ndarray, h: np.ndarray, constant: float, s: np.ndarray) -> float:
    """Evaluate the Ising energy of a spin vector."""
    spins = np.asarray(s, dtype=np.float64).ravel()
    return float(spins @ (np.asarray(J) @ spins) + np.asarray(h) @ spins + constant)


def exact_qubo_solver(
    Q: np.ndarray,
    c: np.ndarray,
    *,
    max_variables: int = 20,
    chunk: int = 4096,
) -> dict[str, Any]:
    """Exhaustive enumeration for small instances (strong exact baseline)."""
    matrix = np.asarray(Q, dtype=np.float64)
    linear = np.asarray(c, dtype=np.float64)
    size = matrix.shape[0]
    if size > max_variables:
        return {
            "z": None,
            "energy": None,
            "evaluations": 0,
            "enumeration_infeasible": True,
            "note": f"exact enumeration capped at {max_variables} variables",
        }
    total = 1 << size
    best_z: np.ndarray | None = None
    best_energy = float("inf")
    evaluated = 0
    offsets = np.arange(size)
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        integers = np.arange(start, stop, dtype=np.int64)
        bits = ((integers[:, None] >> offsets[None, :]) & 1).astype(np.float64)
        energies = np.einsum("ij,jk,ik->i", bits, matrix, bits) + bits @ linear
        position = int(np.argmin(energies))
        evaluated += stop - start
        if energies[position] < best_energy:
            best_energy = float(energies[position])
            best_z = bits[position].astype(int)
    return {
        "z": best_z,
        "energy": best_energy,
        "evaluations": evaluated,
        "enumeration_infeasible": False,
    }


def simulated_annealing(
    Q: np.ndarray,
    c: np.ndarray,
    *,
    seed: int = 21,
    sweeps: int = 60,
    initial_temperature: float = 2.0,
    final_temperature: float = 0.05,
) -> dict[str, Any]:
    """Metropolis single-bit-flip simulated annealing for QUBO."""
    matrix = np.asarray(Q, dtype=np.float64)
    linear = np.asarray(c, dtype=np.float64)
    size = matrix.shape[0]
    rng = np.random.default_rng(seed)
    z = (rng.random(size) < 0.5).astype(np.float64)
    energy = qubo_energy(matrix, linear, z)
    best_z = z.copy()
    best_energy = energy
    temperatures = np.linspace(initial_temperature, final_temperature, max(1, sweeps))
    history: list[float] = []
    for temperature in temperatures:
        for _ in range(size):
            index = int(rng.integers(size))
            z[index] = 1.0 - z[index]
            candidate = qubo_energy(matrix, linear, z)
            delta = candidate - energy
            if delta <= 0.0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
                energy = candidate
                if energy < best_energy:
                    best_energy = energy
                    best_z = z.copy()
            else:
                z[index] = 1.0 - z[index]
        history.append(float(best_energy))
    return {
        "z": best_z.astype(int),
        "energy": best_energy,
        "history": history,
        "sweeps": sweeps,
        "seed": seed,
        "solver": "simulated_annealing",
    }


def tabu_search(
    Q: np.ndarray,
    c: np.ndarray,
    *,
    seed: int = 21,
    iterations: int = 200,
    tenure: int = 5,
) -> dict[str, Any]:
    """Greedy-descent tabu search for QUBO with a fixed tenure."""
    matrix = np.asarray(Q, dtype=np.float64)
    linear = np.asarray(c, dtype=np.float64)
    size = matrix.shape[0]
    rng = np.random.default_rng(seed)
    z = (rng.random(size) < 0.5).astype(np.float64)
    energy = qubo_energy(matrix, linear, z)
    best_z, best_energy = z.copy(), energy
    tabu_until = np.zeros(size, dtype=int)
    for step in range(iterations):
        deltas = []
        for index in range(size):
            z[index] = 1.0 - z[index]
            candidate = qubo_energy(matrix, linear, z)
            z[index] = 1.0 - z[index]
            deltas.append(candidate - energy)
        order = np.argsort(np.asarray(deltas))
        chosen = -1
        for index in order:
            if tabu_until[int(index)] <= step:
                chosen = int(index)
                break
        if chosen < 0:
            break
        z[chosen] = 1.0 - z[chosen]
        energy = qubo_energy(matrix, linear, z)
        tabu_until[chosen] = step + tenure
        if energy < best_energy:
            best_energy = energy
            best_z = z.copy()
    return {
        "z": best_z.astype(int),
        "energy": best_energy,
        "iterations": iterations,
        "tenure": tenure,
        "seed": seed,
        "solver": "tabu_search",
    }


def hungarian_assignment(problem: AssignmentProblem) -> dict[str, Any]:
    """Classical exact assignment baseline via the Hungarian algorithm."""
    rows, columns = scipy.optimize.linear_sum_assignment(problem.cost)
    objective = float(problem.cost[rows, columns].sum())
    assignment = [-1] * problem.tasks
    for task, resource in zip(rows, columns):
        assignment[int(task)] = int(resource)
    return {
        "assignment": assignment,
        "objective": objective,
        "solver": "hungarian_linear_sum_assignment",
        "feasible": True,
    }


def greedy_assignment(problem: AssignmentProblem) -> dict[str, Any]:
    """Greedy heuristic baseline: cheapest free resource per task."""
    cost = problem.cost.copy()
    tasks, resources = problem.tasks, problem.resources
    taken = np.zeros(resources, dtype=bool)
    assignment = [-1] * tasks
    order = np.argsort(cost.min(axis=1))
    for task in order:
        row = cost[task]
        best, best_value = -1, float("inf")
        for resource in range(resources):
            if not taken[resource] and row[resource] < best_value:
                best, best_value = resource, row[resource]
        if best >= 0:
            assignment[int(task)] = best
            taken[best] = True
    feasible = all(slot >= 0 for slot in assignment)
    objective = (
        float(sum(cost[t, assignment[t]] for t in range(tasks))) if feasible else None
    )
    return {
        "assignment": assignment,
        "objective": objective,
        "feasible": feasible,
        "solver": "greedy_cheapest_free",
    }


def cp_sat_assignment(problem: AssignmentProblem, time_limit: float = 5.0) -> dict[str, Any]:
    """Optional OR-Tools CP-SAT baseline; recorded when unavailable."""
    if not ORTOOLS_AVAILABLE:
        return {
            "solver": "cp_sat",
            "available": False,
            "feasible": False,
            "objective": None,
            "note": "OR-Tools not installed; Hungarian and heuristic baselines used instead",
        }
    model = _cp_model.CpModel()  # pragma: no cover - requires ortools
    variables = {}
    for task in range(problem.tasks):  # pragma: no cover
        for resource in range(problem.resources):
            variables[task, resource] = model.NewBoolVar(f"x_{task}_{resource}")
    for task in range(problem.tasks):  # pragma: no cover
        model.Add(sum(variables[task, resource] for resource in range(problem.resources)) == 1)
    for resource in range(problem.resources):  # pragma: no cover
        model.Add(sum(variables[task, resource] for task in range(problem.tasks)) <= 1)
    objective_terms = []  # pragma: no cover
    for task in range(problem.tasks):  # pragma: no cover
        for resource in range(problem.resources):
            weight = int(round(float(problem.cost[task, resource]) * 1000))
            objective_terms.append(weight * variables[task, resource])
    model.Minimize(sum(objective_terms))  # pragma: no cover
    solver = _cp_model.CpSolver()  # pragma: no cover
    solver.parameters.max_time_in_seconds = time_limit  # pragma: no cover
    status = solver.Solve(model)  # pragma: no cover
    if status not in (_cp_model.OPTIMAL, _cp_model.FEASIBLE):  # pragma: no cover
        return {"solver": "cp_sat", "available": True, "feasible": False, "objective": None}
    assignment = [  # pragma: no cover
        next(resource for resource in range(problem.resources) if solver.Value(variables[task, resource]))
        for task in range(problem.tasks)
    ]
    objective = float(sum(problem.cost[task, slot] for task, slot in enumerate(assignment)))
    return {  # pragma: no cover
        "solver": "cp_sat", "available": True, "feasible": True,
        "assignment": assignment, "objective": objective,
    }


def compare_solvers(
    problem: AssignmentProblem,
    *,
    penalty: float | None = None,
    seed: int = 21,
) -> dict[str, Any]:
    """Compare classical and QUBO solvers with recorded gaps (step 45)."""
    tasks, resources = problem.tasks, problem.resources
    if penalty is None:
        spread = float(np.abs(problem.cost).max() - np.abs(problem.cost).min()) + 1.0
        penalty = max(spread, 1.0) * 4.0
    encoded = encode_assignment_qubo(problem, penalty)
    Q, c = encoded["Q"], encoded["c"]

    hungarian = hungarian_assignment(problem)
    greedy = greedy_assignment(problem)
    exact = exact_qubo_solver(Q, c, max_variables=20)
    annealed = simulated_annealing(Q, c, seed=seed)
    tabu = tabu_search(Q, c, seed=seed)
    cp_sat = cp_sat_assignment(problem)

    def evaluate(solver_result: dict[str, Any], solver_name: str) -> dict[str, Any]:
        started = time.perf_counter()
        record = {
            "solver": solver_name,
            "feasible": False,
            "assignment_objective": None,
            "qubo_energy": None,
            "runtime_ms": 0.0,
        }
        variables = solver_result.get("z")
        if variables is None:
            record["note"] = solver_result.get("note", "solver unavailable")
            return record
        decode = decode_assignment(variables, tasks, resources)
        record["qubo_energy"] = float(solver_result.get("energy", qubo_energy(Q, c, variables)))
        record["feasible"] = decode["feasible"]
        if decode["feasible"]:
            record["assignment_objective"] = float(
                sum(problem.cost[t, slot] for t, slot in enumerate(decode["assignment"]))
            )
        record["runtime_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return record

    records = [
        evaluate(hungarian, "hungarian_baseline") | {
            "feasible": hungarian["feasible"],
            "assignment_objective": hungarian["objective"],
        },
        evaluate(greedy, "greedy_heuristic") | {
            "feasible": greedy["feasible"],
            "assignment_objective": greedy["objective"],
        },
        evaluate(annealed, "simulated_annealing"),
        evaluate(tabu, "tabu_search"),
        evaluate(exact, "exact_enumeration"),
        {"solver": "cp_sat", "feasible": cp_sat.get("feasible", False),
         "assignment_objective": cp_sat.get("objective"), "qubo_energy": None,
         "runtime_ms": 0.0, "note": cp_sat.get("note")},
    ]
    optimum = hungarian["objective"]
    for record in records:
        objective = record.get("assignment_objective")
        if objective is not None:
            gap = (objective - optimum) / max(1.0, abs(optimum))
            record["optimality_gap"] = float(gap)
        else:
            record["optimality_gap"] = None
    return {
        "problem": problem.describe(),
        "penalty": float(penalty),
        "records": records,
        "hungarian_optimum": optimum,
        "qubo_encoding": encoded["notes"],
        "verdict": (
            "classical baselines first; QUBO remains experimental until it beats "
            "the classical baseline end-to-end on feasibility, gap and runtime"
        ),
    }


def verify_penalty_preserves_optimum(
    problem: AssignmentProblem,
    penalty: float,
) -> dict[str, Any]:
    """Verify QUBO penalties do not distort the intended optimum (step 44)."""
    encoded = encode_assignment_qubo(problem, penalty)
    Q, c = encoded["Q"], encoded["c"]
    exact = exact_qubo_solver(Q, c, max_variables=20)
    hungarian = hungarian_assignment(problem)
    if exact["z"] is None:
        return {
            "verified": False,
            "note": "instance too large for exact verification in this helper",
        }
    decode = decode_assignment(exact["z"], problem.tasks, problem.resources)
    if not decode["feasible"]:
        return {
            "verified": False,
            "failure_mode": "qubo optimum is infeasible for the original problem",
            "decoded": decode,
        }
    objective = float(
        sum(problem.cost[t, slot] for t, slot in enumerate(decode["assignment"]))
    )
    matches = bool(np.allclose(objective, hungarian["objective"], rtol=1e-9, atol=1e-9))
    return {
        "verified": matches,
        "qubo_optimum_objective": objective,
        "classical_optimum": hungarian["objective"],
        "decoded_assignment": decode["assignment"],
        "failure_mode": None if matches else "penalty scaling distorted the objective",
    }
