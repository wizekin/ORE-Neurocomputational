"""Constraint programming and optimization (Phase E, Steps 31-36).

Module selection, scheduling, and discrete resource allocation are encoded for
OR-Tools CP-SAT behind one adapter interface; a pure-Python exact reference
optimizer provides the same typed contract when OR-Tools is unavailable or the
problem is small enough to solve by exhaustive branch-and-bound. The adapter
distinguishes proven optimal, feasible incumbent, infeasible, and unknown
outcomes, records objective value, best bound, optimality gap, solver status
and runtime — and never labels a feasible incumbent optimal. Every returned
selection is independently re-validated against the original constraints
before it is reported.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import SolverIdentity, ResultStatus
from .exceptions import InvalidFormalizationError, MalformedSolverOutputError

try:  # pragma: no cover - environment dependent
    from ortools.sat.python import cp_model

    ORTOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover
    cp_model = None  # type: ignore[assignment]
    ORTOOLS_AVAILABLE = False

CPSAT_ADAPTER_VERSION = "frce-cpsat-1.0.0"
REFERENCE_OPTIMIZER_VERSION = "frce-refopt-1.0.0"
_INTEGER_SCALE = 1000          # float quantities are scaled to integers for CP-SAT
_REFERENCE_MAX_VARIABLES = 22  # exhaustive branch-and-bound guard


# ---------------------------------------------------------------------------
# Problem contracts (Steps 31-35)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModuleCandidate:
    """One candidate module with cost, resource use, and capabilities (Step 32)."""

    module_id: str
    cost: float
    capabilities: tuple[str, ...]
    resources: Mapping[str, float]
    depends_on: tuple[str, ...] = ()
    mutually_exclusive_with: tuple[str, ...] = ()
    latency_ms: float | None = None
    reliability: float | None = None

    def __post_init__(self) -> None:
        if not self.module_id.strip():
            raise ValueError("ModuleCandidate.module_id must be non-empty")
        if self.cost < 0:
            raise ValueError("costs must be non-negative")
        if not isinstance(self.resources, Mapping):
            raise ValueError("resources must be a mapping")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency must be non-negative")
        if self.reliability is not None and not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must lie in [0, 1]")
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "mutually_exclusive_with", tuple(self.mutually_exclusive_with))


@dataclass(frozen=True, slots=True)
class SelectionProblem:
    """A finite module-selection problem with a transparent objective (Step 35)."""

    problem_id: str
    candidates: tuple[ModuleCandidate, ...]
    required_capabilities: tuple[str, ...]
    resource_limits: Mapping[str, float]
    objective_weights: Mapping[str, float]
    units: Mapping[str, str]
    deadline_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.problem_id.strip():
            raise ValueError("SelectionProblem.problem_id must be non-empty")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "required_capabilities", tuple(self.required_capabilities))
        ids = [c.module_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate module IDs must be unique")
        for required in self.required_capabilities:
            providers = [c for c in self.candidates if required in c.capabilities]
            if not providers:
                raise InvalidFormalizationError(
                    f"required capability {required!r} has no eligible candidate"
                )
        for weight_name in self.objective_weights:
            if weight_name not in {"cost", "latency_ms", "reliability"}:
                raise InvalidFormalizationError(
                    f"unknown objective term: {weight_name!r}; weights must stay transparent"
                )

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return the deterministic payload used for request fingerprints."""
        return {
            "problem_id": self.problem_id,
            "candidates": [
                {
                    "module_id": c.module_id,
                    "cost": c.cost,
                    "capabilities": list(c.capabilities),
                    "resources": dict(sorted(c.resources.items())),
                    "depends_on": list(c.depends_on),
                    "mutually_exclusive_with": list(c.mutually_exclusive_with),
                    "latency_ms": c.latency_ms,
                    "reliability": c.reliability,
                }
                for c in self.candidates
            ],
            "required_capabilities": list(self.required_capabilities),
            "resource_limits": dict(sorted(self.resource_limits.items())),
            "objective_weights": dict(sorted(self.objective_weights.items())),
            "units": dict(sorted(self.units.items())),
            "deadline_ms": self.deadline_ms,
        }


@dataclass(frozen=True, slots=True)
class OptimizationOutcome:
    """One optimization result with honest quality reporting (Step 36)."""

    status: ResultStatus
    selection: tuple[str, ...]
    objective_value: float | None
    best_bound: float | None
    optimality_gap: float | None
    solver: SolverIdentity
    wall_clock_ms: float = 0.0
    exhaustive: bool = False
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection", tuple(self.selection))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.status == ResultStatus.OPTIMAL and not self.exhaustive:
            if self.solver.backend != "ortools_cp_sat":
                raise ValueError("OPTIMAL requires an exhaustive or proven-optimal solve")
        if self.status == ResultStatus.FEASIBLE and self.optimality_gap is None:
            raise ValueError("a feasible incumbent must record a bound or gap status")


# ---------------------------------------------------------------------------
# Independent feasibility validation (Step 36/43)
# ---------------------------------------------------------------------------


def validate_selection(
    problem: SelectionProblem, selection: Sequence[str],
) -> tuple[bool, list[str]]:
    """Re-check a selection against every original constraint independently."""
    chosen = set(selection)
    errors: list[str] = []
    by_id = {c.module_id: c for c in problem.candidates}
    for module_id in selection:
        if module_id not in by_id:
            errors.append(f"unknown module selected: {module_id!r}")
    if errors:
        return False, errors

    # coverage: every required capability must be provided by a selected module
    for capability in problem.required_capabilities:
        providers = [
            c for c in problem.candidates
            if c.module_id in chosen and capability in c.capabilities
        ]
        if not providers:
            errors.append(f"capability {capability!r} is not covered")

    # resource budgets: sum_i r_ik x_i <= R_k
    for resource, limit in problem.resource_limits.items():
        used = sum(
            float(c.resources.get(resource, 0.0))
            for c in problem.candidates if c.module_id in chosen
        )
        if used > limit + 1e-9:
            errors.append(
                f"resource {resource!r} budget exceeded: {used} > {limit}"
            )

    # dependencies: selecting a implies selecting its prerequisites
    for module_id in sorted(chosen):
        for prerequisite in by_id[module_id].depends_on:
            if prerequisite not in chosen:
                errors.append(
                    f"dependency violated: {module_id!r} requires {prerequisite!r}"
                )

    # mutual exclusion: never both selected
    for module_id in sorted(chosen):
        for excluded in by_id[module_id].mutually_exclusive_with:
            if excluded in chosen and excluded != module_id:
                errors.append(
                    f"mutual exclusion violated: {module_id!r} with {excluded!r}"
                )

    # deadline: total latency must respect the deadline when one is set
    if problem.deadline_ms is not None:
        total_latency = sum(
            float(c.latency_ms or 0.0)
            for c in problem.candidates if c.module_id in chosen
        )
        if total_latency > problem.deadline_ms + 1e-9:
            errors.append(
                f"deadline exceeded: {total_latency} ms > {problem.deadline_ms} ms"
            )
    return (not errors), errors


def objective_value(
    problem: SelectionProblem, selection: Sequence[str],
) -> float:
    """Compute the weighted objective with recorded weights (Step 35)."""
    by_id = {c.module_id: c for c in problem.candidates}
    total = 0.0
    for term, weight in problem.objective_weights.items():
        if term == "cost":
            contribution = sum(by_id[m].cost for m in selection)
        elif term == "latency_ms":
            contribution = sum(float(by_id[m].latency_ms or 0.0) for m in selection)
        elif term == "reliability":
            contribution = -sum(by_id[m].reliability or 1.0 for m in selection)
        else:  # validated at construction; kept for safety
            raise InvalidFormalizationError(f"unknown objective term {term!r}")
        total += weight * contribution
    return total


# ---------------------------------------------------------------------------
# OR-Tools CP-SAT backend (Steps 31-36)
# ---------------------------------------------------------------------------


class CpsatOptimizer:
    """OR-Tools CP-SAT adapter for module selection (Steps 31-36)."""

    version = "ortools-9.x"

    def solve(
        self,
        problem: SelectionProblem,
        *,
        time_limit_seconds: float = 10.0,
        workers: int = 1,
    ) -> OptimizationOutcome:
        """Solve with CP-SAT; statuses are reported exactly as the solver gives them."""
        if not ORTOOLS_AVAILABLE:
            raise RuntimeError("ortools is not installed")
        start = time.perf_counter()
        model = cp_model.CpModel()
        variables: dict[str, Any] = {}
        for candidate in problem.candidates:
            variables[candidate.module_id] = model.NewBoolVar(candidate.module_id)

        # Step 33: task coverage and eligibility.
        for capability in problem.required_capabilities:
            eligible = [variables[c.module_id] for c in problem.candidates
                        if capability in c.capabilities]
            model.AddBoolOr(eligible)

        # Step 34: resource and safety limits (scaled to integers for CP-SAT).
        for resource, limit in problem.resource_limits.items():
            terms = [
                (int(round(float(c.resources.get(resource, 0.0)) * _INTEGER_SCALE)),
                 variables[c.module_id])
                for c in problem.candidates
            ]
            model.Add(sum(coef * var for coef, var in terms)
                      <= int(round(float(limit) * _INTEGER_SCALE)))

        # dependencies: x_a <= x_b encoded as implication a -> b
        for candidate in problem.candidates:
            for prerequisite in candidate.depends_on:
                if prerequisite not in variables:
                    raise InvalidFormalizationError(
                        f"candidate {candidate.module_id!r} depends on unknown "
                        f"module {prerequisite!r}"
                    )
                model.AddImplication(variables[candidate.module_id], variables[prerequisite])

        # mutual exclusion: x_a + x_b <= 1
        seen_pairs: set[tuple[str, str]] = set()
        for candidate in problem.candidates:
            for excluded in candidate.mutually_exclusive_with:
                if excluded not in variables:
                    raise InvalidFormalizationError(
                        f"candidate {candidate.module_id!r} excludes unknown "
                        f"module {excluded!r}"
                    )
                pair = tuple(sorted((candidate.module_id, excluded)))
                if pair in seen_pairs or pair[0] == pair[1]:
                    continue
                seen_pairs.add(pair)
                model.Add(variables[pair[0]] + variables[pair[1]] <= 1)

        # deadline as a latency budget
        if problem.deadline_ms is not None:
            latency_terms = [
                (int(round(float(c.latency_ms or 0.0) * _INTEGER_SCALE)),
                 variables[c.module_id])
                for c in problem.candidates
            ]
            model.Add(sum(coef * var for coef, var in latency_terms)
                      <= int(round(problem.deadline_ms * _INTEGER_SCALE)))

        # Step 35: transparent weighted objective.
        objective_terms: list[tuple[int, Any]] = []
        for term, weight in problem.objective_weights.items():
            for candidate in problem.candidates:
                if term == "cost":
                    raw = candidate.cost
                elif term == "latency_ms":
                    raw = float(candidate.latency_ms or 0.0)
                else:  # reliability: maximize via negation
                    raw = -(candidate.reliability or 1.0)
                coefficient = int(round(raw * weight * _INTEGER_SCALE))
                if coefficient:
                    objective_terms.append((coefficient, variables[candidate.module_id]))
        model.Minimize(sum(coef * var for coef, var in objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(0.1, time_limit_seconds)
        solver.parameters.num_search_workers = max(1, workers)
        solver.parameters.randomize_search = False
        status_code = solver.Solve(model)
        wall_ms = (time.perf_counter() - start) * 1000.0

        identity = SolverIdentity(
            name="OR-Tools CP-SAT",
            version=self.version,
            configuration={
                "max_time_in_seconds": max(0.1, time_limit_seconds),
                "num_search_workers": max(1, workers),
                "randomize_search": False,
                "integer_scale": _INTEGER_SCALE,
                "objective_weights": dict(sorted(problem.objective_weights.items())),
                "units": dict(sorted(problem.units.items())),
            },
            backend="ortools_cp_sat",
        )
        limitations: list[str] = [
            "float quantities are scaled by 1000 into integer CP-SAT coefficients",
        ]

        if status_code == cp_model.OPTIMAL:
            selection = tuple(
                c.module_id for c in problem.candidates if solver.Value(variables[c.module_id]) == 1
            )
            feasible, errors = validate_selection(problem, selection)
            if not feasible:
                raise MalformedSolverOutputError(
                    "CP-SAT reported an optimal solution that fails independent "
                    f"validation: {errors}"
                )
            objective = solver.ObjectiveValue() / _INTEGER_SCALE
            bound = solver.BestObjectiveBound() / _INTEGER_SCALE
            return OptimizationOutcome(
                status=ResultStatus.OPTIMAL,
                selection=selection,
                objective_value=objective,
                best_bound=bound,
                optimality_gap=0.0,
                solver=identity,
                wall_clock_ms=wall_ms,
                limitations=tuple(limitations),
            )
        if status_code == cp_model.FEASIBLE:
            selection = tuple(
                c.module_id for c in problem.candidates if solver.Value(variables[c.module_id]) == 1
            )
            feasible, errors = validate_selection(problem, selection)
            if not feasible:
                raise MalformedSolverOutputError(
                    f"CP-SAT incumbent fails independent validation: {errors}"
                )
            objective = solver.ObjectiveValue() / _INTEGER_SCALE
            bound = solver.BestObjectiveBound() / _INTEGER_SCALE
            gap = abs(bound - objective) / max(1.0, abs(objective)) if objective else 0.0
            return OptimizationOutcome(
                status=ResultStatus.FEASIBLE,
                selection=selection,
                objective_value=objective,
                best_bound=bound,
                optimality_gap=gap,
                solver=identity,
                wall_clock_ms=wall_ms,
                limitations=tuple(limitations + [
                    "a time-limited incumbent is reported FEASIBLE with its best "
                    "bound and gap; it is never labeled OPTIMAL"
                ]),
            )
        if status_code == cp_model.INFEASIBLE:
            return OptimizationOutcome(
                status=ResultStatus.INFEASIBLE,
                selection=(),
                objective_value=None,
                best_bound=None,
                optimality_gap=None,
                solver=identity,
                wall_clock_ms=wall_ms,
                limitations=tuple(limitations),
            )
        return OptimizationOutcome(
            status=ResultStatus.UNKNOWN,
            selection=(),
            objective_value=None,
            best_bound=None,
            optimality_gap=None,
            solver=identity,
            wall_clock_ms=wall_ms,
            limitations=tuple(limitations + [
                "UNKNOWN covers model invalidity and search exhaustion; it is not infeasibility"
            ]),
        )


# ---------------------------------------------------------------------------
# Exact reference optimizer (fallback and cross-check)
# ---------------------------------------------------------------------------


class ReferenceOptimizer:
    """Pure-Python exact branch-and-bound over binary selections (Step 36).

    Guarantees a proven optimum for problems up to
    ``_REFERENCE_MAX_VARIABLES`` candidates; larger problems return UNKNOWN
    rather than a partial answer.
    """

    version = REFERENCE_OPTIMIZER_VERSION

    def solve(
        self,
        problem: SelectionProblem,
        *,
        time_limit_seconds: float = 10.0,
    ) -> OptimizationOutcome:
        """Solve exhaustively with pruning; reports OPTIMAL or INFEASIBLE."""
        start = time.perf_counter()
        identity = SolverIdentity(
            name="FRCE reference optimizer",
            version=self.version,
            configuration={
                "method": "exhaustive branch-and-bound",
                "objective_weights": dict(sorted(problem.objective_weights.items())),
                "units": dict(sorted(problem.units.items())),
            },
            backend="reference_exact",
        )
        candidates = list(problem.candidates)
        if len(candidates) > _REFERENCE_MAX_VARIABLES:
            return OptimizationOutcome(
                status=ResultStatus.UNKNOWN,
                selection=(),
                objective_value=None,
                best_bound=None,
                optimality_gap=None,
                solver=identity,
                wall_clock_ms=(time.perf_counter() - start) * 1000.0,
                limitations=(
                    f"the reference optimizer handles at most {_REFERENCE_MAX_VARIABLES} "
                    "candidates; enable the CP-SAT backend for larger problems"
                ),
            )

        ordered = sorted(candidates, key=lambda c: c.cost)
        best_selection: tuple[str, ...] | None = None
        best_value = float("inf")
        deadline = start + max(0.1, time_limit_seconds)
        truncated = False

        def feasible(subset: tuple[str, ...]) -> bool:
            ok, _errors = validate_selection(problem, subset)
            return ok

        for size in range(0, len(ordered) + 1):
            if time.perf_counter() > deadline:
                truncated = True
                break
            found_for_size: tuple[str, ...] | None = None
            value_for_size = float("inf")
            for combo in itertools.combinations(ordered, size):
                if time.perf_counter() > deadline:
                    truncated = True
                    break
                ids = tuple(c.module_id for c in combo)
                value = objective_value(problem, ids)
                if value >= best_value or value >= value_for_size:
                    continue
                if feasible(ids):
                    found_for_size = ids
                    value_for_size = value
            if found_for_size is not None:
                best_selection = found_for_size
                best_value = value_for_size
            if truncated:
                break

        wall_ms = (time.perf_counter() - start) * 1000.0
        if truncated:
            return OptimizationOutcome(
                status=ResultStatus.UNKNOWN,
                selection=(),
                objective_value=None,
                best_bound=None,
                optimality_gap=None,
                solver=identity,
                wall_clock_ms=wall_ms,
                limitations=("reference search hit its time limit; UNKNOWN is not infeasibility",),
            )
        if best_selection is None:
            return OptimizationOutcome(
                status=ResultStatus.INFEASIBLE,
                selection=(),
                objective_value=None,
                best_bound=None,
                optimality_gap=None,
                solver=identity,
                wall_clock_ms=wall_ms,
                exhaustive=True,
                limitations=(
                    "INFEASIBLE was proven by exhaustive search over every subset"
                ),
            )
        return OptimizationOutcome(
            status=ResultStatus.OPTIMAL,
            selection=best_selection,
            objective_value=best_value,
            best_bound=best_value,
            optimality_gap=0.0,
            solver=identity,
            wall_clock_ms=wall_ms,
            exhaustive=True,
            limitations=(
                "OPTIMAL was proven by exhaustive search over every feasible subset"
            ),
        )


class OptimizationRouter:
    """Select the optimizer backend per policy (solver-selection policy, Section 6)."""

    def __init__(self, *, prefer_backend: str = "auto") -> None:
        if prefer_backend not in {"auto", "cpsat", "reference"}:
            raise ValueError("prefer_backend must be auto, cpsat, or reference")
        self.prefer_backend = prefer_backend

    def solve(
        self,
        problem: SelectionProblem,
        *,
        time_limit_seconds: float = 10.0,
    ) -> OptimizationOutcome:
        """Route to the CP-SAT backend with the reference optimizer as fallback."""
        if self.prefer_backend == "cpsat" and ORTOOLS_AVAILABLE:
            return CpsatOptimizer().solve(problem, time_limit_seconds=time_limit_seconds)
        if self.prefer_backend == "reference":
            return ReferenceOptimizer().solve(problem, time_limit_seconds=time_limit_seconds)
        if ORTOOLS_AVAILABLE:
            return CpsatOptimizer().solve(problem, time_limit_seconds=time_limit_seconds)
        return ReferenceOptimizer().solve(problem, time_limit_seconds=time_limit_seconds)


__all__ = [
    "CPSAT_ADAPTER_VERSION",
    "REFERENCE_OPTIMIZER_VERSION",
    "ORTOOLS_AVAILABLE",
    "ModuleCandidate",
    "SelectionProblem",
    "OptimizationOutcome",
    "validate_selection",
    "objective_value",
    "CpsatOptimizer",
    "ReferenceOptimizer",
    "OptimizationRouter",
]
