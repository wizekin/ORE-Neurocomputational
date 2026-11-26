"""Algorithm discovery, candidate synthesis, and comparison (spec Phase F).

Steps 40-44: generate candidate algorithm families compatible with the formal
specification (greedy selection, bipartite/Hungarian matching, exhaustive
optimum), build candidate-specific plans that keep the same external
specification, synthesize candidate procedures from templates, compare them
systematically on correctness obligations, cost, robustness, maintainability,
effort and ORE compatibility, and either select the best justified trade-off
or defer with a shortlist. Candidates that violate hard requirements are
rejected regardless of speed (spec Step 43).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

from .contracts import ArtifactKind, CandidateFamily, CanonicalArtifact, GateViolation
from .mathmodel import MathModel, assignment_objective_value
from .session import CompilationSession

# ---------------------------------------------------------------------------
# Problem instance and report
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    demand: float


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    module_id: str
    capacity: float
    load: float


@dataclass(frozen=True, slots=True)
class AssignmentProblem:
    """One concrete instance of the spec 3.3 task-assignment model."""

    tasks: tuple[TaskSpec, ...]
    modules: tuple[ModuleSpec, ...]
    eligibility: Mapping[tuple[str, str], bool]
    capability: Mapping[tuple[str, str], float]
    costs: Mapping[tuple[str, str], float]
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if not self.tasks or not self.modules:
            raise ValueError("assignment problem needs at least one task and one module")
        for task in self.tasks:
            if task.demand < 0 or not math.isfinite(task.demand):
                raise ValueError(f"task {task.task_id} has invalid demand")
        for module in self.modules:
            if module.capacity < 0 or module.load < 0:
                raise ValueError(f"module {module.module_id} has invalid capacity/load")

    def score(self, task_id: str, module_id: str, loads: Mapping[str, float]) -> float:
        return (
            self.alpha * self.capability.get((task_id, module_id), 0.0)
            - self.beta * loads.get(module_id, 0.0)
            - self.gamma * self.costs.get((task_id, module_id), 0.0)
        )

    def eligible(self, task_id: str, module_id: str) -> bool:
        return bool(self.eligibility.get((task_id, module_id), False))


@dataclass(frozen=True, slots=True)
class AssignmentReport:
    """The structured output required by the RequirementSpec."""

    status: str  # COMPLETED | FAILED
    assignment: Mapping[str, str]
    objective_value: float
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "assignment": dict(sorted(self.assignment.items())),
            "objective_value": self.objective_value,
            "diagnostics": list(self.diagnostics),
        }


# ---------------------------------------------------------------------------
# Candidate plans and procedures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CandidatePlan:
    """A candidate-specific refinement plan (Step 41)."""

    data_structures: tuple[str, ...]
    preconditions: tuple[str, ...]
    invariants: tuple[str, ...]
    complexity: str
    failure_behavior: str
    external_specification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_structures": list(self.data_structures),
            "preconditions": list(self.preconditions),
            "invariants": list(self.invariants),
            "complexity": self.complexity,
            "failure_behavior": self.failure_behavior,
            "external_specification": self.external_specification,
        }


Procedure = Callable[[AssignmentProblem], AssignmentReport]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One synthesized candidate algorithm (Steps 40-42)."""

    candidate_id: str
    family: CandidateFamily
    display_name: str
    plan: CandidatePlan
    procedure: Procedure
    applicability: str  # APPLICABLE | INAPPLICABLE
    inapplicable_reason: str
    estimated_time: str
    estimated_space: str
    assumptions: tuple[str, ...]
    selection_rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family.value,
            "display_name": self.display_name,
            "plan": self.plan.to_dict(),
            "applicability": self.applicability,
            "inapplicable_reason": self.inapplicable_reason,
            "estimated_time": self.estimated_time,
            "estimated_space": self.estimated_space,
            "assumptions": list(self.assumptions),
            "selection_rationale": self.selection_rationale,
        }


@dataclass(frozen=True, slots=True)
class CandidateComparison:
    """The systematic comparison of one candidate (Step 43)."""

    candidate_id: str
    hard_constraints_satisfied: bool
    correctness_obligations_met: int
    correctness_obligations_total: int
    time_cost: str
    space_cost: str
    robustness: str
    maintainability: str
    implementation_effort: str
    ore_compatibility: str
    verdict: str  # SELECTED | REJECTED | SHORTLISTED
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "hard_constraints_satisfied": self.hard_constraints_satisfied,
            "correctness_obligations_met": self.correctness_obligations_met,
            "correctness_obligations_total": self.correctness_obligations_total,
            "time_cost": self.time_cost,
            "space_cost": self.space_cost,
            "robustness": self.robustness,
            "maintainability": self.maintainability,
            "implementation_effort": self.implementation_effort,
            "ore_compatibility": self.ore_compatibility,
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


class CandidateSet(CanonicalArtifact):
    """Canonical artifact: candidate algorithms and their comparison."""

    kind = ArtifactKind.CANDIDATE_SET

    def __init__(
        self,
        *,
        artifact_id: str,
        model_id: str,
        candidates: Iterable[Candidate],
        comparisons: Iterable[CandidateComparison],
        selected_id: str | None,
        shortlist: Iterable[str],
        deferred: bool,
        version: int = 1,
    ) -> None:
        self.artifact_id = artifact_id
        self.model_id = model_id
        self.candidates = tuple(candidates)
        self.comparisons = tuple(comparisons)
        self.selected_id = selected_id
        self.shortlist = tuple(shortlist)
        self.deferred = deferred
        self.version = version

    def selected(self) -> Candidate | None:
        if self.selected_id is None:
            return None
        return next((c for c in self.candidates if c.candidate_id == self.selected_id), None)

    def content_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "model_id": self.model_id,
            "candidates": [c.to_dict() for c in self.candidates],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "selected_id": self.selected_id,
            "shortlist": list(self.shortlist),
            "deferred": self.deferred,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Template procedures (Step 42: template-based synthesis)
# ---------------------------------------------------------------------------

def _infeasible(reason: str) -> AssignmentReport:
    return AssignmentReport("FAILED", {}, 0.0, (reason,))


def greedy_by_task(problem: AssignmentProblem) -> AssignmentReport:
    """Assign each task, in canonical order, to its best feasible module.

    Tasks are processed by descending demand (ties by id) so the outcome
    depends only on the task SET, not on input order: the algorithm is
    structurally permutation-invariant, matching acceptance criterion
    acc-006. Placing larger demands first also reduces false-negative
    infeasibility on tight instances.
    """
    loads = {m.module_id: m.load for m in problem.modules}
    remaining = {m.module_id: m.capacity - m.load for m in problem.modules}
    assignment: dict[str, str] = {}
    ordered_tasks = sorted(problem.tasks, key=lambda t: (-t.demand, t.task_id))
    for task in ordered_tasks:
        best_module: str | None = None
        best_score = -math.inf
        for module in problem.modules:
            if not problem.eligible(task.task_id, module.module_id):
                continue
            if remaining[module.module_id] + 1e-9 < task.demand:
                continue
            score = problem.score(task.task_id, module.module_id, loads)
            if score > best_score or (score == best_score and best_module is not None
                                      and module.module_id < best_module):
                best_score = score
                best_module = module.module_id
        if best_module is None:
            return _infeasible(
                f"no eligible module with remaining capacity for task '{task.task_id}'"
            )
        assignment[task.task_id] = best_module
        remaining[best_module] -= task.demand
        loads[best_module] += task.demand
    objective = assignment_objective_value(
        assignment,
        capability=dict(problem.capability),
        loads=loads,
        costs=dict(problem.costs),
        alpha=problem.alpha,
        beta=problem.beta,
        gamma=problem.gamma,
    )
    return AssignmentReport("COMPLETED", assignment, objective, ())


def greedy_by_pair(problem: AssignmentProblem) -> AssignmentReport:
    """Globally greedy: best (task, module) pairs first, capacity-aware."""
    loads = {m.module_id: m.load for m in problem.modules}
    remaining = {m.module_id: m.capacity - m.load for m in problem.modules}
    assigned: dict[str, str] = {}
    pairs = [
        (problem.score(t.task_id, m.module_id, loads), t, m)
        for t in problem.tasks
        for m in problem.modules
        if problem.eligible(t.task_id, m.module_id)
    ]
    pairs.sort(key=lambda item: (-item[0], item[1].task_id, item[2].module_id))
    for score, task, module in pairs:
        if task.task_id in assigned or remaining[module.module_id] + 1e-9 < task.demand:
            continue
        assigned[task.task_id] = module.module_id
        remaining[module.module_id] -= task.demand
        loads[module.module_id] += task.demand
    if len(assigned) != len(problem.tasks):
        missing = [t.task_id for t in problem.tasks if t.task_id not in assigned]
        return _infeasible(f"greedy pair selection left tasks unassigned: {missing}")
    objective = assignment_objective_value(
        assigned,
        capability=dict(problem.capability),
        loads=loads,
        costs=dict(problem.costs),
        alpha=problem.alpha,
        beta=problem.beta,
        gamma=problem.gamma,
    )
    return AssignmentReport("COMPLETED", assigned, objective, ())


def exhaustive_optimal(problem: AssignmentProblem) -> AssignmentReport:
    """Exact optimum by enumerating every assignment map (small instances)."""
    if len(problem.modules) ** len(problem.tasks) > 200_000:
        return _infeasible("instance too large for exhaustive enumeration")
    loads = {m.module_id: m.load for m in problem.modules}
    best: tuple[float, dict[str, str]] | None = None
    for choice in product(problem.modules, repeat=len(problem.tasks)):
        assignment = {t.task_id: m.module_id for t, m in zip(problem.tasks, choice)}
        used = {m.module_id: 0.0 for m in problem.modules}
        ok = True
        for task, module in zip(problem.tasks, choice):
            if not problem.eligible(task.task_id, module.module_id):
                ok = False
                break
            used[module.module_id] += task.demand
        if not ok:
            continue
        for module in problem.modules:
            if module.capacity + 1e-9 < module.load + used[module.module_id]:
                ok = False
                break
        if not ok:
            continue
        objective = assignment_objective_value(
            assignment,
            capability=dict(problem.capability),
            loads=loads,
            costs=dict(problem.costs),
            alpha=problem.alpha,
            beta=problem.beta,
            gamma=problem.gamma,
        )
        if best is None or objective > best[0] + 1e-12:
            best = (objective, assignment)
    if best is None:
        return _infeasible("no feasible assignment exists")
    return AssignmentReport("COMPLETED", best[1], best[0], ("exhaustive enumeration",))


def hungarian_assignment(problem: AssignmentProblem) -> AssignmentReport:
    """Classic Hungarian algorithm; applies only to unit-demand unit-capacity instances."""
    if not (all(abs(t.demand - 1.0) < 1e-9 for t in problem.tasks)
            and all(abs(m.capacity - m.load - 1.0) < 1e-9 for m in problem.modules)):
        return _infeasible(
            "Hungarian candidate inapplicable: non-unit demands or capacities "
            "(multi-capacity assignment is not a square matching problem)"
        )
    n = len(problem.tasks)
    m = len(problem.modules)
    size = max(n, m)
    loads = {mod.module_id: mod.load for mod in problem.modules}
    big = 1e9
    cost = [[big] * size for _ in range(size)]
    for i, task in enumerate(problem.tasks):
        for j, module in enumerate(problem.modules):
            if problem.eligible(task.task_id, module.module_id):
                cost[i][j] = -problem.score(task.task_id, module.module_id, loads)
    # Hungarian algorithm (square cost matrix, O(n^3)).
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assignment: dict[str, str] = {}
    for j in range(1, size + 1):
        if p[j] and p[j] <= n and j <= m:
            task = problem.tasks[p[j] - 1]
            module = problem.modules[j - 1]
            assignment[task.task_id] = module.module_id
    if len(assignment) != n:
        return _infeasible("Hungarian matching could not place every task")
    objective = assignment_objective_value(
        assignment,
        capability=dict(problem.capability),
        loads=loads,
        costs=dict(problem.costs),
        alpha=problem.alpha,
        beta=problem.beta,
        gamma=problem.gamma,
    )
    return AssignmentReport("COMPLETED", assignment, objective, ("hungarian matching",))


_SHARED_SPEC = (
    "inputs: tasks, modules, eligibility, capability, costs; "
    "outputs: assignment map and AssignmentReport; hard constraints: exactly-one, capacity, eligibility"
)


class AlgorithmDiscovery:
    """Generate, synthesize, and compare candidates (Steps 40-44)."""

    def generate_candidates(self, model: MathModel) -> tuple[Candidate, ...]:
        """Steps 40-42: families selected by problem structure."""
        if model.model_family != "TASK_ASSIGNMENT_ILP":
            raise GateViolation("discovery currently implements the TASK_ASSIGNMENT_ILP family only")
        return (
            Candidate(
                candidate_id="cand-001",
                family=CandidateFamily.GREEDY_SELECTION,
                display_name="greedy-by-task selection",
                plan=CandidatePlan(
                    data_structures=("dict task->module", "dict module->remaining capacity"),
                    preconditions=("non-empty tasks/modules", "finite scores"),
                    invariants=("exactly-one per processed task", "capacity never negative"),
                    complexity="O(N*M) time, O(N+M) space",
                    failure_behavior="structured FAILED report naming the blocking task",
                    external_specification=_SHARED_SPEC,
                ),
                procedure=greedy_by_task,
                applicability="APPLICABLE",
                inapplicable_reason="",
                estimated_time="O(N*M)",
                estimated_space="O(N+M)",
                assumptions=("greedy choices do not need to be globally optimal"),
                selection_rationale="fast, simple, always respects hard constraints",
            ),
            Candidate(
                candidate_id="cand-002",
                family=CandidateFamily.GREEDY_SELECTION,
                display_name="greedy-by-pair selection",
                plan=CandidatePlan(
                    data_structures=("sorted pair list", "dict module->remaining capacity"),
                    preconditions=("non-empty tasks/modules",),
                    invariants=("capacity never negative",),
                    complexity="O(N*M log(N*M)) time, O(N*M) space",
                    failure_behavior="FAILED report listing unassigned tasks",
                    external_specification=_SHARED_SPEC,
                ),
                procedure=greedy_by_pair,
                applicability="APPLICABLE",
                inapplicable_reason="",
                estimated_time="O(N*M log(N*M))",
                estimated_space="O(N*M)",
                assumptions=("globally best pairs first improves average quality"),
                selection_rationale="slightly better average quality, higher memory",
            ),
            Candidate(
                candidate_id="cand-003",
                family=CandidateFamily.EXHAUSTIVE_OPTIMAL,
                display_name="exhaustive enumeration optimum",
                plan=CandidatePlan(
                    data_structures=("iterator over M^N assignment maps",),
                    preconditions=("small instance: M^N <= 200000",),
                    invariants=("optimal by definition when it completes",),
                    complexity="O(M^N * N) time, O(N) space",
                    failure_behavior="FAILED report when instance too large or infeasible",
                    external_specification=_SHARED_SPEC,
                ),
                procedure=exhaustive_optimal,
                applicability="APPLICABLE",
                inapplicable_reason="",
                estimated_time="O(M^N)",
                estimated_space="O(N)",
                assumptions=("small instances only"),
                selection_rationale="exact optimum; usable as trusted reference for differential testing",
            ),
            Candidate(
                candidate_id="cand-004",
                family=CandidateFamily.HUNGARIAN_ASSIGNMENT,
                display_name="Hungarian assignment",
                plan=CandidatePlan(
                    data_structures=("square cost matrix", "dual potentials"),
                    preconditions=("unit demands and unit remaining capacities",),
                    invariants=("perfect matching on the padded square matrix",),
                    complexity="O(n^3) time, O(n^2) space",
                    failure_behavior="FAILED report when the instance is not a square matching",
                    external_specification=_SHARED_SPEC,
                ),
                procedure=hungarian_assignment,
                applicability="INAPPLICABLE",
                inapplicable_reason=(
                    "the formal specification admits arbitrary real capacities and demands; "
                    "Hungarian matching only solves unit-capacity square matchings, so this "
                    "family is incompatible with the general specification (spec Step 43: "
                    "generate only candidates compatible with the formal specification)"
                ),
                estimated_time="O(n^3)",
                estimated_space="O(n^2)",
                assumptions=("unit-demand, unit-capacity instances only"),
                selection_rationale="exact for unit-capacity matchings; inapplicable to general capacities",
            ),
        )

    def compare(
        self,
        candidates: Sequence[Candidate],
        *,
        obligations_met: Mapping[str, tuple[int, int]] | None = None,
        hard_ok: Mapping[str, bool] | None = None,
    ) -> tuple[CandidateComparison, ...]:
        """Step 43: systematic comparison; hard violators and incompatible
        candidates are rejected outright regardless of speed."""
        obligations_met = obligations_met or {}
        hard_ok = hard_ok or {}
        comparisons: list[CandidateComparison] = []
        for candidate in candidates:
            met, total = obligations_met.get(candidate.candidate_id, (0, 0))
            satisfied = hard_ok.get(candidate.candidate_id, True)
            if candidate.applicability != "APPLICABLE":
                comparisons.append(
                    CandidateComparison(
                        candidate_id=candidate.candidate_id,
                        hard_constraints_satisfied=False,
                        correctness_obligations_met=0,
                        correctness_obligations_total=0,
                        time_cost=candidate.estimated_time,
                        space_cost=candidate.estimated_space,
                        robustness="inapplicable",
                        maintainability="n/a",
                        implementation_effort="n/a",
                        ore_compatibility="n/a",
                        verdict="REJECTED",
                        notes=(candidate.inapplicable_reason,),
                    )
                )
                continue
            comparisons.append(
                CandidateComparison(
                    candidate_id=candidate.candidate_id,
                    hard_constraints_satisfied=satisfied,
                    correctness_obligations_met=met,
                    correctness_obligations_total=total,
                    time_cost=candidate.estimated_time,
                    space_cost=candidate.estimated_space,
                    robustness=(
                        "high" if satisfied else "violates hard constraints"
                    ) + "; may report infeasibility on tight feasible instances (false negatives)"
                    if candidate.family is CandidateFamily.GREEDY_SELECTION and satisfied
                    else ("high" if satisfied else "violates hard constraints"),
                    maintainability="high" if candidate.family is CandidateFamily.GREEDY_SELECTION else "medium",
                    implementation_effort="low" if candidate.family is CandidateFamily.GREEDY_SELECTION else "medium",
                    ore_compatibility="pure Python, no external dependencies",
                    verdict="REJECTED" if not satisfied else "SHORTLISTED",
                    notes=() if satisfied else ("hard-constraint violation: candidate rejected regardless of speed",),
                )
            )
        return tuple(comparisons)

    def select(
        self,
        candidates: Sequence[Candidate],
        comparisons: Sequence[CandidateComparison],
        *,
        defer_threshold: float = 0.10,
    ) -> tuple[str | None, tuple[str, ...], bool]:
        """Step 44: select the best justified trade-off or defer with a shortlist."""
        alive = [
            (c, comp) for c, comp in zip(candidates, comparisons)
            if comp.verdict != "REJECTED" and c.applicability == "APPLICABLE"
        ]
        if not alive:
            return None, (), True

        def effort_key(item: tuple[Candidate, CandidateComparison]) -> tuple[int, int, str]:
            candidate, comp = item
            simplicity = 0 if candidate.family is CandidateFamily.GREEDY_SELECTION else 1
            return (
                simplicity,
                -comp.correctness_obligations_met,
                candidate.candidate_id,
            )

        ranked = sorted(alive, key=effort_key)
        best = ranked[0]
        # Defer when the top two differ materially in speed/optimality claims.
        if len(ranked) > 1 and best[0].family is not CandidateFamily.GREEDY_SELECTION:
            return None, tuple(c.candidate_id for c, _ in ranked[:3]), True
        return best[0].candidate_id, (best[0].candidate_id,), False

    def discover(
        self,
        session: CompilationSession,
        model: MathModel,
        *,
        obligations_met: Mapping[str, tuple[int, int]] | None = None,
        hard_ok: Mapping[str, bool] | None = None,
    ) -> CandidateSet:
        """Run Steps 40-44 and register the CandidateSet artifact."""
        candidates = self.generate_candidates(model)
        comparisons = self.compare(candidates, obligations_met=obligations_met, hard_ok=hard_ok)
        selected_id, shortlist, deferred = self.select(candidates, comparisons)
        result = CandidateSet(
            artifact_id=session.next_artifact_id(ArtifactKind.CANDIDATE_SET),
            model_id=model.artifact_id,
            candidates=candidates,
            comparisons=comparisons,
            selected_id=selected_id,
            shortlist=shortlist,
            deferred=deferred,
        )
        session.register(result)
        session.ledger.record(
            stage="F",
            step_id="HIC40-44",
            description="Candidate discovery, synthesis, comparison, selection",
            inputs=[model],
            outputs=[result],
            rule_references={"HIC-DISC-1", "HIC-DISC-2"},
        )
        return result
