"""Phase F tests: candidate discovery, synthesis, comparison (Steps 40-44)."""
from __future__ import annotations

import pytest

from ore_hic import (
    AlgorithmDiscovery,
    AssignmentProblem,
    AssignmentReport,
    Candidate,
    CandidateFamily,
    CandidatePlan,
    ModuleSpec,
    TaskSpec,
    greedy_by_task,
)


@pytest.fixture(scope="module")
def discovered():
    discovery = AlgorithmDiscovery()

    class FakeModel:
        model_family = "TASK_ASSIGNMENT_ILP"
        artifact_id = "m-1"

    candidates = discovery.generate_candidates(FakeModel())
    comparisons = discovery.compare(candidates)
    selected_id, shortlist, deferred = discovery.select(candidates, comparisons)
    return discovery, candidates, comparisons, selected_id, shortlist, deferred


def _problem(**overrides):
    values = {
        "tasks": (TaskSpec("t1", 1.0), TaskSpec("t2", 2.0)),
        "modules": (ModuleSpec("m1", 4.0, 0.0), ModuleSpec("m2", 3.0, 1.0)),
        "eligibility": {("t1", "m1"): True, ("t1", "m2"): True, ("t2", "m1"): True, ("t2", "m2"): True},
        "capability": {("t1", "m1"): 0.9, ("t1", "m2"): 0.3, ("t2", "m1"): 0.4, ("t2", "m2"): 0.8},
        "costs": {("t1", "m1"): 0.1, ("t1", "m2"): 0.2, ("t2", "m1"): 0.2, ("t2", "m2"): 0.1},
    }
    values.update(overrides)
    return AssignmentProblem(**values)


def test_step40_candidates_from_multiple_families(discovered):
    _, candidates, _, _, _, _ = discovered
    families = {c.family for c in candidates}
    assert CandidateFamily.GREEDY_SELECTION in families
    assert CandidateFamily.EXHAUSTIVE_OPTIMAL in families
    assert CandidateFamily.HUNGARIAN_ASSIGNMENT in families


def test_step41_candidates_share_one_external_specification(discovered):
    _, candidates, _, _, _, _ = discovered
    specs = {c.plan.external_specification for c in candidates}
    assert len(specs) == 1, "candidates must be compared against the same specification"
    for candidate in candidates:
        assert candidate.plan.invariants
        assert candidate.plan.complexity
        assert candidate.plan.failure_behavior


def test_step42_synthesized_procedures_solve_valid_instances(discovered):
    _, candidates, _, _, _, _ = discovered
    problem = _problem()
    for candidate in candidates:
        if candidate.applicability != "APPLICABLE":
            continue
        report = candidate.procedure(problem)
        assert isinstance(report, AssignmentReport)
        assert report.status in {"COMPLETED", "FAILED"}
        if report.status == "COMPLETED":
            assert set(report.assignment) == {"t1", "t2"}


def test_step43_incompatible_candidates_are_rejected(discovered):
    _, _, comparisons, _, _, _ = discovered
    by_id = {c.candidate_id: c for c in comparisons}
    # Hungarian is incompatible with the general (multi-capacity) specification.
    assert by_id["cand-004"].verdict == "REJECTED"
    assert by_id["cand-004"].hard_constraints_satisfied is False


def test_step43_hard_violators_are_rejected_even_if_faster():
    discovery = AlgorithmDiscovery()

    def fast_but_wrong(problem: AssignmentProblem) -> AssignmentReport:
        # Violates eligibility and capacity deliberately.
        assignment = {t.task_id: problem.modules[0].module_id for t in problem.tasks}
        return AssignmentReport("COMPLETED", assignment, 0.0, ("ignores all constraints",))

    fast = Candidate(
        candidate_id="cand-fast",
        family=CandidateFamily.GREEDY_SELECTION,
        display_name="fast but unsafe",
        plan=CandidatePlan((), (), (), "O(1)", "none", "shared"),
        procedure=fast_but_wrong,
        applicability="APPLICABLE",
        inapplicable_reason="",
        estimated_time="O(1)",
        estimated_space="O(1)",
        assumptions=(),
        selection_rationale="speed",
    )
    comparisons = discovery.compare([fast], hard_ok={"cand-fast": False})
    assert comparisons[0].verdict == "REJECTED"
    selected, _, _ = discovery.select([fast], comparisons)
    assert selected is None, "a hard violator must never be selected"


def test_step44_selection_picks_a_verified_candidate(discovered):
    _, candidates, _comparisons, selected_id, shortlist, _deferred = discovered
    assert selected_id in {c.candidate_id for c in candidates}
    assert selected_id in shortlist
    applicable = {c.candidate_id for c in candidates if c.applicability == "APPLICABLE"}
    assert selected_id in applicable


def test_greedy_respects_hard_constraints_on_tight_instances():
    # m2 has only 1 unit of remaining capacity; t2 needs 2 -> must go to m1.
    problem = _problem()
    problem = AssignmentProblem(
        tasks=problem.tasks,
        modules=(ModuleSpec("m1", 4.0, 0.0), ModuleSpec("m2", 2.0, 1.0)),
        eligibility=problem.eligibility,
        capability=problem.capability,
        costs=problem.costs,
    )
    report = greedy_by_task(problem)
    assert report.status == "COMPLETED"
    assert report.assignment["t2"] == "m1"


def test_infeasible_instances_produce_structured_failure():
    problem = _problem(
        eligibility={("t1", "m1"): False, ("t1", "m2"): False, ("t2", "m1"): True, ("t2", "m2"): True}
    )
    report = greedy_by_task(problem)
    assert report.status == "FAILED"
    assert report.assignment == {}
    assert report.diagnostics  # failure reason recorded
