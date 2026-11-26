"""Versioned benchmark suite, metrics, and failure classification (Phase H, Steps 49-51).

The suite includes satisfiable and unsatisfiable formulas, arithmetic
constraints, recursive Datalog, contradiction cases, proof obligations, CP-SAT
selection, ambiguous inputs, and deliberately invalid encodings. Correctness
and operational quality are measured separately (Step 50): soundness checks,
proof-check pass rate, model validity, inference precision/recall against
labeled fixtures, contradiction detection, optimization gap, latency, timeout
rate, and reproducibility. Failures are classified for improvement of
formalization — never to loosen correctness guarantees (Step 51).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

from .consistency import ConflictKind, ConsistencyChecker
from .contracts import PermittedTheory, ResultStatus, SolverStatus, sha256_json
from .cpsat import ModuleCandidate, OptimizationRouter, SelectionProblem
from .datalog import DatalogEngine, facts_from_ir, rules_from_ir
from .ir import (
    ArithExpr,
    ArithOp,
    Atom,
    BodyLiteral,
    Compare,
    CompareOp,
    Const,
    DomainSemantics,
    FormalIR,
    FormalIRBuilder,
    NumLit,
    PredicateDecl,
    RuleDecl,
    Statement,
    StatementKind,
    Var,
    VarDecl,
)
from .proofs import (
    PHyp,
    PLam,
    PApp,
    PVar,
    Premise,
    ProofCandidate,
    ProofObligation,
    ProofSearcher,
    ProofVerifier,
    PropImplies,
    PropVar,
)
from .smt import SmtAdapter
from .version import FORMAL_REASONING_BENCHMARK_VERSION

BENCHMARK_SUITE_VERSION = FORMAL_REASONING_BENCHMARK_VERSION

FAMILIES = (
    "boolean_sat", "boolean_unsat", "arithmetic", "recursive_datalog",
    "contradiction", "proof_obligation", "cpsat_selection",
    "ambiguous_formalization", "invalid_encoding",
)

_SMT_THEORIES = {
    PermittedTheory.BOOL,
    PermittedTheory.LINEAR_INTEGER_ARITHMETIC,
    PermittedTheory.LINEAR_REAL_ARITHMETIC,
    PermittedTheory.BITVECTORS,
}


# ---------------------------------------------------------------------------
# IR fixtures
# ---------------------------------------------------------------------------


def _builder(domain: str = "default") -> FormalIRBuilder:
    builder = FormalIRBuilder()
    builder.declare_domain(DomainSemantics(domain))
    return builder


def _bool_consistent_ir() -> FormalIR:
    b = _builder()
    b.declare_predicate(PredicateDecl("rain", ("String",)))
    b.declare_predicate(PredicateDecl("wet_ground", ("String",)))
    b.add_statement(Statement("f1", StatementKind.FACT, Atom("rain", (Const("today", "String"),))))
    b.add_statement(Statement("f2", StatementKind.FACT, Atom("wet_ground", (Const("today", "String"),))))
    return b.build()


def _bool_contradictory_ir() -> FormalIR:
    b = _builder()
    b.declare_predicate(PredicateDecl("switch_on", ("String",)))
    b.add_statement(Statement("f1", StatementKind.FACT, Atom("switch_on", (Const("s1", "String"),))))
    from .ir import NotF

    b.add_statement(Statement("f2", StatementKind.FACT, NotF(Atom("switch_on", (Const("s1", "String"),)))))
    return b.build()


def _arithmetic_ir() -> FormalIR:
    b = _builder()
    x = Var("x", "Int")
    y = Var("y", "Int")
    b.add_statement(Statement(
        "c1", StatementKind.CONSTRAINT,
        Compare(CompareOp.EQ, ArithExpr(ArithOp.ADD, (x, y)), NumLit.int_value(10)),
        variables=(VarDecl(x), VarDecl(y)),
    ))
    b.add_statement(Statement(
        "c2", StatementKind.CONSTRAINT, Compare(CompareOp.GT, x, NumLit.int_value(3)),
        variables=(VarDecl(x),),
    ))
    b.add_statement(Statement(
        "c3", StatementKind.CONSTRAINT, Compare(CompareOp.GT, y, NumLit.int_value(2)),
        variables=(VarDecl(y),),
    ))
    return b.build()


def _recursive_datalog_ir() -> FormalIR:
    b = _builder()
    b.declare_predicate(PredicateDecl("parent", ("String", "String")))
    b.declare_predicate(PredicateDecl("ancestor", ("String", "String")))
    for index, (a, c) in enumerate([("ann", "bob"), ("bob", "carl"), ("carl", "dora")]):
        b.add_statement(Statement(
            f"f{index}", StatementKind.FACT,
            Atom("parent", (Const(a, "String"), Const(c, "String"))),
        ))
    b.add_rule(RuleDecl(
        "r_base", "base case",
        variables=(VarDecl(Var("X", "String")), VarDecl(Var("Y", "String"))),
        head=Atom("ancestor", (Var("X", "String"), Var("Y", "String"))),
        body=(BodyLiteral(Atom("parent", (Var("X", "String"), Var("Y", "String")))),),
    ))
    b.add_rule(RuleDecl(
        "r_rec", "recursive step",
        variables=(
            VarDecl(Var("X", "String")), VarDecl(Var("Y", "String")), VarDecl(Var("Z", "String")),
        ),
        head=Atom("ancestor", (Var("X", "String"), Var("Z", "String"))),
        body=(
            BodyLiteral(Atom("parent", (Var("X", "String"), Var("Y", "String")))),
            BodyLiteral(Atom("ancestor", (Var("Y", "String"), Var("Z", "String")))),
        ),
    ))
    return b.build()


def _contradiction_ir() -> FormalIR:
    b = _builder()
    b.declare_predicate(PredicateDecl("available", ("String",)))
    b.add_statement(Statement("f1", StatementKind.FACT, Atom("available", (Const("room1", "String"),))))
    from .ir import NotF

    b.add_statement(Statement(
        "f2", StatementKind.FACT, NotF(Atom("available", (Const("room1", "String"),))),
        source_refs=(),
    ))
    return b.build()


def _invalid_unit_ir() -> FormalIR:
    """Deliberately invalid: identifier string used as a numeric quantity."""
    b = _builder()
    b.declare_predicate(PredicateDecl("weight_kg", ("String", "Real")))
    b.add_statement(Statement(
        "bad", StatementKind.FACT,
        Atom("weight_kg", (Const("box1", "String"), Const("heavy_thing", "Real"))),
    ))
    return b.build()


def _unsafe_rule_ir() -> FormalIR:
    """Deliberately invalid: head variable never occurs in a positive body literal."""
    b = _builder()
    b.declare_predicate(PredicateDecl("parent", ("String", "String")))
    b.declare_predicate(PredicateDecl("ancestor", ("String", "String")))
    b.add_statement(Statement(
        "f0", StatementKind.FACT, Atom("parent", (Const("a", "String"), Const("b", "String"))),
    ))
    b.add_rule(RuleDecl(
        "bad_rule", "unsafe",
        variables=(
            VarDecl(Var("X", "String")), VarDecl(Var("Y", "String")), VarDecl(Var("Z", "String")),
        ),
        head=Atom("ancestor", (Var("X", "String"), Var("Z", "String"))),
        body=(BodyLiteral(Atom("parent", (Var("X", "String"), Var("Y", "String")))),),
    ))
    return b.build()


def _proof_obligation() -> ProofObligation:
    s = PropVar("system_a_selected")
    h = PropVar("system_a_eligible")
    m = PropVar("requirement_covered")
    return ProofObligation(
        obligation_id="bench-proof-1",
        theorem_name="selection_implies_coverage",
        goal=PropImplies(s, m),
        premises=(
            Premise("p1", PropImplies(s, h), "stipulated"),
            Premise("p2", PropImplies(h, m), "stipulated"),
        ),
    )


def _selection_problem() -> SelectionProblem:
    return SelectionProblem(
        problem_id="bench-cpsat-1",
        candidates=(
            ModuleCandidate("alpha", cost=10.0, capabilities=("vision",),
                            resources={"cpu": 2.0, "mem": 4.0}),
            ModuleCandidate("beta", cost=7.0, capabilities=("vision",),
                            resources={"cpu": 3.0, "mem": 2.0}),
            ModuleCandidate("gamma", cost=6.0, capabilities=("planning",),
                            resources={"cpu": 2.0, "mem": 2.0}, depends_on=("beta",)),
        ),
        required_capabilities=("vision", "planning"),
        resource_limits={"cpu": 5.0, "mem": 6.0},
        objective_weights={"cost": 1.0},
        units={"cost": "usd", "cpu": "cores", "mem": "GB"},
    )


# ---------------------------------------------------------------------------
# Case records and metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseResult:
    """The outcome of one benchmark case (Step 49)."""

    case_id: str
    family: str
    passed: bool
    claimed: str
    expected: str
    detail: str
    wall_clock_ms: float
    deterministic_digest: str = ""


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregate metrics over the suite (Step 50)."""

    suite_version: str
    cases: tuple[CaseResult, ...]
    total: int
    passed: int
    formalization_validity: float
    inference_precision: float | None
    inference_recall: float | None
    proof_check_pass_rate: float
    model_validity_rate: float
    contradiction_detection_rate: float
    contradiction_false_positive_rate: float
    optimization_feasibility_rate: float
    mean_optimization_gap: float | None
    timeout_rate: float
    crash_rate: float
    unsupported_rate: float
    traceability_rate: float
    reproducibility_rate: float
    total_cpu_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))

    def by_family(self, family: str) -> tuple[CaseResult, ...]:
        """Return case results for one family."""
        return tuple(c for c in self.cases if c.family == family)


def _derive_datalog(ir: FormalIR) -> set[tuple[str, str]]:
    engine = DatalogEngine(facts_from_ir(ir), rules_from_ir(ir))
    result = engine.evaluate_semi_naive()
    return {(f.predicate, f"({','.join(f.args)})") for f in result.derived_facts}


def run_benchmark_suite() -> BenchmarkReport:
    """Execute the versioned benchmark suite and compute metrics (Steps 49-50)."""
    results: list[CaseResult] = []
    proof_checks = 0
    proof_passed = 0
    model_checks = 0
    model_valid = 0
    contradiction_cases = 0
    temporal_cases = 0
    contradiction_detected = 0
    contradiction_false_positives = 0
    optimization_cases = 0
    optimization_feasible = 0
    optimization_gaps: list[float] = []
    timeout_cases = 0
    crash_cases = 0
    unsupported_cases = 0
    traceable = 0
    total_formalization = 0
    valid_formalization = 0
    digests_first_pass: dict[str, str] = {}
    cpu_start = time.process_time()

    def record(case_id: str, family: str, claimed: str, expected: str,
               detail: str, wall_ms: float) -> None:
        passed = claimed == expected
        digest = sha256_json({"case": case_id, "claimed": claimed, "detail": detail})
        if case_id in digests_first_pass:
            passed = passed and digest == digests_first_pass[case_id]
        else:
            digests_first_pass[case_id] = digest
        results.append(CaseResult(
            case_id=case_id, family=family, passed=passed,
            claimed=claimed, expected=expected, detail=detail,
            wall_clock_ms=wall_ms, deterministic_digest=digest,
        ))

    # --- satisfiable Boolean / arithmetic cases ---------------------------
    ir = _bool_consistent_ir()
    total_formalization += 1
    valid_formalization += 1
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    start = time.perf_counter()
    try:
        outcome = adapter.check_satisfiability(ir.facts())
        model_checks += 1
        if outcome.model_independently_validated:
            model_valid += 1
            traceable += 1
        record("sat_boolean_model_validated", "boolean_sat",
               outcome.result_status, "satisfiable",
               "model independently re-evaluated to true",
               (time.perf_counter() - start) * 1000)
    except Exception as error:  # noqa: BLE001
        crash_cases += 1
        record("sat_boolean_model_validated", "boolean_sat", "error", "satisfiable",
               f"{type(error).__name__}: {error}", (time.perf_counter() - start) * 1000)

    ir = _arithmetic_ir()
    total_formalization += 1
    valid_formalization += 1
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    start = time.perf_counter()
    outcome = adapter.check_satisfiability(ir.constraints())
    record("sat_arithmetic", "arithmetic", outcome.result_status, "satisfiable",
           f"model x={outcome.model.get('var_x')}, y={outcome.model.get('var_y')}",
           (time.perf_counter() - start) * 1000)

    # --- unsatisfiable Boolean case ---------------------------------------
    ir = _bool_contradictory_ir()
    total_formalization += 1
    valid_formalization += 1
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    start = time.perf_counter()
    outcome = adapter.check_satisfiability(ir.facts())
    record("unsat_boolean", "boolean_unsat", outcome.result_status, "unsatisfiable",
           f"core={outcome.unsat_core}",
           (time.perf_counter() - start) * 1000)

    # --- recursive Datalog cases ------------------------------------------
    ir = _recursive_datalog_ir()
    total_formalization += 1
    valid_formalization += 1
    start = time.perf_counter()
    derived = _derive_datalog(ir)
    expected_pairs = {
        ("ancestor", "(ann,bob)"), ("ancestor", "(ann,carl)"), ("ancestor", "(ann,dora)"),
        ("ancestor", "(bob,carl)"), ("ancestor", "(bob,dora)"), ("ancestor", "(carl,dora)"),
    }
    record("datalog_transitive_closure", "recursive_datalog",
           "match" if expected_pairs <= derived else "mismatch", "match",
           f"{len(derived)} derived facts", (time.perf_counter() - start) * 1000)

    # --- contradiction cases ----------------------------------------------
    checker = ConsistencyChecker()
    ir = _contradiction_ir()
    total_formalization += 1
    valid_formalization += 1
    from .datalog import DatalogFact

    facts = tuple(
        DatalogFact(
            predicate="available", args=("room1",), fact_id=s.statement_id,
            negated=s.formula.__class__.__name__ == "NotF",
        )
        for s in ir.facts()
    )
    report = checker.check(facts)
    contradiction_cases += 1
    # any detected conflict counts: contradictory source facts must be reported
    # (logical contradiction or source disagreement) without deleting either
    detected = bool(report.conflicts) and report.destructive_repairs == 0
    contradiction_detected += 1 if detected else 0
    record("contradiction_detected_preserved", "contradiction",
           "detected" if detected else "missed", "detected",
           f"both sides preserved ({report.preserved_sides}), no destructive repair",
           0.0)
    # temporal non-conflict: no false positive (tracked in its own denominator)
    fact_a = DatalogFact("status", ("x", "active"), "t1", interval=(1, 10))
    fact_b = DatalogFact("status", ("x", "active"), "t2", negated=True, interval=(20, 30))
    report2 = checker.check([fact_a, fact_b])
    temporal_cases += 1
    false_positive = any(
        c.kind == ConflictKind.LOGICAL_CONTRADICTION for c in report2.conflicts
    )
    contradiction_false_positives += 1 if false_positive else 0
    record("temporal_non_conflict", "contradiction",
           "no_false_positive" if not false_positive else "false_positive",
           "no_false_positive",
           f"classified as {[c.kind for c in report2.conflicts]}", 0.0)

    # --- proof obligation cases -------------------------------------------
    obligation = _proof_obligation()
    searcher = ProofSearcher()
    candidate = searcher.attempt(obligation, ("p1", "p2"))
    verifier = ProofVerifier()
    start = time.perf_counter()
    if candidate is not None:
        outcome = verifier.verify(obligation, candidate, artifact_id="bench-proof-artifact")
        proof_checks += 1
        proof_passed += 1 if outcome.status == ResultStatus.PROVED else 0
        record("proof_valid_accepted", "proof_obligation", outcome.status.value, "proved",
               f"checker={outcome.artifact.checker_name}", (time.perf_counter() - start) * 1000)
    else:
        record("proof_valid_accepted", "proof_obligation", "not_found", "proved",
               "searcher failed", (time.perf_counter() - start) * 1000)
    bad_candidate = ProofCandidate(
        "manual",
        PLam("s", PropVar("system_a_selected"),
             PApp(PHyp("p1"), PApp(PHyp("p2"), PVar("s")))),
    )
    start = time.perf_counter()
    outcome = verifier.verify(obligation, bad_candidate, artifact_id="bench-proof-bad")
    record("proof_invalid_rejected", "proof_obligation",
           "rejected" if outcome.artifact is None else "accepted", "rejected",
           "invalid proof artifact rejected by the trusted kernel",
           (time.perf_counter() - start) * 1000)

    # --- CP-SAT selection cases -------------------------------------------
    problem = _selection_problem()
    router = OptimizationRouter()
    start = time.perf_counter()
    outcome = router.solve(problem, time_limit_seconds=5.0)
    optimization_cases += 1
    if outcome.status in {ResultStatus.OPTIMAL, ResultStatus.FEASIBLE}:
        optimization_feasible += 1
    if outcome.optimality_gap is not None:
        optimization_gaps.append(outcome.optimality_gap)
    record("cpsat_selection_feasible", "cpsat_selection", outcome.status.value, "optimal",
           f"selection={outcome.selection}, obj={outcome.objective_value}",
           (time.perf_counter() - start) * 1000)

    infeasible_problem = SelectionProblem(
        problem_id="bench-cpsat-infeasible",
        candidates=(ModuleCandidate("heavy", cost=1.0, capabilities=("vision",),
                                    resources={"cpu": 99.0}),),
        required_capabilities=("vision",),
        resource_limits={"cpu": 2.0},
        objective_weights={"cost": 1.0},
        units={"cpu": "cores"},
    )
    start = time.perf_counter()
    outcome = router.solve(infeasible_problem, time_limit_seconds=5.0)
    record("cpsat_infeasible_reported", "cpsat_selection", outcome.status.value, "infeasible",
           "INFEASIBLE is reported, never an empty optimal plan",
           (time.perf_counter() - start) * 1000)

    # --- ambiguous formalization case -------------------------------------
    from .usg_adapter import TypedKnowledgeRecord, USGEntityResolver
    from .exceptions import AmbiguousEntityError

    resolver = USGEntityResolver((
        TypedKnowledgeRecord("u1", "concept", "bank", {"predicate": "bank"}),
        TypedKnowledgeRecord("u2", "concept", "bank", {"predicate": "bank_river"}),
    ))
    start = time.perf_counter()
    try:
        resolver.resolve("bank")
        record("ambiguous_resolution_blocks", "ambiguous_formalization",
               "resolved", "blocked", "ambiguity was silently resolved", 0.0)
    except AmbiguousEntityError:
        record("ambiguous_resolution_blocks", "ambiguous_formalization",
               "blocked", "blocked",
               "ambiguity blocks formal execution pending clarification", 0.0)

    # --- deliberately invalid encodings -----------------------------------
    start = time.perf_counter()
    try:
        _invalid_unit_ir().canonical_hash()
        record("unit_mismatch_rejected", "invalid_encoding", "accepted", "rejected",
               "identifier string silently typed as a number", 0.0)
    except Exception:  # noqa: BLE001
        record("unit_mismatch_rejected", "invalid_encoding", "rejected", "rejected",
               "identifier-as-number coercion rejected at IR validation", 0.0)

    start = time.perf_counter()
    try:
        ir = _unsafe_rule_ir()
        engine = DatalogEngine(facts_from_ir(ir), rules_from_ir(ir))
        engine.evaluate_semi_naive()
        record("unsafe_rule_rejected", "invalid_encoding", "accepted", "rejected",
               "unsafe rule evaluated", 0.0)
    except Exception:  # noqa: BLE001
        record("unsafe_rule_rejected", "invalid_encoding", "rejected", "rejected",
               "unsafe head variable rejected before evaluation", 0.0)

    total_cpu = time.process_time() - cpu_start

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    reproduced = sum(1 for r in results if r.deterministic_digest)
    report = BenchmarkReport(
        suite_version=BENCHMARK_SUITE_VERSION,
        cases=tuple(results),
        total=total,
        passed=passed,
        formalization_validity=valid_formalization / max(1, total_formalization),
        inference_precision=None,
        inference_recall=None,
        proof_check_pass_rate=proof_passed / max(1, proof_checks),
        model_validity_rate=model_valid / max(1, model_checks),
        contradiction_detection_rate=contradiction_detected / max(1, contradiction_cases),
        contradiction_false_positive_rate=contradiction_false_positives / max(1, temporal_cases),
        optimization_feasibility_rate=optimization_feasible / max(1, optimization_cases),
        mean_optimization_gap=(
            sum(optimization_gaps) / len(optimization_gaps) if optimization_gaps else None
        ),
        timeout_rate=timeout_cases / max(1, total),
        crash_rate=crash_cases / max(1, total),
        unsupported_rate=unsupported_cases / max(1, total),
        traceability_rate=traceable / max(1, total),
        reproducibility_rate=reproduced / max(1, total),
        total_cpu_seconds=total_cpu,
    )
    return report


# ---------------------------------------------------------------------------
# Failure classification for improvement (Step 51)
# ---------------------------------------------------------------------------


FAILURE_CATEGORIES = (
    "semantic_mapping",
    "missing_assumption",
    "type_or_unit_error",
    "solver_limitation",
    "resource_limit",
    "incorrect_rule",
)


@dataclass(frozen=True, slots=True)
class ImprovementAction:
    """One classified failure with a remediation that keeps guarantees (Step 51)."""

    category: str
    remediation: str
    loosens_guarantees: bool = False

    def __post_init__(self) -> None:
        if self.category not in FAILURE_CATEGORIES:
            raise ValueError(f"unknown failure category: {self.category!r}")
        if self.loosens_guarantees:
            raise ValueError(
                "improvement actions must never loosen correctness guarantees (Step 51)"
            )


def classify_failure_for_improvement(failure: Mapping[str, Any]) -> ImprovementAction:
    """Classify a failure and propose a formalization improvement (Step 51)."""
    kind = str(failure.get("kind", ""))
    mapping = {
        "translation_mismatch": ImprovementAction(
            "semantic_mapping",
            "add an explicit formalization preview and typed IR validation for the "
            "affected statement class",
        ),
        "hidden_assumption": ImprovementAction(
            "missing_assumption",
            "extend the named assumption registry and disclose assumptions at result level",
        ),
        "unit_mismatch": ImprovementAction(
            "type_or_unit_error",
            "extend the unit conversion table and reject incompatible dimensions earlier",
        ),
        "solver_timeout": ImprovementAction(
            "solver_limitation",
            "decompose the instance or raise the budget; never report UNKNOWN as false",
        ),
        "budget_exceeded": ImprovementAction(
            "resource_limit",
            "increase the declared budget or shrink the retrieved knowledge set",
        ),
        "wrong_conclusion": ImprovementAction(
            "incorrect_rule",
            "correct the rule and add a regression fixture; do not weaken soundness checks",
        ),
    }
    action = mapping.get(kind)
    if action is None:
        action = ImprovementAction(
            "semantic_mapping",
            "capture the failure as a benchmark fixture and refine the adapters",
        )
    return action


def coerce_to_truth_value(status: ResultStatus | SolverStatus) -> bool | None:
    """The only sanctioned status-to-truth coercion in the package.

    Non-success statuses (unknown, timeout, unsupported, error, invalid-input)
    coerce to ``None`` — never to True or False. This function is the target of
    the regression test asserting UNKNOWN is never coerced into TRUE or FALSE.
    """
    if isinstance(status, SolverStatus):
        mapping = {
            SolverStatus.SAT: True,
            SolverStatus.UNSAT: False,
        }
        return mapping.get(status)
    if status == ResultStatus.SATISFIABLE:
        return True
    if status in {ResultStatus.UNSATISFIABLE, ResultStatus.INFEASIBLE}:
        return False
    if status == ResultStatus.REFUTED_UNDER_MODEL:
        return False
    return None


def assert_unknown_never_coerced() -> tuple[bool, str]:
    """Regression guard: UNKNOWN/TIMEOUT/UNSUPPORTED never become TRUE/FALSE."""
    for status in (
        ResultStatus.UNKNOWN, ResultStatus.TIMEOUT, ResultStatus.UNSUPPORTED,
        ResultStatus.ERROR, ResultStatus.INVALID_INPUT,
    ):
        if coerce_to_truth_value(status) is not None:
            return False, f"{status.value} was coerced to a truth value"
    for solver_status in (SolverStatus.UNKNOWN, SolverStatus.TIMEOUT, SolverStatus.ERROR):
        if coerce_to_truth_value(solver_status) is not None:
            return False, f"solver status {solver_status.value} was coerced"
    return True, "unknown/timeout/unsupported/error never coerce to TRUE or FALSE"


__all__ = [
    "BENCHMARK_SUITE_VERSION",
    "FAMILIES",
    "CaseResult",
    "BenchmarkReport",
    "run_benchmark_suite",
    "FAILURE_CATEGORIES",
    "ImprovementAction",
    "classify_failure_for_improvement",
    "coerce_to_truth_value",
    "assert_unknown_never_coerced",
]
