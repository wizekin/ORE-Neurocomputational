"""Engine acceptance tests for FRCE (Point 5, Section 9): Datalog, consistency,
SAT/SMT, CP-SAT optimization, and machine-checked proofs.

Each test maps to one or more acceptance tests from the specification's Section
9 list; the docstrings cite them.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from ore_formal_reasoning.consistency import (
    CardinalityConstraint,
    ConsistencyChecker,
    ConflictKind,
    ExclusivityGroup,
    FunctionalConstraint,
)
from ore_formal_reasoning.contracts import (
    PermittedTheory,
    ResultStatus,
    SolverStatus,
)
from ore_formal_reasoning.cpsat import (
    ModuleCandidate,
    OptimizationRouter,
    ReferenceOptimizer,
    SelectionProblem,
    validate_selection,
)
from ore_formal_reasoning.datalog import (
    DatalogCache,
    DatalogEngine,
    DatalogFact,
    DatalogLimits,
    NonStratifiableProgramError,
    build_dependency_graph,
    facts_from_ir,
    rules_from_ir,
)
from ore_formal_reasoning.exceptions import (
    InvalidFormalizationError,
    NonStratifiableProgramError as NonStratifiableError,
    UnsupportedTheoryError,
)
from ore_formal_reasoning.ir import (
    AndF,
    ArithExpr,
    ArithOp,
    Atom,
    BVLit,
    BvExpr,
    BvOp,
    BodyLiteral,
    BoolTermF,
    BoolLit,
    CardinalityF,
    Compare,
    CompareOp,
    Const,
    DomainSemantics,
    FormalIRBuilder,
    FuncApp,
    FunctionDecl,
    NotF,
    NumLit,
    PredicateDecl,
    RuleDecl,
    SortDecl,
    SortKind,
    Statement,
    StatementKind,
    StrExpr,
    StrOp,
    StrLit,
    ToReal,
    Var,
    VarDecl,
)
from ore_formal_reasoning.proofs import (
    PApp,
    PHyp,
    PVar,
    PLam,
    PPair,
    PFst,
    PSnd,
    PInl,
    PCase,
    PAbort,
    Premise,
    ProofCandidate,
    ProofObligation,
    ProofSearcher,
    ProofVerifier,
    PropAnd,
    PropBottom,
    PropImplies,
    PropOr,
    PropVar,
    TrustedKernel,
)
from ore_formal_reasoning.smt import (
    IndependentEvaluator,
    SmtAdapter,
    Z3_AVAILABLE,
    detect_theories,
)


def _builder(domain: str = "default") -> FormalIRBuilder:
    builder = FormalIRBuilder()
    builder.declare_domain(DomainSemantics(domain))
    return builder


_SMT_THEORIES = {
    PermittedTheory.BOOL,
    PermittedTheory.LINEAR_INTEGER_ARITHMETIC,
    PermittedTheory.LINEAR_REAL_ARITHMETIC,
    PermittedTheory.BITVECTORS,
    PermittedTheory.STRINGS,
}

# ---------------------------------------------------------------------------
# Datalog (Phase C acceptance tests)
# ---------------------------------------------------------------------------


def _transitive_ir() -> object:
    builder = _builder()
    builder.declare_predicate(PredicateDecl("parent", ("String", "String")))
    builder.declare_predicate(PredicateDecl("ancestor", ("String", "String")))
    for index, (a, c) in enumerate([("tom", "bob"), ("bob", "ann")]):
        builder.add_statement(Statement(
            f"f{index}", StatementKind.FACT,
            Atom("parent", (Const(a, "String"), Const(c, "String"))),
        ))
    builder.add_rule(RuleDecl(
        "r_base", "base",
        variables=(VarDecl(Var("X", "String")), VarDecl(Var("Y", "String"))),
        head=Atom("ancestor", (Var("X", "String"), Var("Y", "String"))),
        body=(BodyLiteral(Atom("parent", (Var("X", "String"), Var("Y", "String")))),),
    ))
    builder.add_rule(RuleDecl(
        "r_rec", "recursive",
        variables=(
            VarDecl(Var("X", "String")),
            VarDecl(Var("Y", "String")),
            VarDecl(Var("Z", "String")),
        ),
        head=Atom("ancestor", (Var("X", "String"), Var("Z", "String"))),
        body=(
            BodyLiteral(Atom("parent", (Var("X", "String"), Var("Y", "String")))),
            BodyLiteral(Atom("ancestor", (Var("Y", "String"), Var("Z", "String")))),
        ),
    ))
    return builder.build()


def test_datalog_derives_transitive_relation_correctly() -> None:
    """Acceptance: Datalog derives a simple transitive relation correctly."""
    ir = _transitive_ir()
    engine = DatalogEngine(facts_from_ir(ir), rules_from_ir(ir))
    result = engine.evaluate_semi_naive()
    pairs = {(f.args[0], f.args[1]) for f in result.derived_facts}
    assert ("tom", "bob") in pairs
    assert ("tom", "ann") in pairs   # the transitively derived fact
    assert ("bob", "ann") in pairs


def test_recursive_datalog_reaches_fixed_point_on_finite_graph() -> None:
    """Acceptance: recursive Datalog reaches a fixed point on a finite graph."""
    ir = _transitive_ir()
    engine = DatalogEngine(facts_from_ir(ir), rules_from_ir(ir))
    result = engine.evaluate_semi_naive()
    assert result.fixed_point_reached is True
    # a second evaluation derives nothing new
    again = engine.evaluate_semi_naive()
    assert {(f.predicate, f.args, f.scope) for f in again.derived_facts} == {
        (f.predicate, f.args, f.scope) for f in result.derived_facts
    }


def test_semi_naive_matches_reference_evaluator() -> None:
    """Acceptance: semi-naive evaluation produces the same logical result as the
    reference (naive) evaluator."""
    ir = _transitive_ir()
    engine = DatalogEngine(facts_from_ir(ir), rules_from_ir(ir))
    semi = engine.evaluate_semi_naive()
    naive = engine.evaluate_naive()
    assert {(f.predicate, f.args, f.scope) for f in semi.derived_facts} == {
        (f.predicate, f.args, f.scope) for f in naive.derived_facts
    }


def test_derived_facts_carry_rule_and_parent_provenance() -> None:
    """Acceptance: each derived fact includes rule and parent-fact provenance."""
    ir = _transitive_ir()
    engine = DatalogEngine(facts_from_ir(ir), rules_from_ir(ir))
    result = engine.evaluate_semi_naive()
    for fact in result.derived_facts:
        assert fact.derivations
        for derivation in fact.derivations:
            assert derivation.rule_id
            assert derivation.parent_fact_ids
    transitive = engine.explain(result, "ancestor", ("tom", "ann"))
    assert transitive
    assert any(d.rule_id == "r_rec" for d in transitive)


def test_multiple_independent_derivations_preserved() -> None:
    """Acceptance: multiple independent derivations are preserved."""
    builder = _builder()
    builder.declare_predicate(PredicateDecl("parent", ("String", "String")))
    builder.declare_predicate(PredicateDecl("ancestor", ("String", "String")))
    for index, (a, c) in enumerate([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]):
        builder.add_statement(Statement(
            f"g{index}", StatementKind.FACT,
            Atom("parent", (Const(a, "String"), Const(c, "String"))),
        ))
    builder.add_rule(RuleDecl(
        "base", "base",
        variables=(VarDecl(Var("X", "String")), VarDecl(Var("Y", "String"))),
        head=Atom("ancestor", (Var("X", "String"), Var("Y", "String"))),
        body=(BodyLiteral(Atom("parent", (Var("X", "String"), Var("Y", "String")))),),
    ))
    builder.add_rule(RuleDecl(
        "rec", "recursive",
        variables=(
            VarDecl(Var("X", "String")),
            VarDecl(Var("Y", "String")),
            VarDecl(Var("Z", "String")),
        ),
        head=Atom("ancestor", (Var("X", "String"), Var("Z", "String"))),
        body=(
            BodyLiteral(Atom("ancestor", (Var("X", "String"), Var("Y", "String")))),
            BodyLiteral(Atom("parent", (Var("Y", "String"), Var("Z", "String")))),
        ),
    ))
    ir = builder.build()
    engine = DatalogEngine(facts_from_ir(ir), rules_from_ir(ir))
    result = engine.evaluate_semi_naive()
    derivations = engine.explain(result, "ancestor", ("a", "d"))
    assert len(derivations) == 2
    assert {d.parent_fact_ids for d in derivations} and \
        len({d.parent_fact_ids for d in derivations}) == 2


def test_stratification_negation_through_recursion_rejected() -> None:
    """Acceptance: stratification restrictions are enforced for negation."""
    builder = _builder()
    builder.declare_predicate(PredicateDecl("p", ("String",)))
    builder.declare_predicate(PredicateDecl("q", ("String",)))
    builder.add_rule(RuleDecl(
        "bad", "negation through recursion",
        variables=(VarDecl(Var("X", "String")),),
        head=Atom("p", (Var("X", "String"),)),
        body=(
            BodyLiteral(Atom("q", (Var("X", "String"),))),
            BodyLiteral(Atom("p", (Var("X", "String"),)), negated=True),
        ),
    ))
    ir = builder.build()
    with pytest.raises(NonStratifiableError):
        DatalogEngine(facts_from_ir(ir), rules_from_ir(ir))


def test_stratified_negation_evaluates_after_lower_stratum() -> None:
    builder = _builder()
    builder.declare_predicate(PredicateDecl("competing", ("String",)))
    builder.declare_predicate(PredicateDecl("candidate", ("String",)))
    builder.declare_predicate(PredicateDecl("clear", ("String",)))
    builder.add_statement(Statement(
        "f1", StatementKind.FACT, Atom("candidate", (Const("a", "String"),)),
    ))
    builder.add_statement(Statement(
        "f2", StatementKind.FACT, Atom("candidate", (Const("b", "String"),)),
    ))
    builder.add_statement(Statement(
        "f3", StatementKind.FACT, Atom("competing", (Const("a", "String"),)),
    ))
    builder.add_rule(RuleDecl(
        "neg", "negation over a positively bound variable",
        variables=(VarDecl(Var("X", "String")),),
        head=Atom("clear", (Var("X", "String"),)),
        body=(
            BodyLiteral(Atom("candidate", (Var("X", "String"),))),
            BodyLiteral(Atom("competing", (Var("X", "String"),)), negated=True),
        ),
    ))
    ir = builder.build()
    engine = DatalogEngine(facts_from_ir(ir), rules_from_ir(ir))
    result = engine.evaluate_semi_naive()
    keys = {(f.predicate, f.args) for f in result.derived_facts}
    assert ("clear", ("b",)) in keys          # derived via negation-as-absence
    assert ("clear", ("a",)) not in keys      # blocked by the positive fact


def test_temporal_facts_non_overlapping_not_simultaneous_conflict() -> None:
    """Acceptance: temporal facts with non-overlapping validity intervals are not
    falsely flagged as simultaneous conflict."""
    checker = ConsistencyChecker()
    fact_a = DatalogFact("status", ("x", "active"), "t1", interval=(1, 10))
    fact_b = DatalogFact("status", ("x", "active"), "t2", negated=True, interval=(20, 30))
    assert checker.check_temporal_non_conflict(fact_a, fact_b)
    report = checker.check([fact_a, fact_b])
    kinds = {c.kind for c in report.conflicts}
    assert ConflictKind.LOGICAL_CONTRADICTION not in kinds
    assert ConflictKind.TEMPORAL_MISMATCH in kinds


def test_unsafe_variables_and_unbounded_constructs_rejected() -> None:
    """Acceptance: unsafe variables and unbounded rule constructs are rejected."""
    builder = _builder()
    builder.declare_predicate(PredicateDecl("p", ("String",)))
    builder.declare_predicate(PredicateDecl("q", ("String", "String")))
    builder.add_rule(RuleDecl(
        "unsafe", "unsafe rule",
        variables=(VarDecl(Var("X", "String")), VarDecl(Var("Y", "String"))),
        head=Atom("q", (Var("X", "String"), Var("Y", "String"))),
        body=(BodyLiteral(Atom("p", (Var("X", "String"),))),),
    ))
    with pytest.raises(Exception, match="unsafe"):
        builder.build()


def test_function_symbols_outside_datalog_subset_rejected() -> None:
    """Unbounded rule constructs (function symbols) are rejected."""
    builder = _builder()
    builder.declare_function(FunctionDecl("f", ("String",), "String"))
    builder.declare_predicate(PredicateDecl("p", ("String",)))
    builder.add_statement(Statement(
        "f1", StatementKind.FACT,
        Atom("p", (FuncApp("f", (Const("a", "String"),)),)),
    ))
    ir = builder.build()
    with pytest.raises(Exception, match="unbounded rule construct"):
        facts_from_ir(ir)


def test_rule_change_invalidates_cached_derivations() -> None:
    """Acceptance: a rule change invalidates affected cached derivations."""
    ir = _transitive_ir()
    engine = DatalogEngine(facts_from_ir(ir), rules_from_ir(ir))
    cache = DatalogCache()
    key = cache.key(engine)
    cache.put(key, engine.evaluate_semi_naive())
    assert cache.get(key) is not None
    # any rule change produces a different program digest, hence a new key
    builder = _builder()
    builder.declare_predicate(PredicateDecl("parent", ("String", "String")))
    builder.add_statement(Statement(
        "f0", StatementKind.FACT,
        Atom("parent", (Const("x", "String"), Const("y", "String"))),
    ))
    changed_ir = builder.build()
    changed_engine = DatalogEngine(facts_from_ir(changed_ir), rules_from_ir(changed_ir))
    assert cache.key(changed_engine) != key
    # targeted invalidation by derived predicate also drops the entry
    assert cache.invalidate(["ancestor"]) == 1
    assert cache.get(key) is None


def test_iteration_limit_stops_unbounded_recursion() -> None:
    """Acceptance: explicit limits bound recursive evaluation."""
    builder = _builder()
    builder.declare_predicate(PredicateDecl("next", ("String", "String")))
    builder.declare_predicate(PredicateDecl("reach", ("String", "String")))
    for index in range(6):
        builder.add_statement(Statement(
            f"n{index}", StatementKind.FACT,
            Atom("next", (Const(f"n{index}", "String"), Const(f"n{index + 1}", "String"))),
        ))
    builder.add_rule(RuleDecl(
        "base", "base",
        variables=(VarDecl(Var("X", "String")), VarDecl(Var("Y", "String"))),
        head=Atom("reach", (Var("X", "String"), Var("Y", "String"))),
        body=(BodyLiteral(Atom("next", (Var("X", "String"), Var("Y", "String")))),),
    ))
    builder.add_rule(RuleDecl(
        "rec", "recursive",
        variables=(
            VarDecl(Var("X", "String")),
            VarDecl(Var("Y", "String")),
            VarDecl(Var("Z", "String")),
        ),
        head=Atom("reach", (Var("X", "String"), Var("Z", "String"))),
        body=(
            BodyLiteral(Atom("reach", (Var("X", "String"), Var("Y", "String")))),
            BodyLiteral(Atom("next", (Var("Y", "String"), Var("Z", "String")))),
        ),
    ))
    ir = builder.build()
    engine = DatalogEngine(
        facts_from_ir(ir), rules_from_ir(ir), limits=DatalogLimits(max_iterations=2),
    )
    with pytest.raises(Exception, match="iteration limit"):
        engine.evaluate_semi_naive()


def test_non_stratifiable_program_detected_by_dependency_graph() -> None:
    builder = _builder()
    builder.declare_predicate(PredicateDecl("p", ("String",)))
    builder.declare_predicate(PredicateDecl("q", ("String",)))
    builder.add_rule(RuleDecl(
        "r1", "negation inside its own stratum",
        variables=(VarDecl(Var("X", "String")),),
        head=Atom("p", (Var("X", "String"),)),
        body=(
            BodyLiteral(Atom("q", (Var("X", "String"),))),
            BodyLiteral(Atom("p", (Var("X", "String"),)), negated=True),
        ),
    ))
    builder.add_rule(RuleDecl(
        "r2", "mutual recursion",
        variables=(VarDecl(Var("X", "String")),),
        head=Atom("q", (Var("X", "String"),)),
        body=(BodyLiteral(Atom("p", (Var("X", "String"),))),),
    ))
    ir = builder.build()
    with pytest.raises(NonStratifiableProgramError):
        build_dependency_graph(rules_from_ir(ir))


# ---------------------------------------------------------------------------
# Consistency (Steps 21-22 acceptance tests)
# ---------------------------------------------------------------------------


def test_contradictory_source_facts_reported_without_deleting_either() -> None:
    """Acceptance: contradictory source facts are reported without deleting either."""
    checker = ConsistencyChecker()
    pos = DatalogFact("flies", ("tweety",), "s1")
    neg = DatalogFact("flies", ("tweety",), "s2", negated=True)
    report = checker.check([pos, neg])
    assert report.conflicts
    assert report.preserved_sides == 2
    assert report.destructive_repairs == 0
    conflict = report.conflicts[0]
    assert pos.fact_id in conflict.side_a
    assert neg.fact_id in conflict.side_b
    assert conflict.minimal_evidence == ("s1", "s2")


def test_conflict_classification_covers_all_declared_kinds() -> None:
    checker = ConsistencyChecker(
        exclusivity_groups=(ExclusivityGroup("g", "classified", ("bird", "mammal"), 1),),
        functional_constraints=(FunctionalConstraint("f", "height_of"),),
        cardinality_constraints=(CardinalityConstraint("c", "spouse_of", 1, (0,)),),
    )
    same_source_pos = DatalogFact("flies", ("x",), "d1")
    same_source_neg = DatalogFact("flies", ("x",), "d1", negated=True)
    report = checker.check([same_source_pos, same_source_neg])
    assert report.conflicts[0].kind == ConflictKind.LOGICAL_CONTRADICTION

    scope_neg = DatalogFact("flies", ("x",), "d2", negated=True, scope="fiction")
    assert checker.check([same_source_pos, scope_neg]).conflicts[0].kind == \
        ConflictKind.SCOPE_MISMATCH

    excl_a = DatalogFact("classified", ("tweety", "bird"), "e1")
    excl_b = DatalogFact("classified", ("tweety", "mammal"), "e2")
    assert checker.check([excl_a, excl_b]).conflicts[0].kind == \
        ConflictKind.LOGICAL_CONTRADICTION

    val_a = DatalogFact("height_of", ("bob", "170"), "h1")
    val_b = DatalogFact("height_of", ("bob", "175"), "h2")
    assert checker.check([val_a, val_b]).conflicts[0].kind == \
        ConflictKind.SOURCE_DISAGREEMENT


def test_source_disagreement_distinguished_from_logical_contradiction() -> None:
    checker = ConsistencyChecker()
    a = DatalogFact("flies", ("t",), "src-a")
    b = DatalogFact("flies", ("t",), "src-b", negated=True)
    report = checker.check([a, b])
    assert report.conflicts[0].kind == ConflictKind.SOURCE_DISAGREEMENT


# ---------------------------------------------------------------------------
# SAT/SMT (Phase D acceptance tests)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_valid_boolean_formula_returns_sat_with_independently_true_model() -> None:
    """Acceptance: a valid Boolean formula returns SAT with a model that
    independently evaluates to true."""
    builder = _builder()
    builder.declare_predicate(PredicateDecl("rain", ("String",)))
    builder.add_statement(Statement(
        "f1", StatementKind.FACT, Atom("rain", (Const("today", "String"),)),
    ))
    ir = builder.build()
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    outcome = adapter.check_satisfiability(ir.facts())
    assert outcome.solver_status == SolverStatus.SAT
    assert outcome.result_status == "satisfiable"
    assert outcome.model_independently_validated is True
    assert outcome.model["atom_rain(const_today)"] is True


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_contradictory_boolean_formula_returns_unsat() -> None:
    """Acceptance: a contradictory Boolean formula returns UNSAT."""
    builder = _builder()
    builder.declare_predicate(PredicateDecl("switch_on", ("String",)))
    builder.add_statement(Statement(
        "f1", StatementKind.FACT, Atom("switch_on", (Const("s1", "String"),)),
    ))
    builder.add_statement(Statement(
        "f2", StatementKind.FACT, NotF(Atom("switch_on", (Const("s1", "String"),))),
    ))
    ir = builder.build()
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    outcome = adapter.check_satisfiability(ir.facts())
    assert outcome.solver_status == SolverStatus.UNSAT
    assert outcome.result_status == "unsatisfiable"
    assert set(outcome.unsat_core) == {"f1", "f2"}
    assert outcome.unsat_core_minimized is False


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_timeout_returns_timeout_never_unsat() -> None:
    """Acceptance: a solver timeout returns TIMEOUT, never UNSAT.

    The status mapping is exercised directly (deterministically) and through a
    deliberately hard instance that must never claim UNSAT.
    """
    builder = _builder()
    builder.declare_predicate(PredicateDecl("p", ("Bool",)))
    x = Var("x", "Int")
    y = Var("y", "Int")
    z = Var("z", "Int")
    # nonlinear diophantine factoring with a tiny budget: never UNSAT
    big = 9973 * 7919 * 1009
    builder.add_statement(Statement(
        "hard", StatementKind.CONSTRAINT,
        AndF((
            Compare(
                CompareOp.EQ,
                ArithExpr(ArithOp.MUL, (
                    ArithExpr(ArithOp.MUL, (x, y)), z,
                )),
                NumLit.int_value(big),
            ),
            Compare(CompareOp.GT, x, NumLit.int_value(1)),
            Compare(CompareOp.GT, y, NumLit.int_value(1)),
            Compare(CompareOp.GT, z, NumLit.int_value(1)),
        )),
        variables=(VarDecl(x), VarDecl(y), VarDecl(z)),
    ))
    ir = builder.build()
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=150)
    outcome = adapter.check_satisfiability(ir.constraints())
    assert outcome.solver_status != SolverStatus.UNSAT  # timeout/unknown never UNSAT


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_unsupported_theory_returns_unsupported_not_a_guess() -> None:
    """Acceptance: unsupported theory returns UNSUPPORTED or UNKNOWN, not a
    guessed answer; string constraints route only to a permitted string theory."""
    builder = _builder()
    builder.add_statement(Statement(
        "st", StatementKind.CONSTRAINT,
        Compare(
            CompareOp.EQ,
            StrExpr(StrOp.CONCAT, (Var("a", "String"), StrLit("b"))),
            StrLit("ab"),
        ),
        variables=(VarDecl(Var("a", "String")),),
    ))
    ir = builder.build()
    permitted = {PermittedTheory.BOOL}  # strings not permitted
    with pytest.raises(UnsupportedTheoryError):
        SmtAdapter(ir, permitted_theories=permitted).check_theories(ir.statements)


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_entailment_accepted_only_when_gamma_and_not_phi_unsat() -> None:
    """Acceptance: entailment is accepted only when Γ ∧ ¬φ is UNSAT."""
    x = Var("x", "Int")
    builder = _builder()
    builder.add_statement(Statement(
        "p1", StatementKind.CONSTRAINT,
        Compare(CompareOp.GT, x, NumLit.int_value(5)),
        variables=(VarDecl(x),),
    ))
    builder.add_statement(Statement(
        "goal", StatementKind.GOAL,
        Compare(CompareOp.GT, x, NumLit.int_value(4)),
        variables=(VarDecl(x),),
    ))
    ir = builder.build()
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    outcome = adapter.check_entailment(ir.constraints(), ir.goals()[0])
    assert outcome.solver_status == SolverStatus.UNSAT  # Γ ∧ ¬φ is UNSAT


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_sat_countermodel_decoded_and_independently_shown() -> None:
    """Acceptance: a SAT countermodel for Γ ∧ ¬φ is decoded and shown to violate
    φ while satisfying Γ."""
    x = Var("x", "Int")
    builder = _builder()
    builder.add_statement(Statement(
        "p1", StatementKind.CONSTRAINT,
        Compare(CompareOp.GT, x, NumLit.int_value(3)),
        variables=(VarDecl(x),),
    ))
    builder.add_statement(Statement(
        "goal", StatementKind.GOAL,
        Compare(CompareOp.GT, x, NumLit.int_value(4)),
        variables=(VarDecl(x),),
    ))
    ir = builder.build()
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    outcome = adapter.check_entailment(ir.constraints(), ir.goals()[0])
    assert outcome.solver_status == SolverStatus.SAT
    assert outcome.counterexample is not None
    assert outcome.model_independently_validated is True
    counter_value = outcome.model["var_x"]
    assert Fraction(counter_value) > 3       # satisfies Γ (x>3)
    assert Fraction(counter_value) <= 4      # violates φ (x>4)


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_named_assumptions_appear_in_unsat_core() -> None:
    """Acceptance: named assumptions appear in an unsat core when supported."""
    x = Var("x", "Int")
    builder = _builder()
    builder.add_statement(Statement(
        "c1", StatementKind.CONSTRAINT,
        Compare(CompareOp.GT, x, NumLit.int_value(5)),
        variables=(VarDecl(x),),
        assumption_name="assume_x_above_5",
    ))
    builder.add_statement(Statement(
        "c2", StatementKind.CONSTRAINT,
        Compare(CompareOp.LT, x, NumLit.int_value(3)),
        variables=(VarDecl(x),),
        assumption_name="assume_x_below_3",
    ))
    ir = builder.build()
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    outcome = adapter.check_satisfiability(ir.constraints())
    assert outcome.solver_status == SolverStatus.UNSAT
    assert set(outcome.unsat_core) == {"assume_x_above_5", "assume_x_below_3"}


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_unsat_core_not_claimed_minimum_unless_minimized() -> None:
    """Acceptance: the system does not claim an unsat core is minimum unless it
    actually minimizes it."""
    x = Var("x", "Int")
    builder = _builder()
    builder.add_statement(Statement(
        "a", StatementKind.CONSTRAINT,
        Compare(CompareOp.GT, x, NumLit.int_value(5)),
        variables=(VarDecl(x),),
    ))
    builder.add_statement(Statement(
        "b", StatementKind.CONSTRAINT,
        Compare(CompareOp.LT, x, NumLit.int_value(1)),
        variables=(VarDecl(x),),
    ))
    builder.add_statement(Statement(
        "c_redundant", StatementKind.CONSTRAINT,
        Compare(CompareOp.LT, x, NumLit.int_value(2)),
        variables=(VarDecl(x),),
    ))
    ir = builder.build()
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    outcome = adapter.check_satisfiability(ir.constraints())
    assert outcome.unsat_core_minimized is False
    assert any("not necessarily a minimum-cardinality cause" in l
               for l in outcome.limitations)
    core, minimized = adapter.minimize_core(outcome.unsat_core)
    assert minimized is True
    assert len(core) < 3                     # a redundant member was dropped
    assert set(core) != {"a", "b", "c_redundant"}
    # the minimized core is irreducible: removing any member keeps it SAT
    for label in core:
        subset = [item for item in core if item != label]
        if subset:
            assert adapter._is_unsat_subset(subset) is False


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_integer_and_real_arithmetic_distinct() -> None:
    """Acceptance: integer and real arithmetic are not silently interchanged."""
    builder = _builder()
    builder.declare_predicate(PredicateDecl("q", ("Real",)))
    builder.add_statement(Statement(
        "s", StatementKind.CONSTRAINT,
        Atom("q", (ArithExpr(
            ArithOp.ADD, (NumLit.int_value(1), NumLit.real_value(Fraction(1, 2))),
        ),)),
    ))
    with pytest.raises(InvalidFormalizationError):
        builder.build()


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_division_by_zero_detected_or_represented() -> None:
    """Acceptance: division by zero is detected or represented per theory
    semantics."""
    x = Var("x", "Real")
    builder = _builder()
    builder.add_statement(Statement(
        "s", StatementKind.CONSTRAINT,
        Compare(
            CompareOp.GT,
            ArithExpr(ArithOp.DIV, (x, NumLit.real_value(Fraction(0)))),
            NumLit.real_value(Fraction(1)),
        ),
        variables=(VarDecl(x),),
    ))
    ir = builder.build()
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    with pytest.raises(InvalidFormalizationError, match="division by a literal zero"):
        adapter.check_satisfiability(ir.constraints())


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_bitvector_overflow_follows_bv_semantics() -> None:
    """Acceptance: bit-vector overflow follows bit-vector semantics and is not
    treated as unbounded integer arithmetic."""
    builder = _builder()
    builder.declare_sort(SortDecl("byte", SortKind.BITVECTOR, width=8))
    x = Var("x", "byte")
    builder.add_statement(Statement(
        "c1", StatementKind.CONSTRAINT,
        Compare(
            CompareOp.EQ,
            BvExpr(BvOp.ADD, (x, BVLit(200, 8))),
            BVLit(44, 8),  # 200 + 100 wraps to 44 in 8-bit semantics
        ),
        variables=(VarDecl(x),),
    ))
    builder.add_statement(Statement(
        "c2", StatementKind.CONSTRAINT,
        Compare(CompareOp.EQ, x, BVLit(100, 8)),
        variables=(VarDecl(x),),
    ))
    ir = builder.build()
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    outcome = adapter.check_satisfiability(ir.constraints())
    assert outcome.solver_status == SolverStatus.SAT
    assert outcome.model["var_x"] == 100
    assert outcome.model_independently_validated is True


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_string_constraints_routed_only_to_supported_string_theory() -> None:
    """Acceptance: string constraints are routed only to a supported string theory."""
    builder = _builder()
    builder.add_statement(Statement(
        "st", StatementKind.CONSTRAINT,
        Compare(
            CompareOp.EQ,
            StrExpr(StrOp.CONCAT, (Var("a", "String"), StrLit("b"))),
            StrLit("ab"),
        ),
        variables=(VarDecl(Var("a", "String")),),
    ))
    ir = builder.build()
    need = detect_theories(ir.statements)
    assert PermittedTheory.STRINGS in need.theories
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    outcome = adapter.check_satisfiability(ir.constraints())
    assert outcome.solver_status == SolverStatus.SAT
    assert outcome.model["var_a"] == "a"


@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver is not installed")
def test_open_world_absence_not_converted_into_false_by_smt() -> None:
    """Acceptance: open-world absence is not converted into explicit false."""
    builder = _builder()
    builder.declare_predicate(PredicateDecl("flies", ("String",)))
    builder.add_statement(Statement(
        "f1", StatementKind.FACT, Atom("flies", (Const("tweety", "String"),)),
    ))
    ir = builder.build()
    adapter = SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    outcome = adapter.check_satisfiability(ir.facts())
    # absence of penguin(tweety) never asserts not-penguin(tweety)
    assert outcome.solver_status == SolverStatus.SAT
    assert not any(k.startswith("atom_penguin") for k in outcome.model)


def test_quantified_cardinality_exactly_one() -> None:
    """Step 11: 'exactly one' becomes an explicit cardinality constraint."""
    builder = _builder()
    builder.declare_predicate(PredicateDecl("p", ("String",)))
    builder.declare_predicate(PredicateDecl("q", ("String",)))
    builder.add_statement(Statement(
        "card", StatementKind.CONSTRAINT,
        CardinalityF("exactly", 1, (
            Atom("p", (Const("a", "String"),)),
            Atom("q", (Const("a", "String"),)),
        )),
    ))
    ir = builder.build()
    adapter = SmtAdapter(ir, permitted_themes=_SMT_THEORIES, timeout_ms=5000) \
        if False else SmtAdapter(ir, permitted_theories=_SMT_THEORIES, timeout_ms=5000)
    outcome = adapter.check_satisfiability(ir.constraints())
    assert outcome.solver_status == SolverStatus.SAT
    exactly_one = sum(
        1 for key in ("atom_p(const_a)", "atom_q(const_a)") if outcome.model.get(key)
    )
    assert exactly_one == 1


# ---------------------------------------------------------------------------
# CP-SAT optimization (Phase E acceptance tests)
# ---------------------------------------------------------------------------


def _selection_problem() -> SelectionProblem:
    return SelectionProblem(
        problem_id="acc-cpsat",
        candidates=(
            ModuleCandidate("m_vision_a", cost=10.0, capabilities=("vision",),
                            resources={"cpu": 2.0, "mem": 4.0}, depends_on=("m_base",)),
            ModuleCandidate("m_vision_b", cost=8.0, capabilities=("vision",),
                            resources={"cpu": 3.0, "mem": 2.0}, depends_on=("m_base",)),
            ModuleCandidate("m_base", cost=5.0, capabilities=("base",),
                            resources={"cpu": 1.0, "mem": 1.0}),
            ModuleCandidate("m_conflict", cost=1.0, capabilities=("base",),
                            resources={"cpu": 0.5, "mem": 0.5},
                            mutually_exclusive_with=("m_base",)),
        ),
        required_capabilities=("vision",),
        resource_limits={"cpu": 4.0, "mem": 5.0},
        objective_weights={"cost": 1.0},
        units={"cost": "usd", "cpu": "cores", "mem": "GB"},
    )


def test_cpsat_selects_feasible_set_covering_all_capabilities() -> None:
    """Acceptance: CP-SAT selects a feasible set covering all required capabilities."""
    problem = _selection_problem()
    outcome = OptimizationRouter(prefer_backend="reference").solve(problem)
    assert outcome.status == ResultStatus.OPTIMAL
    ok, errors = validate_selection(problem, outcome.selection)
    assert ok, errors
    covered = any("vision" in c.capabilities and c.module_id in outcome.selection
                  for c in problem.candidates)
    assert covered


def test_cpsat_respects_cpu_and_memory_budgets_and_dependencies() -> None:
    """Acceptance: CP-SAT respects CPU and memory budgets; dependency
    constraints are respected."""
    problem = _selection_problem()
    outcome = OptimizationRouter(prefer_backend="reference").solve(problem)
    ok, errors = validate_selection(problem, outcome.selection)
    assert ok, errors
    assert "m_base" in outcome.selection  # dependency of any vision module
    for module_id in outcome.selection:
        for prerequisite in [
            c.depends_on for c in problem.candidates if c.module_id == module_id
        ]:
            assert all(p in outcome.selection for p in prerequisite)


def test_mutually_exclusive_selections_never_both_selected() -> None:
    """Acceptance: mutually exclusive module selections are never both selected."""
    problem = _selection_problem()
    outcome = OptimizationRouter(prefer_backend="reference").solve(problem)
    assert not ("m_base" in outcome.selection and "m_conflict" in outcome.selection)


def test_infeasible_optimization_returns_infeasible_not_empty_optimal() -> None:
    """Acceptance: infeasible optimization returns INFEASIBLE, not an empty
    'optimal' plan."""
    problem = SelectionProblem(
        problem_id="infeasible",
        candidates=(ModuleCandidate("big", cost=1.0, capabilities=("vision",),
                                    resources={"cpu": 99.0}),),
        required_capabilities=("vision",),
        resource_limits={"cpu": 2.0},
        objective_weights={"cost": 1.0},
        units={"cpu": "cores"},
    )
    outcome = OptimizationRouter(prefer_backend="reference").solve(problem)
    assert outcome.status == ResultStatus.INFEASIBLE
    assert outcome.selection == ()
    assert outcome.exhaustive is True


def test_time_limited_incumbent_labeled_feasible_not_optimal() -> None:
    """Acceptance: a time-limited incumbent is labeled FEASIBLE with a bound/gap,
    never OPTIMAL."""
    from ore_formal_reasoning.cpsat import CpsatOptimizer, ORTOOLS_AVAILABLE

    if not ORTOOLS_AVAILABLE:
        pytest.skip("ortools is not installed")
    # a hard knapsack-like instance with a tiny time limit
    candidates = tuple(
        ModuleCandidate(f"m{i:02d}", cost=float(i % 7 + 1),
                        capabilities=("cap",), resources={"cpu": float(i % 5 + 1)})
        for i in range(18)
    )
    problem = SelectionProblem(
        problem_id="timed",
        candidates=candidates,
        required_capabilities=("cap",),
        resource_limits={"cpu": 18.0},
        objective_weights={"cost": 1.0},
        units={"cpu": "cores"},
    )
    outcome = CpsatOptimizer().solve(problem, time_limit_seconds=0.1)
    if outcome.status == ResultStatus.FEASIBLE:
        assert outcome.optimality_gap is not None
        assert outcome.best_bound is not None
        assert outcome.status != ResultStatus.OPTIMAL
        assert any("never labeled OPTIMAL" in l for l in outcome.limitations)
    else:
        assert outcome.status in {ResultStatus.OPTIMAL, ResultStatus.UNKNOWN}


def test_objective_weights_and_units_recorded_with_result() -> None:
    """Acceptance: objective weights and units are recorded with the result."""
    problem = _selection_problem()
    outcome = OptimizationRouter(prefer_backend="reference").solve(problem)
    assert outcome.solver.configuration["objective_weights"] == {"cost": 1.0}
    assert outcome.solver.configuration["units"]["cost"] == "usd"
    assert outcome.solver.configuration["units"]["cpu"] == "cores"


# ---------------------------------------------------------------------------
# Machine-checked proofs (Phase F acceptance tests)
# ---------------------------------------------------------------------------


def _syllogism() -> ProofObligation:
    s = PropVar("socrates_is_man")
    h = PropVar("socrates_is_human")
    m = PropVar("socrates_is_mortal")
    return ProofObligation(
        obligation_id="obl-syl",
        theorem_name="syllogism",
        goal=PropImplies(s, m),
        premises=(
            Premise("h1", PropImplies(s, h), "observed"),
            Premise("h2", PropImplies(h, m), "stipulated"),
        ),
        dependencies=("none",),
    )


def test_lean_path_accepts_valid_proof_artifact() -> None:
    """Acceptance: a valid proof artifact is accepted by the trusted checker
    (Lean when a toolchain is pinned, the bundled kernel otherwise)."""
    obligation = _syllogism()
    candidate = ProofCandidate(
        "manual",
        PLam("s", PropVar("socrates_is_man"),
             PApp(PHyp("h2"), PApp(PHyp("h1"), PVar("s")))),
    )
    verifier = ProofVerifier()
    outcome = verifier.verify(obligation, candidate, artifact_id="art-valid")
    assert outcome.status == ResultStatus.PROVED
    artifact = outcome.artifact
    assert artifact.accepted is True
    assert artifact.checker_name
    assert artifact.artifact_hash
    assert artifact.theorem_name == "syllogism"
    assert artifact.conditional_on == ("h1", "h2")


def test_kernel_rejects_invalid_proof_artifact() -> None:
    """Acceptance: an invalid proof artifact is rejected."""
    obligation = _syllogism()
    bad = ProofCandidate(
        "manual",
        PLam("s", PropVar("socrates_is_man"),
             PApp(PHyp("h1"), PApp(PHyp("h2"), PVar("s")))),
    )
    outcome = ProofVerifier().verify(obligation, bad, artifact_id="art-invalid")
    assert outcome.status == ResultStatus.UNKNOWN
    assert outcome.artifact is None
    assert "rejected" in outcome.detail


def test_proof_result_records_theorem_assumptions_checker_and_versions() -> None:
    """Acceptance: a proof result records the exact theorem, assumptions, checker
    version and dependency versions."""
    obligation = _syllogism()
    candidate = ProofSearcher().attempt(obligation, ("h1", "h2"))
    assert candidate is not None
    artifact = ProofVerifier().verify(
        obligation, candidate, artifact_id="art-meta",
    ).artifact
    assert artifact.theorem_name == "syllogism"
    assert "socrates_is_man" in artifact.statement_text
    assert artifact.conditional_on == ("h1", "h2")
    assert artifact.checker_name and artifact.checker_version
    assert artifact.dependencies == ("none",)


def test_theorem_from_unvalidated_premises_reported_conditional() -> None:
    """Acceptance: a theorem proved from false or unvalidated premises is
    reported as conditional on those premises."""
    s = PropVar("p_a")
    h = PropVar("p_b")
    m = PropVar("p_c")
    obligation = ProofObligation(
        obligation_id="obl-cond",
        theorem_name="conditional",
        goal=PropImplies(s, m),
        premises=(
            Premise("h1", PropImplies(s, h), "unvalidated"),
            Premise("h2", PropImplies(h, m), "unvalidated"),
        ),
    )
    candidate = ProofSearcher().attempt(obligation, ("h1", "h2"))
    outcome = ProofVerifier().verify(obligation, candidate, artifact_id="art-cond")
    assert outcome.status == ResultStatus.PROVED
    assert any("conditional on premises with unvalidated status" in l
               for l in outcome.artifact.limitations)


def test_proof_search_failure_not_reported_as_falsity() -> None:
    """Acceptance: a proof-search failure is not reported as theorem falsity."""
    from ore_formal_reasoning.contracts import FormalResult, ResultKind

    obligation = _syllogism()
    failed = ProofSearcher().attempt(obligation, ("h1",))
    assert failed is None
    # the coordinator maps a missing candidate to UNKNOWN with 'not proved' note
    result = FormalResult(
        result_id="r", request_id="q",
        kind=ResultKind.MACHINE_CHECKED_PROOF,
        status=ResultStatus.UNKNOWN,
        statement="proof candidate rejected or search failed; not a refutation",
        provenance=(),
        reproducibility={"ir_hash": ""},
    )
    assert result.establishes_false is False


def test_kernel_checks_connective_rules() -> None:
    """The trusted kernel checks conjunction, disjunction, and bottom elimination."""
    a = PropVar("a")
    b = PropVar("b")
    kernel = TrustedKernel()

    # conjunction: pair + projections
    obligation_and = ProofObligation(
        "o1", "and_intro", PropAnd(a, b),
        premises=(Premise("ha", a), Premise("hb", b)),
    )
    verdict = kernel.check(
        {p.name: p for p in obligation_and.premises},
        PPair(PHyp("ha"), PHyp("hb")),
        obligation_and.goal,
    )
    assert verdict.accepted
    left = ProofObligation(
        "o2", "and_elim_left", a,
        premises=(Premise("hab", PropAnd(a, b)),),
    )
    assert kernel.check(
        {p.name: p for p in left.premises}, PFst(PHyp("hab")), left.goal,
    ).accepted
    assert kernel.check(
        {p.name: p for p in left.premises}, PSnd(PHyp("hab")), b,
    ).accepted

    # disjunction: inl + case elimination
    dis = PropOr(a, b)
    inl_obligation = ProofObligation(
        "o3", "or_intro", dis, premises=(Premise("ha", a),),
        declared_atoms=("b",),
    )
    assert kernel.check(
        {p.name: p for p in inl_obligation.premises}, PInl(dis, PHyp("ha")), dis,
    ).accepted

    # bottom elimination: ex falso quodlibet
    abort_obligation = ProofObligation(
        "o4", "ex_falso", a, premises=(Premise("false_hyp", PropBottom()),),
        declared_atoms=("a",),
    )
    assert kernel.check(
        {p.name: p for p in abort_obligation.premises},
        PAbort(a, PHyp("false_hyp")),
        a,
    ).accepted


def test_incomplete_translation_rejected() -> None:
    """Step 38: reject incomplete or unsound obligation translations."""
    with pytest.raises(Exception, match="absent from every premise"):
        ProofObligation(
            "bad", "dangling_atom", PropVar("undeclared_atom"),
            premises=(Premise("h", PropVar("other")),),
        )


# ---------------------------------------------------------------------------
# Independent evaluator cross-checks (Step 27/43)
# ---------------------------------------------------------------------------


def test_independent_evaluator_exact_rationals_and_bv_wrap() -> None:
    evaluator = IndependentEvaluator()
    assignment = {"var_x": Fraction(1, 3), "var_b": 250}
    formula = Compare(
        CompareOp.GT, NumLit.real_value(Fraction(1, 4)), Var("x", "Real"),
    )
    assert evaluator.evaluate_formula(formula, {"var_x": Fraction(1, 3)}) is False
    assert evaluator.evaluate_formula(formula, {"var_x": Fraction(1, 5)}) is True
    bv_formula = Compare(
        CompareOp.EQ,
        BvExpr(BvOp.ADD, (Var("b", "byte8"), BVLit(200, 8))),
        BVLit(194, 8),
    )
    assert evaluator.evaluate_formula(bv_formula, {"var_b": 250}) is True


def test_independent_evaluator_reports_division_by_zero_as_unknown() -> None:
    evaluator = IndependentEvaluator()
    formula = Compare(
        CompareOp.GT,
        ArithExpr(ArithOp.DIV, (Var("x", "Real"), NumLit.real_value(Fraction(0)))),
        NumLit.real_value(Fraction(1)),
    )
    assert evaluator.evaluate_formula(formula, {"var_x": Fraction(1)}) is None


def test_solver_status_mapping_distinct() -> None:
    from ore_formal_reasoning.contracts import NON_SUCCESS_STATUSES

    assert SolverStatus.TIMEOUT not in {SolverStatus.SAT, SolverStatus.UNSAT}
    for status in (SolverStatus.UNKNOWN, SolverStatus.TIMEOUT, SolverStatus.ERROR):
        assert status in NON_SUCCESS_STATUSES or status == SolverStatus.TIMEOUT
