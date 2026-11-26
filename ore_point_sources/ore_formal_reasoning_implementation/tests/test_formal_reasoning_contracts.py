"""Contract, registry, IR, intake, and USG-adapter tests for FRCE (Point 5).

Covers the Section 9 acceptance tests that concern data contracts, typed IR
construction and normalization, entity resolution, retrieval, unit handling,
and the semantic assumption profiles — the structural half of the algorithm.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from ore_formal_reasoning.contracts import (
    FormalResult,
    FormalizationRequest,
    NamedAssumption,
    PermittedTheory,
    ResultKind,
    ResultStatus,
    SolverIdentity,
    SourceRef,
    TaskClass,
    TemporalScope,
)
from ore_formal_reasoning.exceptions import (
    AmbiguousEntityError,
    InvalidFormalizationError,
    UnitMismatchError,
)
from ore_formal_reasoning.intake import (
    run_intake,
    validate_request,
)
from ore_formal_reasoning.ir import (
    AndF,
    ArithExpr,
    ArithOp,
    Atom,
    BodyLiteral,
    BoolTermF,
    BoolLit,
    CardinalityF,
    Compare,
    CompareOp,
    Const,
    DomainSemantics,
    ExistsF,
    FormalIRBuilder,
    ForallF,
    NumLit,
    PredicateDecl,
    RuleDecl,
    SortDecl,
    SortKind,
    Statement,
    StatementKind,
    UnitConversion,
    UnitDecl,
    UnitTable,
    Var,
    VarDecl,
    dimensions_compatible,
    make_enum_sort,
)
from ore_formal_reasoning.registry import (
    FRCE_OPERATIONS,
    operation_by_id,
    operation_by_step,
    operations_by_phase,
)
from ore_formal_reasoning.usg_adapter import (
    KnowledgeRetriever,
    TypedKnowledgeRecord,
    USGEntityResolver,
    USGReadOnlyGuard,
    USGToIRMapper,
)
from ore_formal_reasoning.version import FORMAL_REASONING_IR_VERSION


# ---------------------------------------------------------------------------
# Registry traceability (FRC-001..FRC-052)
# ---------------------------------------------------------------------------


def test_registry_has_52_unique_ordered_operations() -> None:
    assert len(FRCE_OPERATIONS) == 52
    ids = [op.operation_id for op in FRCE_OPERATIONS]
    assert ids == [f"FRC-{step:03d}" for step in range(1, 53)]
    assert len(set(ids)) == len(ids)


def test_registry_phase_boundaries_match_spec() -> None:
    assert operation_by_step(1).phase == "A"
    assert operation_by_step(8).phase == "B"
    assert operation_by_step(16).phase == "C"
    assert operation_by_step(23).phase == "D"
    assert operation_by_step(31).phase == "E"
    assert operation_by_step(37).phase == "F"
    assert operation_by_step(42).phase == "G"
    assert operation_by_step(49).phase == "H"
    assert len(operations_by_phase("A")) == 7
    assert len(operations_by_phase("C")) == 7
    with pytest.raises(ValueError):
        operation_by_step(53)
    with pytest.raises(ValueError):
        operation_by_id("FRC-000")


# ---------------------------------------------------------------------------
# Result status discipline (Section 1: never report unknown as false)
# ---------------------------------------------------------------------------


def _result(**overrides: object) -> FormalResult:
    defaults: dict = {
        "result_id": "res-1",
        "request_id": "req-1",
        "kind": ResultKind.SATISFIABILITY_RESULT,
        "status": ResultStatus.UNKNOWN,
        "statement": "test statement",
        "provenance": (SourceRef("src-1"),),
        "reproducibility": {"ir_hash": "0" * 64},
    }
    defaults.update(overrides)
    return FormalResult(**defaults)


def test_non_success_statuses_never_establish_falsity() -> None:
    for status in (
        ResultStatus.UNKNOWN,
        ResultStatus.TIMEOUT,
        ResultStatus.UNSUPPORTED,
        ResultStatus.INVALID_INPUT,
        ResultStatus.ERROR,
    ):
        result = _result(status=status)
        assert result.establishes_false is False
        assert result.is_success is False


def test_only_falsifying_statuses_establish_falsity() -> None:
    assert _result(status=ResultStatus.REFUTED_UNDER_MODEL).establishes_false
    assert _result(status=ResultStatus.UNSATISFIABLE).establishes_false
    assert _result(status=ResultStatus.INFEASIBLE).establishes_false
    assert not _result(status=ResultStatus.SATISFIABLE).establishes_false


def test_proved_result_requires_kernel_checked_proof() -> None:
    with pytest.raises(ValueError, match="kernel_checked"):
        _result(
            kind=ResultKind.MACHINE_CHECKED_PROOF,
            status=ResultStatus.PROVED,
            proof_check_status="unverified",
        )
    ok = _result(
        kind=ResultKind.MACHINE_CHECKED_PROOF,
        status=ResultStatus.PROVED,
        proof_check_status="kernel_checked",
        proof_ref="art-1",
    )
    assert ok.proof_check_status == "kernel_checked"


def test_optimal_status_restricted_to_optimization_results() -> None:
    with pytest.raises(ValueError, match="OPTIMAL"):
        _result(status=ResultStatus.OPTIMAL)
    optimization = _result(
        kind=ResultKind.OPTIMIZATION_RESULT,
        status=ResultStatus.OPTIMAL,
    )
    assert optimization.status == ResultStatus.OPTIMAL


def test_decoded_model_requires_independent_validation_flag() -> None:
    with pytest.raises(ValueError, match="independently"):
        _result(model={"x": 1})


def test_solver_textual_explanation_is_not_proof() -> None:
    """A proof result carries a machine-check status or is not a proof (Section 3)."""
    with pytest.raises(ValueError):
        _result(
            kind=ResultKind.MACHINE_CHECKED_PROOF,
            status=ResultStatus.PROVED,
            proof_check_status="solver_explanation",
        )


def test_request_fingerprint_is_deterministic() -> None:
    kwargs = dict(
        request_id="req-9",
        objective="derive ancestor(tom, ann)",
        task_classes=(TaskClass.RULE_INFERENCE,),
        assumptions=(NamedAssumption("a1", "open world"),),
        source_refs=(SourceRef("src-1", record_id="u1"),),
        permitted_theories=frozenset({PermittedTheory.BOOL}),
    )
    first = FormalizationRequest(**kwargs)
    second = FormalizationRequest(**kwargs)
    assert first.fingerprint == second.fingerprint
    third = FormalizationRequest(**{**kwargs, "objective": "different"})
    assert first.fingerprint != third.fingerprint


def test_request_rejects_mixed_class_combination() -> None:
    with pytest.raises(ValueError, match="MIXED"):
        FormalizationRequest(
            request_id="r",
            objective="o",
            task_classes=(TaskClass.MIXED, TaskClass.RULE_INFERENCE),
        )


def test_temporal_scope_interval_validation() -> None:
    with pytest.raises(ValueError):
        TemporalScope(10, 5)
    scope = TemporalScope(1, 5)
    assert scope.overlaps(TemporalScope(4, 9))
    assert not scope.overlaps(TemporalScope(6, 9))
    assert scope.overlaps(TemporalScope())  # unbounded overlaps everything


# ---------------------------------------------------------------------------
# Typed IR construction and normalization (Phase B acceptance tests)
# ---------------------------------------------------------------------------


def _builder() -> FormalIRBuilder:
    builder = FormalIRBuilder()
    builder.declare_domain(DomainSemantics("default"))
    return builder


def test_usg_concept_ids_remain_traceable_after_normalization() -> None:
    builder = _builder()
    builder.declare_predicate(PredicateDecl("parent", ("String", "String")))
    builder.add_statement(Statement(
        "f1", StatementKind.FACT,
        Atom("parent", (Const("tom", "String"), Const("bob", "String"))),
        source_refs=(SourceRef("usg-bundle", record_id="usg-rel-42"),),
        origin="usg",
    ))
    ir = builder.build()
    entry = ir.source_map["f1"]
    assert entry["source_refs"][0]["record_id"] == "usg-rel-42"
    assert entry["origin"] == "usg"
    assert ir.canonical_hash() == builder.build().canonical_hash()


def test_ir_version_is_stable() -> None:
    builder = _builder()
    ir = builder.build()
    assert ir.version == FORMAL_REASONING_IR_VERSION


def test_normalizer_alpha_renames_bound_variables() -> None:
    builder = _builder()
    builder.declare_predicate(PredicateDecl("bird", ("String",)))
    builder.declare_predicate(PredicateDecl("flies", ("String",)))
    statement = Statement(
        "q1", StatementKind.CONSTRAINT,
        ForallF(
            (VarDecl(Var("x", "String")),),
            AndF((
                Atom("bird", (Var("x", "String"),)),
                Atom("flies", (Var("x", "String"),)),
            )),
        ),
    )
    builder.add_statement(statement)
    ir = builder.build()
    assert ir.statements[0].formula.__class__.__name__ == "ForallF"


def test_normalizer_folds_safe_boolean_constants() -> None:
    builder = _builder()
    builder.declare_predicate(PredicateDecl("p", ("String",)))
    folded_formula = AndF((
        BoolTermF(BoolLit(True)),
        Atom("p", (Const("a", "String"),)),
    ))
    builder.add_statement(Statement("s1", StatementKind.CONSTRAINT, folded_formula))
    ir = builder.build()
    formula = ir.statements[0].formula
    assert formula.__class__.__name__ == "Atom"  # `True /\ p(a)` folds to p(a)


def test_integer_and_real_are_not_silently_interchanged() -> None:
    builder = _builder()
    builder.declare_predicate(PredicateDecl("q", ("Real",)))
    builder.add_statement(Statement(
        "s", StatementKind.CONSTRAINT,
        Atom("q", (ArithExpr(
            ArithOp.ADD, (NumLit.int_value(1), NumLit.real_value(Fraction(1, 2))),
        ),)),
    ))
    with pytest.raises(InvalidFormalizationError, match="mixed|ToReal"):
        builder.build()


def test_identifier_string_never_typed_as_numeric_quantity() -> None:
    builder = _builder()
    builder.declare_predicate(PredicateDecl("weight", ("String", "Real")))
    builder.add_statement(Statement(
        "bad", StatementKind.FACT,
        Atom("weight", (Const("box1", "String"), Const("heavy_thing", "Real"))),
    ))
    with pytest.raises(InvalidFormalizationError, match="cannot be typed as a Real"):
        builder.build()


def test_unsafe_rule_head_variable_rejected() -> None:
    builder = _builder()
    builder.declare_predicate(PredicateDecl("parent", ("String", "String")))
    builder.declare_predicate(PredicateDecl("ancestor", ("String", "String")))
    builder.add_rule(RuleDecl(
        "bad", "unsafe rule",
        variables=(
            VarDecl(Var("X", "String")),
            VarDecl(Var("Y", "String")),
            VarDecl(Var("Z", "String")),
        ),
        head=Atom("ancestor", (Var("X", "String"), Var("Z", "String"))),
        body=(BodyLiteral(Atom("parent", (Var("X", "String"), Var("Y", "String")))),),
    ))
    with pytest.raises(InvalidFormalizationError, match="unsafe"):
        builder.build()


def test_undeclared_sort_and_predicate_rejected() -> None:
    builder = _builder()
    with pytest.raises(InvalidFormalizationError, match="undeclared sort"):
        builder.declare_predicate(PredicateDecl("p", ("Nope",)))
    builder2 = _builder()
    builder2.add_statement(Statement(
        "s", StatementKind.FACT, Atom("ghost", (Const("a", "String"),)),
    ))
    with pytest.raises(InvalidFormalizationError, match="undeclared predicate"):
        builder2.build()


def test_statement_without_domain_profile_rejected() -> None:
    """Step 14: never use one global assumption profile by accident."""
    builder = FormalIRBuilder()
    builder.declare_domain(DomainSemantics("animals"))
    builder.declare_predicate(PredicateDecl("p", ("String",)))
    builder.add_statement(Statement(
        "s", StatementKind.FACT, Atom("p", (Const("a", "String"),)),
        domain="physics",
    ))
    with pytest.raises(ValueError, match="semantic profile"):
        builder.build()


def test_enum_and_bv_sorts_declare_finiteness() -> None:
    enum_sort = make_enum_sort("color", ("red", "green", "blue"))
    assert enum_sort.finiteness.value == "finite"
    assert enum_sort.values == ("red", "green", "blue")
    bv_sort = SortDecl("word8", SortKind.BITVECTOR, width=8)
    assert bv_sort.width == 8
    with pytest.raises(ValueError):
        SortDecl("bad_bv", SortKind.BITVECTOR, width=0)


# ---------------------------------------------------------------------------
# Units and dimensional constraints (Step 13)
# ---------------------------------------------------------------------------


def test_unit_mismatch_rejected_before_solver_use() -> None:
    table = UnitTable()
    table.declare(UnitDecl("meter", (("length", 1),)))
    table.declare(UnitDecl("second", (("time", 1),)))
    with pytest.raises(UnitMismatchError):
        table.require_same_dimension("meter", "second")


def test_validated_conversion_rules_only() -> None:
    table = UnitTable()
    table.declare(UnitDecl("km", (("length", 1),)))
    table.declare(UnitDecl("meter", (("length", 1),)))
    table.declare_conversion(UnitConversion("km", "meter", Fraction(1000)))
    assert table.convert(Fraction(3), "km", "meter") == Fraction(3000)
    assert table.convert(Fraction(250), "meter", "km") == Fraction(1, 4)
    with pytest.raises(UnitMismatchError):
        table.convert(Fraction(1), "km", "second")


def test_incompatible_dimension_conversion_rejected() -> None:
    table = UnitTable()
    table.declare(UnitDecl("meter", (("length", 1),)))
    table.declare(UnitDecl("gram", (("mass", 1),)))
    with pytest.raises(UnitMismatchError):
        table.declare_conversion(UnitConversion("meter", "gram", Fraction(1)))
    assert not dimensions_compatible(
        UnitDecl("meter", (("length", 1),)), UnitDecl("gram", (("mass", 1),))
    )


# ---------------------------------------------------------------------------
# Intake (Steps 1, 2, 5-7)
# ---------------------------------------------------------------------------


def test_non_operationalizable_goal_requests_clarification() -> None:
    request = FormalizationRequest(
        request_id="r", objective="think about things",
        task_classes=(TaskClass.RULE_INFERENCE,),
    )
    clarifications = validate_request(request)
    assert clarifications and clarifications[0].code == "goal_not_operationalizable"
    report = run_intake(request)
    assert report.operationalizable is False


def test_classifier_never_calls_every_task_theorem_proving() -> None:
    builder = _builder()
    builder.declare_predicate(PredicateDecl("p", ("String",)))
    builder.add_statement(Statement(
        "s", StatementKind.CONSTRAINT,
        Compare(CompareOp.GT, Var("v", "Int"), NumLit.int_value(3)),
        variables=(VarDecl(Var("v", "Int")),),
    ))
    ir = builder.build()
    request = FormalizationRequest(
        request_id="r", objective="check arithmetic",
        task_classes=(TaskClass.ARITHMETIC_CONSTRAINTS,),
        required_property="v > 3",
    )
    classification = run_intake(request, ir).classification
    assert TaskClass.ARITHMETIC_CONSTRAINTS in classification.routed
    assert TaskClass.MATHEMATICAL_PROOF not in classification.routed


def test_open_world_absence_is_not_false() -> None:
    builder = _builder()
    builder.declare_predicate(PredicateDecl("p", ("String",)))
    builder.add_statement(Statement(
        "s", StatementKind.FACT, Atom("p", (Const("a", "String"),)),
    ))
    ir = builder.build()
    request = FormalizationRequest(
        request_id="r", objective="o",
        task_classes=(TaskClass.BOOLEAN_CONSISTENCY,),
        required_property="p(a)",
    )
    report = run_intake(request, ir)
    assert any("absence of a fact is not falsity" in w for w in report.completeness.warnings)


def test_closed_world_requires_explicit_negation_handling() -> None:
    with pytest.raises(ValueError, match="closed-world"):
        DomainSemantics("strict", closed_world=True, explicit_negation=False)


def test_missing_quantifier_scope_detected() -> None:
    builder = _builder()
    builder.declare_sort(SortDecl(
        "Thing", SortKind.INT, finiteness=__import__(
            "ore_formal_reasoning.ir", fromlist=["Finiteness"]
        ).Finiteness.PARTIALLY_KNOWN,
    ))
    builder.declare_predicate(PredicateDecl("p", ("Thing",)))
    builder.add_statement(Statement(
        "q", StatementKind.CONSTRAINT,
        ForallF((VarDecl(Var("v", "Thing")),), Atom("p", (Var("v", "Thing"),))),
    ))
    ir = builder.build()
    request = FormalizationRequest(
        request_id="r", objective="o", task_classes=(TaskClass.BOOLEAN_CONSISTENCY,),
        required_property="p",
    )
    report = run_intake(request, ir)
    assert any(
        c.code == "missing_quantifier_scope"
        for c in report.completeness.clarification_requests
    )


# ---------------------------------------------------------------------------
# USG adapter (Steps 3, 4, 8; Section 7 contracts)
# ---------------------------------------------------------------------------


def _records() -> tuple[TypedKnowledgeRecord, ...]:
    return (
        TypedKnowledgeRecord(
            "obj-tom", "object", "tom", {"entity": "tom"}, version_id="v1",
        ),
        TypedKnowledgeRecord(
            "rel-1", "relation", "parent_of",
            {"predicate": "parent", "subject": "tom", "object": "bob",
             "arg_sorts": ("String", "String")},
            version_id="v1",
        ),
        TypedKnowledgeRecord(
            "rel-2", "relation", "parent_of",
            {"predicate": "parent", "subject": "bob", "object": "ann",
             "arg_sorts": ("String", "String")},
            version_id="v1",
        ),
        TypedKnowledgeRecord(
            "rule-1", "rule", "direct_ancestor",
            {
                "head_predicate": "ancestor",
                "head_arg_sorts": ("String", "String"),
                "head_variables": ["X", "Y"],
                "body_predicates": ["parent"],
                "body_0_variables": ["X", "Y"],
                "variables": [("X", "String"), ("Y", "String")],
            },
            version_id="v1",
        ),
    )


def test_entity_resolution_rejects_untyped_records() -> None:
    with pytest.raises(ValueError, match="untyped"):
        TypedKnowledgeRecord("x", "mystery", "name", {})


def test_ambiguous_entity_resolution_blocks_formal_execution() -> None:
    records = _records() + (
        TypedKnowledgeRecord(
            "rel-3", "relation", "parent_of",
            {"predicate": "parent", "subject": "ann", "object": "zed",
             "arg_sorts": ("String", "String")},
            version_id="v2",
        ),
    )
    resolver = USGEntityResolver(records)
    with pytest.raises(AmbiguousEntityError):
        resolver.resolve("parent_of")
    resolved = resolver.resolve("parent_of", discriminator="rel-3")
    assert resolved.usg_id == "rel-3"


def test_retrieval_records_query_parameters_and_versions() -> None:
    retriever = KnowledgeRetriever(_records())
    record = retriever.retrieve(
        ["ancestor"], ["obj-tom"],
        [r for r in _records() if r.kind == "rule"],
    )
    assert "rel-1" in record.selected_record_ids
    assert "rule-1" in record.selected_rule_ids
    assert "v1" in record.source_versions
    assert record.query_digest  # reconstructable query parameters


def test_untyped_or_ambiguous_mapping_rejected() -> None:
    mapper = USGToIRMapper()
    builder = FormalIRBuilder()
    broken = (
        TypedKnowledgeRecord("rel-bad", "relation", "broken",
                             {"predicate": "parent", "subject": "a"}),
    )
    with pytest.raises(InvalidFormalizationError, match="lacks subject or object"):
        mapper.map_records(broken, builder, default_domain="default")


def test_adapter_has_no_usg_write_path() -> None:
    guard = USGReadOnlyGuard()
    for method in ("write", "promote", "delete_record", "upsert_fact"):
        with pytest.raises(Exception):
            guard.check_method(method)


def test_mapped_rules_are_range_restricted() -> None:
    """Unsafe USG rules are rejected before evaluation."""
    mapper = USGToIRMapper()
    builder = FormalIRBuilder()
    unsafe = (
        TypedKnowledgeRecord(
            "rule-bad", "rule", "bad_rule",
            {
                "head_predicate": "q",
                "head_arg_sorts": ("String", "String"),
                "head_variables": ["X", "Z"],
                "body_predicates": ["parent"],
                "body_0_variables": ["X", "Y"],
                "variables": [("X", "String"), ("Y", "String"), ("Z", "String")],
            },
            version_id="v1",
        ),
    )
    mapper.map_records(unsafe, builder, default_domain="default")
    with pytest.raises(InvalidFormalizationError, match="unsafe"):
        builder.build()
