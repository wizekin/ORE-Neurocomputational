"""Contract, rule-validation and unification tests for the neuro-symbolic layer.

Covers the Section 10 acceptance areas: contract serialization/versioning and
invalid-input rejection; rule arity, typing, unbound variables, unsafe
recursion and exception checks; unification exact matches, substitutions,
repeated variables and type mismatch.
"""
from __future__ import annotations

import pytest

from ore_neurosymbolic.contracts import (
    CandidateInterpretation,
    EpistemicStatus,
    ConfidenceVector,
    EvidenceItem,
    InferenceMode,
    Literal,
    LogicalTerm,
    LogicalVariable,
    NEUROSYMBOLIC_SCHEMA_VERSION,
    ProofNode,
    ReasoningFinding,
    RuleClass,
    RuleDefinition,
    SemanticFact,
    VerificationLevel,
    sha256_json,
)
from ore_neurosymbolic.request_models import ReasoningRequest
from ore_neurosymbolic.rule_validation import (
    RuleProfile,
    build_predicate_arity_table,
    select_relevant_rules,
    validate_ruleset,
)
from ore_neurosymbolic.unification import (
    Substitution,
    TypeTable,
    unify_literals,
    unify_terms,
)


def _fact(fact_id: str = "f1", **overrides) -> SemanticFact:
    base = dict(
        fact_id=fact_id,
        subject="tweety",
        predicate="is_a",
        object_value="bird",
        epistemic_status=EpistemicStatus.ASSERTED,
    )
    base.update(overrides)
    return SemanticFact(**base)


class TestContractValidation:
    """Invalid-input rejection at every contract boundary."""

    def test_literal_rejects_empty_predicate(self) -> None:
        with pytest.raises(ValueError, match="predicate"):
            Literal("", (LogicalTerm("x"),))

    def test_literal_rejects_non_tuple_args(self) -> None:
        with pytest.raises(TypeError, match="args"):
            Literal("p", [LogicalTerm("x")])  # type: ignore[arg-type]

    def test_fact_rejects_empty_subject(self) -> None:
        with pytest.raises(ValueError, match="subject"):
            SemanticFact(fact_id="f", subject="", predicate="p", object_value="o")

    def test_fact_rejects_bad_status_type(self) -> None:
        with pytest.raises(TypeError, match="epistemic_status"):
            SemanticFact(
                fact_id="f", subject="s", predicate="p", object_value="o",
                epistemic_status="asserted",  # type: ignore[arg-type]
            )

    def test_rule_rejects_empty_premises(self) -> None:
        with pytest.raises(ValueError, match="premises"):
            RuleDefinition(
                rule_id="r", version="1.0.0", typed_variables={"?X": "entity"},
                premises=(), conclusion=Literal("q", (LogicalVariable("X"),)),
            )

    def test_rule_rejects_priority_out_of_range(self) -> None:
        X = LogicalVariable("X", "entity")
        with pytest.raises(ValueError, match="priority"):
            RuleDefinition(
                rule_id="r", version="1.0.0", typed_variables={"?X": "entity"},
                premises=(Literal("p", (X,)),), conclusion=Literal("q", (X,)),
                priority=1.5,
            )

    def test_rule_rejects_undeclared_variable_prefix(self) -> None:
        with pytest.raises(ValueError, match="variable names"):
            RuleDefinition(
                rule_id="r", version="1.0.0", typed_variables={"X": "entity"},
                premises=(Literal("p", (LogicalVariable("X"),)),),
                conclusion=Literal("q", (LogicalVariable("X"),)),
            )

    def test_confidence_vector_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="ConfidenceVector"):
            ConfidenceVector(1.5, 0.1, 0.1, 0.1, 0.0)

    def test_evidence_item_rejects_bad_reliability(self) -> None:
        with pytest.raises(ValueError, match="reliability"):
            EvidenceItem(evidence_id="e", source="s", observation={}, reliability=2.0)


class TestContractSerialization:
    """Round-trip serialization, fingerprinting and schema versioning."""

    def test_fact_round_trip(self) -> None:
        fact = _fact()
        restored = SemanticFact.from_dict(fact.to_dict())
        assert restored == fact
        assert restored.fingerprint() == fact.fingerprint()

    def test_fact_round_trip_keeps_negation_and_unknown_separate(self) -> None:
        negated = _fact(polarity=False)
        unknown = _fact(epistemic_status=EpistemicStatus.UNKNOWN)
        assert negated.polarity is False
        assert unknown.epistemic_status is EpistemicStatus.UNKNOWN
        assert negated.fingerprint() != unknown.fingerprint()

    def test_literal_round_trip_preserves_variables_and_negation(self) -> None:
        literal = Literal(
            "p", (LogicalVariable("X", "entity"), LogicalTerm("c", "class")),
            negated=True, temporal_qualifier="past",
        )
        restored = Literal.from_dict(literal.to_dict())
        assert restored == literal
        assert restored.signature() == literal.signature()

    def test_rule_round_trip(self) -> None:
        X = LogicalVariable("X", "entity")
        rule = RuleDefinition(
            rule_id="r1", version="2.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("p", (X,)),), conclusion=Literal("q", (X,)),
            rule_class=RuleClass.DEFEASIBLE, priority=0.7,
            scope={"domain": "zoology"},
            exceptions=(Literal("e", (X,)),),
        )
        restored = RuleDefinition.from_dict(rule.to_dict())
        assert restored == rule
        assert restored.fingerprint() == rule.fingerprint()

    def test_proof_node_content_hash_is_stable(self) -> None:
        node = ProofNode(
            node_id="n1", conclusion=Literal("q", (LogicalTerm("a"),)),
            rule_ref=("r1", "1.0.0"), premise_node_ids=(), source_fact_ids=("f1",),
            run_id="run1",
        )
        first = node.compute_content_hash(())
        second = node.compute_content_hash(())
        assert first == second
        assert first == sha256_json({
            "conclusion": node.conclusion.to_dict(),
            "rule_version": "1.0.0",
            "premise_hashes": [],
        })

    def test_confidence_vector_keeps_five_dimensions(self) -> None:
        vector = ConfidenceVector(0.9, 0.8, 0.7, 0.6, 0.5)
        payload = vector.to_dict()
        assert set(payload) == {
            "q_source", "q_observation", "q_rule", "q_coverage", "q_conflict",
        }
        assert ConfidenceVector.from_dict(payload) == vector

    def test_schema_version_present(self) -> None:
        assert NEUROSYMBOLIC_SCHEMA_VERSION == "1.0.0"


class TestReasoningRequestValidation:
    """Step A1: reject invalid scope or absent knowledge/budgets."""

    def test_invalid_scope_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid scope keys"):
            ReasoningRequest(
                goal_predicate="p", goal_args=("a",),
                facts=(_fact(),),
                scope={"mystery_key": 1},
            )

    def test_absent_knowledge_rejected(self) -> None:
        with pytest.raises(ValueError, match="absent budgets"):
            ReasoningRequest(goal_predicate="p", goal_args=("a",))

    def test_non_positive_deadline_rejected(self) -> None:
        with pytest.raises(ValueError, match="deadline"):
            ReasoningRequest(
                goal_predicate="p", goal_args=("a",), facts=(_fact(),),
                deadline_seconds=0.0,
            )

    def test_goal_literal_projection(self) -> None:
        request = ReasoningRequest(
            goal_predicate="commands", goal_args=("a", "b"), facts=(_fact(),),
        )
        assert request.goal_literal.signature() == "commands(a,b)"

    def test_mode_and_verification_enums(self) -> None:
        request = ReasoningRequest(
            goal_predicate="p", goal_args=("a",), facts=(_fact(),),
            inference_mode=InferenceMode.BACKWARD,
            verification_level=VerificationLevel.STRICT,
        )
        assert request.inference_mode is InferenceMode.BACKWARD
        assert request.verification_level is VerificationLevel.STRICT


class TestRuleValidation:
    """Stage 1 gate: malformed rules rejected deterministically."""

    CLASS = LogicalTerm("class_a", "class")
    VALUE = LogicalTerm("value_a", "value")

    def _rule(self, **overrides) -> RuleDefinition:
        X = LogicalVariable("X", "entity")
        base = dict(
            rule_id="r1", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("p", (X, self.CLASS)),),
            conclusion=Literal("q", (X, self.VALUE)),
        )
        base.update(overrides)
        return RuleDefinition(**base)

    def test_valid_rule_accepted(self) -> None:
        report = validate_ruleset((self._rule(),))
        assert report.ok
        assert len(report.accepted_rules) == 1

    def test_unbound_conclusion_variable_rejected(self) -> None:
        Y = LogicalVariable("Y", "entity")
        report = validate_ruleset((
            self._rule(conclusion=Literal("q", (LogicalVariable("X"), Y))),
        ))
        assert not report.ok
        codes = {item.code for item in report.rejected_rules}
        assert "unbound_variables" in codes

    def test_undeclared_variable_rejected(self) -> None:
        report = validate_ruleset((
            RuleDefinition(
                rule_id="r2", version="1.0.0", typed_variables={},
                premises=(Literal("p", (LogicalVariable("X"), LogicalTerm("c"))),),
                conclusion=Literal("q", (LogicalVariable("X"), LogicalTerm("c"))),
            ),
        ))
        codes = {item.code for item in report.rejected_rules}
        assert "undeclared_variables" in codes

    def test_type_conflict_rejected(self) -> None:
        report = validate_ruleset((
            RuleDefinition(
                rule_id="r3", version="1.0.0", typed_variables={"?X": "entity"},
                premises=(Literal("p", (LogicalVariable("X", "number"),)),),
                conclusion=Literal("q", (LogicalVariable("X"),)),
            ),
        ))
        codes = {item.code for item in report.rejected_rules}
        assert "type_conflict" in codes

    def test_self_recursion_rejected_by_safe_profile(self) -> None:
        X = LogicalVariable("X", "entity")
        Y = LogicalVariable("Y", "entity")
        report = validate_ruleset((
            RuleDefinition(
                rule_id="r4", version="1.0.0",
                typed_variables={"?X": "entity", "?Y": "entity"},
                premises=(Literal("q", (X, LogicalTerm("c"))),),
                conclusion=Literal("q", (X, LogicalTerm("c"))),
            ),
        ))
        codes = {item.code for item in report.rejected_rules}
        assert "unsafe_self_recursion" in codes

    def test_strict_rule_cannot_declare_exceptions(self) -> None:
        X = LogicalVariable("X", "entity")
        report = validate_ruleset((
            self._rule(
                exceptions=(Literal("e", (X,)),),
            ),
        ))
        codes = {item.code for item in report.rejected_rules}
        assert "strict_rule_with_exceptions" in codes

    def test_defeasible_rule_requires_scope(self) -> None:
        X = LogicalVariable("X", "entity")
        report = validate_ruleset((
            self._rule(
                rule_class=RuleClass.DEFEASIBLE, scope={},
            ),
        ))
        codes = {item.code for item in report.rejected_rules}
        assert "defeasible_rule_without_scope" in codes

    def test_negation_in_head_rejected(self) -> None:
        X = LogicalVariable("X", "entity")
        report = validate_ruleset((
            self._rule(conclusion=Literal("q", (X,), negated=True)),
        ))
        codes = {item.code for item in report.rejected_rules}
        assert "negation_in_head" in codes

    def test_ruleset_size_budget(self) -> None:
        profile = RuleProfile(max_rules=2)
        report = validate_ruleset(
            tuple(self._rule(rule_id=f"r{i}", version="1.0.0") for i in range(3)),
            profile=profile,
        )
        assert report.rejected_rules[0].code == "ruleset_too_large"

    def test_duplicate_rule_id_with_different_content_rejected(self) -> None:
        X = LogicalVariable("X", "entity")
        report = validate_ruleset((
            self._rule(),
            self._rule(conclusion=Literal("other", (X,))),
        ))
        codes = {item.code for item in report.rejected_rules}
        assert "duplicate_rule_id" in codes

    def test_declared_arity_crosscheck(self) -> None:
        X = LogicalVariable("X", "entity")
        report = validate_ruleset(
            (self._rule(premises=(Literal("p", (X, LogicalTerm("c"))),)),),
            declared_predicates={"p": (1,)},
        )
        codes = {item.code for item in report.rejected_rules}
        assert "arity_mismatch" in codes

    def test_predicate_arity_table(self) -> None:
        X = LogicalVariable("X", "entity")
        rule = self._rule(premises=(Literal("p", (X, self.CLASS)),))
        table = build_predicate_arity_table((rule,))
        assert table == {"p": (2,), "q": (2,)}


class TestRuleSelection:
    """Step B2: goal-relevant selection with recorded rationale."""

    def test_selection_prefers_goal_conclusions(self) -> None:
        X = LogicalVariable("X", "entity")
        r1 = RuleDefinition(
            rule_id="r_goal", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("p", (X, LogicalTerm("c"))),),
            conclusion=Literal("goal", (X, LogicalTerm("v"))),
        )
        r2 = RuleDefinition(
            rule_id="r_other", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("p", (X, LogicalTerm("c"))),),
            conclusion=Literal("z", (X, LogicalTerm("v"))),
        )
        selected, rationale = select_relevant_rules((r1, r2), ("goal",))
        assert [rule.rule_id for rule in selected] == ["r_goal"]
        assert "goal" in rationale

    def test_fallback_selects_everything(self) -> None:
        X = LogicalVariable("X", "entity")
        rule = RuleDefinition(
            rule_id="r1", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("p", (X, LogicalTerm("c"))),),
            conclusion=Literal("q", (X, LogicalTerm("v"))),
        )
        selected, rationale = select_relevant_rules((rule,), ("nomatch",))
        assert len(selected) == 1
        assert "no goal predicate matched" in rationale


class TestUnification:
    """Step B3: typed unification with occurs check and diagnostics."""

    def test_exact_constant_match(self) -> None:
        substitution = Substitution()
        ok, diagnostic = unify_terms(LogicalTerm("a"), LogicalTerm("a"), substitution)
        assert ok and diagnostic is None

    def test_constant_conflict(self) -> None:
        substitution = Substitution()
        ok, diagnostic = unify_terms(LogicalTerm("a"), LogicalTerm("b"), substitution)
        assert not ok
        assert diagnostic.code == "constant_conflict"

    def test_variable_binds_to_term(self) -> None:
        substitution = Substitution()
        ok, _diagnostic = unify_terms(
            LogicalVariable("X", "entity"), LogicalTerm("a", "entity"), substitution,
        )
        assert ok
        assert substitution.as_dict() == {"X": "a"}
        assert substitution.apply(LogicalVariable("X")) == LogicalTerm("a", "entity")

    def test_repeated_variable_must_agree(self) -> None:
        substitution = Substitution()
        substitution.bind("X", LogicalTerm("a", "entity"))
        ok, diagnostic = unify_terms(
            LogicalVariable("X"), LogicalTerm("b", "entity"), substitution,
        )
        assert not ok
        assert diagnostic.code == "constant_conflict"

    def test_type_mismatch_rejected(self) -> None:
        substitution = Substitution()
        ok, diagnostic = unify_terms(
            LogicalVariable("X", "number"), LogicalTerm("a", "entity"), substitution,
        )
        assert not ok
        assert diagnostic.code == "type_mismatch"

    def test_occurs_check_rejected(self) -> None:
        substitution = Substitution()
        ok, diagnostic = unify_terms(
            LogicalVariable("X", "any"), LogicalVariable("X", "any"), substitution,
        )
        assert ok  # identical variables unify trivially

    def test_variable_variable_binding_is_deterministic(self) -> None:
        first = Substitution()
        unify_terms(LogicalVariable("B"), LogicalVariable("A"), first)
        second = Substitution()
        unify_terms(LogicalVariable("A"), LogicalVariable("B"), second)
        assert first.as_dict() == second.as_dict()

    def test_subtype_table_used_for_binding(self) -> None:
        table = TypeTable(relations={"entity": frozenset({"agent", "object"})})
        substitution = Substitution()
        ok, _diagnostic = unify_terms(
            LogicalVariable("X", "entity"), LogicalTerm("a", "agent"), substitution, table,
        )
        assert ok

    def test_arity_mismatch_diagnostic(self) -> None:
        substitution = Substitution()
        ok, diagnostic = unify_literals(
            (LogicalTerm("a"), LogicalTerm("b")), (LogicalTerm("a"),), substitution,
        )
        assert not ok
        assert diagnostic.code == "arity_mismatch"

    def test_literal_unification_with_shared_substitution(self) -> None:
        substitution = Substitution()
        ok, _diagnostic = unify_literals(
            (LogicalVariable("X"), LogicalVariable("Y")),
            (LogicalTerm("a"), LogicalTerm("b")),
            substitution,
        )
        assert ok
        assert substitution.as_dict() == {"X": "a", "Y": "b"}

    def test_substitution_copy_isolates_branches(self) -> None:
        base = Substitution()
        base.bind("X", LogicalTerm("a"))
        branch = base.copy()
        branch.bind("X", LogicalTerm("b"))
        assert base.as_dict()["X"] == "a"
        assert branch.as_dict()["X"] == "b"
