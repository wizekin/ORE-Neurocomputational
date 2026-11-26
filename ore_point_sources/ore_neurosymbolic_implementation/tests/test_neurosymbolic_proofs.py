"""Proof construction, integrity, tamper detection and lineage tests.

Covers the Section 10 acceptance areas: proof graph integrity; verifier
catches tampered rule, substitution, premise and conclusion; full
source/rule-version lineage; and the Step C5 non-conflatable classification.
"""
from __future__ import annotations

import pytest

from ore_neurosymbolic.contracts import (
    EpistemicStatus,
    Literal,
    LogicalTerm,
    LogicalVariable,
    ProofClass,
    RuleClass,
    RuleDefinition,
    SemanticFact,
    VerificationStatus,
)
from ore_neurosymbolic.proof_builder import ProofBuilder
from ore_neurosymbolic.proof_model import ProofDAG
from ore_neurosymbolic.proof_verifier import IndependentProofVerifier
from ore_neurosymbolic.symbolic_engine import EngineReport, SymbolicEngine

X, Y, Z = (
    LogicalVariable("X", "entity"),
    LogicalVariable("Y", "entity"),
    LogicalVariable("Z", "entity"),
)


def _fact(fact_id: str, subject: str, predicate: str, object_value: str) -> SemanticFact:
    return SemanticFact(
        fact_id=fact_id, subject=subject, predicate=predicate,
        object_value=object_value, epistemic_status=EpistemicStatus.ASSERTED,
    )


def _rule() -> RuleDefinition:
    return RuleDefinition(
        rule_id="r_superior", version="1.0.0",
        typed_variables={"?X": "entity", "?Y": "entity", "?Z": "entity"},
        premises=(Literal("commanded_by", (X, Y)), Literal("commands", (Y, Z))),
        conclusion=Literal("indirect_command", (X, Z)),
        rule_class=RuleClass.STRICT, priority=1.0,
    )


def _engine_with_derivation() -> tuple[SymbolicEngine, EngineReport]:
    engine = SymbolicEngine((_rule(),))
    engine.add_fact(_fact("f1", "captain", "commanded_by", "admiral"))
    engine.add_fact(_fact("f2", "admiral", "commands", "fleet"))
    report = engine.backward_chain(
        Literal("indirect_command", (LogicalTerm("captain"), LogicalTerm("fleet"))),
    )
    return engine, report


class TestProofDAG:
    """Step C1: content addressing, deduplication, immutability."""

    def test_identical_content_reuses_node(self) -> None:
        dag = ProofDAG()
        conclusion = Literal("q", (LogicalTerm("a"),))
        first = dag.add_node(conclusion, ("r1", "1.0.0"), (), ("f1",), "run1")
        second = dag.add_node(conclusion, ("r1", "1.0.0"), (), ("f1",), "run1")
        assert first.node_id == second.node_id
        assert len(dag.nodes) == 1

    def test_different_rule_version_is_a_different_node(self) -> None:
        dag = ProofDAG()
        conclusion = Literal("q", (LogicalTerm("a"),))
        first = dag.add_node(conclusion, ("r1", "1.0.0"), (), ("f1",), "run1")
        second = dag.add_node(conclusion, ("r1", "2.0.0"), (), ("f1",), "run1")
        assert first.node_id != second.node_id

    def test_content_hash_chains_premises(self) -> None:
        dag = ProofDAG()
        premise = dag.add_node(
            Literal("p", (LogicalTerm("a"),)), ("r1", "1.0.0"), (), ("f1",), "run1",
        )
        conclusion = dag.add_node(
            Literal("q", (LogicalTerm("a"),)), ("r2", "1.0.0"),
            (premise.node_id,), ("f2",), "run1",
        )
        recomputed = conclusion.compute_content_hash((premise.content_hash,))
        assert conclusion.content_hash == recomputed

    def test_verify_integrity_detects_tampering(self) -> None:
        dag = ProofDAG()
        conclusion = dag.add_node(
            Literal("q", (LogicalTerm("a"),)), ("r1", "1.0.0"), (), ("f1",), "run1",
        )
        ok, tampered = dag.verify_integrity()
        assert ok and not tampered
        # Simulate tampering by replacing the stored hash.
        victim = dag.nodes[conclusion.node_id]
        from dataclasses import replace as _replace

        dag.nodes[conclusion.node_id] = _replace(victim, content_hash="deadbeef")
        ok, tampered = dag.verify_integrity()
        assert not ok
        assert conclusion.node_id in tampered

    def test_validate_reports_unresolved_premise(self) -> None:
        dag = ProofDAG()
        dag.add_node(
            Literal("q", (LogicalTerm("a"),)), ("r1", "1.0.0"),
            ("missing_node",), ("f1",), "run1",
        )
        issues = dag.validate({})
        assert any(item.code == "unresolved_premise" for item in issues)

    def test_validate_reports_unresolved_rule_version(self) -> None:
        dag = ProofDAG()
        dag.add_node(
            Literal("q", (LogicalTerm("a"),)), ("r_ghost", "9.9.9"), (), ("f1",), "run1",
        )
        issues = dag.validate({})
        assert any(item.code == "unresolved_rule_version" for item in issues)

    def test_validate_reports_substitution_type_error(self) -> None:
        dag = ProofDAG()
        dag.add_node(
            Literal("other_predicate", (LogicalTerm("a"),)), ("r1", "1.0.0"),
            (), ("f1",), "run1",
        )
        issues = dag.validate({("r1", "1.0.0"): _rule()})
        assert any(item.code == "substitution_type_error" for item in issues)

    def test_ancestors_closure(self) -> None:
        dag = ProofDAG()
        base = dag.add_node(
            Literal("p", (LogicalTerm("a"),)), ("r1", "1.0.0"), (), ("f1",), "run1",
        )
        mid = dag.add_node(
            Literal("m", (LogicalTerm("a"),)), ("r2", "1.0.0"), (base.node_id,), ("f2",), "run1",
        )
        top = dag.add_node(
            Literal("q", (LogicalTerm("a"),)), ("r3", "1.0.0"), (mid.node_id,), ("f3",), "run1",
        )
        ancestors = dag.ancestors(top.node_id)
        assert base.node_id in ancestors and mid.node_id in ancestors


class TestProofBuilder:
    """Steps C1-C5: build, validate, trace lineage, classify."""

    def test_build_from_derivations(self) -> None:
        engine, report = _engine_with_derivation()
        builder = ProofBuilder((_rule(),))
        build = builder.build(report, dict(engine.facts._facts), "run1")
        assert len(build.graph.nodes) == 1
        assert not build.issues

    def test_classification_strict_unverified_before_verification(self) -> None:
        engine, report = _engine_with_derivation()
        builder = ProofBuilder((_rule(),))
        build = builder.build(report, dict(engine.facts._facts), "run1")
        for label in build.classification.values():
            assert label is ProofClass.UNVERIFIED_CANDIDATE

    def test_classification_after_verification(self) -> None:
        engine, report = _engine_with_derivation()
        builder = ProofBuilder((_rule(),))
        build = builder.build(report, dict(engine.facts._facts), "run1")
        verifier = IndependentProofVerifier((_rule(),), engine.facts)
        verification = verifier.verify_graph(build.graph.nodes)
        builder.classify_with_verification(build, verification.verified)
        labels = set(build.classification.values())
        assert labels == {ProofClass.STRICT_VERIFIED}

    def test_defeasible_classification_without_verification(self) -> None:
        defeasible_rule = RuleDefinition(
            rule_id="r_def", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("is_a", (X, LogicalTerm("bird"))),),
            conclusion=Literal("ability", (X, LogicalTerm("fly", "activity"))),
            rule_class=RuleClass.DEFEASIBLE, priority=0.5,
            scope={"domain": "zoology"},
        )
        engine = SymbolicEngine((defeasible_rule,), scope_context={"domain": "zoology"})
        engine.add_fact(_fact("f1", "tweety", "is_a", "bird"))
        report = engine.forward_chain()
        builder = ProofBuilder((defeasible_rule,))
        build = builder.build(report, dict(engine.facts._facts), "run1")
        assert set(build.classification.values()) == {ProofClass.DEFEASIBLE}

    def test_lineage_traces_to_source_facts(self) -> None:
        engine, report = _engine_with_derivation()
        builder = ProofBuilder((_rule(),))
        build = builder.build(report, dict(engine.facts._facts), "run1")
        builder.trace_lineage(build, dict(engine.facts._facts))
        records = list(build.lineage.values())
        assert records
        all_sources = {item for record in records for item in record.source_fact_ids}
        assert {"f1", "f2"} <= all_sources

    def test_strict_unverified_pending_detection(self) -> None:
        engine, report = _engine_with_derivation()
        builder = ProofBuilder((_rule(),))
        build = builder.build(report, dict(engine.facts._facts), "run1")
        pending = builder.strict_unverified_nodes(build)
        assert pending


class TestIndependentVerifier:
    """Step C3: re-execution without trusting engine claims."""

    def test_strict_node_verifies(self) -> None:
        engine, report = _engine_with_derivation()
        builder = ProofBuilder((_rule(),))
        build = builder.build(report, dict(engine.facts._facts), "run1")
        verifier = IndependentProofVerifier((_rule(),), engine.facts)
        verification = verifier.verify_graph(build.graph.nodes)
        assert all(verification.verified.values())
        assert not verification.tampered

    def test_tampered_conclusion_detected(self) -> None:
        engine, report = _engine_with_derivation()
        builder = ProofBuilder((_rule(),))
        build = builder.build(report, dict(engine.facts._facts), "run1")
        from dataclasses import replace as _replace

        node = next(iter(build.graph.nodes.values()))
        tampered_node = _replace(
            node,
            conclusion=Literal(
                "fabricated", node.conclusion.args,
            ),
        )
        verifier = IndependentProofVerifier((_rule(),), engine.facts)
        verification = verifier.verify_graph({node.node_id: tampered_node})
        assert node.node_id in verification.tampered
        assert verification.diagnostics[node.node_id] == "conclusion_mismatch"

    def test_tampered_rule_version_detected(self) -> None:
        engine, report = _engine_with_derivation()
        builder = ProofBuilder((_rule(),))
        build = builder.build(report, dict(engine.facts._facts), "run1")
        ghost_rule = RuleDefinition(
            rule_id="r_superior", version="9.9.9",
            typed_variables=_rule().typed_variables,
            premises=_rule().premises, conclusion=_rule().conclusion,
            rule_class=RuleClass.STRICT,
        )
        verifier = IndependentProofVerifier((ghost_rule,), engine.facts)
        verification = verifier.verify_graph(build.graph.nodes)
        assert verification.tampered
        assert verification.diagnostics[next(iter(verification.tampered))] == (
            "rule_version_unresolvable"
        )

    def test_missing_premise_fact_detected(self) -> None:
        engine, report = _engine_with_derivation()
        builder = ProofBuilder((_rule(),))
        build = builder.build(report, dict(engine.facts._facts), "run1")
        # Remove one premise fact behind the verifier's back.
        engine.facts.remove("f2")
        verifier = IndependentProofVerifier((_rule(),), engine.facts)
        verification = verifier.verify_graph(build.graph.nodes)
        assert verification.tampered

    def test_verification_level_none_keeps_unverified(self) -> None:
        engine, report = _engine_with_derivation()
        builder = ProofBuilder((_rule(),))
        build = builder.build(report, dict(engine.facts._facts), "run1")
        verifier = IndependentProofVerifier((_rule(),), engine.facts)
        verification = verifier.verify_graph(build.graph.nodes, verification_level="none")
        assert not any(verification.verified.values())
        assert not verification.tampered

    def test_verifier_ignores_defeasible_nodes(self) -> None:
        defeasible_rule = RuleDefinition(
            rule_id="r_def", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("is_a", (X, LogicalTerm("bird"))),),
            conclusion=Literal("ability", (X, LogicalTerm("fly", "activity"))),
            rule_class=RuleClass.DEFEASIBLE, priority=0.5,
            scope={"domain": "zoology"},
        )
        engine = SymbolicEngine((defeasible_rule,), scope_context={"domain": "zoology"})
        engine.add_fact(_fact("f1", "tweety", "is_a", "bird"))
        report = engine.forward_chain()
        builder = ProofBuilder((defeasible_rule,))
        build = builder.build(report, dict(engine.facts._facts), "run1")
        verifier = IndependentProofVerifier((defeasible_rule,), engine.facts)
        verification = verifier.verify_graph(build.graph.nodes)
        assert not any(verification.verified.values())
        assert not verification.tampered


class TestVerificationStatusMarking:
    """Step C3 integration with the DAG verifier status."""

    def test_mark_verifier_status_preserves_content_hash(self) -> None:
        dag = ProofDAG()
        node = dag.add_node(
            Literal("q", (LogicalTerm("a"),)), ("r1", "1.0.0"), (), ("f1",), "run1",
        )
        original_hash = node.content_hash
        assert dag.mark_verifier_status(node.node_id, VerificationStatus.VERIFIED)
        updated = dag.nodes[node.node_id]
        assert updated.verifier_status is VerificationStatus.VERIFIED
        assert updated.content_hash == original_hash

    def test_mark_unknown_node_returns_false(self) -> None:
        dag = ProofDAG()
        assert not dag.mark_verifier_status("ghost", VerificationStatus.VERIFIED)
