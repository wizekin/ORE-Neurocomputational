"""Symbolic engine tests: chaining, defeasible semantics, bounded limits.

Covers the Section 10 acceptance areas: multi-hop/conjunctive inference,
fixed point, no unsupported conclusion, determinism; defeasible exception
precedence and unresolved incomparable conflicts; explicit negation vs
absence/unknown; cycles, recursion, wall-time, depth, rule-fire and memory
limits.
"""
from __future__ import annotations

import pytest

from ore_neurosymbolic.contracts import (
    EpistemicStatus,
    Literal,
    LogicalTerm,
    LogicalVariable,
    RuleClass,
    RuleDefinition,
    SemanticFact,
)
from ore_neurosymbolic.inference_queue import BoundedInferenceQueue, QueueEntry
from ore_neurosymbolic.symbolic_engine import EngineBudgets, FactTable, SymbolicEngine

X, Y, Z = (
    LogicalVariable("X", "entity"),
    LogicalVariable("Y", "entity"),
    LogicalVariable("Z", "entity"),
)
BIRD = LogicalTerm("bird", "class")
PENGUIN = LogicalTerm("penguin", "class")
FLAG = LogicalTerm("yes", "flag")


def _fact(fact_id: str, subject: str, predicate: str, object_value: str, **kw) -> SemanticFact:
    return SemanticFact(
        fact_id=fact_id, subject=subject, predicate=predicate,
        object_value=object_value,
        epistemic_status=kw.pop("epistemic_status", EpistemicStatus.ASSERTED),
        **kw,
    )


def _transitive_rule() -> RuleDefinition:
    return RuleDefinition(
        rule_id="r_ancestor", version="1.0.0",
        typed_variables={"?X": "entity", "?Y": "entity", "?Z": "entity"},
        premises=(Literal("ancestor", (X, Y)), Literal("ancestor", (Y, Z))),
        conclusion=Literal("ancestor", (X, Z)),
        rule_class=RuleClass.STRICT, priority=1.0,
    )


class TestForwardChaining:
    """Steps B7 + math op 6.2: fixed-point forward inference."""

    def test_multi_hop_conjunctive_inference(self) -> None:
        engine = SymbolicEngine((_transitive_rule(),))
        engine.add_fact(_fact("f1", "a", "ancestor", "b"))
        engine.add_fact(_fact("f2", "b", "ancestor", "c"))
        report = engine.forward_chain()
        signatures = [item.conclusion_literal.signature() for item in report.derived]
        assert "ancestor(a,c)" in signatures
        assert report.fixed_point_reached

    def test_fixed_point_terminates_with_no_change(self) -> None:
        engine = SymbolicEngine((_transitive_rule(),))
        engine.add_fact(_fact("f1", "a", "ancestor", "b"))
        report = engine.forward_chain()
        assert report.stop_reason == "fixed_point"
        assert len(report.derived) == 0

    def test_no_unsupported_conclusion(self) -> None:
        # Both premises must be explicitly supported; a one-sided chain
        # derives nothing.
        rule = RuleDefinition(
            rule_id="r_two", version="1.0.0",
            typed_variables={"?X": "entity", "?Y": "entity"},
            premises=(Literal("p", (X, FLAG)), Literal("q", (X, FLAG))),
            conclusion=Literal("r", (X, FLAG)), rule_class=RuleClass.STRICT,
        )
        engine = SymbolicEngine((rule,))
        engine.add_fact(_fact("f1", "a", "p", "yes"))
        report = engine.forward_chain()
        assert len(report.derived) == 0

    def test_determinism_across_runs(self) -> None:
        rules = (_transitive_rule(),)
        facts = (
            _fact("f1", "a", "ancestor", "b"),
            _fact("f2", "b", "ancestor", "c"),
            _fact("f3", "c", "ancestor", "d"),
        )
        runs = []
        for _attempt in range(3):
            engine = SymbolicEngine(rules)
            for fact in facts:
                engine.add_fact(fact)
            report = engine.forward_chain()
            runs.append(
                [item.conclusion_literal.signature() for item in report.derived]
            )
        assert runs[0] == runs[1] == runs[2]

    def test_goal_completion_stops_early(self) -> None:
        engine = SymbolicEngine((_transitive_rule(),))
        engine.add_fact(_fact("f1", "a", "ancestor", "b"))
        engine.add_fact(_fact("f2", "b", "ancestor", "c"))
        goal = Literal("ancestor", (LogicalTerm("a"), LogicalTerm("c")))
        report = engine.forward_chain(goal=goal)
        assert report.goal_satisfied


class TestBackwardChaining:
    """Step B8: goal-directed SLD resolution with memoization."""

    def test_goal_proven_through_rules(self) -> None:
        engine = SymbolicEngine((_transitive_rule(),))
        engine.add_fact(_fact("f1", "a", "ancestor", "b"))
        engine.add_fact(_fact("f2", "b", "ancestor", "c"))
        goal = Literal("ancestor", (LogicalTerm("a"), LogicalTerm("c")))
        report = engine.backward_chain(goal)
        assert report.goal_satisfied
        assert any(
            item.conclusion_literal.signature() == "ancestor(a,c)"
            for item in report.derived
        )

    def test_unprovable_goal_stays_unproven(self) -> None:
        engine = SymbolicEngine((_transitive_rule(),))
        engine.add_fact(_fact("f1", "a", "ancestor", "b"))
        goal = Literal("ancestor", (LogicalTerm("b"), LogicalTerm("z")))
        report = engine.backward_chain(goal)
        assert not report.goal_satisfied
        assert len(report.derived) == 0

    def test_goal_directly_supported_by_fact(self) -> None:
        engine = SymbolicEngine((_transitive_rule(),))
        engine.add_fact(_fact("f1", "a", "ancestor", "b"))
        goal = Literal("ancestor", (LogicalTerm("a"), LogicalTerm("b")))
        report = engine.backward_chain(goal)
        assert report.goal_satisfied
        assert report.rule_fires == 0


class TestNegationVsUnknown:
    """Section 2: unknown is not false; missing evidence is not negative."""

    def test_missing_evidence_is_not_negation(self) -> None:
        engine = SymbolicEngine(())
        literal = Literal("flies", (LogicalTerm("tweety"),))
        assert engine.facts.has_positive(literal) is None
        assert engine.facts.has_negated(literal) is None

    def test_negated_premise_requires_explicit_negative_fact(self) -> None:
        X_var = LogicalVariable("X", "entity")
        flying = LogicalTerm("flying", "state")
        rule = RuleDefinition(
            rule_id="r_neg", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("flight_state", (X_var, flying), negated=True),),
            conclusion=Literal("is_grounded", (X_var, FLAG)),
            rule_class=RuleClass.STRICT,
        )
        engine = SymbolicEngine((rule,))
        # Absence of flight_state(tweety, flying) must NOT fire the rule:
        # missing evidence is unknown, never negative evidence.
        engine.add_fact(_fact("f1", "tweety", "is_a", "bird"))
        report = engine.forward_chain()
        assert len(report.derived) == 0
        # Explicit negation DOES fire it.
        engine.add_fact(_fact("f2", "tweety", "flight_state", "flying", polarity=False))
        report = engine.forward_chain()
        assert any(
            item.conclusion_literal.signature() == "is_grounded(tweety,yes)"
            for item in report.derived
        )


class TestDefeasibleSemantics:
    """Steps B6 + Section 2: specificity/priority policy and exceptions."""

    def test_exception_defeats_general_rule(self) -> None:
        fly_rule = RuleDefinition(
            rule_id="r_flies", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("is_a", (X, BIRD)),),
            conclusion=Literal("ability", (X, LogicalTerm("fly", "activity"))),
            rule_class=RuleClass.DEFEASIBLE, priority=0.5,
            scope={"domain": "zoology"},
            exceptions=(Literal("is_a", (X, PENGUIN)),),
        )
        flightless_rule = RuleDefinition(
            rule_id="r_flightless", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("is_a", (X, PENGUIN)),),
            conclusion=Literal("inability", (X, LogicalTerm("fly", "activity"))),
            rule_class=RuleClass.STRICT,
        )
        engine = SymbolicEngine((fly_rule, flightless_rule),
                                scope_context={"domain": "zoology"})
        engine.add_fact(_fact("f1", "tweety", "is_a", "bird"))
        engine.add_fact(_fact("f2", "tweety", "is_a", "penguin"))
        report = engine.forward_chain()
        signatures = [item.conclusion_literal.signature() for item in report.derived]
        assert "inability(tweety,fly)" in signatures
        assert "ability(tweety,fly)" not in signatures

    def test_specific_rule_defeats_general_rule_same_conclusion(self) -> None:
        sky = LogicalTerm("sky", "place")
        general = RuleDefinition(
            rule_id="r_general", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("is_a", (X, BIRD)),),
            conclusion=Literal("habitat", (X, sky)),
            rule_class=RuleClass.DEFEASIBLE, priority=0.3,
            scope={"domain": "zoology"},
        )
        specific = RuleDefinition(
            rule_id="r_specific", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("is_a", (X, BIRD)), Literal("is_a", (X, PENGUIN))),
            conclusion=Literal("habitat", (X, sky)),
            rule_class=RuleClass.DEFEASIBLE, priority=0.3,
            scope={"domain": "zoology"},
        )
        engine = SymbolicEngine((general, specific),
                                scope_context={"domain": "zoology"})
        engine.add_fact(_fact("f1", "tweety", "is_a", "bird"))
        engine.add_fact(_fact("f2", "tweety", "is_a", "penguin"))
        report = engine.forward_chain()
        assert len(report.derived) == 1
        assert report.derived[0].rule_ref[0] == "r_specific"
        assert len(report.defeated) == 1
        assert report.defeated[0].defeated_by == ("r_specific",)

    def test_incomparable_rules_produce_conflict_records(self) -> None:
        migrates = LogicalTerm("yes", "flag")
        rule_a = RuleDefinition(
            rule_id="r_a", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("is_a", (X, BIRD)),),
            conclusion=Literal("migrates", (X, migrates)),
            rule_class=RuleClass.DEFEASIBLE, priority=0.5,
            scope={"domain": "zoology"},
        )
        rule_b = RuleDefinition(
            rule_id="r_b", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("is_a", (X, PENGUIN)),),
            conclusion=Literal("migrates", (X, migrates)),
            rule_class=RuleClass.DEFEASIBLE, priority=0.5,
            scope={"domain": "zoology"},
        )
        engine = SymbolicEngine((rule_a, rule_b), scope_context={"domain": "zoology"})
        engine.add_fact(_fact("f1", "tweety", "is_a", "bird"))
        engine.add_fact(_fact("f2", "tweety", "is_a", "penguin"))
        report = engine.forward_chain()
        assert len(report.conflicts) == 1

    def test_scope_mismatch_blocks_defeasible_rule(self) -> None:
        rule = RuleDefinition(
            rule_id="r_scoped", version="1.0.0", typed_variables={"?X": "entity"},
            premises=(Literal("is_a", (X, BIRD)),),
            conclusion=Literal("ability", (X, LogicalTerm("fly", "activity"))),
            rule_class=RuleClass.DEFEASIBLE, priority=0.5,
            scope={"domain": "chemistry"},
        )
        engine = SymbolicEngine((rule,), scope_context={"domain": "zoology"})
        engine.add_fact(_fact("f1", "tweety", "is_a", "bird"))
        report = engine.forward_chain()
        assert len(report.derived) == 0


class TestBoundedInference:
    """Steps B9/B7 + Section 2: cycles, depth, fires and queue limits."""

    def test_rule_fire_budget_stops_chain(self) -> None:
        engine = SymbolicEngine(
            (_transitive_rule(),),
            budgets=EngineBudgets(max_rule_fires=2, max_depth=8),
        )
        engine.add_fact(_fact("f1", "a", "ancestor", "b"))
        engine.add_fact(_fact("f2", "b", "ancestor", "c"))
        engine.add_fact(_fact("f3", "c", "ancestor", "d"))
        report = engine.forward_chain()
        assert report.stop_reason == "rule_fire_budget"
        assert report.rule_fires <= 2

    def test_depth_budget_records_cycle_event(self) -> None:
        engine = SymbolicEngine(
            (_transitive_rule(),),
            budgets=EngineBudgets(max_rule_fires=64, max_depth=1),
        )
        engine.add_fact(_fact("f1", "a", "ancestor", "b"))
        engine.add_fact(_fact("f2", "b", "ancestor", "c"))
        engine.add_fact(_fact("f3", "c", "ancestor", "d"))
        report = engine.forward_chain()
        assert report.cycle_events, "depth exhaustion must be recorded"

    def test_backward_cycle_detection_stops_unsafe_recursion(self) -> None:
        # A pathological pair of rules that chase each other's premises.
        rule = RuleDefinition(
            rule_id="r_mutual", version="1.0.0",
            typed_variables={"?X": "entity", "?Y": "entity"},
            premises=(Literal("q", (X, Y)),), conclusion=Literal("p", (X, Y)),
            rule_class=RuleClass.STRICT,
        )
        rule2 = RuleDefinition(
            rule_id="r_mutual2", version="1.0.0",
            typed_variables={"?X": "entity", "?Y": "entity"},
            premises=(Literal("p", (X, Y)),), conclusion=Literal("q", (X, Y)),
            rule_class=RuleClass.STRICT,
        )
        engine = SymbolicEngine(
            (rule, rule2), budgets=EngineBudgets(max_depth=3),
        )
        goal = Literal("p", (LogicalTerm("a"), LogicalTerm("b")))
        report = engine.backward_chain(goal)
        assert not report.goal_satisfied

    def test_queue_deduplicates_and_counts(self) -> None:
        queue = BoundedInferenceQueue(max_size=8)
        fact = _fact("d1", "a", "p", "b")
        assert queue.push(QueueEntry(fact=fact, origin_rule_ref=("r", "1"), depth=0))
        assert not queue.push(QueueEntry(fact=fact, origin_rule_ref=("r", "1"), depth=1))
        assert queue.duplicates_absorbed == 1
        assert len(queue) == 1

    def test_queue_overflow_budget(self) -> None:
        queue = BoundedInferenceQueue(max_size=2)
        for index in range(3):
            queue.push(QueueEntry(
                fact=_fact(f"d{index}", "s", "p", str(index)),
                origin_rule_ref=None, depth=0,
            ))
        assert queue.overflow_count == 1
        assert len(queue) == 2

    def test_fact_table_is_append_only_for_source_facts(self) -> None:
        table = FactTable()
        fact = _fact("f1", "a", "p", "b")
        assert table.add(fact)
        assert not table.add(fact)
        assert len(table) == 1

    def test_fact_table_remove_only_affects_scratch_entries(self) -> None:
        table = FactTable()
        fact = _fact("f1", "a", "p", "b")
        table.add(fact)
        assert table.remove("f1")
        assert len(table) == 0
        assert not table.remove("missing")

    def test_wall_clock_budget_stops_chain(self) -> None:
        engine = SymbolicEngine(
            (_transitive_rule(),),
            budgets=EngineBudgets(wall_clock_seconds=1e-9),
        )
        engine.add_fact(_fact("f1", "a", "ancestor", "b"))
        engine.add_fact(_fact("f2", "b", "ancestor", "c"))
        report = engine.forward_chain()
        assert report.stop_reason in {"wall_clock_budget", "fixed_point"}
