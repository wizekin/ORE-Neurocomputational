"""Phase C tests: requirements and behavior formalization (Steps 17-26)."""
from __future__ import annotations

import pytest
from test_ore_hic_helpers import ASSIGNMENT_IDEA

from ore_hic import (
    AmbiguityAnalyzer,
    CompilationSession,
    ConstraintKind,
    GateViolation,
    Normalizer,
    RequirementsFormalizer,
    SemanticInterpreter,
    TextInputAdapter,
)
from ore_hic.requirements import StateMachine, StateTransition
from ore_hic.segmentation import IdeaSegmenter


@pytest.fixture(scope="module")
def spec():
    session = CompilationSession(user="u")
    record = TextInputAdapter().capture(session, ASSIGNMENT_IDEA, source_identity="u")
    view = Normalizer().normalize(record)
    segmentation = IdeaSegmenter().run(session, record, view)
    graph = SemanticInterpreter().interpret(session, record, segmentation)
    ambiguity = AmbiguityAnalyzer().analyze(session, graph, resolved={"amb-001"}, confirmed={"as-model-linearity"})
    return RequirementsFormalizer().formalize(session, graph, segmentation, ambiguity)


def test_requirements_have_stable_ids_and_provenance(spec):
    ids = [r.requirement_id for r in spec.requirements]
    assert len(ids) == len(set(ids)), "requirement ids must be unique"
    assert ids[0] == "req-001"
    for requirement in spec.requirements:
        assert requirement.source_rule  # every requirement has provenance


def test_step17_objectives_separate_primary_secondary_prohibited(spec):
    assert spec.objectives[0].title == "Primary objective"
    prohibited = [o for o in spec.objectives if "Prohibited" in o.title]
    assert prohibited, "prohibited outcomes must be explicit"


def test_step18_interfaces_have_types_and_units(spec):
    by_name = {i.name: i for i in spec.interfaces}
    for name in ("tasks", "modules", "eligibility", "assignment", "report"):
        assert name in by_name
    assert by_name["modules"].unit is not None  # capacity unit declared
    assert by_name["assignment"].direction == "OUTPUT"
    assert by_name["tasks"].direction == "INPUT"
    for interface in spec.interfaces:
        assert interface.data_type
        assert interface.cardinality
        assert interface.requirement_id in {r.requirement_id for r in spec.requirements}


def test_step19_actors_and_permissions(spec):
    actors = {a.actor: a for a in spec.actors}
    assert "user" in actors and actors["user"].may_approve is True
    assert actors["hic-compiler"].may_execute is True


def test_step20_preconditions_and_postconditions(spec):
    assert spec.preconditions and spec.postconditions
    formalizable = [p for p in (*spec.preconditions, *spec.postconditions) if p.formalizable]
    assert formalizable, "predicates should be executable where possible"
    post_kinds = {p.kind for p in spec.postconditions}
    assert "ALL_TASKS_ASSIGNED" in post_kinds
    assert "CAPACITY_RESPECTED" in post_kinds


def test_step21_invariants(spec):
    kinds = {p.kind for p in spec.invariants}
    assert "NO_DUPLICATE_ASSIGNMENT" in kinds
    informal = [p for p in spec.invariants if not p.formalizable]
    assert informal, "informal obligations must be labeled, not hidden"


def test_step22_hard_constraints_cannot_carry_weights(spec):
    hard = spec.hard_constraints()
    soft = spec.soft_constraints()
    assert hard and soft
    for constraint in hard:
        assert constraint.weight is None
    for constraint in soft:
        assert constraint.weight is not None
    with pytest.raises(GateViolation):
        # A hard constraint with an objective weight is a schema violation.
        from ore_hic.requirements import ConstraintSpec

        ConstraintSpec("bad", ConstraintKind.HARD, "x", "x <= 1", "global", 0.5, 0.0, "req-x")


def test_step23_state_machine_rejects_illegal_changes():
    machine = StateMachine(
        states=("A", "B", "C"),
        transitions=(StateTransition("t1", "A", "B", "go", "true", "", False),),
        initial="A",
    )
    assert machine.can_transition("A", "go")
    assert machine.transition("A", "go").target == "B"
    with pytest.raises(GateViolation):
        machine.transition("B", "go")  # illegal from B
    with pytest.raises(GateViolation):
        machine.transition("A", "skip")


def test_step23_spec_state_machine_has_recovery(spec):
    transitions = spec.state_machine.transitions
    assert any(t.recovery for t in transitions), "recovery transitions must exist"
    events = {(t.source, t.event) for t in transitions}
    assert ("RECEIVED", "validate_input") in events


def test_step24_failure_cases_are_enumerated(spec):
    triggers = " ".join(f.trigger for f in spec.failure_cases)
    assert "input" in triggers
    assert "timeout" in triggers or "cancellation" in triggers
    reversible = [f for f in spec.failure_cases if f.reversible]
    compensating = [f for f in spec.failure_cases if not f.reversible]
    assert reversible and compensating, "reversibility must be classified"


def test_step25_termination_is_explicit(spec):
    termination = spec.termination
    assert termination.termination_argument
    assert termination.max_iterations is not None
    assert termination.success and termination.failure and termination.cancellation and termination.timeout


def test_step26_acceptance_criteria_are_measurable(spec):
    assert spec.acceptance_criteria
    for criterion in spec.acceptance_criteria:
        assert criterion.test_kind in {"PROPERTY", "EXAMPLE", "NEGATIVE", "BOUNDARY", "METRIC"}
        assert criterion.requirement_id
    negative = [c for c in spec.acceptance_criteria if c.test_kind == "NEGATIVE"]
    boundary = [c for c in spec.acceptance_criteria if c.test_kind == "BOUNDARY"]
    assert negative and boundary, "negative and boundary tests are required"


def test_unresolved_obligations_are_visible(spec):
    # The spec fixture resolved everything; unresolved items must surface when blocked.
    assert isinstance(spec.unresolved_obligations, tuple)
