"""Phase B tests: semantic interpretation, USG grounding, ambiguity (Steps 8-16)."""
from __future__ import annotations

import pytest
from test_ore_hic_helpers import ASSIGNMENT_IDEA

from ore_hic import (
    AmbiguityAnalyzer,
    CompilationSession,
    HICUSGAdapter,
    Modality,
    Normalizer,
    SemanticInterpreter,
    StarterUSGRegistry,
    TextInputAdapter,
    entropy,
)
from ore_hic.segmentation import IdeaSegmenter


def _interpret(idea: str = ASSIGNMENT_IDEA):
    session = CompilationSession(user="u")
    record = TextInputAdapter().capture(session, idea, source_identity="u")
    view = Normalizer().normalize(record)
    segmentation = IdeaSegmenter().run(session, record, view)
    interpreter = SemanticInterpreter(HICUSGAdapter())
    graph = interpreter.interpret(session, record, segmentation)
    return session, record, graph


def test_step8_concepts_link_to_usg_or_stay_unresolved():
    _, _, graph = _interpret()
    resolved = graph.resolved_concepts()
    assert resolved, "assignment ideas must ground concepts in USG"
    for concept in resolved:
        assert concept.usg_reference is not None
        assert concept.usg_reference.concept_id.startswith("USG-")
        assert concept.usg_reference.rule_id  # rule traceability


def test_step8_unknown_words_do_not_become_usg_concepts():
    session = CompilationSession(user="u")
    record = TextInputAdapter().capture(
        session, "The blorpmatic quibbleotron assigns tasks to modules.", source_identity="u"
    )
    view = Normalizer().normalize(record)
    segmentation = IdeaSegmenter().run(session, record, view)
    graph = SemanticInterpreter(HICUSGAdapter()).interpret(session, record, segmentation)
    unresolved = [c for c in graph.concepts if c.unresolved]
    assert unresolved, "unknown words must surface as unresolved candidates"
    for concept in unresolved:
        assert concept.usg_reference is None


def test_step9_relations_cite_predicate_frames():
    _, _, graph = _interpret()
    assert graph.relations, "assign phrasing must produce relations"
    for relation in graph.relations:
        assert relation.frame_id.startswith("USG-FRAME-")
        assert relation.predicate
        assert relation.roles
        assert relation.span.artifact_id


def test_step10_attributes_carry_units_and_modality():
    _, _, graph = _interpret()
    assert graph.attributes
    for attribute in graph.attributes:
        assert attribute.modality in set(Modality)
        assert attribute.measurement_status in {"MEASURED", "ASSUMED", "ESTIMATED", "USER_DEFINED"}
        assert attribute.rule_id


def test_step11_usg_class_hierarchy_resolves_inheritance():
    usg = HICUSGAdapter()
    assert usg.compatible("USG-TASK", "USG-WORK-ITEM")
    assert usg.compatible("USG-TASK", "USG-THING")
    assert not usg.compatible("USG-TASK", "USG-RESOURCE")  # task is not a resource


def test_step13_competing_interpretations_are_preserved():
    _, _, graph = _interpret()
    assert graph.interpretations, "ambiguous 'capacity' must keep alternatives"
    interpretation = graph.interpretations[0]
    readings = [h.reading for h in interpretation.hypotheses]
    assert len(readings) >= 2
    # Posterior weights follow P(h|X,C) proportional to P(X|h,C) * P(h|C).
    for hypothesis in interpretation.hypotheses:
        assert hypothesis.posterior_weight == pytest.approx(
            hypothesis.likelihood * hypothesis.prior
        )


def test_entropy_and_information_gain_math():
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy([1.0, 0.0]) == pytest.approx(0.0)
    analyzer = AmbiguityAnalyzer()
    ig = analyzer.expected_information_gain("amb", [0.5, 0.5])
    assert ig == pytest.approx(1.0, abs=1e-9)
    ig_skewed = analyzer.expected_information_gain("amb", [0.99, 0.01])
    assert ig_skewed < ig


def test_step14_blocking_vs_optional_ambiguity():
    _, _, graph = _interpret()
    analyzer = AmbiguityAnalyzer()
    ambiguities = analyzer.detect(graph)
    assert ambiguities
    blocking = [a for a in ambiguities if a.blocking]
    optional = [a for a in ambiguities if not a.blocking]
    assert blocking, "near-tie 'capacity' readings must block"
    assert optional, "missing optional parameters must not block"


def test_step15_questions_are_ranked_by_value():
    _, _, graph = _interpret()
    analyzer = AmbiguityAnalyzer()
    ambiguities = analyzer.detect(graph)
    questions = analyzer.rank_questions(ambiguities, graph)
    assert questions
    assert [q.rank for q in questions] == list(range(1, len(questions) + 1))
    scores = [q.priority_score() for q in questions]
    assert scores == sorted(scores, reverse=True)
    # Blocking (high-impact) questions outrank low-impact parameter questions.
    top = questions[0]
    low_impact = [q for q in questions if q.decision_impact.value == "LOW"]
    if low_impact:
        assert top.priority_score() >= low_impact[0].priority_score()


def test_step16_assumptions_record_source_and_impact():
    _, _, graph = _interpret()
    analyzer = AmbiguityAnalyzer()
    ambiguities = analyzer.detect(graph)
    assumptions = analyzer.record_assumptions(ambiguities, graph)
    by_id = {a.assumption_id: a for a in assumptions}
    linearity = by_id["as-model-linearity"]
    assert linearity.impact.value == "HIGH"
    assert linearity.requires_confirmation is True
    assert linearity.source.value == "COMPILER"
    # Once confirmed by the user, the assumption no longer blocks.
    confirmed = analyzer.record_assumptions(ambiguities, graph, confirmed={"as-model-linearity"})
    by_id = {a.assumption_id: a for a in confirmed}
    assert by_id["as-model-linearity"].requires_confirmation is False
    assert by_id["as-model-linearity"].source.value == "USER"


def test_user_facts_are_distinguishable_from_compiler_assumptions():
    _, _, graph = _interpret()
    analyzer = AmbiguityAnalyzer()
    assumptions = analyzer.record_assumptions(graph and analyzer.detect(graph), graph)
    compiler_sourced = [a for a in assumptions if a.source.value == "COMPILER"]
    assert compiler_sourced  # compiler assumptions are labeled as such


def test_starter_registry_is_explicit_and_versioned():
    registry = StarterUSGRegistry()
    assert registry.REGISTRY_VERSION
    assert registry.resolve_concept("task") is not None
    assert registry.resolve_concept("taskz") is None
