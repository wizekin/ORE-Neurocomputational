"""Contract, mathematical, biological, pipeline, and SQLite tests for Point 04."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from ore_testing_effect import (
    AnteriorCingulateErrorMonitor,
    BiologicalTestingEffectNetwork,
    Feedback,
    LearningTarget,
    NeuralPopulation,
    RetrievalAttempt,
    TestingEffectProcessor,
    TestingEffectRequest,
    TestingEffectState,
    OPERATION_REGISTRY,
    assess_attempt,
    calculate_future_priority,
    classify_errors,
    cosine_similarity,
    record_result_to_sqlite,
    sigmoid,
    softmax,
    weighted_coverage,
)
from ore_logger.database import LoggerDatabase


def build_target(criterion_reliability: float = 0.95) -> LearningTarget:
    """Build a structured target with weighted content, relations, and sequence criteria."""
    components = (
        "active generation from memory",
        "improved later accessibility",
        "retrieval reveals knowledge gaps",
        "benefits depend on learning conditions",
    )
    relations = (
        "retrieval practice requires generating information without direct exposure",
        "retrieval attempts can improve later access under suitable conditions",
    )
    return LearningTarget(
        target_id="testing-effect",
        prompt="Explain why retrieval practice can improve later retention.",
        components=components,
        relations=relations,
        component_weights={item: 1.0 for item in components},
        relation_weights={item: 1.0 for item in relations},
        expected_order=components,
        criterion_source="validated_answer_key",
        criterion_reliability=criterion_reliability,
        initial_encoding_quality=0.8,
    )


def build_attempt(target: LearningTarget, partial: bool = False,
                  confidence: float = 0.72, cue_type: str = "causal") -> RetrievalAttempt:
    """Construct either a complete independent attempt or a deliberately incomplete attempt."""
    if partial:
        recalled = target.components[:2]
        recalled_relations = target.relations[:1]
        component_status = {
            target.components[0]: "correct",
            target.components[1]: "correct",
            target.components[2]: "missing",
            target.components[3]: "missing",
        }
        relation_status = {
            target.relations[0]: "correct",
            target.relations[1]: "missing",
        }
    else:
        recalled = target.components
        recalled_relations = target.relations
        component_status = {item: "correct" for item in target.components}
        relation_status = {item: "correct" for item in target.relations}
    return RetrievalAttempt(
        cue="Explain why retrieval practice can improve later memory and reveal gaps.",
        response="Retrieval requires generating information and can improve later access.",
        confidence=confidence,
        cue_type=cue_type,
        recalled_components=recalled,
        recalled_relations=recalled_relations,
        component_status=component_status,
        relation_status=relation_status,
        claim_confidence={item: confidence for item in target.components},
        response_time_ms=12000.0,
    )


def build_feedback() -> Feedback:
    """Return reliable corrective information for the testing-effect target."""
    return Feedback(
        correct_representation=(
            "Retrieval practice generates information from memory, can improve later access, "
            "reveals knowledge gaps, and has condition-dependent benefits."
        ),
        explanation="Correct missing details, then retrieve the corrected explanation without notes.",
        source="validated_answer_key",
        source_reliability=0.98,
        verified=True,
    )


def test_registry_has_exactly_22_source_ordered_operations():
    """Keep stage identifiers and order aligned with the source algorithm."""
    assert len(OPERATION_REGISTRY) == 22
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P04-{index:02d}" for index in range(1, 23)
    ]
    assert all(item.mathematical_form and item.biological_reference for item in OPERATION_REGISTRY)


def test_mathematical_primitives_are_stable_and_bounded():
    """Validate sigmoid, softmax, cosine similarity, and weighted coverage."""
    assert sigmoid(-1000.0) == 0.0
    assert sigmoid(1000.0) == 1.0
    probabilities = softmax({"a": 1000.0, "b": 999.0, "c": -1000.0})
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["a"] > probabilities["b"] > probabilities["c"]
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert weighted_coverage(("important", "minor"), ("important",),
                             {"important": 3.0, "minor": 1.0}) == pytest.approx(0.75)


def test_attempt_scoring_separates_components_relations_order_and_confidence():
    """Score explicit evidence by dimension rather than equating confidence with correctness."""
    target = build_target()
    attempt = build_attempt(target, partial=True, confidence=0.95)
    result = assess_attempt(target, attempt)
    assert result["component_coverage"] == pytest.approx(0.5)
    assert result["relation_coverage"] == pytest.approx(0.5)
    assert result["order_accuracy"] < 1.0
    assert result["confidence"] == 0.95
    assert result["evidenced_accuracy"] < result["confidence"]
    diagnosis = classify_errors(target, attempt, result)
    assert diagnosis["primary_error"] == "omission"
    assert diagnosis["diagnosis_allowed"] is True


def test_unreliable_criterion_blocks_definitive_error_diagnosis():
    """Do not diagnose the learner when the answer criterion is itself uncertain."""
    target = build_target(criterion_reliability=0.4)
    attempt = build_attempt(target, partial=True)
    diagnosis = classify_errors(target, attempt, assess_attempt(target, attempt))
    assert diagnosis["primary_error"] == "criterion_uncertain"
    assert diagnosis["diagnosis_allowed"] is False
    assert diagnosis["probabilities"] == {}


def test_pipeline_runs_all_22_operations_and_preserves_initial_attempt():
    """Integrate feedback, retest, alternate cues, delayed retrieval, and the scheduler handoff."""
    target = build_target()
    initial = build_attempt(target, partial=True, confidence=0.90)
    retest = build_attempt(target, partial=False, confidence=0.88)
    alternative = build_attempt(target, partial=False, confidence=0.82, cue_type="application")
    delayed = build_attempt(target, partial=False, confidence=0.84, cue_type="relational")
    processor = TestingEffectProcessor()
    result = processor.process(TestingEffectRequest(
        target=target,
        initial_attempt=initial,
        feedback=build_feedback(),
        retest_attempt=retest,
        alternative_attempts=(alternative,),
        delayed_attempts=(delayed,),
        current_time_utc="2026-01-01T00:00:00Z",
        delay_after_hours=72.0,
    ))
    assert len(result.operation_trace) == 22
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"P04-{index:02d}" for index in range(1, 23)
    ]
    assert result.feedback_decision["verified_feedback_accepted"] is True
    assert result.retest_assessment["evidenced_accuracy"] > result.initial_assessment["evidenced_accuracy"]
    assert result.output["review_handoff"]["suggested_delay_hours"] == 72.0
    assert result.output["delayed_accuracy"] == pytest.approx(1.0)
    history = result.target_state["attempt_history"]
    assert history[0]["response"] == initial.response
    assert history[0]["label"] == "initial_unaided_or_assisted_attempt"
    assert history[1]["response"] == retest.response
    assert result.biological["literal_neuron_simulation"] is False


def test_assisted_or_guessed_attempt_does_not_count_as_independent_recall():
    """Exclude hinted or guessed observations from independent retrieval calibration."""
    target = build_target()
    base = build_attempt(target)
    assisted = RetrievalAttempt(
        cue=base.cue, response=base.response, confidence=0.99,
        cue_type=base.cue_type, aid_level=2, recalled_components=target.components,
        recalled_relations=target.relations,
        component_status={item: "correct" for item in target.components},
        relation_status={item: "correct" for item in target.relations},
    )
    result = TestingEffectProcessor().process(TestingEffectRequest(target, assisted))
    assert result.initial_assessment["independent_retrieval"] is False
    assert result.target_state["successes"] == 0
    assert result.target_state["confidence_pairs"] == []


def test_unverified_feedback_does_not_strengthen_conceptual_edges_or_cortical_weights():
    """Prevent an unreliable correction from changing the verified conceptual network."""
    target = build_target(criterion_reliability=0.45)
    attempt = build_attempt(target, partial=True)
    untrusted = Feedback(
        correct_representation="An unverified proposed correction.",
        source="unverified_source",
        source_reliability=0.99,
        verified=True,
    )
    result = TestingEffectProcessor().process(TestingEffectRequest(
        target=target, initial_attempt=attempt, feedback=untrusted
    ))
    assert result.feedback_decision["verified_feedback_accepted"] is False
    assert result.operation_trace[18]["functional_output"]["updated_edges"] == {}
    assert result.biological["cortical_plasticity"]["update_count"] == 0


def test_biological_surrogates_are_bounded_and_update_only_with_verified_evidence():
    """Exercise recurrent activity, pattern completion, discrepancy monitoring, and gated plasticity."""
    population = NeuralPopulation("test", dimensions=16)
    first = population.update(np.ones(16) * 0.5)
    assert np.all((first >= 0.0) & (first <= 1.0))
    network = BiologicalTestingEffectNetwork(dimensions=16)
    encoded = network.encode_target("target", ["memory", "retrieval", "cue"])
    assert encoded["hippocampal_binding"]["bound"] is True
    retrieval = network.retrieve("target", "memory retrieval cue", 0.5, 0.8)
    assert retrieval["pattern_completion"]["target_match"] is True
    monitor = network.monitor_attempt(0.8, 0.4, {"a": 0.9, "b": 0.2})
    assert monitor["absolute_error_signal"] == pytest.approx(0.4)
    rejected = network.learn_from_verified_evidence(
        ["memory", "retrieval"], "incorrect answer", 0.1, 0.8, 0.8, 0.5, False
    )
    assert rejected["cortical_plasticity"]["plasticity_applied"] is False
    accepted = network.learn_from_verified_evidence(
        ["memory", "retrieval"], "correct answer", 0.9, 0.1, 0.8, 0.5, True
    )
    assert accepted["cortical_plasticity"]["plasticity_applied"] is True
    assert network.snapshot()["literal_neuron_simulation"] is False


def test_state_reuse_tracks_different_targets_independently():
    """Keep cue history and access estimates separated by target identity."""
    processor = TestingEffectProcessor()
    first_target = build_target()
    second_target = LearningTarget(
        target_id="second-target",
        prompt="Explain how a synapse supports communication.",
        components=("junction", "signal transmission"),
    )
    first = processor.process(TestingEffectRequest(
        first_target, build_attempt(first_target)
    ))
    second = processor.process(TestingEffectRequest(
        second_target,
        RetrievalAttempt(cue="Explain synapse communication", response="A junction transmits signals.",
                         confidence=0.6, recalled_components=("junction", "signal transmission"),
                         component_status={"junction": "correct", "signal transmission": "correct"}),
    ))
    assert set(second.final_state["targets"]) == {"testing-effect", "second-target"}
    assert second.final_state["process_count"] == 2
    assert first.target_state["target_id"] == "testing-effect"


def test_sqlite_stores_annotations_and_22_ordered_operation_events(tmp_path):
    """Register all package definitions and persist every ordered operation into the shared logger."""
    target = build_target()
    result = TestingEffectProcessor().process(TestingEffectRequest(
        target, build_attempt(target, partial=True), feedback=build_feedback()
    ))
    database_path = tmp_path / "point04.sqlite3"
    repository_root = Path(__file__).resolve().parents[1]
    report = record_result_to_sqlite(result, database_path, repository_root=repository_root)
    database = LoggerDatabase(database_path)
    annotations = database.search_annotations("ore_testing_effect", limit=1000)
    definitions = [row for row in annotations
                   if row["symbol_kind"] in {"class", "function", "async_function"}]
    assert definitions
    assert all(row["annotation"].strip() for row in definitions)
    assert all(row["source_docstring"] for row in definitions)
    with sqlite3.connect(database_path) as connection:
        run = connection.execute(
            "SELECT status, event_count FROM runs WHERE run_id=?",
            (report["run_id"],),
        ).fetchone()
        events = connection.execute(
            "SELECT sequence_no, event_kind, metadata_json FROM events "
            "WHERE run_id=? ORDER BY sequence_no",
            (report["run_id"],),
        ).fetchall()
    assert run == ("completed", 22)
    assert [row[0] for row in events] == list(range(1, 23))
    assert all(row[1] == "operation" and "P04-" in row[2] for row in events)


def test_request_rejects_naive_timestamps_and_negative_delay():
    """Reject invalid temporal metadata before it enters the retrieval-practice pipeline."""
    target = build_target()
    attempt = build_attempt(target)
    with pytest.raises(ValueError):
        TestingEffectRequest(target, attempt, current_time_utc="2026-01-01T00:00:00")
    with pytest.raises(ValueError):
        TestingEffectRequest(target, attempt, delay_after_hours=-1.0)
