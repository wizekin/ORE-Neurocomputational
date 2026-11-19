"""Focused acceptance tests for the eighteen-step Point 07 feedback pipeline."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ore_feedback_seeking.functional import FUNCTIONAL_STEPS
from ore_feedback_seeking import (
    DistributedFeedbackNetwork, FeedbackConfig, FeedbackEvidence,
    FeedbackRequest, FeedbackSource, FeedbackState, PerformanceContext,
    Point07FeedbackProcessor, ReasoningNode, TargetCriterion, TransferProbe,
    brier_score, entropy, normalize_distribution, record_result_to_sqlite,
    register_annotations,
)


def make_request(**overrides):
    """Build a correlation-versus-causation misconception with supported correction data."""
    source = FeedbackSource(
        "study-review", "Methods-aware scientific reviewer", domain="research methods",
        reliability=0.97, relevance=1.0, specificity=0.95, evidence_quality=0.98,
        outcome_likelihoods={
            "observational": {"correlation_only": 0.95, "causal": 0.05},
            "randomized": {"correlation_only": 0.10, "causal": 0.90},
        },
    )
    evidence = FeedbackEvidence(
        "method-check", "study-review", "The cited report is observational and cannot alone prove causation.",
        correction="The variables are associated; causality needs additional evidence.",
        hypothesis_likelihoods={"correlation_only": 0.95, "causal": 0.05},
        independence_key="original-method-review", relevance=1.0, actionability=0.95,
        evidence_quality=0.98, affected_components=("causal_rule",),
        corrected_component_values={"causal_rule": 1.0},
    )
    values = {
        "request_id": "point07-correlation-causation",
        "learning_target": "Distinguish observational association from causal evidence.",
        "candidate_response": "Exercise always causes improved mood.",
        "domain": "research methods",
        "criteria": (
            TargetCriterion("association", "State what the measured association shows", 0.4, 1.0),
            TargetCriterion("causality", "Avoid claiming causation from correlation alone", 0.6, 0.0),
        ),
        "reasoning_chain": (
            ReasoningNode("read-study", "Read the observed association correctly", 0.95),
            ReasoningNode("infer-cause", "Infer that the association proves causation", 0.25, ("read-study",)),
        ),
        "confidence": 0.7, "ambiguity": 0.1, "importance": 0.8, "stakes": 0.4,
        "feedback_sources": (source,), "feedback_evidence": (evidence,),
        "hypothesis_priors": {"causal": 0.65, "correlation_only": 0.35},
        "error_hints": {"inference_error": 0.9},
        "knowledge_components": {"causal_rule": 0.2, "unrelated_semantic_fact": 0.8},
        "affected_components": ("causal_rule",),
        "corrected_components": {"causal_rule": 1.0},
        "corrected_response": "The variables are associated; causality needs additional evidence.",
        "supporting_concepts": ("confounding", "reverse_causation"),
        "conditions": ("The original study is observational.",),
        "counterexample": "Ice-cream sales and drowning can co-vary because of temperature.",
        "positive_example": "A well-controlled randomized intervention can strengthen causal inference.",
        "boundary_examples": ("An observational result may still be useful for prediction.",),
        "retrieval_response": "Association is not proof of causation; a third variable may explain both.",
        "retrieval_correctness": 1.0,
        "transfer_probes": (
            TransferProbe("new-study", "A new observational study", "association_only",
                          correctness=1.0, structural_similarity=0.9, surface_similarity=0.2),
            TransferProbe("counterexample", "Two outcomes share a common cause", "not_causal",
                          is_counterexample=True, correctness=1.0, structural_similarity=0.95,
                          surface_similarity=0.15),
        ),
        "delayed_check_days": 2.0,
        "performance": PerformanceContext(fatigue=0.1, distraction=0.0,
                                           working_memory_load=0.2, time_pressure=0.0),
    }
    values.update(overrides)
    return FeedbackRequest(**values)


def test_all_eighteen_operations_run_in_source_order_with_two_tracks():
    result = Point07FeedbackProcessor().process(make_request())
    assert len(result.operation_trace) == 18
    assert [item["operation_id"] for item in result.operation_trace] == [f"P07-{i:02d}" for i in range(1, 19)]
    assert all(item["status"] == "completed" for item in result.operation_trace)
    assert all("functional_output" in item and "biological_output" in item for item in result.operation_trace)
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["biological_step_count"] == 18


def test_target_coverage_uncertainty_and_first_error_localization():
    result = Point07FeedbackProcessor().process(make_request())
    outputs = result.functional_outputs
    assert outputs["step_01_define_target"]["completeness"] == pytest.approx(0.4)
    assert outputs["step_03_monitor_comprehension"]["check_required"] is True
    assert outputs["step_08_classify_error"]["most_likely_error"] == "inference_error"
    assert outputs["step_08_classify_error"]["first_divergence"]["node_id"] == "infer-cause"


def test_bayesian_source_selection_and_dependent_evidence_collapse():
    request = make_request()
    original = request.feedback_evidence[0]
    duplicate = FeedbackEvidence(
        "copied-method-check", original.source_id, "A copied copy of the same method review.",
        correction=original.correction,
        hypothesis_likelihoods=original.hypothesis_likelihoods,
        independence_key=original.independence_key, evidence_quality=1.0,
    )
    result = Point07FeedbackProcessor().process(make_request(feedback_evidence=(original, duplicate)))
    route = result.functional_outputs["step_05_select_feedback_route"]
    trust = result.functional_outputs["step_07_verify_feedback"]
    assert route["selected_source"] == "study-review"
    assert route["source_scores"][0]["domain_alignment"] == pytest.approx(1.0)
    assert trust["dependent_evidence_groups_collapsed"] == 1
    assert trust["feedback_trustworthy"] is True
    assert trust["posterior"]["correlation_only"] > trust["prior"]["correlation_only"]


def test_local_correction_preserves_unrelated_knowledge_and_links_provenance():
    result = Point07FeedbackProcessor().process(make_request())
    correction = result.functional_outputs["step_10_apply_targeted_correction"]
    assert correction["component_updates"]["causal_rule"]["updated"] == pytest.approx(0.6)
    assert "unrelated_semantic_fact" in correction["preserved_component_names"]
    graph = result.functional_outputs["step_11_integrate_knowledge"]
    assert graph["provenance"]
    assert any(edge["relation"] == "supports" for edge in graph["edges"])


def test_retrieval_transfer_and_confidence_calibration_use_observed_labels():
    result = Point07FeedbackProcessor().process(make_request())
    outputs = result.functional_outputs
    assert outputs["step_13_test_retrieval"]["retrieval_success"] is True
    assert outputs["step_14_calibrate_confidence"]["current_brier_score"] == pytest.approx(0.09)
    assert outputs["step_16_test_transfer"]["transfer_score"] == pytest.approx(1.0)
    assert outputs["step_16_test_transfer"]["counterexamples_tested"] == 1


def test_absent_correctness_label_is_not_fabricated():
    result = Point07FeedbackProcessor().process(make_request(retrieval_correctness=None, retrieval_response=None, transfer_probes=()))
    outputs = result.functional_outputs
    assert outputs["step_13_test_retrieval"]["retrieval_success"] is None
    assert outputs["step_14_calibrate_confidence"]["calibration_updated"] is False
    assert outputs["step_14_calibrate_confidence"]["current_brier_score"] is None
    assert outputs["step_16_test_transfer"]["transfer_score"] is None


def test_feedback_load_is_chunked_and_delayed_review_is_scheduled():
    result = Point07FeedbackProcessor(FeedbackConfig(feedback_load_threshold=0.1)).process(make_request())
    outputs = result.functional_outputs
    assert outputs["step_15_control_feedback_load"]["chunking_required"] is True
    assert outputs["step_15_control_feedback_load"]["chunk_count"] >= 1
    assert outputs["step_17_schedule_delayed_check"]["review_queue_size"] == 1
    assert outputs["step_17_schedule_delayed_check"]["accessibility"] < 1.0


def test_biology_keeps_reward_error_separate_from_conceptual_mismatch():
    network = DistributedFeedbackNetwork()
    before = network.advance({"medial_frontal_error_monitoring": 1.0})
    after = network.advance({"medial_frontal_error_monitoring": 0.1}, reward_prediction_error=0.4)
    assert before["semantic_error_is_reward_error"] is False
    assert after["reward_prediction_error"] == pytest.approx(0.4)
    assert "striatal_value_learning" in after["population_rates"]
    assert len(after["population_rates"]) == 10


def test_math_helpers_are_stable_and_normalized():
    distribution = normalize_distribution({"a": 2.0, "b": 2.0})
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert entropy(distribution) == pytest.approx(0.693147, rel=1e-5)
    assert brier_score(0.9, 1.0) == pytest.approx(0.01)


def test_sqlite_annotations_cover_every_package_definition_and_trace_events(tmp_path):
    database_path = tmp_path / "ore_logger.sqlite3"
    root = Path(__file__).resolve().parents[1]
    report = register_annotations(database_path, root)
    assert report["scope"] == "ore_feedback_seeking"
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT qualified_name, symbol_kind, annotation FROM annotations WHERE source_path LIKE 'ore_feedback_seeking/%'"
        ).fetchall()
    finally:
        connection.close()
    symbols = {(name, kind) for name, kind, annotation in rows if annotation}
    assert ("Point07FeedbackProcessor", "class") in symbols
    assert ("Point07FeedbackProcessor.process", "function") in symbols
    assert ("step_01_define_target", "function") in symbols
    assert all((function.__name__, "function") in symbols for function in FUNCTIONAL_STEPS)
    assert all((name, "class") in symbols for name in (
        "FeedbackRequest", "FeedbackState", "FeedbackConfig", "FeedbackResult",
        "DistributedFeedbackNetwork", "NeuralPopulation", "OperationSpec",
    ))
    result = Point07FeedbackProcessor().process(make_request())
    persisted = record_result_to_sqlite(result, database_path, root)
    assert persisted["events_written"] == 18
    assert persisted["class_annotation_linked"] is True
    connection = sqlite3.connect(database_path)
    try:
        runs = connection.execute("SELECT status,event_count FROM runs").fetchall()
        event_rows = connection.execute("SELECT COUNT(*),COUNT(symbol_id),COUNT(class_symbol_id) FROM events").fetchone()
    finally:
        connection.close()
    assert runs[-1] == ("completed", 18)
    assert event_rows == (18, 18, 18)


def test_reusable_state_is_updated_in_place_across_feedback_episodes():
    state = FeedbackState()
    Point07FeedbackProcessor(simulate_biology=False).process(make_request(), state=state)
    assert state.episode_count == 1
    assert state.error_frequencies["inference_error"] > 0.0
    assert state.review_queue


def test_untrusted_feedback_does_not_overwrite_knowledge_components():
    source = FeedbackSource("weak", "Unverified commenter", reliability=0.1,
                            relevance=0.1, specificity=0.1, evidence_quality=0.1)
    weak_evidence = FeedbackEvidence(
        "weak-claim", "weak", "An unsupported claim.", correction="Replace the model with this.",
        relevance=0.1, actionability=0.1, evidence_quality=0.1,
        affected_components=("causal_rule",), corrected_component_values={"causal_rule": 1.0},
    )
    result = Point07FeedbackProcessor(simulate_biology=False).process(
        make_request(feedback_sources=(source,), feedback_evidence=(weak_evidence,))
    )
    correction = result.functional_outputs["step_10_apply_targeted_correction"]
    assert correction["correction_applied"] is False
    assert correction["correction_provisional"] is True
    assert correction["revised_knowledge"]["causal_rule"] == pytest.approx(0.2)


def test_request_validation_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        make_request(confidence=1.2)
