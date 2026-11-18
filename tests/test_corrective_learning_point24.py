"""Acceptance tests for Point 24 feedback and misconception correction."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ore_corrective_learning import (
    ClarificationOption, CorrectionRequest, CorrectiveConfig, CorrectiveState,
    FeedbackEvidence, KnowledgeHypothesis, Point24CorrectiveLearningProcessor,
    RetrievalAttempt, TransferProbe, record_result_to_sqlite, register_annotations,
)


def make_request(**overrides):
    """A false hot-outcome prediction corrected by reliable cold-outcome evidence."""
    old = KnowledgeHypothesis(
        "old-rule", "The control valve opens when the temperature is high.",
        prior=0.75, predictions={"hot": 0.80, "cold": 0.20},
        components={"valve_rule": 0.0, "unrelated_fact": 4.0}, source="inferred",
    )
    corrected = KnowledgeHypothesis(
        "corrected-rule", "The control valve closes when the temperature is high.",
        prior=0.25, predictions={"hot": 0.10, "cold": 0.90},
        components={"valve_rule": 1.0, "unrelated_fact": 4.0}, source="instruction",
    )
    evidence = FeedbackEvidence(
        "sensor-log", "An independently checked sensor log shows the valve closed at high temperature.",
        {"old-rule": 0.05, "corrected-rule": 0.95},
        source_name="calibrated sensor log", source_type="measurement",
        reliability=0.98, relevance=0.98, diagnosticity=0.98, directness="direct",
        independence_key="sensor-campaign-1",
    )
    clarification = ClarificationOption(
        "ask-condition", "clarify",
        "Ask whether the recorded temperature was above the specified threshold.",
        {"old-rule": {"yes": 0.90, "no": 0.10},
         "corrected-rule": {"yes": 0.10, "no": 0.90}},
        cost=0.05,
    )
    weak_option = ClarificationOption(
        "ask-vague", "clarify",
        "Ask a vague question that both explanations predict similarly.",
        {"old-rule": {"yes": 0.55, "no": 0.45},
         "corrected-rule": {"yes": 0.52, "no": 0.48}},
        cost=0.0,
    )
    values = {
        "request_id": "valve-correction-001",
        "hypotheses": (corrected, old),
        "current_hypothesis_id": "old-rule",
        "observed_outcome": "cold",
        "context": "A temperature-control experiment under specified operating conditions.",
        "error_hints": {"knowledge": 0.7, "inference": 0.3},
        "evidence": (evidence,),
        "clarification_options": (weak_option, clarification),
        "verified_hypothesis_id": "corrected-rule",
        "affected_components": ("valve_rule",),
        "explanation": {
            "nodes": ["temperature", "control signal", "valve state"],
            "edges": [
                {"from": "temperature", "relation": "triggers", "to": "control signal"},
                {"from": "control signal", "relation": "sets", "to": "valve state"},
            ],
            "conditions": ["within the tested operating range"],
            "example": "At the specified high-temperature threshold, the valve closes.",
            "non_example": "A low-temperature reading does not invoke this high-temperature rule.",
            "boundary_cases": ["outside the calibrated operating range"],
            "learner_explanation": "The controller closes the valve at the high-temperature threshold.",
        },
        "retrieval_attempt": RetrievalAttempt(
            "corrected-rule", confidence=0.90, delay_days=2.0,
            response_text="The valve closes when the threshold is exceeded.",
        ),
        "transfer_probes": (
            TransferProbe("new-temperature", "A different specimen at the same relative threshold.",
                          "corrected-rule", "corrected-rule"),
            TransferProbe("counterexample", "A low-temperature counterexample.",
                          "old-rule", "old-rule", is_counterexample=True),
        ),
        "confidence_estimate": 0.90,
        "task_stakes": "normal",
    }
    values.update(overrides)
    return CorrectionRequest(**values)


def test_pipeline_runs_twenty_ordered_operations_and_two_tracks():
    result = Point24CorrectiveLearningProcessor().process(make_request())
    assert len(result.operation_trace) == 20
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"P24-{index:02d}" for index in range(1, 21)
    ]
    assert all(row["status"] == "completed" for row in result.operation_trace)
    assert all("functional_output" in row and "biological_output" in row
               for row in result.operation_trace)
    assert result.operation_trace[-1]["functional_output"]["validation"]["operation_order_verified"]


def test_prediction_error_and_bayesian_feedback_move_belief_to_supported_candidate():
    result = Point24CorrectiveLearningProcessor().process(make_request())
    assert result.mismatch["mismatch_detected"] is True
    assert result.mismatch["observed_outcome"] == "cold"
    assert result.selected_hypothesis_id == "corrected-rule"
    assert result.posterior_probabilities["corrected-rule"] > 0.80
    assert result.correction["interpretation_changed"] is True
    assert result.stopping["status"] == "corrected"


def test_information_gain_prefers_question_that_separates_hypotheses():
    result = Point24CorrectiveLearningProcessor().process(make_request())
    selected = result.information_action["selected_option"]
    assert selected["action_id"] == "ask-condition"
    assert selected["expected_information_gain"] > 0.10


def test_error_classification_retains_multiple_possible_causes():
    result = Point24CorrectiveLearningProcessor().process(make_request())
    assert result.error_classification["knowledge"] == pytest.approx(0.7)
    assert result.error_classification["inference"] == pytest.approx(0.3)


def test_dependent_copies_of_feedback_are_not_counted_as_independent_evidence():
    request = make_request()
    original = request.evidence[0]
    duplicate = FeedbackEvidence(
        "copied-report", "A copied version of the same sensor campaign.",
        original.likelihoods, source_name="copied report", source_type="reported",
        reliability=1.0, relevance=1.0, diagnosticity=1.0, directness="direct",
        independence_key="sensor-campaign-1",
    )
    result = Point24CorrectiveLearningProcessor().process(make_request(evidence=(original, duplicate)))
    assert result.evidence_assessment["independent_group_count"] == 1
    assert result.evidence_assessment["dependent_sources_collapsed"] == 1


def test_component_revision_is_scoped_to_affected_feature_and_uses_current_belief():
    result = Point24CorrectiveLearningProcessor().process(make_request())
    updates = result.component_updates["component_updates"]
    assert updates["valve_rule"]["previous"] == 0.0
    assert updates["valve_rule"]["target"] == 1.0
    assert "unrelated_fact" not in updates
    assert "unrelated_fact" in result.component_updates["preserved_component_names"]


def test_corrected_explanation_keeps_relations_conditions_and_boundary_cases():
    result = Point24CorrectiveLearningProcessor().process(make_request())
    explanation = result.explanation_model
    assert explanation["graph"]["edge_count"] == 2
    assert explanation["conditions"] == ["within the tested operating range"]
    assert explanation["boundary_cases"] == ["outside the calibrated operating range"]
    assert explanation["relational_model_supplied"] is True


def test_retrieval_failure_triggers_specific_feedback_and_prevents_closure():
    result = Point24CorrectiveLearningProcessor().process(
        make_request(retrieval_attempt=RetrievalAttempt("old-rule", 0.95, 1.0))
    )
    assert result.retrieval["success"] is False
    assert result.retrieval_feedback["feedback_required"] is True
    assert result.stopping["status"] == "uncertain"
    assert result.stopping["earliest_failed_stage"] == "retrieve_corrected_understanding"


def test_transfer_checks_new_case_and_counterexample():
    result = Point24CorrectiveLearningProcessor().process(make_request())
    assert result.transfer["score"] == 1.0
    assert result.transfer["counterexamples_tested"] == 1
    assert result.transfer["passed"] is True


def test_confidence_calibration_uses_verified_outcome_not_prior_confidence_as_evidence():
    result = Point24CorrectiveLearningProcessor().process(make_request())
    assert result.calibration["verified_correctness"] == 1.0
    assert result.calibration["brier_score"] == pytest.approx(0.01)
    assert result.calibration["confidence_not_used_as_evidence"] is True


def test_consolidation_and_regression_state_are_persisted():
    state = CorrectiveState()
    result = Point24CorrectiveLearningProcessor(state=state).process(make_request())
    assert result.consolidation["consolidation_strength"] > 0
    assert result.consolidation["correction_accessibility"] > 0.25
    assert result.final_state["correction_history"][-1]["status"] == "corrected"
    assert result.regression["belief_regression_detected"] is False


def test_missing_observation_and_feedback_remain_explicitly_uncertain():
    hypotheses = (
        KnowledgeHypothesis("a", "Candidate A", 0.6, {"yes": 0.8, "no": 0.2}),
        KnowledgeHypothesis("b", "Candidate B", 0.4, {"yes": 0.2, "no": 0.8}),
    )
    result = Point24CorrectiveLearningProcessor().process(
        CorrectionRequest("insufficient-data", hypotheses, "a")
    )
    assert result.stopping["status"] == "uncertain"
    assert result.stopping["stop"] is False
    assert "no_observed_outcome_or_numeric_measurement" in result.unresolved_questions
    assert "no_corrective_evidence_supplied" in result.unresolved_questions


def test_verified_retrieval_target_is_not_confused_with_posterior_winner():
    result = Point24CorrectiveLearningProcessor().process(
        make_request(evidence=(), retrieval_attempt=RetrievalAttempt("corrected-rule", 0.8))
    )
    assert result.retrieval["target_hypothesis_id"] == "corrected-rule"
    assert result.retrieval["success"] is True
    assert result.stopping["status"] == "uncertain"


def test_high_stakes_raise_the_posterior_threshold():
    normal = Point24CorrectiveLearningProcessor().process(make_request())
    high = Point24CorrectiveLearningProcessor().process(make_request(task_stakes="high"))
    assert normal.stopping["status"] == "corrected"
    assert high.stopping["status"] == "uncertain"


def test_no_meaningful_mismatch_with_observation_does_not_trigger_correction():
    result = Point24CorrectiveLearningProcessor(
        CorrectiveConfig(mismatch_threshold=0.99)
    ).process(make_request())
    assert result.stopping["status"] == "no_correction_needed"


def test_biological_reference_can_be_disabled_without_removing_functional_trace():
    result = Point24CorrectiveLearningProcessor(
        CorrectiveConfig(simulate_biology=False)
    ).process(make_request())
    assert len(result.operation_trace) == 20
    assert result.biological["model"] == "disabled"
    assert all(row["biological_output"]["model"] == "disabled"
               for row in result.operation_trace)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="hypothesis prior"):
        KnowledgeHypothesis("bad", "Bad prior", prior=1.5)
    with pytest.raises(ValueError, match="source_type"):
        FeedbackEvidence("bad-source", "Invalid source", {"a": 0.5}, source_type="unknown")
    with pytest.raises(ValueError, match="current_hypothesis_id"):
        CorrectionRequest("bad-request", (KnowledgeHypothesis("a", "Candidate A"),), "missing")


def test_sqlite_persists_twenty_events_and_registers_package_annotations(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    database = tmp_path / "ore_logger.sqlite3"
    result = Point24CorrectiveLearningProcessor(
        CorrectiveConfig(simulate_biology=False)
    ).process(make_request())
    report = record_result_to_sqlite(result, database, repository_root=root)
    with sqlite3.connect(database) as connection:
        event_count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=?", (report["run_id"],)
        ).fetchone()[0]
        annotation_count = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE module_name LIKE 'ore_corrective_learning%'"
        ).fetchone()[0]
        status = connection.execute(
            "SELECT status FROM runs WHERE run_id=?", (report["run_id"],)
        ).fetchone()[0]
    assert event_count == 20
    assert annotation_count >= 20
    assert status == "completed"
    catalogue = register_annotations(tmp_path / "catalog.sqlite3", repository_root=root)
    assert catalogue["scope"] == "ore_corrective_learning"
    assert catalogue["annotation_table"] == "annotations"
