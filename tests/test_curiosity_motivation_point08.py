"""Acceptance tests for the 24-step Point 08 curiosity and motivated-learning implementation."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ore_curiosity_motivation import (
    DistributedMotivationNetwork, LearningAction, LearningOutcome, LearningRequest,
    LearningState, LearningUnit, Point08MotivatedLearningProcessor,
    RetrievalCheck, record_result_to_sqlite, register_annotations,
)
from ore_curiosity_motivation.functional import FUNCTIONAL_STEPS, shannon_entropy, softmax


def make_request(**overrides):
    """Build one target learning unit and one context-linked incidental unit."""
    target = LearningUnit(
        "moon-phases", "Explain why the Moon's illuminated appearance changes.",
        relevance=1.0, novelty=0.8, uncertainty=0.85, utility=0.9,
        intrinsic_interest=0.95, transfer_value=0.8, personal_relevance=0.9,
        conceptual_distance=0.35, missing_prerequisites=0.1, available_support=0.9,
        expected_features=("Sun", "Moon", "Earth", "viewing geometry"),
        target=True, elaboration_hint=0.9, reliability=0.95, future_utility=0.9,
    )
    incidental = LearningUnit(
        "lunar-calendar", "Some lunar calendars use phase observations.",
        relevance=0.3, novelty=0.5, uncertainty=0.4, utility=0.5,
        intrinsic_interest=0.4, transfer_value=0.4, personal_relevance=0.3,
        conceptual_distance=0.3, available_support=0.8, target=False,
        temporal_context_link=0.9, semantic_context_link=0.8,
        elaboration_hint=0.3, reliability=0.9, future_utility=0.4,
    )
    actions = (
        LearningAction(
            "read-diagram", "moon-phases", "inspect_diagram",
            expected_learning_gain=0.7, expected_information_gain=0.8,
            expected_interest_gain=0.6, effort_cost=0.15, time_cost=0.1,
            stress_cost=0.05, source_reliability=0.95, learner_choice=True,
            question="How does viewing geometry change the visible illuminated fraction?",
            evidence=({"source": "lesson-diagram", "claim": "Sun illuminates half of the Moon."},),
        ),
        LearningAction(
            "predict-phase", "moon-phases", "make_prediction",
            expected_learning_gain=0.65, expected_information_gain=0.75,
            expected_interest_gain=0.5, effort_cost=0.25, time_cost=0.2,
            stress_cost=0.05, source_reliability=0.9,
        ),
    )
    values = {
        "request_id": "point08-moon-phases",
        "goal": "Explain and predict lunar phases from the Sun-Earth-Moon geometry.",
        "units": (target, incidental),
        "actions": actions,
        "outcomes": (LearningOutcome(
            "read-diagram", observed_learning=0.85, observed_interest_gain=0.7,
            completed=True, evidence_quality=0.95, notes="Learner explained the geometry.",
        ),),
        "retrieval_checks": (RetrievalCheck(
            "delayed-check-1", factual_accuracy=0.9, relational_accuracy=0.85,
            transfer_accuracy=0.8, confidence=0.85, delayed_days=1.0,
            target_unit_id="moon-phases",
        ),),
        "intrinsic_motivation": 0.9, "extrinsic_motivation": 0.3, "personal_value": 0.8,
        "stress": 0.1, "fatigue": 0.1, "boredom": 0.05, "frustration": 0.0,
        "distraction": 0.05, "attention_available": 0.9, "time_available": 0.8,
        "emotional_valence": 0.7, "emotional_arousal": 0.4,
        "perceived_control": 0.9, "emotional_relevance": 0.7,
        "observed_error_rate": 0.15, "task_switch_rate": 0.0,
        "observed_milestones": 1.0, "planned_milestones": 2.0,
        "target_success_rate": 0.75, "sleep_quality": 0.8,
        "time_since_learning_days": 1.0,
        "hypothesis_priors": {"geometry": 0.7, "distance_shadow": 0.3},
        "expected_learning_baseline": 0.5,
    }
    values.update(overrides)
    return LearningRequest(**values)


def test_all_twenty_four_steps_execute_once_in_source_order():
    result = Point08MotivatedLearningProcessor().process(make_request())
    assert len(result.operation_trace) == 24
    assert [item["operation_id"] for item in result.operation_trace] == [
        f"P08-{index:02d}" for index in range(1, 25)
    ]
    assert all(item["status"] == "completed" for item in result.operation_trace)
    assert len(FUNCTIONAL_STEPS) == 24
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["biological_step_count"] == 24


def test_curiosity_value_learnability_and_motivation_are_separate():
    result = Point08MotivatedLearningProcessor(simulate_biology=False).process(make_request())
    outputs = result.functional_outputs
    assert outputs["step_02_detect_novelty_and_information_gaps"]["mean_information_gap"] > 0
    assert outputs["step_03_estimate_information_value"]["mean_value"] > 0.5
    assert outputs["step_04_appraise_learnability_and_safety"]["exploration_feasibility"] > 0.5
    assert 0 <= outputs["step_05_estimate_curiosity"]["curiosity"] <= 1
    profile = outputs["step_06_separate_motivation_sources"]["motivation_profile"]
    assert set(profile) == {"intrinsic", "extrinsic", "personal", "mastery", "avoidance"}
    assert outputs["step_06_separate_motivation_sources"]["value_channels_remain_separate"]


def test_attention_is_normalized_and_action_selection_respects_available_resources():
    result = Point08MotivatedLearningProcessor(simulate_biology=False).process(make_request())
    allocation = result.functional_outputs["step_10_allocate_attention"]
    assert allocation["attention_weights_sum"] == pytest.approx(1.0)
    assert sum(item["attention_amount"] for item in allocation["attention_allocations"]) == pytest.approx(0.9)
    selected = result.functional_outputs["step_20_preserve_autonomy_and_agency"]
    assert selected["selected_path"] == "read-diagram"
    assert selected["learner_choice_honored"] is True
    assert selected["agency_value"] > 0


def test_exploration_records_evidence_and_outcome_prediction_error():
    result = Point08MotivatedLearningProcessor().process(make_request())
    exploration = result.functional_outputs["step_11_explore_and_sample_evidence"]
    update = result.functional_outputs["step_12_update_expectations"]
    assert len(exploration["sampled_evidence"]) == 1
    assert exploration["sampled_action_ids"] == ["read-diagram"]
    assert update["prediction_errors"][0]["prediction_error"] == pytest.approx(0.15)
    assert update["reward_prediction_error_available"] is True
    assert update["semantic_mismatch_is_not_reward_prediction_error"] is True
    assert result.biological_outputs[11]["reward_prediction_error"] == pytest.approx(0.15)


def test_target_and_incidental_encoding_are_distinguished():
    result = Point08MotivatedLearningProcessor().process(make_request())
    outputs = result.functional_outputs
    target = outputs["step_13_encode_target_information"]["target_memory_candidates"]
    incidental = outputs["step_14_encode_incidental_information"]["incidental_memory_candidates"]
    assert [item["unit_id"] for item in target] == ["moon-phases"]
    assert [item["unit_id"] for item in incidental] == ["lunar-calendar"]
    assert outputs["step_14_encode_incidental_information"]["target_and_incidental_kept_separate"]


def test_high_stress_or_fatigue_recommends_regulation_not_more_effort():
    request = make_request(stress=0.95, fatigue=0.9, emotional_arousal=0.95)
    result = Point08MotivatedLearningProcessor(simulate_biology=False).process(request)
    outputs = result.functional_outputs
    assert outputs["step_04_appraise_learnability_and_safety"]["learnability"][0]["decision"] == "pause_or_regulate"
    assert outputs["step_15_appraise_emotion"]["arousal_regulation"] == "reduce_pressure_or_pause"
    assert outputs["step_18_adapt_challenge_and_explanation"]["adaptation_strategy"] == "pause_or_shorten_session"


def test_delayed_learning_requires_observed_retrieval_measurements():
    result = Point08MotivatedLearningProcessor(simulate_biology=False).process(make_request())
    durable = result.functional_outputs["step_23_test_durable_understanding"]
    assert durable["durable_learning"] == pytest.approx((0.9+0.85+0.8)/3)
    assert durable["all_checks_unmeasured"] is False
    missing = Point08MotivatedLearningProcessor(simulate_biology=False).process(
        make_request(retrieval_checks=())
    )
    missing_output = missing.functional_outputs["step_23_test_durable_understanding"]
    assert missing_output["durable_learning"] is None
    assert missing_output["all_checks_unmeasured"] is True


def test_emotional_arousal_is_non_monotonic_and_affect_dimensions_remain_separate():
    processor = Point08MotivatedLearningProcessor(simulate_biology=False)
    moderate = processor.process(make_request(emotional_arousal=0.45))
    high = processor.process(make_request(emotional_arousal=1.0))
    assert moderate.functional_outputs["step_15_appraise_emotion"]["arousal_performance_factor"] > high.functional_outputs["step_15_appraise_emotion"]["arousal_performance_factor"]
    affect = moderate.functional_outputs["step_15_appraise_emotion"]["affect_vector"]
    assert set(affect) == {"valence", "arousal", "perceived_control", "relevance"}


def test_biological_network_is_distributed_and_does_not_infer_rpe_from_novelty():
    network = DistributedMotivationNetwork()
    result = network.advance({"anterior_cingulate_monitoring": 1.0})
    assert len(result["population_rates"]) == 10
    assert result["reward_prediction_error"] is None
    assert result["semantic_or_novelty_mismatch_is_not_reward_error"] is True
    reward_result = network.advance({"ventral_striatal_value": 0.2}, reward_prediction_error=0.35)
    assert reward_result["reward_prediction_error"] == pytest.approx(0.35)
    assert 0 <= reward_result["population_rates"]["midbrain_dopamine_proxy"] <= 1


def test_state_updates_and_future_policy_are_persisted():
    state = LearningState()
    result = Point08MotivatedLearningProcessor(simulate_biology=False).process(make_request(), state=state)
    assert state.episode_count == 1
    assert state.action_values["read-diagram"] == pytest.approx(0.73)
    assert state.history[-1]["request_id"] == "point08-moon-phases"
    assert result.functional_outputs["step_24_update_future_motivation"]["future_policy"]


def test_no_unobserved_outcomes_or_correctness_labels_are_invented():
    request = make_request(outcomes=(), retrieval_checks=())
    result = Point08MotivatedLearningProcessor().process(request)
    update = result.functional_outputs["step_12_update_expectations"]
    durable = result.functional_outputs["step_23_test_durable_understanding"]
    assert update["reward_prediction_error_available"] is False
    assert update["prediction_errors"] == []
    assert durable["durable_learning"] is None
    assert result.biological_outputs[11]["reward_prediction_error"] is None


def test_math_helpers_are_stable_for_empty_and_extreme_inputs():
    assert softmax(()) == ()
    assert sum(softmax((1000.0, 0.0, -1000.0))) == pytest.approx(1.0)
    assert shannon_entropy({"a": 0.5, "b": 0.5}) == pytest.approx(0.693147, rel=1e-5)
    assert shannon_entropy({"a": 0.0, "b": 0.0}) == 0.0


def test_sqlite_catalogue_annotates_all_classes_and_functions_and_links_each_event(tmp_path):
    database_path = tmp_path / "ore_logger.sqlite3"
    root = Path(__file__).resolve().parents[1]
    catalogue = register_annotations(database_path, root)
    assert catalogue["scope"] == "ore_curiosity_motivation"
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT qualified_name, symbol_kind, annotation FROM annotations "
            "WHERE source_path LIKE 'ore_curiosity_motivation/%'"
        ).fetchall()
    finally:
        connection.close()
    symbols = {(name, kind) for name, kind, annotation in rows if annotation}
    assert ("Point08MotivatedLearningProcessor", "class") in symbols
    assert ("Point08MotivatedLearningProcessor.process", "function") in symbols
    assert all((function.__name__, "function") in symbols for function in FUNCTIONAL_STEPS)
    assert all((name, "class") in symbols for name in (
        "LearningUnit", "LearningAction", "LearningOutcome", "RetrievalCheck",
        "LearningRequest", "LearningState", "MotivationConfig", "LearningResult",
        "NeuralPopulation", "DistributedMotivationNetwork", "OperationSpec",
    ))
    result = Point08MotivatedLearningProcessor().process(make_request())
    persisted = record_result_to_sqlite(result, database_path, root)
    assert persisted["events_written"] == 24
    assert persisted["class_annotation_linked"] is True
    connection = sqlite3.connect(database_path)
    try:
        run = connection.execute("SELECT status,event_count FROM runs ORDER BY started_at_utc DESC LIMIT 1").fetchone()
        events = connection.execute(
            "SELECT COUNT(*), COUNT(symbol_id), COUNT(class_symbol_id) FROM events"
        ).fetchone()
    finally:
        connection.close()
    assert run == ("completed", 24)
    assert events == (24, 24, 24)


def test_request_validation_rejects_missing_units_and_invalid_state():
    with pytest.raises(ValueError):
        LearningRequest(request_id="bad", goal="No input", units=())
    with pytest.raises(ValueError):
        make_request(stress=1.5)
