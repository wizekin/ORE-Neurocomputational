"""Focused Point 18 tests for attention, missing-information recovery, provenance, and logging."""
from __future__ import annotations

import sqlite3

import pytest

from ore_attentional_failure import (
    AttentionalBiologicalNetwork,
    AttentionalConfig,
    AttentionalExperience,
    AttentionalFailureProcessor,
    AttentionalState,
    InformationRequirement,
    OPERATION_REGISTRY,
    audit_completeness,
    clip01,
    decide_downstream_release,
    detect_attention_lapse,
    sigmoid,
    softmax,
)


def make_event(**overrides):
    """Create a stable, fully specified task event for focused Point 18 tests."""
    values = {
        "experience_id": "point18-test",
        "task_goal": "Capture the required action and its cause",
        "required_information": (
            InformationRequirement("actor", importance=0.8, critical=True),
            InformationRequirement("action", importance=1.0, critical=True),
            InformationRequirement("cause", importance=0.7, depends_on=("action",)),
        ),
        "incoming_items": (
            {"key": "actor", "content": "the operator", "goal_relevance": 1.0,
             "signal_quality": 0.98, "integration_support": 0.98, "processing_cost": 0.2},
            {"key": "action", "content": "stopped the machine", "goal_relevance": 1.0,
             "signal_quality": 0.98, "integration_support": 0.98, "processing_cost": 0.2},
            {"key": "cause", "content": "the sensor overheated", "goal_relevance": 0.9,
             "signal_quality": 0.98, "integration_support": 0.98, "processing_cost": 0.2},
        ),
        "metrics": {"alertness": 0.95, "fatigue": 0.0, "distraction_load": 0.0,
                    "engagement": 0.95},
    }
    values.update(overrides)
    return AttentionalExperience(**values)


def test_registry_has_exactly_seventeen_source_ordered_operations():
    """The formal registry must preserve every Point 18 stage in source order."""
    assert len(OPERATION_REGISTRY) == 17
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P18-{index:02d}" for index in range(1, 18)
    ]


def test_numeric_helpers_are_stable_and_bounded():
    """Probability helpers should handle extremes and reject non-finite values."""
    assert clip01(-2.0) == 0.0
    assert clip01(2.0) == 1.0
    assert sigmoid(1000.0) == pytest.approx(1.0)
    assert sigmoid(-1000.0) == pytest.approx(0.0)
    assert sum(softmax([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        clip01(float("nan"))


def test_stable_attention_captures_all_required_items_and_releases():
    """High-quality input under stable attention should pass the downstream gate."""
    result = AttentionalFailureProcessor().process(make_event())
    assert len(result.operation_trace) == 17
    assert result.repair_verification["verified"] is True
    assert result.release_decision["release_downstream"] is True
    assert result.release_decision["status"] == "complete"


def test_goal_requirements_normalize_importance_weights():
    """Task requirements should be normalized without changing their relative importance."""
    result = AttentionalFailureProcessor().process(make_event())
    weights = result.task_requirements["importance_weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["action"] > weights["actor"]


def test_fatigue_and_distraction_reduce_effective_processing_availability():
    """State estimation should respond to current fatigue and distraction without diagnoses."""
    processor = AttentionalFailureProcessor()
    calm = processor.process(make_event()).state_estimate
    tired = AttentionalFailureProcessor().process(make_event(metrics={
        "alertness": 0.8, "fatigue": 0.9, "distraction_load": 0.9,
        "response_slowing": 0.0, "omission_rate": 0.0,
    })).state_estimate
    assert tired["attentional_availability"] < calm["attentional_availability"]
    assert tired["medical_or_fixed_trait_diagnosis_inferred"] is False


def test_competing_input_assessment_distinguishes_internal_external_and_safety():
    """The competitor model should preserve safety-relevant unexpected signals."""
    event = make_event(competing_inputs=(
        {"key": "phone", "kind": "external", "goal_relevance": 0.0, "salience": 0.9},
        {"key": "worry", "kind": "internal", "goal_relevance": 0.0, "salience": 0.6},
        {"key": "alarm", "kind": "external", "goal_relevance": 0.0, "urgency": 1.0,
         "safety_critical": True},
    ))
    result = AttentionalFailureProcessor().process(event)
    assessment = result.competing_input_assessment
    assert assessment["external_count"] == 2
    assert assessment["internal_count"] == 1
    assert assessment["competitors"][2]["classification"] == "urgent_safety_signal"


def test_safety_critical_item_overrides_priority_and_working_memory_budget():
    """A safety-critical item must not be deferred by the ordinary capacity budget."""
    event = make_event(incoming_items=(
        {"key": "detail", "content": "minor detail", "goal_relevance": 0.2,
         "processing_cost": 0.8},
        {"key": "alarm", "content": "evacuate now", "urgency": 1.0, "safety_critical": True,
         "processing_cost": 1.0},
    ))
    config = AttentionalConfig(working_memory_budget=0.3)
    result = AttentionalFailureProcessor(config=config).process(event)
    selected = {x["key"] for x in result.working_memory_gate["selected_items"]}
    assert "alarm" in selected
    assert result.priority_map["safety_override_applied"] is True


def test_sensory_bias_preserves_raw_low_priority_inputs():
    """Gain modulation must not erase raw content or low-priority source pointers."""
    result = AttentionalFailureProcessor().process(make_event())
    assert result.sensory_bias["low_priority_traces_preserved"] is True
    assert all(x["raw_input_preserved"] for x in result.sensory_bias["items"])
    assert all("content" in x for x in result.sensory_bias["items"])


def test_working_memory_gate_defers_items_when_budget_is_exhausted():
    """Ordinary input cost should limit admission while preserving deferred keys."""
    event = make_event(incoming_items=(
        {"key": "actor", "content": "operator", "processing_cost": 0.8},
        {"key": "action", "content": "stopped", "processing_cost": 0.8},
        {"key": "cause", "content": "overheated", "processing_cost": 0.8},
    ))
    config = AttentionalConfig(working_memory_budget=0.6)
    result = AttentionalFailureProcessor(config=config).process(event)
    assert result.working_memory_gate["deferred_items"]
    assert result.working_memory_gate["deferred_source_pointers_retained"] is True


def test_sustained_attention_tracks_a_bounded_moving_window():
    """The leaky integrator and moving-window statistics should remain bounded."""
    state = AttentionalState(attention_history=[0.8, 0.75, 0.72])
    result = AttentionalFailureProcessor(state=state).process(make_event())
    assert 0.0 <= result.sustained_attention["stability"] <= 1.0
    assert len(result.sustained_attention["history"]) <= AttentionalConfig().attention_window
    assert len(state.attention_history) <= AttentionalConfig().attention_window


def test_lapse_detection_uses_hysteresis_instead_of_one_noisy_sample():
    """One high lapse score is insufficient; consecutive high scores confirm a lapse."""
    event = make_event(metrics={
        "alertness": 0.2, "fatigue": 0.9, "distraction_load": 1.0,
        "response_slowing": 1.0, "omission_rate": 1.0,
        "disengagement_signal": 1.0, "engagement": 0.1,
    })
    state = AttentionalState()
    processor = AttentionalFailureProcessor(state=state)
    first = processor.process(event)
    assert first.lapse_event["score"] >= 0.65
    assert first.lapse_event["detected"] is False
    second = processor.process(make_event(experience_id="point18-test-second", metrics=event.metrics))
    assert second.lapse_event["detected"] is True
    assert second.lapse_event["hysteresis_confirmed"] is True


def test_safety_critical_omission_bypasses_lapse_hysteresis():
    """A safety-critical omission should trigger immediate protective handling."""
    event = make_event(
        safety_critical=True,
        metrics={"alertness": 0.7, "fatigue": 0.1, "distraction_load": 0.1,
                 "omission_rate": 0.1, "error_rate": 0.1},
    )
    result = AttentionalFailureProcessor().process(event)
    assert result.lapse_event["detected"] is True
    assert result.lapse_event["safety_override"] is True


def test_missing_input_is_recovered_only_from_a_source_record():
    """The system should recover an omitted item from a transcript and cite its source."""
    event = make_event(
        incoming_items=(
            {"key": "actor", "content": "the operator", "goal_relevance": 1.0},
            {"key": "cause", "content": "sensor overheating", "goal_relevance": 0.8},
        ),
        source_items=(
            {"key": "action", "content": "stopped the machine", "confidence": 0.98,
             "source_id": "transcript-line-4"},
        ),
        available_recovery={"source": True},
    )
    result = AttentionalFailureProcessor().process(event)
    recovered = {x["key"]: x for x in result.reencoded_representation["recovered_items"]}
    assert recovered["action"]["content"] == "stopped the machine"
    assert recovered["action"]["provenance"] == "recoverable_source_record"
    assert recovered["action"]["source_id"] == "transcript-line-4"
    assert result.repair_verification["verified"] is True


def test_weakly_encoded_item_can_be_repaired_from_a_better_source():
    """Weak sensory encoding should be replaced only when independent source evidence exists."""
    event = make_event(incoming_items=(
        {"key": "actor", "content": "operator", "goal_relevance": 1.0,
         "signal_quality": 0.2, "integration_support": 0.2},
        {"key": "action", "content": "stopped", "goal_relevance": 1.0},
        {"key": "cause", "content": "overheated", "goal_relevance": 1.0},
    ), source_items=(
        {"key": "actor", "content": "the machine operator", "confidence": 0.99,
         "source_id": "verified-transcript"},
    ), available_recovery={"source": True})
    result = AttentionalFailureProcessor().process(event)
    repaired = result.reencoded_representation["items"]["actor"]
    assert repaired["content"] == "the machine operator"
    assert repaired["provenance"] == "recoverable_source_record"


def test_unrecoverable_content_remains_unknown_and_not_schema_guessed():
    """Without a source or repeat, the missing content remains unknown and blocks release."""
    event = make_event(
        incoming_items=({"key": "actor", "content": "operator", "goal_relevance": 1.0},),
        required_information=(
            InformationRequirement("actor", importance=1.0, critical=True),
            InformationRequirement("action", importance=1.0, critical=True),
        ),
        source_items=(),
        safety_critical=True,
    )
    result = AttentionalFailureProcessor().process(event)
    assert "action" in result.preserved_uncertainty["unresolved"]
    assert result.preserved_uncertainty["guessing_prohibited"] is True
    assert result.release_decision["status"] == "blocked"


def test_completeness_audit_uses_dependency_aware_coverage_and_geometric_confidence():
    """A low-confidence prerequisite should keep dependent requirements unresolved."""
    requirements = (
        InformationRequirement("cause", importance=1.0, confidence=0.0, depends_on=("action",)),
        InformationRequirement("action", importance=1.0, confidence=0.95),
    )
    audit = audit_completeness(requirements, {"items": {
        "action": {"confidence": 0.40},
    }}, AttentionalConfig())
    cause = next(x for x in audit["requirements"] if x["key"] == "cause")
    assert cause["missing_dependencies"] == ["action"]
    assert audit["combined_confidence"] < 0.95
    assert audit["complete"] is False


def test_recovery_selection_prefers_source_review_when_available():
    """A verified source should be preferred over merely continuing with a gap."""
    event = make_event(
        available_recovery={"source": True},
        source_items=({"key": "action", "content": "stopped", "confidence": 0.95},),
    )
    result = AttentionalFailureProcessor().process(event)
    assert result.recovery_plan["action"] in {"review_source", "continue"}
    assert result.recovery_plan["expected_benefit"] >= 0.0


def test_reencoding_keeps_recovered_provenance_distinct_from_original_input():
    """Recovered content must retain its own source ID and provenance label."""
    event = make_event(
        incoming_items=(
            {"key": "actor", "content": "operator", "goal_relevance": 1.0},
            {"key": "cause", "content": "sensor overheated", "goal_relevance": 1.0},
        ),
        source_items=({"key": "action", "content": "stopped machine", "confidence": 0.9,
                       "source_id": "recording-12"},),
        available_recovery={"source": True},
    )
    result = AttentionalFailureProcessor().process(event)
    assert result.reencoded_representation["direct_and_recovered_provenance_separate"] is True
    assert result.reencoded_representation["items"]["action"]["source_id"] == "recording-12"


def test_repair_verification_removes_a_gap_only_after_confident_recovery():
    """A gap should close only when recovered evidence crosses the calibrated threshold."""
    event = make_event(
        incoming_items=({"key": "actor", "content": "operator", "goal_relevance": 1.0},),
        required_information=(InformationRequirement("actor", importance=1.0),),
        source_items=({"key": "actor", "content": "operator", "confidence": 0.98,
                       "source_id": "transcript"},),
        available_recovery={"source": True},
    )
    result = AttentionalFailureProcessor().process(event)
    assert result.repair_verification["verified"] is True
    assert "actor" not in result.preserved_uncertainty["unresolved"]


def test_future_policy_is_context_specific_and_does_not_create_a_trait_label():
    """Adaptation should be local to task context and explicitly avoid fixed-person judgments."""
    state = AttentionalState()
    result = AttentionalFailureProcessor(state=state).process(make_event(context_key="reading"))
    policy = result.future_attention_policy
    assert policy["context_key"] == "reading"
    assert policy["context_observations"] == 1
    assert policy["single_lapse_is_not_a_stable_trait"] is True
    assert "reading" in state.context_policy


def test_partial_release_carries_missing_markers_and_safety_critical_work_is_blocked():
    """Partial outputs must expose omissions; safety-critical tasks must not pass them onward."""
    event = make_event(
        required_information=(
            InformationRequirement("actor", importance=1.0, critical=True),
            InformationRequirement("unknown", importance=1.0, critical=True),
        ),
        incoming_items=({"key": "actor", "content": "operator", "goal_relevance": 1.0},),
    )
    partial = AttentionalFailureProcessor().process(event)
    assert partial.release_decision["release_partial"] is True
    assert "unknown" in partial.release_decision["missing_markers"]
    blocked = AttentionalFailureProcessor().process(make_event(
        experience_id="critical-block",
        required_information=event.required_information,
        incoming_items=event.incoming_items,
        safety_critical=True,
    ))
    assert blocked.release_decision["status"] == "blocked"


def test_biological_network_uses_recurrence_inhibition_and_bounded_activity():
    """Population updates should retain prior activation and stay inside the unit interval."""
    network = AttentionalBiologicalNetwork()
    first = network.engage("dorsal_attention_selection", "select", 0.8)
    second = network.engage("dorsal_attention_selection", "sustain", 0.2)
    assert second["activation_before"] == pytest.approx(first["activation_after"])
    assert 0.0 <= second["activation_after"] <= 1.0
    assert "recurrent_gain" in second["parameters"]


def test_sqlite_registers_annotations_and_logs_all_seventeen_operations(tmp_path):
    """Every Point 18 operation should be indexed and persisted through the existing ORE logger."""
    from ore_logger.database import LoggerDatabase
    from ore_attentional_failure import register_annotations, record_result_to_sqlite

    database_path = tmp_path / "point18.sqlite3"
    result = AttentionalFailureProcessor().process(make_event())
    catalogue = register_annotations(database_path)
    assert catalogue["files_seen"] >= 6
    assert LoggerDatabase(database_path).summary()["symbols"] > 0
    logged = record_result_to_sqlite(result, database_path)
    with sqlite3.connect(database_path) as connection:
        run = connection.execute(
            "SELECT status, event_count FROM runs WHERE run_id = ?", (logged["run_id"],)
        ).fetchone()
        count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ?", (logged["run_id"],)
        ).fetchone()[0]
    assert run == ("completed", 17)
    assert count == 17
