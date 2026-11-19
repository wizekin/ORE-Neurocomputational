"""Contract, scheduler, biological, persistence, and SQLite tests for Point 03."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from ore_spaced_repetition import (
    BiologicalSpacedRepetitionNetwork,
    LearningItem,
    OfflineConsolidationObservation,
    ReviewObservation,
    SpacedRepetitionConfig,
    SpacedRepetitionProcessor,
    SpacedRepetitionRequest,
    SpacedRepetitionState,
    OPERATION_REGISTRY,
    assess_calibration,
    assess_maintenance,
    choose_next_interval,
    estimate_accessibility,
    estimate_encoding_quality,
    initialize_stability,
)


START = "2026-01-01T00:00:00Z"


def build_item(**overrides) -> LearningItem:
    """Create one structured science-learning item with measurable encoding indicators."""
    values = {
        "item_id": "synapse-definition",
        "content": "A synapse is a specialized junction through which one neuron communicates with another cell.",
        "components": ("specialized junction", "communication", "neuron", "another cell"),
        "relations": ("neuron communicates across junction with another cell",),
        "context": {"topic": "neuroscience", "lesson": "cell communication"},
        "difficulty": 0.45,
        "target_retention_hours": 24.0 * 30.0,
        "encoding_measurements": {
            "attention": 0.9, "comprehension": 0.85,
            "relational_organization": 0.8, "meaningful_practice": 0.75,
        },
        "source": "biology-lesson",
        "created_at_utc": START,
    }
    values.update(overrides)
    return LearningItem(**values)


def build_review(occurred_at: str, **overrides) -> ReviewObservation:
    """Create an independent retrieval observation with complete, accurate relations."""
    values = {
        "occurred_at_utc": occurred_at,
        "outcome": "success",
        "accuracy": 0.95, "completeness": 0.9,
        "relation_accuracy": 0.9, "fluency": 0.75,
        "effort": 0.65, "response_mode": "free_recall",
        "hint_level": 0.0, "feedback_provided": False,
        "learner_response": "A synapse is a specialized junction used by one neuron to communicate with another cell.",
    }
    values.update(overrides)
    return ReviewObservation(**values)


def test_registry_contains_twelve_source_ordered_operations():
    """Keep the registry identifiers aligned with the 12 source algorithm steps."""
    assert len(OPERATION_REGISTRY) == 12
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P03-{index:02d}" for index in range(1, 13)
    ]
    assert all(item.mathematical_form and item.biological_reference for item in OPERATION_REGISTRY)


def test_encoding_quality_and_stability_are_bounded():
    """Estimate initial encoding quality while explicitly reporting missing evidence."""
    complete = estimate_encoding_quality({
        "attention": 1.0, "comprehension": 1.0,
        "relational_organization": 1.0, "meaningful_practice": 1.0,
    })
    incomplete = estimate_encoding_quality({"attention": 0.8})
    assert complete["encoding_quality"] == pytest.approx(1.0)
    assert complete["encoding_uncertainty"] < incomplete["encoding_uncertainty"]
    config = SpacedRepetitionConfig()
    stability = initialize_stability(build_item(), complete["encoding_quality"], config)
    assert config.minimum_stability_hours <= stability <= config.maximum_stability_hours


def test_accessibility_decreases_monotonically_with_elapsed_time_for_both_models():
    """Check monotonic time dependence without declaring one forgetting curve universal."""
    for model in ("exponential", "power_law"):
        config = SpacedRepetitionConfig(forgetting_model=model)
        early = estimate_accessibility(48.0, 12.0, 1, 0.3, [0.8], config)
        late = estimate_accessibility(48.0, 96.0, 1, 0.3, [0.8], config)
        assert early["predicted_recall_probability"] > late["predicted_recall_probability"]
        assert 0.0 <= late["memory_uncertainty"] <= 1.0


def test_accessibility_is_separate_from_recall_probability_and_interval_hits_target():
    """Invert the logistic retrieval mapping rather than equating availability with recall."""
    from ore_spaced_repetition import choose_next_interval

    config = SpacedRepetitionConfig()
    immediate = estimate_accessibility(24.0, 0.0, 0, 0.2, [], config)
    assert immediate["memory_accessibility"] == pytest.approx(1.0)
    assert immediate["predicted_recall_probability"] < immediate["memory_accessibility"]
    item = build_item()
    review = build_review(START)
    schedule = choose_next_interval(24.0, item, review, START, config)
    at_target = estimate_accessibility(24.0, schedule["interval_hours"], 0, 0.2, [], config)
    assert at_target["predicted_recall_probability"] == pytest.approx(
        config.target_recall_probability, abs=1e-6
    )


def test_first_registration_creates_nonzero_review_delay_and_twelve_events():
    """Register an item once and create a future due timestamp without a fake review."""
    item = build_item()
    request = SpacedRepetitionRequest("learner-03", item, START)
    result = SpacedRepetitionProcessor("learner-03").process(request)
    assert len(result.operation_trace) == 12
    assert result.output["new_item_registered"] is True
    assert result.review_event is None
    assert result.memory_record["next_interval_hours"] > 0
    assert result.memory_record["next_due_at_utc"] > START
    assert [row["operation_id"] for row in result.operation_trace] == [
        spec.operation_id for spec in OPERATION_REGISTRY
    ]


def test_successful_review_updates_state_and_uses_stability_for_next_interval():
    """A successful delayed free-recall attempt updates state and extends the future interval."""
    item = build_item()
    state = SpacedRepetitionState("learner-03")
    processor = SpacedRepetitionProcessor("learner-03", state=state)
    first = processor.process(SpacedRepetitionRequest("learner-03", item, START))
    due = first.memory_record["next_due_at_utc"]
    review = build_review(due)
    second = processor.process(SpacedRepetitionRequest(
        "learner-03", item, due, review_observation=review
    ))
    assert second.review_event["independent_retrieval"] is True
    assert second.memory_record["stability_hours"] > first.memory_record["stability_hours"]
    assert second.memory_record["last_learning_at_utc"] == due
    assert second.memory_record["review_history"]
    assert second.output["review_attempt_recorded"] is True


def test_recognition_does_not_count_as_independent_recall_or_reset_learning_clock():
    """Prevent recognition-only success from being counted as mastery or a new retrieval success."""
    item = build_item()
    state = SpacedRepetitionState("learner-03")
    processor = SpacedRepetitionProcessor("learner-03", state=state)
    first = processor.process(SpacedRepetitionRequest("learner-03", item, START))
    due = first.memory_record["next_due_at_utc"]
    recognition = build_review(due, outcome="success", response_mode="recognition")
    second = processor.process(SpacedRepetitionRequest(
        "learner-03", item, due, review_observation=recognition
    ))
    assert second.review_event["outcome"] == "recognition_only"
    assert second.review_event["independent_retrieval"] is False
    assert second.memory_record["last_learning_at_utc"] == START
    assert second.calibration["sample_count"] == 0
    assert second.memory_record["provisional_maintenance"] is False


def test_failure_and_corrective_relearning_produce_shorter_next_interval_than_success():
    """Ensure correction after failure is recorded separately from independent recall success."""
    item = build_item()
    config = SpacedRepetitionConfig()
    success = build_review(START, outcome="success")
    failure = build_review(START, outcome="failure", accuracy=0.2,
                           completeness=0.2, relation_accuracy=0.1, fluency=0.2)
    correction = build_review(START, outcome="corrected_relearning", accuracy=0.5,
                              completeness=0.5, relation_accuracy=0.5,
                              feedback_provided=True,
                              corrected_response="A synapse is a junction for communication.")
    success_schedule = choose_next_interval(80.0, item, success, START, config)
    failure_schedule = choose_next_interval(80.0, item, failure, START, config)
    corrected_schedule = choose_next_interval(80.0, item, correction, START, config)
    assert failure_schedule["interval_hours"] < corrected_schedule["interval_hours"]
    assert corrected_schedule["interval_hours"] < success_schedule["interval_hours"]
    assert correction.feedback_provided is True


def test_offline_consolidation_is_noop_without_observation_and_gated_when_present():
    """Do not apply an automatic sleep boost when no evidence has been supplied."""
    config = SpacedRepetitionConfig()
    baseline = {"observed": False, "duration_hours": 0.0, "quality": 0.5,
                "interference": 0.0, "evidence_confidence": 0.0}
    no_observation = __import__(
        "ore_spaced_repetition.functional", fromlist=["apply_offline_consolidation"]
    ).apply_offline_consolidation(100.0, 0.3, baseline, config)
    assert no_observation["stability_hours"] == 100.0
    assert no_observation["stability_adjustment_hours"] == 0.0
    assert no_observation["guaranteed_sleep_boost"] is False
    observed = OfflineConsolidationObservation(
        observed=True, duration_hours=8.0, quality=0.9,
        interference=0.1, evidence_confidence=0.8,
    )
    adjustment = __import__(
        "ore_spaced_repetition.functional", fromlist=["apply_offline_consolidation"]
    ).apply_offline_consolidation(100.0, 0.3, {
        "observed": observed.observed, "duration_hours": observed.duration_hours,
        "quality": observed.quality, "interference": observed.interference,
        "evidence_confidence": observed.evidence_confidence,
    }, config)
    assert adjustment["stability_adjustment_hours"] > 0
    assert adjustment["guaranteed_sleep_boost"] is False


def test_maintenance_requires_multiple_spaced_independent_successes():
    """Do not label an item provisionally stable after a single easy or aided attempt."""
    config = SpacedRepetitionConfig(maintenance_min_span_hours=168.0,
                                    maintenance_successes_required=3)
    history = [
        {"occurred_at_utc": "2026-01-01T00:00:00Z", "independent_retrieval": True,
         "outcome": "success", "quality_score": 0.95},
        {"occurred_at_utc": "2026-01-08T00:00:00Z", "independent_retrieval": True,
         "outcome": "success", "quality_score": 0.96},
        {"occurred_at_utc": "2026-01-15T00:00:00Z", "independent_retrieval": True,
         "outcome": "success", "quality_score": 0.94},
    ]
    result = assess_maintenance(history, START, "2026-01-15T01:00:00Z", config)
    assert result["provisionally_stable"] is True
    assert result["permanent_mastery_claimed"] is False
    assert assess_maintenance(history[:1], START, "2026-01-02T00:00:00Z",
                              config)["provisionally_stable"] is False


def test_calibration_uses_only_observed_independent_binary_results():
    """Compute Brier score from supplied outcome labels and preserve an empty-data state."""
    config = SpacedRepetitionConfig()
    empty = assess_calibration([], config)
    assert empty["calibration_available"] is False
    populated = assess_calibration([
        {"predicted": 0.8, "observed": 1.0},
        {"predicted": 0.7, "observed": 0.0},
    ], config)
    assert populated["sample_count"] == 2
    assert populated["brier_score"] == pytest.approx((0.2 ** 2 + 0.7 ** 2) / 2)


def test_biological_surrogates_are_bounded_and_keep_spacing_effect_explicit():
    """Check pattern binding/reactivation and non-negative spacing-sensitive plasticity."""
    network = BiologicalSpacedRepetitionNetwork(dimensions=16)
    encoded = network.encode(["synapse", "junction", "neuron"], {"item_id": "syn-1"})
    assert encoded["hippocampal_binding"]["bound"] is True
    review = network.review("syn-1", "What is a synapse?", 0.4, 0.9,
                            0.6, 72.0, np.zeros(16))
    assert review["reactivation"]["reactivation_count"] == 1
    assert review["plasticity"]["spacing_factor"] > 0.0
    assert network.snapshot()["literal_neuron_simulation"] is False


def test_contracts_reject_invalid_timestamps_and_review_scores():
    """Reject naive timestamps, out-of-range performance, and unsupported modes."""
    with pytest.raises(ValueError):
        LearningItem("bad-time", "content", created_at_utc="2026-01-01T00:00:00")
    with pytest.raises(ValueError):
        build_review(START, accuracy=1.5)
    with pytest.raises(ValueError):
        ReviewObservation(START, "unknown", 0.5)
    with pytest.raises(ValueError):
        SpacedRepetitionConfig(target_recall_probability=0.99)


def test_state_reuse_keeps_independent_review_history_for_multiple_items():
    """Maintain distinct records and schedules in the same learner memory bank."""
    state = SpacedRepetitionState("learner-bank")
    processor = SpacedRepetitionProcessor("learner-bank", state=state)
    first = processor.process(SpacedRepetitionRequest(
        "learner-bank", build_item(item_id="item-one"), START
    ))
    second = processor.process(SpacedRepetitionRequest(
        "learner-bank", build_item(item_id="item-two"), START
    ))
    assert set(second.final_state["items"]) == {"item-one", "item-two"}
    assert first.final_state["process_count"] == 1
    assert second.final_state["process_count"] == 2


def test_sqlite_registers_all_definitions_and_persists_twelve_operation_events(tmp_path):
    """Index Point 03 classes/functions and persist the numbered event trace to SQLite."""
    from ore_spaced_repetition import record_result_to_sqlite
    from ore_logger.database import LoggerDatabase

    result = SpacedRepetitionProcessor("learner-sqlite").process(
        SpacedRepetitionRequest("learner-sqlite", build_item(), START)
    )
    database_path = tmp_path / "point03.sqlite3"
    root = Path(__file__).resolve().parents[1]
    report = record_result_to_sqlite(result, database_path, repository_root=root)
    database = LoggerDatabase(database_path)
    rows = database.search_annotations("ore_spaced_repetition", limit=1000)
    definitions = [row for row in rows if row["symbol_kind"] in {"class", "function", "async_function"}]
    assert definitions
    assert all(row["annotation"].strip() and row["source_docstring"] for row in definitions)
    with sqlite3.connect(database_path) as connection:
        run = connection.execute("SELECT status, event_count FROM runs WHERE run_id=?",
                                 (report["run_id"],)).fetchone()
        events = connection.execute(
            "SELECT sequence_no, event_kind, metadata_json FROM events "
            "WHERE run_id=? ORDER BY sequence_no", (report["run_id"],)
        ).fetchall()
    assert run == ("completed", 12)
    assert [row[0] for row in events] == list(range(1, 13))
    assert all(row[1] == "operation" and "P03-" in row[2] for row in events)
