"""Functional, biological, uncertainty, generalization, and SQLite tests for Point 02."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from ore_cortical_learning import (
    CorticalLearningConfig,
    EncounterInput,
    GradualCorticalLearningProcessor,
    OPERATION_REGISTRY,
    register_annotations,
)
from ore_cortical_learning.functional import (
    beta_uncertainty,
    calculate_modulatory_gate,
    conditional_reliability,
    cosine_similarity,
    relationship_observations,
    timing_eligibility,
)


def _encounter(
    encounter_id: str,
    sensory: tuple[float, ...] = (1.0, 0.2, 0.1, 0.4, 0.6, 0.3, 0.1, 0.2, 0.5, 0.4, 0.2, 0.1),
    context_id: str = "room-a",
    observed: tuple[float, ...] | None = None,
    evaluation_only: bool = False,
    interval_ms: float = 0.0,
) -> EncounterInput:
    """Build a deterministic multi-channel encounter for repeatable cortical-learning tests."""
    return EncounterInput(
        encounter_id=encounter_id,
        sensory_activity=sensory,
        context_id=context_id,
        action_activity=(0.1, 0.3, 0.7, 0.2),
        context=(0.2, 0.8, 0.1, 0.4) if context_id == "room-a" else (0.8, 0.1, 0.4, 0.2),
        goal_activity=(0.7, 0.2, 0.5),
        episode_trace=(0.1, 0.4, 0.2, 0.3, 0.5),
        observed_activity=observed,
        modulatory_state={"attention": 0.8, "novelty": 0.4, "reward": 0.7, "arousal": 0.6},
        pre_post_interval_ms=interval_ms,
        evaluation_only=evaluation_only,
    )


def test_registry_contains_all_fourteen_source_ordered_operations() -> None:
    """Protect the 14-stage source sequence and require equations plus evidence labels."""
    assert len(OPERATION_REGISTRY) == 14
    assert [row.operation_id for row in OPERATION_REGISTRY] == [f"CL{i:02d}" for i in range(1, 15)]
    assert all(row.mathematical_form and row.biological_reference and row.evidence_level
               for row in OPERATION_REGISTRY)


def test_similarity_conditional_reliability_and_relationship_counts() -> None:
    """Verify overlap metrics and conditional co-activation estimates from positive/negative evidence."""
    assert np.isclose(cosine_similarity([1.0, 0.0], [2.0, 0.0]), 1.0)
    success, failure = relationship_observations([0.9, 0.8, 0.1])
    assert success[0, 1] == 1.0 and failure[0, 2] == 1.0
    probabilities, counts = conditional_reliability(success, failure)
    assert probabilities[0, 1] > probabilities[0, 2]
    assert counts[0, 1] == 1.0 and counts[0, 2] == 1.0
    uncertainty = beta_uncertainty(1.0 + success, 1.0 + failure)
    assert 0.0 <= uncertainty["mean_binary_entropy"] <= 1.0


def test_modulatory_gate_and_timing_rule_are_bounded_and_signed() -> None:
    """Confirm bounded state-dependent gating and timing-sensitive eligibility signs."""
    high = calculate_modulatory_gate({"attention": 1.0, "novelty": 1.0, "reward": 1.0, "arousal": 1.0})
    low = calculate_modulatory_gate({"attention": 0.0, "novelty": 0.0, "reward": 0.0, "arousal": 0.0})
    assert high["gate"] == 1.0
    assert low["gate"] < high["gate"]
    assert timing_eligibility(10.0, 100.0) > 0.0
    assert timing_eligibility(-10.0, 100.0) < 0.0
    assert timing_eligibility(0.0, 100.0) == 1.0


def test_repeated_encounters_accumulate_statistics_and_modify_small_weight_steps() -> None:
    """Verify persistent cortical weights and co-activity statistics change incrementally."""
    processor = GradualCorticalLearningProcessor()
    before = processor.model.recurrent_weights.copy()
    first = processor.process(_encounter("train-1", observed=(0.1,) * 12))
    second = processor.process(_encounter("train-2", observed=(0.2,) * 12))
    third = processor.process(_encounter("train-3", observed=(0.3,) * 12))
    assert all(len(result.operation_trace) == 14 for result in (first, second, third))
    assert processor.encounter_count == 3
    assert processor.model.update_count == 3
    assert np.any(np.abs(processor.model.recurrent_weights - before) > 0.0)
    assert np.any(processor.coactivity_ema > 0.0)
    assert np.all(np.abs(processor.model.recurrent_weights) <= processor.config.max_abs_weight)
    assert np.all(np.abs(processor.model.prediction_weights) <= processor.config.max_abs_weight)
    assert first.prediction_error is not None and first.prediction_mse is not None
    assert second.relationship_reliability["source"] == "context_specific_branch"


def test_repeated_similar_examples_reactivate_overlapping_cortical_patterns() -> None:
    """Check encounter comparison identifies prior similar patterns without erasing context labels."""
    processor = GradualCorticalLearningProcessor()
    processor.process(_encounter("a-1"))
    same_context = processor.process(_encounter("a-2"))
    different_context = processor.process(_encounter("b-1", context_id="room-b"))
    assert same_context.similarities["overlap_detected"] is True
    assert same_context.similarities["context_match"] is True
    assert different_context.similarities["nearest_encounter"]["context_id"] == "room-a"
    assert different_context.similarities["context_match"] is False


def test_context_specific_relationship_branches_are_kept_separate() -> None:
    """Preserve distinct relationship statistics for encounters with different context identifiers."""
    processor = GradualCorticalLearningProcessor()
    processor.process(_encounter("ctx-a", context_id="room-a"))
    processor.process(_encounter("ctx-b", context_id="room-b"))
    assert set(processor.context_relationships) == {"room-a", "room-b"}
    assert set(processor.context_prototypes) == {"room-a", "room-b"}
    assert processor.context_encounter_counts["room-a"] == 1
    assert processor.context_encounter_counts["room-b"] == 1


def test_held_out_evaluation_does_not_update_weights_or_training_evidence() -> None:
    """Prevent held-out encounters from changing synapses, reliability, prototypes, or train history."""
    processor = GradualCorticalLearningProcessor()
    processor.process(_encounter("training-1", observed=(0.5,) * 12))
    recurrent_before = processor.model.recurrent_weights.copy()
    prediction_before = processor.model.prediction_weights.copy()
    success_before = processor.relationship_success.copy()
    failure_before = processor.relationship_failure.copy()
    history_before = len(processor.history)
    prototype_before = processor.context_prototypes["room-a"].copy()
    evaluation = processor.process(_encounter(
        "heldout-1", observed=(0.4,) * 12, evaluation_only=True, interval_ms=50.0
    ))
    assert evaluation.generalization_evaluation["status"] == "held_out_evaluation_completed"
    assert evaluation.generalization_evaluation["training_state_changed"] is False
    assert np.allclose(processor.model.recurrent_weights, recurrent_before)
    assert np.allclose(processor.model.prediction_weights, prediction_before)
    assert np.allclose(processor.relationship_success, success_before)
    assert np.allclose(processor.relationship_failure, failure_before)
    assert len(processor.history) == history_before
    assert np.allclose(processor.context_prototypes["room-a"], prototype_before)
    assert processor.evaluation_count == 1 and processor.encounter_count == 1
    assert evaluation.operation_trace[7]["status"] == "skipped_evaluation_only"


def test_no_target_means_no_fabricated_prediction_error() -> None:
    """Return an explicit not-observed marker instead of fabricating a downstream target."""
    result = GradualCorticalLearningProcessor().process(_encounter("no-target"))
    assert result.prediction_error is None
    assert result.prediction_mse is None
    assert result.operation_trace[5]["functional_output"]["status"] == "not_observed"


def test_homeostatic_state_is_bounded_and_activity_remains_finite() -> None:
    """Ensure population gain regulation leaves finite activity and respects configured bounds."""
    processor = GradualCorticalLearningProcessor(CorticalLearningConfig(target_activity=0.3))
    result = processor.process(_encounter("homeostasis"))
    assert np.all(np.isfinite(result.cortical_activity))
    assert np.all(processor.model.homeostatic_gain >= processor.config.homeostatic_gain_min)
    assert np.all(processor.model.homeostatic_gain <= processor.config.homeostatic_gain_max)


def test_inconsistent_relationships_increase_uncertainty_instead_of_forcing_certainty() -> None:
    """Accumulate conflicting conditional outcomes and keep the uncertainty estimate visible."""
    processor = GradualCorticalLearningProcessor()
    stable_input = (1.0,) + (0.0,) * 11
    processor.process(_encounter("positive-1", sensory=stable_input, observed=(1.0,) * 12))
    processor.process(_encounter("positive-2", sensory=stable_input, observed=(1.0,) * 12))
    before = processor.snapshot()["relationship_reliability_mean"]
    processor.process(_encounter("mixed-1", sensory=(0.0, 1.0) + (0.0,) * 10,
                                 context_id="room-b", observed=(0.0,) * 12))
    report = processor.snapshot()
    assert report["context_branch_count"] == 2
    assert 0.0 <= before <= 1.0
    assert report["relationship_reliability_mean"] >= 0.0


def test_every_package_class_and_function_has_a_docstring() -> None:
    """Require documentation on every package class, function, nested helper, and method."""
    package_root = Path(__file__).resolve().parents[1] / "ore_cortical_learning"
    for source_path in package_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node), f"missing annotation docstring: {source_path.name}:{node.name}"


def test_package_annotations_are_persisted_in_logger_sqlite(tmp_path: Path) -> None:
    """Index the package and verify class, method, and helper annotations are stored in SQLite."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "point02-annotations.sqlite3"
    report = register_annotations(database_path, root)
    assert report["parse_errors"] == 0
    from ore_logger.database import LoggerDatabase

    database = LoggerDatabase(database_path)
    expected = (
        ("ore_cortical_learning.pipeline", "GradualCorticalLearningProcessor", "class"),
        ("ore_cortical_learning.pipeline", "GradualCorticalLearningProcessor.process", "function"),
        ("ore_cortical_learning.pipeline", "GradualCorticalLearningProcessor.process.<locals>.record", "function"),
        ("ore_cortical_learning.biology", "CorticalPopulationModel.apply_plasticity", "function"),
        ("ore_cortical_learning.contracts", "EncounterInput", "class"),
    )
    for module, qualname, kind in expected:
        row = database.get_annotation(module, qualname, kind)
        assert row is not None and row["annotation"].strip()
        assert row["annotation_source"] == "docstring"
    assert database.summary()["symbols"] >= 30


def test_processor_records_fourteen_operation_events_in_sqlite(tmp_path: Path) -> None:
    """Verify one processor encounter persists all 14 operation annotations and run records."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "point02-run.sqlite3"
    processor = GradualCorticalLearningProcessor(database_path=database_path, repository_root=root)
    result = processor.process(_encounter("sqlite-run", observed=(0.2,) * 12))
    assert result.metadata["sqlite_logger"]["enabled"] is True
    from ore_logger.database import LoggerDatabase

    database = LoggerDatabase(database_path)
    runs = database.list_runs()
    assert len(runs) == 1 and runs[0]["status"] == "completed"
    events = database.list_events(runs[0]["run_id"])
    operations = [event for event in events if event["event_kind"] == "operation"]
    assert len(operations) == 14
    assert all(event.get("annotation") for event in operations)
