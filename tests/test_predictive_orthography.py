import numpy as np
import pytest

from ore_n3 import LabeledN2Patch, N3Config, N3Patch, N3Processor, N3Sequence
from ore_n4 import N4Config, N4Processor
from ore_predictive_orthography import (
    LIFConfig, MASTER_OPERATION_REGISTRY, MASTER_STAGE_REGISTRY,
    PredictiveOrthographyProcessor,
)

def _build_processor(*, simulate_biology=True):
    references = [
        LabeledN2Patch("a", features=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        LabeledN2Patch("a", features=np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)),
        LabeledN2Patch("b", features=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
        LabeledN2Patch("b", features=np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32)),
        LabeledN2Patch("c", features=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)),
        LabeledN2Patch("c", features=np.array([0.0, 0.1, 0.9, 0.0], dtype=np.float32)),
    ]
    n3 = N3Processor(N3Config(max_iterations=6)).fit(
        references,
        grapheme_sequences=[("a", "b", "c"), ("a", "a", "b"), ("b", "c", "a")],
    )
    return PredictiveOrthographyProcessor(
        n3, N4Processor(N4Config(max_iterations=6)),
        simulate_biology=simulate_biology,
        lif_config=LIFConfig(neuron_count=8, simulation_steps=120),
    )

def _sequence():
    return N3Sequence("seven-stage-test", [
        N3Patch("p1", features=np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32), segmentation_confidence=0.95),
        N3Patch("p2", features=np.array([0.05, 0.95, 0.0, 0.0], dtype=np.float32), segmentation_confidence=0.85),
        N3Patch("p3", features=np.array([0.0, 0.05, 0.95, 0.0], dtype=np.float32), segmentation_confidence=0.90),
    ])

def test_master_registry_is_exactly_seven_stages_and_forty_operations():
    assert len(MASTER_STAGE_REGISTRY) == 7
    assert len(MASTER_OPERATION_REGISTRY) == 40
    assert [len(stage.operation_ids) for stage in MASTER_STAGE_REGISTRY] == [8, 7, 5, 8, 4, 5, 3]
    assert [item.operation_id for item in MASTER_OPERATION_REGISTRY] == [
        f"MPO-{index:03d}" for index in range(1, 41)
    ]
    assert all(stage.operation_ids for stage in MASTER_STAGE_REGISTRY)

def test_functional_and_biological_tracks_execute_and_remain_separate():
    result = _build_processor().process(
        _sequence(), task_relevance=np.array([0.1, 1.0, 0.2], dtype=np.float32)
    )
    assert result.diagnostics["valid"] is True
    assert len(result.stage_trace) == 7
    assert len(result.operation_trace) == 40
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"MPO-{index:03d}" for index in range(1, 41)
    ]
    assert result.posterior_beliefs.shape == (3, 4)
    assert np.allclose(result.posterior_beliefs.sum(axis=1), 1.0, atol=1e-5)
    assert result.left_neighbor_available.tolist() == [False, True, True]
    assert result.right_neighbor_available.tolist() == [True, True, False]
    assert result.attended_feature_residuals.shape == (3, 4)
    assert result.biological["enabled"] is True
    assert result.biological["numerically_valid"] is True
    assert len(result.biological["populations"]) == 7
    assert result.biological["simulated_time_ms"] == pytest.approx(120.0)
    assert result.n3_result.posterior_beliefs is not result.n4_result.posterior_beliefs

def test_biological_track_can_be_disabled_without_changing_functional_contract():
    with_biology = _build_processor(simulate_biology=True).process(_sequence())
    without_biology = _build_processor(simulate_biology=False).process(_sequence())
    assert with_biology.diagnostics["valid"] is True
    assert without_biology.diagnostics["valid"] is True
    assert without_biology.biological["enabled"] is False
    assert len(without_biology.operation_trace) == 40
    assert np.allclose(with_biology.posterior_beliefs, without_biology.posterior_beliefs)
    assert np.allclose(without_biology.posterior_beliefs.sum(axis=1), 1.0, atol=1e-5)


def test_operation_trace_has_unique_outputs_and_complete_source_step_mapping():
    from ore_predictive_orthography.source_mapping import SOURCE_STEP_MAP

    result = _build_processor(simulate_biology=False).process(_sequence())
    assert len(SOURCE_STEP_MAP) == 38
    assert result.diagnostics["source_step_mapping_complete"] is True
    assert result.diagnostics["operation_specific_outputs"] is True
    assert all(row["source_step_ids"] for row in result.operation_trace)
    assert all(row["outputs_produced"] for row in result.operation_trace)
    assert all(
        row["execution_evidence"]["stage_summary_reused_as_operation_output"] is False
        for row in result.operation_trace
    )
    output_fields = [tuple(row["execution_evidence"]["output_field_names"])
                     for row in result.operation_trace]
    assert len(set(output_fields)) > 1
    assert result.operation_trace[8]["outputs_produced"]["observed_grapheme_exposure_counts"]["size"] > 0


def test_prediction_diagnostics_feed_an_explicit_next_processing_cycle():
    """Use MPO stage-7.7 residuals and uncertainty as validated cues for a subsequent sequence."""
    processor = _build_processor(simulate_biology=False)
    first = processor.process(_sequence())
    feedback = processor.feedback_for_next_cycle(first)

    assert feedback["source_operation_id"] == "MPO-040"
    assert feedback["source_step_id"] == "7.7"
    assert feedback["source_sequence_id"] == first.sequence_id
    assert feedback["bottom_up_salience"].shape == (3,)
    assert feedback["task_relevance"].shape == (3,)
    assert feedback["seed_attention"].shape == (3,)
    assert np.all((feedback["bottom_up_salience"] >= 0.0) &
                  (feedback["bottom_up_salience"] <= 1.0))

    second = processor.process(_sequence(), feedback=feedback)
    assert second.diagnostics["feedback_consumed"] is True
    assert second.metadata["next_cycle_feedback_input"]["source_sequence_id"] == first.sequence_id
    assert second.diagnostics["valid"] is True


def test_feedback_shape_and_range_are_validated_before_n4_processing():
    """Reject feedback that cannot align with the next sequence's reading positions."""
    processor = _build_processor(simulate_biology=False)
    with pytest.raises(ValueError, match="length 3"):
        processor.process(_sequence(), feedback={"seed_attention": np.array([0.1, 0.2])})
    with pytest.raises(ValueError, match="lie in \\[0, 1\\]"):
        processor.process(_sequence(), feedback={"task_relevance": np.array([0.1, 1.2, 0.2])})


def test_every_registered_predictive_operation_is_covered_by_a_source_step():
    """Require at least one source operation mapping for each MPO operation."""
    from ore_predictive_orthography.source_mapping import OPERATION_TO_SOURCE_STEPS

    assert set(OPERATION_TO_SOURCE_STEPS) == {
        f"MPO-{index:03d}" for index in range(1, 41)
    }
    assert all(OPERATION_TO_SOURCE_STEPS.values())


def test_unfitted_n3_processor_is_rejected_before_inference():
    processor = PredictiveOrthographyProcessor(N3Processor(), simulate_biology=False)
    with pytest.raises(RuntimeError, match="fit the N3 processor"):
        processor.process(_sequence())
