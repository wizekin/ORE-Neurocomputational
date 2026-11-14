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

def test_unfitted_n3_processor_is_rejected_before_inference():
    processor = PredictiveOrthographyProcessor(N3Processor(), simulate_biology=False)
    with pytest.raises(RuntimeError, match="fit the N3 processor"):
        processor.process(_sequence())
