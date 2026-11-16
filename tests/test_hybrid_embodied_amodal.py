"""Software tests for the 19-stage hybrid embodied–amodal implementation."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from ore_hybrid_embodied_amodal import (
    ALL_OPERATIONS,
    STAGE_REGISTRY,
    HybridConcept,
    HybridConfig,
    HybridEmbodiedAmodalProcessor,
    HybridLearningExample,
    HybridRequest,
    index_hybrid_annotations,
    record_hybrid_result,
)


@pytest.fixture
def guitar_concept() -> HybridConcept:
    """Provide a small, transparent multimodal concept prototype for deterministic tests."""
    return HybridConcept(
        concept_id="guitar",
        label="guitar",
        aliases=("guitar",),
        category="musical instrument",
        features={"instrument": 1.0, "strings": 0.9, "music": 0.8, "neck": 0.7},
        modality_features={
            "visual": {"strings": 0.9, "neck": 0.8},
            "auditory": {"music": 0.9, "sound": 0.8},
            "motor": {"strumming": 0.9, "finger_position": 0.8},
        },
        verified_by="test",
        source_reference="synthetic://test/guitar",
    )


@pytest.fixture
def multimodal_request() -> HybridRequest:
    """Provide temporally aligned visual, auditory, motor, and tactile input evidence."""
    return HybridRequest(
        sequence_id="test-guitar-001",
        modality_inputs={
            "visual": {"strings": 0.9, "neck": 0.8},
            "auditory": {"music": 0.9, "sound": 0.8},
            "motor": {"strumming": 0.9, "finger_position": 0.8},
            "somatosensory": {"vibration": 0.7, "contact": 0.8},
        },
        task="recognition",
        context_features={"instrument": 0.9, "music": 0.8},
        task_features={"instrument": 1.0, "strings": 0.8},
        modality_uncertainty={
            "visual": 0.1, "auditory": 0.2, "motor": 0.3, "somatosensory": 0.2
        },
        modality_times={
            "visual": 1.0, "auditory": 1.05, "motor": 1.08, "somatosensory": 1.1
        },
        spatial_features={
            "visual": {"neck": 0.8},
            "somatosensory": {"contact": 0.8},
        },
        response_mode="recognition",
    )


def test_registry_has_nineteen_ordered_stages_and_stable_operations() -> None:
    """Check the requested 19-stage order and unique hierarchical sub-operation IDs."""
    assert len(STAGE_REGISTRY) == 19
    assert [stage.stage_id for stage in STAGE_REGISTRY] == [
        f"HEC-01.{number:02d}" for number in range(1, 20)
    ]
    operation_ids = [item.operation_id for item in ALL_OPERATIONS]
    assert len(operation_ids) == len(set(operation_ids))
    assert all(stage.steps for stage in STAGE_REGISTRY)
    assert all(item.mathematical_operation for item in ALL_OPERATIONS)
    assert all(item.biological_reference for item in ALL_OPERATIONS)


def test_end_to_end_pipeline_runs_all_stages_and_preserves_both_tracks(
    guitar_concept: HybridConcept,
    multimodal_request: HybridRequest,
) -> None:
    """Run the full functional algorithm and check shared plus modality-specific outputs."""
    processor = HybridEmbodiedAmodalProcessor(
        [guitar_concept], HybridConfig(simulate_biology=False)
    )
    result = processor.process(multimodal_request)

    assert result.diagnostics["valid"] is True
    assert result.diagnostics["stage_count"] == 19
    assert result.diagnostics["operation_count"] == len(ALL_OPERATIONS)
    assert len(result.stage_trace) == 19
    assert len(result.operation_trace) == len(ALL_OPERATIONS)
    assert result.selected_concept is not None
    assert result.selected_concept["concept_id"] == "guitar"
    assert len(result.common_representation) == processor.config.hub_dimensions
    assert set(result.modality_representations) == {
        "visual", "auditory", "motor", "somatosensory"
    }
    assert set(result.modality_residuals) == set(result.modality_representations)
    assert set(result.reactivated_representations) == set(result.modality_representations)
    assert result.biological["enabled"] is False
    assert np.all(np.isfinite(result.common_representation))


def test_reliability_weights_normalize_and_temporal_binding_is_visible(
    guitar_concept: HybridConcept,
    multimodal_request: HybridRequest,
) -> None:
    """Verify precision weighting sums to one and temporally close channels bind more strongly."""
    result = HybridEmbodiedAmodalProcessor(
        [guitar_concept], HybridConfig(simulate_biology=False)
    ).process(multimodal_request)
    stage_9 = result.stage_outputs["HEC-01.09"]
    assert stage_9["weight_sum"] == pytest.approx(1.0)
    assert sum(stage_9["modality_weights"].values()) == pytest.approx(1.0)
    bindings = result.stage_outputs["HEC-01.07"]["binding_strengths"]
    assert bindings["visual"]["auditory"] > 0.0
    assert bindings["visual"]["auditory"] > bindings["visual"]["motor"]


def test_attention_inhibition_and_contextual_output_are_traceable(
    guitar_concept: HybridConcept,
    multimodal_request: HybridRequest,
) -> None:
    """Check that task/context gating and caller-supplied irrelevant features are recorded."""
    request = HybridRequest(
        sequence_id="attention-test",
        modality_inputs=multimodal_request.modality_inputs,
        task="visual recognition",
        context_features={"instrument": 0.8},
        task_features={"strings": 1.0},
        irrelevant_features={"vibration": 1.0},
        response_mode="description",
    )
    result = HybridEmbodiedAmodalProcessor(
        [guitar_concept], HybridConfig(simulate_biology=False)
    ).process(request)
    assert result.stage_outputs["HEC-01.04"]["attention_weights"]["strings"] > 0.5
    assert result.stage_outputs["HEC-01.04"]["inhibited_features"]["vibration"] == 1.0
    assert result.response["kind"] in ("description", "recognition")
    assert result.stage_outputs["HEC-01.14"]["adapted_representation"]


def test_learning_is_preview_only_without_authorization(
    guitar_concept: HybridConcept,
    multimodal_request: HybridRequest,
) -> None:
    """Ensure a proposed new experience cannot change long-term memory without authorization."""
    example = HybridLearningExample(
        concept_id="guitar",
        modality_inputs={"visual": {"strings": 1.0}, "auditory": {"music": 0.9}},
        shared_features={"instrument": 1.0, "music": 0.9},
        confidence=0.9,
        verified_by="reviewer",
        source_reference="dataset://reviewed/guitar-episode",
    )
    request = HybridRequest(
        sequence_id="learning-preview",
        modality_inputs=multimodal_request.modality_inputs,
        learning_examples=(example,),
        learning_authorized=False,
    )
    processor = HybridEmbodiedAmodalProcessor(
        [guitar_concept], HybridConfig(simulate_biology=False)
    )
    result = processor.process(request)
    assert result.learning["status"] == "preview_only"
    assert result.learning["committed_examples"] == 0
    assert result.learning["learned_memory_size"] == 0


def test_authorized_learning_requires_source_and_reviewer_provenance(
    guitar_concept: HybridConcept,
    multimodal_request: HybridRequest,
) -> None:
    """Commit only explicitly authorized learning examples carrying reviewer and source metadata."""
    verified = HybridLearningExample(
        concept_id="guitar",
        modality_inputs={"visual": {"strings": 1.0}, "auditory": {"music": 0.9}},
        shared_features={"instrument": 1.0, "music": 0.9},
        confidence=0.9,
        verified_by="independent-reviewer",
        source_reference="dataset://verified/guitar-002",
    )
    unverified = HybridLearningExample(
        concept_id="unknown",
        modality_inputs={"visual": {"shape": 1.0}},
        shared_features={"shape": 1.0},
        confidence=0.9,
        verified_by="unspecified",
        source_reference="unspecified",
    )
    request = HybridRequest(
        sequence_id="learning-authorized",
        modality_inputs=multimodal_request.modality_inputs,
        learning_examples=(verified, unverified),
        learning_authorized=True,
    )
    processor = HybridEmbodiedAmodalProcessor(
        [guitar_concept], HybridConfig(simulate_biology=False)
    )
    result = processor.process(request)
    assert result.learning["status"] == "committed"
    assert result.learning["committed_examples"] == 1
    assert result.learning["blocked_examples"] == 1
    assert result.learning["learned_memory_size"] == 1


def test_lif_reference_reports_physical_units_and_19_stage_trace(
    guitar_concept: HybridConcept,
    multimodal_request: HybridRequest,
) -> None:
    """Check independent neural-reference simulation state, units, and finite measurements."""
    config = HybridConfig(simulate_biology=True, population_size=4, steps_per_stage=1)
    result = HybridEmbodiedAmodalProcessor([guitar_concept], config).process(multimodal_request)
    assert result.biological["enabled"] is True
    assert result.biological["stage_count"] == 19
    assert result.biological["units"]["membrane_voltage"] == "mV"
    assert result.biological["units"]["synaptic_current"] == "pA"
    assert result.biological["units"]["firing_rate"] == "Hz"
    assert result.biological["units"]["simulated_time"] == "ms"
    assert result.biological["numerically_valid"] is True
    for row in result.biological["layers"].values():
        assert np.isfinite(row["mean_voltage_mv"])
        assert np.isfinite(row["mean_current_pa"])
        assert np.isfinite(row["mean_firing_rate_hz"])


def test_existing_semantic_concept_contract_is_accepted(
    multimodal_request: HybridRequest,
) -> None:
    """Adapt the repository's existing SemanticConcept object without changing its contract."""
    from ore_semantic.contracts import SemanticConcept

    concept = SemanticConcept(
        concept_id="guitar",
        label="guitar",
        aliases=("guitar",),
        semantic_features={"instrument": 1.0, "strings": 0.9, "music": 0.8},
        modality_features={"visual": {"strings": 0.9}, "auditory": {"music": 0.8}},
        lexical_category="musical instrument",
        source_reference="dataset://semantic/guitar",
    )
    result = HybridEmbodiedAmodalProcessor(
        [concept], HybridConfig(simulate_biology=False)
    ).process(multimodal_request)
    assert result.diagnostics["stage_count"] == 19


def test_annotations_and_operation_events_are_written_to_sqlite(
    guitar_concept: HybridConcept,
    multimodal_request: HybridRequest,
    tmp_path: Path,
) -> None:
    """Index package symbols and persist run/operation explanations to the shared logger schema."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "hybrid-test.sqlite3"
    annotations = index_hybrid_annotations(root, database_path)
    assert annotations["hybrid_symbol_count"] > 0
    assert annotations["hybrid_annotations"].get("class", 0) > 0
    assert annotations["hybrid_annotations"].get("function", 0) > 0

    result = HybridEmbodiedAmodalProcessor(
        [guitar_concept], HybridConfig(simulate_biology=False)
    ).process(multimodal_request)
    run = record_hybrid_result(result, root, database_path)
    assert run["stage_count"] == 19
    assert run["operation_count"] == len(ALL_OPERATIONS)
    with sqlite3.connect(database_path) as connection:
        run_count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        event_count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=?", (run["run_id"],)
        ).fetchone()[0]
        linked_annotations = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND symbol_id IS NOT NULL",
            (run["run_id"],),
        ).fetchone()[0]
    assert run_count == 1
    assert event_count >= len(ALL_OPERATIONS)
    assert linked_annotations > 0


def test_request_rejects_nonfinite_or_out_of_range_feature_values() -> None:
    """Reject malformed input before the pipeline can introduce invalid numerical states."""
    with pytest.raises(ValueError):
        HybridRequest(
            sequence_id="invalid",
            modality_inputs={"visual": {"edge": float("nan")}},
        )
    with pytest.raises(ValueError):
        HybridRequest(
            sequence_id="invalid",
            modality_inputs={"visual": {"edge": 1.5}},
        )
