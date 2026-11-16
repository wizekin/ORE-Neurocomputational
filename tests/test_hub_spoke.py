"""Acceptance and regression tests for the 21-stage Hub-and-Spoke implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ore_logger import LoggerDatabase
from ore_semantic.contracts import SemanticConcept, SemanticRequest
from ore_hub_spoke import (
    ALL_OPERATIONS,
    STAGE_REGISTRY,
    HubAndSpokeConfig,
    HubAndSpokeProcessor,
    HubAndSpokeRequest,
    LearningExperience,
    index_hub_spoke_annotations,
    record_hub_spoke_result,
    request_from_semantic_request,
)


def _concepts() -> list[SemanticConcept]:
    """Create a compact, provenance-tagged set of overlapping concepts for controlled tests."""
    return [
        SemanticConcept(
            concept_id="c-giraffe",
            label="giraffe",
            aliases=("giraffe",),
            semantic_features={
                "ANIMAL": 1.0, "MAMMAL": 0.9, "LONG_NECK": 0.95,
                "SPOTTED": 0.85, "TALL": 0.95, "BROWSER": 0.7,
            },
            modality_features={
                "visual": {"LONG_NECK": 1.0, "SPOTTED": 0.9, "TALL": 1.0},
                "auditory": {"LOW_CALL": 0.6},
                "spatial_motor": {"BROWSE_HIGH_LEAVES": 0.9},
            },
            lexical_category="noun",
            frequency=10.0,
            prototype_id="mammal",
            verified_by="test",
            source_reference="test://giraffe",
        ),
        SemanticConcept(
            concept_id="c-zebra",
            label="zebra",
            aliases=("zebra",),
            semantic_features={
                "ANIMAL": 1.0, "MAMMAL": 0.9, "STRIPED": 1.0,
                "TALL": 0.7, "HERBIVORE": 0.8,
            },
            modality_features={
                "visual": {"STRIPED": 1.0, "FOUR_LEGS": 0.9},
                "auditory": {"BARKING_CALL": 0.6},
                "spatial_motor": {"GRAZING": 0.8},
            },
            lexical_category="noun",
            frequency=9.0,
            prototype_id="mammal",
            verified_by="test",
            source_reference="test://zebra",
        ),
        SemanticConcept(
            concept_id="c-violin",
            label="violin",
            aliases=("fiddle",),
            semantic_features={"ARTIFACT": 1.0, "MUSICAL_INSTRUMENT": 1.0},
            modality_features={
                "visual": {"WOODEN_BODY": 0.9, "STRINGS": 1.0},
                "auditory": {"BOWED_TONE": 1.0, "PITCH": 0.9},
                "spatial_motor": {"BOWING": 0.9, "FINGERING": 0.8},
            },
            lexical_category="noun",
            frequency=6.0,
            prototype_id="instrument",
            verified_by="test",
            source_reference="test://violin",
        ),
    ]


def _processor(simulate_biology: bool = True) -> HubAndSpokeProcessor:
    """Create a deterministic processor with short recurrent and neural simulations."""
    return HubAndSpokeProcessor(
        _concepts(),
        config=HubAndSpokeConfig(
            simulate_biology=simulate_biology,
            recurrent_steps=6,
            steps_per_stage=1,
            population_size=4,
            random_seed=7,
        ),
        associations={"c-violin": {"c-giraffe": 0.05}},
    )


def test_registry_contains_21_ordered_stages_and_unique_suboperations() -> None:
    """Verify every stage and sub-operation has a stable, complete registry record."""
    assert len(STAGE_REGISTRY) == 21
    assert [stage.stage_id for stage in STAGE_REGISTRY] == [
        f"HS-{index:02d}" for index in range(1, 22)
    ]
    ids = [operation.operation_id for operation in ALL_OPERATIONS]
    assert len(ids) == len(set(ids))
    assert len(ids) > 90
    assert all(operation.name and operation.mathematical_operation for operation in ALL_OPERATIONS)
    assert all(stage.biological_reference for stage in STAGE_REGISTRY)


def test_end_to_end_processing_executes_all_stages_and_operation_trace() -> None:
    """Run a visual-plus-linguistic concept and validate all ordered outputs."""
    result = _processor().process(HubAndSpokeRequest(
        sequence_id="sample-giraffe",
        text="giraffe",
        modality_inputs={"visual": {"LONG_NECK": 0.95, "SPOTTED": 0.8}},
        task="describe",
    ))

    assert result.diagnostics["valid"] is True
    assert len(result.stage_trace) == 21
    assert [row["stage_id"] for row in result.stage_trace] == [
        f"HS-{index:02d}" for index in range(1, 22)
    ]
    assert len(result.operation_trace) == len(ALL_OPERATIONS)
    assert result.output["accepted"] is True
    assert result.output["concept_id"] == "c-giraffe"
    assert result.stage_outputs["HS-06"]["gate_sum"] == pytest.approx(1.0)
    assert result.stage_outputs["HS-09"]["observations_overwritten"] is False
    assert result.to_dict()["schema"] == "ore.hub_spoke.21_stage.v1"


def test_image_or_nonlinguistic_evidence_can_access_concepts_without_a_text_cue() -> None:
    """Allow a sensory-only request and retrieve candidates from visual properties."""
    result = _processor(simulate_biology=False).process(HubAndSpokeRequest(
        sequence_id="visual-only",
        modality_inputs={"visual": {"LONG_NECK": 0.9, "SPOTTED": 0.8}},
    ))
    assert result.stage_outputs["HS-01"]["available_modalities"] == ["visual"]
    assert result.stage_outputs["HS-11"]["ranked_candidates"]
    assert result.diagnostics["stage_count"] == 21


def test_missing_modality_is_not_encoded_as_negative_evidence() -> None:
    """Confirm absent spokes receive zero gates rather than negative feature activation."""
    result = _processor(simulate_biology=False).process(HubAndSpokeRequest(
        sequence_id="auditory-only",
        modality_inputs={"auditory": {"LOW_CALL": 0.9}},
    ))
    stage = result.stage_outputs["HS-06"]
    gates = stage["modality_gates"]
    assert gates["auditory"] > 0.0
    assert stage["direct_input_modalities"] == ["auditory"]
    assert stage["missing_channels_encoded_as_negative"] is False
    assert sum(gates.values()) == pytest.approx(1.0)


def test_learning_is_preview_only_until_explicitly_authorized() -> None:
    """Ensure experience proposals do not mutate the model until the caller authorizes learning."""
    experience = LearningExperience(
        concept_id="c-giraffe",
        modality_features={
            "visual": {"LONG_NECK": 1.0, "SPOTTED": 0.9},
            "auditory": {"LOW_CALL": 0.5},
        },
        semantic_features={"ANIMAL": 1.0, "MAMMAL": 0.9},
        confidence=0.95,
        verified_by="reviewer",
        source_reference="dataset://giraffe-episode-1",
        episode_id="giraffe-1",
    )
    processor = _processor(simulate_biology=False)
    initial_generation = processor.model_generation
    request = HubAndSpokeRequest(
        sequence_id="learning-preview",
        text="giraffe",
        modality_inputs={"visual": {"LONG_NECK": 0.9}},
    )
    preview = processor.process(request, [experience], learning_authorized=False)
    assert preview.learning["stage_18"]["status"] == "preview_only_authorization_required"
    assert processor.model_generation == initial_generation
    assert processor.experience_memory("c-giraffe") == {}

    learned = processor.process(request, [experience], learning_authorized=True)
    assert learned.learning["stage_18"]["status"] == "committed"
    assert processor.model_generation == initial_generation + 1
    assert "visual" in processor.experience_memory("c-giraffe")
    assert "ANIMAL" in processor.learned_feature_probabilities("c-giraffe")


def test_cross_modal_alignment_and_recurrent_refinement_are_reported() -> None:
    """Check that the shared hub, modality residuals, and recurrent convergence are observable."""
    result = _processor(simulate_biology=False).process(HubAndSpokeRequest(
        sequence_id="cross-modal",
        text="violin",
        modality_inputs={
            "visual": {"WOODEN_BODY": 0.9, "STRINGS": 1.0},
            "auditory": {"BOWED_TONE": 0.95, "PITCH": 0.8},
            "spatial_motor": {"BOWING": 0.9},
        },
    ))
    assert result.stage_outputs["HS-08"]["iterations"] >= 1
    assert "shared_hub_state" in result.stage_outputs["HS-10"]
    routes = set(result.stage_outputs["HS-10"]["modality_specific_residual_norms"])
    assert {"visual", "auditory", "spatial_motor"} <= routes


def test_hub_spoke_and_connection_lesions_are_separate_counterfactuals() -> None:
    """Verify three lesion families expose distinct masks and pathway profiles."""
    result = _processor(simulate_biology=False).process(HubAndSpokeRequest(
        sequence_id="lesion-test",
        text="giraffe",
        modality_inputs={
            "visual": {"LONG_NECK": 1.0, "SPOTTED": 0.9},
            "auditory": {"LOW_CALL": 0.7},
        },
    ))
    assert result.lesions["hub"]["degraded_dimensions"]
    assert len(result.lesions["spokes"]["individual_spoke_profiles"]) == 7
    assert len(result.lesions["connections"]["individual_connection_profiles"]) == 7
    assert result.lesions["hub"]["impaired"]["specificity_margin"] >= 0.0
    assert result.lesions["connections"]["endpoint_populations_destroyed"] is False


def test_biological_reference_records_millivolts_picoamps_hz_and_ms() -> None:
    """Validate all 21 LIF stage snapshots and the declared physical units."""
    result = _processor().process(HubAndSpokeRequest(
        sequence_id="lif-sample",
        text="giraffe",
        modality_inputs={"visual": {"LONG_NECK": 1.0}},
    ))
    biological = result.biological
    assert biological["enabled"] is True
    assert biological["stage_count"] == 21
    assert len(biological["operation_trace"]) == 21
    assert biological["numerically_valid"] is True
    assert biological["units"]["membrane_voltage"] == "mV"
    assert biological["units"]["synaptic_current"] == "pA"
    assert biological["units"]["firing_rate"] == "Hz"
    assert biological["units"]["simulated_time"] == "ms"
    assert "atl_hub" in biological["layers"]
    assert "visual_spoke" in biological["layers"]


def test_biology_can_be_disabled_without_shortening_functional_trace() -> None:
    """Keep the functional algorithm complete when neural dynamics are disabled."""
    result = _processor(simulate_biology=False).process(HubAndSpokeRequest(
        sequence_id="no-biology",
        text="giraffe",
        modality_inputs={"visual": {"LONG_NECK": 1.0}},
    ))
    assert len(result.stage_trace) == 21
    assert result.biological["enabled"] is False
    assert all(row["biological_status"] == "skipped_disabled" for row in result.operation_trace)


def test_adapter_preserves_existing_semantic_request() -> None:
    """Adapt the existing ore_semantic request contract without modifying its contents."""
    original = SemanticRequest(
        sequence_id="legacy-semantic-request",
        text="giraffe",
        context={"ANIMAL": 0.8},
        modality_evidence={"visual": {"LONG_NECK": 0.9}},
    )
    adapted = request_from_semantic_request(original)
    assert adapted.sequence_id == original.sequence_id
    assert adapted.concept_cue == original.text
    assert adapted.context_features == original.context
    assert adapted.modality_inputs == original.modality_evidence


def test_logger_indexes_every_package_class_and_function_in_sqlite(tmp_path: Path) -> None:
    """Use the repository logger AST catalog to persist documented package symbols."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "hub-spoke-annotations.sqlite3"
    report = index_hub_spoke_annotations(root, database_path)
    database = LoggerDatabase(database_path)

    processor_class = database.get_annotation(
        "ore_hub_spoke.pipeline", "HubAndSpokeProcessor", "class", "ore_hub_spoke/pipeline.py"
    )
    process_method = database.get_annotation(
        "ore_hub_spoke.pipeline", "HubAndSpokeProcessor.process", "function",
        "ore_hub_spoke/pipeline.py",
    )
    assert report["hub_spoke_symbol_count"] > 40
    assert processor_class is not None
    assert process_method is not None
    assert processor_class["annotation_source"] == "docstring"
    assert "21 functional stages" in process_method["annotation"]


def test_operation_and_stage_events_are_written_to_existing_logger(tmp_path: Path) -> None:
    """Record an executed result as reusable operation annotations and SQLite run events."""
    root = Path(__file__).resolve().parents[1]
    result = _processor(simulate_biology=False).process(HubAndSpokeRequest(
        sequence_id="sqlite-run",
        text="giraffe",
        modality_inputs={"visual": {"LONG_NECK": 0.9}},
    ))
    metadata = record_hub_spoke_result(
        result, root, tmp_path / "hub-spoke-run.sqlite3"
    )
    database = LoggerDatabase(metadata["database_path"])
    run = database.get_run(metadata["run_id"])
    events = database.list_events(metadata["run_id"])
    assert run is not None
    assert run["status"] == "completed"
    assert metadata["stage_count"] == 21
    assert metadata["operation_count"] == len(ALL_OPERATIONS)
    assert len(events) >= len(ALL_OPERATIONS)
    assert any(event["event_kind"] == "operation" for event in events)
