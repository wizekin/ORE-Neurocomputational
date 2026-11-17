"""Acceptance tests for the 19-stage amodal ATL semantic processing model."""

from __future__ import annotations

from pathlib import Path

import pytest

from ore_logger import LoggerDatabase
from ore_atl_amodal import (
    ALL_OPERATIONS,
    STAGE_REGISTRY,
    AmodalATLConfig,
    AmodalATLProcessor,
    AmodalATLRequest,
    AmodalConcept,
    index_atl_amodal_annotations,
    record_atl_amodal_result,
)


def _concepts() -> list[AmodalConcept]:
    """Build compact, overlapping semantic prototypes for reproducible cross-modal tests."""
    return [
        AmodalConcept(
            concept_id="dog",
            label="dog",
            aliases=("canine",),
            semantic_features={
                "ANIMAL": 1.0, "MAMMAL": 0.95, "FOUR_LEGS": 0.85,
                "BARKS": 0.8, "DOMESTIC": 0.75,
            },
            modality_features={
                "visual": {"FOUR_LEGS": 0.9, "FUR": 0.8, "TAIL": 0.7},
                "auditory": {"BARKS": 1.0, "WHINES": 0.6},
                "spatial_motor": {"RUNS": 0.8, "WAGS_TAIL": 0.7},
                "tactile": {"SOFT_FUR": 0.8},
            },
            category="animal",
            source_reference="test://dog",
        ),
        AmodalConcept(
            concept_id="diamond",
            label="diamond",
            aliases=("gem",),
            semantic_features={
                "MINERAL": 1.0, "CRYSTAL": 0.95, "HARD": 0.9,
                "TRANSPARENT": 0.8, "GEMSTONE": 0.9,
            },
            modality_features={
                "visual": {"FACETED": 0.9, "TRANSPARENT": 0.8},
                "tactile": {"HARD": 1.0, "SMOOTH": 0.6},
            },
            category="mineral",
            source_reference="test://diamond",
        ),
    ]


def _processor(simulate_biology: bool = False) -> AmodalATLProcessor:
    """Create a deterministic small-dimensional processor for fast acceptance tests."""
    return AmodalATLProcessor(
        _concepts(),
        config=AmodalATLConfig(
            hidden_dim=16,
            recurrent_steps=6,
            simulate_biology=simulate_biology,
            population_size=4,
            steps_per_stage=2,
            random_seed=11,
        ),
    )


def test_registry_contains_nineteen_ordered_stages_and_unique_suboperations() -> None:
    """Verify all 19 stages have stable identifiers and mathematically specified operations."""
    assert len(STAGE_REGISTRY) == 19
    assert [stage.stage_id for stage in STAGE_REGISTRY] == [
        f"ATL-{index:02d}" for index in range(1, 20)
    ]
    identifiers = [operation.operation_id for operation in ALL_OPERATIONS]
    assert len(identifiers) == len(set(identifiers))
    assert len(ALL_OPERATIONS) >= 50
    assert all(operation.mathematical_operation for operation in ALL_OPERATIONS)
    assert all(stage.biological_reference for stage in STAGE_REGISTRY)


def test_multimodal_concept_access_executes_all_stages_and_operations() -> None:
    """Run visual and auditory evidence through all functional stages and verify concept retrieval."""
    result = _processor().process(AmodalATLRequest(
        sequence_id="dog-cross-modal",
        text="dog",
        modality_inputs={
            "visual": {"FOUR_LEGS": 0.95, "FUR": 0.8},
            "auditory": {"BARKS": 0.9},
        },
    ))
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["stage_count"] == 19
    assert len(result.stage_trace) == 19
    assert [row["stage_id"] for row in result.stage_trace] == [
        f"ATL-{index:02d}" for index in range(1, 20)
    ]
    assert len(result.operation_trace) == len(ALL_OPERATIONS)
    assert result.output["kind"] == "concept"
    assert result.output["concept_id"] == "dog"
    assert result.stage_outputs["ATL-04"]["gate_sum"] == pytest.approx(1.0)
    assert result.stage_outputs["ATL-09"]["raw_observations_preserved"] is True
    assert result.to_dict()["schema"] == "ore.atl_amodal.19_stage.v1"


def test_lexical_only_input_can_access_shared_semantics() -> None:
    """Confirm a written word can enter semantic processing without an image or sound feature map."""
    result = _processor().process(AmodalATLRequest(
        sequence_id="written-dog",
        text="dog",
    ))
    assert result.stage_outputs["ATL-01"]["available_modalities"] == ["linguistic"]
    assert len(result.stage_trace) == 19
    assert result.output["concept_id"] == "dog"


def test_privative_and_subsective_adjectives_separate_similarity_from_membership() -> None:
    """Ensure a fake diamond can resemble a diamond without being categorized as a genuine diamond."""
    processor = _processor()
    fake = processor.process(AmodalATLRequest(
        sequence_id="fake-diamond",
        text="fake diamond",
    ))
    small = processor.process(AmodalATLRequest(
        sequence_id="small-diamond",
        text="small diamond",
    ))

    fake_denotation = fake.output["denotation"]
    small_denotation = small.output["denotation"]
    assert fake.output["kind"] == "phrase"
    assert fake_denotation["adjective_type"] == "privative"
    assert fake_denotation["noun_category_membership"] < 0.5
    assert fake_denotation["belongs_to_noun_category"] is False
    assert small_denotation["adjective_type"] == "subsective"
    assert small_denotation["noun_category_membership"] >= 0.65
    assert small_denotation["belongs_to_noun_category"] is True


def test_explicit_phrase_components_and_feature_overrides_are_supported() -> None:
    """Allow caller-supplied semantic feature vectors to control noun and adjective encodings."""
    result = _processor().process(AmodalATLRequest(
        sequence_id="caller-defined-phrase",
        noun="diamond",
        adjective="fake",
        noun_features={"MINERAL": 1.0, "HARD": 0.9},
        adjective_features={"NONAUTHENTIC": 1.0, "IMITATION": 0.9},
        adjective_type="privative",
    ))
    assert result.output["noun"] == "diamond"
    assert result.output["adjective"] == "fake"
    assert result.stage_outputs["ATL-13"]["feature_source"] == "caller_features"
    assert result.output["denotation"]["belongs_to_noun_category"] is False


def test_bilateral_lif_reference_records_all_nineteen_stages_and_physical_units() -> None:
    """Validate separate LIF population dynamics for sensory spokes, bilateral ATL, and phrase processing."""
    result = AmodalATLProcessor(
        _concepts(),
        config=AmodalATLConfig(
            hidden_dim=16,
            recurrent_steps=5,
            simulate_biology=True,
            population_size=4,
            steps_per_stage=2,
            random_seed=13,
        ),
    ).process(AmodalATLRequest(
        sequence_id="biology-dog",
        text="dog",
        modality_inputs={"visual": {"FOUR_LEGS": 1.0}},
    ))
    biological = result.biological
    assert biological["enabled"] is True
    assert biological["stage_count"] == 19
    assert len(biological["operation_trace"]) == 19
    assert biological["numerically_valid"] is True
    assert biological["units"]["membrane_voltage"] == "mV"
    assert biological["units"]["synaptic_current"] == "pA"
    assert biological["units"]["firing_rate"] == "Hz"
    assert biological["units"]["simulated_time"] == "ms"
    assert {"atl_left", "atl_right", "visual_spoke"} <= set(biological["layers"])


def test_disabling_biology_does_not_disable_the_functional_algorithm() -> None:
    """Keep all functional stages and mathematical traces available when LIF simulation is disabled."""
    result = _processor(simulate_biology=False).process(AmodalATLRequest(
        sequence_id="functional-only",
        text="dog",
        modality_inputs={"visual": {"FOUR_LEGS": 0.9}},
    ))
    assert len(result.stage_trace) == 19
    assert result.biological["enabled"] is False
    assert all(row["biological_status"] == "skipped_disabled" for row in result.operation_trace)


def test_hebbian_learning_is_preview_only_until_explicit_authorization() -> None:
    """Propose co-activation weight updates without mutating the pathway weights by default."""
    processor = _processor()
    request = AmodalATLRequest(
        sequence_id="learning-preview",
        text="dog",
        modality_inputs={"visual": {"FOUR_LEGS": 0.95, "FUR": 0.9}},
        learning_experiences=({
            "modality_features": {"visual": {"FOUR_LEGS": 1.0, "FUR": 0.9}},
            "semantic_features": {"ANIMAL": 1.0, "MAMMAL": 0.9},
        },),
    )
    initial_generation = processor.model_generation
    preview = processor.process(request)
    assert preview.learning["status"] == "preview_only_authorization_required"
    assert processor.model_generation == initial_generation

    committed = processor.process(AmodalATLRequest(
        sequence_id="learning-committed",
        text="dog",
        modality_inputs={"visual": {"FOUR_LEGS": 0.95, "FUR": 0.9}},
        learning_experiences=request.learning_experiences,
        learning_authorized=True,
    ))
    assert committed.learning["status"] == "committed"
    assert processor.model_generation == initial_generation + 1


def test_every_package_class_and_function_is_indexed_in_the_logger_database(tmp_path: Path) -> None:
    """Use the repository AST catalogue to store class and callable explanations in SQLite."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "atl-annotations.sqlite3"
    report = index_atl_amodal_annotations(root, database_path)
    database = LoggerDatabase(database_path)

    processor_class = database.get_annotation(
        "ore_atl_amodal.pipeline",
        "AmodalATLProcessor",
        "class",
        "ore_atl_amodal/pipeline.py",
    )
    process_method = database.get_annotation(
        "ore_atl_amodal.pipeline",
        "AmodalATLProcessor.process",
        "function",
        "ore_atl_amodal/pipeline.py",
    )
    assert report["atl_amodal_symbol_count"] > 30
    assert processor_class is not None
    assert process_method is not None
    assert "Coordinate multimodal encoding" in processor_class["annotation"]
    assert "Execute all 19 registered functional stages" in process_method["annotation"]


def test_operation_and_stage_events_are_recorded_in_the_existing_sqlite_logger(tmp_path: Path) -> None:
    """Record every operation and stage trace as events linked to persisted symbol annotations."""
    root = Path(__file__).resolve().parents[1]
    result = _processor().process(AmodalATLRequest(
        sequence_id="sqlite-recording",
        text="dog",
        modality_inputs={"visual": {"FOUR_LEGS": 0.9}},
    ))
    metadata = record_atl_amodal_result(
        result, root, tmp_path / "atl-execution.sqlite3"
    )
    database = LoggerDatabase(metadata["database_path"])
    run = database.get_run(metadata["run_id"])
    events = database.list_events(metadata["run_id"])

    assert run is not None
    assert run["status"] == "completed"
    assert metadata["stage_count"] == 19
    assert metadata["operation_count"] == len(ALL_OPERATIONS)
    assert metadata["diagnostics_valid"] is True
    assert len(events) >= len(ALL_OPERATIONS) + 19
    assert any(event["event_kind"] == "operation" for event in events)
    assert any(event["event_kind"] == "component_trace" for event in events)
