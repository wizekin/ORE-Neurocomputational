from __future__ import annotations

from pathlib import Path

from ore_distributed_only import (
    ConceptPattern,
    DistributedOnlyProcessor,
    DistributedSemanticRequest,
    EpisodicExperience,
    STAGE_REGISTRY,
    index_distributed_annotations,
    record_distributed_result,
)
from ore_logger import LoggerDatabase


def _concepts() -> list[ConceptPattern]:
    """Create a small, explicit multimodal concept inventory for deterministic tests."""
    return [
        ConceptPattern(
            label="violin",
            category="musical_instrument",
            description="A stringed musical instrument played with a bow.",
            typical_action="Hold the violin and move the bow across its strings.",
            properties={
                "linguistic": {"token:violin": 1.0},
                "visual": {"shape:curved_wooden_body": 0.9, "parts:strings": 1.0},
                "auditory": {"sound:bowed_string_timbre": 0.95},
                "action": {"action:hold_and_bow": 1.0},
                "tactile": {"texture:polished_wood": 0.7},
                "affective": {"affect:musical_pleasure": 0.6},
                "spatial": {"posture:instrument_under_chin": 0.8},
            },
        ),
        ConceptPattern(
            label="guitar",
            category="musical_instrument",
            description="A stringed instrument commonly played by plucking or strumming.",
            typical_action="Hold the guitar and pluck or strum its strings.",
            properties={
                "linguistic": {"token:guitar": 1.0},
                "visual": {"shape:waisted_body": 0.9, "parts:strings": 1.0},
                "auditory": {"sound:plucked_string_timbre": 0.95},
                "action": {"action:pluck_or_strum": 1.0},
                "tactile": {"texture:polished_wood": 0.6},
                "spatial": {"posture:instrument_across_lap": 0.8},
            },
        ),
    ]


def test_registry_contains_all_21_stages_and_source_operations():
    """Ensure the source algorithm is represented by 21 sequential stages."""
    assert len(STAGE_REGISTRY) == 21
    assert [stage.stage_id for stage in STAGE_REGISTRY] == [
        f"DS-{number:02d}" for number in range(1, 22)
    ]
    assert sum(len(stage.operations) for stage in STAGE_REGISTRY) >= 70


def test_pipeline_executes_all_stages_and_preserves_distributed_populations():
    """Exercise all stages and verify that outputs remain population-specific."""
    processor = DistributedOnlyProcessor(_concepts(), auto_log=False, random_seed=4)
    processor.add_experience(EpisodicExperience(
        concept_label="violin",
        properties={
            "auditory": {"sound:bowed_string_timbre": 0.9},
            "action": {"action:hold_and_bow": 0.9},
            "visual": {"shape:curved_wooden_body": 0.8},
        },
        familiarity=0.8,
        durability=0.9,
    ))
    result = processor.process(DistributedSemanticRequest(
        linguistic_input="violin",
        modality_inputs={"auditory": {"sound:bowed_string_timbre": 0.85}},
        context={"venue:concert_hall": 0.7},
        task="describe",
        reliability={"auditory": 0.95, "linguistic": 0.9},
    ))
    expected_operations = sum(len(stage.operations) for stage in STAGE_REGISTRY)
    assert result.diagnostics["valid"]
    assert result.diagnostics["central_semantic_hub"] is False
    assert len(result.stage_trace) == 21
    assert len(result.operation_trace) == expected_operations
    assert result.operation_trace[0]["operation_id"] == "DS-01.signal_sampling"
    assert result.stage_trace[-1]["stage_id"] == "DS-21"
    assert set(result.population_activations) == {
        "visual", "auditory", "action", "tactile",
        "affective", "linguistic", "contextual", "spatial",
    }
    assert result.observed_indices["linguistic"]
    assert result.episodic_retrieval
    assert result.timing_coordination["global_phase_coherence"] >= 0.0
    biological = result.diagnostics["biological_simulation"]
    assert biological["enabled"] is True
    assert biological["stage_count"] == 21
    assert biological["numerically_valid"] is True
    assert "mean_membrane_voltage_mv" in biological["populations"]["auditory"]
    assert result.interpretation in {"violin", "guitar", None}


def test_learning_requires_explicit_authorization_and_known_target():
    """Verify that model weights change only after an authorized learning request."""
    processor = DistributedOnlyProcessor(_concepts(), auto_log=False, random_seed=5)
    before = processor.population_connectivity.copy()
    processor.process(DistributedSemanticRequest(linguistic_input="violin"))
    assert (processor.population_connectivity == before).all()

    processor.process(DistributedSemanticRequest(
        linguistic_input="violin",
        learning=True,
        learning_target="violin",
        spike_times_ms={"linguistic": 0.0, "auditory": 4.0, "action": 8.0},
    ))
    assert (processor.population_connectivity != before).any()
    assert len(processor.episodes) == 1


def test_lesions_are_counterfactual_and_leave_model_connectivity_unchanged():
    """Check that the lesion stage attenuates selected populations without mutating the base graph."""
    processor = DistributedOnlyProcessor(_concepts(), auto_log=False, random_seed=6)
    before = processor.population_connectivity.copy()
    result = processor.process(DistributedSemanticRequest(
        linguistic_input="violin",
        lesion_strengths={"auditory": 1.0},
    ))
    assert result.lesion_analysis["population_mask"][1] == 0.0
    assert (processor.population_connectivity == before).all()
    assert result.lesion_analysis["interpretation"].startswith("Counterfactual")


def test_annotation_index_and_run_trace_are_persisted_to_sqlite(tmp_path):
    """Index Python symbols and confirm every stage operation is stored in the ORE logger database."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "distributed-only.sqlite3"
    processor = DistributedOnlyProcessor(_concepts(), auto_log=False, random_seed=7)
    result = processor.process(DistributedSemanticRequest(linguistic_input="violin"))
    index_report = index_distributed_annotations(root, database_path)
    run_report = record_distributed_result(result, root, database_path)

    database = LoggerDatabase(database_path)
    class_annotation = database.get_annotation(
        "ore_distributed_only.processor",
        "DistributedOnlyProcessor",
        "class",
        "ore_distributed_only/processor.py",
    )
    process_annotation = database.get_annotation(
        "ore_distributed_only.processor",
        "DistributedOnlyProcessor.process",
        "function",
        "ore_distributed_only/processor.py",
    )
    with database.connect() as connection:
        operation_count = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE symbol_kind='operation' "
            "AND source_path='ore_distributed_only/registry.py'"
        ).fetchone()[0]
        event_count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND event_kind='operation'",
            (run_report["run_id"],),
        ).fetchone()[0]

    assert class_annotation and class_annotation["annotation"]
    assert process_annotation and process_annotation["annotation"]
    assert index_report["stage_count"] == 21
    assert operation_count == run_report["operation_count"]
    assert event_count == run_report["operation_count"]
    assert run_report["database_summary"]["events"] > 0
