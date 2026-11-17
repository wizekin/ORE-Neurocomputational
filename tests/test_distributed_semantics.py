"""Regression tests for all 36 distributed semantic stages and ORE SQLite integration."""
from __future__ import annotations

from pathlib import Path

import pytest

from ore_logger import LoggerDatabase
from ore_semantic.contracts import SemanticConcept
from ore_distributed_semantics import (
    ALL_OPERATIONS,
    STAGE_REGISTRY,
    DistributedSemanticConfig,
    DistributedSemanticProcessor,
    DistributedSemanticRequest,
    index_distributed_semantic_annotations,
    record_distributed_semantic_result,
)


def _concepts() -> list[SemanticConcept]:
    """Build a compact, overlapping concept inventory with explicit provenance."""
    return [
        SemanticConcept(
            concept_id="musician",
            label="musician",
            aliases=("player",),
            semantic_features={"PERSON": 1.0, "HUMAN": 1.0, "MUSICIAN": 1.0},
            modality_features={"visual": {"HUMAN_FORM": 0.8}, "spatial_motor": {"PERFORMS_MUSIC": 0.9}},
            lexical_category="noun",
            prototype_id="person",
            verified_by="test",
            source_reference="test://musician",
        ),
        SemanticConcept(
            concept_id="violin",
            label="violin",
            aliases=("fiddle",),
            semantic_features={"ARTIFACT": 1.0, "MUSICAL_INSTRUMENT": 1.0, "STRING_INSTRUMENT": 1.0},
            modality_features={"visual": {"WOODEN_BODY": 0.9, "STRINGS": 1.0}, "auditory": {"BOWED_TONE": 1.0}},
            lexical_category="noun",
            prototype_id="instrument",
            verified_by="test",
            source_reference="test://violin",
        ),
        SemanticConcept(
            concept_id="sandwich",
            label="sandwich",
            semantic_features={"FOOD": 1.0, "EDIBLE": 1.0},
            modality_features={"visual": {"BREAD": 1.0}, "tactile": {"SOFT_TEXTURE": 0.7}},
            lexical_category="noun",
            prototype_id="food",
            verified_by="test",
            source_reference="test://sandwich",
        ),
    ]


def _request(**kwargs) -> DistributedSemanticRequest:
    """Create a reproducible linguistic-plus-multimodal example with explicit role evidence."""
    base = {
        "sequence_id": "semantic-test-001",
        "text": "musician tuned violin",
        "modality_inputs": {
            "visual": {"HUMAN_FORM": 0.7, "WOODEN_BODY": 0.8, "STRINGS": 0.9},
            "auditory": {"BOWED_TONE": 0.8},
        },
        "syntactic_relations": (
            {"head": "tuned", "dependent": "musician", "relation": "subject"},
            {"head": "tuned", "dependent": "violin", "relation": "object"},
        ),
        "event_roles": {"Agent": "musician", "Action": "tuned", "Theme": "violin"},
    }
    base.update(kwargs)
    return DistributedSemanticRequest(**base)


def _processor(tmp_path: Path, simulate_biology: bool = True) -> DistributedSemanticProcessor:
    """Create a deterministic, lightweight processor that preserves biological and functional tracks."""
    return DistributedSemanticProcessor(
        _concepts(),
        config=DistributedSemanticConfig(
            simulate_biology=simulate_biology,
            random_seed=5,
            steps_per_stage=1,
            population_size=4,
            recurrent_steps=2,
        ),
        associations={"musician": {"violin": 0.9}, "violin": {"musician": 0.9, "sandwich": 0.01}},
    )


def test_registry_contains_36_ordered_stages_and_108_math_operations() -> None:
    """Verify the master registry is complete, stable, and mathematically annotated."""
    assert len(STAGE_REGISTRY) == 36
    assert [stage.stage_id for stage in STAGE_REGISTRY] == [f"DS-{n:02d}" for n in range(1, 37)]
    assert len(ALL_OPERATIONS) == 108
    assert len({operation.operation_id for operation in ALL_OPERATIONS}) == 108
    assert all(operation.mathematical_operation for operation in ALL_OPERATIONS)
    assert all(stage.biological_reference for stage in STAGE_REGISTRY)


def test_end_to_end_runs_functional_and_lif_tracks_for_every_stage(tmp_path: Path) -> None:
    """Check the pipeline emits 36 functional stages, 108 operations, and 36 neural snapshots."""
    result = _processor(tmp_path).process(_request())

    assert result.diagnostics["valid"] is True
    assert len(result.stage_trace) == 36
    assert len(result.operation_trace) == 108
    assert len(result.biological_trace) == 36
    assert [row["stage_id"] for row in result.stage_trace] == [f"DS-{n:02d}" for n in range(1, 37)]
    assert result.stage_outputs["DS-06"]["atl_state"]["capacity"] == pytest.approx(1.0)
    assert result.stage_outputs["DS-11"]["event_frame"]["roles"]["Agent"] == "musician"
    assert result.stage_outputs["DS-20"]["scientific_status"] == "illustrative_computational_proxy_not_eeg"
    assert result.stage_outputs["DS-31"]["bilateral_atl_profile"]["clinical_reference"] == "semantic_dementia"
    assert result.stage_outputs["DS-36"]["evaluation"]["biological_validation_status"] == "not_tested_without_independent_human_data"
    assert result.to_dict()["schema"] == "ore.distributed_semantics.36_stage.v1"


def test_biological_reference_can_be_disabled_without_changing_functional_stage_count(tmp_path: Path) -> None:
    """Ensure synthetic LIF simulation is optional and separate from semantic function."""
    result = _processor(tmp_path, simulate_biology=False).process(_request())
    assert len(result.stage_trace) == 36
    assert len(result.biological_trace) == 0
    assert result.diagnostics["biology_simulated"] is False


def test_learning_is_preview_only_unless_explicitly_authorized(tmp_path: Path) -> None:
    """Prevent accidental mutation of long-term semantic state without allow_learning."""
    processor = _processor(tmp_path, simulate_biology=False)
    preview = processor.process(_request(
        sequence_id="preview",
        learning_feedback={"UNUSUAL_PLAYING_TECHNIQUE": 0.9},
        allow_learning=False,
    ))
    assert preview.stage_outputs["DS-26"]["learning_update"]["applied"] is False
    assert processor.model_generation == 0

    learned = processor.process(_request(
        sequence_id="authorized",
        learning_feedback={"UNUSUAL_PLAYING_TECHNIQUE": 0.9},
        allow_learning=True,
    ))
    assert learned.stage_outputs["DS-26"]["learning_update"]["applied"] is True
    assert processor.model_generation == 1


def test_annotations_for_every_new_python_symbol_are_indexed_in_sqlite(tmp_path: Path) -> None:
    """Index source docstrings and verify package classes/functions appear in the SQLite catalogue."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "distributed-semantic.sqlite3"
    report = index_distributed_semantic_annotations(root, database_path)
    assert report["distributed_semantic_symbol_count"] > 20

    database = LoggerDatabase(database_path)
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT qualified_name, symbol_kind, annotation
            FROM annotations
            WHERE source_path LIKE 'ore_distributed_semantics/%'
            """
        ).fetchall()
    indexed = {(row["qualified_name"], row["symbol_kind"]) for row in rows}
    assert any("DistributedSemanticProcessor" in name and kind == "class" for name, kind in indexed)
    assert any("process" in name and kind == "function" for name, kind in indexed)
    assert all(str(row["annotation"]).strip() for row in rows)


def test_run_recorder_persists_stage_and_operation_events(tmp_path: Path) -> None:
    """Verify one pipeline result is persisted through the repository's standard run/event logger."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "distributed-run.sqlite3"
    result = _processor(tmp_path, simulate_biology=False).process(_request())
    metadata = record_distributed_semantic_result(result, root, database_path)
    assert metadata["stage_count"] == 36
    assert metadata["operation_count"] == 108
    assert metadata["diagnostics_valid"] is True

    database = LoggerDatabase(database_path)
    with database.connect() as connection:
        run = connection.execute("SELECT status, event_count FROM runs").fetchone()
        events = connection.execute("SELECT COUNT(*) AS n FROM events").fetchone()
    assert run["status"] in {"completed", "completed_with_warnings"}
    assert run["event_count"] >= 108
    assert events["n"] >= 108
