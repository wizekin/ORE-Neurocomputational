"""Regression tests for the 22-stage grounded semantic integration subsystem."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ore_logger import LoggerDatabase
from ore_grounded_integration import (
    GroundedSemanticConfig,
    GroundedSemanticIntegrationProcessor,
    GroundedSemanticRequest,
    OPERATION_REGISTRY,
    STAGE_REGISTRY,
    record_grounded_result,
    register_grounded_annotations,
)
from ore_grounded_integration.math_ops import (
    cosine_similarity,
    hebbian_update,
    stable_softmax,
    temporal_association,
)


def _request(sample_id: str = "bird-1", **overrides) -> GroundedSemanticRequest:
    """Create a compact deterministic multimodal bird example for pipeline tests."""
    values = {
        "sample_id": sample_id,
        "text": "The bird builds a nest",
        "modality_inputs": {
            "visual": {"wings": 0.9, "feathers": 0.8},
            "auditory": {"chirp": 0.7},
            "motor": {"build": 0.8, "fly": 0.6},
        },
        "context": {"forest": 1.0, "tree": 0.7},
        "task": "describe",
        "candidate_senses": {
            "bird_animal": {"bird": 1.0, "feathers": 0.5},
            "sports_bat": {"bat": 1.0, "ball": 0.5},
        },
        "learner_response": {
            "recall": 0.8,
            "identification": 0.9,
            "action": 0.6,
            "contextual": 0.7,
            "generalization": 0.5,
        },
        "usage_examples": ("A bird builds a nest.", "The bird sings."),
        "source_reference": "unit_test",
    }
    values.update(overrides)
    return GroundedSemanticRequest(**values)


def test_registry_contains_22_contiguous_stages_and_operations():
    """Verify that all stages and each registered mathematical operation are discoverable."""
    assert len(STAGE_REGISTRY) == 22
    assert [stage.stage_id for stage in STAGE_REGISTRY] == [
        f"GS-{index:02d}" for index in range(1, 23)
    ]
    assert len(OPERATION_REGISTRY) >= 40
    assert all(stage.operations for stage in STAGE_REGISTRY)
    assert all(operation.mathematical_form for operation in OPERATION_REGISTRY)


def test_all_22_stages_run_and_emit_optional_biological_reference():
    """Verify complete functional traces and the separate, explicitly scoped LIF simulation."""
    processor = GroundedSemanticIntegrationProcessor(
        GroundedSemanticConfig(hidden_dim=32, population_size=4, simulate_biology=True)
    )
    result = processor.process(_request())
    assert result.diagnostics["valid"] is True
    assert len(result.stage_trace) == 22
    assert len(result.operation_trace) == len(OPERATION_REGISTRY)
    assert len(result.biological_trace) == 22
    assert result.stage_outputs["GS-09"]["selected_meaning"] == "bird_animal"
    assert result.stage_outputs["GS-21"]["evaluated"] is True
    assert result.stage_outputs["GS-22"]["target_dimension"] == "generalization"
    assert "computational approximation" in result.diagnostics["scientific_scope"].lower()
    json.dumps(result.to_dict(), default=str)


def test_learning_updates_word_transitions_graph_and_retrievable_memory():
    """Verify that repeated experiences alter associative state and enable later prediction."""
    processor = GroundedSemanticIntegrationProcessor(
        GroundedSemanticConfig(hidden_dim=32, simulate_biology=False)
    )
    first = processor.process(_request(sample_id="bird-first"))
    assert first.stage_outputs["GS-11"]["prediction_available"] is False
    assert processor.next_token_counts["bird"]["builds"] == 1
    assert "bird" in processor.semantic_graph
    assert "bird_animal" in processor.semantic_memory

    second = processor.process(_request(
        sample_id="bird-second",
        text="bird",
        modality_inputs={"visual": {"wings": 0.95, "feathers": 0.85}},
        context={"forest": 1.0},
        learner_response=None,
        usage_examples=(),
    ))
    assert second.stage_outputs["GS-14"]["semantic_memories"]
    assert second.stage_outputs["GS-12"]["available"] is True
    assert second.stage_outputs["GS-22"]["activity"] == "collect_baseline_response"


def test_mathematical_primitives_are_finite_and_well_behaved():
    """Test softmax normalization, vector similarity, temporal association and Hebbian updates."""
    probabilities = stable_softmax({"a": 1.0, "b": 2.0}, temperature=0.5)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-12
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert 0.0 < temporal_association(0.25, 1.0) < 1.0
    updated = hebbian_update(
        np.zeros((2, 2)),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        learning_rate=0.1,
        decay=0.0,
    )
    assert updated[1, 0] == 0.1
    assert np.all(np.isfinite(updated))


def test_sqlite_catalog_indexes_every_new_class_and_function(tmp_path):
    """Verify docstring annotations for the new processor and its stage methods are stored in SQLite."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "grounded-catalog.sqlite3"
    report = register_grounded_annotations(root, database_path)
    database = LoggerDatabase(database_path)
    class_row = database.get_annotation(
        "ore_grounded_integration.pipeline",
        "GroundedSemanticIntegrationProcessor",
        "class",
        "ore_grounded_integration/pipeline.py",
    )
    method_row = database.get_annotation(
        "ore_grounded_integration.pipeline",
        "GroundedSemanticIntegrationProcessor._stage_22_adapt",
        "function",
        "ore_grounded_integration/pipeline.py",
    )
    assert report["stage_count"] == 22
    assert report["symbol_count"] > 0
    assert class_row and "22 functional stages" in class_row["annotation"]
    assert method_row and "Stage 22" in method_row["annotation"]


def test_sqlite_run_persists_all_operations_and_stage_traces(tmp_path):
    """Verify a processed result creates a SQLite run with operation and component-trace events."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "grounded-run.sqlite3"
    processor = GroundedSemanticIntegrationProcessor(
        GroundedSemanticConfig(hidden_dim=32, population_size=4, simulate_biology=False)
    )
    result = processor.process(_request())
    report = record_grounded_result(result, root, database_path)
    database = LoggerDatabase(database_path)
    runs = database.list_runs()
    assert report["stage_trace_count"] == 22
    assert report["operation_trace_count"] == len(OPERATION_REGISTRY)
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    events = database.list_events(runs[0]["run_id"])
    assert sum(event["event_kind"] == "operation" for event in events) == len(OPERATION_REGISTRY)
    assert sum(event["event_kind"] == "component_trace" for event in events) == 22
    assert all(event.get("annotation") for event in events if event["event_kind"] == "operation")
