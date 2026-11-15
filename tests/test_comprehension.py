"""Behavioral and annotation-storage tests for the 38-operation comprehension subsystem."""
from __future__ import annotations

from pathlib import Path

import pytest

from ore_comprehension import (
    ComprehensionConfig,
    ComprehensionProcessor,
    ComprehensionRequest,
    OPERATION_REGISTRY,
    STAGE_REGISTRY,
    tokenize_text,
)
from ore_logger.catalog import sync_repository
from ore_logger.database import LoggerDatabase


def test_registry_contains_exactly_38_operations_in_seven_stages() -> None:
    """Verify that the implemented operation registry preserves all 38 source-specification steps."""
    assert len(OPERATION_REGISTRY) == 38
    assert [item.number for item in OPERATION_REGISTRY] == list(range(1, 39))
    assert len(STAGE_REGISTRY) == 7
    assert sum(stage.last_operation - stage.first_operation + 1 for stage in STAGE_REGISTRY) == 38
    assert all(item.method_name.startswith("op_") for item in OPERATION_REGISTRY)


def test_tokenizer_preserves_sentence_boundary_markers() -> None:
    """Ensure the tokenizer retains punctuation needed by the sentence-completion operation."""
    assert tokenize_text("The chef prepared food.") == ["The", "chef", "prepared", "food", "."]


def test_unknown_context_defaults_to_generic_event_instead_of_arbitrary_script() -> None:
    """Ensure missing event cues yield a neutral hypothesis rather than the first registered schema."""
    processor = ComprehensionProcessor(ComprehensionConfig(simulate_biology=False))
    result = processor.process(ComprehensionRequest(
        sequence_id="generic-event-test",
        text="The glorp blorped.",
    ))
    assert result.sentence_interpretations[0]["event_hypothesis"] == "generic"


def test_functional_pipeline_executes_all_38_operations_for_each_token() -> None:
    """Check per-token operation traces, mathematical outputs, and final sentence consolidation."""
    processor = ComprehensionProcessor(ComprehensionConfig(simulate_biology=False))
    result = processor.process(ComprehensionRequest(
        sequence_id="functional-test",
        text="The chef prepared the meal with a knife.",
    ))
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["registered_operation_count"] == 38
    assert result.diagnostics["operations_executed"] == len(result.tokens) * 38
    assert len(result.operation_trace) == len(result.tokens) * 38
    assert len(result.sentence_interpretations) == 1
    assert result.sentence_interpretations[0]["sentence_complete"] is True
    assert all(token["surprisal_bits"] >= 0.0 for token in result.token_records)
    assert all(token["prediction_entropy_bits"] >= 0.0 for token in result.token_records)
    assert all(token["n400_estimate_uv"] <= 0.0 for token in result.token_records)
    assert all(token["p600_estimate_uv"] >= 0.0 for token in result.token_records)
    assert result.operation_trace[0]["operation_id"] == "TDPI-001"
    assert result.operation_trace[37]["operation_id"] == "TDPI-038"


def test_caller_supplied_lexical_semantic_and_world_evidence_is_used() -> None:
    """Confirm caller-provided knowledge can extend the transparent built-in reference lexicon."""
    processor = ComprehensionProcessor(ComprehensionConfig(simulate_biology=False))
    result = processor.process(ComprehensionRequest(
        sequence_id="evidence-test",
        text="The glorp moved.",
        lexical_categories={"glorp": "NOUN"},
        semantic_features={"glorp": {"physical": 1.0, "tool": 1.0}},
        world_knowledge={"glorp": {"tool": 1.0, "used_for": 0.8}},
    ))
    word = next(item for item in result.token_records if item["normalized"] == "glorp")
    assert word["lexical_category"] == "NOUN"
    assert result.diagnostics["valid"] is True


def test_lif_reference_network_runs_independently_from_functional_outputs() -> None:
    """Verify the optional LIF population simulation returns finite unit-labelled diagnostics."""
    processor = ComprehensionProcessor(ComprehensionConfig(
        simulate_biology=True,
        lif_population_size=4,
        lif_steps_per_operation=1,
    ))
    result = processor.process(ComprehensionRequest(
        sequence_id="biology-test",
        text="The chef.",
    ))
    assert result.biological["enabled"] is True
    assert result.biological["numerically_valid"] is True
    assert result.biological["operation_count"] == len(result.tokens) * 38
    assert set(result.biological["layers"]) >= {
        "sentence_context", "syntactic_prediction", "semantic_prediction",
        "world_knowledge", "lexical_input", "prediction_error",
        "conflict_monitor", "working_memory", "learning",
    }


def test_operation_docstrings_are_indexed_in_logger_sqlite(tmp_path: Path) -> None:
    """Run the project annotation catalogue and verify operation class/method explanations are stored."""
    repository_root = Path(__file__).resolve().parents[1]
    database = LoggerDatabase(tmp_path / "annotations.sqlite3")
    report = sync_repository(database, repository_root)
    assert isinstance(report, dict)
    class_annotation = database.get_annotation(
        "ore_comprehension.pipeline",
        "ComprehensionProcessor",
        "class",
        "ore_comprehension/pipeline.py",
    )
    method_annotation = database.get_annotation(
        "ore_comprehension.pipeline",
        "ComprehensionProcessor.op_19_calculate_surprisal",
        "function",
        "ore_comprehension/pipeline.py",
    )
    assert class_annotation is not None
    assert "38" in class_annotation["annotation"]
    assert method_annotation is not None
    assert "log2" in method_annotation["annotation"]
    assert "Biological analogy" in method_annotation["annotation"]


@pytest.mark.parametrize("bad_probability", [-0.1, 1.1])
def test_config_rejects_invalid_bounded_parameters(bad_probability: float) -> None:
    """Ensure bounded configuration values reject impossible probabilities and thresholds."""
    with pytest.raises(ValueError):
        ComprehensionConfig(context_decay=bad_probability)
