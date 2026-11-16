"""Acceptance and regression tests for the 18-stage Controlled Semantic Cognition package."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ore_logger import LoggerDatabase
from ore_semantic.contracts import SemanticConcept

from ore_csc import (
    ALL_OPERATIONS,
    STAGE_REGISTRY,
    CSCConfig,
    CSCProcessor,
    CSCRequest,
    index_csc_annotations,
    record_csc_result,
)


def _concepts() -> list[SemanticConcept]:
    """Create competing financial and geographical senses with deliberately unequal priors."""
    return [
        SemanticConcept(
            concept_id="financial-bank",
            label="financial bank",
            aliases=("bank", "financial institution"),
            semantic_features={
                "FINANCIAL_INSTITUTION": 1.0, "MONEY": 0.95,
                "TRANSACTIONS": 0.9, "CUSTOMERS": 0.8, "BUILDING": 0.6,
            },
            frequency=100.0,
            verified_by="test",
            source_reference="test://bank/financial",
        ),
        SemanticConcept(
            concept_id="river-bank",
            label="river bank",
            aliases=("bank", "riverbank", "shore"),
            semantic_features={
                "RIVER": 1.0, "WATER": 0.95,
                "SHORE": 0.9, "LANDSCAPE": 0.8, "OUTDOORS": 0.7,
            },
            frequency=8.0,
            verified_by="test",
            source_reference="test://bank/river",
        ),
    ]


def _request(sequence_id: str = "bank-001") -> CSCRequest:
    """Create a sentence where river and water context should override the frequent financial sense."""
    return CSCRequest(
        sequence_id=sequence_id,
        text="The fisherman sat beside the bank.",
        task="interpret the geographical meaning",
        context_features={"RIVER": 0.95, "WATER": 0.9, "FISHERMAN": 0.8},
        task_features={"RIVER": 0.9, "WATER": 0.9, "SHORE": 0.85},
    )


def _processor(simulate_biology: bool = False) -> CSCProcessor:
    """Create a deterministic processor with short reference dynamics for focused tests."""
    return CSCProcessor(
        _concepts(),
        CSCConfig(
            simulate_biology=simulate_biology,
            max_control_iterations=2,
            steps_per_stage=1,
            population_size=4,
            random_seed=5,
        ),
    )


def test_registry_has_18_stages_and_complete_unique_operation_ids() -> None:
    """Verify the exact CSC stage sequence and stable hierarchical operation registry."""
    assert len(STAGE_REGISTRY) == 18
    assert [stage.stage_id for stage in STAGE_REGISTRY] == [
        "CSC-{0:02d}".format(index) for index in range(1, 19)
    ]
    ids = [operation.operation_id for operation in ALL_OPERATIONS]
    assert len(ids) == len(set(ids))
    assert len(ids) == 78
    assert all(operation.mathematical_operation for operation in ALL_OPERATIONS)
    assert all(stage.biological_reference for stage in STAGE_REGISTRY)


def test_end_to_end_context_overrides_stronger_frequency_prior() -> None:
    """Confirm semantic control selects the contextually appropriate river sense."""
    result = _processor().process(_request())
    assert result.diagnostics["valid"] is True
    assert len(result.stage_trace) == 18
    assert len(result.operation_trace) == len(ALL_OPERATIONS)
    assert result.selected_concept is not None
    assert result.selected_concept["concept_id"] == "river-bank"
    assert result.response["emitted"] is True
    assert result.stage_outputs["CSC-14"]["alternatives_preserved"]
    assert result.stage_outputs["CSC-11"]["knowledge_deleted"] is False


def test_ambiguous_candidates_are_preserved_and_selection_can_abstain() -> None:
    """Check equal-evidence alternatives remain visible when confidence is insufficient."""
    concepts = [
        SemanticConcept(
            concept_id="sense-a", label="bank", aliases=("bank",),
            semantic_features={"SHARED": 1.0, "ENTITY": 0.7}, frequency=10.0,
            source_reference="test://sense-a",
        ),
        SemanticConcept(
            concept_id="sense-b", label="bank", aliases=("bank",),
            semantic_features={"SHARED": 1.0, "ENTITY": 0.7}, frequency=10.0,
            source_reference="test://sense-b",
        ),
    ]
    processor = CSCProcessor(
        concepts,
        CSCConfig(simulate_biology=False, confidence_threshold=0.7, selection_margin_threshold=0.1),
    )
    result = processor.process(CSCRequest(
        sequence_id="ambiguous",
        text="bank",
        task="comprehension",
    ))
    assert result.selected_concept is None
    assert result.response["emitted"] is False
    assert len(result.candidate_scores) == 2
    assert result.diagnostics["selection_accepted"] is False


def test_biological_reference_reports_18_finite_stage_snapshots() -> None:
    """Verify the optional neural reference advances all stages and reports explicit units."""
    result = _processor(simulate_biology=True).process(_request("lif-bank"))
    assert result.biological["enabled"] is True
    assert result.biological["stage_count"] == 18
    assert result.biological["numerically_valid"] is True
    assert result.biological["units"]["membrane_voltage"] == "mV"
    assert result.biological["units"]["synaptic_current"] == "pA"
    assert result.biological["units"]["firing_rate"] == "Hz"
    assert len(result.biological["operation_trace"]) == 18


def test_learning_is_preview_only_without_authorization_and_requires_feedback() -> None:
    """Ensure temporary control changes do not silently mutate long-term associations."""
    processor = _processor()
    preview = processor.process(_request("preview"))
    assert preview.learning["committed"] is False
    assert preview.learning["long_term_memory_changed"] is False

    with pytest.raises(ValueError):
        processor.process(_request("invalid-learning"), learning_authorized=True)

    feedback_request = CSCRequest(
        sequence_id="authorized-learning",
        text="The fisherman sat beside the bank.",
        task="geographical sense",
        context_features={"RIVER": 0.95, "WATER": 0.9},
        task_features={"RIVER": 0.9, "SHORE": 0.85},
        feedback_concept_id="river-bank",
    )
    learned = processor.process(
        feedback_request,
        learning_authorized=True,
        feedback_reward=0.9,
    )
    assert learned.learning["committed"] is True
    assert learned.learning["learning_generation"] == 1
    assert learned.learning["feedback_concept_id"] == "river-bank"
    assert learned.learning["error_gradient"] != 0.0
    assert learned.learning["control_bias_delta"] > 0.0


def test_signal_fft_and_context_task_ema_are_executed_and_reported() -> None:
    """Verify optional numeric signal samples, temporal input, and history EMAs reach the hub."""
    processor = _processor()
    request = CSCRequest(
        sequence_id="signal-history",
        text="bank",
        task="geographical interpretation",
        context_features={"WATER": 0.2},
        task_features={"SHORE": 0.3},
        signal_samples=(0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0),
        signal_modality="auditory",
        previous_context_features={"RIVER": 1.0},
        previous_task_features={"RIVER": 1.0},
    )
    result = processor.process(request)
    first = result.stage_outputs["CSC-01"]
    fourth = result.stage_outputs["CSC-04"]
    fifth = result.stage_outputs["CSC-05"]
    sixth = result.stage_outputs["CSC-06"]

    assert first["fourier_bins"] == 5
    assert first["z_score_normalization"] is True
    assert "auditory" in fourth["spoke_names"]
    assert any(key.startswith("spectral_bin_") for key in fourth["hub_state"])
    assert fifth["previous_context_supplied"] is True
    assert fifth["context_features"]["river"] == pytest.approx(0.65)
    assert fifth["context_features"]["water"] == pytest.approx(0.07)
    assert sixth["previous_task_supplied"] is True
    assert sixth["task_features"]["river"] == pytest.approx(0.70)
    assert sixth["task_features"]["shore"] == pytest.approx(0.09)


def test_result_serialization_contains_full_stage_and_operation_traces() -> None:
    """Ensure result data can be serialized without NumPy or non-finite objects."""
    result = _processor().process(_request("json"))
    encoded = json.dumps(result.to_dict(), ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["schema"] == "ore.csc.18_stage.v1"
    assert len(decoded["stage_trace"]) == 18
    assert len(decoded["operation_trace"]) == 78


def test_logger_indexes_every_csc_class_and_function_docstring(tmp_path: Path) -> None:
    """Verify package symbols are persisted in the shared SQLite annotation table."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    report = index_csc_annotations(root, database_path)
    assert report["csc_symbol_count"] >= 20
    database = LoggerDatabase(database_path)
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT qualified_name, symbol_kind, annotation, source_docstring
            FROM annotations
            WHERE source_path LIKE 'ore_csc/%'
            """
        ).fetchall()
    assert any(row["qualified_name"] == "CSCProcessor" and row["symbol_kind"] == "class" for row in rows)
    assert any(row["qualified_name"] == "CSCProcessor.process" and row["symbol_kind"] == "function" for row in rows)
    assert all(row["annotation"].strip() for row in rows)


def test_logger_records_operation_events_and_reuses_symbol_annotations(tmp_path: Path) -> None:
    """Write a CSC run through the existing ORE event logger and verify linked annotations exist."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_events.sqlite3"
    result = _processor().process(_request("logger"))
    report = record_csc_result(result, root, database_path)
    assert report["operation_count"] == 78
    assert report["stage_count"] == 18
    database = LoggerDatabase(database_path)
    with database.connect() as connection:
        event_count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        annotation_count = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE source_path LIKE 'ore_csc/%'"
        ).fetchone()[0]
    assert event_count >= 78
    assert annotation_count >= 20
