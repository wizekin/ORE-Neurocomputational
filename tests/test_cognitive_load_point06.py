"""Focused acceptance tests for the Point 06 cognitive-load algorithm."""
from __future__ import annotations

import json
import math

import pytest

from ore_cognitive_load import (
    OPERATION_REGISTRY,
    Point06CognitiveLoadProcessor,
    Point06Config,
    Point06Request,
    Point06State,
    estimate_element_interactivity,
    feature_similarity,
    form_meaningful_chunks,
    monitor_overload,
    order_by_prerequisites,
    reduce_presentation_demands,
    segment_material,
    softmax,
    update_retrieval_and_support,
    verify_comprehension,
)


def test_registry_contains_all_sixteen_source_steps_in_order() -> None:
    """Preserve all sixteen operations and their source sequence identifiers."""
    assert len(OPERATION_REGISTRY) == 16
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P06-{index:02d}" for index in range(1, 17)
    ]
    assert [item.source_step for item in OPERATION_REGISTRY] == list(range(1, 17))


def test_request_requires_exactly_one_material_representation() -> None:
    """Reject ambiguous or empty material before processing begins."""
    with pytest.raises(ValueError, match="exactly one"):
        Point06Request(sequence_id="bad", goal="learn", text="x", units=["x"])
    with pytest.raises(ValueError, match="exactly one"):
        Point06Request(sequence_id="bad", goal="learn")
    with pytest.raises(ValueError, match="non-empty"):
        Point06Request(sequence_id="bad", goal="learn", units=[])


def test_softmax_is_stable_and_normalized() -> None:
    """Normalize extreme utilities without overflow or invalid probabilities."""
    values = softmax([10000.0, 9999.0, -10000.0], temperature=0.5)
    assert sum(values) == pytest.approx(1.0)
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
    assert softmax([], 1.0) == []


def test_sentence_segmentation_preserves_meaningful_source_units() -> None:
    """Split source text at clear sentence and paragraph boundaries."""
    request = Point06Request(
        sequence_id="segmentation", goal="understand learning", text="First idea. Second idea.\nThird idea."
    )
    units = segment_material(request)
    assert [unit.text for unit in units] == ["First idea.", "Second idea.", "Third idea."]
    assert [unit.identifier for unit in units] == ["u0001", "u0002", "u0003"]


def test_similarity_uses_explicit_lexical_overlap_only() -> None:
    """Keep the similarity implementation inspectable and avoid pretending to use embeddings."""
    assert feature_similarity(["brain", "learning"], ["learning", "brain", "memory"]) == pytest.approx(2 / 3)
    assert feature_similarity([], ["brain"]) == 0.0


def test_prior_knowledge_reduces_modelled_effective_complexity() -> None:
    """Apply the stated familiarity adjustment while keeping it a configurable model."""
    cfg = Point06Config(simulate_biology=False)
    request = Point06Request(
        sequence_id="complexity", goal="cell energy", units=["Cellular energy uses ATP.", "ATP stores energy."],
        unit_metadata={"u0001": {"intrinsic_load": 0.8}, "u0002": {"intrinsic_load": 0.8}},
        dependencies=[{"source": "u0001", "target": "u0002", "strength": 0.8}],
        prior_knowledge=["cellular energy ATP"],
    )
    units = segment_material(request)
    edges = [{"source": "u0001", "target": "u0002", "strength": 0.8}]
    without_familiarity = estimate_element_interactivity(units, edges, cfg)
    familiar_units = [type(unit)(**{**unit.to_dict(), "familiarity": 1.0}) for unit in units]
    with_familiarity = estimate_element_interactivity(familiar_units, edges, cfg)
    assert with_familiarity["effective_complexity"] < without_familiarity["effective_complexity"]
    assert with_familiarity["raw_complexity"] == pytest.approx(without_familiarity["raw_complexity"])


def test_prerequisite_order_and_cycles_are_explicit() -> None:
    """Use a stable topological order and report cyclic dependencies instead of hiding them."""
    request = Point06Request(sequence_id="graph", goal="learn", units=["A", "B", "C"])
    units = segment_material(request)
    acyclic = order_by_prerequisites(units, [
        {"source": "u0001", "target": "u0003"},
        {"source": "u0002", "target": "u0003"},
    ])
    assert acyclic["ordered_unit_ids"].index("u0001") < acyclic["ordered_unit_ids"].index("u0003")
    cyclic = order_by_prerequisites(units, [
        {"source": "u0001", "target": "u0002"},
        {"source": "u0002", "target": "u0001"},
    ])
    assert set(cyclic["cyclic_unit_ids"]) == {"u0001", "u0002"}


def test_chunking_keeps_unrelated_neighboring_units_separate() -> None:
    """Do not form chunks merely because information is adjacent in the input."""
    request = Point06Request(
        sequence_id="chunks", goal="learn", units=["The hippocampus supports memory.", "A triangle has three sides."]
    )
    units = segment_material(request)
    result = form_meaningful_chunks(units, [], threshold=0.5)
    assert result["chunk_count"] == 2
    assert all(chunk["reversible"] for chunk in result["chunks"])


def test_presentation_costs_produce_targeted_redesign_suggestions() -> None:
    """Identify avoidable presentation cost without deleting essential subject content."""
    result = reduce_presentation_demands({"split_attention": 0.9, "ambiguity": 0.8})
    assert result["avoidable_load"] > 0.0
    assert result["essential_content_preserved"] is True
    assert len(result["redesign_recommendations"]) == 2


def test_monitor_distinguishes_productive_effort_from_overload() -> None:
    """High reported effort with high accuracy should not automatically count as overload."""
    productive = monitor_overload(
        {"subjective_effort": 0.95, "accuracy": 0.95, "error_rate": 0.05}, 0.3, 0.4
    )
    breakdown = monitor_overload(
        {"subjective_effort": 0.9, "accuracy": 0.2, "error_rate": 0.9, "reread_rate": 0.8}, 0.8, 0.4
    )
    assert productive["productive_effort"] is True
    assert productive["overload_likely"] is False
    assert breakdown["overload_likely"] is True


def test_support_fades_only_after_repeated_successful_retrieval() -> None:
    """Keep scaffolding unchanged without evidence and fade only after repeated success."""
    state = Point06State(support_level=0.8)
    cfg = Point06Config(simulate_biology=False)
    no_evidence = update_retrieval_and_support(state, {}, cfg)
    first_success = update_retrieval_and_support(state, {"reconstruction_accuracy": 0.9}, cfg)
    second_success = update_retrieval_and_support(state, {"reconstruction_accuracy": 0.9}, cfg)
    assert no_evidence["support_faded"] is False
    assert first_success["support_faded"] is False
    assert second_success["support_faded"] is True
    assert state.support_level < 0.8


def test_comprehension_verification_does_not_infer_missing_evidence() -> None:
    """Report missing assessment dimensions rather than filling them with false success values."""
    missing = verify_comprehension({})
    partial = verify_comprehension({"local_accuracy": 0.8})
    assert missing["status"] == "not_assessed"
    assert missing["score"] is None
    assert partial["status"] == "partially_assessed"
    assert partial["score"] == pytest.approx(0.8)


def test_processor_runs_all_steps_and_separates_functional_and_biological_outputs() -> None:
    """Execute a complete source-ordered episode and verify the separate biological proxy."""
    request = Point06Request(
        sequence_id="p06-main", goal="understand the causal stages of cellular respiration",
        units=["Glycolysis breaks down glucose.",
               "Electron carriers transport high-energy electrons.",
               "The electron transport chain builds a proton gradient.",
               "ATP synthase uses the gradient to produce ATP."],
        dependencies=[
            {"source": "u0001", "target": "u0002", "relation": "precedes"},
            {"source": "u0002", "target": "u0003", "relation": "supports"},
            {"source": "u0003", "target": "u0004", "relation": "enables"},
        ],
        prior_knowledge=["ATP is a cellular energy carrier."],
        presentation_costs={"split_attention": 0.8, "ambiguity": 0.2},
        metrics={"local_accuracy": 0.8, "relation_accuracy": 0.7,
                 "transfer_accuracy": 0.6, "explanation_accuracy": 0.8,
                 "reconstruction_accuracy": 0.85},
    )
    processor = Point06CognitiveLoadProcessor(Point06Config(simulate_biology=True))
    result = processor.process(request)
    assert len(result.operation_trace) == 16
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"P06-{index:02d}" for index in range(1, 17)
    ]
    assert result.operation_trace[0]["functional_output"]["selected_unit_id"]
    assert result.biological["operation_count"] == 16
    assert result.metadata["source_sequence_steps"] == 16
    assert result.comprehension["status"] == "assessed"
    assert result.external_representation["relation_invention"] is False
    assert all("mathematical_form" in row and "biological_reference" in row for row in result.operation_trace)
    encoded = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "ore.cognitive_load.point06.1.0" in encoded


def test_disabled_biology_does_not_disable_functional_processing() -> None:
    """Run all functional operations even when the optional neural reference model is disabled."""
    result = Point06CognitiveLoadProcessor(
        Point06Config(simulate_biology=False)
    ).process(Point06Request(sequence_id="p06-no-bio", goal="learn", text="A concise explanation."))
    assert len(result.operation_trace) == 16
    assert result.biological["model"] == "disabled"
    assert all(row["biological_output"]["model"] == "disabled" for row in result.operation_trace)


def test_reject_unknown_dependency_identifiers_before_processing() -> None:
    """Refuse edges whose endpoints are absent from the supplied learning material."""
    request = Point06Request(
        sequence_id="bad-edge", goal="learn", units=["A", "B"],
        dependencies=[{"source": "missing", "target": "u0002"}],
    )
    with pytest.raises(ValueError, match="unknown unit"):
        Point06CognitiveLoadProcessor().process(request)


def test_sqlite_annotations_and_event_logging_when_repository_logger_is_available(tmp_path) -> None:
    """Store module, class, function, and method annotations plus all sixteen operation events."""
    sqlite3 = pytest.importorskip("sqlite3")
    pytest.importorskip("ore_logger")
    from pathlib import Path
    from ore_cognitive_load import record_result_to_sqlite, register_annotations

    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    registration = register_annotations(database_path, repository_root=root)
    assert registration["files_indexed"] >= 1
    with sqlite3.connect(database_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE module_name LIKE 'ore_cognitive_load%'"
        ).fetchone()[0]
    assert count >= 20
    result = Point06CognitiveLoadProcessor(Point06Config(simulate_biology=False)).process(
        Point06Request(sequence_id="sqlite-run", goal="learn", text="One idea. A second idea.")
    )
    report = record_result_to_sqlite(result, database_path, repository_root=root)
    assert report["database_summary"]["events"] == 16
    with sqlite3.connect(database_path) as connection:
        event_count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=?", (report["run_id"],)
        ).fetchone()[0]
        status = connection.execute(
            "SELECT status FROM runs WHERE run_id=?", (report["run_id"],)
        ).fetchone()[0]
    assert event_count == 16
    assert status == "completed"