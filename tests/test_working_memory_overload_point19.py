"""Focused acceptance tests for the Point 19 working-memory overload implementation."""
from __future__ import annotations

import sqlite3

import pytest

from ore_working_memory_overload import (
    Dependency, Point19Config, Point19Request, Point19State,
    Point19WorkingMemoryOverloadProcessor, Representation,
    OPERATION_REGISTRY, allocate_attention, create_dependencies,
    estimate_interference, feature_similarity, form_chunks, retrieve_by_cue,
    segment_units, softmax,
)


def test_registry_has_all_nineteen_ordered_operations() -> None:
    """Keep source order and the complete nineteen-step contract stable."""
    assert len(OPERATION_REGISTRY) == 19
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P19-{index:02d}" for index in range(1, 20)
    ]


def test_segmentation_and_softmax_are_stable() -> None:
    """Token boundaries are retained and numerical normalization remains finite."""
    assert segment_units("Maya, who arrived, smiled.") == [
        "Maya", ",", "who", "arrived", ",", "smiled", "."
    ]
    probabilities = softmax([10000.0, 9999.0, -10000.0], temperature=0.5)
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(0.0 <= item <= 1.0 for item in probabilities)


def test_request_requires_exactly_one_input_representation() -> None:
    """Reject an ambiguous request before any processor state is changed."""
    with pytest.raises(ValueError, match="exactly one"):
        Point19Request(sequence_id="bad", text="hello", units=["hello"])
    with pytest.raises(ValueError, match="exactly one"):
        Point19Request(sequence_id="bad")


def test_feature_similarity_respects_categories_and_numeric_vectors() -> None:
    """Support categorical identity and graded numeric overlap as distinct features."""
    assert feature_similarity({"role": "subject"}, {"role": "subject"}) == 1.0
    assert feature_similarity({"role": "subject"}, {"role": "object"}) == 0.0
    assert feature_similarity({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0}) == 1.0


def test_chunks_only_group_explicitly_related_items() -> None:
    """Do not guess chunk boundaries when no schema or caller annotation supplies them."""
    state = Point19State()
    state.representations["a"] = Representation("a", "New", {}, 0, chunk_id="np1")
    state.representations["b"] = Representation("b", "York", {}, 1, chunk_id="np1")
    state.representations["c"] = Representation("c", "arrived", {}, 2)
    result = form_chunks(state)
    assert result["chunk_count"] == 1
    assert result["chunks"][0]["member_ids"] == ["a", "b"]
    assert result["chunks"][0]["reversible"] is True


def test_retrieval_does_not_treat_a_lone_weak_candidate_as_certain() -> None:
    """Relative softmax probability must be combined with absolute evidence strength."""
    config = Point19Config(simulate_biology=False, retrieval_threshold=0.62)
    dependency = Dependency("d1", {"syntactic_role": "subject"})
    weak = Representation("x", "unrelated", {"semantic_type": "place"}, 0)
    result = retrieve_by_cue(dependency, [weak], config)
    assert result["candidates"][0]["probability"] == pytest.approx(1.0)
    assert result["selected"] is None
    assert result["confidence"] < config.retrieval_threshold


def test_processor_emits_all_operations_and_separates_biological_proxy() -> None:
    """Run a small example through all nineteen steps with explicit dependency evidence."""
    request = Point19Request(
        sequence_id="p19-test-001",
        units=["Maya", "arrived", "."],
        unit_annotations=[
            {"features": {"syntactic_role": "subject", "referent": "Maya"}},
            {"features": {"syntactic_role": "predicate", "event": "arrive"}},
            {"features": {"punctuation": "period"}},
        ],
        dependencies=[{
            "identifier": "subject-dependency",
            "required_features": {"syntactic_role": "subject"},
            "cue_features": {"referent": "Maya"},
        }],
        goals={"identify_agent": 1.0, "understand_event": 0.8},
        context={"context_support": 0.6, "semantic_coherence": 0.9},
    )
    processor = Point19WorkingMemoryOverloadProcessor(Point19Config(simulate_biology=True))
    result = processor.process(request)

    assert len(result.operation_trace) == 19
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"P19-{index:02d}" for index in range(1, 20)
    ]
    assert result.operation_trace[9]["functional_output"]["retrieval_count"] == 1
    assert result.operation_trace[10]["functional_output"]["integrated_count"] == 1
    assert result.biological["operation_count"] == 19
    assert result.interpretation["durable_episodic_storage_performed"] is False
    assert result.metadata["source_sequence_steps"] == 19


def test_disabled_biology_still_runs_functional_track() -> None:
    """The biological proxy is optional and cannot suppress the functional algorithm."""
    processor = Point19WorkingMemoryOverloadProcessor(Point19Config(simulate_biology=False))
    result = processor.process(Point19Request(sequence_id="p19-disabled", text="A short sentence."))
    assert len(result.operation_trace) == 19
    assert result.biological["model"] == "disabled"
    assert all(row["biological_output"]["model"] == "disabled" for row in result.operation_trace)


def test_processor_retains_unknowns_instead_of_inventing_missing_relations() -> None:
    """An unresolved dependency remains explicit when no supported candidate can be retrieved."""
    request = Point19Request(
        sequence_id="p19-unknown",
        units=["word"],
        unit_annotations=[{"features": {"semantic_type": "object"}}],
        dependencies=[{
            "identifier": "missing-agent",
            "required_features": {"syntactic_role": "agent"},
        }],
    )
    result = Point19WorkingMemoryOverloadProcessor(
        Point19Config(simulate_biology=False)
    ).process(request)
    assert result.interpretation["unresolved"]
    assert result.interpretation["inferred"] == []
    assert result.interpretation["durable_episodic_storage_performed"] is False


def test_interference_is_bounded_and_attention_is_normalized() -> None:
    """Competition and attention distributions remain within their mathematical domains."""
    state = Point19State()
    state.representations["one"] = Representation("one", "he", {"person": "A"}, 0)
    state.representations["two"] = Representation("two", "him", {"person": "A"}, 1)
    interference = estimate_interference(state)
    attention = allocate_attention(state, temperature=0.5)
    assert 0.0 <= interference["mean_interference"] <= 1.0
    assert attention["attention_sum"] == pytest.approx(1.0)


def test_point19_symbols_are_registered_in_logger_sqlite_when_available(tmp_path) -> None:
    """AST catalogue synchronization persists package classes and functions in annotations."""
    from ore_working_memory_overload import register_annotations

    pytest.importorskip("ore_logger")
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    report = register_annotations(database_path, repository_root=root)
    assert report["files_indexed"] >= 1
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE module_name LIKE 'ore_working_memory_overload%'"
        ).fetchone()[0]
    assert rows >= 20  # module + classes + functions + methods


def test_result_logging_persists_nineteen_events_when_logger_is_available(tmp_path) -> None:
    """The result logger writes every step through the existing logger run/event tables."""
    from ore_working_memory_overload import record_result_to_sqlite

    pytest.importorskip("ore_logger")
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    result = Point19WorkingMemoryOverloadProcessor(
        Point19Config(simulate_biology=False)
    ).process(Point19Request(sequence_id="p19-log", text="Maya smiled."))
    database_path = tmp_path / "ore_logger.sqlite3"
    report = record_result_to_sqlite(result, database_path, repository_root=root)
    assert report["database_summary"]["events"] == 19
    with sqlite3.connect(database_path) as connection:
        event_count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=?", (report["run_id"],)
        ).fetchone()[0]
        status = connection.execute(
            "SELECT status FROM runs WHERE run_id=?", (report["run_id"],)
        ).fetchone()[0]
    assert event_count == 19
    assert status == "completed"
