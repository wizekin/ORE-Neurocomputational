from __future__ import annotations

import ast
from pathlib import Path
import shutil

import pytest

from ore_logger import LoggerDatabase

from ore_language_comprehension.point08 import (
    POINT8_STAGE_NAMES,
    POINT8_STEP_SPECS,
    Point8CrossStageIntegrator,
    Point8LIFReference,
    Point8Request,
)
from ore_language_comprehension.annotations import register_point08_annotations


def _stage_inputs(confidence: float = 0.95) -> dict:
    """Return a complete, provenance-labelled fixture from sensory input to proposition."""
    proposition = {
        "predicate": "OPEN",
        "semantic_roles": {"agent": "child", "theme": "door"},
        "operators": {"tense": "past"},
        "explicitness": {"agent": "explicit", "theme": "explicit"},
    }
    return {
        "sensory": [{"candidate_id": "s1", "content": "the child opened the door", "confidence": confidence}],
        "lexical": [{
            "candidate_id": "w1", "content": "opened", "confidence": confidence,
            "sensory_score": confidence, "lexical_score": confidence, "context_score": confidence,
        }],
        "syntactic": [{"candidate_id": "g1", "content": "S(NP child, VP opened door)", "confidence": confidence}],
        "semantic": [{
            "candidate_id": "m1", "content": proposition, "confidence": confidence,
            "source": "literal_semantic_composition",
        }],
        "propositions": [{
            "candidate_id": "p1", "content": proposition, "confidence": confidence,
            "source": "explicit_linguistic_input", "status": "supported",
        }],
    }


def test_point8_exposes_eighteen_source_ordered_operations_and_all_feedback_loops() -> None:
    result = Point8CrossStageIntegrator().process(
        Point8Request(
            sequence_id="p8-steps",
            stage_outputs=_stage_inputs(),
            context={"topic": "opening a door"},
        )
    )

    assert len(POINT8_STEP_SPECS) == 18
    assert len(result.functional_trace) == 18
    assert [item["step"] for item in result.functional_trace] == list(range(1, 19))
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["durable_memory_write_performed"] is False
    assert len(result.feedback_loops) == 8
    assert all(item["operation_id"] == POINT8_STEP_SPECS[index]["operation_id"]
               for index, item in enumerate(result.functional_trace))


def test_streaming_state_persists_across_increments_without_restarting_the_pipeline() -> None:
    processor = Point8CrossStageIntegrator()
    first = processor.process(Point8Request(
        sequence_id="stream-1",
        increment={"text": "The child"},
        stage_outputs={
            "sensory": [{"candidate_id": "s1", "content": "The child", "confidence": 0.9}],
            "lexical": [{"candidate_id": "w1", "content": "child", "confidence": 0.9}],
        },
    ))
    second = processor.process(Point8Request(
        sequence_id="stream-2",
        increment={"text": "opened the door"},
        stage_outputs={
            "sensory": [{"candidate_id": "s2", "content": "opened the door", "confidence": 0.95}],
            "lexical": [{"candidate_id": "w2", "content": "opened", "confidence": 0.95}],
            "syntactic": [{"candidate_id": "g1", "content": "subject + verb", "confidence": 0.9}],
            "semantic": [{"candidate_id": "m1", "content": {"predicate": "OPEN"}, "confidence": 0.9}],
            "propositions": [{"candidate_id": "p1", "content": {"predicate": "OPEN"}, "confidence": 0.9}],
        },
    ))

    assert first.time_index == 1
    assert second.time_index == 2
    assert second.state["sensory_candidates"]
    assert second.state["lexical_candidates"]
    assert second.diagnostics["candidate_counts"]["propositions"] == 1


def test_repeated_activation_does_not_inflate_candidate_confidence_without_new_evidence() -> None:
    processor = Point8CrossStageIntegrator()
    payload = {
        "lexical": [{
            "candidate_id": "same-word", "content": "open", "confidence": 0.7,
            "sensory_score": 0.7, "lexical_score": 0.7, "context_score": 0.7,
        }]
    }
    first = processor.process(Point8Request(sequence_id="repeat-1", stage_outputs=payload))
    second = processor.process(Point8Request(sequence_id="repeat-2", stage_outputs=payload))

    assert first.state["lexical_candidates"][0]["confidence"] == pytest.approx(0.7)
    assert second.state["lexical_candidates"][0]["confidence"] == pytest.approx(0.7)


def test_literal_and_inferred_propositions_remain_separate_and_context_conflicts_are_kept() -> None:
    processor = Point8CrossStageIntegrator()
    result = processor.process(Point8Request(
        sequence_id="p8-provenance",
        stage_outputs=_stage_inputs(),
        context={"room": {"temperature": "cold"}, "entities": ["window"]},
        inference_candidates=[{
            "candidate_id": "i1",
            "content": {"speech_act": "request", "action": "close_window"},
            "confidence": 0.95,
            "linguistic_fit": 0.95,
            "context_fit": 0.95,
            "goal_fit": 0.95,
        }],
        task={"pragmatic_inference_required": True},
    ))

    assert result.finalized_interpretation is not None
    assert result.finalized_interpretation["selected_literal_proposition"]["source"] == "explicit_linguistic_input"
    assert result.finalized_interpretation["inferred_propositions"][0]["status"] == "inferred"
    assert result.finalized_interpretation["provenance_policy"] == "literal_and_inferred_content_remain_separate"


def test_prediction_errors_are_typed_and_attention_allocation_respects_budget() -> None:
    result = Point8CrossStageIntegrator().process(Point8Request(
        sequence_id="p8-errors",
        stage_outputs=_stage_inputs(),
        predictions={"sensory": "clear", "pragmatic": "literal"},
        observed_evidence={"sensory": "noisy", "pragmatic": "request"},
        task={"control_budget": 0.6},
    ))

    assert result.state["prediction_errors"]["sensory"] == 1.0
    assert result.state["prediction_errors"]["pragmatic"] == 1.0
    allocations = result.state["attention_allocation"]
    assert allocations
    assert sum(item["allocated_attention"] for item in allocations) == pytest.approx(0.6)
    assert {item["kind"] for item in allocations} >= {"sensory", "pragmatic"}


def test_revision_requires_new_evidence_to_clear_the_margin() -> None:
    weak = Point8CrossStageIntegrator(revision_margin=0.2).process(Point8Request(
        sequence_id="p8-weak-revision",
        stage_outputs={"semantic": [{"candidate_id": "m1", "content": "old sense", "confidence": 0.8}]},
        revision_candidates=[{
            "stage": "semantic", "candidate_id": "m2", "content": "new sense",
            "confidence": 0.6, "evidence_fit": 0.5,
        }],
    ))
    strong_processor = Point8CrossStageIntegrator(revision_margin=0.08)
    strong = strong_processor.process(Point8Request(
        sequence_id="p8-strong-revision",
        stage_outputs={"semantic": [{"candidate_id": "m1", "content": "old sense", "confidence": 0.5}]},
        revision_candidates=[{
            "stage": "semantic", "candidate_id": "m2", "content": "new sense",
            "confidence": 0.99, "evidence_fit": 0.99,
        }],
    ))

    assert weak.state["semantic_candidates"][0]["candidate_id"] == "m1"
    assert strong.state["semantic_candidates"][0]["candidate_id"] == "m2"
    assert strong.state["revision_log"]


def test_pragmatic_inference_is_not_engaged_for_an_ordinary_literal_sentence() -> None:
    result = Point8CrossStageIntegrator().process(Point8Request(
        sequence_id="p8-literal",
        stage_outputs=_stage_inputs(),
        inference_candidates=[{
            "candidate_id": "social1", "content": "unstated intention",
            "confidence": 0.99, "linguistic_fit": 0.99, "context_fit": 0.99, "goal_fit": 0.99,
        }],
        task={"social_inference_demand": 0.1},
    ))

    assert result.state["inferred_propositions"] == []
    assert result.functional_trace[10]["output"]["social_inference_engaged"] is False


def test_working_memory_is_bounded_and_keeps_pointers_to_deprioritized_items() -> None:
    propositions = [
        {"candidate_id": f"p{i}", "content": {"predicate": f"EVENT_{i}"}, "confidence": 1 - i * 0.1}
        for i in range(5)
    ]
    result = Point8CrossStageIntegrator(working_memory_capacity=2).process(Point8Request(
        sequence_id="p8-wm",
        stage_outputs={"propositions": propositions},
    ))

    assert len(result.state["working_memory"]) == 2
    assert len(result.state["deprioritized_ids"]) == 3
    assert set(result.state["deprioritized_ids"]).isdisjoint(
        item["candidate_id"] for item in result.state["working_memory"]
    )


def test_inadequate_interpretation_is_not_forced_into_a_final_proposition() -> None:
    result = Point8CrossStageIntegrator().process(Point8Request(
        sequence_id="p8-incomplete",
        stage_outputs={"propositions": [{
            "candidate_id": "partial", "content": {"predicate": "OPEN", "theme": None}, "confidence": 0.4,
        }]},
        unresolved=[{"issue_id": "missing-theme", "critical": True, "uncertainty": 0.9}],
    ))

    assert result.finalized_interpretation is None
    assert result.interpretation_quality["status"] == "continue_or_reanalyse"


def test_downstream_routes_prepare_a_point7_handoff_without_writing_memory() -> None:
    result = Point8CrossStageIntegrator().process(Point8Request(
        sequence_id="p8-routing",
        stage_outputs=_stage_inputs(),
        task={
            "requires_answer": True,
            "requires_action_plan": True,
            "encode_episode": True,
            "generalizable_knowledge": True,
        },
    ))

    handoff = result.routing["memory_handoff"]
    assert set(result.routing["routes"]) == {
        "discourse_continuation", "question_answering", "decision_support",
        "episodic_memory", "semantic_memory_candidate",
    }
    assert handoff["target"] == "Point7MemoryDecisionProcessor"
    assert handoff["durable_storage_performed"] is False
    assert handoff["task"]["semantic_verified"] is False


def test_later_discourse_can_revise_an_earlier_finalized_proposition_and_memory_handoff() -> None:
    revised = {
        "predicate": "OPEN",
        "semantic_roles": {"agent": "child", "theme": "window"},
    }
    result = Point8CrossStageIntegrator(revision_margin=0.02).process(Point8Request(
        sequence_id="p8-later-revision",
        stage_outputs=_stage_inputs(),
        task={"encode_episode": True},
        later_evidence={"revision_candidates": [{
            "stage": "proposition",
            "candidate_id": "p2",
            "content": revised,
            "confidence": 0.99,
            "evidence_fit": 0.99,
        }]},
    ))

    assert result.state["revision_log"]
    assert result.finalized_interpretation["selected_literal_proposition"]["content"] == revised
    assert result.routing["memory_handoff"]["proposition"] == revised
    assert result.functional_trace[15]["output"]["interpretation_rechecked"] is True


def test_optional_lif_reference_tracks_ten_populations_and_eighteen_operations() -> None:
    result = Point8LIFReference(population_size=3, steps_per_operation=1).run()

    assert result["operation_count"] == 18
    assert len(result["population_names"]) == len(POINT8_STAGE_NAMES) == 10
    assert len(result["trace"]) == 18
    assert result["biological_reference_is_measured_human_data"] is False


def test_stage_result_factory_maps_prior_points_into_shared_channels() -> None:
    request = Point8Request.from_stage_results(
        "p8-factory",
        {
            "point3": {"parse_hypotheses": [{"candidate_id": "g1", "content": "parse"}]},
            "point4": {"propositions": [{"candidate_id": "p1", "content": {"predicate": "HELP"}}]},
            "point5": {"enriched_interpretation": {"situation_model": {"events": ["help"]}}},
        },
    )
    result = Point8CrossStageIntegrator().process(request)

    assert "syntactic" in request.stage_outputs
    assert "semantic" in request.stage_outputs
    assert "discourse" in request.stage_outputs
    assert result.state["syntactic_hypotheses"]
    assert result.state["propositions"]
    assert result.state["discourse_model"]["situation_model"] == {"events": ["help"]}


def test_reset_clears_stream_state_but_not_processor_configuration() -> None:
    processor = Point8CrossStageIntegrator(max_candidates=3)
    processor.process(Point8Request(
        sequence_id="p8-reset",
        stage_outputs={"lexical": [{"candidate_id": "w1", "content": "word", "confidence": 0.9}]},
    ))
    processor.reset()

    assert processor.state.time_index == 0
    assert processor.state.lexical_candidates == []
    assert processor.max_candidates == 3


def test_every_point8_class_and_function_has_a_docstring() -> None:
    module_path = Path(__file__).resolve().parents[1] / "ore_language_comprehension" / "point08.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    missing = [
        f"{node.name}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and ast.get_docstring(node) is None
    ]
    assert missing == []



def test_point8_annotations_are_indexed_in_the_existing_sqlite_catalogue(tmp_path: Path) -> None:
    """Index Point 8's classes and methods in the same annotation database used by earlier points."""
    source_root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "ore-project"
    isolated.mkdir()
    shutil.copytree(source_root / "ore_logger", isolated / "ore_logger")
    shutil.copytree(source_root / "ore_language_comprehension", isolated / "ore_language_comprehension")
    (isolated / "pyproject.toml").write_text("[project]\nname='point8-annotation-test'\n", encoding="utf-8")
    database_path = tmp_path / "point8-annotations.sqlite3"

    report = register_point08_annotations(isolated, database_path)
    rows = LoggerDatabase(database_path).search_annotations("ore_language_comprehension", limit=2000)

    assert report["symbols_written"] > 0
    assert report["database_path"] == str(database_path)
    assert any(row["qualified_name"] == "Point8CrossStageIntegrator.process" for row in rows)
    assert any(row["qualified_name"] == "Point8Request.from_stage_results" for row in rows)
    assert all(row["source_docstring"] for row in rows)
