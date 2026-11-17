from __future__ import annotations

import ast
from pathlib import Path
import shutil

import numpy as np
import pytest

from ore_logger import LoggerDatabase
from ore_language_comprehension.annotations import register_point09_annotations
from ore_language_comprehension.point09 import (
    POINT9_DEFAULT_CONNECTIONS,
    POINT9_STEP_SPECS,
    POINT9_SYSTEM_NAMES,
    Point9LIFReference,
    Point9NeuralArchitectureCoordinator,
    Point9Request,
)


def _stage_outputs(confidence: float = 0.95) -> dict:
    """Return a literal, role-bound sentence with separated upstream evidence."""
    proposition = {
        "predicate": "OPEN",
        "semantic_roles": {"agent": "child", "theme": "door"},
        "source": "explicit_linguistic_input",
        "explicitness": {"agent": "explicit", "theme": "explicit"},
    }
    return {
        "perceptual": [{"candidate_id": "p1", "content": "the child opened the door", "confidence": confidence}],
        "lexical": [{"candidate_id": "l1", "content": "opened", "confidence": confidence}],
        "syntactic": [{"candidate_id": "s1", "content": "subject-verb-object", "confidence": confidence}],
        "propositions": [{"candidate_id": "q1", "content": proposition, "confidence": confidence,
                          "source": "explicit_linguistic_input", "status": "supported"}],
    }


def test_point9_exposes_twenty_ordered_network_coordination_operations() -> None:
    result = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-steps",
        stage_outputs=_stage_outputs(),
        context={"topic": "opening a door"},
    ))

    assert len(POINT9_STEP_SPECS) == 20
    assert len(result.functional_trace) == 20
    assert [item["step"] for item in result.functional_trace] == list(range(1, 21))
    assert result.diagnostics["valid"] is True
    assert list(result.state["systems"]) == list(POINT9_SYSTEM_NAMES)
    assert result.diagnostics["biological_reference_is_measured_human_data"] is False


def test_ordinary_literal_sentence_does_not_force_mentalizing_or_high_executive_control() -> None:
    result = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-literal",
        stage_outputs=_stage_outputs(),
        context={"scene": "a child opens a door"},
    ))

    assert result.state["mentalizing_demand"]["engaged"] is False
    assert result.state["systems"]["mentalizing"]["activation"] == 0.0
    assert result.state["executive_demand"]["engaged"] is False
    assert result.state["systems"]["language"]["activation"] > 0.0


def test_explicit_belief_question_recruits_mentalizing_without_merging_literal_state() -> None:
    result = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-belief",
        stage_outputs={
            "propositions": [{
                "candidate_id": "q-belief", "content": {"predicate": "BELIEVE", "agent": "Maya",
                    "content": "meeting cancelled"}, "confidence": 0.95, "source": "literal_statement",
            }]
        },
        context={"meeting": "still scheduled"},
        task={"mentalizing_required": True},
        mental_state_candidates=[{
            "candidate_id": "m-belief", "agent": "Maya", "content": "Maya believes the meeting was cancelled",
            "belief_fit": 0.95, "context_fit": 0.9, "utterance_fit": 0.95,
        }],
    ))

    assert result.state["mentalizing_demand"]["explicit_task_override"] is True
    assert result.state["systems"]["mentalizing"]["activation"] > 0.0
    assert result.interpretation["literal_propositions"][0]["source"] == "literal_statement"
    assert result.interpretation["mental_state_hypotheses"][0]["status"] == "candidate_supported"
    assert result.functional_trace[10]["output"]["representations_merged"] is False


def test_executive_support_is_gated_by_load_and_conflict() -> None:
    easy = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-easy",
        stage_outputs=_stage_outputs(),
    ))
    hard = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-hard",
        stage_outputs={
            **_stage_outputs(),
            "syntactic": [
                {"candidate_id": "parse-a", "content": "parse A", "confidence": 0.6, "conflict": 0.9},
                {"candidate_id": "parse-b", "content": "parse B", "confidence": 0.55, "conflict": 0.9},
                {"candidate_id": "parse-c", "content": "parse C", "confidence": 0.5, "conflict": 0.9},
            ],
        },
        unresolved=[{"issue_id": "open-attachment", "critical": True, "uncertainty": 0.9, "conflict": 0.9}],
        task={"working_memory_load": 1.0, "time_pressure": 1.0, "executive_task_demand": 1.0},
    ))

    assert easy.state["executive_demand"]["engaged"] is False
    assert hard.state["executive_demand"]["engaged"] is True
    assert hard.state["systems"]["multiple_demand"]["activation"] > easy.state["systems"]["multiple_demand"]["activation"]


def test_individual_localizer_calculates_region_selectivity_and_preserves_source() -> None:
    result = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-localizer",
        stage_outputs=_stage_outputs(),
        person_profile={
            "subject_id": "participant-01",
            "lateralization": "left",
            "functional_localizer": {
                "left_IFG": {
                    "language_values": [2.1, 2.3, 2.2, 2.0],
                    "control_values": [0.4, 0.5, 0.45, 0.55],
                    "source": "participant_fmri_localizer",
                },
                "left_MTG": {"language_mean": 1.8, "control_mean": 0.8},
            },
        },
    ))

    assert result.localization["status"] == "participant_specific_estimate"
    assert result.localization["region_scores"]["left_IFG"]["contrast"] == pytest.approx(1.675)
    assert result.localization["region_scores"]["left_IFG"]["standardized_selectivity"] > 1.0
    assert result.localization["region_scores"]["left_IFG"]["evidence_source"] == "participant_fmri_localizer"
    assert result.functional_trace[18]["output"]["demographic_anatomy_inference_performed"] is False


def test_missing_localizer_is_marked_as_approximate_prior_not_individual_measurement() -> None:
    result = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-no-localizer",
        stage_outputs=_stage_outputs(),
        person_profile={"age_group": "adult", "language_history": "not supplied"},
    ))

    assert result.localization["status"] == "approximate_prior_only"
    assert result.diagnostics["participant_localizer_supplied"] is False


def test_sparse_graph_keeps_functional_edges_distinct_from_anatomical_claims() -> None:
    result = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-graph",
        stage_outputs=_stage_outputs(),
        communication_weights={"perceptual->language": 0.9},
    ))

    edge = next(item for item in result.communication_graph["edges"]
                if item["source"] == "perceptual" and item["target"] == "language")
    assert edge["weight"] == pytest.approx(0.9)
    assert "not_anatomical" in edge["edge_type"]
    assert result.communication_graph["sparsity"] > 0.5


def test_custom_system_coupling_changes_the_recurrent_message_weight() -> None:
    result = Point9NeuralArchitectureCoordinator(recurrent_gain=1.0).process(Point9Request(
        sequence_id="p9-coupling",
        stage_outputs=_stage_outputs(),
        communication_weights={"perceptual->language": 0.91},
    ))
    messages = result.functional_trace[9]["output"]["incoming_messages"]["language"]
    edge_message = next(item for item in messages if item["source"] == "perceptual")

    assert edge_message["weight"] == pytest.approx(0.91)
    assert edge_message["weighted_activation"] > 0.0


def test_conceptual_knowledge_is_ranked_but_not_fabricated() -> None:
    no_knowledge = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-no-knowledge",
        stage_outputs=_stage_outputs(),
    ))
    with_knowledge = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-knowledge",
        stage_outputs=_stage_outputs(),
        context={"scene": "a fragile glass object fell and broke"},
        knowledge_candidates=[
            {"candidate_id": "k1", "content": "fragile objects can shatter after impact",
             "confidence": 0.9, "relevance": 0.95, "source": "caller_knowledge_base"},
            {"candidate_id": "k2", "content": "the meeting has been postponed",
             "confidence": 0.9, "relevance": 0.2, "source": "caller_knowledge_base"},
        ],
    ))

    assert no_knowledge.state["knowledge_candidates"] == []
    assert with_knowledge.state["knowledge_candidates"][0]["candidate_id"] == "k1"
    assert with_knowledge.functional_trace[5]["output"]["unprovided_world_knowledge_invented"] is False


def test_prediction_errors_route_to_the_subsystem_that_was_challenged() -> None:
    result = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-errors",
        stage_outputs=_stage_outputs(),
        predictions={"sensory": "clear", "syntactic": "object-attachment", "pragmatic": "literal"},
        observed_evidence={"sensory": "noisy", "syntactic": "clause-attachment", "pragmatic": "sarcastic"},
    ))

    targets = {item["channel"]: item["primary_system"]
               for item in result.functional_trace[13]["output"]["revision_notices"]}
    assert targets == {"sensory": "perceptual", "syntactic": "language", "pragmatic": "mentalizing"}
    assert result.state["systems"]["mentalizing"]["status"] != "active" or result.state["mentalizing_demand"]["engaged"]


def test_functional_connectivity_uses_only_supplied_time_series() -> None:
    t = np.linspace(0.0, 1.0, 10)
    result = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-fc",
        stage_outputs=_stage_outputs(),
        system_time_series={
            "language": t,
            "conceptual_knowledge": 2.0 * t + 1.0,
            "multiple_demand": t[::-1],
        },
    ))

    assert result.functional_connectivity["status"] == "estimated_from_caller_supplied_time_series"
    corr = result.functional_connectivity["correlation"]
    assert corr["language"]["conceptual_knowledge"] == pytest.approx(1.0)
    assert corr["language"]["multiple_demand"] == pytest.approx(-1.0)
    assert "not_directional" in result.functional_connectivity["interpretation"]


def test_connectivity_is_not_estimated_without_enough_measurement_rows() -> None:
    result = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-no-fc",
        stage_outputs=_stage_outputs(),
    ))

    assert result.functional_connectivity["status"] == "not_computed_insufficient_system_series"
    assert result.functional_connectivity["correlation"] == {}


def test_person_profile_can_adjust_gates_without_fabricating_localization() -> None:
    standard = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-standard-gate",
        stage_outputs=_stage_outputs(),
        task={"perspective_dependence": 0.5, "intention_relevance": 0.5},
    ))
    personalized = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-person-gate",
        stage_outputs=_stage_outputs(),
        task={"perspective_dependence": 0.5, "intention_relevance": 0.5},
        person_profile={"mentalizing_threshold": 0.25, "executive_threshold": 0.7},
    ))

    assert personalized.state["mentalizing_demand"]["threshold_used"] == pytest.approx(0.25)
    assert standard.state["mentalizing_demand"]["threshold_used"] == pytest.approx(0.55)
    assert personalized.localization["status"] == "approximate_prior_only"


def test_stage_factory_maps_points_1_to_8_into_specialized_channels() -> None:
    request = Point9Request.from_stage_results(
        "p9-factory",
        {
            "point1": {"form_hypotheses": [{"candidate_id": "p", "content": "signal", "confidence": 0.8}]},
            "point2": {"lexical_candidates": [{"candidate_id": "l", "content": "read", "confidence": 0.9}]},
            "point3": {"parse_hypotheses": [{"candidate_id": "s", "content": "parse", "confidence": 0.9}]},
            "point4": {"propositions": [{"candidate_id": "q", "content": {"predicate": "READ"}, "confidence": 0.95}]},
            "point5": {"enriched_interpretation": {"situation_model": {"entities": ["reader"]}}},
            "point6": {"working_memory_state": {"active_items": [{"candidate_id": "w", "content": "reader"}]}},
            "point7": {"retrieved_memories": [{"candidate_id": "m", "content": "reading event"}]},
            "point8": {"finalized_interpretation": {"literal_propositions": [{"candidate_id": "q", "content": {"predicate": "READ"}}]}},
        },
    )
    result = Point9NeuralArchitectureCoordinator().process(request)

    assert "sensory" in request.stage_outputs
    assert "lexical" in request.stage_outputs
    assert "syntactic" in request.stage_outputs
    assert result.interpretation["literal_propositions"]
    assert result.state["systems"]["working_memory"]["representation"]


def test_downstream_routing_is_preparatory_and_does_not_write_memory() -> None:
    result = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-routing",
        stage_outputs=_stage_outputs(),
        task={
            "encode_episode": True,
            "generalizable_knowledge": True,
            "requires_answer": True,
            "requires_action_plan": True,
        },
    ))

    handoff = result.routing["memory_handoff"]
    assert "episodic_memory_handoff" in result.routing["routes"]
    assert "semantic_memory_candidate" in result.routing["routes"]
    assert "question_answering" in result.routing["routes"]
    assert "decision_support" in result.routing["routes"]
    assert handoff["target"] == "Point7MemoryDecisionProcessor"
    assert handoff["durable_storage_performed"] is False
    assert result.diagnostics["durable_memory_write_performed"] is False


def test_optional_lif_reference_models_seven_separate_populations() -> None:
    result = Point9NeuralArchitectureCoordinator(
        simulate_biology=True,
        lif_population_size=4,
        lif_steps_per_operation=1,
    ).process(Point9Request(
        sequence_id="p9-lif",
        stage_outputs=_stage_outputs(),
        task={"mentalizing_required": True},
    ))

    assert result.biological_summary["operation_count"] == 20
    assert result.biological_summary["population_names"] == list(POINT9_SYSTEM_NAMES)
    assert len(result.biological_trace) == 20
    assert result.biological_summary["biological_reference_is_measured_human_data"] is False


def test_lif_reference_validates_drives_and_operation_count() -> None:
    lif = Point9LIFReference(population_size=2, steps_per_operation=1)
    with pytest.raises(KeyError):
        lif.run({"unknown_system": 0.5})
    with pytest.raises(ValueError):
        lif.run({"language": float("nan")})
    with pytest.raises(ValueError):
        lif.run(operation_count=0)


def test_custom_network_edge_validation_rejects_unknown_nodes_and_out_of_range_weights() -> None:
    with pytest.raises(ValueError):
        Point9NeuralArchitectureCoordinator().process(Point9Request(
            sequence_id="p9-bad-edge",
            communication_weights={"unknown->language": 0.3},
        ))
    with pytest.raises(ValueError):
        Point9NeuralArchitectureCoordinator().process(Point9Request(
            sequence_id="p9-bad-weight",
            communication_weights={"perceptual->language": 1.5},
        ))


def test_interpretation_sufficiency_preserves_critical_unresolved_items() -> None:
    result = Point9NeuralArchitectureCoordinator().process(Point9Request(
        sequence_id="p9-critical",
        stage_outputs=_stage_outputs(),
        unresolved=[{"issue_id": "speaker-identity", "critical": True, "uncertainty": 0.9}],
    ))

    assert result.interpretation["sufficiency"]["adequate"] is False
    assert result.interpretation["sufficiency"]["critical_unresolved"] == ["speaker-identity"]


def test_reset_clears_recurrent_state_and_preserves_configuration() -> None:
    processor = Point9NeuralArchitectureCoordinator(candidate_capacity=3)
    processor.process(Point9Request(sequence_id="p9-reset-a", stage_outputs=_stage_outputs()))
    processor.reset()

    assert processor.state.time_index == 0
    assert processor.state.literal_propositions == []
    assert processor.candidate_capacity == 3


def test_every_point9_class_and_function_has_a_docstring() -> None:
    module_path = Path(__file__).resolve().parents[1] / "ore_language_comprehension" / "point09.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    missing = [
        f"{node.name}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and ast.get_docstring(node) is None
    ]
    assert missing == []


def test_point9_annotations_are_indexed_in_the_existing_sqlite_catalogue(tmp_path: Path) -> None:
    """Index Point 9 classes and methods in the same annotation database used by earlier points."""
    source_root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "ore-project"
    isolated.mkdir()
    shutil.copytree(source_root / "ore_logger", isolated / "ore_logger")
    shutil.copytree(source_root / "ore_language_comprehension", isolated / "ore_language_comprehension")
    shutil.copytree(source_root / "ore_brain_networks", isolated / "ore_brain_networks")
    (isolated / "pyproject.toml").write_text("[project]\nname='point9-annotation-test'\n", encoding="utf-8")
    database_path = tmp_path / "point9-annotations.sqlite3"

    report = register_point09_annotations(isolated, database_path)
    rows = LoggerDatabase(database_path).search_annotations("ore_language_comprehension", limit=2500)

    assert report["symbols_written"] > 0
    assert report["database_path"] == str(database_path)
    assert any(row["qualified_name"] == "Point9NeuralArchitectureCoordinator.process" for row in rows)
    assert any(row["qualified_name"] == "Point9Request.from_stage_results" for row in rows)
    assert all(row["source_docstring"] for row in rows)
