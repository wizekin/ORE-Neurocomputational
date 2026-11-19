"""Functional, biological, and SQLite integration tests for Point 02."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from ore_prior_knowledge import (
    BiologicalPriorKnowledgeNetwork,
    KnowledgeNode,
    KnowledgeRelation,
    KnowledgeSchema,
    LearningUnit,
    NeuralPopulation,
    PriorKnowledgeConfig,
    PriorKnowledgeProcessor,
    PriorKnowledgeRequest,
    PriorKnowledgeState,
    RelationInput,
    OPERATION_REGISTRY,
    cosine_similarity,
    registry_as_dicts,
)


def build_request(text: str | None = None) -> PriorKnowledgeRequest:
    """Create a small biology knowledge graph and schema for integration tests."""
    nodes = (
        KnowledgeNode(
            "membrane", "cell membrane", features={"cell": 0.9, "membrane": 1.0,
                                                      "selective": 0.9, "boundary": 0.8,
                                                      "transport": 0.8},
            context={"topic": "biology"}, reliability=0.95,
            memory_type="semantic", provenance="previous_biology_lesson",
        ),
        KnowledgeNode(
            "filter", "water filter", features={"selective": 0.9, "boundary": 0.8,
                                                   "passage": 0.9, "substances": 0.6},
            context={"topic": "biology"}, reliability=0.9,
            memory_type="semantic", provenance="known_analogy",
        ),
        KnowledgeNode(
            "cell", "cell", features={"cell": 1.0, "living": 0.9},
            context={"topic": "biology"}, reliability=0.98,
            memory_type="semantic", provenance="previous_biology_lesson",
        ),
        KnowledgeNode(
            "substances", "substances", features={"substances": 1.0, "molecules": 0.7},
            context={"topic": "biology"}, reliability=0.9,
            memory_type="semantic", provenance="previous_biology_lesson",
        ),
    )
    edges = (
        KnowledgeRelation("membrane", "part_of", "cell", 0.9, 0.95, "observed"),
        KnowledgeRelation("filter", "regulates", "substances", 0.7, 0.8, "inferred"),
    )
    schema = KnowledgeSchema(
        "selective-boundary", "Selective boundary schema",
        roles=("cell membrane", "substances", "selective", "boundary"),
        relation_patterns=(("cell membrane", "regulates", "substances"),),
        expectations={"selective": 0.9, "boundary": 0.8, "transport": 0.7},
        context={"topic": "biology"}, confidence=0.8,
        supporting_episode_ids=("bio-episode-1",),
    )
    unit = LearningUnit(
        "membrane-lesson-1",
        text or "A cell membrane selectively regulates the movement of substances across a boundary.",
        entities=("cell membrane", "substances"),
        features={"cell": 0.9, "membrane": 1.0, "selective": 1.0,
                  "boundary": 0.95, "transport": 0.8, "substances": 0.7},
        relations=(RelationInput("cell membrane", "regulates", "substances", 0.95),),
        context={"topic": "biology", "lesson": "cell structure"},
        reliability=0.95,
    )
    return PriorKnowledgeRequest(
        "learner-01", unit, current_context={"topic": "biology", "goal": "understand cell membranes"},
        knowledge_nodes=nodes, knowledge_relations=edges, schemas=(schema,),
        learner_state={"attention": 0.9, "working_memory": 0.85},
    )


def test_registry_has_fourteen_source_ordered_operations():
    """Keep the formal registry aligned with the 14-stage source specification."""
    assert len(OPERATION_REGISTRY) == 14
    assert [row["operation_id"] for row in registry_as_dicts()] == [
        f"P02-{index:02d}" for index in range(1, 15)
    ]


def test_full_pipeline_integrates_prior_knowledge_and_emits_trace():
    """Run all steps and preserve evidence labels, analogy limits, and retrieval routes."""
    request = build_request()
    result = PriorKnowledgeProcessor(request.learner_id).process(request)
    assert len(result.operation_trace) == 14
    assert result.operation_trace[0]["operation_id"] == "P02-01"
    assert result.operation_trace[-1]["operation_id"] == "P02-14"
    assert result.integrated_representation["knowledge_node_id"].startswith("p02-knowledge-")
    assert result.integrated_representation["observed_inference_distinction_preserved"] is True
    assert result.analogy_maps
    assert all("limitations" in item for item in result.analogy_maps)
    assert result.retrieval_routes
    candidate_rows = result.operation_trace[3]["functional_output"]["candidate_memories"]
    assert candidate_rows
    assert sum(row["retrieval_probability"] for row in candidate_rows) == pytest.approx(1.0)
    assert result.output["biological_fidelity"] == "computational surrogate"


def test_negated_schema_consistent_material_is_stored_as_conflicting_episode():
    """Avoid blindly assimilating a claim that conflicts with a strongly matching schema."""
    request = build_request("A cell membrane does not regulate the movement of substances.")
    result = PriorKnowledgeProcessor(request.learner_id).process(request)
    assert result.confidence_and_uncertainty["fit_state"] == "schema_conflict"
    assert result.output["strategy"] in {"revise_candidate_schema", "store_separately"}
    assert result.new_episode is not None
    assert result.new_episode["observed_inference_distinction_preserved"] is True
    integrated_node = result.final_state["knowledge_nodes"][result.integrated_representation["knowledge_node_id"]]
    assert integrated_node["memory_type"] == "episodic"
    assert result.schema_updates == () or all(not item.get("automatic_overwrite", True)
                                               for item in result.schema_updates)


def test_low_capacity_and_empty_prior_graph_are_handled_without_crashing():
    """Allow a novel item to remain provisional when there is no candidate prior knowledge."""
    request = PriorKnowledgeRequest(
        "learner-empty", "An unfamiliar concept has several novel attributes.",
        current_context={"topic": "new topic"}, learner_state={"attention": 0.2,
                                                               "working_memory": 0.3},
    )
    result = PriorKnowledgeProcessor("learner-empty").process(request)
    assert len(result.operation_trace) == 14
    assert result.output["candidate_count"] == 0
    assert result.new_episode is not None
    assert result.confidence_and_uncertainty["fit_state"] in {
        "high_novelty", "ambiguous_or_partial_fit"
    }


def test_state_reuse_accumulates_nodes_associations_and_episode_history():
    """Preserve knowledge and consolidation state across sequential learning units."""
    request = build_request()
    state = PriorKnowledgeState("learner-01")
    processor = PriorKnowledgeProcessor("learner-01", state=state)
    first = processor.process(request)
    second_request = build_request("A membrane is a selective boundary that controls passage.")
    second = processor.process(second_request)
    assert first.final_state["process_count"] == 1
    assert second.final_state["process_count"] == 2
    assert second.final_state["consolidation_count"] == 2
    assert len(second.final_state["knowledge_nodes"]) >= 4
    assert second.final_state["associative_weights"]


def test_neural_population_and_biological_network_are_finite_and_bounded():
    """Verify recurrent activity, pattern binding, and biological surrogate diagnostics."""
    population = NeuralPopulation("test", dimensions=16)
    values = population.update(np.ones(16), gain=1.0, inhibition=0.1)
    assert np.all(np.isfinite(values))
    assert np.all(np.abs(values) <= 1.0)
    bio = BiologicalPriorKnowledgeNetwork(dimensions=16)
    diag = bio.process_step(1, ["cell", "membrane"], 0.4, 0.9, 0.2)
    assert diag["learning_gate"]["learning_gate"] >= 0.0
    bound = bio.bind_and_learn("episode-1", ["cell", "membrane"], {"topic": "biology"},
                               ["cell", "membrane"], ["membrane-node"], 0.7)
    assert bound["hippocampal_binding"]["bound"] is True
    assert bio.snapshot()["literal_neuron_simulation"] is False


def test_similarity_validates_shapes_and_handles_zero_vectors():
    """Keep the mathematical primitives numerically defined and shape-safe."""
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        PriorKnowledgeConfig(dimensions=4)


def test_sqlite_registers_definitions_and_persists_fourteen_operation_events(tmp_path):
    """Index package definitions and persist every numbered operation in ORE SQLite."""
    from ore_prior_knowledge import record_result_to_sqlite
    from ore_logger.database import LoggerDatabase

    request = build_request()
    result = PriorKnowledgeProcessor(request.learner_id).process(request)
    database_path = tmp_path / "point02.sqlite3"
    repository_root = Path(__file__).resolve().parents[1]
    report = record_result_to_sqlite(result, database_path, repository_root=repository_root)
    database = LoggerDatabase(database_path)
    rows = database.search_annotations("ore_prior_knowledge", limit=1000)
    definitions = [item for item in rows if item["symbol_kind"] in {"class", "function", "async_function"}]
    assert definitions
    assert all(item["annotation"].strip() and item["source_docstring"] for item in definitions)
    with sqlite3.connect(database_path) as connection:
        run = connection.execute("SELECT status, event_count FROM runs WHERE run_id = ?",
                                 (report["run_id"],)).fetchone()
        events = connection.execute("SELECT sequence_no, event_kind, metadata_json FROM events "
                                    "WHERE run_id = ? ORDER BY sequence_no",
                                    (report["run_id"],)).fetchall()
    assert run == ("completed", 14)
    assert [row[0] for row in events] == list(range(1, 15))
    assert all(row[1] == "operation" and "P02-" in row[2] for row in events)


def test_repeated_identical_input_creates_distinct_binding_episodes():
    """Allow repeated study of the same unit without colliding hippocampal episode identities."""
    request = build_request()
    processor = PriorKnowledgeProcessor(request.learner_id)
    first = processor.process(request)
    second = processor.process(request)
    first_binding = first.operation_trace[12]["functional_output"]["consolidation"]["binding"]
    second_binding = second.operation_trace[12]["functional_output"]["consolidation"]["binding"]
    assert first_binding["hippocampal_binding"]["episode_id"] != second_binding["hippocampal_binding"]["episode_id"]
    assert second.final_state["process_count"] == 2


def test_relation_object_mismatch_is_flagged_as_schema_conflict():
    """Detect a contradictory relation mapping even when surface features strongly overlap."""
    base = build_request()
    original = base.new_input
    contradictory = LearningUnit(
        "membrane-contradiction", "A cell membrane regulates ions.",
        entities=("cell membrane", "ions"),
        features=dict(original.features),
        relations=(RelationInput("cell membrane", "regulates", "ions", 0.95),),
        context={"topic": "biology"}, reliability=0.95,
    )
    request = PriorKnowledgeRequest(
        base.learner_id, contradictory, current_context=base.current_context,
        knowledge_nodes=base.knowledge_nodes, knowledge_relations=base.knowledge_relations,
        schemas=base.schemas, learner_state=base.learner_state,
    )
    result = PriorKnowledgeProcessor(request.learner_id).process(request)
    assert result.confidence_and_uncertainty["fit_state"] == "schema_conflict"
    assert result.operation_trace[7]["functional_output"]["schema_fit"]["relational_conflicts"]
    assert result.new_episode is not None


def test_learner_capacity_values_must_be_probabilities():
    """Reject non-finite or out-of-range attention and working-memory state estimates."""
    with pytest.raises(ValueError):
        PriorKnowledgeRequest("learner", "new information", learner_state={"attention": 1.5})
    with pytest.raises(ValueError):
        PriorKnowledgeRequest("learner", "new information", learner_state={"working_memory": float("nan")})
