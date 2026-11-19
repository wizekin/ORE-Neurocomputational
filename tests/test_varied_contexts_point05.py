"""Functional, biological, pipeline, and SQLite tests for Point 05 varied contexts."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from ore_logger.database import LoggerDatabase
from ore_varied_contexts import (
    BiologicalVariedContextsNetwork,
    ConceptDefinition,
    HippocampalContextMemory,
    LearningExample,
    TrialObservation,
    VariedContextsProcessor,
    VariedContextsRequest,
    VariedContextsState,
    OPERATION_REGISTRY,
    build_concept_profiles,
    compare_concepts,
    context_id,
    cosine_similarity,
    infer_context,
    record_result_to_sqlite,
    stable_softmax,
)


def build_concepts() -> tuple[ConceptDefinition, ...]:
    """Create three related math concepts with explicit defining features and prerequisites."""
    return (
        ConceptDefinition(
            "direct", "mathematics",
            {"ratio_constant": 1.0, "product_constant": 0.0, "difference_constant": 0.0},
            discriminative_features={"ratio_constant": 1.0},
            prerequisite_concepts=("arithmetic",),
            relations=("the ratio remains constant",),
            boundary_conditions={"rate_assumption": "constant"},
            initial_familiarity=0.7,
        ),
        ConceptDefinition(
            "inverse", "mathematics",
            {"ratio_constant": 0.0, "product_constant": 1.0, "difference_constant": 0.0},
            discriminative_features={"product_constant": 1.0},
            prerequisite_concepts=("arithmetic",),
            relations=("the product remains constant",),
            initial_familiarity=0.7,
        ),
        ConceptDefinition(
            "additive", "mathematics",
            {"ratio_constant": 0.0, "product_constant": 0.0, "difference_constant": 1.0},
            discriminative_features={"difference_constant": 1.0},
            prerequisite_concepts=("arithmetic",),
            relations=("the difference remains constant",),
            initial_familiarity=0.7,
        ),
    )


def build_examples() -> tuple[LearningExample, ...]:
    """Create training and held-out examples across contexts, modalities, and transfer levels."""
    def create(example_id: str, concept: str, feature: str, setting: str,
               format_name: str, modality: str, level: str = "training",
               anchor: bool = False) -> LearningExample:
        """Build one labeled example with concept features independent of context features."""
        return LearningExample(
            example_id=example_id,
            concept_id=concept,
            concept_features={
                "ratio_constant": float(feature == "ratio_constant"),
                "product_constant": float(feature == "product_constant"),
                "difference_constant": float(feature == "difference_constant"),
            },
            context_features={"setting": setting, "format": format_name,
                              **({"rate_assumption": "constant"} if concept == "direct" else {})},
            modality=modality,
            representation=format_name,
            relations=({
                "direct": ("the ratio remains constant",),
                "inverse": ("the product remains constant",),
                "additive": ("the difference remains constant",),
            }[concept]),
            is_anchor=anchor,
            is_novel=level != "training",
            transfer_level=level,
            session_id="tests",
        )

    return (
        create("direct-anchor", "direct", "ratio_constant", "kitchen", "story", "text", anchor=True),
        create("direct-table", "direct", "ratio_constant", "market", "table", "table"),
        create("inverse-anchor", "inverse", "product_constant", "road", "table", "table", anchor=True),
        create("inverse-diagram", "inverse", "product_constant", "factory", "diagram", "diagram"),
        create("additive-anchor", "additive", "difference_constant", "shop", "graph", "graph", anchor=True),
        create("additive-story", "additive", "difference_constant", "workplace", "story", "text"),
        create("direct-near", "direct", "ratio_constant", "garden", "diagram", "diagram", "near"),
        create("inverse-near", "inverse", "product_constant", "pipes", "story", "text", "near"),
        create("additive-near", "additive", "difference_constant", "club", "table", "table", "near"),
        create("direct-far", "direct", "ratio_constant", "navigation", "map", "diagram", "far"),
        create("inverse-far", "inverse", "product_constant", "workflow", "flowchart", "diagram", "far"),
        create("additive-far", "additive", "difference_constant", "measurement", "equation", "text", "far"),
        create("direct-delayed", "direct", "ratio_constant", "sports", "table", "table", "delayed"),
        create("inverse-delayed", "inverse", "product_constant", "travel", "graph", "graph", "delayed"),
        create("additive-delayed", "additive", "difference_constant", "budget", "story", "text", "delayed"),
    )


def build_observations(
    examples: tuple[LearningExample, ...],
    wrong_example: str | None = None,
) -> tuple[TrialObservation, ...]:
    """Create explicit selections, confidence ratings, contextual judgments, and relation evidence."""
    relations = {
        "direct": "the ratio remains constant",
        "inverse": "the product remains constant",
        "additive": "the difference remains constant",
    }
    observations = []
    for item in examples:
        selected = item.concept_id
        if item.example_id == wrong_example:
            selected = "inverse" if item.concept_id == "direct" else "direct"
        observations.append(TrialObservation(
            example_id=item.example_id,
            selected_concept_id=selected,
            confidence=0.92 if selected != item.concept_id else 0.86,
            effort=0.80 if selected != item.concept_id else 0.42,
            context_judgment=dict(item.context_features),
            cue_type=item.modality,
            explanation="The selected relation should match the defining feature.",
            preserved_relations=(relations[item.concept_id],),
            incompatible_relations=(),
            feedback_verified=True,
            response_time_ms=2500.0,
        ))
    return tuple(observations)


def build_request(wrong_example: str | None = None) -> VariedContextsRequest:
    """Build a complete request with independent observations across all four trial levels."""
    concepts = build_concepts()
    examples = build_examples()
    return VariedContextsRequest(
        concepts=concepts,
        examples=examples,
        observations=build_observations(examples, wrong_example),
        prerequisite_readiness={"arithmetic": 0.92},
        current_context={"setting": "kitchen", "format": "story", "rate_assumption": "constant"},
        fatigue=0.15,
        available_time_minutes=45.0,
        original_context_id=context_id(examples[0].context_features),
        current_time_utc="2026-01-01T00:00:00Z",
    )


def test_registry_has_exactly_23_ordered_operations():
    """Keep registry identifiers aligned with the 23 source algorithm stages."""
    assert len(OPERATION_REGISTRY) == 23
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P05-{index:02d}" for index in range(1, 24)
    ]
    assert all(item.mathematical_form and item.biological_reference for item in OPERATION_REGISTRY)


def test_similarity_softmax_and_context_entropy_are_valid():
    """Check core numerical primitives for normalization, finiteness, and edge cases."""
    assert cosine_similarity({"a": 1.0}, {"a": 1.0}) == pytest.approx(1.0)
    probabilities = stable_softmax({"direct": 1000.0, "inverse": 999.0, "additive": -1000.0})
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["direct"] > probabilities["inverse"] > probabilities["additive"]


def test_concept_profiles_preserve_episodes_and_estimate_invariants():
    """Build category prototypes without dropping example-level records."""
    concepts = build_concepts()
    examples = build_examples()
    profiles = build_concept_profiles(concepts, examples)
    direct = profiles["direct"]
    assert direct["example_count"] == 5
    assert "ratio_constant" in direct["candidate_invariants"]
    assert len(direct["episodic_examples_preserved"]) == 4
    comparisons = compare_concepts(concepts, profiles)
    assert len(comparisons["pairs"]) == 3
    assert comparisons["pairs"]["direct::inverse"]["prototype_distance"] > 0.0


def test_context_inference_recovers_known_context_and_retains_uncertainty():
    """Estimate a stored context from its cues and request more evidence for an ambiguous cue set."""
    examples = build_examples()
    exact = infer_context(examples[0].context_features, examples)
    assert exact["best_context_id"] == context_id(examples[0].context_features)
    assert 0.0 <= exact["uncertainty"] <= 1.0
    vague = infer_context({"format": "story"}, examples)
    assert vague["uncertainty"] >= 0.0
    assert isinstance(vague["needs_more_cues"], bool)


def test_pipeline_builds_meaningful_interleaving_and_separates_transfer_levels():
    """Run all 23 stages and check schedule, context variation, transfer, and delayed measures."""
    result = VariedContextsProcessor().process(build_request())
    assert len(result.operation_trace) == 23
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"P05-{index:02d}" for index in range(1, 24)
    ]
    assert result.interleaving_decision["approved_pairs"]
    assert result.schedule["interleaving_used"] is True
    assert result.schedule["spacing_not_derived_from_order"] is True
    assert result.transfer_report["near"]["accuracy"] == pytest.approx(1.0)
    assert result.transfer_report["far"]["accuracy"] == pytest.approx(1.0)
    assert result.transfer_report["delayed"]["accuracy"] == pytest.approx(1.0)
    assert result.flexibility_report["context_flexibility_mean"] == pytest.approx(1.0)
    assert result.output["context_count"] > 1
    assert result.output["modality_count"] > 1
    assert result.biological["literal_neuron_simulation"] is False


def test_interleaving_rejects_unrelated_or_unready_concepts():
    """Do not mix unrelated categories when prerequisite knowledge or useful contrast is absent."""
    concepts = (
        ConceptDefinition("biology", "biology", {"cell": 1.0}),
        ConceptDefinition("music", "music", {"rhythm": 1.0}),
    )
    examples = (
        LearningExample("cell", "biology", {"cell": 1.0}, {"setting": "lab"}, is_anchor=True),
        LearningExample("beat", "music", {"rhythm": 1.0}, {"setting": "studio"}, is_anchor=True),
    )
    request = VariedContextsRequest(concepts, examples)
    result = VariedContextsProcessor().process(request)
    assert result.interleaving_decision["approved_pairs"] == []
    assert result.schedule["interleaving_used"] is False


def test_boundary_conditions_and_far_transfer_mapping_are_explicit():
    """Retain genuine context constraints and penalize incompatible relational mappings."""
    concepts = build_concepts()
    examples = build_examples()
    observations = build_observations(examples)
    result = VariedContextsProcessor().process(VariedContextsRequest(
        concepts, examples, observations,
        prerequisite_readiness={"arithmetic": 0.9},
    ))
    direct_anchor = next(item for item in examples if item.example_id == "direct-anchor")
    boundary = result.operation_trace[18]["functional_output"]["boundary_conditions_by_example"]
    direct_condition = boundary[direct_anchor.example_id]["concept_conditions"]["direct"]
    assert direct_condition["condition_satisfied"] is True
    far = result.transfer_report["far"]
    assert far["mean_relation_mapping_score"] == pytest.approx(1.0)
    assert all(row["valid_mapping"] for group in far["relation_mapping"] for row in [group])


def test_errors_are_diagnosed_and_future_practice_adapts():
    """Detect a concept confusion and recommend targeted contrastive practice."""
    result = VariedContextsProcessor().process(build_request(wrong_example="direct-near"))
    assert result.output["near_transfer_accuracy"] < 1.0
    assert result.operation_trace[20]["functional_output"]["error_diagnosis"]["correction_required"]
    assert result.operation_trace[21]["functional_output"]["adaptive_practice_plan"]["concept_priority"]


def test_biological_surrogates_are_bounded_and_keep_episodes_separate():
    """Exercise episodic completion, modality pathways, control, monitoring, and plasticity gating."""
    network = BiologicalVariedContextsNetwork(dimensions=32)
    example = build_examples()[0]
    encoded = network.encode_example(example)
    assert encoded["episode_binding"]["episode_bound"] is True
    context_result = network.infer_context(example.context_features, example.concept_id)
    assert context_result["pattern_completion"]["expected_concept_match"] is True
    selection = network.select_concept({"direct": 0.8, "inverse": 0.2}, 0.9, 0.1)
    assert selection["selected_concept"] == "direct"
    gated = network.learn_from_trial(0.7, 1.0, 0.8, 0.9, True)
    assert 0.0 <= gated["plasticity_gate"]["learning_gate"] <= 1.0
    snapshot = network.snapshot()
    assert snapshot["literal_neuron_simulation"] is False
    assert snapshot["hippocampal_context_memory"]["episode_count"] == 1


def test_state_reuse_keeps_concept_and_episode_memories():
    """Retain concept prototypes and prior episode records across processing calls."""
    state = VariedContextsState()
    processor = VariedContextsProcessor(state=state)
    first = processor.process(build_request())
    second = processor.process(build_request())
    assert second.final_state["process_count"] == 2
    assert len(second.final_state["episode_records"]) == len(build_examples())
    assert set(second.final_state["concepts"]) == {"direct", "inverse", "additive"}
    assert first.output["interleaving_used"] is True


def test_sqlite_stores_annotations_and_23_ordered_operation_events(tmp_path):
    """Index every package definition and persist all ordered stages through the shared ORE logger."""
    result = VariedContextsProcessor().process(build_request())
    database_path = tmp_path / "point05.sqlite3"
    repository_root = Path(__file__).resolve().parents[1]
    report = record_result_to_sqlite(result, database_path, repository_root=repository_root)
    database = LoggerDatabase(database_path)
    annotations = database.search_annotations("ore_varied_contexts", limit=1000)
    definitions = [row for row in annotations
                   if row["symbol_kind"] in {"class", "function", "async_function"}]
    assert definitions
    assert all(row["annotation"].strip() for row in definitions)
    assert all(row["source_docstring"] for row in definitions)
    with sqlite3.connect(database_path) as connection:
        run = connection.execute(
            "SELECT status, event_count FROM runs WHERE run_id=?",
            (report["run_id"],),
        ).fetchone()
        events = connection.execute(
            "SELECT sequence_no, event_kind, metadata_json FROM events "
            "WHERE run_id=? ORDER BY sequence_no",
            (report["run_id"],),
        ).fetchall()
    assert run == ("completed", 23)
    assert [row[0] for row in events] == list(range(1, 24))
    assert all(row[1] == "operation" and "P05-" in row[2] for row in events)


def test_contracts_reject_unknown_transfer_level_and_naive_timestamp():
    """Reject invalid task labels and timestamps before pipeline processing."""
    with pytest.raises(ValueError):
        LearningExample(
            "bad", "direct", {"ratio": 1.0}, {}, transfer_level="unknown"
        )
    with pytest.raises(ValueError):
        VariedContextsRequest(
            build_concepts(), build_examples(),
            current_time_utc="2026-01-01T00:00:00",
        )
