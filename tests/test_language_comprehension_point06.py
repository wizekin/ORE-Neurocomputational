"""Acceptance tests for source-ordered Point 6 working-memory integration."""

from __future__ import annotations

from ore_working_memory import WorkingMemoryConfig
from ore_language_comprehension import (
    POINT6_STEP_SPECS,
    Point6PropositionalWorkingMemory,
    Point6Request,
)


def _point5_sample():
    """Provide a compact Point 5 payload with literal, inferred and unresolved content."""
    return {
        "schema": "ore.language_comprehension.point05.v1",
        "literal_proposition": {
            "propositions": [{
                "proposition_id": "p1",
                "predicate": "GIVE",
                "arguments": {"agent": "Mira", "theme": "book"},
            }],
            "proposition_graph": {
                "nodes": [{"node_id": "p1", "node_type": "event"}],
                "edges": [],
            },
            "operators": {"tense": "past"},
            "thematic_roles": [{"role": "Agent", "filler": "Mira"}],
            "proposition_completeness": [{
                "predicate": "GIVE",
                "required_roles": ["Agent", "Theme", "Recipient"],
                "filled_required_roles": ["Agent", "Theme"],
                "missing_required_roles": ["Recipient"],
            }],
            "competing_readings": [],
            "confidence": 0.9,
        },
        "enriched_interpretation": {
            "discourse_links": [{"source": "p1", "relation": "after", "target": "p0"}],
            "coreference": [{"surface": "she", "entity_id": "mira"}],
            "active_information": {"prior_turn_count": 1},
            "speaker_intentions": [{"type": "inform", "confidence": 0.7}],
            "indirect_speech_acts": [],
            "irony_hypotheses": [],
            "situation_model": {
                "people": [{"entity_id": "mira", "label": "Mira"}],
                "events": [{"event_id": "p1", "predicate": "GIVE"}],
            },
            "temporal_relations": [{"relation": "after", "source": "p1", "target": "p0"}],
            "coherence": {"score": 0.8, "topic_distribution": {"book": 0.7, "gift": 0.3}},
            "common_ground": {"facts": ["Mira knows Lina."]},
            "context_conflicts": [],
            "unresolved_alternatives": [],
            "world_knowledge_evidence": {},
            "conversation_goal": {"task": "understand"},
            "inference_policy": {
                "literal_content_preserved": True,
                "inferences_are_defeasible": True,
            },
        },
        "discourse_result": {
            "sentences": ["Mira gave a book."],
            "propositions": [{"predicate": "GIVE", "text": "Mira gave a book."}],
        },
        "handoff": {"target": "Point 6"},
    }


def _no_biology(**overrides):
    """Build a reproducible working-memory processor without LIF simulation."""
    values = {"simulate_biology": False}
    values.update(overrides)
    return Point6PropositionalWorkingMemory(WorkingMemoryConfig(**values))


def test_point6_has_eighteen_steps_and_maps_all_forty_operations() -> None:
    request = Point6Request(
        sequence_id="point6-trace",
        text="Mira gave a book to Lina, and she thanked her.",
    )
    result = _no_biology().process(request)

    assert len(POINT6_STEP_SPECS) == 18
    assert len(result.functional_trace) == 18
    assert sum(
        len(row["underlying_working_memory_operation_ids"])
        for row in result.functional_trace
    ) == 42  # 40 canonical operations plus two explicit source extensions
    assert result.diagnostics["underlying_working_memory_operation_count"] == 40
    assert result.diagnostics["underlying_operation_order_valid"] is True
    assert result.diagnostics["source_extension_ids"] == ["WMX-001", "WMX-002"]
    assert result.diagnostics["valid"] is True


def test_point5_factory_preserves_literal_enriched_and_situation_tracks() -> None:
    payload = _point5_sample()
    request = Point6Request.from_point5_result("point6-from-p5", payload)
    result = _no_biology().process(request)

    assert request.text == "Mira gave a book."
    assert request.point5_interpretation["literal_proposition"] == payload["literal_proposition"]
    assert result.point5_interpretation["enriched_interpretation"]["coreference"] == [
        {"surface": "she", "entity_id": "mira"}
    ]
    assert result.handoff["situation_model"] == payload["enriched_interpretation"]["situation_model"]
    assert result.handoff["active_propositions"] == payload["literal_proposition"]["propositions"]


def test_unfilled_semantic_role_is_kept_as_an_open_working_memory_slot() -> None:
    request = Point6Request.from_point5_result("point6-open-role", _point5_sample())
    result = _no_biology().process(request)

    incomplete = result.working_memory_state["incomplete_structures"]
    assert incomplete["unresolved_slot_count"] == 1
    assert incomplete["complete"] is False
    assert incomplete["structures"][0]["unresolved_slots"] == ["Recipient"]


def test_competing_interpretations_can_remain_unresolved() -> None:
    request = Point6Request(
        sequence_id="point6-abstention",
        text="Nadia met Salma after she arrived.",
        interpretations=[
            {"hypothesis_id": "she-nadia", "interpretation": "she refers to Nadia", "prior": 0.5},
            {"hypothesis_id": "she-salma", "interpretation": "she refers to Salma", "prior": 0.5},
        ],
    )
    result = _no_biology().process(request)

    assert result.working_memory_state["semantic_resolution"]["abstained"] is True
    assert len(result.working_memory_state["semantic_resolution"]["alternatives"]) == 2
    assert result.diagnostics["valid"] is True


def test_point6_is_temporary_and_does_not_write_durable_memory() -> None:
    result = _no_biology().process(
        Point6Request(
            sequence_id="point6-temporary",
            text="The glass fell from the table.",
        )
    )

    assert result.working_memory_state["temporary_state_only"] is True
    assert result.working_memory_state["durable_memory_written"] is False
    assert result.handoff["durable_storage_performed"] is False
    assert result.diagnostics["temporary_state_only"] is True


def test_optional_lif_reference_tracks_all_forty_canonical_operations() -> None:
    processor = Point6PropositionalWorkingMemory(
        WorkingMemoryConfig(
            simulate_biology=True,
            population_size=3,
            steps_per_operation=1,
        )
    )
    result = processor.process(
        Point6Request(
            sequence_id="point6-lif",
            text="Maya said that Omar had left.",
        )
    )

    assert result.biological_summary["operation_count"] == 40
    assert result.diagnostics["biological_reference_is_measured_human_data"] is False
    assert len(result.biological_trace) == 18
