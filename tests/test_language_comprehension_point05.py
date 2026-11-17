"""Acceptance tests for the source-ordered Point 5 contextual/pragmatic adapter."""

from __future__ import annotations

from ore_discourse import DiscourseConfig
from ore_language_comprehension import (
    POINT5_STEP_SPECS,
    Point5ContextualPragmaticIntegrator,
    Point5Request,
)


def test_point5_exposes_twenty_source_steps_and_maps_all_discourse_operations() -> None:
    request = Point5Request(
        sequence_id="point5-trace",
        sentences=["Mira arrived early.", "However, the meeting had already ended."],
        entities=[{"entity_id": "mira", "label": "Mira", "entity_type": "person"}],
    )
    result = Point5ContextualPragmaticIntegrator(
        DiscourseConfig(simulate_biology=False)
    ).process(request)

    assert len(POINT5_STEP_SPECS) == 20
    assert len(result.functional_trace) == 20
    assert sum(len(row["underlying_discourse_operation_ids"]) for row in result.functional_trace) == 39
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["underlying_operation_order_valid"] is True
    assert result.biological_summary["model"] == "disabled"


def test_literal_point4_representation_is_preserved_separately_from_inference() -> None:
    literal = {
        "propositions": [{"predicate": "APOLOGIZE", "arguments": {"agent": "Lisa"}}],
        "proposition_graph": {"nodes": [{"id": "event-1"}], "edges": []},
        "confidence": 0.82,
    }
    request = Point5Request(
        sequence_id="point5-literal",
        sentences=["Lisa apologized."],
        literal_proposition=literal,
        unresolved_alternatives=[{"candidate": "recipient", "status": "unknown"}],
    )
    result = Point5ContextualPragmaticIntegrator(
        DiscourseConfig(simulate_biology=False)
    ).process(request)

    assert result.literal_proposition == literal
    assert result.enriched_interpretation["literal_proposition"] == literal
    assert result.enriched_interpretation["inference_policy"]["literal_content_preserved"] is True
    assert result.handoff["unresolved_alternatives"] == [{"candidate": "recipient", "status": "unknown"}]


def test_reference_resolution_uses_explicit_entity_evidence() -> None:
    request = Point5Request(
        sequence_id="point5-coreference",
        sentences=["Mira entered the room.", "She sat down."],
        entities=[{"entity_id": "mira", "label": "Mira", "entity_type": "person"}],
    )
    result = Point5ContextualPragmaticIntegrator(
        DiscourseConfig(simulate_biology=False)
    ).process(request)
    assert any(
        row.get("surface", "").casefold() == "she" and row.get("entity_id") == "mira"
        for row in result.enriched_interpretation["coreference"]
    )


def test_indirect_request_and_irony_are_hypotheses_not_literal_rewrites() -> None:
    request = Point5Request(
        sequence_id="point5-pragmatics",
        sentences=[
            "The deployment failed and the service was broken.",
            "Could you please restart the service?",
            "Great work on the release!",
        ],
        prosody={2: {"intonation_mismatch": 0.8, "exaggeration": 0.6}},
    )
    result = Point5ContextualPragmaticIntegrator(
        DiscourseConfig(simulate_biology=False)
    ).process(request)
    assert result.enriched_interpretation["indirect_speech_acts"]
    assert result.enriched_interpretation["irony_hypotheses"]
    assert result.enriched_interpretation["inference_policy"]["inferences_are_defeasible"] is True


def test_point4_factory_preserves_semantic_payload_and_context() -> None:
    point4 = {
        "schema": "ore.language_comprehension.point04.v1",
        "propositions": [{"predicate": "OPEN", "arguments": {"agent": "Mira", "patient": "door"}}],
        "proposition_graph": {"nodes": [{"id": "e1"}], "edges": []},
        "competing_readings": [{"reading_id": "r2", "confidence": 0.2}],
        "confidence": 0.9,
    }
    request = Point5Request.from_point4_result(
        "point5-from-p4",
        point4,
        sentences=["Mira opened the door.", "Then she left."],
        prior_turns=["Mira was trying to leave."],
        common_ground={"facts": ["Mira has a key."]},
    )
    result = Point5ContextualPragmaticIntegrator(
        DiscourseConfig(simulate_biology=False)
    ).process(request)
    assert request.literal_proposition["proposition_graph"] == point4["proposition_graph"]
    assert request.prior_turns == ("Mira was trying to leave.",)
    assert result.handoff["target"].startswith("Point 6")
