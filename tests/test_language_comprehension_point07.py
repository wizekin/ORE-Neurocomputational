from __future__ import annotations

import pytest

from ore_language_comprehension.point07 import (
    POINT7_STEP_SPECS,
    Point7LIFReference,
    Point7MemoryDecisionProcessor,
    Point7MemoryStore,
    Point7Request,
)


def _proposition() -> dict:
    return {
        "predicate": "APOLOGIZE",
        "semantic_roles": {"agent": "Lisa", "recipient": "Peter"},
        "components": [
            {"content": "Lisa apologized", "provenance": "explicit", "confidence": 0.95},
            {"content": "Peter was the recipient", "provenance": "inferred", "confidence": 0.65},
        ],
        "relations": [{"source": "P1", "relation": "follows", "target": "P0"}],
    }


def test_point7_exposes_twenty_eight_ordered_steps() -> None:
    result = Point7MemoryDecisionProcessor().process(
        Point7Request(sequence_id="p7-steps", proposition=_proposition())
    )
    assert len(POINT7_STEP_SPECS) == 28
    assert len(result.functional_trace) == 28
    assert [x["step"] for x in result.functional_trace] == list(range(1, 29))
    assert result.diagnostics["valid"] is True


def test_episodic_encoding_preserves_context_and_component_provenance() -> None:
    store = Point7MemoryStore()
    processor = Point7MemoryDecisionProcessor(store)
    result = processor.process(
        Point7Request(
            sequence_id="p7-episode",
            proposition=_proposition(),
            context={"time": "yesterday", "place": "library", "source": "conversation"},
            task={"encode_episode": True},
        )
    )
    assert "episodic" in result.routes
    assert result.episodic_trace is not None
    assert result.episodic_trace["time"] == "yesterday"
    assert result.episodic_trace["source_provenance"] == ["explicit", "inferred"]
    assert store.episodes()[0]["event_core"]["participants_and_roles"]["agent"] == "Lisa"


def test_memory_store_uses_valid_sqlite_schema_and_persists_when_path_is_supplied(tmp_path) -> None:
    path = str(tmp_path / "point7.sqlite3")
    store = Point7MemoryStore(path)
    processor = Point7MemoryDecisionProcessor(store)
    processor.process(
        Point7Request(
            sequence_id="p7-durable",
            proposition=_proposition(),
            task={"encode_episode": True},
            context={"source": "direct conversation"},
        )
    )
    assert len(store.episodes()) == 1
    store.close()

    reopened = Point7MemoryStore(path)
    assert len(reopened.episodes()) == 1
    assert reopened.episodes()[0]["sequence_id"] == "p7-durable"
    reopened.close()


def test_semantic_candidate_is_pending_unless_explicitly_verified() -> None:
    store = Point7MemoryStore()
    processor = Point7MemoryDecisionProcessor(store)
    candidate = processor.process(
        Point7Request(
            sequence_id="p7-semantic-pending",
            proposition={"predicate": "WATER_BOILS_AT", "claim": "Water boils at 100 C"},
            components=[{"content": "Water boils at 100 C", "provenance": "reported", "confidence": 0.8}],
            task={"generalizable_knowledge": True},
        )
    )
    assert candidate.semantic_candidate["status"] == "candidate_requires_prior_knowledge_and_reliability_check"
    assert store.semantic_records()[0]["status"] == "pending_verification"

    accepted = processor.process(
        Point7Request(
            sequence_id="p7-semantic-verified",
            proposition={"predicate": "WATER_BOILS_AT", "claim": "Water boils at 100 C"},
            components=[{"content": "Water boils at 100 C", "provenance": "explicit", "confidence": 0.99}],
            task={"generalizable_knowledge": True, "semantic_verified": True},
        )
    )
    assert accepted.semantic_candidate["knowledge_id"]
    assert store.semantic_records()[-1]["status"] == "accepted"


def test_answering_who_question_extracts_explicit_agent_role() -> None:
    result = Point7MemoryDecisionProcessor().process(
        Point7Request(
            sequence_id="p7-answer",
            proposition=_proposition(),
            query="Who apologized?",
            task={"requires_answer": True},
        )
    )
    assert result.answer["status"] == "selected"
    assert result.answer["value"] == "Lisa"
    assert result.answer["provenance"] == "explicit"


def test_decision_branch_applies_hard_constraints_and_high_stakes_threshold() -> None:
    options = [
        {"option": "unsafe-shortcut", "expected_utility": 10, "cost": 1, "risk": 0, "hard_constraints_satisfied": False},
        {"option": "safe-route", "expected_utility": 7, "cost": 1, "risk": 1, "evidence_sufficient": True},
        {"option": "wait", "expected_utility": 2, "cost": 0, "risk": 0, "evidence_sufficient": True},
    ]
    result = Point7MemoryDecisionProcessor().process(
        Point7Request(
            sequence_id="p7-decision",
            proposition=_proposition(),
            feasible_options=options,
            task={"requires_action_plan": True, "high_consequence": True, "risk_weight": 1.0},
        )
    )
    assert result.action_plan["status"] == "selected"
    assert result.action_plan["selected_option"]["option"] == "safe-route"
    assert all(x["option"] != "unsafe-shortcut" for x in result.action_plan["candidates"])


def test_ambiguous_decision_defers_instead_of_forcing_choice() -> None:
    result = Point7MemoryDecisionProcessor(decision_margin=0.2).process(
        Point7Request(
            sequence_id="p7-defer",
            proposition=_proposition(),
            feasible_options=[
                {"option": "A", "utility": 1.0},
                {"option": "B", "utility": 0.95},
            ],
            task={"requires_action_plan": True, "minimum_decision_margin": 0.2},
        )
    )
    assert result.action_plan["selected_option"] is None
    assert result.action_plan["status"] == "defer_or_gather_more_information"


def test_low_retention_does_not_force_episodic_storage() -> None:
    store = Point7MemoryStore()
    result = Point7MemoryDecisionProcessor(store).process(
        Point7Request(sequence_id="p7-transient", proposition=_proposition())
    )
    assert result.routes == ["transient"]
    assert result.episodic_trace is None
    assert store.episodes() == []
    assert result.diagnostics["durable_storage_written"] is False


def test_point6_factory_preserves_handoff_and_lif_reference_is_labelled_as_analogy() -> None:
    request = Point7Request.from_point6_result(
        "p7-from-p6",
        {"handoff": {"active_propositions": [{"predicate": "LEAVE"}], "situation_model": {"events": ["leave"]}}},
    )
    assert request.proposition["active_propositions"] == [{"predicate": "LEAVE"}]
    assert request.context["point6_handoff"]["situation_model"] == {"events": ["leave"]}
    assert request.context["situation_model"] == {"events": ["leave"]}
    assert request.components[0]["provenance"] == "explicit"

    biology = Point7LIFReference(population_size=3).run(operation_count=28)
    assert len(biology["population_names"]) == 10
    assert biology["operation_count"] == 28
    assert biology["biological_reference_is_measured_human_data"] is False


def test_invalid_provenance_is_reported_without_crashing() -> None:
    result = Point7MemoryDecisionProcessor().process(
        Point7Request(
            sequence_id="p7-invalid-source",
            proposition=_proposition(),
            components=[{"content": "unknown source", "provenance": "invented-tag", "confidence": 0.9}],
        )
    )
    assert result.diagnostics["valid"] is True
    assert result.functional_trace[1]["output"]["invalid_provenance_count"] == 1


def test_semantic_verification_flag_without_source_components_does_not_accept_claim() -> None:
    store = Point7MemoryStore()
    result = Point7MemoryDecisionProcessor(store).process(
        Point7Request(
            sequence_id="p7-unsubstantiated",
            proposition={"predicate": "UNSUPPORTED_CLAIM", "claim": "unverified general rule"},
            task={"generalizable_knowledge": True, "semantic_verified": True},
        )
    )
    assert result.semantic_candidate is not None
    assert store.semantic_records()[0]["status"] == "pending_verification"


def test_retrieval_reactivates_episode_and_feedback_is_persisted_separately() -> None:
    store = Point7MemoryStore()
    processor = Point7MemoryDecisionProcessor(store)
    processor.process(
        Point7Request(
            sequence_id="p7-reactivation",
            proposition=_proposition(),
            task={"encode_episode": True},
        )
    )
    retrieved = processor.process(
        Point7Request(
            sequence_id="p7-retrieval",
            proposition={},
            query="apologized",
        )
    )
    assert retrieved.retrieved_memories
    assert "immediate_answer" not in retrieved.routes
    assert retrieved.retrieved_memories[0]["reactivation_count"] == 1
    assert store.episodes()[0]["trace_strength"] > 0.25

    feedback = processor.process(
        Point7Request(
            sequence_id="p7-feedback",
            proposition={},
            outcome_feedback={"predicted_outcome": 0.8, "observed_outcome": 0.5},
        )
    )
    assert feedback.functional_trace[-1]["output"]["prediction_error"] == pytest.approx(-0.3)
    assert store.feedback_events()[0]["sequence_id"] == "p7-feedback"
