"""Software-contract tests for ORE's eight-stage discourse subsystem."""

from __future__ import annotations

from ore_discourse import (
    DiscourseConfig,
    DiscourseN7Adapter,
    DiscourseProcessor,
    DiscourseRequest,
    OPERATION_REGISTRY,
    STAGE_REGISTRY,
)


def test_registry_trace_and_biological_reference_are_complete() -> None:
    """Check stable 39-operation ordering and one neural step per functional operation."""
    request = DiscourseRequest(
        sequence_id="trace-001",
        sentences=["Mira opened the door.", "Then she entered the room."],
        entities=[{"entity_id": "mira", "label": "Mira", "entity_type": "person"}],
    )
    result = DiscourseProcessor(DiscourseConfig(simulate_biology=True)).process(request)
    assert len(STAGE_REGISTRY) == 8
    assert len(OPERATION_REGISTRY) == 39
    assert len(result.operation_trace) == 39
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"DC-{index:03d}" for index in range(1, 40)
    ]
    assert result.biological["operation_count"] == 39
    assert result.biological["units"]["voltage"] == "mV"
    assert result.biological["units"]["current"] == "pA"
    assert result.diagnostics["valid"] is True


def test_coreference_and_common_ground_are_preserved() -> None:
    """Resolve a person pronoun from explicit entity evidence and retain declared common ground."""
    request = DiscourseRequest(
        sequence_id="coref-001",
        sentences=["Mira entered the room.", "She sat down."],
        entities=[{"entity_id": "mira", "label": "Mira", "entity_type": "person"}],
        common_ground={"facts": ["The room is part of the office."]},
        context={"prior_turns": ["Mira was expected at the office."]},
    )
    result = DiscourseProcessor(DiscourseConfig(simulate_biology=False)).process(request)
    references = result.discourse_context["coreference"]
    assert any(item["surface"] == "she" and item["entity_id"] == "mira" for item in references)
    assert result.discourse_context["common_ground"]["facts"] == [
        "The room is part of the office."
    ]
    assert result.discourse_context["prior_turn_count"] == 1


def test_sentence_integration_detects_contrast_and_causal_connectives() -> None:
    """Link neighboring sentence propositions using explicit relation markers."""
    request = DiscourseRequest(
        sequence_id="links-001",
        sentences=["Mira arrived early.", "However, the meeting had already ended."],
    )
    result = DiscourseProcessor(DiscourseConfig(simulate_biology=False)).process(request)
    assert result.proposition_links[0]["relation"] == "contrast"
    assert result.proposition_links[0]["connective"] == "however"


def test_conventional_indirect_request_retains_literal_question() -> None:
    """Represent a conventional ability question and its inferred listener action separately."""
    request = DiscourseRequest(
        sequence_id="request-001",
        sentences=["Could you please open the window?"],
    )
    result = DiscourseProcessor(DiscourseConfig(simulate_biology=False)).process(request)
    item = result.indirect_requests[0]
    assert item["literal_interpretation"]["type"] == "question"
    assert item["intended_action"] == "open the window"
    assert item["is_indirect_request"] is True
    assert item["preserved_literal_content"] is True


def test_irony_detection_uses_context_and_optional_prosody() -> None:
    """Raise an irony hypothesis when positive wording follows an explicitly negative event."""
    request = DiscourseRequest(
        sequence_id="irony-001",
        sentences=["The deployment failed and the service was broken.", "Great work on the release!"],
        prosody={1: {"intonation_mismatch": 0.8, "exaggeration": 0.6}},
    )
    result = DiscourseProcessor(DiscourseConfig(simulate_biology=False)).process(request)
    assert result.irony[1]["semantic_mismatch"] == 1.0
    assert result.irony[1]["prosody_available"] is True
    assert result.irony[1]["irony_probability"] > 0.58


def test_temporal_relations_and_duration_are_tracked() -> None:
    """Extract temporal order and explicit duration from a small two-sentence narrative."""
    request = DiscourseRequest(
        sequence_id="time-001",
        sentences=["The alarm rang for 3 minutes.", "Afterward, the guard arrived."],
        temporal_evidence={0: {"start": 0, "end": 3, "duration": {"value": 3, "unit": "minutes"}}},
    )
    result = DiscourseProcessor(DiscourseConfig(simulate_biology=False)).process(request)
    assert any(item["relation"] == "after" for item in result.temporal_relations)
    assert result.coherence["timeline"] == [0, 1]
    assert result.diagnostics["temporal_constraints_valid"] is True


def test_n7_adapter_returns_the_fields_consumed_by_n9() -> None:
    """Verify that the adapter maps N9's discourse payload into its documented output contract."""
    adapter = DiscourseN7Adapter(
        DiscourseProcessor(DiscourseConfig(simulate_biology=False))
    )
    output = adapter.predict({
        "propositions": [
            {"sentence_index": 0, "text": "The team launched the service."},
            {"sentence_index": 1, "text": "Then the service became available."},
        ],
        "semantic_graph": {
            "nodes": [{"id": "team", "type": "entity", "label": "team"}],
            "edges": [],
        },
        "lexical_hypotheses": [],
        "discourse_context": {
            "sequence_id": "n9-001",
            "prior_turns": ["The team prepared a release."],
        },
    })
    for field in (
        "intents", "predictions", "situation_model", "coherence", "uncertainty",
        "temporal_links", "causal_links", "proposition_compatibility",
        "knowledge_conflicts",
    ):
        assert field in output
    assert 0.0 <= output["coherence"] <= 1.0
    assert 0.0 <= output["uncertainty"] <= 1.0


def test_serialization_and_disabled_biology() -> None:
    """Ensure the result serializes to JSON-compatible structures when biology is disabled."""
    request = DiscourseRequest(
        sequence_id="json-001",
        propositions=[{"sentence_index": 0, "predicate": "arrive", "arguments": {"agent": "Mira"}}],
    )
    result = DiscourseProcessor(DiscourseConfig(simulate_biology=False)).process(request)
    payload = result.to_dict()
    assert payload["schema"] == "ore.discourse.eight_stage.v2"
    assert len(payload["operation_trace"]) == 39
    assert payload["biological"]["model"] == "disabled"


def test_request_validation_rejects_empty_input() -> None:
    """Reject a discourse request that provides neither surface sentences nor propositions."""
    try:
        DiscourseRequest(sequence_id="invalid-001")
    except ValueError as error:
        assert "provide sentences" in str(error)
    else:
        raise AssertionError("empty request should fail validation")


def test_stage_registry_counts_match_the_39_operation_distribution() -> None:
    """Check that the eight stage counts add up to exactly 39 registered operations."""
    assert [stage.operation_count for stage in STAGE_REGISTRY] == [5, 5, 5, 5, 4, 5, 5, 5]
    assert sum(stage.operation_count for stage in STAGE_REGISTRY) == 39
    assert all(
        row["functional_output"]["source_stage_outputs"]
        for row in DiscourseProcessor(DiscourseConfig(simulate_biology=False)).process(
            DiscourseRequest(
                sequence_id="grouping-001",
                sentences=["Mira arrived.", "Then she opened the door."],
            )
        ).operation_trace
    )


def test_discourse_symbols_are_annotated_in_the_logger_sqlite_catalog(tmp_path) -> None:
    """Index discourse docstrings and verify class/function annotations are stored in SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    database = LoggerDatabase(tmp_path / "discourse-annotations.sqlite3")
    report = sync_repository(database, root)
    assert report["parse_errors"] == 0

    expected_symbols = (
        ("ore_discourse.pipeline", "DiscourseProcessor", "class"),
        ("ore_discourse.pipeline", "DiscourseProcessor.process", "function"),
        ("ore_discourse.pipeline", "DiscourseProcessor._stage_context", "function"),
        ("ore_discourse.biology", "DiscourseLIFSimulator", "class"),
        ("ore_discourse.biology", "DiscourseLIFSimulator.step", "function"),
        ("ore_discourse.registry", "OperationSpec", "class"),
        ("ore_discourse.registry", "registry_as_dicts", "function"),
        ("ore_discourse.contracts", "DiscourseRequest", "class"),
    )
    for module, qualified_name, kind in expected_symbols:
        annotation = database.get_annotation(module, qualified_name, kind)
        assert annotation is not None
        assert annotation["annotation"].strip()

    with database.connection() as connection:
        stored = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE module_name LIKE 'ore_discourse.%'"
        ).fetchone()[0]
    assert stored >= len(expected_symbols)
