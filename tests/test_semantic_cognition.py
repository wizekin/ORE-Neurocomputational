"""Regression tests for the 44-operation semantic-cognition subsystem."""
from __future__ import annotations

import json

import pytest

from ore_semantic import (
    OPERATION_REGISTRY,
    SemanticConcept,
    SemanticConfig,
    SemanticCognitionProcessor,
    SemanticN8Adapter,
    SemanticRequest,
    WordFormEpisode,
    semantic_request_from_reading_result,
)


def concept(
    concept_id: str,
    label: str,
    *,
    aliases: tuple[str, ...] = (),
    semantic_features: dict[str, float] | None = None,
    modality_features: dict[str, dict[str, float]] | None = None,
    category: str = "noun",
    frequency: float = 10.0,
    prototype_id: str | None = None,
) -> SemanticConcept:
    return SemanticConcept(
        concept_id=concept_id,
        label=label,
        aliases=aliases,
        semantic_features=semantic_features or {"ENTITY": 1.0},
        modality_features=modality_features or {},
        lexical_category=category,
        frequency=frequency,
        prototype_id=prototype_id,
        verified_by="test",
        source_reference=f"test://{concept_id}",
    )


def test_registry_has_44_contiguous_unique_operation_ids() -> None:
    ids = [operation.operation_id for operation in OPERATION_REGISTRY]
    assert len(ids) == 44
    assert len(set(ids)) == 44
    assert ids == [f"SC-{index:03d}" for index in range(1, 45)]
    assert all(operation.functional_responsibility for operation in OPERATION_REGISTRY)
    assert all(operation.biological_reference for operation in OPERATION_REGISTRY)


def test_exact_lexical_match_returns_semantic_concept_and_44_trace_rows() -> None:
    processor = SemanticCognitionProcessor([
        concept("c-cat", "cat", semantic_features={"ANIMAL": 1.0, "FELINE": 1.0},
                prototype_id="animal")
    ])
    result = processor.process(SemanticRequest("s-1", text="Cat"))

    assert result.tokens[0]["selected_concept"]["concept_id"] == "c-cat"
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["operation_count"] == 44
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"SC-{index:03d}" for index in range(1, 45)
    ]
    assert result.to_dict()["schema"] == "ore.semantic_cognition.v1"


def test_ambiguous_alias_is_preserved_and_can_abstain() -> None:
    processor = SemanticCognitionProcessor([
        concept("c-bank-river", "bank", aliases=("bank",),
                semantic_features={"RIVER": 1.0}, frequency=10.0),
        concept("c-bank-money", "financial institution", aliases=("bank",),
                semantic_features={"MONEY": 1.0}, frequency=10.0),
    ])
    result = processor.process(SemanticRequest("s-ambiguous", text="bank"))

    assert result.tokens[0]["abstained"] is True
    assert len(result.tokens[0]["candidates"]) == 2
    assert {row["concept_id"] for row in result.tokens[0]["candidates"]} == {
        "c-bank-river", "c-bank-money"
    }


def test_episodic_trace_can_supply_candidate_without_lexical_alias() -> None:
    processor = SemanticCognitionProcessor([
        concept("c-remembered", "unrelated-label", semantic_features={"OBJECT": 1.0})
    ])
    episode = WordFormEpisode(
        episode_id="episode-1",
        orthography="unseenform",
        concept_id="c-remembered",
        phonological_form=("UW", "N"),
        semantic_features={"OBJECT": 1.0},
        strength=0.95,
        recency=0.90,
        source_reference="dataset://episode-1",
    )
    result = processor.process(SemanticRequest(
        "s-episode", text="unseenform", episodes=(episode,)
    ))

    assert result.tokens[0]["selected_concept"]["concept_id"] == "c-remembered"
    assert result.episodic_retrieval[0]["episode_id"] == "episode-1"


def test_modality_specific_evidence_changes_competition_without_erasing_alternatives() -> None:
    processor = SemanticCognitionProcessor([
        concept(
            "c-fish", "bass",
            semantic_features={"ANIMAL": 1.0},
            modality_features={"visual": {"FINS": 1.0, "WATER": 1.0}},
            frequency=10.0,
        ),
        concept(
            "c-instrument", "bass",
            semantic_features={"ARTIFACT": 1.0},
            modality_features={"visual": {"STRINGS": 1.0, "WOOD": 1.0}},
            frequency=10.0,
        ),
    ])
    result = processor.process(SemanticRequest(
        "s-modality", text="bass",
        modality_evidence={"visual": {"FINS": 1.0, "WATER": 1.0}},
    ))

    rows = result.tokens[0]["candidates"]
    assert len(rows) == 2
    assert rows[0]["concept_id"] == "c-fish"
    assert rows[0]["modality_fit"] > rows[1]["modality_fit"]


def test_biological_reference_has_44_snapshots_and_labeled_units() -> None:
    processor = SemanticCognitionProcessor(
        [concept("c-apple", "apple")],
        config=SemanticConfig(steps_per_operation=1, dt_ms=1.0),
    )
    result = processor.process(SemanticRequest("s-neural", text="apple"))

    biological = result.biological
    assert biological["enabled"] is True
    assert len(biological["operation_trace"]) == 44
    assert biological["simulated_time_ms"] == pytest.approx(44.0)
    assert biological["numerically_valid"] is True
    assert biological["units"]["membrane_voltage"] == "mV"
    assert biological["units"]["synaptic_current"] == "pA"
    assert biological["units"]["firing_rate"] == "Hz"
    assert biological["units"]["simulated_time"] == "ms"
    assert "atl_hub" in biological["layers"]
    assert "ifg_control" in biological["layers"]


def test_biology_can_be_disabled_without_shortening_registry_trace() -> None:
    processor = SemanticCognitionProcessor(
        [concept("c-apple", "apple")],
        config=SemanticConfig(simulate_biology=False),
    )
    result = processor.process(SemanticRequest("s-no-bio", text="apple"))

    assert len(result.operation_trace) == 44
    assert result.biological["enabled"] is False
    assert len(result.biological["operation_trace"]) == 44
    assert {row["status"] for row in result.biological["operation_trace"]} == {"skipped"}


def test_inference_does_not_mutate_lexicon_or_model_fingerprint() -> None:
    processor = SemanticCognitionProcessor([
        concept("c-apple", "apple", semantic_features={"FOOD": 1.0})
    ])
    before = processor.model_fingerprint
    processor.process(SemanticRequest("s-readonly", text="apple"))
    assert processor.model_fingerprint == before
    assert processor.concepts[0].semantic_features == {"FOOD": 1.0}


def test_reading_result_adapter_preserves_sequence_and_token_order() -> None:
    reading_payload = {
        "sequence_id": "reading-001",
        "tokens": [{"orthography": "one"}, {"orthography": "two"}],
        "normalized_text": "one two",
    }
    request = semantic_request_from_reading_result(reading_payload)
    assert request.sequence_id == "reading-001"
    assert tuple(request.tokens or ()) == ("one", "two")


def test_n9_adapter_returns_valid_concept_graph_without_fabricated_relations() -> None:
    processor = SemanticCognitionProcessor([
        concept("c-apple", "apple", semantic_features={"FOOD": 1.0})
    ], config=SemanticConfig(simulate_biology=False))
    adapter = SemanticN8Adapter(processor)
    output = adapter.compose({
        "sequence_id": "n9-semantic-001",
        "lexical_hypotheses": [{"form": "apple", "score": 0.99}],
        "syntax": {},
        "context": {},
    })

    assert output["status"] == "concept_access_only"
    assert output["semantic_graph"]["nodes"][0]["id"] == "c-apple"
    assert output["semantic_graph"]["edges"] == []
    assert output["propositions"] == []
    assert output["completeness"] == pytest.approx(1.0)
    assert len(output["semantic_result"]["operation_trace"]) == 44
    json.dumps(output, ensure_ascii=False)


def test_n9_adapter_abstains_when_upstream_has_no_lexical_input() -> None:
    adapter = SemanticN8Adapter(SemanticCognitionProcessor([]))
    output = adapter.compose({"sequence_id": "n9-empty", "lexical_hypotheses": []})
    assert output["status"] == "unavailable_no_lexical_hypotheses"
    assert output["semantic_graph"] == {"nodes": [], "edges": []}
    assert output["completeness"] == 0.0
    assert output["ambiguity"] == 1.0


@pytest.mark.parametrize("invalid", [-0.1, 1.1, float("nan")])
def test_concept_rejects_invalid_feature_values(invalid: float) -> None:
    with pytest.raises(ValueError):
        concept("c-invalid", "invalid", semantic_features={"BAD": invalid})
