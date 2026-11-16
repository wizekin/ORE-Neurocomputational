"""Tests for persistent episodic word-form memory and explicit consolidation."""
from __future__ import annotations

from ore_semantic import (
    MEMORY_OPERATION_REGISTRY,
    SemanticConcept,
    SemanticCognitionProcessor,
    SemanticMemoryStore,
    SemanticRequest,
    SemanticPrototypeEvidence,
    SOURCE_OPERATION_MAP,
)


def _concept() -> SemanticConcept:
    """Build one concept entry that can be linked to learned word forms."""
    return SemanticConcept(
        concept_id="c-cat",
        label="cat",
        semantic_features={"ANIMAL": 1.0, "FELINE": 1.0},
        lexical_category="noun",
        prototype_id="animal",
        source_reference="test://cat",
    )


def test_capture_store_recall_and_explicit_memory_trace(tmp_path) -> None:
    """Persist a bound word-form episode and retrieve it through semantic inference."""
    database = tmp_path / "episodes.sqlite3"
    store = SemanticMemoryStore(str(database))
    processor = SemanticCognitionProcessor([_concept()], memory_store=store)

    episode = processor.learn_word_form_episode(
        "kitten",
        concept_id="c-cat",
        phonological_form=("K", "IH", "T", "AH", "N"),
        semantic_features={"ANIMAL": 1.0, "FELINE": 0.9},
        context_features={"young": 1.0, "animal": 1.0},
        source_reference="test://observed-episode",
        prototype_id="animal",
    )
    assert store.episode_count() == 1
    result = processor.process(SemanticRequest("memory-read-1", text="kitten"))
    assert result.tokens[0]["selected_concept"]["concept_id"] == "c-cat"
    assert result.episodic_retrieval[0]["episode_id"] == episode.episode_id
    assert store.episode_count() == 1
    assert [row["operation_id"] for row in store.operation_trace[:5]] == [
        "SM-001", "SM-002", "SM-003", "SM-004", "SM-005"
    ]
    assert len(MEMORY_OPERATION_REGISTRY) == 16
    assert len(SOURCE_OPERATION_MAP) == 44
    assert len(result.metadata["semantic_source_operation_mapping"]) == 44
    assert any(row["operation_id"] == "SM-008" for row in result.metadata["persistent_memory_operation_trace"])
    store.close()

    reopened = SemanticMemoryStore(str(database))
    assert reopened.episode_count() == 1
    assert reopened.recall("kitten")[0].concept_id == "c-cat"
    reopened.close()


def test_pattern_separation_preserves_distinct_encounters_and_merges_duplicates(tmp_path) -> None:
    """Keep similar episodes distinct when phonological or context bindings differ."""
    store = SemanticMemoryStore(str(tmp_path / "separation.sqlite3"))
    first = store.capture_word_form(
        "read", concept_id="c-read", phonological_form=("R", "EH", "D"),
        context_features={"past": 1.0}, prototype_id="read-verb",
    )
    second = store.capture_word_form(
        "read", concept_id="c-read", phonological_form=("R", "IY", "D"),
        context_features={"present": 1.0}, prototype_id="read-verb",
    )
    assert first.episode_id != second.episode_id
    duplicate = store.store_episode(first, prototype_id="read-verb")
    assert duplicate["duplicate_merged"] is True
    assert duplicate["episode_id"] == first.episode_id
    assert store.episode_count() == 2
    store.close()


def test_consolidation_strengthens_a_prototype_from_accumulated_experience(tmp_path) -> None:
    """Build and update a durable abstract prototype from separately stored experiences."""
    store = SemanticMemoryStore(str(tmp_path / "prototype.sqlite3"))
    store.capture_word_form(
        "apple", concept_id="c-apple", phonological_form=("AE", "P", "AH", "L"),
        semantic_features={"FOOD": 1.0, "FRUIT": 1.0}, context_features={"red": 1.0},
        prototype_id="fruit",
    )
    store.capture_word_form(
        "pear", concept_id="c-pear", phonological_form=("P", "EH", "R"),
        semantic_features={"FOOD": 1.0, "FRUIT": 0.9}, context_features={"green": 1.0},
        prototype_id="fruit",
    )

    report = store.consolidate("fruit", learning_rate=0.25)
    prototype = store.get_prototype("fruit")
    assert report["updated_prototypes"][0]["experience_count"] == 2
    assert prototype is not None
    assert json_value(prototype["semantic_features"])["FOOD"] == 1.0
    assert prototype["experience_count"] == 2
    assert [row["operation_id"] for row in store.operation_trace[-7:]] == [
        "SM-009", "SM-010", "SM-011", "SM-012", "SM-013", "SM-014", "SM-015"
    ]
    store.close()



def test_pattern_completion_recovers_a_stored_episode_from_partial_context_cue(tmp_path) -> None:
    """Complete missing orthographic and phonological channels from a supported episode."""
    store = SemanticMemoryStore(str(tmp_path / "completion.sqlite3"))
    episode = store.capture_word_form(
        "kitten",
        concept_id="c-cat",
        phonological_form=("K", "IH", "T", "AH", "N"),
        semantic_features={"ANIMAL": 1.0, "FELINE": 0.9},
        context_features={"young": 1.0, "animal": 1.0},
    )

    completion = store.pattern_complete(context_features={"young": 1.0})

    assert completion["status"] == "completed"
    assert completion["completed_episode_id"] == episode.episode_id
    assert {"orthography", "phonology", "semantics"} == set(completion["completed_channels"])
    assert completion["no_unobserved_feature_invention"] is True
    assert store.operation_trace[-1]["operation_id"] == "SM-007"
    store.close()


def test_abstract_prototype_generalizes_using_ordered_phonology_and_semantic_features(tmp_path) -> None:
    """Consolidate multiple experiences and match an unfamiliar cue against the resulting prototype."""
    store = SemanticMemoryStore(str(tmp_path / "generalization.sqlite3"))
    store.capture_word_form(
        "apple", concept_id="c-apple", phonological_form=("AE", "P", "AH", "L"),
        semantic_features={"FOOD": 1.0, "FRUIT": 1.0}, context_features={"red": 1.0},
        prototype_id="fruit",
    )
    store.capture_word_form(
        "pear", concept_id="c-pear", phonological_form=("P", "EH", "R"),
        semantic_features={"FOOD": 1.0, "FRUIT": 0.9}, context_features={"green": 1.0},
        prototype_id="fruit",
    )
    store.consolidate("fruit", learning_rate=0.25)

    matches = store.generalize(
        semantic_features={"FRUIT": 1.0},
        context_features={"red": 1.0},
    )

    assert matches
    assert matches[0]["prototype_id"] == "fruit"
    assert matches[0]["generalization_status"] == "matched_consolidated_prototype"
    assert matches[0]["phonological_sequence"] in (["AE", "P", "AH", "L"], ["P", "EH", "R"])
    assert store.operation_trace[-1]["operation_id"] == "SM-016"
    store.close()


def test_semantic_pipeline_uses_pattern_completion_and_generalized_prototype_evidence(tmp_path) -> None:
    """Route persistent prototype matches back into candidate selection without making inference write memory."""
    store = SemanticMemoryStore(str(tmp_path / "semantic-integration.sqlite3"))
    processor = SemanticCognitionProcessor([_concept()], memory_store=store)
    processor.learn_word_form_episode(
        "cat",
        concept_id="c-cat",
        phonological_form=("K", "AE", "T"),
        semantic_features={"ANIMAL": 1.0, "FELINE": 1.0},
        context_features={"domestic": 1.0},
        prototype_id="animal",
    )
    processor.consolidate_memory("animal", learning_rate=0.2)

    result = processor.process(SemanticRequest(
        "generalization-read-1",
        text="Cat",
        context={"domestic": 1.0},
    ))

    candidate = result.tokens[0]["candidates"][0]
    assert candidate["concept_id"] == "c-cat"
    assert candidate["prototype_evidence_ids"] == ["animal"]
    assert any(
        row["source"] == "sqlite_consolidated_experience_generalization"
        for row in result.abstract_prototypes
    )
    assert result.diagnostics["recognition_is_read_only"] is True
    assert result.metadata["persistent_episode_count"] == 1
    store.close()


def json_value(value: str) -> dict[str, float]:
    """Decode a JSON numeric feature map returned from SQLite."""
    import json
    return json.loads(value)
