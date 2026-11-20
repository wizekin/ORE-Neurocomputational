"""Tests for deterministic exact retrieval from unified USG knowledge."""
from __future__ import annotations

import pytest

from ore_usg import USGKnowledgeAdapter, USGKnowledgeIndex
from test_usg_structure_adapter import _structure_bundle


def test_exact_lookup_returns_lexical_class_and_frame_with_provenance():
    resource = USGKnowledgeAdapter().convert(_structure_bundle())
    index = USGKnowledgeIndex(resource)

    lexical = index.lookup("BANK", kind="LEXICAL_ENTRY", language="en")
    semantic_class = index.lookup("financial_entity", kind="SEMANTIC_CLASS")
    frame = index.lookup("transfer", kind="PREDICATE_FRAME")

    assert lexical.exact_match and lexical.items
    assert lexical.items[0].label == "bank"
    assert lexical.items[0].source_reference.startswith("usg-knowledge://")
    assert any(item.details for item in lexical.items)
    assert semantic_class.items[0].label == "financial_entity"
    assert frame.items[0].kind == "PREDICATE_FRAME"
    assert '"role":"AGENT"' in frame.items[0].details[0]


def test_exact_lookup_does_not_stem_or_fuzzy_match_and_filters_language():
    index = USGKnowledgeIndex(USGKnowledgeAdapter().convert(_structure_bundle()))

    assert not index.lookup("banking", kind="LEXICAL_ENTRY").items
    assert not index.lookup("bank", kind="LEXICAL_ENTRY", language="fr").items
    assert index.lookup("bank", kind="LEXICAL_ENTRY", language="en").items


def test_relation_lookup_uses_explicit_namespaced_record_identity():
    resource = USGKnowledgeAdapter().convert(_structure_bundle())
    index = USGKnowledgeIndex(resource)
    class_id = resource.structures.classes[0].class_id

    relations = index.relations_for(class_id)

    assert len(relations) == 1
    assert relations[0].predicate == "INSTANCE_OF"
    assert relations[0].object_id == class_id


def test_lookup_rejects_unknown_kind_and_empty_query():
    index = USGKnowledgeIndex(USGKnowledgeAdapter().convert(_structure_bundle()))

    with pytest.raises(ValueError, match="query must be non-empty"):
        index.lookup(" ")
    with pytest.raises(ValueError, match="kind must be one of"):
        index.lookup("bank", kind="FUZZY")


def test_explicit_lexical_sense_relation_links_to_concept_without_guessing():
    resource = USGKnowledgeAdapter().convert(_structure_bundle())
    index = USGKnowledgeIndex(resource)
    bank_entry = next(
        entry for entry in resource.lexical.entries
        if entry.orthography == "bank" and entry.meanings
    )
    financial_meaning = next(
        meaning for meaning in bank_entry.meanings
        if "financial institution" in meaning.gloss.casefold()
    )
    relations = index.relations_for(financial_meaning.meaning_id)

    assert len(relations) == 1
    assert relations[0].predicate == "DENOTES"
    assert relations[0].subject_id == financial_meaning.meaning_id
    assert relations[0].object_id in {item.concept_id for item in resource.semantic.concepts}
