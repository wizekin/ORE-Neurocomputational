"""Tests for direct bridges from USG resources to existing ORE processors."""
from __future__ import annotations

from dataclasses import replace

import pytest

from ore_usg import (
    USGKnowledgeAdapter,
    USGPredicateFrameConversionError,
    build_lexical_access_processor,
    build_semantic_composition_processor,
    build_usg_semantic_composition_processor,
    to_ore_predicate_frame,
)
from ore_semantic_composition import SemanticCompositionConfig, SemanticCompositionRequest
from test_usg_structure_adapter import _structure_bundle


def test_lexical_access_processor_uses_imported_ore_lexical_entries():
    resource = USGKnowledgeAdapter().convert(_structure_bundle())

    processor = build_lexical_access_processor(resource)

    assert processor.entries == resource.lexical.entries
    assert processor.by_form["bank"]
    assert processor.entries[0].source_reference.startswith("usg-knowledge://")


def test_semantic_composition_processor_receives_imported_concepts_and_frames():
    resource = USGKnowledgeAdapter().convert(_structure_bundle())

    processor = build_semantic_composition_processor(resource)

    assert processor.concepts == resource.semantic.concepts
    assert len(processor.frames) == 1
    assert processor.frames[0].predicate == "transfer"
    assert processor.frames[0].roles == ("AGENT", "THEME", "RECIPIENT")
    assert processor.frames[0].required_roles == ("AGENT",)
    assert "frame_version=1.2" in processor.frames[0].source_reference
    assert processor.frames[0].source_reference.startswith("usg-knowledge://")


def test_predicate_frame_conversion_rejects_uninterpretable_role_shapes():
    resource = USGKnowledgeAdapter().convert(_structure_bundle())
    frame = resource.structures.predicate_frames[0]
    malformed = replace(frame, roles_json='{"AGENT": true}')

    with pytest.raises(USGPredicateFrameConversionError, match="roles array"):
        to_ore_predicate_frame(malformed)



def test_usg_aware_composition_uses_classes_relations_mappings_and_frames():
    bundle = _structure_bundle()
    timestamp = "2026-11-12T00:00:00Z"
    bundle["records"].append({
        "id": "concept:transfer",
        "kind": "CONCEPT",
        "version": 1,
        "state": "ACCEPTED",
        "created_at": timestamp,
        "source_ids": ["source:permitted"],
        "evidence_ids": ["evidence:lemma"],
        "context_ids": ["context:finance"],
        "payload": {
            "concept_type": "EVENT",
            "namespace": "urn:test",
            "label": "transfer",
            "language": "en",
            "lexical_category": "verb",
        },
    })
    bundle["mappings"].append({
        "id": "mapping:money-related-transfer",
        "source_record_id": "concept:money",
        "target_record_id": "concept:transfer",
        "mapping_type": "RELATED_TO",
        "state": "ACCEPTED",
        "evidence_ids": ["evidence:lemma"],
        "context_ids": ["context:finance"],
    })
    original = deepcopy(bundle)
    resource = USGKnowledgeAdapter().convert(bundle)
    processor = build_usg_semantic_composition_processor(
        resource,
        config=SemanticCompositionConfig(
            simulate_biology=False,
            confidence_threshold=0.0,
            ambiguity_margin=0.0,
        ),
    )

    result = processor.process(SemanticCompositionRequest(
        sequence_id="usg-phase2-transfer",
        tokens=("money", "transfer"),
    ))

    overlay = result.sentence_representation["usg_knowledge_graph"]
    counts = overlay["counts"]
    assert counts["semantic_concepts"] == 2
    assert counts["semantic_classes"] == 1
    assert counts["predicate_frames"] == 1
    assert counts["explicit_relations"] == 1
    assert counts["concept_mappings"] == 1
    assert any(
        edge["edge_type"] == "usg_explicit_relation"
        and edge["predicate"] == "INSTANCE_OF"
        for edge in overlay["edges"]
    )
    assert any(
        edge["edge_type"] == "usg_concept_mapping"
        and edge["mapping_type"] == "RELATED_TO"
        for edge in overlay["edges"]
    )
    class_node = next(
        node for node in overlay["nodes"]
        if node["node_type"] == "usg_semantic_class"
    )
    assert class_node["class_key"] == "financial_entity"
    frame_node = next(
        node for node in overlay["nodes"]
        if node["node_type"] == "usg_predicate_frame"
    )
    assert frame_node["frame_key"] == "transfer"
    assert frame_node["roles"] == ["AGENT", "THEME", "RECIPIENT"]
    assert result.sentence_representation["validation"]["valid_references"] is True
    assert result.diagnostics["usg_semantic_integration"]["graph_references_valid"] is True
    assert result.metadata["usg_semantic_integration"]["bundle_digest"] == resource.bundle_digest
    assert result.metadata["usg_semantic_integration"]["semantic_features_inferred"] is False
    assert bundle == original


def test_usg_aware_composition_does_not_attach_unrelated_relations():
    resource = USGKnowledgeAdapter().convert(_structure_bundle())
    processor = build_usg_semantic_composition_processor(
        resource,
        config=SemanticCompositionConfig(
            simulate_biology=False,
            confidence_threshold=0.0,
            ambiguity_margin=0.0,
        ),
    )

    result = processor.process(SemanticCompositionRequest(
        sequence_id="usg-phase2-unrelated",
        tokens=("money",),
    ))

    overlay = result.sentence_representation["usg_knowledge_graph"]
    assert not any(
        edge.get("predicate") == "DENOTES"
        for edge in overlay["edges"]
        if edge["edge_type"] == "usg_explicit_relation"
    )
