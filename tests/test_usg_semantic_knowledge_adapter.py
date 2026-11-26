"""Local regression tests for complete ORE-USG Phase 2 semantic integration."""
from __future__ import annotations

from copy import deepcopy

from ore_usg import USGKnowledgeAdapter, USGKnowledgeIndex, build_semantic_composition_processor
from test_usg_lexical_adapter import _bundle


def _phase2_bundle():
    bundle = deepcopy(_bundle())
    timestamp = "2026-11-12T00:00:00Z"
    bundle["records"].extend([
        {
            "id": "concept:finance", "kind": "CONCEPT", "version": 1,
            "state": "ACCEPTED", "created_at": timestamp,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
            "context_ids": ["context:finance"],
            "payload": {
                "concept_type": "GENERAL", "namespace": "urn:usg:test",
                "label": "Finance", "language": "en",
                "definition": "The management of money and assets.",
            },
        },
        {
            "id": "class:entity", "kind": "SEMANTIC_CLASS", "version": 1,
            "state": "ACCEPTED", "created_at": timestamp,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
            "context_ids": ["context:finance"],
            "payload": {"class_key": "entity", "definition": "An entity."},
        },
        {
            "id": "class:financial-entity", "kind": "SEMANTIC_CLASS", "version": 1,
            "state": "ACCEPTED", "created_at": timestamp,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
            "context_ids": ["context:finance"],
            "payload": {
                "class_key": "financial_entity",
                "definition": "An entity participating in financial activity.",
                "parent_class_keys": ["entity"],
            },
        },
        {
            "id": "class:pending", "kind": "SEMANTIC_CLASS", "version": 1,
            "state": "UNRESOLVED", "created_at": timestamp,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
            "context_ids": ["context:finance"],
            "payload": {"class_key": "pending_class", "definition": "Unresolved class."},
        },
        {
            "id": "frame:transfer", "kind": "PREDICATE_FRAME", "version": 1,
            "state": "VALIDATED_FOR_PROFILE", "created_at": timestamp,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
            "context_ids": ["context:finance"],
            "payload": {
                "frame_key": "transfer", "version": "1.0",
                "roles": [
                    {"role": "AGENT", "description": "Initiator", "required": True},
                    {"role": "THEME", "description": "Transferred item", "required": True},
                    {"role": "RECIPIENT", "description": "Destination", "required": False},
                ],
            },
        },
        {
            "id": "frame:incomplete", "kind": "PREDICATE_FRAME", "version": 1,
            "state": "UNRESOLVED", "created_at": timestamp,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
            "context_ids": ["context:finance"],
            "payload": {
                "frame_key": "unresolved_transfer", "version": "draft",
                "roles": [], "complete": False, "resolved": False,
            },
        },
    ])
    bundle["evidence"].append({
        "id": "evidence:contradicts", "source_id": "source:permitted",
        "evidence_type": "RELATION_OBSERVATION", "support_status": "CONTRADICTS",
        "captured_at": timestamp, "transformation_ids": [],
    })
    bundle["relations"].extend([
        {
            "id": "relation:financial-subclass",
            "subject_record_id": "class:financial-entity",
            "predicate": "SUBCLASS_OF",
            "object_record_id": "class:entity",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:lemma"], "context_ids": ["context:finance"],
        },
        {
            "id": "relation:finance-membership",
            "subject_record_id": "concept:finance",
            "predicate": "INSTANCE_OF",
            "object_record_id": "class:financial-entity",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:lemma", "evidence:contradicts"],
            "context_ids": ["context:finance"],
        },
        {
            "id": "relation:unresolved-membership",
            "subject_record_id": "sense:en:bank:river",
            "predicate": "INSTANCE_OF",
            "object_record_id": "class:pending",
            "direction": "DIRECTED", "state": "UNRESOLVED",
            "evidence_ids": ["evidence:lemma"], "context_ids": ["context:finance"],
        },
    ])
    bundle["mappings"].append({
        "id": "mapping:financial-sense-to-finance",
        "source_record_id": "sense:en:bank:financial",
        "target_record_id": "concept:finance",
        "mapping_type": "RELATED_TO", "state": "ACCEPTED",
        "evidence_ids": ["evidence:lemma"], "context_ids": ["context:finance"],
    })
    return bundle


def test_senses_and_concepts_become_distinct_provenance_tracked_ore_concepts():
    bundle = _phase2_bundle()
    original = deepcopy(bundle)
    _, resource = USGKnowledgeAdapter().load_and_convert(bundle)
    semantic = resource.semantic_knowledge

    senses = [item for item in semantic.concept_records if item.record_kind == "LEXICAL_SENSE"]
    concepts = [item for item in semantic.concept_records if item.record_kind == "CONCEPT"]
    assert len(senses) == 2  # both accepted/profile-validated senses remain distinct
    assert len(concepts) == 1
    assert len({item.concept.concept_id for item in senses}) == len(senses)
    financial = next(item for item in senses if item.usg_record_id == "sense:en:bank:financial")
    assert financial.definition == "A financial institution."
    assert financial.concept.definition == financial.definition
    assert financial.lemma_record_id == "lemma:en:bank"
    assert financial.lemma_source_reference.startswith("usg-knowledge://")
    assert "bank" in financial.concept.aliases
    assert financial.concept.lexical_category == "noun"
    assert financial.sense_source_id == "source:permitted"
    assert "evidence%3Afinancial" in financial.concept.source_reference
    assert all(item.concept.semantic_features == {} for item in semantic.concept_records)
    assert all(item.active for item in semantic.concept_records if item.lifecycle_state != "UNRESOLVED")
    assert bundle == original


def test_categories_hierarchy_memberships_and_conflicts_are_explicit():
    semantic = USGKnowledgeAdapter().convert(_phase2_bundle()).semantic_knowledge
    by_key = {item.class_key: item for item in semantic.categories}
    assert by_key["financial_entity"].parent_category_ids == (by_key["entity"].category_id,)
    assert by_key["pending_class"].lifecycle_state == "UNRESOLVED"

    conflicted = next(item for item in semantic.memberships if item.predicate == "INSTANCE_OF" and item.member_id.endswith("concept%3Afinance@1"))
    assert conflicted.status == "CONFLICTED"
    unresolved = next(item for item in semantic.memberships if item.category_id == by_key["pending_class"].category_id)
    assert unresolved.status == "UNRESOLVED"
    assert unresolved.source_object_id == unresolved.category_id


def test_relations_keep_type_direction_and_endpoint_identity_without_merging():
    semantic = USGKnowledgeAdapter().convert(_phase2_bundle()).semantic_knowledge
    by_predicate = {item.predicate: item for item in semantic.relations}
    hierarchy = by_predicate["SUBCLASS_OF"]
    assert hierarchy.relation_type == "HIERARCHY"
    assert hierarchy.direction == "DIRECTED"
    assert hierarchy.subject_id != hierarchy.object_id

    mapping = next(item for item in semantic.relations if item.source_kind == "MAPPING")
    assert mapping.predicate == "RELATED_TO"
    assert mapping.relation_type == "ASSOCIATION"
    assert mapping.subject_id != mapping.object_id


def test_predicate_frames_preserve_complete_and_unresolved_forms():
    semantic = USGKnowledgeAdapter().convert(_phase2_bundle()).semantic_knowledge
    by_key = {item.frame_key: item for item in semantic.predicate_frames}
    executable = by_key["transfer"]
    assert executable.status == "EXECUTABLE"
    assert executable.ore_frame is not None
    assert executable.ore_frame.roles == ("AGENT", "THEME", "RECIPIENT")
    assert executable.ore_frame.required_roles == ("AGENT", "THEME")

    incomplete = by_key["unresolved_transfer"]
    assert incomplete.status == "UNRESOLVED"
    assert incomplete.ore_frame is None
    assert "roles" in incomplete.unresolved_fields
    assert len(semantic.executable_frames) == 1


def test_unified_retrieval_and_semantic_composition_use_the_new_inventory():
    resource = USGKnowledgeAdapter().convert(_phase2_bundle())
    index = USGKnowledgeIndex(resource)
    sense_hits = index.lookup("bank", kind="SEMANTIC_CONCEPT").items
    assert len(sense_hits) == 2
    assert index.lookup("A financial institution.", kind="SEMANTIC_CONCEPT").items
    assert index.lookup("financial_entity", kind="SEMANTIC_CATEGORY").items
    assert index.lookup("unresolved_transfer", kind="PREDICATE_FRAME").items
    processor = build_semantic_composition_processor(resource)
    assert {item.concept_id for item in processor.concepts} == {
        item.concept_id for item in resource.semantic_knowledge.active_concepts
    }
    assert len(processor.frames) == 1
