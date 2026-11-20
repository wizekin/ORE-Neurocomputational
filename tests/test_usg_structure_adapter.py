"""Tests for USG semantic-class, predicate-frame, and relation imports."""
from __future__ import annotations

from copy import deepcopy

from ore_usg import USGSemanticStructureAdapter
from test_usg_lexical_adapter import _bundle


def _structure_bundle():
    bundle = deepcopy(_bundle())
    timestamp = "2026-11-12T00:00:00Z"
    bundle["records"].extend([
        {
            "id": "class:financial-entity", "kind": "SEMANTIC_CLASS", "version": 1,
            "state": "ACCEPTED", "created_at": timestamp,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
            "context_ids": ["context:finance"],
            "payload": {
                "class_key": "financial_entity",
                "definition": "An entity that participates in financial activity.",
                "parent_class_keys": ["entity"],
                "custom_metadata": {"origin": "test-fixture"},
            },
        },
        {
            "id": "frame:transfer", "kind": "PREDICATE_FRAME", "version": 1,
            "state": "VALIDATED_FOR_PROFILE", "created_at": timestamp,
            "source_ids": ["source:internal"], "evidence_ids": ["evidence:river"],
            "context_ids": ["context:finance"],
            "payload": {
                "frame_key": "transfer",
                "version": "1.2",
                "roles": [
                    {"role": "AGENT", "description": "Entity initiating transfer"},
                    {"role": "THEME", "description": "Item transferred"},
                    {"role": "RECIPIENT", "description": "Transfer destination"},
                ],
                "custom_constraint": {"required_roles": ["AGENT", "THEME"]},
            },
        },
        {
            "id": "concept:money", "kind": "CONCEPT", "version": 1,
            "state": "ACCEPTED", "created_at": timestamp,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
            "context_ids": ["context:finance"],
            "payload": {"concept_type": "GENERAL", "namespace": "urn:test", "label": "Money"},
        },
    ])
    bundle["relations"].append({
        "id": "relation:classifies-money",
        "subject_record_id": "concept:money",
        "predicate": "INSTANCE_OF",
        "object_record_id": "class:financial-entity",
        "direction": "DIRECTED",
        "state": "ACCEPTED",
        "evidence_ids": ["evidence:lemma"],
        "context_ids": ["context:finance"],
    })
    bundle["relations"].append({
        "id": "relation:bank-sense-denotes-money",
        "subject_record_id": "sense:en:bank:financial",
        "predicate": "DENOTES",
        "object_record_id": "concept:money",
        "direction": "DIRECTED",
        "state": "ACCEPTED",
        "evidence_ids": ["evidence:financial"],
        "context_ids": ["context:finance"],
    })
    return bundle


def test_imports_semantic_classes_frames_and_directed_relations_losslessly():
    bundle = _structure_bundle()
    original = deepcopy(bundle)
    resource = USGSemanticStructureAdapter().convert(bundle)

    assert len(resource.classes) == 1
    assert resource.classes[0].class_key == "financial_entity"
    assert '"parent_class_keys":["entity"]' in resource.classes[0].payload_json
    assert len(resource.predicate_frames) == 1
    assert resource.predicate_frames[0].frame_key == "transfer"
    assert resource.predicate_frames[0].frame_version == "1.2"
    assert '"role":"AGENT"' in resource.predicate_frames[0].roles_json
    assert '"custom_constraint"' in resource.predicate_frames[0].payload_json
    assert len(resource.relations) == 2
    relation = next(item for item in resource.relations if item.predicate == "INSTANCE_OF")
    assert relation.subject_id.startswith("usg:")
    assert relation.object_id == resource.classes[0].class_id
    assert relation.subject_id != relation.object_id
    assert any(item.predicate == "DENOTES" for item in resource.relations)
    assert bundle == original


def test_candidate_structures_and_relations_are_excluded_by_default():
    bundle = _structure_bundle()
    next(row for row in bundle["records"] if row["id"] == "class:financial-entity")["state"] = "CANDIDATE"

    resource = USGSemanticStructureAdapter().convert(bundle)

    assert not resource.classes
    assert {item.predicate for item in resource.relations} == {"DENOTES"}
    reasons = {item.record_id: item.reason for item in resource.report.excluded_records}
    assert reasons["class:financial-entity"] == "lifecycle_state_not_enabled:CANDIDATE"
    assert reasons["relation:classifies-money"] == (
        "both_relation_endpoints_must_be_active_rights_permitted_supported_records"
    )


def test_restricted_source_structure_is_not_imported():
    bundle = _structure_bundle()
    record = next(row for row in bundle["records"] if row["id"] == "class:financial-entity")
    record["source_ids"] = ["source:restricted"]

    resource = USGSemanticStructureAdapter().convert(bundle)

    assert not resource.classes
    assert any(
        item.record_id == "class:financial-entity"
        and item.reason == "source_rights_not_permitted_for_local_import"
        for item in resource.report.excluded_records
    )
