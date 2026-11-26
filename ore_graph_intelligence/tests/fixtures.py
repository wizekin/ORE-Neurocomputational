"""Deterministic USG bundle fixtures for the graph intelligence test suite.

The fixtures follow the exact USG knowledge-bundle contract used by ``ore_usg``
adapters: records (with kind-specific payload fields), sources with rights
metadata, and typed directed relations with optional qualifiers. Bundles are
plain dictionaries so tests can deepcopy and mutate them per scenario.
"""
from __future__ import annotations

from copy import deepcopy

TIMESTAMP = "2026-11-12T00:00:00Z"
NAMESPACE = "urn:usg:graph-intelligence-test"


def graph_bundle() -> dict:
    """A rich deterministic bundle: senses, concepts, classes, a frame, typed relations."""
    bundle = {
        "contract": "USG-KNOWLEDGE/0.1.0",
        "namespace": NAMESPACE,
        "bundle_id": "ore-graph-intelligence-test",
        "created_at": TIMESTAMP,
        "records": [
            {
                "id": "lemma:en:bank", "kind": "LEMMA", "version": 1,
                "state": "ACCEPTED", "created_at": TIMESTAMP,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
                "context_ids": ["context:english"],
                "payload": {"lemma": "bank", "language": "en", "part_of_speech": "NOUN"},
            },
            {
                "id": "sense:en:bank:financial", "kind": "LEXICAL_SENSE", "version": 1,
                "state": "ACCEPTED", "created_at": TIMESTAMP,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:financial"],
                "context_ids": ["context:finance"],
                "payload": {
                    "lemma_record_id": "lemma:en:bank", "language": "en",
                    "definition": "A financial institution.",
                    "sense_source_id": "source:permitted",
                },
            },
            {
                "id": "sense:en:bank:river", "kind": "LEXICAL_SENSE", "version": 1,
                "state": "VALIDATED_FOR_PROFILE", "created_at": TIMESTAMP,
                "source_ids": ["source:internal"], "evidence_ids": ["evidence:river"],
                "context_ids": ["context:geography"],
                "payload": {
                    "lemma_record_id": "lemma:en:bank", "language": "en",
                    "definition": "The land alongside a river.",
                    "sense_source_id": "source:internal",
                },
            },
            {
                "id": "form:en:banks", "kind": "LEXICAL_FORM", "version": 1,
                "state": "ACCEPTED", "created_at": TIMESTAMP,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:form"],
                "context_ids": ["context:english"],
                "payload": {"form": "banks", "language": "en", "script": "Latn"},
            },
            {
                "id": "concept:money", "kind": "CONCEPT", "version": 1,
                "state": "ACCEPTED", "created_at": TIMESTAMP,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:concept"],
                "context_ids": ["context:finance"],
                "payload": {"concept_type": "GENERAL", "namespace": "urn:test", "label": "Money"},
            },
            {
                "id": "concept:river", "kind": "CONCEPT", "version": 1,
                "state": "ACCEPTED", "created_at": TIMESTAMP,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:concept-river"],
                "context_ids": ["context:geography"],
                "payload": {"concept_type": "GENERAL", "namespace": "urn:test", "label": "River"},
            },
            {
                "id": "class:financial-entity", "kind": "SEMANTIC_CLASS", "version": 1,
                "state": "ACCEPTED", "created_at": TIMESTAMP,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
                "context_ids": ["context:finance"],
                "payload": {
                    "class_key": "financial_entity",
                    "definition": "An entity that participates in financial activity.",
                    "parent_class_keys": ["entity"],
                },
            },
            {
                "id": "class:entity", "kind": "SEMANTIC_CLASS", "version": 1,
                "state": "ACCEPTED", "created_at": TIMESTAMP,
                "source_ids": ["source:internal"], "evidence_ids": ["evidence:class"],
                "context_ids": ["context:general"],
                "payload": {"class_key": "entity", "definition": "A general thing."},
            },
            {
                "id": "frame:transfer", "kind": "PREDICATE_FRAME", "version": 1,
                "state": "VALIDATED_FOR_PROFILE", "created_at": TIMESTAMP,
                "source_ids": ["source:internal"], "evidence_ids": ["evidence:river"],
                "context_ids": ["context:finance"],
                "payload": {
                    "frame_key": "transfer", "version": "1.2",
                    "roles": [
                        {"role": "AGENT", "description": "Entity initiating transfer",
                         "required": True},
                        {"role": "THEME", "description": "Item transferred"},
                        {"role": "RECIPIENT", "description": "Transfer destination"},
                    ],
                    "custom_constraint": {"required_roles": ["AGENT", "THEME"]},
                },
            },
            {
                "id": "lemma:en:restricted", "kind": "LEMMA", "version": 1,
                "state": "ACCEPTED", "created_at": TIMESTAMP,
                "source_ids": ["source:restricted"], "evidence_ids": ["evidence:restricted"],
                "context_ids": ["context:english"],
                "payload": {"lemma": "restricted", "language": "en", "part_of_speech": "ADJ"},
            },
        ],
        "sources": _sources(),
        "evidence": _evidence(),
        "contexts": _contexts(),
        "mappings": [],
        "relations": _relations(),
    }
    return bundle


def _sources() -> list[dict]:
    return [
        {
            "id": "source:permitted", "provider": "Graph fixture", "source_type": "OTHER",
            "source_locator": None, "retrieved_at": TIMESTAMP, "license_id": "MIT",
            "rights_status": "KNOWN_PERMITTED", "redistribution": "ALLOWED",
            "attribution": "Graph fixture source",
        },
        {
            "id": "source:internal", "provider": "Internal fixture", "source_type": "USG_DOCUMENT",
            "source_locator": None, "retrieved_at": TIMESTAMP, "license_id": None,
            "rights_status": "INTERNAL", "redistribution": "NOT_APPLICABLE",
            "attribution": "Internal fixture source",
        },
        {
            "id": "source:restricted", "provider": "Restricted fixture", "source_type": "DICTIONARY",
            "source_locator": None, "retrieved_at": TIMESTAMP, "license_id": None,
            "rights_status": "UNKNOWN", "redistribution": "UNKNOWN",
            "attribution": "Restricted fixture source",
        },
    ]


def _evidence() -> list[dict]:
    entries = (
        ("evidence:lemma", "source:permitted"),
        ("evidence:financial", "source:permitted"),
        ("evidence:river", "source:internal"),
        ("evidence:form", "source:permitted"),
        ("evidence:concept", "source:permitted"),
        ("evidence:concept-river", "source:permitted"),
        ("evidence:class", "source:internal"),
        ("evidence:restricted", "source:restricted"),
    )
    return [
        {
            "id": evidence_id, "source_id": source_id, "evidence_type": "DEFINITION",
            "support_status": "SUPPORTS", "captured_at": TIMESTAMP, "transformation_ids": [],
        }
        for evidence_id, source_id in entries
    ]


def _contexts() -> list[dict]:
    return [
        {"id": "context:english", "context_type": "LANGUAGE", "value": "en"},
        {"id": "context:finance", "context_type": "DOMAIN", "value": "finance"},
        {"id": "context:geography", "context_type": "DOMAIN", "value": "geography"},
        {"id": "context:general", "context_type": "DOMAIN", "value": "general"},
    ]


def _relations() -> list[dict]:
    return [
        {
            "id": "relation:classifies-money", "subject_record_id": "concept:money",
            "predicate": "INSTANCE_OF", "object_record_id": "class:financial-entity",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:lemma"], "context_ids": ["context:finance"],
        },
        {
            "id": "relation:bank-sense-denotes-money", "subject_record_id": "sense:en:bank:financial",
            "predicate": "DENOTES", "object_record_id": "concept:money",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:financial"], "context_ids": ["context:finance"],
        },
        {
            "id": "relation:river-sense-denotes-river", "subject_record_id": "sense:en:bank:river",
            "predicate": "DENOTES", "object_record_id": "concept:river",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:river"], "context_ids": ["context:geography"],
        },
        {
            "id": "relation:negated-bank-sense-denotes-river",
            "subject_record_id": "sense:en:bank:financial",
            "predicate": "DENOTES", "object_record_id": "concept:river",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:river"], "context_ids": ["context:geography"],
            "payload": {
                "qualifiers": {
                    "polarity": "negative",
                    "modality": "factual",
                    "temporal_scope": "2026",
                },
            },
        },
        {
            "id": "relation:transfer-money", "subject_record_id": "concept:money",
            "predicate": "transfer", "object_record_id": "concept:river",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:lemma"], "context_ids": ["context:finance"],
            "payload": {"arguments": {
                "AGENT": "concept:money",
                "THEME": "concept:river",
                "RECIPIENT": "sense:en:bank:financial",
            }},
        },
        {
            "id": "relation:money-part-of-restricted", "subject_record_id": "concept:money",
            "predicate": "PART_OF", "object_record_id": "lemma:en:restricted",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:lemma"], "context_ids": ["context:finance"],
        },
    ]


def contradiction_bundle() -> dict:
    """Bundle with a reversed negative DENOTES edge: novel candidate hits counter-evidence."""
    from copy import deepcopy

    bundle = deepcopy(graph_bundle())
    bundle["relations"] = [
        relation for relation in bundle["relations"]
        if relation["id"] != "relation:negated-bank-sense-denotes-river"
    ]
    bundle["relations"].append({
        "id": "relation:negated-river-denotes-financial",
        "subject_record_id": "concept:river",
        "predicate": "DENOTES",
        "object_record_id": "sense:en:bank:financial",
        "direction": "DIRECTED", "state": "ACCEPTED",
        "evidence_ids": ["evidence:river"], "context_ids": ["context:geography"],
        "payload": {
            "qualifiers": {"polarity": "negative", "modality": "factual"},
        },
    })
    return bundle


def bundle_with_dangling_only() -> dict:
    """Bundle whose single relation targets a restricted (non-admitted) record."""
    bundle = deepcopy(graph_bundle())
    bundle["relations"] = [
        relation for relation in bundle["relations"]
        if relation["id"] == "relation:money-part-of-restricted"
    ]
    return bundle


__all__ = [
    "NAMESPACE", "TIMESTAMP", "bundle_with_dangling_only", "contradiction_bundle",
    "graph_bundle",
]
