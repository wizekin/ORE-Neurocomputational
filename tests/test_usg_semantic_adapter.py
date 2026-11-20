"""Tests for the read-only USG concept and explicit mapping adapter."""
from __future__ import annotations

from copy import deepcopy

from ore_usg import USGConceptAdapter
from test_usg_lexical_adapter import _bundle


def _semantic_bundle():
    bundle = deepcopy(_bundle())
    timestamp = "2026-11-12T00:00:00Z"
    for record_id, label in (("concept:river", "River bank"), ("concept:finance", "Financial institution")):
        bundle["records"].append({
            "id": record_id,
            "kind": "CONCEPT",
            "version": 1,
            "state": "ACCEPTED",
            "created_at": timestamp,
            "source_ids": ["source:permitted"],
            "evidence_ids": ["evidence:lemma"],
            "context_ids": ["context:english"],
            "payload": {
                "concept_type": "GENERAL",
                "namespace": "urn:usg:concepts",
                "label": label,
                "language": "en",
                "aliases": [],
            },
        })
    bundle["mappings"].append({
        "id": "mapping:river-related-finance",
        "source_record_id": "concept:river",
        "target_record_id": "concept:finance",
        "mapping_type": "RELATED_TO",
        "state": "ACCEPTED",
        "evidence_ids": ["evidence:lemma"],
        "context_ids": ["context:english"],
    })
    return bundle


def test_concepts_and_mapping_are_preserved_without_feature_inference_or_merging():
    bundle = _semantic_bundle()
    original = deepcopy(bundle)
    resource = USGConceptAdapter().convert(bundle)

    assert len(resource.concepts) == 2
    assert len(resource.mappings) == 1
    assert resource.mappings[0].mapping_type == "RELATED_TO"
    assert resource.mappings[0].source_concept_id != resource.mappings[0].target_concept_id
    assert all(item.semantic_features == {} for item in resource.concepts)
    assert all(item.modality_features == {} for item in resource.concepts)
    assert all(item.concept_id.startswith("usg:") for item in resource.concepts)
    assert resource.report.imported_record_ids == (
        "concept:finance", "concept:river", "mapping:river-related-finance",
    )
    assert bundle == original


def test_candidate_concepts_are_excluded_by_default_and_opted_in_explicitly():
    bundle = _semantic_bundle()
    next(row for row in bundle["records"] if row["id"] == "concept:river")["state"] = "CANDIDATE"

    default = USGConceptAdapter().convert(bundle)
    opted_in = USGConceptAdapter(include_candidate_states=True).convert(bundle)

    assert len(default.concepts) == 1
    assert not default.mappings
    assert len(opted_in.concepts) == 2
    assert len(opted_in.mappings) == 1
