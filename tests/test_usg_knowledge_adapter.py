"""Tests for the unified USG knowledge adapter."""
from __future__ import annotations

from copy import deepcopy

import pytest

from ore_usg import USGKnowledgeAdapter, USGKnowledgeResource
from test_usg_lexical_adapter import _bundle


def test_unified_adapter_assembles_all_resources_with_one_bundle_identity():
    bundle = deepcopy(_bundle())
    snapshot, resource = USGKnowledgeAdapter().load_and_convert(bundle)

    assert snapshot.bundle == bundle
    assert resource.bundle_id == bundle["bundle_id"]
    assert resource.bundle_digest == snapshot.bundle_digest
    assert resource.lexical.bundle_digest == resource.semantic.bundle_digest
    assert resource.lexical.bundle_digest == resource.structures.bundle_digest
    assert resource.contexts == resource.lexical.report.context_summaries
    assert resource.evidence == resource.lexical.report.evidence_summaries
    assert resource.sources == resource.lexical.report.source_summaries
    assert resource.lexical_entries == resource.lexical.entries
    assert resource.concepts == resource.semantic.concepts
    assert resource.relations == resource.structures.relations


def test_unified_resource_rejects_mixed_bundle_snapshots():
    resource = USGKnowledgeAdapter().convert(_bundle())
    with pytest.raises(ValueError, match="same namespace, bundle ID, and digest"):
        USGKnowledgeResource(
            lexical=resource.lexical,
            semantic=resource.semantic,
            structures=resource.structures,
            namespace=resource.namespace,
            bundle_id=resource.bundle_id,
            bundle_digest="0" * 64,
        )


def test_unified_adapter_uses_one_candidate_policy_for_all_subsystems():
    bundle = deepcopy(_bundle())
    default = USGKnowledgeAdapter().convert(bundle)
    candidate = USGKnowledgeAdapter(include_candidate_states=True).convert(bundle)

    assert len(candidate.lexical.entries) >= len(default.lexical.entries)
    assert candidate.lexical.bundle_digest == default.lexical.bundle_digest
