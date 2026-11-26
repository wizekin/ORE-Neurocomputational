"""Local tests for Phase 4 unified USG knowledge access and registry routing."""
from __future__ import annotations

from dataclasses import replace

from ore_contracts import (
    ComputeBudget,
    ContractCatalog,
    ModuleStatus,
    OperationMode,
    RunContext,
)
from ore_contracts.module_registry import ModuleRegistry
from ore_usg import (
    ORE_USG_RETRIEVAL_CONTRACT,
    USGKnowledgeAccess,
    USGKnowledgeAdapter,
    USGKnowledgeQuery,
    USG_KNOWLEDGE_ACCESS_MODULE_ID,
    build_lexical_access_processor,
    build_semantic_cognition_processor,
    build_semantic_composition_processor,
    make_usg_bundle_artifact,
    make_usg_knowledge_query_artifact,
    register_usg_knowledge_module,
)
from test_usg_semantic_knowledge_adapter import _phase2_bundle


def _resource():
    return USGKnowledgeAdapter().convert(_phase2_bundle())


def test_unified_resource_exposes_cross_domain_references_and_snapshot_identity():
    resource = _resource()

    assert resource.resource_id.endswith(resource.bundle_digest)
    assert {
        "LEXICAL", "SEMANTIC", "RELATIONAL", "PREDICATE", "CONTEXT",
        "EVIDENCE", "SOURCE_PROVENANCE",
    }.issubset(set(resource.available_domains))
    references = resource.reference_inventory
    assert "lexical_to_semantic" in references
    assert "context_assignments" in references
    assert "evidence_attachments" in references
    assert references["predicate_frames"] == resource.predicate_frames
    assert resource.semantic_relations == resource.semantic_knowledge.relations


def test_exact_lookup_returns_lexical_meanings_concepts_frames_and_provenance():
    resource = _resource()
    access = USGKnowledgeAccess(resource)

    entries = access.lexical_entries("BANK", language="en")
    meanings = access.lexical_meanings("bank", language="en")
    concepts = access.semantic_concepts("Finance", language="en")
    frames = access.predicate_frames("transfer", executable_only=True)
    exact = access.query(USGKnowledgeQuery(
        mode="EXACT", query="bank", kind="LEXICAL_ENTRY", language="en",
    ))
    exact_by_id = access.query(USGKnowledgeQuery(
        mode="EXACT", record_id=entries[0].entry_id,
    ))

    assert entries
    assert meanings
    assert concepts
    assert len(frames) == 1
    assert frames[0].predicate == "transfer"
    assert exact.evidence
    assert exact.source_provenance
    assert exact_by_id.candidates[0].record_id == entries[0].entry_id


def test_related_lookup_follows_explicit_links_and_keeps_uncertainty_visible():
    resource = _resource()
    access = USGKnowledgeAccess(resource)
    finance = access.semantic_concepts("Finance", language="en")[0]

    result = access.query(USGKnowledgeQuery(
        mode="RELATED_CONCEPT",
        record_id=finance.concept_id,
    ))

    assert result.candidates
    assert result.relations
    assert result.uncertainty_present
    assert any(evidence.support_status == "CONTRADICTS" for evidence in result.evidence)
    assert result.source_provenance
    assert all(item.relationship_direction == "DIRECTED" for item in result.candidates)
    assert all(item.confidence_value is None for item in result.candidates)
    assert result.confidence_available is False
    assert result.bundle_digest == resource.bundle_digest


def test_contextual_lookup_uses_only_explicit_context_assignments():
    resource = _resource()
    access = USGKnowledgeAccess(resource)
    finance = access.semantic_concepts("Finance", language="en")[0]

    result = access.query(USGKnowledgeQuery(
        mode="CONTEXTUAL",
        query="bank",
        kind="LEXICAL_ENTRY",
        language="en",
        context_type="DOMAIN",
        context_value="finance",
    ))

    assert result.contexts
    assert any(item.context_id == "context:finance" for item in result.contexts)
    assert any(item.record_id == finance.concept_id for item in result.candidates)
    assert result.bundle_id == resource.bundle_id


def test_missing_optional_integration_domains_do_not_break_other_retrieval():
    original = _resource()
    partial = replace(original, language_integration=None, semantic_knowledge=None)
    access = USGKnowledgeAccess(partial)

    exact = access.query(USGKnowledgeQuery(
        mode="EXACT", query="bank", kind="LEXICAL_ENTRY", language="en",
    ))
    assert exact.candidates
    assert "CONTEXT" in exact.unavailable_domains
    assert "SOURCE_PROVENANCE" in exact.unavailable_domains
    assert access.semantic_concepts("bank", language="en")
    assert build_semantic_cognition_processor(partial)
    assert build_semantic_composition_processor(partial)


def test_ore_language_processor_bridges_share_the_same_knowledge_access():
    resource = _resource()

    lexical = build_lexical_access_processor(resource)
    semantic = build_semantic_cognition_processor(resource)
    composition = build_semantic_composition_processor(resource)

    assert isinstance(lexical.usg_knowledge, USGKnowledgeAccess)
    assert isinstance(semantic.usg_knowledge, USGKnowledgeAccess)
    assert isinstance(composition.usg_knowledge, USGKnowledgeAccess)
    assert lexical.usg_knowledge.lexical_meanings("bank", language="en")
    assert semantic.usg_knowledge.semantic_relations(
        resource.semantic_concepts[0].concept_id
    ) is not None
    assert composition.usg_knowledge.predicate_frames("transfer")


def test_registry_access_module_validates_contracts_and_tracks_both_parent_artifacts():
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    importer = register_usg_knowledge_module(registry)

    assert registry.get_backend(USG_KNOWLEDGE_ACCESS_MODULE_ID) is not None
    capability = registry.find_by_capability("language.knowledge.related_lookup")
    assert any(item.module_id == USG_KNOWLEDGE_ACCESS_MODULE_ID for item in capability)

    context = RunContext(
        run_id="usg-phase4-access",
        mode=OperationMode.FUNCTIONAL,
        deterministic_seed=41,
        compute_budget=ComputeBudget(),
    )
    source = make_usg_bundle_artifact(
        _phase2_bundle(),
        run_id=context.run_id,
        producer_id="test.usg_phase4",
    )
    imported = importer.execute({"knowledge_bundle": source}, context)
    assert imported.status is ModuleStatus.SUCCESS
    knowledge_artifact = imported.artifacts[0]
    assert catalog.validate_artifact(knowledge_artifact).valid

    request_artifact = make_usg_knowledge_query_artifact(
        USGKnowledgeQuery(
            mode="EXACT", query="bank", kind="LEXICAL_ENTRY", language="en",
        ),
        run_id=context.run_id,
    )
    access_backend = registry.get_backend(USG_KNOWLEDGE_ACCESS_MODULE_ID)
    retrieved = access_backend.execute(
        {"knowledge": knowledge_artifact, "request": request_artifact},
        context,
    )

    assert retrieved.status is ModuleStatus.SUCCESS
    output = retrieved.artifacts[0]
    assert output.envelope.contract_id == ORE_USG_RETRIEVAL_CONTRACT
    assert catalog.validate_artifact(output).valid
    assert output.envelope.parent_artifact_ids == (
        knowledge_artifact.payload_hash,
        request_artifact.payload_hash,
    )
    assert output.metadata["confidence_available"] is False
