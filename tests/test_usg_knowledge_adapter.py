"""Tests for the unified USG knowledge adapter."""
from __future__ import annotations

from copy import deepcopy

import pytest

from ore_contracts import ComputeBudget, ContractCatalog, ModuleStatus, OperationMode, RunContext
from ore_contracts.module_registry import ModuleRegistry
from ore_contracts.resource_coordination import AdaptiveResourceCoordinator
from ore_contracts.resource_profiles import ModuleResourceEstimate
from ore_usg import (
    USGKnowledgeAdapter,
    USGKnowledgeResource,
    make_usg_bundle_artifact,
    register_usg_knowledge_module,
    coordinate_usg_knowledge_import,
)
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


def test_unified_knowledge_module_registers_and_emits_one_typed_artifact():
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_usg_knowledge_module(registry)
    context = RunContext(
        run_id="usg-knowledge-adapter-test",
        mode=OperationMode.FUNCTIONAL,
        deterministic_seed=19,
        compute_budget=ComputeBudget(),
    )
    bundle = _bundle()
    source = make_usg_bundle_artifact(
        bundle,
        run_id=context.run_id,
        producer_id="test.usg_bundle",
    )
    backend = registry.get_backend("ore.usg_knowledge_import")
    manifest = registry.get_manifest("ore.usg_knowledge_import")
    assert backend is not None
    assert manifest is not None
    assert "language.semantic_resource.import" in manifest.capabilities
    assert "language.semantic_structure.import" in manifest.capabilities

    result = backend.execute({"knowledge_bundle": source}, context)

    assert result.status is ModuleStatus.SUCCESS
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert catalog.validate_artifact(artifact).valid
    assert artifact.envelope.contract_id == "ORE.USG.KNOWLEDGE"
    assert artifact.payload.lexical.bundle_digest == artifact.payload.bundle_digest
    assert artifact.metadata["source_database_mutation_performed"] is False
    assert artifact.envelope.parent_artifact_ids == (source.payload_hash,)


def test_unified_usg_import_is_routed_by_adaptive_resource_coordinator(tmp_path):
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_usg_knowledge_module(registry)
    coordinator = AdaptiveResourceCoordinator(
        registry,
        catalog,
        repository_root=__import__("pathlib").Path(__file__).resolve().parents[1],
        logger_database_path=tmp_path / "ore_logger.sqlite3",
        enable_logging=False,
        declared_estimates={
            "ore.usg_knowledge_import": ModuleResourceEstimate(
                "ore.usg_knowledge_import", 20.0, 16.0, 0.1, 0.95, 0.99,
                "declared_test_estimate",
            ),
        },
    )
    context = RunContext(
        run_id="usg-adaptive-import-test",
        mode=OperationMode.FUNCTIONAL,
        deterministic_seed=23,
        compute_budget=ComputeBudget(),
    )
    source = make_usg_bundle_artifact(
        _bundle(),
        run_id=context.run_id,
        producer_id="test.usg_bundle",
    )

    result = coordinate_usg_knowledge_import(
        coordinator, source, "usg-adaptive-import-task", context=context,
    )

    assert result.completed
    assert result.total_module_calls == 1
    assert result.final_support["language.lexical_resource.import"] == 1.0
    assert result.final_support["language.semantic_resource.import"] == 1.0
    assert result.final_support["language.semantic_structure.import"] == 1.0
    assert result.final_support["language.knowledge.provenance"] == 1.0
    assert "ore.usg_knowledge_import" in result.module_results
