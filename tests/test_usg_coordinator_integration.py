"""Local acceptance coverage for Phase 5 Main Brain / USG coordination."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from ore_contracts import (
    ContractCatalog,
    ContractEnvelope,
    EvidenceKind,
    sha256_json,
    utc_now_iso,
)
from ore_contracts.module_artifacts import ContractArtifact, content_hash
from ore_blackboard import (
    BlackboardCoordinator,
    BlackboardPolicy,
    BlackboardResourceCoordinator,
    BlackboardTaskRequest,
    LexicalSpecialist,
)
from ore_blackboard.bridge import (
    BLACKBOARD_SNAPSHOT_CONTRACT,
    BLACKBOARD_USG_MODULE_KNOWLEDGE_CONTRACT,
    register_blackboard_contracts,
)
from ore_blackboard.specialists import SemanticSpecialist
from ore_blackboard.usg_integration import BlackboardUSGSnapshot
from ore_blackboard.contracts import build_input_root
from test_usg_semantic_knowledge_adapter import _phase2_bundle


def _snapshot():
    return BlackboardUSGSnapshot(deepcopy(_phase2_bundle()), expected_digest=None)


def _external_artifact(contract_id, payload, run_id, producer_id):
    return ContractArtifact(
        ContractEnvelope(
            contract_id=contract_id,
            schema_version="1.0.0",
            run_id=run_id,
            producer_id=producer_id,
            config_hash=sha256_json({}),
            created_at=utc_now_iso(),
            evidence_kind=EvidenceKind.EXTERNAL_REFERENCE,
            data_hash=content_hash(payload),
        ),
        payload,
    )


def test_router_returns_only_requested_domain_slice_and_reuses_cached_lookups():
    snapshot = _snapshot()
    router = snapshot.router

    routed = router.route(
        "ore.blackboard.lexical_specialist",
        domains=("LEXICAL", "MORPHOLOGY", "CONTEXT", "EVIDENCE"),
        lexical_terms=("bank", "mystery"),
        language="en",
        workspace_revision=1,
    )

    assert len(routed.lexical_entries) == 1
    assert routed.lexical_entries[0].orthography == "bank"
    assert len(routed.lexical_entries) < len(snapshot.resource.lexical.entries)
    assert routed.evidence
    assert routed.contexts
    assert routed.evidence_by_record
    assert routed.context_by_record
    assert routed.bundle_digest == snapshot.reference.bundle_digest
    assert routed.feedback
    assert routed.feedback[0].feedback_kind == "MISSING_LEXICAL_ENTRY"
    assert routed.feedback[0].status == "OPEN"

    before = router.statistics()
    repeated = router.route(
        "ore.blackboard.lexical_specialist",
        domains=("LEXICAL", "MORPHOLOGY", "CONTEXT", "EVIDENCE"),
        lexical_terms=("bank", "mystery"),
        language="en",
        workspace_revision=2,
    )
    cached = router.route(
        "ore.blackboard.lexical_specialist",
        domains=("LEXICAL", "MORPHOLOGY", "CONTEXT", "EVIDENCE"),
        lexical_terms=("bank", "mystery"),
        language="en",
        workspace_revision=3,
    )
    after = router.statistics()

    assert not repeated.feedback
    assert cached is repeated
    assert after["cache_hits"] > before["cache_hits"]
    assert after["cached_module_routes"] >= 2
    assert router.feedback_for("ore.blackboard.lexical_specialist") == ()


def test_downstream_feedback_is_deduplicated_and_returned_to_target_module():
    snapshot = _snapshot()
    router = snapshot.router

    first, created = router.record_feedback(
        source_module="ore.blackboard.semantic_specialist",
        target_module="ore.blackboard.lexical_specialist",
        feedback_kind="MISSING_DISAMBIGUATING_CONTEXT",
        query="bank",
        reason="Two lexical readings remain without disambiguating context.",
        source_finding_id="semantic-finding-1",
        workspace_revision=4,
    )
    repeated, repeated_created = router.record_feedback(
        source_module="ore.blackboard.semantic_specialist",
        target_module="ore.blackboard.lexical_specialist",
        feedback_kind="MISSING_DISAMBIGUATING_CONTEXT",
        query="bank",
        reason="Two lexical readings remain without disambiguating context.",
        source_finding_id="semantic-finding-2",
        workspace_revision=5,
    )

    assert created
    assert not repeated_created
    assert first.feedback_id == repeated.feedback_id

    routed = router.route(
        "ore.blackboard.lexical_specialist",
        domains=("LEXICAL", "CONTEXT", "EVIDENCE"),
        lexical_terms=("bank",),
        language="en",
    )
    delivered = next(item for item in routed.feedback if item.feedback_id == first.feedback_id)

    assert delivered.status == "OPEN"
    assert delivered.target_module == "ore.blackboard.lexical_specialist"
    assert delivered.query == "bank"
    assert delivered.to_dict()["source_mutation_performed"] is False
    assert router.feedback_for("ore.blackboard.lexical_specialist") == ()
    later_task_route = router.route(
        "ore.blackboard.lexical_specialist",
        domains=("LEXICAL", "CONTEXT", "EVIDENCE"),
        lexical_terms=("bank",),
        language="en",
        delivery_scope="later-task",
    )
    assert any(item.feedback_id == first.feedback_id for item in later_task_route.feedback)
    assert router.unannounced_feedback()

    closed = router.resolve_feedback(first.feedback_id, status="RESOLVED")
    assert closed.status == "RESOLVED"
    assert router.feedback_for(
        "ore.blackboard.lexical_specialist", delivery_scope="another-task",
    ) == ()
    assert router.unannounced_feedback() == ()


def test_blackboard_coordinator_only_requests_usg_for_specialists_that_need_it():
    coordinator = BlackboardCoordinator()

    verification_only = BlackboardTaskRequest(
        input="The statement requires verification.",
        goal="verify the statement",
        required_capabilities=("cognition.verification",),
        optional_capabilities=(),
    )
    no_usg_demands = coordinator._selected_usg_route_demands(verification_only)
    assert no_usg_demands == ()

    lexical_task = BlackboardTaskRequest(
        input="The bank is near the river.",
        goal="resolve lexical meanings",
        required_capabilities=("language.lexical_access",),
        optional_capabilities=(),
        usg_bundle=_phase2_bundle(),
    )
    demands = coordinator._selected_usg_route_demands(lexical_task)

    assert len(demands) == 1
    assert demands[0]["module_id"] == "ore.blackboard.lexical_specialist"
    assert {"LEXICAL", "CONTEXT", "EVIDENCE"}.issubset(set(demands[0]["domains"]))


def test_dynamic_specialist_input_is_a_lineage_linked_typed_slice_not_the_full_bundle():
    snapshot = _snapshot()
    coordinator = BlackboardCoordinator()
    input_root = build_input_root("bank river")
    provider = coordinator._make_usg_artifact_provider(
        snapshot,
        LexicalSpecialist(),
        run_id="phase5-routed-input",
        input_root=input_root,
        language="en",
    )
    assert provider is not None

    dynamic_inputs = provider(SimpleNamespace(state={}, revision=7))
    artifact = dynamic_inputs["usg_knowledge"]

    assert artifact.envelope.contract_id == BLACKBOARD_USG_MODULE_KNOWLEDGE_CONTRACT
    assert artifact.payload.module_id == "ore.blackboard.lexical_specialist"
    assert artifact.payload.lexical_entries
    assert len(artifact.payload.lexical_entries) < len(snapshot.resource.lexical.entries)
    assert artifact.envelope.parent_artifact_ids == (
        snapshot.knowledge_artifact("phase5-routed-input").payload_hash,
    )
    assert artifact.metadata["source_database_mutation_performed"] is False

    catalog = ContractCatalog()
    register_blackboard_contracts(catalog)
    assert catalog.validate_artifact(artifact).valid


def test_semantic_specialist_compares_support_and_contradiction_without_collapsing():
    snapshot = _snapshot()
    routed = snapshot.router.route(
        "ore.blackboard.semantic_specialist",
        domains=("SEMANTIC", "RELATIONAL", "PREDICATE", "CONTEXT", "EVIDENCE", "SOURCE_PROVENANCE"),
        lexical_terms=("bank", "river", "transfer"),
        predicate_terms=("transfer",),
        language="en",
    )
    finance = next(item for item in routed.semantic_concepts if item.label.casefold() == "finance")
    bank_entry = next(item for item in routed.lexical_entries if item.orthography == "bank")
    index_stats = snapshot.router.statistics()
    assert index_stats["semantic_alias_index_built"] is True
    assert index_stats["predicate_frame_index_built"] is True
    assert index_stats["related_edge_index_built"] is True
    financial_meaning = next(
        item for item in bank_entry.meanings
        if "financial institution" in item.gloss.casefold()
    )
    state = {
        "syntactic_hypotheses": [{
            "subject": "bank",
            "verb": "transfer",
            "object": "river",
            "finding_id": "syntax-finding-1",
        }],
        "lexical_hypotheses": {
            "bank": [{
                "gloss": financial_meaning.gloss,
                "finding_id": "bank-finding-1",
                "meaning_id": financial_meaning.meaning_id,
                "semantic_concept_ids": [finance.concept_id],
                "semantic_link_status": "LINKED",
                "evidence": [],
                "context_ids": ["context:finance"],
                "usg_references": [{
                    "bundle_id": routed.bundle_id,
                    "bundle_digest": routed.bundle_digest,
                    "record_id": financial_meaning.meaning_id,
                    "source_reference": financial_meaning.source_reference,
                }],
            }],
            "river": [{
                "gloss": "A flowing natural watercourse.",
                "finding_id": "river-finding-1",
                "meaning_id": None,
                "semantic_concept_ids": [],
                "semantic_link_status": "UNLINKED",
                "evidence": [],
                "context_ids": [],
                "usg_references": [],
            }],
            "transfer": [{
                "gloss": "To convey ownership or possession.",
                "finding_id": "transfer-finding-1",
                "meaning_id": None,
                "semantic_concept_ids": [],
                "semantic_link_status": "UNLINKED",
                "evidence": [],
                "context_ids": [],
                "usg_references": [],
            }],
        },
    }
    run_id = "phase5-semantic-evidence"
    workspace_artifact = _external_artifact(
        BLACKBOARD_SNAPSHOT_CONTRACT,
        {"workspace_id": "phase5-test", "revision": 0, "state": state},
        run_id,
        "test.workspace_snapshot",
    )
    knowledge_artifact = snapshot.module_knowledge_artifact(run_id, routed)
    result = SemanticSpecialist().execute(
        {
            "workspace_snapshot": workspace_artifact,
            "usg_knowledge": knowledge_artifact,
        },
        SimpleNamespace(run_id=run_id),
    )

    findings = result.artifacts[0].payload["findings"]
    proposition = findings[0]["payload"]

    assert proposition["evidence_conflict"] is True
    assert proposition["evidence_assertion_status"] == "CONFLICTED"
    assert proposition["usg_evidence_score_calibrated_probability"] is False
    assert {item["support_status"] for item in proposition["usg_evidence"]} >= {
        "SUPPORTS", "CONTRADICTS",
    }
    assert any(
        item["feedback_kind"] == "CONTRADICTORY_SEMANTIC_EVIDENCE"
        for item in proposition["usg_feedback_requests"]
    )
    assert result.provenance["usg_source_mutation_performed"] is False


def test_resource_coordinator_prioritizes_mandatory_knowledge_routes():
    resource_coordinator = BlackboardResourceCoordinator(
        BlackboardPolicy(),
        task_family="phase5_usg_coordination",
    )
    try:
        plan = resource_coordinator.prioritize_knowledge_routes([
            {
                "module_id": "ore.blackboard.semantic_specialist",
                "domains": ("SEMANTIC", "RELATIONAL", "EVIDENCE"),
                "priority": 0.95,
                "mandatory": False,
            },
            {
                "module_id": "ore.blackboard.lexical_specialist",
                "domains": ("LEXICAL", "CONTEXT", "EVIDENCE"),
                "priority": 1.0,
                "mandatory": True,
            },
        ])
    finally:
        resource_coordinator.close()

    assert plan["mandatory_routes_preserved"] is True
    assert plan["route_order"][0] == "ore.blackboard.lexical_specialist"
    assert all(item["heuristic"] is True for item in plan["routes"])
    assert all(item["score_definition"] for item in plan["routes"])
