"""Discovery and verification acceptance tests 12-17.

12. Type-invalid candidates rejected before ranking.
13. The prediction service cannot write authoritative USG.
14. The workspace distinguishes hypotheses from verified facts.
15. Supported dispositions identify evidence/rules.
16. Contradictory fixtures are detected.
17. Low-confidence/ambiguous cases can abstain.
"""
from __future__ import annotations

import pytest

from ore_graph_intelligence.discovery import (
    AbstentionPolicy,
    GraphCandidateService,
    LinkPredictor,
    ComplementaryScorer,
    UtilityRanker,
)
from ore_graph_intelligence.projection import USGGraphProjector
from ore_graph_intelligence.schemas.candidate_schema import (
    CandidateDisposition,
    CandidateStatus,
    GraphCandidate,
    GraphQuery,
    GraphQueryAnchor,
    new_hypothesis_namespace,
)
from ore_graph_intelligence.verification import (
    CandidateVerifier,
    PromotionGate,
)

from .fixtures import contradiction_bundle, graph_bundle


@pytest.fixture()
def snapshot():
    return USGGraphProjector().project(graph_bundle())


def _sense_query(relation: str = "DENOTES"):
    return GraphQuery(
        anchor=GraphQueryAnchor(label="A financial institution."),
        relation_types=(relation,),
        budget_candidates=32,
    )


def _river_sense_query():
    return GraphQuery(
        anchor=GraphQueryAnchor(label="The land alongside a river."),
        relation_types=("DENOTES",),
        budget_candidates=32,
    )


def test_type_invalid_candidates_rejected_before_ranking(snapshot):
    """Test 12: candidates violating domain/range never reach ranking."""
    service = GraphCandidateService(snapshot)
    # A CONCEPT subject on the DENOTES relation violates the declared domain.
    concept_query = GraphQuery(
        anchor=GraphQueryAnchor(label="Money"),
        relation_types=("DENOTES",), budget_candidates=32,
    )
    kept, rejected = service.propose(concept_query)
    assert not kept, "every CONCEPT-subject DENOTES candidate must be type-invalid"
    # A sense anchor on INSTANCE_OF instead violates the domain of that relation.
    instance_query = GraphQuery(
        anchor=GraphQueryAnchor(label="Money"),
        relation_types=("INSTANCE_OF",), budget_candidates=32,
    )
    kept, rejected = service.propose(instance_query)
    assert rejected, "object type violations must be recorded as hard rejections"
    for candidate in rejected:
        assert candidate.status == CandidateStatus.REJECTED_PRE_RANK
        assert candidate.rejection_reasons
        assert candidate.utility is None, "rejected candidates must never be ranked"
    # The river sense instead produces a well-typed novel candidate.
    river_kept, _ = GraphCandidateService(snapshot).propose(_river_sense_query())
    assert river_kept, "the fixture must produce at least one well-typed novel candidate"
    for candidate in river_kept:
        assert candidate.status == CandidateStatus.PROPOSED
        assert not candidate.rejection_reasons


def test_prediction_service_cannot_write_authoritative_usg(snapshot):
    """Test 13: proposal services expose no USG write path."""
    service = GraphCandidateService(snapshot)
    write_methods = [
        name for name in dir(service)
        if name.startswith(("write", "insert", "update_", "promote", "commit"))
    ]
    assert not write_methods, "candidate services must not expose authoritative writes"
    kept, _rejected = service.propose(_sense_query())
    verifier = CandidateVerifier(snapshot)
    results = verifier.verify_all(kept)
    proposals = PromotionGate().gate(results)
    for proposal in proposals:
        assert proposal.requires_authorization == "authorized_usg_update_workflow"
        assert not hasattr(proposal, "execute"), "proposals are data, not workflows"


def test_workspace_distinguishes_hypotheses_from_verified_facts(snapshot):
    """Test 14: hypothesis and verification findings carry distinct types."""
    namespace = new_hypothesis_namespace("model-test", snapshot.graph_hash)
    assert namespace.startswith("graph_hypotheses:")
    from ore_graph_intelligence.orchestration import (
        HYPOTHESIS_FINDING_TYPE,
        VERIFICATION_FINDING_TYPE,
    )

    assert HYPOTHESIS_FINDING_TYPE != VERIFICATION_FINDING_TYPE
    from ore_workspace import PermissionPolicy, WorkspaceManager

    manager = WorkspaceManager(enable_logging=False)
    from ore_graph_intelligence.orchestration.workspace_bridge import (
        graph_workspace_permissions,
    )

    workspace = manager.create_workspace(
        "graph acceptance test",
        permissions=PermissionPolicy(graph_workspace_permissions()),
    )
    from ore_graph_intelligence.orchestration import WorkspaceGraphPublisher

    publisher = WorkspaceGraphPublisher(workspace)
    service = GraphCandidateService(snapshot)
    kept, _rejected = service.propose(_river_sense_query())
    assert kept
    receipts = [publisher.publish_hypothesis(candidate, namespace) for candidate in kept]
    assert all(receipt["finding_type"] == "graph_hypothesis" for receipt in receipts)
    results = CandidateVerifier(snapshot).verify_all(kept)
    verification_receipts = publisher.publish_verifications(results)
    assert all(
        receipt["finding_type"] == "graph_verification" for receipt in verification_receipts
    )
    types = {finding["finding_type"] for finding in publisher.workspace_findings()}
    assert "graph_hypothesis" in types
    assert "graph_verification" in types


def test_supported_disposition_identifies_evidence_and_rules(snapshot):
    """Test 15: SUPPORTED requires evidence or rules, never scores alone."""
    # A candidate matching an existing projected triple classifies as direct evidence.
    existing_triple = next(
        edge for edge in snapshot.edges if edge.relation_type == "INSTANCE_OF"
    )
    candidate = GraphCandidate(
        subject_id=existing_triple.source_id,
        relation_type="INSTANCE_OF",
        object_id=existing_triple.target_id,
        snapshot_hash=snapshot.graph_hash,
    )
    verifier = CandidateVerifier(snapshot)
    result = verifier.verify(candidate)
    assert result.disposition == CandidateDisposition.SUPPORTED
    assert result.evidence_classification == "direct_source"
    assert result.evidence_ids, "supported dispositions must identify evidence"
    assert "direct_source_support" in result.rules_applied

    # Novel candidates are model-only suggestions and can never be SUPPORTED.
    kept, _rejected = GraphCandidateService(snapshot).propose(_river_sense_query())
    novel_results = CandidateVerifier(snapshot).verify_all(kept)
    for result in novel_results:
        assert result.disposition in {
            CandidateDisposition.UNVERIFIABLE, CandidateDisposition.AMBIGUOUS,
        }, "a model-only suggestion is not a proof"
        assert result.notes


def test_contradictory_fixture_detected():
    """Test 16: a candidate contradicted by a negative edge becomes AMBIGUOUS."""
    snapshot = USGGraphProjector().project(contradiction_bundle())
    service = GraphCandidateService(snapshot)
    kept, rejected = service.propose(_sense_query())
    assert kept, "the contradiction fixture leaves the financial-sense candidate novel"
    verifier = CandidateVerifier(snapshot)
    results = verifier.verify_all(kept)
    contradicted = [
        result for result in results if result.counter_edge_ids
    ]
    assert contradicted, "the contradiction candidate must be present"
    for result in contradicted:
        assert "river" in result.triple[2] or "river" in result.triple[0], (
            "the counter-evidenced candidate is the financial/river sense pair"
        )
    for result in contradicted:
        assert result.disposition == CandidateDisposition.AMBIGUOUS, (
            "counter-evidence must be detected, never silently supported"
        )
        assert result.counter_edge_ids


def test_low_confidence_cases_can_abstain(snapshot):
    """Test 17: near-tied utilities abstain instead of forcing top-1."""
    kept, _rejected = GraphCandidateService(snapshot).propose(_river_sense_query())
    assert kept
    scored = LinkPredictor(ComplementaryScorer(snapshot)).score_all(kept)
    ranked = UtilityRanker(snapshot).rank(scored)
    policy = AbstentionPolicy(min_utility=0.99)
    candidates, decision = policy.apply(ranked)
    assert decision.abstained
    assert decision.rule == "low_confidence"
    assert candidates == []
    # A permissive policy does not abstain on the same ranking.
    permissive, permissive_decision = AbstentionPolicy(
        utility_margin=0.0, min_utility=0.0,
    ).apply(ranked)
    assert not permissive_decision.abstained
    assert permissive