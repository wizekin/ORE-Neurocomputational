"""Acceptance tests for the three-tier cognitive memory and hybrid RAG (Point 15).

Covers specification acceptance tests 34-38: the canonical USG promotion
path, episodic linkage, test-gated procedural promotion, hybrid retrieval
with provenance and ranking evaluation, and secret redaction.
"""
from __future__ import annotations

import pytest

from ore_agentic import (
    EpisodeKind,
    HybridRetriever,
    LifecycleState,
    LexicalProvider,
    MemoryKind,
    MemoryLinkStore,
    MemoryRelation,
    ProcedureNotPromotedError,
    ProcedureStatus,
    ProceduralMemory,
    PromotionReview,
    RetrievedDocument,
    SemanticMemory,
    SemanticProposal,
    USGPromotionGate,
    evaluate_ranking,
)

from agentic_point15_helpers import FakeUSGIndex, make_engine, make_request


# ------------------------------------------------------- semantic memory (34)

def test_semantic_update_uses_canonical_usg_promotion_path():
    """Acceptance 34: semantic update uses the canonical USG promotion path."""
    index = FakeUSGIndex()
    memory = SemanticMemory(usg_index=index)
    proposal = memory.propose(
        "attention span correlates with reading speed",
        concepts=("attention", "reading"),
        origin_workflow_id="wf-1")
    # Agent output is recorded as a candidate, never an authoritative fact.
    assert proposal.lifecycle is LifecycleState.CANDIDATE
    assert memory.proposal(proposal.proposal_id) is not None
    # The canonical USG resource is untouched: no write path exists.
    assert index.write_calls == []
    assert set(index.records) == {"attention"}
    # The gate blocks evidence-less proposals.
    gate = USGPromotionGate()
    review = gate.review(proposal)
    assert isinstance(review, PromotionReview)
    assert not review.eligible
    assert review.blockers == ("proposal lacks evidence references",)
    assert review.required_path == "canonical_usg_bundle_import"
    with pytest.raises(ValueError):
        gate.validate_for_import(proposal)
    # An evidenced proposal can only be marked VALIDATED_FOR_IMPORT; it still
    # cannot mutate the USG resource from agent code.
    evidenced = memory.propose(
        "attention span correlates with reading speed",
        concepts=("attention",), evidence_refs=("ev-1",),
        origin_workflow_id="wf-1")
    validated = gate.validate_for_import(evidenced)
    assert validated.lifecycle is LifecycleState.VALIDATED_FOR_IMPORT
    assert index.write_calls == []  # canonical path never bypassed


def test_semantic_lookup_goes_through_the_canonical_index():
    """Semantic retrieval reads the canonical USG index with provenance (Step 37)."""
    index = FakeUSGIndex(labels={"attention": "attention (cognitive process)"})
    memory = SemanticMemory(usg_index=index)
    records = memory.lookup("attention")
    assert len(records) == 1
    assert records[0]["record_id"] == "attention"
    assert records[0]["source_reference"] == "usg:test-bundle"
    assert memory.lookup("nonexistent") == ()


# ------------------------------------------------------- episodic memory (35)

def test_episodic_record_links_workflow_task_time_and_outcome():
    """Acceptance 35: episodic record links workflow, task, time and outcome."""
    engine = make_engine()
    result = engine.run(make_request())
    episodes = engine.episodic.recall(workflow_id=result.workflow_id)
    assert episodes
    for record in episodes:
        assert record.workflow_id == result.workflow_id
        assert record.recorded_at  # time
    task_episodes = [record for record in episodes
                     if record.kind is EpisodeKind.TASK]
    assert task_episodes
    assert all(record.task_id and record.worker_id for record in task_episodes)
    assert any(record.outcome == "succeeded" for record in task_episodes)
    outcome_episode = next(record for record in episodes
                           if record.kind is EpisodeKind.OUTCOME)
    assert outcome_episode.outcome == result.status.value
    assert outcome_episode.software_versions["ore_agentic"]


def test_episodic_memory_privacy_redaction_and_retention():
    """Episodic memory redacts secrets and applies retention policy (Step 38)."""
    engine = make_engine()
    record = engine.episodic.record(
        EpisodeKind.OBSERVATION, "observed an external call",
        workflow_id="wf-priv", payload={"api_key": "sk-abcdef123456",
                                        "safe": "value",
                                        "nested": {"password": "hunter2"}})
    assert record.redacted
    assert record.payload["api_key"] == "***REDACTED***"
    assert record.payload["nested"]["password"] == "***REDACTED***"
    assert record.payload["safe"] == "value"
    removed = engine.episodic.record(
        EpisodeKind.SNAPSHOT, "ephemeral snapshot", workflow_id="wf-priv",
        retention_class="ephemeral")
    assert removed.retention_class == "ephemeral"
    deleted = engine.episodic.apply_retention(keep_classes=("standard", "audit"))
    assert removed.episode_id in deleted
    assert engine.episodic.episode(removed.episode_id) is None
    assert engine.episodic.episode(record.episode_id) is not None


# ----------------------------------------------------- procedural memory (36)

def test_procedure_cannot_be_promoted_without_tests():
    """Acceptance 36: procedure cannot be promoted without tests."""
    memory = ProceduralMemory(min_passing_tests=1)
    procedure = memory.propose("publish workflow", "1.0.0",
                               steps=("compile", "delegate", "close"))
    assert procedure.status is ProcedureStatus.DRAFT
    with pytest.raises(ProcedureNotPromotedError):
        memory.promote(procedure.procedure_id)
    memory.attach_test(procedure.procedure_id, passed=False,
                       detail="test failed")
    with pytest.raises(ProcedureNotPromotedError):
        memory.promote(procedure.procedure_id)
    memory.attach_test(procedure.procedure_id, passed=True,
                       detail="test passed")
    promoted = memory.promote(procedure.procedure_id)
    assert promoted.status is ProcedureStatus.VALIDATED
    successor = memory.propose("publish workflow", "1.1.0")
    memory.attach_test(successor.procedure_id, passed=True)
    memory.promote(successor.procedure_id)
    superseded = memory.supersede(procedure.procedure_id,
                                  successor.procedure_id)
    assert superseded.status is ProcedureStatus.SUPERSEDED


def test_engine_proposes_procedures_only_as_drafts():
    """Engine close-out proposes procedures only after validation (Steps 39, 50)."""
    engine = make_engine()
    result = engine.run(make_request())
    assert result.procedure_proposal_ids
    for proposal_id in result.procedure_proposal_ids:
        procedure = engine.procedural.procedure(proposal_id)
        assert procedure.status is ProcedureStatus.DRAFT
        with pytest.raises(ProcedureNotPromotedError):
            engine.procedural.promote(proposal_id)


# ------------------------------------------------------- memory links (40)

def test_memory_types_link_without_conflation():
    """Acceptance 40: memory types link through typed relations, never conflated."""
    links = MemoryLinkStore()
    links.link(MemoryKind.SEMANTIC, "sem-1", MemoryRelation.VALIDATED_BY,
               MemoryKind.EPISODIC, "ep-1")
    links.link(MemoryKind.PROCEDURAL, "proc-1", MemoryRelation.DERIVED_FROM,
               MemoryKind.EPISODIC, "ep-1")
    links.link(MemoryKind.PROCEDURAL, "proc-1", MemoryRelation.SUPERSEDES,
               MemoryKind.PROCEDURAL, "proc-0")
    assert len(links.links()) == 3
    assert links.neighbors("ep-1") == ("sem-1", "proc-1")
    assert links.neighbors("proc-1", relation=MemoryRelation.SUPERSEDES) == (
        "proc-0",)
    # The relation vocabulary is closed and typed.
    assert {relation.value for relation in MemoryRelation} == {
        "derived_from", "uses_concept", "validated_by", "observed_in",
        "supersedes"}
    # Engine close-out records the links between tiers.
    engine = make_engine()
    engine.run(make_request())
    assert engine.links.links()
    assert any(link.relation is MemoryRelation.DERIVED_FROM
               for link in engine.links.links())


# ------------------------------------------------------ hybrid RAG (37)

def test_hybrid_retrieval_returns_provenance_and_supports_ranking_evaluation():
    """Acceptance 37: hybrid retrieval returns provenance and supports ranking."""
    index = FakeUSGIndex(
        labels={"attention": "attention (cognitive process)"},
        relations=(("attention", "modulates", "reading_speed"),))
    documents = (
        RetrievedDocument(doc_id="doc-1", content="attention controls reading",
                          source_id="corpus-a", locator="p.1", authority=0.6),
        RetrievedDocument(doc_id="doc-2", content="unrelated gardening notes",
                          source_id="corpus-b", locator="p.9", authority=0.9),
        RetrievedDocument(doc_id="doc-3", content="reading speed and attention "
                          "span measured together", source_id="corpus-a",
                          locator="p.4", authority=0.7),
    )
    from ore_agentic import GraphProvider, SemanticProvider

    retriever = HybridRetriever((LexicalProvider(documents),
                                 SemanticProvider(index),
                                 GraphProvider(index)))
    result = retriever.retrieve("attention", limit=5)
    assert result.ranked
    top = result.ranked[0]
    assert top.document.doc_id == "usg:attention"  # exact semantic match first
    assert top.exact_id_match
    assert top.document.source_id == "usg:test-bundle"
    assert top.document.locator  # provenance: source identity + location
    assert 0.0 <= top.score <= 3.0  # blended score with boosts
    # Graph traversal surfaced the relation document.
    assert any(item.document.doc_id == "usg-relation:rel-0"
               for item in result.ranked)
    # Ranking evaluation metrics are computable on labeled relevance.
    metrics = evaluate_ranking(result, relevant_ids=("usg:attention", "doc-1"),
                               k=5)
    assert 0.0 <= metrics.recall_at_k <= 1.0
    assert metrics.mrr > 0.0
    assert metrics.ndcg > 0.0
    assert metrics.recall_at_k == 1.0  # both relevant docs retrieved in top-5


def test_retrieval_metadata_filtering_and_freshness():
    """Metadata filters and freshness weighting shape retrieval (Steps 41-42)."""
    documents = (
        RetrievedDocument(doc_id="doc-1", content="attention reading",
                          source_id="a", authority=0.5,
                          metadata={"kind": "primary"}),
        RetrievedDocument(doc_id="doc-2", content="attention reading",
                          source_id="b", authority=0.9,
                          metadata={"kind": "summary"}),
    )
    retriever = HybridRetriever((LexicalProvider(documents),),
                                metadata_filter={"kind": "primary"})
    result = retriever.retrieve("attention reading")
    assert [item.document.doc_id for item in result.ranked] == ["doc-1"]
    # The blended score is a ranking score, not a probability.
    from ore_agentic import hybrid_retrieval_score
    with pytest.raises(ValueError):
        hybrid_retrieval_score(1.5, 0.0, 0.0, 0.0, 0.0,
                               weights={"lexical": 1.0})


# ---------------------------------------------------------- redaction (38)

def test_secrets_are_redacted_from_logs_and_model_visible_outputs():
    """Acceptance 38: secrets are redacted from logs and model-visible outputs."""
    from ore_agentic import Tracer, redact, redact_value

    payload = {"api_key": "sk-abcdef123456",
               "nested": {"authorization": "Bearer abcdef123456",
                          "safe": "value"},
               "note": "token=abc123 and password=hunter2",
               "count": 3}
    safe = redact(payload)
    assert safe["api_key"] == "***REDACTED***"
    assert safe["nested"]["authorization"] == "***REDACTED***"
    assert "hunter2" not in str(safe)
    assert safe["nested"]["safe"] == "value"
    assert safe["count"] == 3
    assert redact_value("key sk-abcdef123456 here") == (
        "key ***REDACTED*** here")
    tracer = Tracer(__import__("ore_agentic").TraceContext.create())
    with tracer.span("operation", api_key="sk-abcdef123456", safe="ok"):
        pass
    span = tracer.spans()[0]
    assert span.attributes["api_key"] == "***REDACTED***"
    assert span.attributes["safe"] == "ok"
    assert "api_key" in span.redacted_fields
    # Engine spans carry no secret material.
    engine = make_engine()
    result = engine.run(make_request())
    serialized = str(result.spans) + str(result.trace)
    assert "sk-" not in serialized
    assert "password" not in serialized.lower()


def test_semantic_proposal_serialization_round_trip():
    """Semantic proposals serialize with lifecycle and provenance (Step 37)."""
    proposal = SemanticProposal(
        proposal_id="sem-1", statement="a durable fact",
        concepts=("c1",), relations=(("c1", "modulates", "c2"),),
        evidence_refs=("ev-1",), origin_workflow_id="wf-1",
        origin_task_id="t-1")
    serialized = proposal.to_dict()
    assert serialized["lifecycle"] == "CANDIDATE"
    assert serialized["relations"] == [["c1", "modulates", "c2"]]
    assert serialized["origin_workflow_id"] == "wf-1"
    with pytest.raises(ValueError):
        SemanticProposal(proposal_id="sem-2", statement="  ")
