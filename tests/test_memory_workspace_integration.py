"""Tests for memory recall, provenance safety, workspace publication, and module routing."""
from __future__ import annotations

from ore_language_comprehension.point07 import Point7MemoryStore
from ore_logger.database import LoggerDatabase
from ore_memory_integration import (
    EvidenceValidationPolicy,
    MemoryEncodingPolicy,
    MemoryEvidence,
    MemoryIntegrationConfig,
    MemoryIntegrationCoordinator,
    MemoryQuery,
    MemoryRetrievalCoordinator,
    SemanticMemoryAdapter,
    SourceIndependencePolicy,
    WorkingMemoryAdapter,
)


def _store() -> Point7MemoryStore:
    """Create one in-memory Point 7 store with independent episodic and semantic evidence."""
    store = Point7MemoryStore(":memory:")
    store.add_episode("older-sequence", {
        "event_core": {
            "predicate": "adopted",
            "participants_and_roles": {"agent": "Lina", "patient": "puppy"},
            "outcome": "puppy joined household",
        },
        "context": {"place": "animal shelter", "goal": "remember adoption"},
        "time": 100.0,
        "trace_strength": 0.85,
        "confidence_by_component": [0.90, 0.95],
        "source_provenance": ["witnessed"],
        "source_reference": "older observation",
        "relations": [{"subject": "Lina", "predicate": "adopted", "object": "puppy"}],
    })
    store.add_episode("current-sequence", {
        "event_core": {"predicate": "adopted", "participants_and_roles": {"agent": "Lina"}},
        "context": {"place": "animal shelter"},
        "trace_strength": 1.0,
    })
    store.add_semantic_candidate({
        "claim": "Dogs are mammals",
        "claim_key": "dogs_mammals",
        "confidence": 0.96,
        "source": "verified biology source",
        "provenance": {"source": "verified_reference"},
    }, verified=True)
    store.add_semantic_candidate({
        "claim": "All puppies are adopted in shelters",
        "claim_key": "puppies_adopted_in_shelters",
        "confidence": 0.65,
        "source": "unverified suggestion",
        "provenance": {"source": "unverified"},
    }, verified=False)
    return store


def test_recall_uses_old_records_and_excludes_current_sequence() -> None:
    """Recall ranks only prior episodes and never retrieves the input as its own memory."""
    store = _store()
    try:
        query = MemoryQuery(
            sequence_id="current-sequence",
            text="Lina adopted a puppy at an animal shelter",
            goal="recall adoption event",
            event_cues={"agent": "Lina", "predicate": "adopted", "patient": "puppy"},
            context_cues={"place": "animal shelter"},
        )
        batch = MemoryRetrievalCoordinator(store).retrieve(query)
        assert batch.diagnostics["same_sequence_excluded"] is True
        episodic = [item for item in batch.items if item.memory_type == "episodic"]
        assert episodic
        assert all(item.sequence_id != "current-sequence" for item in episodic)
        assert episodic[0].memory_id
        assert episodic[0].relevance_score > 0.0
    finally:
        store.close()


def test_pending_semantic_candidate_is_not_promoted_to_knowledge() -> None:
    """Keep accepted claims and pending candidates in separate working-memory fields."""
    store = _store()
    try:
        batch = MemoryRetrievalCoordinator(store).retrieve(MemoryQuery(
            sequence_id="new",
            text="dogs mammals puppies shelter",
            goal="understand animal biology",
        ))
        context = WorkingMemoryAdapter().to_context(batch.items)
        accepted = context["semantic_knowledge"]
        pending = context["pending_semantic_candidates"]
        assert any(item.get("claim") == "Dogs are mammals" for item in accepted.values())
        assert any(item["verification_status"] == "pending_verification" for item in pending)
        assert all(item.get("claim") != "All puppies are adopted in shelters" for item in accepted.values())
    finally:
        store.close()


def test_source_independence_prevents_duplicate_support_counting() -> None:
    """Treat downstream records derived from one episode as a single evidence group."""
    first = MemoryEvidence(
        memory_id="episode-1", memory_type="episodic", content={"event": "adopted"},
        relevance_score=0.8, confidence=0.9, independence_group="source-A",
    )
    derived = MemoryEvidence(
        memory_id="schema-1", memory_type="semantic", content={"rule": "adoption"},
        relevance_score=0.95, confidence=0.8, independence_group="source-A",
    )
    independent = MemoryEvidence(
        memory_id="episode-2", memory_type="episodic", content={"event": "rescued"},
        relevance_score=0.7, confidence=0.8, independence_group="source-B",
    )
    representatives = SourceIndependencePolicy().independent_representatives(
        (first, derived, independent)
    )
    assert {item.memory_id for item in representatives} == {"schema-1", "episode-2"}

    score = EvidenceValidationPolicy().score_hypothesis([
        {"independence_group": "source-A", "support": 0.9, "reliability": 0.9},
        {"independence_group": "source-A", "support": 0.8, "reliability": 0.9},
        {"independence_group": "source-B", "support": 0.6, "reliability": 0.8},
    ])
    assert score["independent_group_count"] == 2
    assert score["support_total"] == 0.9 * 0.9 + 0.6 * 0.8


def test_encoding_is_opt_in_and_rejects_reconstructed_content() -> None:
    """Recall and reconstruction never silently create new long-term observations."""
    policy = MemoryEncodingPolicy()
    assert not policy.evaluate(requested=False, payload={"source_provenance": ["explicit"], "confidence": 0.9})["allowed"]
    rejected = policy.evaluate(
        requested=True,
        payload={"source_provenance": ["schema_inference"], "confidence": 0.95},
    )
    assert rejected["allowed"] is False
    accepted = policy.evaluate(
        requested=True,
        payload={"source_provenance": ["explicit"], "confidence": 0.9, "event_core": {"predicate": "adopted"}},
    )
    assert accepted["allowed"] is True


def test_integration_publishes_and_routes_retrieved_memories(tmp_path) -> None:
    """Publish previous memories to the workspace and expose module-specific handoffs."""
    store = _store()
    observed = []
    handlers = {
        "semantic_processor": observed.append,
        "schema_formation": observed.append,
        "schema_guided_inference": observed.append,
        "analogical_transfer": observed.append,
        "reconstructive_memory": observed.append,
        "cognitive_control": observed.append,
        "working_memory_overload": observed.append,
    }
    config = MemoryIntegrationConfig(
        memory_database_path=":memory:",
        max_retrieval_items=8,
        max_active_memories=6,
    )
    coordinator = MemoryIntegrationCoordinator(
        store,
        config=config,
        module_handlers=handlers,
        annotation_database_path=tmp_path / "ore_logger.sqlite3",
    )
    try:
        result = coordinator.process(
            "Lina adopted a puppy at an animal shelter",
            "recall adoption event",
            sequence_id="current-sequence",
            context={"place": "animal shelter"},
            event_cues={"agent": "Lina", "predicate": "adopted", "patient": "puppy"},
            context_cues={"place": "animal shelter"},
        )
        state = result.workspace_snapshot["state"]
        assert state["retrieved_memories"]
        assert all(item.get("sequence_id") != "current-sequence" for item in state["retrieved_memories"] if item.get("memory_type") == "episodic")
        active = state["working_memory"]["current-sequence"]["active_memory_references"]
        assert active
        assert "working_memory" in result.handoffs
        assert "schema_formation" in result.handoffs
        assert "reconstructive_memory" in result.handoffs
        assert observed
        assert result.diagnostics["recall_precedes_encoding"] is True
        assert result.diagnostics["feedback_rounds"] <= 1
        assert result.encoding["write_performed"] is False
    finally:
        coordinator.close()
        store.close()


def test_annotations_are_registered_in_the_logger_database(tmp_path) -> None:
    """Index every new package definition in the existing annotations table."""
    store = _store()
    database_path = tmp_path / "ore_logger.sqlite3"
    coordinator = MemoryIntegrationCoordinator(
        store,
        config=MemoryIntegrationConfig(memory_database_path=":memory:"),
        annotation_database_path=database_path,
    )
    try:
        assert database_path.exists()
        rows = LoggerDatabase(database_path).search_annotations("MemoryIntegrationCoordinator")
        assert any(row["symbol_kind"] == "class" for row in rows)
        assert any(row["qualified_name"].endswith(".process") for row in rows)
    finally:
        coordinator.close()
        store.close()
