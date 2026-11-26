"""Orchestration acceptance tests 23-28: budgets, caches, audit, rollback, isolation.

23. Oversized tasks are sampled/deferred/rejected within budget.
24. Snapshot/model changes invalidate stale caches.
25. SQLite events reconstruct the candidate lifecycle.
26. Previous model/config rollback happens without USG changes.
27. Access controls prevent restricted features leaking.
28. Model crash/timeout does not block exact USG reasoning.
"""
from __future__ import annotations

import sqlite3

from ore_graph_intelligence.nar import (
    NARTrainer,
    NeuralAlgorithmicService,
    TaskGenerator,
    TraceDataset,
)
from ore_graph_intelligence.observability import CandidateLifecycleAudit, GraphEventRecorder
from ore_graph_intelligence.orchestration import (
    AdaptiveResourceCoordinator,
    CancelToken,
    GraphIntelligenceCoordinator,
    GraphIntelligenceTask,
    GraphResultCache,
    ModelRegistry,
    ResourceBudget,
    CacheKey,
)
from ore_graph_intelligence.projection import USGGraphProjector
from ore_graph_intelligence.schemas.candidate_schema import (
    GraphQuery,
    GraphQueryAnchor,
)
from ore_graph_intelligence.schemas.model_manifest import ModelManifest

from .fixtures import graph_bundle


def _river_query():
    return GraphQuery(
        anchor=GraphQueryAnchor(label="The land alongside a river."),
        relation_types=("DENOTES",), budget_candidates=32,
    )


def _manifest(version: str) -> ModelManifest:
    return ModelManifest(
        model_id="embeddings", model_type="TransE", version=version,
        weights_hash=f"{version}{'0' * (64 - len(version))}",
        snapshot_hash="f" * 64, split_digest="e" * 64,
        trained_at_utc="2026-11-22T00:00:00+00:00",
    )


def test_oversized_tasks_sampled_deferred_rejected():
    """Test 23: admission control samples, defers or rejects oversized tasks."""
    coordinator = AdaptiveResourceCoordinator(ResourceBudget(
        deadline_ms=1.0, max_ram_mb=1.0, max_candidates=4, max_models=1,
    ))
    from ore_graph_intelligence.orchestration.resource_estimator import ResourceEstimate

    # Reject: RAM exceeds the budget.
    huge = ResourceEstimate(
        n_nodes=10, n_edges=10, neighborhood_nodes=8, estimated_ram_mb=999.0,
        model_load_mb=1.0, estimated_latency_ms=0.5, candidate_budget=4,
    )
    decision = coordinator.admit(huge)
    assert decision.decision == "reject"
    # Sample: candidate budget beyond the cap gets sampled down.
    oversampled = ResourceEstimate(
        n_nodes=10, n_edges=10, neighborhood_nodes=8, estimated_ram_mb=0.5,
        model_load_mb=0.0, estimated_latency_ms=0.1, candidate_budget=64,
    )
    decision = coordinator.admit(oversampled)
    assert decision.decision == "sample"
    assert decision.sampled_candidate_budget <= 4
    # Defer: latency over deadline with a small budget defers.
    slow = ResourceEstimate(
        n_nodes=10, n_edges=10, neighborhood_nodes=8, estimated_ram_mb=0.5,
        model_load_mb=0.0, estimated_latency_ms=10.0, candidate_budget=4,
    )
    decision = coordinator.admit(slow)
    assert decision.decision == "defer"
    assert decision.deferred_retry_after_ms is not None


def test_snapshot_and_model_changes_invalidate_stale_caches():
    """Test 24: cache entries die when the snapshot hash or model version changes."""
    cache = GraphResultCache(capacity=8)
    old_key = CacheKey("snapshot-a", "model-1", "candidates")
    cache.put(old_key, {"result": 1})
    new_snapshot_key = CacheKey("snapshot-b", "model-1", "candidates")
    cache.put(new_snapshot_key, {"result": 2})
    assert cache.get(CacheKey("snapshot-b", "model-1", "candidates")) is not None
    removed = cache.invalidate_stale(snapshot_hash="snapshot-b")
    assert removed == 1, "entries from stale snapshots must be invalidated"
    assert cache.get(old_key) is None
    model_key = CacheKey("snapshot-b", "model-1", "candidates")
    cache.put(model_key, {"result": 3})
    removed = cache.invalidate_stale(model_version="model-2")
    assert removed == 1, "entries from stale model versions must be invalidated"
    assert cache.get(model_key) is None
    assert cache.stats()["invalidations"] == 2


def test_sqlite_events_reconstruct_candidate_lifecycle(tmp_path):
    """Test 25: ore_logger SQLite events rebuild the per-candidate lifecycle."""
    from ore_logger.database import LoggerDatabase

    database = LoggerDatabase(tmp_path / "graph-lifecycle.sqlite")
    recorder = GraphEventRecorder(logger_database=database)
    coordinator = GraphIntelligenceCoordinator(event_recorder=recorder)
    bundle = graph_bundle()
    task = GraphIntelligenceTask(
        task_type="link_discovery", query=_river_query(), bundle=bundle,
    )
    result = coordinator.execute(task)
    assert result.candidates, "the fixture must produce candidates to audit"

    audit = CandidateLifecycleAudit()
    trail = audit.audit_correlation(recorder, result.correlation_id)
    assert trail.event_count > 0
    assert trail.candidate_audits
    verified_ids = {v.candidate_id for v in result.verification}
    for candidate_audit in trail.candidate_audits:
        assert "proposed" in candidate_audit.stages
        if candidate_audit.candidate_id in verified_ids:
            assert "verified" in candidate_audit.stages
            assert candidate_audit.final_disposition is not None

    # Reconstruct the same lifecycle from the SQLite rows written by ore_logger.
    with sqlite3.connect(str(tmp_path / "graph-lifecycle.sqlite")) as connection:
        connection.row_factory = sqlite3.Row
        rows = [
            dict(row) for row in connection.execute(
                "SELECT sequence_no, timestamp_utc, event_kind, metadata_json "
                "FROM events WHERE run_id=? ORDER BY sequence_no",
                (recorder.run_id,),
            )
        ]
    assert rows, "lifecycle events must be persisted to SQLite"
    sqlite_trail = audit.audit_sqlite_rows(rows)
    assert sqlite_trail.event_count == trail.event_count
    assert len(sqlite_trail.candidate_audits) == len(trail.candidate_audits)
    sqlite_stages = {
        item.candidate_id: item.stages for item in sqlite_trail.candidate_audits
    }
    for item in trail.candidate_audits:
        assert sqlite_stages[item.candidate_id] == item.stages


def test_model_rollback_without_usg_changes():
    """Test 26: rollback swaps model versions only; USG state is never touched."""
    registry = ModelRegistry()
    bundle = graph_bundle()
    snapshot_hash = USGGraphProjector().project(bundle).graph_hash
    first = registry.register("embeddings", _manifest("1"))
    registry.promote("embeddings", first.version, to_stage="monitored")
    assert registry.current("embeddings").version == "1"
    second = registry.register("embeddings", _manifest("2"))
    registry.promote("embeddings", second.version, to_stage="monitored")
    assert registry.current("embeddings").version == "2"
    rolled = registry.rollback("embeddings")
    assert rolled is not None and rolled.version == "1"
    assert registry.current("embeddings").version == "1"
    # The USG bundle is unchanged by every registry operation.
    assert USGGraphProjector().project(bundle).graph_hash == snapshot_hash
    # Shadow evaluation retires a failing version without touching decisions.
    shadow = registry.register("embeddings", _manifest("3"), stage="shadow")
    retired = registry.evaluate_shadow(
        "embeddings", shadow.version, improved=False,
        false_positive_increase=True, resource_increase_unacceptable=False,
    )
    assert retired.stage == "retired"


def test_access_controls_prevent_restricted_features_leaking():
    """Test 27: restricted-source records never enter the projection."""
    snapshot = USGGraphProjector().project(graph_bundle())
    labels = " ".join(node.label or "" for node in snapshot.nodes)
    assert "restricted" not in labels.lower()
    assert all(
        "source:restricted" not in (node.provenance.source_ids or ())
        for node in snapshot.nodes
    )
    quarantined = {record.record_id for record in snapshot.quarantine}
    assert "lemma:en:restricted" in quarantined, "the restricted record is reported"


def test_model_crash_does_not_block_exact_reasoning():
    """Test 28: a crashing learned scorer degrades to baselines; exact lookup works."""
    class ExplodingScorer:
        def raw_score(self, *_args):
            raise RuntimeError("model crash")

    coordinator = GraphIntelligenceCoordinator(
        embedding_scorer=ExplodingScorer(),
    )
    task = GraphIntelligenceTask(
        task_type="link_discovery", query=_river_query(), bundle=graph_bundle(),
    )
    result = coordinator.execute(task)
    assert result.errors, "the crash is recorded, not swallowed"
    assert result.candidates, "baseline fallback keeps the pipeline alive"

    exact_task = GraphIntelligenceTask(
        task_type="exact_lookup",
        query=GraphQuery(
            anchor=GraphQueryAnchor(label="Money"),
            relation_types=("INSTANCE_OF",),
        ),
        bundle=graph_bundle(),
    )
    exact_result = coordinator.execute(exact_task)
    assert exact_result.selected_path == "exact_lookup"
    assert exact_result.exact_lookup, "exact USG reasoning is unaffected by model failures"
    assert not exact_result.errors


def test_cache_hits_on_identical_cycles():
    """Step 51: identical snapshot/model keys serve cached results."""
    coordinator = GraphIntelligenceCoordinator()
    task = GraphIntelligenceTask(
        task_type="exact_lookup",
        query=GraphQuery(
            anchor=GraphQueryAnchor(label="Money"),
            relation_types=("INSTANCE_OF",),
        ),
        bundle=graph_bundle(),
    )
    first = coordinator.execute(task)
    second = coordinator.execute(task)
    assert first.exact_lookup == second.exact_lookup
    assert second.cache_hits == 1
    assert second.metrics.get("cache_reused") is True


def test_cancellation_stops_the_cycle_at_a_phase_boundary():
    """Contract requirement: cooperative cancellation keeps partial results."""
    token = CancelToken()
    coordinator = GraphIntelligenceCoordinator()
    task = GraphIntelligenceTask(
        task_type="link_discovery", query=_river_query(), bundle=graph_bundle(),
        cancel_token=token,
    )
    token.cancel()
    result = coordinator.execute(task)
    assert result.cancelled
    assert result.snapshot is not None, "partial phase-A results are preserved"
    assert not result.verification, "no further phases run after cancellation"
    kinds = coordinator.event_recorder.kinds()
    assert "error" in kinds


def test_nar_proposal_flows_through_the_coordinator():
    """Phase E inside the coordinator: advisory NAR proposal with oracle check."""
    generator = TaskGenerator()
    tasks = [
        generator.generate("graph_traversal_bfs", n_nodes=8, seed=seed, topology="chain")
        for seed in range(3)
    ]
    traces = TraceDataset.from_tasks(tasks).traces
    model = NARTrainer(latent_dim=12, rounds=2, epochs=20, seed=0).train(traces)
    service = NeuralAlgorithmicService({"graph_traversal_bfs": model})
    nar_task = generator.generate("graph_traversal_bfs", n_nodes=8, seed=9, topology="chain")
    coordinator = GraphIntelligenceCoordinator(nar_service=service)
    task = GraphIntelligenceTask(
        task_type="exact_lookup",
        query=GraphQuery(
            anchor=GraphQueryAnchor(label="Money"),
            relation_types=("INSTANCE_OF",),
        ),
        bundle=graph_bundle(), nar_task=nar_task,
    )
    result = coordinator.execute(task)
    assert result.nar_proposal is not None
    assert result.nar_proposal.advisory is True or result.nar_proposal.disposition in {
        "advisory", "fallback_to_reference", "abstained",
    }
    kinds = coordinator.event_recorder.kinds()
    assert "nar_proposed" in kinds
