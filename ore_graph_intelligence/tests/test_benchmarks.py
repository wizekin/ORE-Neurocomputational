"""Operational benchmark smoke tests (spec section 8, operations metrics).

Projection time, graph size, inference latency, candidate counts and cache behavior
are measured against generous bounds so regressions surface in CI.
"""
from __future__ import annotations

from time import perf_counter

from ore_graph_intelligence.observability import (
    graph_prediction_metrics,
    hits_at_k,
    mean_reciprocal_rank,
)
from ore_graph_intelligence.orchestration import (
    GraphIntelligenceCoordinator,
    GraphIntelligenceTask,
    GraphResultCache,
    CacheKey,
)
from ore_graph_intelligence.orchestration.resource_estimator import (
    AdaptiveResourceCoordinator,
    GraphResourceEstimator,
)
from ore_graph_intelligence.projection import USGGraphProjector
from ore_graph_intelligence.schemas.candidate_schema import (
    GraphQuery,
    GraphQueryAnchor,
)

from .fixtures import graph_bundle


def test_projection_benchmark():
    """Projection of the fixture bundle stays well under a second."""
    started = perf_counter()
    snapshot = USGGraphProjector().project(graph_bundle())
    elapsed = perf_counter() - started
    assert snapshot.node_count >= 5
    assert snapshot.edge_count >= 3
    assert elapsed < 1.0, f"projection too slow: {elapsed:.3f}s"


def test_candidate_pipeline_benchmark():
    """The full learned cycle completes quickly and reports candidate counts."""
    coordinator = GraphIntelligenceCoordinator()
    task = GraphIntelligenceTask(
        task_type="link_discovery",
        query=GraphQuery(
            anchor=GraphQueryAnchor(label="The land alongside a river."),
            relation_types=("DENOTES",), budget_candidates=32,
        ),
        bundle=graph_bundle(),
    )
    started = perf_counter()
    result = coordinator.execute(task)
    elapsed = perf_counter() - started
    assert result.candidates or result.abstention
    assert elapsed < 5.0, f"coordinated cycle too slow: {elapsed:.3f}s"


def test_resource_estimator_reports_footprint():
    """Estimates cover nodes, edges, RAM, model load and latency."""
    snapshot = USGGraphProjector().project(graph_bundle())
    estimate = GraphResourceEstimator().estimate(snapshot)
    assert estimate.n_nodes == snapshot.node_count
    assert estimate.n_edges == snapshot.edge_count
    assert estimate.estimated_ram_mb > 0
    assert estimate.model_load_mb >= 0
    assert estimate.estimated_latency_ms > 0
    decision = AdaptiveResourceCoordinator().admit(estimate)
    assert decision.decision == "admit"


def test_cache_hit_benchmark():
    """Repeated identical cycles serve from the bounded cache."""
    cache = GraphResultCache(capacity=16)
    key = CacheKey("snapshot", "model", "candidates")
    for round_index in range(32):
        if cache.get(key) is None:
            cache.put(key, {"round": round_index})
    assert cache.stats()["hits"] == 31
    assert cache.stats()["misses"] == 1


def test_ranking_metrics_benchmark():
    """MRR/Hits@K are computed quickly on moderate ranking lists."""
    ranks = list(range(1, 21))
    assert mean_reciprocal_rank(ranks) > 0.0
    assert hits_at_k(ranks, 10) == 0.5
    metrics = graph_prediction_metrics([], frozenset(), ranks=[1, 2, 3])
    assert metrics["mrr"] > 0
    assert metrics["hits_at_1"] == 1 / 3
