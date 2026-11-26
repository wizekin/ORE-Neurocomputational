"""End-to-end integration test: the full 56-step Point 8 pipeline in one cycle.

Phases A-F run together on the USG fixture bundle with every subsystem live:
typed projection and integrity (A), trained embeddings and R-GCN (B), candidate
discovery, scoring, ranking, abstention, workspace publication (C), symbolic
verification and promotion gating (D), an advisory NAR proposal (E), and
coordination with resource admission, caching, SQLite lifecycle logging and
metrics closure (F). The USG bundle is never mutated.
"""
from __future__ import annotations

import sqlite3
from copy import deepcopy

from ore_graph_intelligence.embeddings import EmbeddingTrainer, TripleRecord
from ore_graph_intelligence.gnn import RelationGraph, RGCNLinkPredictor, build_snapshot_features
from ore_graph_intelligence.nar import (
    NARTrainer,
    NeuralAlgorithmicService,
    TaskGenerator,
    TraceDataset,
)
from ore_graph_intelligence.observability import (
    CandidateLifecycleAudit,
    GraphEventRecorder,
    SemanticSafetyMetrics,
    relation_type_subgroup_metrics,
)
from ore_graph_intelligence.orchestration import (
    GraphIntelligenceCoordinator,
    GraphIntelligenceTask,
    ModelRegistry,
    WorkspaceGraphPublisher,
)
from ore_graph_intelligence.orchestration.workspace_bridge import graph_workspace_permissions
from ore_graph_intelligence.orchestration.resource_estimator import ResourceBudget
from ore_graph_intelligence.projection import USGGraphProjector
from ore_graph_intelligence.schemas.candidate_schema import (
    CandidateDisposition,
    GraphQuery,
    GraphQueryAnchor,
)
from ore_graph_intelligence.schemas.model_manifest import ModelManifest

from .fixtures import graph_bundle


def _train_embedding(snapshot):
    records = [
        TripleRecord(
            subject=edge.source_id, relation=edge.relation_type, obj=edge.target_id,
        )
        for edge in snapshot.edges
    ]
    if len(records) < 4:
        from ore_graph_intelligence.embeddings import TripleRecord as Record

        records = [
            Record(subject=f"n{index}", relation="r", obj=f"n{(index + 1) % 6}")
            for index in range(6)
        ] + records
    return EmbeddingTrainer(
        model_type="TransE", embedding_dim=8, epochs=5, negative_samples=2, seed=0,
    ).train(records)


def _train_rgcn(snapshot):
    relations = sorted(snapshot.relation_types())
    node_ids = sorted(node.node_id for node in snapshot.nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    relation_index = {relation: index for index, relation in enumerate(relations)}
    features, _names = build_snapshot_features(snapshot)
    triples = [
        (node_index[edge.source_id], relation_index[edge.relation_type],
         node_index[edge.target_id])
        for edge in snapshot.edges
    ]
    graph = RelationGraph.from_triples(triples, relations, n_nodes=snapshot.node_count)
    predictor = RGCNLinkPredictor(
        snapshot.node_count, relations, features.shape[1],
        hidden_dim=8, embedding_dim=8, epochs=4, seed=0,
    )
    predictor.train(features, graph, triples, triples)
    return predictor


def _train_nar_service():
    generator = TaskGenerator()
    tasks = [
        generator.generate("graph_traversal_bfs", n_nodes=8, seed=seed, topology="chain")
        for seed in range(4)
    ]
    traces = TraceDataset.from_tasks(tasks).traces
    model = NARTrainer(latent_dim=12, rounds=2, epochs=25, seed=0).train(traces)
    return NeuralAlgorithmicService({"graph_traversal_bfs": model})


def test_full_56_step_cycle(tmp_path):
    """One bounded task traverses every Point 8 phase end to end."""
    bundle = graph_bundle()
    original = deepcopy(bundle)
    snapshot = USGGraphProjector().project(bundle)

    # Phase B artifacts (steps 13-24): deterministic baselines + learned models.
    embedding = _train_embedding(snapshot)
    gnn_predictor = _train_rgcn(snapshot)
    nar_service = _train_nar_service()

    from ore_logger.database import LoggerDatabase
    from ore_workspace import PermissionPolicy, WorkspaceManager

    database = LoggerDatabase(tmp_path / "e2e.sqlite")
    recorder = GraphEventRecorder(logger_database=database)
    manager = WorkspaceManager(enable_logging=False)
    workspace = manager.create_workspace(
        "graph e2e", permissions=PermissionPolicy(graph_workspace_permissions()),
    )
    publisher = WorkspaceGraphPublisher(workspace)

    registry = ModelRegistry()
    manifest = ModelManifest(
        model_id="embeddings", model_type="TransE", version="1.0.0",
        weights_hash="0" * 64, snapshot_hash=snapshot.graph_hash,
        split_digest=embedding.split.train_digest or "1" * 64,
        config={"embedding_dim": 8}, metrics={"validation_mrr": embedding.report.validation_mrr},
        trained_at_utc="2026-11-22T00:00:00+00:00",
    )
    registered = registry.register("embeddings", manifest, stage="shadow")
    registry.evaluate_shadow(
        "embeddings", registered.version, improved=True,
        false_positive_increase=False, resource_increase_unacceptable=False,
    )
    registry.promote("embeddings", registered.version, to_stage="monitored")

    coordinator = GraphIntelligenceCoordinator(
        embedding=embedding,
        gnn_predictor=gnn_predictor,
        nar_service=nar_service,
        event_recorder=recorder,
        workspace_publisher=publisher,
    )
    nar_task = TaskGenerator().generate("graph_traversal_bfs", n_nodes=8, seed=11, topology="chain")
    task = GraphIntelligenceTask(
        task_type="link_discovery",
        query=GraphQuery(
            anchor=GraphQueryAnchor(label="The land alongside a river."),
            relation_types=("DENOTES",), budget_candidates=32,
        ),
        bundle=bundle,
        budget=ResourceBudget(deadline_ms=500.0),
        nar_task=nar_task,
    )

    result = coordinator.execute(task)

    # Phase A: projection, integrity, provenance (steps 3-12).
    assert result.snapshot is not None and result.snapshot.node_count > 0
    assert result.integrity is not None and result.integrity.passed
    assert result.snapshot.quarantine, "quarantined records are reported"

    # Phase F: admission and path selection (steps 49-50).
    assert result.admission is not None and result.admission.decision == "admit"
    assert result.selected_path == "learned_graph_intelligence"

    # Phase B/C: candidates scored with complementary components (steps 25-31).
    assert result.candidates or (result.abstention and result.abstention.abstained)
    if result.candidates:
        for candidate in result.candidates:
            scores = candidate.component_scores
            assert scores.baseline_score is not None, "deterministic baseline always scores"
            assert candidate.utility is not None or candidate.status == "abstained"
        for candidate in result.candidates:
            assert candidate.explanation_paths is not None
            break

    # Step 32: hypotheses published into the dedicated namespace.
    hypothesis_publications = [
        item for item in result.workspace_publications
        if item["finding_type"] == "graph_hypothesis"
    ]
    verification_publications = [
        item for item in result.workspace_publications
        if item["finding_type"] == "graph_verification"
    ]
    assert hypothesis_publications or (result.abstention and result.abstention.abstained)
    finding_types = {
        finding["finding_type"] for finding in publisher.workspace_findings()
    }
    assert "graph_hypothesis" in finding_types or (
        result.abstention and result.abstention.abstained
    )

    # Phase D: verification and promotion gate (steps 33-39).
    if result.verification:
        dispositions = {result_item.disposition for result_item in result.verification}
        assert dispositions <= {
            CandidateDisposition.SUPPORTED, CandidateDisposition.REJECTED,
            CandidateDisposition.AMBIGUOUS, CandidateDisposition.UNVERIFIABLE,
        }
        safety = SemanticSafetyMetrics.from_results(result.verification)
        assert safety.n_candidates == len(result.verification)
        subgroups = relation_type_subgroup_metrics(result.verification)
        assert subgroups
        assert verification_publications
        for proposal in result.update_proposals:
            assert proposal.requires_authorization == "authorized_usg_update_workflow"
    else:
        assert result.abstention and result.abstention.abstained

    # Phase E: NAR advisory proposal with deterministic check (steps 41-48).
    assert result.nar_proposal is not None
    assert result.nar_proposal.disposition in {"advisory", "fallback_to_reference"}
    assert result.nar_proposal.exact_agreement >= 0.0

    # Phase F: lifecycle events, SQLite persistence, audit closure (steps 40, 56).
    kinds = recorder.kinds()
    assert "task_received" in kinds and "cycle_closed" in kinds
    assert "snapshot_projected" in kinds and "verification_completed" in kinds or (
        "abstention_evaluated" in kinds
    )
    with sqlite3.connect(str(tmp_path / "e2e.sqlite")) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute(
            "SELECT sequence_no, timestamp_utc, event_kind, metadata_json FROM events "
            "WHERE run_id=? ORDER BY sequence_no", (recorder.run_id,),
        )]
    assert rows, "lifecycle events must be persisted to ore_logger SQLite"
    trail = CandidateLifecycleAudit().audit_sqlite_rows(rows)
    assert trail.event_count == len(recorder.events)
    if trail.candidate_audits:
        assert all("proposed" in item.stages for item in trail.candidate_audits)

    # Core invariant: the authoritative USG bundle is untouched.
    assert bundle == original, "no graph intelligence path may mutate USG records"

    # Step 51: a repeated cycle serves from the cache; the registry tracks versions.
    second = coordinator.execute(GraphIntelligenceTask(
        task_type="link_discovery", query=task.query, bundle=bundle,
    ))
    assert second.cache_hits == 1
    assert registry.current("embeddings").version == registered.version
    rolled = registry.rollback("embeddings")
    assert rolled is None or rolled.version == registered.version
    assert bundle == original
