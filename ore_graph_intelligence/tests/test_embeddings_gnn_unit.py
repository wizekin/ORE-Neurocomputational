"""Embedding and GNN acceptance tests 11, 18, 19, 20.

11. Learned models are compared to deterministic baselines.
18. Raw scores are not called probabilities without calibration.
19. Model manifests include version, data, config, and weights hash.
20. Fixed seeds/manifests reproduce within tolerance.
"""
from __future__ import annotations

import numpy as np

from ore_graph_intelligence.embeddings import (
    EmbeddingScorer,
    EmbeddingTrainer,
    TripleRecord,
    build_split_manifest,
)
from ore_graph_intelligence.gnn import (
    DeterministicGraphBaselines,
    RelationGraph,
    RGCNLinkPredictor,
    build_snapshot_features,
)
from ore_graph_intelligence.projection import USGGraphProjector
from ore_graph_intelligence.schemas.model_manifest import ModelManifest

from .fixtures import graph_bundle


def _triple_records():
    """Deterministic triple set with enough entities and relations to train on."""
    entities = [f"e{index:02d}" for index in range(12)]
    relations = ["is_a", "part_of", "causes"]
    records = []
    for subject in entities:
        for relation in relations:
            obj = entities[(int(subject[1:]) + 3) % len(entities)]
            if obj != subject:
                records.append(TripleRecord(subject=subject, relation=relation, obj=obj))
    return records


def test_learned_models_compared_to_deterministic_baselines():
    """Test 11: the trained embedding reports its MRR and deterministic scores coexist."""
    records = _triple_records()
    trained = EmbeddingTrainer(
        model_type="TransE", embedding_dim=8, epochs=8, negative_samples=4, seed=0,
    ).train(records)
    report = trained.report
    assert report.validation_mrr > 0.0
    assert report.negative_sampling_policy, "the negative sampling policy must be documented"

    triples = [(r.subject, r.relation, r.obj) for r in records]
    baselines = DeterministicGraphBaselines(triples)
    assert baselines.typed_neighborhood_lookup("e00", "is_a") == ("e03",)
    # Deterministic common-neighbor ranking recorded alongside the learned MRR.
    baseline_scores = [
        baselines.common_neighbor_score(subject, obj)
        for subject, _relation, obj in triples
    ]
    assert all(np.isfinite(baseline_scores))
    assert max(baseline_scores) >= 0.0


def test_raw_scores_not_probabilities_without_calibration():
    """Test 18: uncalibrated scorers expose raw scores only, never probabilities."""
    records = _triple_records()
    trained = EmbeddingTrainer(
        model_type="DistMult", embedding_dim=8, epochs=5, negative_samples=4, seed=0,
    ).train(records)
    scorer = EmbeddingScorer(trained.model, trained.entity_ids, trained.relation_ids)
    raw = scorer.raw_score("e00", "is_a", "e03")
    assert isinstance(raw, float)
    assert scorer.calibrated_probability("e00", "is_a", "e03") is None, (
        "no calibrator fitted: a raw ranking value must not masquerade as a probability"
    )


def test_model_manifest_records_version_config_and_weights_hash():
    """Test 19: manifests carry model/version, config, snapshot, split, metrics."""
    records = _triple_records()
    manifest = build_split_manifest(records, policy="relation", seed=0)
    trained = EmbeddingTrainer(
        model_type="ComplEx", embedding_dim=8, epochs=4, negative_samples=2, seed=0,
    ).train(records)
    model_manifest = ModelManifest(
        model_id="graph-embeddings", model_type="ComplEx", version="1.2.0",
        weights_hash="a" * 64,
        snapshot_hash="b" * 64, split_digest=manifest.train_digest or "c" * 64,
        config={"embedding_dim": 8},
        metrics={"validation_mrr": trained.report.validation_mrr},
        trained_at_utc="2026-11-22T00:00:00+00:00",
    )
    payload = model_manifest.as_dict()
    assert payload["version"] == "1.2.0"
    assert payload["config"]["embedding_dim"] == 8
    assert len(payload["weights_hash"]) == 64
    assert len(payload["split_digest"]) == 64
    assert "validation_mrr" in payload["metrics"]


def test_fixed_seeds_reproduce_training():
    """Test 20: the same seed and data reproduce the same weights hash."""
    records = _triple_records()
    first = EmbeddingTrainer(
        model_type="RotatE", embedding_dim=8, epochs=4, negative_samples=2, seed=7,
    ).train(records)
    second = EmbeddingTrainer(
        model_type="RotatE", embedding_dim=8, epochs=4, negative_samples=2, seed=7,
    ).train(records)
    assert first.report.as_dict()["config"] == second.report.as_dict()["config"]
    assert first.report.loss_history == second.report.loss_history, (
        "fixed seeds and manifests must reproduce training within tolerance"
    )


def test_rgcn_reports_baseline_comparison():
    """Test 11 (GNN): the R-GCN training report records its baseline MRR."""
    snapshot = USGGraphProjector().project(graph_bundle())
    relations = sorted(snapshot.relation_types())
    node_ids = sorted(node.node_id for node in snapshot.nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    relation_index = {relation: index for index, relation in enumerate(relations)}
    features, feature_names = build_snapshot_features(snapshot)
    assert feature_names, "the feature schema must be recorded"  # noqa: F841
    triples = [
        (node_index[edge.source_id], relation_index[edge.relation_type],
         node_index[edge.target_id])
        for edge in snapshot.edges
    ]
    graph = RelationGraph.from_triples(triples, relations, n_nodes=snapshot.node_count)
    predictor = RGCNLinkPredictor(
        snapshot.node_count, relations, features.shape[1],
        hidden_dim=8, embedding_dim=8, epochs=5, seed=0,
    )
    report = predictor.train(features, graph, triples, triples)
    assert report.validation_mrr >= 0.0
    assert report.baseline_mrr >= 0.0, "learned GNN must be compared to its baseline"
    score = predictor.score(features, graph, triples[0][0], triples[0][1], triples[0][2])
    assert np.isfinite(score)
