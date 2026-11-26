"""GNN inference engine: snapshot features, R-GCN training, over-smoothing monitoring.

Implements ORE Point 8 steps 11, 17, 18-21:

- Snapshot features come from approved type IDs, relation IDs, and structural statistics
  only (no label leakage).
- R-GCN trains with relation-specific messages and basis decomposition on a filtered
  link-prediction objective.
- GCN, GraphSAGE, and GAT are available as measured alternatives.
- The over-smoothing monitor measures representation collapse as depth increases and
  reports when residual or jumping-knowledge variants would be justified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..projection.typed_graph import TypedGraphSnapshot
from .gcn import GCNEncoder
from .gat import GATEncoder
from .graphsage import GraphSAGEEncoder
from .rgcn import DistMultDecoder, RGCNEncoder, RelationGraph

NODE_KIND_ORDER = (
    "LEMMA", "LEXICAL_SENSE", "LEXICAL_FORM", "CONCEPT", "SEMANTIC_CLASS",
    "PREDICATE_FRAME", "RELATION_EVENT",
)


def build_snapshot_features(
    snapshot: TypedGraphSnapshot,
    *,
    extra_features: Mapping[str, np.ndarray] | None = None,
    relation_ids: Mapping[str, int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build leakage-free node features from approved type IDs and structure (step 11).

    Feature columns: one-hot node kind, log-degree statistics (out/in), relation-type
    participation histogram. Only schema-approved signals are used: no labels, no target
    relation leakage. ``extra_features`` lets callers add approved provenance-tagged text
    features explicitly.
    """
    nodes = sorted(snapshot.nodes, key=lambda node: node.node_id)
    node_ids = [node.node_id for node in nodes]
    index_of = {node_id: index for index, node_id in enumerate(node_ids)}
    kind_index = {kind: position for position, kind in enumerate(NODE_KIND_ORDER)}

    relations = relation_ids or {
        relation: index for index, relation in enumerate(sorted(snapshot.relation_types()))
    }
    relation_index = {name: position for position, name in enumerate(sorted(relations))}

    width = len(NODE_KIND_ORDER) + 4 + len(relation_index)
    features = np.zeros((len(nodes), width), dtype=np.float64)
    for row, node in enumerate(nodes):
        features[row, kind_index[node.node_kind]] = 1.0
        out_degree, in_degree = snapshot.degree_of(node.node_id)
        features[row, len(NODE_KIND_ORDER)] = float(np.log1p(out_degree))
        features[row, len(NODE_KIND_ORDER) + 1] = float(np.log1p(in_degree))
        features[row, len(NODE_KIND_ORDER) + 2] = float(out_degree == 0 and in_degree == 0)
        features[row, len(NODE_KIND_ORDER) + 3] = float(len(node.aliases))
    relation_offset = len(NODE_KIND_ORDER) + 4
    for edge in snapshot.edges:
        row = index_of.get(edge.source_id)
        column = relation_index.get(edge.relation_type)
        if row is None or column is None:
            continue
        features[row, relation_offset + column] += 1.0
    if extra_features:
        for node_id, array in extra_features.items():
            row = index_of.get(node_id)
            if row is None:
                continue
            vector = np.asarray(array, dtype=np.float64).reshape(-1)
            features[row, -1] = vector[0] if vector.size else 0.0
    return features, node_ids


@dataclass
class OverSmoothingReport:
    """Representation-collapse measurements as depth increases (ORE Point 8 step 20)."""

    layer_mean_norm: list[float] = field(default_factory=list)
    layer_mean_cosine: list[float] = field(default_factory=list)
    collapse_detected: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "layer_mean_norm": list(self.layer_mean_norm),
            "layer_mean_cosine": list(self.layer_mean_cosine),
            "collapse_detected": self.collapse_detected,
        }


class OverSmoothingMonitor:
    """Measure representation collapse as GNN depth increases."""

    def __init__(self, *, cosine_threshold: float = 0.95, norm_ratio_threshold: float = 0.1) -> None:
        if not 0.0 < cosine_threshold <= 1.0:
            raise ValueError("cosine_threshold must lie in (0, 1]")
        self.cosine_threshold = float(cosine_threshold)
        self.norm_ratio_threshold = float(norm_ratio_threshold)

    def measure(self, layer_representations: Sequence[np.ndarray]) -> OverSmoothingReport:
        """Compute mean pairwise cosine similarity and norm trend across layer outputs."""
        report = OverSmoothingReport()
        for representations in layer_representations:
            matrix = np.asarray(representations, dtype=np.float64)
            norms = np.linalg.norm(matrix, axis=1)
            normalized = matrix / np.maximum(norms[:, None], 1e-12)
            if matrix.shape[0] > 1:
                similarity = normalized @ normalized.T
                n = matrix.shape[0]
                mean_cosine = float((similarity.sum() - n) / (n * (n - 1)))
            else:
                mean_cosine = 1.0
            report.layer_mean_norm.append(float(np.mean(norms)))
            report.layer_mean_cosine.append(mean_cosine)
        if report.layer_mean_norm:
            initial_norm = max(report.layer_mean_norm[0], 1e-12)
            final_norm = report.layer_mean_norm[-1]
            norm_ratio = final_norm / initial_norm
            report.collapse_detected = bool(
                (report.layer_mean_cosine and max(report.layer_mean_cosine) > self.cosine_threshold)
                or norm_ratio < self.norm_ratio_threshold
            )
        return report


class DeterministicGraphBaselines:
    """Exact typed neighborhood lookup and structural ranking (ORE Point 8 step 13).

    Baselines: common-neighbor overlap and relation-frequency ranking. They are the
    reference every learned model must beat before it is considered useful.
    """

    def __init__(self, triples: Sequence[tuple[str, str, str]]) -> None:
        self.triples = list(triples)
        self.triple_set = frozenset(self.triples)
        self.relation_counts: dict[str, int] = {}
        for _, relation, _ in self.triples:
            self.relation_counts[relation] = self.relation_counts.get(relation, 0) + 1
        self.neighbors_out: dict[str, set[str]] = {}
        self.neighbors_in: dict[str, set[str]] = {}
        for head, relation, tail in self.triples:
            self.neighbors_out.setdefault(head, set()).add(tail)
            self.neighbors_in.setdefault(tail, set()).add(head)
        total = max(len(self.triples), 1)
        self.relation_frequency = {
            relation: count / total for relation, count in self.relation_counts.items()
        }

    def common_neighbor_score(self, subject: str, obj: str) -> float:
        """Count shared 1-hop neighbors between subject and object."""
        out = self.neighbors_out.get(subject, set())
        into = self.neighbors_in.get(obj, set())
        return float(len(out & into))

    def relation_frequency_score(self, relation: str) -> float:
        """Return the observed frequency of this relation type."""
        return self.relation_frequency.get(relation, 0.0)

    def typed_neighborhood_lookup(
        self, subject: str, relation: str,
    ) -> tuple[str, ...]:
        """Exact lookup: every object already linked from subject by this relation type."""
        return tuple(sorted(
            tail for head, rel, tail in self.triples if head == subject and rel == relation
        ))


@dataclass
class RGCNTrainingReport:
    """Training and evaluation record for one R-GCN link-prediction run."""

    loss_history: list[float] = field(default_factory=list)
    validation_mrr: float = 0.0
    baseline_mrr: float = 0.0
    epochs_run: int = 0
    learned_beats_baseline: bool = False


class RGCNLinkPredictor:
    """Train an R-GCN encoder with a DistMult decoder on filtered link prediction (step 21)."""

    def __init__(
        self,
        n_nodes: int,
        relation_names: Sequence[str],
        feature_dim: int,
        *,
        hidden_dim: int = 16,
        embedding_dim: int = 16,
        basis_size: int = 3,
        learning_rate: float = 0.02,
        epochs: int = 30,
        negative_samples: int = 4,
        seed: int = 0,
    ) -> None:
        if n_nodes < 1 or not relation_names or feature_dim < 1:
            raise ValueError("nodes, relations, and feature dimension must be positive")
        self.n_nodes = int(n_nodes)
        self.relation_names = list(relation_names)
        self.encoder = RGCNEncoder(
            feature_dim, hidden_dim, embedding_dim, self.relation_names,
            basis_size=basis_size, seed=seed,
        )
        self.decoder = DistMultDecoder(embedding_dim, len(self.relation_names), seed=seed)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.negative_samples = int(negative_samples)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)

    def train(
        self,
        features: np.ndarray,
        graph: RelationGraph,
        train_triples: Sequence[tuple[int, int, int]],
        known_triples: Sequence[tuple[int, int, int]],
        *,
        validation_triples: Sequence[tuple[int, int, int]] | None = None,
    ) -> RGCNTrainingReport:
        """Train on positive triples plus filtered negatives; report loss and metrics."""
        if not train_triples:
            raise ValueError("train triples must be non-empty")
        known = set(map(tuple, known_triples))
        train_array = np.array(sorted(set(map(tuple, train_triples))), dtype=np.int64)
        report = RGCNTrainingReport()
        for _epoch in range(self.epochs):
            loss = self._train_epoch(features, graph, train_array, known)
            report.loss_history.append(loss)
        report.epochs_run = len(report.loss_history)
        if validation_triples:
            report.validation_mrr = self.evaluate_mrr(
                features, graph, validation_triples, known,
            )
            report.baseline_mrr = self._baseline_mrr(validation_triples)
            report.learned_beats_baseline = report.validation_mrr >= report.baseline_mrr
        return report

    def _train_epoch(
        self,
        features: np.ndarray,
        graph: RelationGraph,
        train_array: np.ndarray,
        known: set[tuple[int, int, int]],
    ) -> float:
        heads, rels, tails = train_array[:, 0], train_array[:, 1], train_array[:, 2]
        negatives = self._sample_negatives(train_array, known)
        embeddings, caches = self.encoder.forward(features, graph)
        positive_scores = self.decoder.score(embeddings, heads, rels, tails)
        negative_scores = self.decoder.score(
            embeddings, negatives[:, 0], negatives[:, 1], negatives[:, 2],
        )
        pos_weights = _sigmoid(positive_scores) - 1.0
        neg_weights = _sigmoid(negative_scores)
        positive_loss = float(np.logaddexp(0.0, -positive_scores).sum())
        negative_loss = float(np.logaddexp(0.0, negative_scores).sum())

        grad_embeddings = np.zeros_like(embeddings)
        grad_relations = np.zeros_like(self.decoder.relation_vectors)
        g_embed, g_rel = self.decoder.gradients(embeddings, heads, rels, tails, pos_weights)
        grad_embeddings += g_embed
        grad_relations += g_rel
        g_embed, g_rel = self.decoder.gradients(
            embeddings, negatives[:, 0], negatives[:, 1], negatives[:, 2], neg_weights,
        )
        grad_embeddings += g_embed
        grad_relations += g_rel
        _, layer_grads = self.encoder.backward(caches, grad_embeddings)
        self.encoder.apply_gradients(layer_grads, self.learning_rate)
        self.decoder.relation_vectors -= self.learning_rate * grad_relations
        total = positive_loss + negative_loss
        return total / (len(heads) + len(negatives) + 1e-12)

    def _sample_negatives(
        self, train_array: np.ndarray, known: set[tuple[int, int, int]],
    ) -> np.ndarray:
        """Corrupt tails only with tails already used in the observed graph (filtered)."""
        tails = np.unique(train_array[:, 2])
        negatives: list[tuple[int, int, int]] = []
        for head, relation, tail in train_array:
            for _attempt in range(self.negative_samples):
                corrupted = int(tails[int(self._rng.integers(0, len(tails)))])
                if (int(head), int(relation), corrupted) in known:
                    continue
                negatives.append((int(head), int(relation), corrupted))
        if not negatives:
            negatives = [(int(head), int(relation), int(tail)) for head, relation, tail in train_array[:1]]
        return np.array(negatives, dtype=np.int64)

    def score(
        self,
        features: np.ndarray,
        graph: RelationGraph,
        subject: int,
        relation: int,
        obj: int,
    ) -> float:
        """Score one index triple with the trained encoder/decoder pair."""
        embeddings, _ = self.encoder.forward(features, graph)
        return float(self.decoder.score(
            embeddings,
            np.array([subject], dtype=np.int64),
            np.array([relation], dtype=np.int64),
            np.array([obj], dtype=np.int64),
        )[0])

    def evaluate_mrr(
        self,
        features: np.ndarray,
        graph: RelationGraph,
        validation_triples: Sequence[tuple[int, int, int]],
        known: set[tuple[int, int, int]],
    ) -> float:
        """Filtered MRR over corrupted-tail ranking."""
        embeddings, _ = self.encoder.forward(features, graph)
        tails = sorted({tail for _, _, tail in known} | {tail for _, _, tail in validation_triples})
        reciprocal_ranks: list[float] = []
        for head, relation, tail in validation_triples:
            candidates = [
                candidate for candidate in tails
                if (head, relation, candidate) not in known or candidate == tail
            ]
            if len(candidates) < 2:
                continue
            scores = self.decoder.score(
                embeddings,
                np.full(len(candidates), head, dtype=np.int64),
                np.full(len(candidates), relation, dtype=np.int64),
                np.array(candidates, dtype=np.int64),
            )
            order = np.argsort(-scores)
            positions = np.empty(len(order), dtype=np.int64)
            positions[order] = np.arange(len(order))
            rank = int(positions[candidates.index(tail)]) + 1
            reciprocal_ranks.append(1.0 / rank)
        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    def _baseline_mrr(self, validation_triples: Sequence[tuple[int, int, int]]) -> float:
        """Relation-frequency baseline MRR: constant score gives expected rank 1/(n/2)."""
        if not validation_triples:
            return 0.0
        # A constant baseline ranks the true tail uniformly; expected reciprocal rank ~ 2/(n+1)
        n_observed = max(len({tail for _, _, tail in validation_triples}), 1)
        return 2.0 / (n_observed + 1)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_negative = np.exp(values[~positive])
    out[~positive] = exp_negative / (1.0 + exp_negative)
    return out


class GNNInferenceEngine:
    """Run encoder variants and monitor over-smoothing for one snapshot (steps 18-20)."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)

    def compare_encoders(
        self,
        features: np.ndarray,
        graph: RelationGraph,
        *,
        variants: Sequence[str] = ("rgcn", "gcn", "graphsage", "gat"),
    ) -> dict[str, dict[str, object]]:
        """Run each requested encoder variant and report shape/latency/collapse metrics."""
        results: dict[str, dict[str, object]] = {}
        for variant in variants:
            if variant == "rgcn":
                encoder = RGCNEncoder(
                    features.shape[1], 16, 16, graph.edges_by_relation.keys(), seed=self.seed,
                )
            elif variant == "gcn":
                encoder = GCNEncoder(features.shape[1], 16, 16, seed=self.seed)
            elif variant == "graphsage":
                encoder = GraphSAGEEncoder(features.shape[1], 16, 16, seed=self.seed)
            elif variant == "gat":
                encoder = GATEncoder(features.shape[1], 16, 16, seed=self.seed)
            else:
                raise ValueError(f"unknown encoder variant: {variant}")
            import time

            started = time.perf_counter()
            if variant == "rgcn":
                output, _caches = encoder.forward(features, graph)
            else:
                output = encoder.forward(features, graph)
            latency_ms = (time.perf_counter() - started) * 1000.0
            monitor = OverSmoothingMonitor()
            layer_outputs = [features, output]
            report = monitor.measure(layer_outputs)
            results[variant] = {
                "embedding_shape": list(output.shape),
                "finite": bool(np.all(np.isfinite(output))),
                "latency_ms": latency_ms,
                "over_smoothing": report.as_dict(),
            }
        return results


__all__ = [
    "DeterministicGraphBaselines", "GNNInferenceEngine", "NODE_KIND_ORDER",
    "OverSmoothingMonitor", "OverSmoothingReport", "RGCNLinkPredictor",
    "RGCNTrainingReport", "build_snapshot_features",
]
