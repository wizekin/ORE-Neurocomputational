"""GraphSAGE-style inductive aggregation over sampled neighborhoods (ORE Point 8 step 18).

GraphSAGE supports sampled/inductive aggregation: h_v' = phi([W_self h_v || AGG(N(v))]) with
a mean aggregator over a bounded neighbor sample. Sampling keeps the aggregation bounded on
large projections and the self-transform keeps isolated nodes representable.
"""
from __future__ import annotations

import numpy as np

from .rgcn import RelationGraph, _xavier


class GraphSAGELayer:
    """One GraphSAGE layer with mean neighbor aggregation over a bounded sample."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        seed: int = 0,
        sample_size: int = 8,
        activation: str = "relu",
        combine: str = "concat",
    ) -> None:
        if in_dim < 1 or out_dim < 1 or sample_size < 1:
            raise ValueError("dimensions and sample size must be positive")
        if activation not in {"relu", "tanh", "identity"}:
            raise ValueError("activation must be relu, tanh, or identity")
        if combine not in {"concat", "sum"}:
            raise ValueError("combine must be concat or sum")
        rng = np.random.default_rng(seed)
        self.self_weights = _xavier(rng, in_dim, out_dim)
        self.neighbor_weights = _xavier(rng, in_dim, out_dim)
        self.sample_size = int(sample_size)
        self.activation = activation
        self.combine = combine

    def _activate(self, values: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(values, 0.0)
        if self.activation == "tanh":
            return np.tanh(values)
        return values

    def output_width(self) -> int:
        """Output dimension: concatenation doubles the declared width."""
        return 2 * self.self_weights.shape[0] if self.combine == "concat" else self.self_weights.shape[0]

    def forward(self, features: np.ndarray, graph: RelationGraph, *, sampled: bool = False) -> np.ndarray:
        """Aggregate mean neighbor features (sampled or full) and combine with self."""
        features = np.asarray(features, dtype=np.float64)
        neighbor_sums = np.zeros_like(features)
        neighbor_counts = np.zeros(graph.n_nodes, dtype=np.float64)
        for sources, targets in graph.edges_by_relation.values():
            if sources.size == 0:
                continue
            if sampled and self.sample_size < len(sources):
                # Deterministic bounded sample: first N edges in stable order keep runs reproducible.
                order = np.argsort(sources, kind="stable")[: self.sample_size]
                sources, targets = sources[order], targets[order]
            np.add.at(neighbor_sums, targets, features[sources])
            np.add.at(neighbor_counts, targets, 1.0)
        means = neighbor_sums / np.maximum(neighbor_counts, 1.0)[:, None]
        self_part = features @ self.self_weights
        neighbor_part = means @ self.neighbor_weights
        if self.combine == "concat":
            combined = np.concatenate([self_part, neighbor_part], axis=1)
        else:
            combined = self_part + neighbor_part
        return self._activate(combined)


class GraphSAGEEncoder:
    """Two-layer GraphSAGE encoder for measured comparison against the R-GCN track."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        seed: int = 0,
        sample_size: int = 8,
    ) -> None:
        self.layer_one = GraphSAGELayer(
            in_dim, hidden_dim, seed=seed, sample_size=sample_size, combine="concat",
        )
        self.layer_two = GraphSAGELayer(
            2 * hidden_dim, out_dim, seed=seed + 1, sample_size=sample_size,
            combine="sum", activation="identity",
        )

    def forward(self, features: np.ndarray, graph: RelationGraph, *, sampled: bool = False) -> np.ndarray:
        hidden = self.layer_one.forward(features, graph, sampled=sampled)
        return self.layer_two.forward(hidden, graph, sampled=sampled)


__all__ = ["GraphSAGEEncoder", "GraphSAGELayer"]
