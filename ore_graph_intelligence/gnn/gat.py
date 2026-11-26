"""Graph attention network layer (ORE Point 8 step 19).

GAT learns attention e_ij = a(W h_i, W h_j) with alpha_ij a softmax over neighbors.
Attention measures model influence over message weights, not truth or causality; relation
direction and roles are encoded by aggregating only over incoming edges. Added only when a
measured task justifies attention over simpler aggregation.
"""
from __future__ import annotations

import numpy as np

from .rgcn import RelationGraph, _xavier


class GATLayer:
    """One multi-head GAT layer over the union of typed incoming edges."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int = 2,
        seed: int = 0,
        activation: str = "elu",
        negative_slope: float = 0.2,
    ) -> None:
        if in_dim < 1 or out_dim < 1 or heads < 1:
            raise ValueError("dimensions and head count must be positive")
        if out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads")
        if activation not in {"elu", "relu", "identity"}:
            raise ValueError("activation must be elu, relu, or identity")
        rng = np.random.default_rng(seed)
        self.heads = int(heads)
        self.head_dim = out_dim // heads
        self.out_dim = out_dim
        self.weights = _xavier(rng, in_dim, out_dim)  # shared W across heads
        self.attention_self = rng.normal(0.0, 0.1, (self.head_dim,))
        self.attention_neighbor = rng.normal(0.0, 0.1, (self.head_dim,))
        self.negative_slope = float(negative_slope)
        self.activation = activation

    def _activate(self, values: np.ndarray) -> np.ndarray:
        if self.activation == "elu":
            return np.where(
                values > 0, values, self.negative_slope * (np.exp(values) - 1.0),
            )
        if self.activation == "relu":
            return np.maximum(values, 0.0)
        return values

    def forward(self, features: np.ndarray, graph: RelationGraph) -> np.ndarray:
        """Compute multi-head attention-weighted aggregation over incoming edges."""
        features = np.asarray(features, dtype=np.float64)
        projected = features @ self.weights  # (n, out_dim)
        n = graph.n_nodes
        output = np.zeros((n, self.out_dim), dtype=np.float64)
        incoming: dict[int, list[int]] = {}
        for sources, targets in graph.edges_by_relation.values():
            for source, target in zip(sources.tolist(), targets.tolist()):
                incoming.setdefault(target, []).append(source)
        for target in range(n):
            neighbors = incoming.get(target, [])
            for head in range(self.heads):
                start = head * self.head_dim
                end = start + self.head_dim
                target_slice = projected[target, start:end]
                if not neighbors:
                    output[target, start:end] = self._activate(target_slice)
                    continue
                keys = np.stack([projected[neighbor, start:end] for neighbor in neighbors])
                scores = (
                    keys @ self.attention_self + target_slice @ self.attention_neighbor
                )
                scores = np.maximum(scores, 0.0) + self.negative_slope * np.minimum(scores, 0.0)
                scores = scores - scores.max()
                weights = np.exp(scores)
                weights = weights / weights.sum()
                output[target, start:end] = self._activate(
                    target_slice + weights @ keys
                )
        return output


class GATEncoder:
    """Two-layer GAT encoder for measured comparison against the R-GCN track."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, heads: int = 2, seed: int = 0) -> None:
        self.layer_one = GATLayer(in_dim, hidden_dim, heads=heads, seed=seed, activation="elu")
        self.layer_two = GATLayer(hidden_dim, out_dim, heads=1, seed=seed + 1, activation="identity")

    def forward(self, features: np.ndarray, graph: RelationGraph) -> np.ndarray:
        hidden = self.layer_one.forward(features, graph)
        return self.layer_two.forward(hidden, graph)


__all__ = ["GATEncoder", "GATLayer"]
