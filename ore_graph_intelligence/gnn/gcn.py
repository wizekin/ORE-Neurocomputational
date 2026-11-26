"""Graph convolutional network baseline (ORE Point 8 step 18).

GCN is the normalized-neighborhood baseline: h' = phi(D^-1/2 (A+I) D^-1/2 h W). It treats
the projection as one homogeneous relation graph and is kept deliberately simple so
R-GCN, GraphSAGE, and GAT can be measured against it.
"""
from __future__ import annotations

import numpy as np

from .rgcn import RelationGraph, _xavier


def normalized_adjacency(graph: RelationGraph) -> np.ndarray:
    """Symmetric normalized adjacency D^-1/2 (A + I) D^-1/2 over the union of relations."""
    adjacency = np.eye(graph.n_nodes, dtype=np.float64)
    for sources, targets in graph.edges_by_relation.values():
        adjacency[targets, sources] += 1.0
        adjacency[sources, targets] += 1.0
    degrees = adjacency.sum(axis=1)
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1.0))
    return adjacency * inverse_sqrt[:, None] * inverse_sqrt[None, :]


class GCNLayer:
    """One GCN layer over the normalized neighborhood baseline."""

    def __init__(self, in_dim: int, out_dim: int, *, seed: int = 0, activation: str = "relu") -> None:
        if in_dim < 1 or out_dim < 1:
            raise ValueError("dimensions must be positive")
        if activation not in {"relu", "tanh", "identity"}:
            raise ValueError("activation must be relu, tanh, or identity")
        self.weights = _xavier(np.random.default_rng(seed), in_dim, out_dim)
        self.activation = activation

    def forward(self, features: np.ndarray, normalized: np.ndarray) -> np.ndarray:
        """Compute phi(normalized @ h @ W)."""
        features = np.asarray(features, dtype=np.float64)
        pre = normalized @ (features @ self.weights)
        if self.activation == "relu":
            return np.maximum(pre, 0.0)
        if self.activation == "tanh":
            return np.tanh(pre)
        return pre


class GCNEncoder:
    """Two-layer GCN encoder for measured comparison against the R-GCN track."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, seed: int = 0) -> None:
        self.layer_one = GCNLayer(in_dim, hidden_dim, seed=seed, activation="relu")
        self.layer_two = GCNLayer(hidden_dim, out_dim, seed=seed + 1, activation="identity")

    def forward(self, features: np.ndarray, graph: RelationGraph) -> np.ndarray:
        normalized = normalized_adjacency(graph)
        hidden = self.layer_one.forward(features, normalized)
        return self.layer_two.forward(hidden, normalized)


__all__ = ["GCNEncoder", "GCNLayer", "normalized_adjacency"]
