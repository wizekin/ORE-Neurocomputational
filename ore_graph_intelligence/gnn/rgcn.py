"""Relational graph convolutional network (R-GCN) with basis decomposition (ORE Point 8 step 17).

Implements the exact R-GCN update

    h_i^(l+1) = sigma( sum_r sum_{j in N_i^r} (1/c_{i,r}) W_r h_j^l + W_0 h_i^l )

with basis decomposition W_r = sum_b a_{rb} V_b so parameter count is controlled when the
relation count makes per-relation full matrices costly. USG relation types (is_a, part_of,
causes, has_property, ...) stay distinct: each relation type gets its own aggregation pass
with its own composed transform; relation types are never treated as interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _xavier(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (rows + columns))
    return rng.uniform(-limit, limit, (rows, columns))


@dataclass
class RelationGraph:
    """Relation-typed adjacency: per relation, edge arrays and per-target normalization."""

    n_nodes: int
    edges_by_relation: dict[str, tuple[np.ndarray, np.ndarray]]

    @classmethod
    def from_triples(
        cls,
        triples: Sequence[tuple[int, int, int]],
        relation_names: Sequence[str],
        *,
        n_nodes: int,
    ) -> "RelationGraph":
        """Build relation adjacency from (subject, relation_index, object) index triples.

        Edges are directed subject -> object, exactly as projected; reversing an edge is a
        distinct graph unless the relation is explicitly symmetric.
        """
        names = list(relation_names)
        buckets: dict[str, list[tuple[int, int]]] = {name: [] for name in names}
        for subject, relation, obj in triples:
            name = names[int(relation)]
            buckets[name].append((int(subject), int(obj)))
        edges_by_relation: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in names:
            pairs = buckets[name]
            if not pairs:
                edges_by_relation[name] = (
                    np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                )
            else:
                sources = np.array([pair[0] for pair in pairs], dtype=np.int64)
                targets = np.array([pair[1] for pair in pairs], dtype=np.int64)
                edges_by_relation[name] = (sources, targets)
        return cls(n_nodes=n_nodes, edges_by_relation=edges_by_relation)

    def normalization_counts(self, relation: str) -> np.ndarray:
        """c_{i,r}: number of r-edges pointing at each target node i."""
        sources, targets = self.edges_by_relation[relation]
        counts = np.zeros(self.n_nodes, dtype=np.float64)
        if targets.size:
            np.add.at(counts, targets, 1.0)
        return counts


class RGCNLayer:
    """One R-GCN layer with basis decomposition W_r = sum_b a_{rb} V_b."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        relations: Sequence[str],
        *,
        basis_size: int = 3,
        seed: int = 0,
        activation: str = "relu",
    ) -> None:
        if in_dim < 1 or out_dim < 1 or basis_size < 1:
            raise ValueError("dimensions and basis size must be positive")
        if not relations:
            raise ValueError("relations must be non-empty")
        if activation not in {"relu", "tanh", "identity"}:
            raise ValueError("activation must be relu, tanh, or identity")
        self.in_dim, self.out_dim = in_dim, out_dim
        self.relations = list(relations)
        self.basis_size = int(basis_size)
        self.activation = activation
        rng = np.random.default_rng(seed)
        self.basis = np.stack([
            _xavier(rng, in_dim, out_dim) for _ in range(self.basis_size)
        ])  # (B, in, out)
        self.coefficients = rng.normal(0.0, 0.1, (len(self.relations), self.basis_size))
        self.self_transform = _xavier(rng, in_dim, out_dim)

    # ------------------------------------------------------------------ forward

    def relation_transform(self, relation_index: int) -> np.ndarray:
        """Compose W_r = sum_b a_{rb} V_b for one relation."""
        composed = np.einsum("b,bio->io", self.coefficients[relation_index], self.basis)
        return composed

    def forward(
        self,
        features: np.ndarray,
        graph: RelationGraph,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Compute h' = phi(W_0 h + sum_r (1/c_{i,r}) sum_{j in N_i^r} W_r h_j).

        Returns the new representations plus the cache needed for exact backpropagation.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (graph.n_nodes, self.in_dim):
            raise ValueError("feature matrix shape mismatch")
        pre_activation = features @ self.self_transform
        aggregation_cache: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for relation_index, relation in enumerate(self.relations):
            sources, targets = graph.edges_by_relation[relation]
            if sources.size == 0:
                continue
            w_r = self.relation_transform(relation_index)
            messages = features[sources] @ w_r  # (edges, out)
            counts = np.zeros(graph.n_nodes, dtype=np.float64)
            np.add.at(counts, targets, 1.0)
            counts = np.maximum(counts, 1.0)
            messages = messages / counts[targets][:, None]
            aggregated = np.zeros((graph.n_nodes, self.out_dim), dtype=np.float64)
            np.add.at(aggregated, targets, messages)
            pre_activation += aggregated
            aggregation_cache.append((relation, w_r, sources, targets, counts))
        output = self._activate(pre_activation)
        cache = {
            "features": features,
            "pre_activation": pre_activation,
            "aggregations": aggregation_cache,
            "counts_all": {
                relation: graph.normalization_counts(relation) for relation in self.relations
            },
        }
        return output, cache

    def _activate(self, values: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(values, 0.0)
        if self.activation == "tanh":
            return np.tanh(values)
        return values

    def _activation_derivative(self, values: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (values > 0.0).astype(np.float64)
        if self.activation == "tanh":
            output = np.tanh(values)
            return 1.0 - output * output
        return np.ones_like(values)

    # ------------------------------------------------------------------ backward

    def backward(
        self,
        cache: Mapping[str, object],
        grad_output: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Backpropagate through this layer; return dL/dh and parameter gradients."""
        features = np.asarray(cache["features"], dtype=np.float64)
        pre_activation = np.asarray(cache["pre_activation"], dtype=np.float64)
        aggregations = list(cache["aggregations"])
        dz = grad_output * self._activation_derivative(pre_activation)
        grad_self = features.T @ dz
        grad_features = dz @ self.self_transform.T
        grad_basis = np.zeros_like(self.basis)
        grad_coefficients = np.zeros_like(self.coefficients)
        for entry in aggregations:
            relation, w_r, sources, targets, counts = entry
            # Reconstruct per-edge normalized messages gradient: dz scattered to edges.
            edge_grad = dz[targets]
            # dW_r += sum over edges (1/c_{i,r}) dz_i h_j^T
            normalized = features[sources] * (1.0 / counts[targets])[:, None]
            grad_w_r = normalized.T @ edge_grad
            # d h_j += (1/c_{i,r}) W_r^T dz_i
            grad_features[sources] += (edge_grad / counts[targets][:, None]) @ w_r.T
            # Basis decomposition gradients: W_r = sum_b a_rb V_b
            for basis_index in range(self.basis_size):
                coefficient = self.coefficients[
                    self.relations.index(relation), basis_index
                ]
                grad_basis[basis_index] += coefficient * grad_w_r
                grad_coefficients[
                    self.relations.index(relation), basis_index
                ] += float(np.sum(grad_w_r * self.basis[basis_index]))
        return grad_features, {
            "self_transform": grad_self,
            "basis": grad_basis,
            "coefficients": grad_coefficients,
        }

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "basis": self.basis,
            "coefficients": self.coefficients,
            "self_transform": self.self_transform,
        }

    def apply_gradients(self, grads: Mapping[str, np.ndarray], learning_rate: float) -> None:
        self.basis -= learning_rate * grads["basis"]
        self.coefficients -= learning_rate * grads["coefficients"]
        self.self_transform -= learning_rate * grads["self_transform"]


class RGCNEncoder:
    """Stacked R-GCN encoder producing node embeddings from typed relation aggregation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        relations: Sequence[str],
        *,
        basis_size: int = 3,
        seed: int = 0,
        activation: str = "relu",
    ) -> None:
        self.layer_one = RGCNLayer(
            in_dim, hidden_dim, relations, basis_size=basis_size, seed=seed,
            activation=activation,
        )
        self.layer_two = RGCNLayer(
            hidden_dim, out_dim, relations, basis_size=basis_size, seed=seed + 1,
            activation="identity",
        )
        self.relations = list(relations)

    def forward(self, features: np.ndarray, graph: RelationGraph) -> tuple[np.ndarray, list[dict[str, object]]]:
        """Encode nodes; returns final embeddings and per-layer caches for backprop."""
        hidden, cache_one = self.layer_one.forward(features, graph)
        output, cache_two = self.layer_two.forward(hidden, graph)
        return output, [cache_one, cache_two]

    def backward(self, caches: list[dict[str, object]], grad_output: np.ndarray) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        """Backpropagate through the stack; returns input gradient and per-layer param grads."""
        grad_hidden, grads_two = self.layer_two.backward(caches[1], grad_output)
        grad_input, grads_one = self.layer_one.backward(caches[0], grad_hidden)
        return grad_input, [grads_one, grads_two]

    def parameters(self) -> dict[str, np.ndarray]:
        merged: dict[str, np.ndarray] = {}
        for prefix, layer in (("l1_", self.layer_one), ("l2_", self.layer_two)):
            for name, array in layer.parameters().items():
                merged[f"{prefix}{name}"] = array
        return merged

    def apply_gradients(self, layer_grads: list[dict[str, np.ndarray]], learning_rate: float) -> None:
        self.layer_one.apply_gradients(layer_grads[0], learning_rate)
        self.layer_two.apply_gradients(layer_grads[1], learning_rate)


class DistMultDecoder:
    """DistMult scoring head over R-GCN node embeddings (task-appropriate objective, step 21)."""

    def __init__(self, embedding_dim: int, n_relations: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.relation_vectors = rng.normal(0.0, 0.1, (n_relations, embedding_dim))

    def score(self, embeddings: np.ndarray, subjects: np.ndarray, relations: np.ndarray, objects: np.ndarray) -> np.ndarray:
        return np.sum(
            embeddings[subjects] * self.relation_vectors[relations] * embeddings[objects],
            axis=1,
        )

    def gradients(
        self,
        embeddings: np.ndarray,
        subjects: np.ndarray,
        relations: np.ndarray,
        objects: np.ndarray,
        grad_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (dL/d embeddings scattered, dL/d relation vectors scattered)."""
        grad_embeddings = np.zeros_like(embeddings)
        grad_relations = np.zeros_like(self.relation_vectors)
        np.add.at(
            grad_embeddings, subjects,
            grad_weights[:, None] * self.relation_vectors[relations] * embeddings[objects],
        )
        np.add.at(
            grad_embeddings, objects,
            grad_weights[:, None] * self.relation_vectors[relations] * embeddings[subjects],
        )
        np.add.at(
            grad_relations, relations,
            grad_weights[:, None] * embeddings[subjects] * embeddings[objects],
        )
        return grad_embeddings, grad_relations


__all__ = ["DistMultDecoder", "RGCNEncoder", "RGCNLayer", "RelationGraph"]
