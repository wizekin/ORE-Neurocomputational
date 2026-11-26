"""Knowledge-graph embedding models: TransE, DistMult, ComplEx, and RotatE (Phase B, steps 14-15).

All four models are implemented natively on numpy with manual gradients so ORE keeps its
declared dependency footprint (numpy/scipy only). PyKEEN remains a replaceable experiment
adapter at the trainer boundary (ORE Point 8 step 14). Learned scores rank candidates;
they never establish truth.

Relation-pattern guidance (step 15): TransE models antisymmetric relations, DistMult
models symmetric relations, ComplEx adds asymmetric expressive power over DistMult, and
RotatE models symmetry, antisymmetry, inverse, and composition patterns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODEL_NAMES = ("TransE", "DistMult", "ComplEx", "RotatE")

_RELATION_PATTERNS: dict[str, frozenset[str]] = {
    "TransE": frozenset({"antisymmetry", "inverse", "composition"}),
    "DistMult": frozenset({"symmetry"}),
    "ComplEx": frozenset({"symmetry", "antisymmetry", "inverse"}),
    "RotatE": frozenset({"symmetry", "antisymmetry", "inverse", "composition"}),
}


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


@dataclass(frozen=True, slots=True)
class RelationBehaviorProfile:
    """Observed structural behavior of one relation type in the projection (step 15)."""

    relation_type: str
    triple_count: int
    symmetry_fraction: float
    inverse_pair_fraction: float
    one_to_many_fraction: float
    recommended_models: tuple[str, ...]
    rejected_models: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "relation_type": self.relation_type,
            "triple_count": self.triple_count,
            "symmetry_fraction": self.symmetry_fraction,
            "inverse_pair_fraction": self.inverse_pair_fraction,
            "one_to_many_fraction": self.one_to_many_fraction,
            "recommended_models": list(self.recommended_models),
            "rejected_models": list(self.rejected_models),
        }


class KGEModel:
    """Base class for numpy knowledge-graph embedding models with manual gradients."""

    name = "KGEModel"

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embedding_dim: int,
        *,
        seed: int = 0,
    ) -> None:
        if n_entities < 1 or n_relations < 1:
            raise ValueError("n_entities and n_relations must be positive")
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        self.n_entities = int(n_entities)
        self.n_relations = int(n_relations)
        self.embedding_dim = int(embedding_dim)
        self.rng = np.random.default_rng(seed)
        self._init_parameters()

    # ------------------------------------------------------------------ parameters

    def _init_parameters(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def parameters(self) -> dict[str, np.ndarray]:
        """Return trainable arrays by name; ordering must be deterministic."""
        raise NotImplementedError  # pragma: no cover - abstract

    def zero_grads(self) -> dict[str, np.ndarray]:
        """Allocate a zero gradient dictionary matching ``parameters``."""
        return {
            name: np.zeros_like(array, dtype=np.float64)
            for name, array in self.parameters().items()
        }

    # ------------------------------------------------------------------ scoring

    def score_triples(
        self, heads: np.ndarray, relations: np.ndarray, tails: np.ndarray,
    ) -> np.ndarray:
        """Score (head, relation, tail) index triples; higher means more plausible."""
        raise NotImplementedError  # pragma: no cover - abstract

    def accumulate_gradients(
        self,
        heads: np.ndarray,
        relations: np.ndarray,
        tails: np.ndarray,
        grad_weights: np.ndarray,
        grads: dict[str, np.ndarray],
    ) -> None:
        """Add ``grad_weights[i] * d score_i / d theta`` into ``grads`` (upstream weights)."""
        raise NotImplementedError  # pragma: no cover - abstract

    # ------------------------------------------------------------------ helpers

    def _gather(self, table: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return table[indices]

    def _scatter_add(
        self, grads: dict[str, np.ndarray], name: str, indices: np.ndarray, values: np.ndarray,
    ) -> None:
        np.add.at(grads[name], indices, values)

    def relation_behavior(self) -> frozenset[str]:
        """Return the relation patterns this model can represent (step 15)."""
        return _RELATION_PATTERNS.get(self.name, frozenset())


class TransE(KGEModel):
    """TransE: f(h, r, t) = -||e_h + e_r - e_t||_p (ORE Point 8 section 2)."""

    name = "TransE"

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embedding_dim: int,
        *,
        seed: int = 0,
        norm_p: int = 2,
        margin_scale: float = 1.0,
    ) -> None:
        if norm_p not in (1, 2):
            raise ValueError("norm_p must be 1 or 2")
        self.norm_p = int(norm_p)
        self.margin_scale = float(margin_scale)
        super().__init__(n_entities, n_relations, embedding_dim, seed=seed)

    def _init_parameters(self) -> None:
        bound = 6.0 / np.sqrt(self.embedding_dim)
        self.entity_embeddings = self.rng.uniform(-bound, bound, (self.n_entities, self.embedding_dim))
        self.relation_embeddings = self.rng.uniform(-bound, bound, (self.n_relations, self.embedding_dim))
        norms = np.linalg.norm(self.relation_embeddings, axis=1, keepdims=True)
        self.relation_embeddings = self.relation_embeddings / np.maximum(norms, 1e-12)

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "entity_embeddings": self.entity_embeddings,
            "relation_embeddings": self.relation_embeddings,
        }

    def score_triples(self, heads, relations, tails):
        difference = (
            self.entity_embeddings[heads]
            + self.relation_embeddings[relations]
            - self.entity_embeddings[tails]
        )
        if self.norm_p == 2:
            distance = np.linalg.norm(difference, axis=1)
        else:
            distance = np.abs(difference).sum(axis=1)
        return -self.margin_scale * distance

    def accumulate_gradients(self, heads, relations, tails, grad_weights, grads):
        difference = (
            self.entity_embeddings[heads]
            + self.relation_embeddings[relations]
            - self.entity_embeddings[tails]
        )
        if self.norm_p == 2:
            norm = np.maximum(np.linalg.norm(difference, axis=1, keepdims=True), 1e-12)
            d_score = -self.margin_scale * difference / norm
        else:
            d_score = -self.margin_scale * np.sign(difference)
        weighted = grad_weights[:, None] * d_score
        self._scatter_add(grads, "entity_embeddings", heads, weighted)
        self._scatter_add(grads, "relation_embeddings", relations, weighted)
        self._scatter_add(grads, "entity_embeddings", tails, -weighted)


class DistMult(KGEModel):
    """DistMult: f(h, r, t) = sum(e_h * e_r * e_t); symmetric by construction."""

    name = "DistMult"

    def _init_parameters(self) -> None:
        bound = 6.0 / np.sqrt(self.embedding_dim)
        self.entity_embeddings = self.rng.uniform(-bound, bound, (self.n_entities, self.embedding_dim))
        self.relation_embeddings = self.rng.uniform(-bound, bound, (self.n_relations, self.embedding_dim))

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "entity_embeddings": self.entity_embeddings,
            "relation_embeddings": self.relation_embeddings,
        }

    def score_triples(self, heads, relations, tails):
        product = (
            self.entity_embeddings[heads]
            * self.relation_embeddings[relations]
            * self.entity_embeddings[tails]
        )
        return product.sum(axis=1)

    def accumulate_gradients(self, heads, relations, tails, grad_weights, grads):
        h = self.entity_embeddings[heads]
        r = self.relation_embeddings[relations]
        t = self.entity_embeddings[tails]
        weighted = grad_weights[:, None]
        self._scatter_add(grads, "entity_embeddings", heads, weighted * r * t)
        self._scatter_add(grads, "relation_embeddings", relations, weighted * h * t)
        self._scatter_add(grads, "entity_embeddings", tails, weighted * h * r)


class ComplEx(KGEModel):
    """ComplEx: f = Re(<e_h, e_r, conj(e_t)>) over complex embeddings stored as [re, im]."""

    name = "ComplEx"

    def _init_parameters(self) -> None:
        bound = 6.0 / np.sqrt(self.embedding_dim)
        shape = (self.n_entities, self.embedding_dim, 2)
        self.entity_embeddings = self.rng.uniform(-bound, bound, shape)
        self.relation_embeddings = self.rng.uniform(-bound, bound, (self.n_relations, self.embedding_dim, 2))

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "entity_embeddings": self.entity_embeddings,
            "relation_embeddings": self.relation_embeddings,
        }

    def score_triples(self, heads, relations, tails):
        h = self.entity_embeddings[heads]
        r = self.relation_embeddings[relations]
        t = self.entity_embeddings[tails]
        return (
            h[:, :, 0] * (r[:, :, 0] * t[:, :, 0] + r[:, :, 1] * t[:, :, 1])
            + h[:, :, 1] * (r[:, :, 0] * t[:, :, 1] - r[:, :, 1] * t[:, :, 0])
        ).sum(axis=1)

    def accumulate_gradients(self, heads, relations, tails, grad_weights, grads):
        h = self.entity_embeddings[heads]
        r = self.relation_embeddings[relations]
        t = self.entity_embeddings[tails]
        weighted = grad_weights[:, None, None]
        # f = a(c*e + d*g) + b(c*g - d*e) with h=a+bi, r=c+di, t=e+gi
        # d f / d a = c*e + d*g ; d f / d b = c*g - d*e
        grad_h = np.stack([
            r[:, :, 0] * t[:, :, 0] + r[:, :, 1] * t[:, :, 1],
            r[:, :, 0] * t[:, :, 1] - r[:, :, 1] * t[:, :, 0],
        ], axis=2)
        # d f / d r_re = h_re*t_re + h_im*t_im ; d f / d r_im = h_re*t_im - h_im*t_re
        grad_r = np.stack([
            h[:, :, 0] * t[:, :, 0] + h[:, :, 1] * t[:, :, 1],
            h[:, :, 0] * t[:, :, 1] - h[:, :, 1] * t[:, :, 0],
        ], axis=2)
        # d f / d t_re = h_re*r_re - h_im*r_im ; d f / d t_im = h_re*r_im + h_im*r_re
        grad_t = np.stack([
            h[:, :, 0] * r[:, :, 0] - h[:, :, 1] * r[:, :, 1],
            h[:, :, 0] * r[:, :, 1] + h[:, :, 1] * r[:, :, 0],
        ], axis=2)
        self._scatter_add(grads, "entity_embeddings", heads, weighted * grad_h)
        self._scatter_add(grads, "relation_embeddings", relations, weighted * grad_r)
        self._scatter_add(grads, "entity_embeddings", tails, weighted * grad_t)


class RotatE(KGEModel):
    """RotatE: f(h, r, t) = -||e_h \\circ e_r - e_t|| with unit-modulus phase rotations."""

    name = "RotatE"

    def _init_parameters(self) -> None:
        bound = 6.0 / np.sqrt(self.embedding_dim)
        self.entity_embeddings = self.rng.uniform(-bound, bound, (self.n_entities, self.embedding_dim, 2))
        self.relation_phases = self.rng.uniform(0.0, 2.0 * np.pi, (self.n_relations, self.embedding_dim))

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "entity_embeddings": self.entity_embeddings,
            "relation_phases": self.relation_phases,
        }

    def _rotate(self, heads: np.ndarray, relations: np.ndarray) -> np.ndarray:
        h = self.entity_embeddings[heads]
        phase = self.relation_phases[relations]
        cos = np.cos(phase)
        sin = np.sin(phase)
        rotated = np.empty_like(h)
        rotated[:, :, 0] = h[:, :, 0] * cos - h[:, :, 1] * sin
        rotated[:, :, 1] = h[:, :, 0] * sin + h[:, :, 1] * cos
        return rotated

    def score_triples(self, heads, relations, tails):
        difference = self._rotate(heads, relations) - self.entity_embeddings[tails]
        return -np.linalg.norm(difference, axis=(1, 2))

    def accumulate_gradients(self, heads, relations, tails, grad_weights, grads):
        h = self.entity_embeddings[heads]
        phase = self.relation_phases[relations]
        cos = np.cos(phase)
        sin = np.sin(phase)
        rotated = np.empty_like(h)
        rotated[:, :, 0] = h[:, :, 0] * cos - h[:, :, 1] * sin
        rotated[:, :, 1] = h[:, :, 0] * sin + h[:, :, 1] * cos
        difference = rotated - self.entity_embeddings[tails]
        norm = np.maximum(np.linalg.norm(difference, axis=(1, 2), keepdims=True), 1e-12)
        d_score_du = -difference / norm  # (batch, d, 2)
        weighted = grad_weights[:, None, None]
        # d f / d h = R(theta)^T d_score_du
        grad_h = np.empty_like(h)
        grad_h[:, :, 0] = d_score_du[:, :, 0] * cos + d_score_du[:, :, 1] * sin
        grad_h[:, :, 1] = -d_score_du[:, :, 0] * sin + d_score_du[:, :, 1] * cos
        # d f / d t = -d_score_du
        self._scatter_add(grads, "entity_embeddings", heads, weighted * grad_h)
        self._scatter_add(grads, "entity_embeddings", tails, weighted * (-d_score_du))
        # d u_x/d theta = -u_y ; d u_y/d theta = u_x  =>  d f/d theta = d_x*(-u_y) + d_y*(u_x)
        d_theta = (
            d_score_du[:, :, 0] * (-rotated[:, :, 1]) + d_score_du[:, :, 1] * rotated[:, :, 0]
        )
        self._scatter_add(grads, "relation_phases", relations, grad_weights[:, None] * d_theta)


def create_model(
    model_name: str,
    n_entities: int,
    n_relations: int,
    embedding_dim: int,
    *,
    seed: int = 0,
    norm_p: int = 2,
) -> KGEModel:
    """Factory for the four supported embedding models."""
    factories: dict[str, type[KGEModel]] = {
        "TransE": TransE,
        "DistMult": DistMult,
        "ComplEx": ComplEx,
        "RotatE": RotatE,
    }
    if model_name not in factories:
        raise ValueError(f"model_name must be one of {sorted(factories)}")
    if model_name == "TransE":
        return TransE(
            n_entities, n_relations, embedding_dim, seed=seed, norm_p=norm_p,
        )
    return factories[model_name](n_entities, n_relations, embedding_dim, seed=seed)


def profile_relation_behavior(
    triples: list[tuple[str, str, str]],
) -> dict[str, RelationBehaviorProfile]:
    """Measure symmetry/inverse/one-to-many behavior per relation type (step 15)."""
    triples_by_relation: dict[str, list[tuple[str, str, str]]] = {}
    for head, relation, tail in triples:
        triples_by_relation.setdefault(relation, []).append((head, relation, tail))
    profiles: dict[str, RelationBehaviorProfile] = {}
    triple_set = set(triples)
    for relation in sorted(triples_by_relation):
        instances = triples_by_relation[relation]
        count = len(instances)
        symmetric = sum(
            1 for head, _, tail in instances
            if head != tail and (tail, relation, head) in triple_set
        )
        inverse_present = sum(
            1 for head, _, tail in instances
            if any(other[0] == tail and other[2] == head and other[1] != relation for other in instances)
        )
        subjects = {head for head, _, _ in instances}
        tails_by_subject: dict[str, int] = {}
        for head, _, tail in instances:
            tails_by_subject[head] = tails_by_subject.get(head, 0) + 1
        one_to_many = sum(1 for total in tails_by_subject.values() if total > 1) / max(len(subjects), 1)
        symmetry_fraction = symmetric / count if count else 0.0
        if symmetry_fraction >= 0.5:
            recommended: tuple[str, ...] = ("DistMult", "ComplEx")
            rejected = ("TransE",)
        elif inverse_present > 0:
            recommended = ("RotatE", "ComplEx")
            rejected = ("DistMult",)
        else:
            recommended = ("TransE", "RotatE")
            rejected = ("DistMult",)
        profiles[relation] = RelationBehaviorProfile(
            relation_type=relation,
            triple_count=count,
            symmetry_fraction=symmetry_fraction,
            inverse_pair_fraction=inverse_present / count if count else 0.0,
            one_to_many_fraction=one_to_many,
            recommended_models=recommended,
            rejected_models=rejected,
        )
    return profiles


__all__ = [
    "MODEL_NAMES", "ComplEx", "DistMult", "KGEModel", "RelationBehaviorProfile", "RotatE",
    "TransE", "create_model", "profile_relation_behavior", "sigmoid",
]
