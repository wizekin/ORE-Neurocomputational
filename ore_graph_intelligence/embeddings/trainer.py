"""Embedding trainer, evaluation, and calibration (ORE Point 8 steps 14-24).

The trainer learns TransE/DistMult/ComplEx/RotatE with the positive/negative ranking loss

    L = -sum log sigmoid(f(h,r,t)) - sum log sigmoid(-f(h',r,t')) + lambda ||theta||^2

using filtered negative sampling, fixed seeds, and a declared config/budget. It calibrates
confidence on held-out data (temperature scaling or isotonic regression), computes filtered
MRR / Hits@K metrics by relation type, and registers a reproducible model manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from ..schemas.model_manifest import SplitManifest
from .model_adapters import KGEModel, create_model, sigmoid
from .splitting import FilteredNegativeSampler, TripleRecord, build_split_manifest


def _canonical_config(config: Mapping[str, Any]) -> str:
    import json

    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class TrainingReport:
    """Loss history and evaluation metrics for one trained embedding model."""

    model_type: str
    config: dict[str, Any]
    loss_history: list[float] = field(default_factory=list)
    validation_mrr: float = 0.0
    validation_hits: dict[int, float] = field(default_factory=dict)
    test_mrr: float = 0.0
    test_hits: dict[int, float] = field(default_factory=dict)
    per_relation_mrr: dict[str, float] = field(default_factory=dict)
    epochs_run: int = 0
    negative_sampling_policy: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "config": self.config,
            "loss_history": list(self.loss_history),
            "validation_mrr": self.validation_mrr,
            "validation_hits": {str(k): v for k, v in self.validation_hits.items()},
            "test_mrr": self.test_mrr,
            "test_hits": {str(k): v for k, v in self.test_hits.items()},
            "per_relation_mrr": dict(self.per_relation_mrr),
            "epochs_run": self.epochs_run,
            "negative_sampling_policy": self.negative_sampling_policy,
        }


class AdamOptimizer:
    """Minimal deterministic Adam implemented on numpy (no external deep-learning dep)."""

    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = float(learning_rate)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._step = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self._step += 1
        bias1 = 1.0 - self.beta1 ** self._step
        bias2 = 1.0 - self.beta2 ** self._step
        for name, param in params.items():
            grad = grads[name]
            if name not in self._m:
                self._m[name] = np.zeros_like(param)
                self._v[name] = np.zeros_like(param)
            self._m[name] = self.beta1 * self._m[name] + (1.0 - self.beta1) * grad
            self._v[name] = self.beta2 * self._v[name] + (1.0 - self.beta2) * grad * grad
            m_hat = self._m[name] / bias1
            v_hat = self._v[name] / bias2
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


@dataclass
class TrainedEmbedding:
    """One trained embedding artifact with its split, vocabulary, and report."""

    model: KGEModel
    report: TrainingReport
    split: SplitManifest
    entity_ids: dict[str, int]
    relation_ids: dict[str, int]

    def scorer(
        self,
        *,
        calibrator: TemperatureCalibrator | IsotonicCalibrator | None = None,
        validation_statistics: Mapping[str, float] | None = None,
    ) -> EmbeddingScorer:
        """Build a scorer bound to the exact vocabulary this model was trained with."""
        return EmbeddingScorer(
            self.model, self.entity_ids, self.relation_ids,
            validation_statistics=validation_statistics, calibrator=calibrator,
        )

    def known_triples(self) -> frozenset[tuple[str, str, str]]:
        """Triples the model was trained or validated on; never include the test split."""
        return frozenset(self.split.train_triples) | frozenset(self.split.validation_triples)


class EmbeddingTrainer:
    """Train one embedding model end-to-end with leakage-safe data handling."""

    def __init__(
        self,
        *,
        model_type: str,
        embedding_dim: int = 32,
        learning_rate: float = 0.05,
        epochs: int = 50,
        negative_samples: int = 8,
        l2_lambda: float = 1e-4,
        seed: int = 0,
        batch_size: int = 256,
        norm_p: int = 2,
    ) -> None:
        if model_type not in {"TransE", "DistMult", "ComplEx", "RotatE"}:
            raise ValueError("model_type must be TransE, DistMult, ComplEx, or RotatE")
        if embedding_dim < 2 or epochs < 1 or negative_samples < 1:
            raise ValueError("embedding_dim, epochs, and negative_samples must be positive")
        if l2_lambda < 0:
            raise ValueError("l2_lambda cannot be negative")
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.l2_lambda = l2_lambda
        self.seed = seed
        self.batch_size = batch_size
        self.norm_p = norm_p

    # ------------------------------------------------------------------ data

    def train(
        self,
        records: Sequence[TripleRecord],
        *,
        split_policy: str = "relation",
    ) -> TrainedEmbedding:
        """Train on the train split, validate, and return the complete trained artifact."""
        if not records:
            raise ValueError("records must be non-empty")
        manifest = build_split_manifest(
            records, policy=split_policy, seed=self.seed,
            notes=("random_edge splits are rejected: they leak through neighboring facts",),
        )
        known = set(manifest.train_triples) | set(manifest.validation_triples)
        entities = sorted({head for head, _, _ in known} | {tail for _, _, tail in known})
        relations = sorted({relation for _, relation, _ in known})
        entity_ids = {name: index for index, name in enumerate(entities)}
        relation_ids = {name: index for index, name in enumerate(relations)}
        train_triples = [item for item in manifest.train_triples if item[1] in relation_ids]
        validation_triples = [item for item in manifest.validation_triples if item[1] in relation_ids]
        if not train_triples:
            raise ValueError("empty training split under the declared leakage policy")

        n_entities, n_relations = len(entity_ids), len(relation_ids)
        model = create_model(
            self.model_type, n_entities, n_relations, self.embedding_dim, seed=self.seed,
            norm_p=self.norm_p,
        )
        sampler = FilteredNegativeSampler(
            set(manifest.train_triples) | set(manifest.validation_triples),
        )
        rng = np.random.default_rng(self.seed)
        optimizer = AdamOptimizer(self.learning_rate)
        train_array = np.array(
            [[entity_ids[h], relation_ids[r], entity_ids[t]] for h, r, t in train_triples],
            dtype=np.int64,
        )
        report = TrainingReport(
            model_type=self.model_type,
            config=self._config_dict(split_policy),
            negative_sampling_policy=sampler.policy_description(),
        )

        for _epoch in range(self.epochs):
            order = rng.permutation(len(train_array))
            epoch_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = train_array[order[start: start + self.batch_size]]
                heads, rels, tails = batch[:, 0], batch[:, 1], batch[:, 2]
                n_neg = self.negative_samples * len(batch)
                negatives = sampler.sample(rng, n_neg)
                if not negatives:
                    continue
                neg_array = np.array(
                    [[entity_ids[h], relation_ids[r], entity_ids[t]] for h, r, t in negatives],
                    dtype=np.int64,
                )
                loss, grads = self._loss_and_grads(model, heads, rels, tails, neg_array)
                for name in grads:
                    grads[name] += 2.0 * self.l2_lambda * model.parameters()[name]
                optimizer.step(model.parameters(), grads)
                epoch_loss += float(loss)
            report.loss_history.append(epoch_loss / max(len(order), 1))

        report.epochs_run = len(report.loss_history)
        if validation_triples:
            ranking = FilteredRankingEvaluator(model, entity_ids, relation_ids)
            metrics = ranking.evaluate(
                validation_triples, all_true=known,
            )
            report.validation_mrr = metrics["mrr"]
            report.validation_hits = {k: metrics[f"hits@{k}"] for k in (1, 3, 10)}
        report.config["trained_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return TrainedEmbedding(
            model=model,
            report=report,
            split=manifest,
            entity_ids=entity_ids,
            relation_ids=relation_ids,
        )

    def _config_dict(self, split_policy: str) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "embedding_dim": self.embedding_dim,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "negative_samples": self.negative_samples,
            "l2_lambda": self.l2_lambda,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "norm_p": self.norm_p,
            "split_policy": split_policy,
        }

    def _loss_and_grads(
        self,
        model: KGEModel,
        heads: np.ndarray,
        rels: np.ndarray,
        tails: np.ndarray,
        neg_array: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray]]:
        positive_scores = model.score_triples(heads, rels, tails)
        negative_scores = model.score_triples(neg_array[:, 0], neg_array[:, 1], neg_array[:, 2])
        positive_weights = sigmoid(positive_scores) - 1.0
        negative_weights = sigmoid(negative_scores)
        grads = model.zero_grads()
        model.accumulate_gradients(heads, rels, tails, positive_weights, grads)
        model.accumulate_gradients(
            neg_array[:, 0], neg_array[:, 1], neg_array[:, 2], negative_weights, grads,
        )
        eps = 1e-12
        positive_loss = np.logaddexp(0.0, -positive_scores).sum()
        negative_loss = np.logaddexp(0.0, negative_scores).sum()
        total = float(positive_loss + negative_loss)
        total += self.l2_lambda * sum(
            float(np.sum(param * param)) for param in model.parameters().values()
        )
        return total / (len(heads) + len(neg_array) + eps), grads


class FilteredRankingEvaluator:
    """Filtered MRR / Hits@K evaluation (never ranks a known true fact above the test triple)."""

    def __init__(
        self,
        model: KGEModel,
        entity_ids: Mapping[str, int],
        relation_ids: Mapping[str, int],
    ) -> None:
        self.model = model
        self.entity_ids = dict(entity_ids)
        self.relation_ids = dict(relation_ids)
        self._entity_index = np.arange(len(self.entity_ids), dtype=np.int64)

    def evaluate(
        self,
        triples: Sequence[tuple[str, str, str]],
        *,
        all_true: set[tuple[str, str, str]],
    ) -> dict[str, float]:
        """Compute filtered MRR, Hits@1/3/10, and per-relation MRR."""
        reciprocal_ranks: list[float] = []
        per_relation: dict[str, list[float]] = {}
        hits = {1: 0, 3: 0, 10: 0}
        counted = 0
        for head, relation, tail in triples:
            if head not in self.entity_ids or tail not in self.entity_ids or relation not in self.relation_ids:
                continue
            h_idx = self.entity_ids[head]
            r_idx = self.relation_ids[relation]
            candidate_tails = [
                entity for entity, index in self.entity_ids.items()
                if (head, relation, entity) not in all_true or entity == tail
            ]
            if len(candidate_tails) < 2:
                continue
            tail_indices = np.array([self.entity_ids[e] for e in candidate_tails], dtype=np.int64)
            scores = self.model.score_triples(
                np.full(len(tail_indices), h_idx, dtype=np.int64),
                np.full(len(tail_indices), r_idx, dtype=np.int64),
                tail_indices,
            )
            order = np.argsort(-scores)
            rank_positions = np.empty(len(order), dtype=np.int64)
            rank_positions[order] = np.arange(len(order))
            rank = int(rank_positions[list(candidate_tails).index(tail)]) + 1
            reciprocal_ranks.append(1.0 / rank)
            per_relation.setdefault(relation, []).append(1.0 / rank)
            for k in hits:
                if rank <= k:
                    hits[k] += 1
            counted += 1
        if counted == 0:
            return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0}
        return {
            "mrr": float(np.mean(reciprocal_ranks)),
            "hits@1": hits[1] / counted,
            "hits@3": hits[3] / counted,
            "hits@10": hits[10] / counted,
            "per_relation_mrr": {
                relation: float(np.mean(values)) for relation, values in per_relation.items()
            },
        }


class TemperatureCalibrator:
    """Temperature scaling fitted on held-out scores (ORE Point 8 step 23)."""

    def __init__(self) -> None:
        self.temperature: float | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Fit one temperature minimizing held-out negative log likelihood."""
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if scores.size < 2 or not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("calibration needs at least two scored examples with 0/1 labels")
        best_temperature, best_loss = 1.0, np.inf
        for candidate in np.concatenate([
            np.linspace(0.25, 4.0, 76), np.linspace(4.0, 16.0, 25),
        ]):
            scaled = scores / candidate
            loss = float((np.logaddexp(0.0, scaled) - labels * scaled).mean())
            if loss < best_loss:
                best_temperature, best_loss = float(candidate), loss
        self.temperature = best_temperature
        return best_temperature

    def probability(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw scores into calibrated probabilities; unavailable before fitting."""
        if self.temperature is None:
            raise RuntimeError("calibrator is not fitted; raw scores stay uncalibrated ranking values")
        return sigmoid(np.asarray(scores, dtype=np.float64) / self.temperature)


class IsotonicCalibrator:
    """Pool-adjacent-violators isotonic regression on held-out scores."""

    def __init__(self) -> None:
        self._thresholds: np.ndarray | None = None
        self._values: np.ndarray | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if scores.size < 2:
            raise ValueError("isotonic calibration needs at least two scored examples")
        order = np.argsort(scores)
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        blocks: list[tuple[float, float, int]] = []
        for score, label in zip(sorted_scores, sorted_labels):
            blocks.append((float(label), float(label), 1))
            while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
                total_sum = blocks[-2][0] * blocks[-2][2] + blocks[-1][0] * blocks[-1][2]
                total_count = blocks[-2][2] + blocks[-1][2]
                blocks.pop()
                blocks.pop()
                blocks.append((total_sum / total_count, 0.0, total_count))
        self._thresholds = np.array([item[0] for item in blocks])
        means = []
        running = 0.0
        counts = []
        for index, (mean, _, count) in enumerate(blocks):
            running += mean * count
            counts.append(count)
            means.append(running / sum(counts))
        self._values = np.array(means)

    def probability(self, scores: np.ndarray) -> np.ndarray:
        if self._thresholds is None:
            raise RuntimeError("calibrator is not fitted; raw scores stay uncalibrated ranking values")
        thresholds = np.unique(self._thresholds)
        values = self._values[: len(thresholds)]
        positions = np.searchsorted(thresholds, np.asarray(scores, dtype=np.float64), side="right")
        return np.clip(values[np.minimum(positions, len(values) - 1)], 0.0, 1.0)


class EmbeddingScorer:
    """Expose raw ranking scores and - only when calibrated - probabilities (step 23)."""

    def __init__(
        self,
        model: KGEModel,
        entity_ids: Mapping[str, int],
        relation_ids: Mapping[str, int],
        *,
        validation_statistics: Mapping[str, float] | None = None,
        calibrator: TemperatureCalibrator | IsotonicCalibrator | None = None,
    ) -> None:
        self.model = model
        self.entity_ids = dict(entity_ids)
        self.relation_ids = dict(relation_ids)
        self.validation_statistics = dict(validation_statistics or {})
        self.calibrator = calibrator

    def raw_score(self, subject: str, relation: str, obj: str) -> float:
        """Return the raw model score; an uncalibrated ranking value, never a probability."""
        if subject not in self.entity_ids or obj not in self.entity_ids or relation not in self.relation_ids:
            raise KeyError(f"unknown entities or relation: {subject}, {relation}, {obj}")
        score = self.model.score_triples(
            np.array([self.entity_ids[subject]], dtype=np.int64),
            np.array([self.relation_ids[relation]], dtype=np.int64),
            np.array([self.entity_ids[obj]], dtype=np.int64),
        )
        return float(score[0])

    def calibrated_probability(self, subject: str, relation: str, obj: str) -> float | None:
        """Return a held-out-calibrated probability, or None when no calibrator is fitted."""
        if self.calibrator is None:
            return None
        score = np.array([self.raw_score(subject, relation, obj)], dtype=np.float64)
        return float(self.calibrator.probability(score)[0])

    def normalized_score(self, subject: str, relation: str, obj: str) -> float:
        """Z-normalize with validation statistics when available; otherwise return raw."""
        raw = self.raw_score(subject, relation, obj)
        mean = self.validation_statistics.get("mean")
        std = self.validation_statistics.get("std")
        if mean is None or std is None or std == 0.0:
            return raw
        return float((raw - mean) / std)


__all__ = [
    "AdamOptimizer", "EmbeddingScorer", "EmbeddingTrainer", "FilteredRankingEvaluator",
    "IsotonicCalibrator", "TemperatureCalibrator", "TrainingReport",
]
