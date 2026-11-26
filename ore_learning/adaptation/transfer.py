"""Transfer learning and adaptation in an isolated laboratory (Steps 12-16).

Frozen feature extraction is tried first; the smallest suitable adaptation method is
selected only when reuse is insufficient; datasets are reproducible with split
lineage and leakage audits; candidates train in isolation with recorded seeds and
hyperparameters; adapted and base models are compared on identical held-out tasks;
promotion requires a predefined improvement or an approved trade-off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import brier_score, mse, normalize_to_unit_interval, sigmoid

__all__ = ("AdaptationComparison", "AdaptationDataset", "AdaptationMethod",
           "AdaptationPolicyError", "IsolatedTrainingRun", "PromotionDecision",
           "adaptation_method_ladder", "evaluate_promotion", "frozen_feature_quality",
           "select_adaptation_method", "split_dataset")


class AdaptationPolicyError(ValueError):
    """Raised when an adaptation step violates the isolation policy."""


class AdaptationMethod(str, Enum):
    """Adaptation methods ordered from smallest to most invasive (Step 13)."""

    FROZEN_REUSE = "frozen_feature_reuse"
    LINEAR_PROBE = "linear_probe"
    PEFT_ADAPTER = "parameter_efficient_finetuning"
    FULL_FINETUNE = "full_finetuning"
    DOMAIN_ADAPTATION = "domain_adaptation"


def adaptation_method_ladder() -> tuple[AdaptationMethod, ...]:
    """The method ladder in increasing order of invasiveness."""
    return (AdaptationMethod.FROZEN_REUSE, AdaptationMethod.LINEAR_PROBE,
            AdaptationMethod.PEFT_ADAPTER, AdaptationMethod.FULL_FINETUNE,
            AdaptationMethod.DOMAIN_ADAPTATION)


def split_dataset(features: np.ndarray, targets: np.ndarray, *,
                  train_ratio: float = 0.6, validation_ratio: float = 0.2,
                  seed: int = 0) -> dict[str, list[int]]:
    """Deterministic shuffled split into train/validation/held-out test (Step 14)."""
    rng = np.random.default_rng(seed)
    n = len(features)
    order = rng.permutation(n)
    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)
    return {"train": [int(i) for i in order[:train_end]],
            "validation": [int(i) for i in order[train_end:validation_end]],
            "test": [int(i) for i in order[validation_end:]]}


@dataclass(slots=True)
class AdaptationDataset:
    """A reproducible adaptation dataset with split lineage (Step 14).

    Dataset identity, licenses, collection period, filtering rules, labels, splits
    and provenance are recorded; evaluation leakage is prevented structurally: the
    held-out test indices never enter training, and a near-duplicate audit flags
    rows that leak across splits.
    """

    dataset_id: str
    license: str
    collection_period: str
    features: np.ndarray
    targets: np.ndarray
    splits: dict[str, list[int]]
    filtering_rules: tuple[str, ...] = ()
    access_policy: str = "restricted"
    retention_policy: str = "project-scoped"
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate shapes and split disjointness."""
        self.features = np.asarray(self.features, dtype=float)
        self.targets = np.asarray(self.targets, dtype=float)
        if self.features.ndim != 2 or self.features.shape[0] != self.targets.shape[0]:
            raise ValueError("features must be (N, d) and aligned with targets")
        if not self.dataset_id or not self.license or not self.collection_period:
            raise ValueError("dataset identity, license and period are mandatory")
        train = set(self.splits.get("train", []))
        test = set(self.splits.get("test", []))
        if train & test:
            raise AdaptationPolicyError(
                "evaluation leakage: train and held-out indices overlap")

    def leakage_audit(self, round_digits: int = 6) -> dict[str, Any]:
        """Audit near-duplicates across splits (acceptance: held-out without leakage)."""
        fingerprint: dict[tuple, str] = {}
        duplicates: list[dict[str, Any]] = []
        for split_name, indices in self.splits.items():
            for index in indices:
                key = tuple(np.round(self.features[index], round_digits).tolist())
                if key in fingerprint and fingerprint[key] != split_name:
                    duplicates.append({"row": index, "splits": [fingerprint[key],
                                                                split_name]})
                fingerprint.setdefault(key, split_name)
        overlap = sorted(set(self.splits.get("train", []))
                         & set(self.splits.get("test", [])))
        return {"status": "leak-free" if not duplicates and not overlap else "leaky",
                "cross_split_duplicates": duplicates,
                "train_test_overlap": overlap,
                "note": ("held-out data is never trained on; near-duplicate rows are "
                         "flagged before any comparison")}

    def describe(self) -> dict[str, Any]:
        """Provenance view of the dataset."""
        return {"dataset_id": self.dataset_id, "license": self.license,
                "collection_period": self.collection_period,
                "filtering_rules": list(self.filtering_rules),
                "splits": {name: len(indices) for name, indices in self.splits.items()},
                "access_policy": self.access_policy,
                "retention_policy": self.retention_policy}


def frozen_feature_quality(model: Any, dataset: AdaptationDataset,
                           metric: Callable[[np.ndarray, np.ndarray], float] | None = None
                           ) -> float:
    """Step 12 -- quality of the pretrained representation without any training."""
    scorer = metric or (lambda prediction, target: float(-mse(prediction, target)))
    test_indices = dataset.splits["test"]
    prediction = np.asarray(model.predict(dataset.features[test_indices]))
    target = dataset.targets[test_indices]
    return float(scorer(prediction, target))


def select_adaptation_method(frozen_quality: float, target: float,
                             adapted_qualities: Mapping[str, float] | None = None
                             ) -> AdaptationMethod:
    """Step 13 -- select the smallest suitable adaptation method.

    Frozen reuse wins when it already meets the target; otherwise the smallest
    method on the ladder whose measured quality meets the target is chosen; when
    nothing measured meets the target the ladder's next step after frozen reuse
    (linear probe) is requested as the cheapest experiment.
    """
    if frozen_quality >= target:
        return AdaptationMethod.FROZEN_REUSE
    ladder = adaptation_method_ladder()
    measured = adapted_qualities or {}
    for method in ladder:
        if method is AdaptationMethod.FROZEN_REUSE:
            continue
        quality = measured.get(method.value)
        if quality is not None and quality >= target:
            return method
    return AdaptationMethod.LINEAR_PROBE


@dataclass(slots=True)
class IsolatedTrainingRun:
    """Step 15 -- one isolated, fully recorded training run.

    The candidate trains only in the laboratory: random seeds, optimizer, learning
    rate, batch size, epochs, checkpoint digest and environment are recorded.  The
    candidate cannot modify USG (no such path exists) and is not active by default
    (lifecycle changes happen only through the registry's promotion gates).
    """

    model: Any
    dataset: AdaptationDataset
    method: AdaptationMethod
    seed: int
    optimizer: str = "sgd"
    learning_rate: float = 0.05
    batch_size: int = 8
    epochs: int = 60
    environment: str = "ore_learning.adaptation.lab"
    started_at: float = field(default_factory=time.time)
    _base_digest: str = field(default="", init=False, repr=False)

    def run(self) -> dict[str, Any]:
        """Train on the train split only; return the run record with digests."""
        if not self._base_digest:
            self._base_digest = self.model.digest()
        rng = np.random.default_rng(self.seed)
        train_indices = np.asarray(self.dataset.splits["train"])
        order = rng.permutation(len(train_indices))
        losses: list[float] = []
        for _ in range(self.epochs):
            for start in range(0, len(order), self.batch_size):
                batch = train_indices[order[start:start + self.batch_size]]
                losses.append(self._train_batch(batch))
        return {
            "method": self.method.value, "seed": self.seed,
            "optimizer": self.optimizer, "learning_rate": self.learning_rate,
            "batch_size": self.batch_size, "epochs": self.epochs,
            "environment": self.environment,
            "base_digest": self._base_digest,
            "candidate_digest": self.model.digest(),
            "final_loss": round(float(losses[-1]), 6) if losses else None,
            "train_split_size": len(train_indices),
            "usg_modified": False,
            "active_by_default": False,
            "note": "adaptation happens only in the lab; USG and production state "
                    "are untouched",
        }

    def _train_batch(self, batch: np.ndarray) -> float:
        """One optimizer step on a batch (method-dispatched)."""
        if self.method is AdaptationMethod.FROZEN_REUSE:
            return 0.0
        if self.method is AdaptationMethod.LINEAR_PROBE:
            # only the output layer trains; hidden/input weights stay frozen
            prediction = self.model.predict(self.dataset.features[batch])
            target = self.dataset.targets[batch].reshape(prediction.shape)
            loss = float(np.mean((prediction - target) ** 2))
            gradient = 2.0 * (prediction - target) / prediction.shape[0]
            if hasattr(self.model, "train_output_layer"):
                self.model.train_output_layer(self.dataset.features[batch], gradient,
                                              self.learning_rate)
            return loss
        prediction = self.model.predict(self.dataset.features[batch])
        target = self.dataset.targets[batch].reshape(prediction.shape)
        return self.model._full_finetune_step(self.dataset.features[batch], target,
                                              self.learning_rate)


@dataclass(slots=True)
class AdaptationComparison:
    """Step 16 -- adapted vs base on identical held-out tasks."""

    base_quality: float
    adapted_quality: float
    base_latency_ms: float
    adapted_latency_ms: float
    base_memory_bytes: int
    adapted_memory_bytes: int
    prior_capability_regression: float
    dataset_id: str
    held_out: bool = True

    @property
    def adaptation_gain(self) -> float:
        """Delta Q = Q_adapted - Q_base on held-out data."""
        return self.adapted_quality - self.base_quality

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"base_quality": round(self.base_quality, 6),
                "adapted_quality": round(self.adapted_quality, 6),
                "adaptation_gain": round(self.adaptation_gain, 6),
                "base_latency_ms": round(self.base_latency_ms, 3),
                "adapted_latency_ms": round(self.adapted_latency_ms, 3),
                "base_memory_bytes": self.base_memory_bytes,
                "adapted_memory_bytes": self.adapted_memory_bytes,
                "prior_capability_regression": round(self.prior_capability_regression, 6),
                "dataset_id": self.dataset_id, "held_out": self.held_out}


def evaluate_promotion(comparison: AdaptationComparison,
                       config: LearningConfig | None = None,
                       approval: Any = None) -> PromotionDecision:
    """Step 16 -- promote only when predefined acceptance criteria pass.

    Automatic promotion requires the held-out gain to meet the configured minimum
    with no prior-capability regression.  A trade-off (quality drop within the
    approved bound offset by, e.g., a latency improvement) is only promotable with a
    recorded human approval.
    """
    effective = config or LearningConfig()
    gain = comparison.adaptation_gain
    regression_free = comparison.prior_capability_regression <= 1e-9
    if gain >= effective.min_adaptation_gain and regression_free:
        return PromotionDecision(
            approved=True,
            reason=(f"held-out gain {gain:.4f} meets the minimum "
                    f"{effective.min_adaptation_gain:.4f} without regression"),
            approved_tradeoff=False)
    quality_drop = -gain
    latency_improved = comparison.adapted_latency_ms < comparison.base_latency_ms
    if (quality_drop <= effective.max_approved_tradeoff_quality_drop
            and latency_improved and approval is not None
            and getattr(approval, "decision", "approved") == "approved"):
        return PromotionDecision(
            approved=True,
            reason=(f"approved trade-off: quality drop {quality_drop:.4f} within the "
                    f"bound {effective.max_approved_tradeoff_quality_drop:.4f}, "
                    "latency improved"),
            approved_tradeoff=True)
    return PromotionDecision(
        approved=False,
        reason=("gain below the configured minimum and no approved trade-off on "
                "record; the base artifact is preserved for rollback"),
        approved_tradeoff=False)


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """The outcome of one promotion evaluation."""

    approved: bool
    reason: str
    approved_tradeoff: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"approved": self.approved, "reason": self.reason,
                "approved_tradeoff": self.approved_tradeoff}
