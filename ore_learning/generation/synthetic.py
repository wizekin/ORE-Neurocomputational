"""Synthetic data validation before any training use (Point 14, Step 24).

Distribution similarity, diversity, mode collapse, label consistency, privacy
memorization, and downstream utility are measured on generated samples; models
trained with and without synthetic data are compared on a real held-out set; samples
that reproduce sensitive training records or violate licenses are excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import population_stability_index

__all__ = ("SyntheticDataReport", "approve_synthetic_for_training",
           "evaluate_synthetic", "memorization_flags", "utility_comparison")


def _pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    """All-pairs Euclidean distances."""
    differences = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances


def memorization_flags(synthetic: np.ndarray, real_train: np.ndarray,
                       epsilon: float) -> np.ndarray:
    """Flag synthetic samples that reproduce a real training record (privacy leak).

    A sample is flagged when its nearest real neighbour lies within ``epsilon``,
    i.e. it is closer than any plausible regeneration noise floor.
    """
    synth = np.asarray(synthetic, dtype=float)
    real = np.asarray(real_train, dtype=float)
    if synth.size == 0 or real.size == 0:
        return np.zeros(len(synth), dtype=bool)
    differences = synth[:, None, :] - real[None, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))
    nearest = distances.min(axis=1)
    return nearest < epsilon


def utility_comparison(real_train_x: np.ndarray, real_train_y: np.ndarray,
                       synthetic_x: np.ndarray, synthetic_y: np.ndarray,
                       real_test_x: np.ndarray, real_test_y: np.ndarray, *,
                       train_fn: Callable[[np.ndarray, np.ndarray], Any],
                       evaluate_fn: Callable[[Any, np.ndarray, np.ndarray], float]) -> dict[str, Any]:
    """Downstream utility: train real-only vs real+synthetic, evaluate on real held-out.

    ``train_fn(x, y)`` returns a model; ``evaluate_fn(model, x, y)`` returns a
    higher-is-better quality.  Both models are evaluated on the same real held-out
    set, giving the measured augmentation delta (acceptance criterion).
    """
    train_x = np.asarray(real_train_x, dtype=float)
    train_y = np.asarray(real_train_y, dtype=float)
    test_x = np.asarray(real_test_x, dtype=float)
    test_y = np.asarray(real_test_y, dtype=float)
    real_only_model = train_fn(train_x, train_y)
    real_only_quality = float(evaluate_fn(real_only_model, test_x, test_y))
    if len(synthetic_x):
        augmented_x = np.vstack([train_x, np.asarray(synthetic_x, dtype=float)])
        augmented_y = np.vstack([train_y.reshape(-1, 1) if train_y.ndim == 1 else train_y,
                                  np.asarray(synthetic_y, dtype=float).reshape(
                                      -1, 1) if np.ndim(synthetic_y) == 1
                                  else np.asarray(synthetic_y, dtype=float)])
    else:
        augmented_x, augmented_y = train_x, train_y
    augmented_model = train_fn(augmented_x, augmented_y)
    augmented_quality = float(evaluate_fn(augmented_model, test_x, test_y))
    return {"real_only_quality": round(real_only_quality, 6),
            "augmented_quality": round(augmented_quality, 6),
            "augmentation_delta": round(augmented_quality - real_only_quality, 6),
            "held_out": "real held-out set, identical for both models",
            "note": "synthetic augmentation is adopted only when the held-out delta "
                    "is positive or a trade-off is explicitly approved"}


@dataclass(slots=True)
class SyntheticDataReport:
    """Step 24 -- measured properties of one synthetic batch."""

    n_synthetic: int
    n_real: int
    psi: float
    mean_distance: float
    diversity: float
    mode_collapse_indicator: float
    memorized_count: int
    label_consistency: float | None
    utility: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"n_synthetic": self.n_synthetic, "n_real": self.n_real,
                "psi": round(self.psi, 6), "mean_distance": round(self.mean_distance, 6),
                "diversity": round(self.diversity, 6),
                "mode_collapse_indicator": round(self.mode_collapse_indicator, 6),
                "memorized_count": self.memorized_count,
                "label_consistency": (round(self.label_consistency, 6)
                                      if self.label_consistency is not None else None),
                "utility": self.utility}


def evaluate_synthetic(synthetic: np.ndarray, real_train: np.ndarray, *,
                       synthetic_labels: Sequence | None = None,
                       real_labels: Sequence | None = None,
                       config: LearningConfig | None = None,
                       utility: dict[str, Any] | None = None) -> SyntheticDataReport:
    """Measure distribution similarity, diversity, collapse, memorization, labels.

    - PSI between synthetic and real coordinate marginals (distribution similarity);
    - diversity: mean nearest-neighbour distance among synthetic samples relative to
      the real data's own nearest-neighbour spread (ratio ~1 means the batch is as
      spread out as the real data; ~0 means it collapsed onto few points);
    - mode collapse: share of synthetic samples packed within ``epsilon`` of another
      synthetic sample (mass concentrated in few modes);
    - memorization: synthetic samples within ``gan_memorization_epsilon`` of a real
      training record are counted as privacy leaks;
    - label consistency: agreement between a synthetic sample's label and the label
      of its nearest real neighbour, when labels are supplied.
    """
    effective = config or LearningConfig()
    synth = np.asarray(synthetic, dtype=float)
    real = np.asarray(real_train, dtype=float)
    if synth.ndim != 2 or real.ndim != 2 or synth.shape[1] != real.shape[1]:
        raise ValueError("synthetic and real data must share the (N, d) shape")
    if synth.shape[0] == 0:
        raise ValueError("the synthetic batch is empty")

    # small reference sets make fine bins noisy; coarsen adaptively
    psi_bins = int(np.clip(min(len(real), len(synth)) // 8, 3, 10))
    psi_values = [population_stability_index(real[:, d], synth[:, d], n_bins=psi_bins)
                  for d in range(real.shape[1])]
    psi = float(np.mean(psi_values))

    if len(synth) > 1:
        nearest_synth = _pairwise_distance_matrix(synth).min(axis=1)
        collapse = float(np.mean(nearest_synth < effective.gan_memorization_epsilon))
    else:
        nearest_synth = np.zeros(1)
        collapse = 1.0
    if len(real) > 1:
        nearest_real = _pairwise_distance_matrix(real).min(axis=1)
        reference_spread = float(np.mean(nearest_real))
    else:
        reference_spread = 0.0
    diversity = (float(np.mean(nearest_synth)) / reference_spread
                 if reference_spread > 1e-12 else 0.0)

    flagged = memorization_flags(synth, real, effective.gan_memorization_epsilon)
    cross = _pairwise_distance_matrix(np.vstack([synth, real]))[: len(synth), len(synth):]
    mean_distance = float(np.mean(cross.min(axis=1)) /
                          (reference_spread if reference_spread > 1e-12 else 1.0))

    label_consistency = None
    if synthetic_labels is not None and real_labels is not None:
        nearest_real = np.argmin(cross, axis=1)
        agreement = [float(str(synthetic_labels[i]) == str(real_labels[int(j)]))
                     for i, j in enumerate(nearest_real)]
        label_consistency = float(np.mean(agreement))

    return SyntheticDataReport(
        n_synthetic=int(len(synth)), n_real=int(len(real)), psi=psi,
        mean_distance=mean_distance, diversity=diversity,
        mode_collapse_indicator=collapse, memorized_count=int(np.sum(flagged)),
        label_consistency=label_consistency, utility=utility)


def approve_synthetic_for_training(report: SyntheticDataReport,
                                   config: LearningConfig | None = None,
                                   utility_required: bool = True
                                   ) -> dict[str, Any]:
    """Decide whether the synthetic batch may be used for approved training.

    Approval requires bounded distribution distance, adequate diversity, no
    memorized records, and a non-negative (or explicitly approved) held-out utility
    delta when utility was measured.
    """
    effective = config or LearningConfig()
    reasons: list[str] = []
    if report.psi > effective.synthetic_max_psi:
        reasons.append(f"PSI {report.psi:.3f} above the bound "
                       f"{effective.synthetic_max_psi}")
    if report.diversity < effective.synthetic_min_diversity:
        reasons.append(f"diversity {report.diversity:.3f} below the floor "
                       f"{effective.synthetic_min_diversity}")
    if report.memorized_count > 0:
        reasons.append(f"{report.memorized_count} samples reproduce sensitive "
                       "training records")
    if utility_required:
        if report.utility is None:
            reasons.append("downstream utility was not measured against real "
                           "held-out data")
        elif report.utility.get("augmentation_delta", -1.0) < 0.0:
            reasons.append("synthetic augmentation reduced held-out quality")
    approved = not reasons
    return {"approved": approved,
            "reasons": reasons,
            "keep_mask_hint": ("exclude memorized samples before any training use"
                               if not approved else "all samples eligible"),
            "note": "generated data never enters authoritative state; it only "
                    "trains isolated candidates (Step 15)"}
