"""SHAP feature-attribution diagnostics (Point 14, Step 28).

Shapley-value attributions follow

    phi_i = sum_{S subset F\\{i}} |S|!(|F|-|S|-1)!/|F|! [f(S u {i}) - f(S)]

with masked features taken from a background distribution.  The report records the
explainer configuration and the background reference because attributions depend on
those choices and are not causal proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import exact_shapley_values

__all__ = ("ShapExplanation", "ShapExplainer", "sampling_shapley_values")


def sampling_shapley_values(evaluate: Callable[[np.ndarray], float],
                            instance: np.ndarray, background: np.ndarray,
                            *, n_samples: int = 32,
                            seed: int = 7) -> np.ndarray:
    """Sampling approximation: average marginal contributions over random subsets.

    Used when the feature count exceeds the exact-enumeration bound; the
    approximation settings are recorded in the report (acceptance criterion).
    """
    x = np.asarray(instance, dtype=float)
    bg = np.asarray(background, dtype=float)
    rng = np.random.default_rng(seed)
    n_features = x.size
    totals = np.zeros(n_features)
    counts = np.zeros(n_features)
    for _ in range(int(n_samples)):
        row = bg[rng.integers(0, len(bg))]
        permutation = rng.permutation(n_features)
        masked = row.copy()
        previous = float(evaluate(masked))
        for feature in permutation:
            masked[feature] = x[feature]
            current = float(evaluate(masked))
            totals[feature] += current - previous
            counts[feature] += 1
            previous = current
    return np.divide(totals, np.maximum(counts, 1.0))


@dataclass(slots=True)
class ShapExplanation:
    """Step 28 -- a SHAP diagnostic record with its full configuration."""

    explanation_id: str
    values: np.ndarray
    instance_digest: str
    background_ref: str
    explainer_variant: str
    background_size: int
    approximation: str
    feature_names: tuple[str, ...]
    base_value: float
    efficiency_residual: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"explanation_id": self.explanation_id,
                "values": [round(float(value), 6) for value in self.values],
                "feature_names": list(self.feature_names),
                "base_value": round(self.base_value, 6),
                "efficiency_residual": round(self.efficiency_residual, 9),
                "background_ref": self.background_ref,
                "explainer_variant": self.explainer_variant,
                "background_size": self.background_size,
                "approximation": self.approximation,
                "channel": "model_diagnostics",
                "note": "attribution depends on the background and representation; "
                        "not causal proof"}


class ShapExplainer:
    """Compute and record SHAP diagnostics for scalar models (Step 28)."""

    def __init__(self, config: LearningConfig | None = None) -> None:
        """Configure the exact/approximate threshold."""
        self.config = config or LearningConfig()
        self.sequence = 0

    def explain(self, evaluate: Callable[[np.ndarray], float], instance: Sequence[float],
                background: np.ndarray, *,
                feature_names: Sequence[str] | None = None,
                background_ref: str = "background:unlabeled") -> ShapExplanation:
        """Return the attribution record with configuration and efficiency check."""
        x = np.asarray(instance, dtype=float)
        bg = np.asarray(background, dtype=float)
        if x.ndim != 1 or bg.ndim != 2 or bg.shape[1] != x.size:
            raise ValueError("instance must be 1-D and background 2-D with matching width")
        self.sequence += 1
        if x.size <= self.config.shap_max_exact_features:
            variant = "exact-enumeration"
            values = exact_shapley_values(evaluate, x, bg)
            approximation = "none (exact Shapley weights)"
        else:
            variant = "sampling-permutation"
            values = sampling_shapley_values(
                evaluate, x, bg, n_samples=self.config.shap_sampling_background_size)
            approximation = (f"permutation sampling with "
                             f"{self.config.shap_sampling_background_size} samples")
        prediction = float(evaluate(x))
        base_value = float(np.mean([evaluate(row) for row in bg]))
        efficiency = float(np.sum(values))
        return ShapExplanation(
            explanation_id=f"shap-{self.sequence:05d}",
            values=values,
            instance_digest=_digest_row(x),
            background_ref=background_ref,
            explainer_variant=variant,
            background_size=int(len(bg)),
            approximation=approximation,
            feature_names=tuple(feature_names or [f"x{i}" for i in range(x.size)]),
            base_value=base_value,
            efficiency_residual=prediction - base_value - efficiency)


def _digest_row(row: np.ndarray) -> str:
    """A compact digest of an instance for provenance."""
    import hashlib

    return hashlib.sha256(np.asarray(row, dtype=float).round(9).tobytes()
                          ).hexdigest()[:12]
