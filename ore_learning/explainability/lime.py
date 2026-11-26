"""LIME local-surrogate diagnostics (Point 14, Step 29).

Perturb an input around the target instance, weight perturbations by proximity
pi_x, and fit an interpretable surrogate g minimizing

    xi(x) = argmin_{g in G} L(f, g, pi_x) + Omega(g).

The report records the perturbation strategy and the local-fidelity measure;
surrogates can mislead when perturbations leave the data manifold or the model is
discontinuous, so the record carries that limitation explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import weighted_r_squared, weighted_ridge_fit

__all__ = ("LimeExplanation", "LimeExplainer")


def _proximity_weights(perturbations: np.ndarray, instance: np.ndarray,
                       kernel_width: float) -> np.ndarray:
    """LIME proximity kernel pi_x = exp(-d(x, z)^2 / kernel_width^2)."""
    distances = np.linalg.norm(perturbations - instance, axis=1)
    return np.exp(-(distances ** 2) / (kernel_width ** 2))


@dataclass(slots=True)
class LimeExplanation:
    """Step 29 -- a LIME diagnostic record with fidelity and perturbation method."""

    explanation_id: str
    surrogate_coefficients: np.ndarray
    intercept: float
    local_fidelity: float
    perturbation_method: str
    n_perturbations: int
    kernel_width: float
    instance_digest: str
    feature_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"explanation_id": self.explanation_id,
                "surrogate_coefficients": [round(float(value), 6)
                                           for value in self.surrogate_coefficients],
                "intercept": round(self.intercept, 6),
                "local_fidelity": round(self.local_fidelity, 6),
                "perturbation_method": self.perturbation_method,
                "n_perturbations": self.n_perturbations,
                "kernel_width": self.kernel_width,
                "instance_digest": self.instance_digest,
                "feature_names": list(self.feature_names),
                "channel": "model_diagnostics",
                "note": ("local surrogate only; fidelity is measured on the "
                         "perturbation neighbourhood, and perturbations off the data "
                         "manifold can mislead")}


class LimeExplainer:
    """Fit weighted local linear surrogates for scalar models (Step 29)."""

    def __init__(self, config: LearningConfig | None = None) -> None:
        """Configure sample count, kernel width, and perturbation scale."""
        self.config = config or LearningConfig()
        self.sequence = 0

    def explain(self, evaluate: Callable[[np.ndarray], float],
                instance: Sequence[float], *,
                feature_names: Sequence[str] | None = None,
                noise_scale: float = 0.3,
                seed: int = 13) -> LimeExplanation:
        """Perturb, weight by proximity, fit, and measure local fidelity (R^2)."""
        x = np.asarray(instance, dtype=float)
        if x.ndim != 1 or x.size < 2:
            raise ValueError("instance must be a 1-D vector with at least 2 features")
        rng = np.random.default_rng(seed)
        self.sequence += 1
        n_samples = self.config.lime_num_samples
        kernel_width = self.config.lime_kernel_width
        perturbations = np.vstack([x] + [
            x + rng.normal(0.0, noise_scale, size=x.shape) for _ in range(n_samples)])
        responses = np.asarray([float(evaluate(row)) for row in perturbations])
        weights = _proximity_weights(perturbations, x, kernel_width)
        design = np.hstack([perturbations, np.ones((len(perturbations), 1))])
        coefficients = weighted_ridge_fit(design, responses, weights)
        beta, intercept = coefficients[:-1], float(coefficients[-1])
        predictions = design @ coefficients
        fidelity = weighted_r_squared(responses, predictions, weights)
        return LimeExplanation(
            explanation_id=f"lime-{self.sequence:05d}",
            surrogate_coefficients=np.asarray(beta, dtype=float),
            intercept=intercept, local_fidelity=float(fidelity),
            perturbation_method=f"gaussian noise (scale={noise_scale}, seed={seed})",
            n_perturbations=int(n_samples), kernel_width=float(kernel_width),
            instance_digest=_digest_row(x),
            feature_names=tuple(feature_names or [f"x{i}" for i in range(x.size)]))


def _digest_row(row: np.ndarray) -> str:
    """A compact digest of an instance for provenance."""
    import hashlib

    return hashlib.sha256(np.asarray(row, dtype=float).round(9).tobytes()
                          ).hexdigest()[:12]
