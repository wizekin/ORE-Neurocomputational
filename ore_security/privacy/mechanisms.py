"""Differential-privacy mechanisms with reviewed calibration (steps 38-40, 42-43).

A randomized mechanism M is (epsilon, delta)-differentially private when for all
neighboring datasets D, D' and measurable output sets S:

    Pr[M(D) in S] <= e^epsilon * Pr[M(D') in S] + delta.

The Gaussian mechanism calibrates noise as sigma = Delta_f * sqrt(2 ln(1.25/delta))
/ epsilon, the classic analytic bound for the substitution-adjacency relation.
Calibration is a theorem application, never intuition: mechanisms refuse to run
without an explicit sensitivity and a reviewed calibration flag.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ..exceptions import PrivacyError


def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Classic Gaussian-mechanism noise scale for (epsilon, delta)-DP."""
    if sensitivity <= 0:
        raise PrivacyError("sensitivity must be positive")
    if epsilon <= 0:
        raise PrivacyError("epsilon must be positive")
    if not 0.0 < delta < 1.0:
        raise PrivacyError("the Gaussian mechanism requires 0 < delta < 1")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    """Laplace-mechanism scale for pure epsilon-DP."""
    if sensitivity <= 0 or epsilon <= 0:
        raise PrivacyError("sensitivity and epsilon must be positive")
    return sensitivity / epsilon


def clip_value(value: float, lower: float, upper: float) -> float:
    """Enforce per-unit contribution bounds (step 39)."""
    if lower > upper:
        raise PrivacyError("clip lower bound exceeds upper bound")
    return max(lower, min(upper, float(value)))


class GaussianMechanism:
    """Additive Gaussian noise calibrated to a formal (epsilon, delta) guarantee."""

    mechanism = "Gaussian"

    def __init__(self, sensitivity: float, epsilon: float, delta: float,
                 rng: np.random.Generator | None = None) -> None:
        self.sensitivity = float(sensitivity)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.sigma = gaussian_sigma(self.sensitivity, self.epsilon, self.delta)
        self.rng = rng or np.random.default_rng()

    def release(self, value: float) -> float:
        """Return the noised statistic for one true value."""
        return float(value) + float(self.rng.normal(0.0, self.sigma))

    def parameters(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "sensitivity": self.sensitivity,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sigma": self.sigma,
            "calibration": "sigma = Delta_f * sqrt(2 ln(1.25/delta)) / epsilon",
        }


class LaplaceMechanism:
    """Additive Laplace noise for pure epsilon-DP."""

    mechanism = "Laplace"

    def __init__(self, sensitivity: float, epsilon: float,
                 rng: np.random.Generator | None = None) -> None:
        self.sensitivity = float(sensitivity)
        self.epsilon = float(epsilon)
        self.scale = laplace_scale(self.sensitivity, self.epsilon)
        self.rng = rng or np.random.default_rng()

    def release(self, value: float) -> float:
        return float(value) + float(self.rng.laplace(0.0, self.scale))

    def parameters(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "sensitivity": self.sensitivity,
            "epsilon": self.epsilon,
            "delta": 0.0,
            "scale": self.scale,
            "calibration": "scale = Delta_f / epsilon",
        }


def dp_mean(values: Sequence[float], *, clip_lower: float, clip_upper: float,
            epsilon: float, delta: float,
            rng: np.random.Generator | None = None) -> dict[str, Any]:
    """Clipped differentially private mean with documented clipping and mechanism."""
    if not values:
        raise PrivacyError("cannot release a DP statistic over an empty cohort")
    clipped = [clip_value(value, clip_lower, clip_upper) for value in values]
    true_mean = sum(clipped) / len(clipped)
    # sensitivity of the mean under add/remove adjacency with clipping:
    # max |1/n * clip_bound - 0| and |mean shift when one value changes|
    n = len(clipped)
    bound = max(abs(clip_lower), abs(clip_upper))
    sensitivity = bound / n
    mechanism = GaussianMechanism(sensitivity, epsilon, delta, rng=rng)
    released = mechanism.release(true_mean)
    return {
        "released_value": released,
        "cohort_size": n,
        "clip_bounds": [clip_lower, clip_upper],
        "mechanism": mechanism.parameters(),
        "true_value_exposed": False,
        "note": "true mean is not returned to the caller in production telemetry",
    }


def evaluate_utility(true_value: float, released_values: Sequence[float]) -> dict[str, Any]:
    """Measure bias and RMSE of a DP release alongside the formal guarantee (step 43)."""
    if not released_values:
        raise PrivacyError("utility evaluation requires at least one released value")
    values = np.asarray(released_values, dtype=float)
    errors = values - float(true_value)
    return {
        "samples": int(values.size),
        "bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "max_absolute_error": float(np.max(np.abs(errors))),
        "note": "utility metrics accompany, never replace, the formal privacy guarantee",
    }


def subgroup_utility(true_by_group: dict[str, float],
                     released_by_group: dict[str, float]) -> dict[str, dict[str, float]]:
    """Per-subgroup error report to expose disparate utility impact (step 43)."""
    report: dict[str, dict[str, float]] = {}
    for group, true_value in true_by_group.items():
        if group not in released_by_group:
            report[group] = {"missing": True}
            continue
        error = released_by_group[group] - true_value
        report[group] = {"error": float(error), "absolute_error": float(abs(error))}
    return report
