"""Statistical baseline estimators with conservative priors and explicit support counts.

Every estimator exposes its sample count so callers can distinguish a learned estimate
from a prior-backed fallback, and historical versions are retained for rollback.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scipy.stats import beta as beta_distribution

from .contracts import CWMConfig

MAD_TO_SIGMA = 1.4826


def _unit_interval(name: str, value: float) -> float:
    """Validate a probability-like parameter inside the unit interval."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and inside [0, 1]")
    return number


class BetaBinomialEstimator:
    """Beta-Binomial smoothed success estimator with credible intervals."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        """Configure symmetric-or-not prior pseudo-counts for success estimation."""
        if not math.isfinite(float(alpha)) or alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(float(beta)) or beta <= 0.0:
            raise ValueError("beta must be finite and positive")
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._successes = 0
        self._failures = 0

    @property
    def successes(self) -> int:
        """Return the count of observed successes."""
        return self._successes

    @property
    def failures(self) -> int:
        """Return the count of observed failures."""
        return self._failures

    def record(self, success: bool) -> None:
        """Update one Bernoulli observation."""
        if success:
            self._successes += 1
        else:
            self._failures += 1

    def sample_count(self) -> int:
        """Return the number of observations backing the posterior."""
        return self._successes + self._failures

    def posterior_alpha(self) -> float:
        """Return the posterior Beta alpha parameter."""
        return self._successes + self._alpha

    def posterior_beta(self) -> float:
        """Return the posterior Beta beta parameter."""
        return self._failures + self._beta

    def posterior_mean(self) -> float:
        """Return the smoothed success probability (s+alpha)/(s+f+alpha+beta)."""
        total = self.posterior_alpha() + self.posterior_beta()
        return self.posterior_alpha() / total

    def credible_interval(self, confidence: float = 0.90) -> tuple[float, float]:
        """Return the equal-tailed credible interval of the success probability."""
        _unit_interval("confidence", confidence)
        if confidence <= 0.0:
            return (self.posterior_mean(), self.posterior_mean())
        tail = (1.0 - confidence) / 2.0
        low = float(beta_distribution.ppf(tail, self.posterior_alpha(), self.posterior_beta()))
        high = float(
            beta_distribution.ppf(1.0 - tail, self.posterior_alpha(), self.posterior_beta()))
        return (max(0.0, low), min(1.0, high))

    def interval_width(self, confidence: float = 0.90) -> float:
        """Return the width of the credible interval as an uncertainty measure."""
        low, high = self.credible_interval(confidence)
        return high - low

    def to_dict(self) -> dict[str, Any]:
        """Return a detached estimator state for persistence and versioning."""
        return {
            "alpha": self._alpha,
            "beta": self._beta,
            "successes": self._successes,
            "failures": self._failures,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BetaBinomialEstimator:
        """Restore one estimator from a persisted state dictionary."""
        estimator = cls(float(value["alpha"]), float(value["beta"]))
        estimator._successes = int(value["successes"])
        estimator._failures = int(value["failures"])
        return estimator


class ExponentialMovingEstimator:
    """Exponential moving mean and variance for execution cost measurements.

    mean_t = rho * observed_t + (1 - rho) * mean_(t-1)
    variance_t = (1 - rho) * (variance_(t-1) + rho * (observed_t - mean_t)^2)
    """

    def __init__(self, rho: float = 0.35) -> None:
        """Configure the smoothing parameter rho in the open unit interval."""
        number = float(rho)
        if not math.isfinite(number) or not 0.0 < number <= 1.0:
            raise ValueError("rho must be finite and inside the open unit interval")
        self._rho = number
        self._mean: float | None = None
        self._variance = 0.0
        self._count = 0

    @property
    def rho(self) -> float:
        """Return the smoothing parameter."""
        return self._rho

    def update(self, value: float) -> None:
        """Fold one finite observation into the moving mean and variance."""
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError("EMA observations must be finite and non-negative")
        self._count += 1
        if self._mean is None:
            self._mean = number
            self._variance = 0.0
            return
        previous_mean = self._mean
        self._mean = self._rho * number + (1.0 - self._rho) * previous_mean
        self._variance = (1.0 - self._rho) * (
            self._variance + self._rho * (number - self._mean) ** 2)

    def mean(self) -> float | None:
        """Return the moving mean or None before any observation arrives."""
        return self._mean

    def variance(self) -> float:
        """Return the moving variance, zero before any observation arrives."""
        return self._variance

    def dispersion(self) -> float:
        """Return the standard deviation implied by the moving variance."""
        return math.sqrt(self._variance)

    def sample_count(self) -> int:
        """Return the number of folded observations."""
        return self._count

    def to_dict(self) -> dict[str, Any]:
        """Return a detached estimator state for persistence and versioning."""
        return {
            "rho": self._rho,
            "mean": self._mean,
            "variance": self._variance,
            "sample_count": self._count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExponentialMovingEstimator:
        """Restore one estimator from a persisted state dictionary."""
        estimator = cls(float(value["rho"]))
        estimator._mean = (
            None if value["mean"] is None else float(value["mean"]))
        estimator._variance = float(value["variance"])
        estimator._count = int(value["sample_count"])
        return estimator


class RobustEstimator:
    """Median and MAD-based robust statistics over a bounded observation window."""

    def __init__(self, window: int = 32) -> None:
        """Configure the bounded observation window."""
        if int(window) < 1:
            raise ValueError("window must be a positive integer")
        self._window = deque(maxlen=int(window))

    def record(self, value: float) -> None:
        """Add one finite observation to the bounded window."""
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("robust observations must be finite")
        self._window.append(number)

    def estimate(self) -> float | None:
        """Return the window median or None when no observation is available."""
        if not self._window:
            return None
        ordered = sorted(self._window)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    def dispersion(self) -> float | None:
        """Return the MAD scaled to a standard-deviation equivalent, or None."""
        median = self.estimate()
        if median is None or len(self._window) < 2:
            return None
        deviations = sorted(abs(item - median) for item in self._window)
        middle = len(deviations) // 2
        mad = (
            deviations[middle] if len(deviations) % 2
            else 0.5 * (deviations[middle - 1] + deviations[middle])
        )
        return MAD_TO_SIGMA * mad

    def sample_count(self) -> int:
        """Return the number of retained observations."""
        return len(self._window)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached estimator state for persistence and versioning."""
        return {"window": list(self._window)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RobustEstimator:
        """Restore one estimator from a persisted state dictionary."""
        estimator = cls()
        for item in value["window"]:
            estimator.record(float(item))
        return estimator


def shannon_entropy(probabilities: Sequence[float]) -> float:
    """Return H = -sum(p_i log p_i) with the zero-times-log-zero convention."""
    total = 0.0
    for probability in probabilities:
        number = float(probability)
        if number < 0.0 or not math.isfinite(number):
            raise ValueError("probabilities must be finite and non-negative")
        if number > 0.0:
            total -= number * math.log(number)
    return total


def expected_information_gain(
    prior: Sequence[float],
    posterior_distributions: Mapping[str, Sequence[float]],
    outcome_probabilities: Mapping[str, float],
) -> float:
    """Return prior entropy minus the outcome-weighted expected posterior entropy."""
    weight_sum = sum(float(value) for value in outcome_probabilities.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("outcome probabilities must sum to one")
    expected_posterior = 0.0
    for outcome, weight in outcome_probabilities.items():
        if outcome not in posterior_distributions:
            raise ValueError(f"missing posterior distribution for outcome {outcome!r}")
        expected_posterior += float(weight) * shannon_entropy(
            posterior_distributions[outcome])
    return shannon_entropy(prior) - expected_posterior


def evidence_coverage_proxy(coverage_now: float, predicted_coverage: float) -> float:
    """Return the labeled empirical proxy: expected increase in evidence coverage."""
    return max(0.0, min(1.0, float(predicted_coverage)) - min(1.0, float(coverage_now)))


@dataclass(frozen=True, slots=True)
class OperationClassBaseline:
    """Conservative per-operation-class fallback estimates used before learning begins."""

    operation_class: str
    prior_successes: float
    prior_failures: float
    expected_duration_ms: float
    expected_cpu_cost: float
    expected_memory_mb: float
    expected_goal_progress: float
    timeout_risk: float

    def __post_init__(self) -> None:
        """Validate the conservative default estimates."""
        if not self.operation_class.strip():
            raise ValueError("operation_class must be non-empty")
        for name in ("prior_successes", "prior_failures", "expected_duration_ms",
                     "expected_cpu_cost", "expected_memory_mb", "expected_goal_progress",
                     "timeout_risk"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def success_estimator(self, config: CWMConfig | None = None) -> BetaBinomialEstimator:
        """Return a Beta-Binomial estimator seeded with the class prior pseudo-counts."""
        alpha = self.prior_successes if self.prior_successes > 0.0 else (
            config.success_prior_alpha if config is not None else 1.0)
        beta = self.prior_failures if self.prior_failures > 0.0 else (
            config.success_prior_beta if config is not None else 1.0)
        return BetaBinomialEstimator(alpha, beta)


DEFAULT_OPERATION_CLASS_BASELINES: Mapping[str, OperationClassBaseline] = {
    "module_execution": OperationClassBaseline(
        "module_execution", 1.0, 1.0, 400.0, 1.0, 128.0, 0.25, 0.10),
    "verification": OperationClassBaseline(
        "verification", 2.0, 0.5, 120.0, 0.25, 32.0, 0.05, 0.02),
    "knowledge_retrieval": OperationClassBaseline(
        "knowledge_retrieval", 1.5, 1.0, 200.0, 0.5, 64.0, 0.20, 0.05),
    "composition": OperationClassBaseline(
        "composition", 1.0, 2.0, 600.0, 2.0, 256.0, 0.30, 0.20),
}


def baseline_for(operation_class: str) -> OperationClassBaseline:
    """Return the conservative baseline for one class, defaulting to module execution."""
    return DEFAULT_OPERATION_CLASS_BASELINES.get(
        operation_class, DEFAULT_OPERATION_CLASS_BASELINES["module_execution"])


__all__ = [
    "DEFAULT_OPERATION_CLASS_BASELINES",
    "MAD_TO_SIGMA",
    "BetaBinomialEstimator",
    "ExponentialMovingEstimator",
    "OperationClassBaseline",
    "RobustEstimator",
    "baseline_for",
    "evidence_coverage_proxy",
    "expected_information_gain",
    "shannon_entropy",
]
