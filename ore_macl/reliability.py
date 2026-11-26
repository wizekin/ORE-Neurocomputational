"""Uncertainty-aware agent reliability using Beta-Binomial posteriors (spec Section 5).

Agent reliability uses p_i | data ~ Beta(alpha + successes, beta + failures). The
layer returns a posterior mean and a credible interval, and never ranks agents
precisely from tiny samples: the interval width is exposed so matching can penalize
estimates with insufficient sample support.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - exercised implicitly through credible_interval
    from scipy.stats import beta as _scipy_beta
except ImportError:  # pragma: no cover - deterministic fallback keeps the package usable
    _scipy_beta = None


DEFAULT_PRIOR_ALPHA = 1.0
DEFAULT_PRIOR_BETA = 1.0
DEFAULT_CREDIBLE_LEVEL = 0.9


def _normal_quantile(probability: float) -> float:
    """Acklam's rational approximation of the standard normal quantile."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if probability < p_low:
        q = math.sqrt(-2 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if probability <= p_high:
        q = probability - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - probability))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
          ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


@dataclass(frozen=True, slots=True)
class BetaBinomialEstimate:
    """Summarize one agent's accepted-result posterior with explicit sample support (Step 56)."""

    successes: int
    failures: int
    prior_alpha: float = DEFAULT_PRIOR_ALPHA
    prior_beta: float = DEFAULT_PRIOR_BETA

    def __post_init__(self) -> None:
        """Reject negative counts and invalid priors."""
        if self.successes < 0 or self.failures < 0:
            raise ValueError("successes and failures cannot be negative")
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")

    @property
    def samples(self) -> int:
        """Return the number of observed outcomes behind this estimate."""
        return self.successes + self.failures

    @property
    def posterior_alpha(self) -> float:
        """Return the Beta posterior alpha parameter."""
        return self.prior_alpha + self.successes

    @property
    def posterior_beta(self) -> float:
        """Return the Beta posterior beta parameter."""
        return self.prior_beta + self.failures

    @property
    def posterior_mean(self) -> float:
        """Return the posterior mean acceptance rate."""
        total = self.posterior_alpha + self.posterior_beta
        return self.posterior_alpha / total

    def credible_interval(self, level: float = DEFAULT_CREDIBLE_LEVEL) -> tuple[float, float]:
        """Return an equal-tailed credible interval for the acceptance rate."""
        if not 0.0 < level < 1.0:
            raise ValueError("credible level must be in (0, 1)")
        tail = (1.0 - level) / 2.0
        if _scipy_beta is not None:
            lower = float(_scipy_beta.ppf(tail, self.posterior_alpha, self.posterior_beta))
            upper = float(_scipy_beta.ppf(1.0 - tail, self.posterior_alpha, self.posterior_beta))
            return lower, upper
        mean = self.posterior_mean
        variance = (
            (self.posterior_alpha * self.posterior_beta)
            / ((self.posterior_alpha + self.posterior_beta)
               * (self.posterior_alpha + self.posterior_beta + 1.0))
        )
        deviation = _normal_quantile(1.0 - tail) * math.sqrt(variance)
        return max(0.0, mean - deviation), min(1.0, mean + deviation)

    @property
    def interval_width(self):
        """Return the default-level credible interval width as an uncertainty penalty."""
        lower, upper = self.credible_interval()
        return upper - lower

    def to_dict(self) -> dict[str, Any]:
        """Return a detached estimate record."""
        lower, upper = self.credible_interval()
        return {
            "successes": self.successes,
            "failures": self.failures,
            "samples": self.samples,
            "posterior_mean": self.posterior_mean,
            "credible_interval": [lower, upper],
            "prior": [self.prior_alpha, self.prior_beta],
        }


@dataclass
class ReliabilityRecord:
    """Accumulate per-agent outcome statistics used by matching and the ledger (Step 55)."""

    successes: int = 0
    failures: int = 0
    retries: int = 0
    policy_violations: int = 0
    verifier_disagreements: int = 0
    latency_ema: float = 0.0
    cost_ema: float = 0.0
    latency_samples: int = 0
    cost_samples: int = 0
    brier_score_sum: float = 0.0
    brier_samples: int = 0
    task_classes: dict[str, int] = field(default_factory=dict)

    def record_outcome(self, *, accepted: bool) -> None:
        """Record one adjudicated outcome for the Beta-Binomial posterior."""
        if accepted:
            self.successes += 1
        else:
            self.failures += 1

    def record_calibration(self, self_confidence: float, accepted: bool) -> None:
        """Accumulate a Brier score comparing self-confidence with acceptance (Step 57)."""
        if not 0.0 <= self_confidence <= 1.0:
            return
        outcome = 1.0 if accepted else 0.0
        self.brier_score_sum += (self_confidence - outcome) ** 2
        self.brier_samples += 1

    def record_resource(self, *, elapsed_seconds: float, cost: float) -> None:
        """Smooth latency and cost observations with a bounded exponential average."""
        if elapsed_seconds < 0 or cost < 0:
            return
        self.latency_samples += 1
        self.cost_samples += 1
        self.latency_ema += 0.3 * (elapsed_seconds - self.latency_ema)
        self.cost_ema += 0.3 * (cost - self.cost_ema)

    def record_task_class(self, task_class: str) -> None:
        """Track which task classes this agent has completed."""
        if task_class.strip():
            self.task_classes[task_class] = self.task_classes.get(task_class, 0) + 1

    @property
    def brier_score(self) -> float | None:
        """Return the mean Brier score, or None before any calibration samples."""
        return self.brier_score_sum / self.brier_samples if self.brier_samples else None

    def estimate(
        self,
        prior_alpha: float = DEFAULT_PRIOR_ALPHA,
        prior_beta: float = DEFAULT_PRIOR_BETA,
    ) -> BetaBinomialEstimate:
        """Return the Beta-Binomial estimate for this record."""
        return BetaBinomialEstimate(self.successes, self.failures, prior_alpha, prior_beta)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached statistics record."""
        return {
            "successes": self.successes,
            "failures": self.failures,
            "retries": self.retries,
            "policy_violations": self.policy_violations,
            "verifier_disagreements": self.verifier_disagreements,
            "latency_ema": self.latency_ema,
            "cost_ema": self.cost_ema,
            "latency_samples": self.latency_samples,
            "cost_samples": self.cost_samples,
            "brier_score": self.brier_score,
            "brier_samples": self.brier_samples,
            "task_classes": dict(self.task_classes),
        }


def uncertainty_penalty(estimate: BetaBinomialEstimate, level: float = DEFAULT_CREDIBLE_LEVEL) -> float:
    """Return the credible-interval width used to shrink scores from tiny samples."""
    lower, upper = estimate.credible_interval(level)
    return upper - lower


def score_reliability(
    estimate: BetaBinomialEstimate,
    *,
    uncertainty_weight: float = 0.5,
    level: float = DEFAULT_CREDIBLE_LEVEL,
) -> float:
    """Combine the posterior mean with an explicit interval-width penalty (Step 56)."""
    if not 0.0 <= uncertainty_weight <= 1.0:
        raise ValueError("uncertainty_weight must be in [0, 1]")
    lower, upper = estimate.credible_interval(level)
    penalty = uncertainty_weight * (upper - lower)
    return max(0.0, min(1.0, estimate.posterior_mean - penalty))


def reliability_report(
    records: Mapping[str, ReliabilityRecord],
    *,
    prior_alpha: float = DEFAULT_PRIOR_ALPHA,
    prior_beta: float = DEFAULT_PRIOR_BETA,
    level: float = DEFAULT_CREDIBLE_LEVEL,
) -> dict[str, dict[str, Any]]:
    """Summarize every tracked agent's reliability with intervals and sample support."""
    report: dict[str, dict[str, Any]] = {}
    for agent_id, record in sorted(records.items()):
        estimate = record.estimate(prior_alpha, prior_beta)
        lower, upper = estimate.credible_interval(level)
        report[agent_id] = {
            "posterior_mean": estimate.posterior_mean,
            "credible_interval": [lower, upper],
            "samples": estimate.samples,
            "sufficient_sample_support": estimate.samples >= 5,
            "latency_ema": record.latency_ema,
            "cost_ema": record.cost_ema,
            "brier_score": record.brier_score,
            "policy_violations": record.policy_violations,
            "task_classes": dict(record.task_classes),
        }
    return report
