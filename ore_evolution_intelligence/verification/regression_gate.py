"""Regression gate: deployment-blocking metric comparisons (section 7; section 9).

A candidate only deploys when it beats the incumbent/ baseline on quality without
regressing reliability, latency, cost or safety beyond documented tolerances — safety
regressions are always blocking, never averaged away (acceptance test 25: hard safety
failure makes a candidate ineligible regardless of fitness; section 8: do not hide
safety regressions inside aggregate fitness). Canary regression triggers rollback
(section 9: deployment-regression safeguard).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

#: Documented regression policy: what blocks promotion. Latency/cost tolerances are
#: deliberately generous relative to quality/reliability/safety because the benchmark's
#: quality gains are only reachable by activating more modules, which trades latency;
#: safety and reliability regressions are always blocking (zero tolerance for new
#: violations; reliability may not regress beyond -0.05).
@dataclass(frozen=True, slots=True)
class RegressionPolicy:
    """Tolerances that guard deployment (documented policy choices)."""

    min_quality_delta: float = 0.0
    max_latency_regression: float = 0.60  # relative
    max_cost_regression: float = 0.40  # relative
    min_reliability_delta: float = -0.05
    safety_zero_tolerance: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_quality_delta": self.min_quality_delta,
            "max_latency_regression": self.max_latency_regression,
            "max_cost_regression": self.max_cost_regression,
            "min_reliability_delta": self.min_reliability_delta,
            "safety_zero_tolerance": self.safety_zero_tolerance,
        }


DEFAULT_REGRESSION_POLICY = RegressionPolicy()


@dataclass(frozen=True, slots=True)
class RegressionDecision:
    """Regression-gate verdict with per-metric deltas."""

    approved: bool
    reasons: tuple[str, ...] = ()
    deltas: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved, "reasons": list(self.reasons),
            "deltas": dict(self.deltas or {}),
        }


class RegressionGate:
    """Compare candidate metrics against baseline under the regression policy."""

    def __init__(self, policy: RegressionPolicy | None = None) -> None:
        self.policy = policy or DEFAULT_REGRESSION_POLICY

    def check(
        self,
        candidate_metrics: Any,
        baseline_metrics: Any,
    ) -> RegressionDecision:
        """Gate the candidate's deployment (section 7)."""
        deltas = {
            "quality": candidate_metrics.quality - baseline_metrics.quality,
            "reliability": candidate_metrics.reliability - baseline_metrics.reliability,
            "latency_mean_ms": candidate_metrics.latency_mean_ms - baseline_metrics.latency_mean_ms,
            "cost": candidate_metrics.cost - baseline_metrics.cost,
            "safety_violations": float(
                candidate_metrics.safety_violations - baseline_metrics.safety_violations
            ),
        }
        reasons: list[str] = []
        if deltas["quality"] < self.policy.min_quality_delta:
            reasons.append(
                f"quality regression {deltas['quality']:+.4f} below "
                f"{self.policy.min_quality_delta:+.4f}"
            )
        if deltas["reliability"] < self.policy.min_reliability_delta:
            reasons.append(
                f"reliability regression {deltas['reliability']:+.4f} below "
                f"{self.policy.min_reliability_delta:+.4f}"
            )
        baseline_latency = max(baseline_metrics.latency_mean_ms, 1e-6)
        latency_regression = deltas["latency_mean_ms"] / baseline_latency
        if latency_regression > self.policy.max_latency_regression:
            reasons.append(
                f"latency regression {latency_regression:+.1%} exceeds "
                f"{self.policy.max_latency_regression:+.1%}"
            )
        baseline_cost = max(baseline_metrics.cost, 1e-6)
        cost_regression = deltas["cost"] / baseline_cost
        if cost_regression > self.policy.max_cost_regression:
            reasons.append(
                f"cost regression {cost_regression:+.1%} exceeds "
                f"{self.policy.max_cost_regression:+.1%}"
            )
        if self.policy.safety_zero_tolerance and deltas["safety_violations"] > 0:
            reasons.append(
                "safety regression: candidate accumulates more safety violations "
                "than the baseline (zero tolerance)"
            )
        if candidate_metrics.safety_violations > 0 and self.policy.safety_zero_tolerance:
            reasons.append("candidate has safety violations at all")
        return RegressionDecision(
            approved=not reasons, reasons=tuple(reasons), deltas=deltas,
        )


__all__ = ["DEFAULT_REGRESSION_POLICY", "RegressionDecision", "RegressionGate", "RegressionPolicy"]
