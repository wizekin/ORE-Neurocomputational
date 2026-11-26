"""Held-out validation and drift envelopes (steps 10, 29; sections 7-8).

The held-out validator is the only component that opens the sealed test set
(:meth:`BenchmarkManifest.open_sealed_test_set`) — and only after search terminates
(step 29). Finalists and the immutable baseline are re-evaluated on *identical*
held-out tasks with identical seeds; improvement must be statistically *and*
practically meaningful, with no safety or reliability regression.

:class:`ValidationEnvelope` captures the validation-metric tolerance band later used
by canary rollout and live monitoring to detect drift (acceptance tests 33, 34).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..domain.evaluation_record import EvaluationRecord, t95
from ..domain.strategy import StrategyCandidate
from .evaluator import StrategyEvaluator

#: Practically meaningful quality improvement threshold (documented policy choice).
DEFAULT_PRACTICAL_THRESHOLD = 0.05

#: Reliability regression tolerance (no meaningful regression allowed).
RELIABILITY_TOLERANCE = 0.05

#: Envelope tolerances (relative) for drift detection. ``cost`` uses the per-task
#: mean (``cost_mean``) so envelopes stay comparable across different task counts.
DEFAULT_ENVELOPE_TOLERANCES: Mapping[str, float] = {
    "quality": 0.10,
    "reliability": 0.10,
    "latency_mean_ms": 0.15,
    "cost": 0.20,
}


@dataclass(frozen=True, slots=True)
class HeldoutComparison:
    """Baseline vs finalist comparison on the sealed held-out set (step 29)."""

    candidate_id: str
    baseline_id: str
    candidate_quality: float
    baseline_quality: float
    quality_delta: float
    quality_delta_ci95: float
    reliability_delta: float
    latency_delta_ms: float
    cost_delta: float
    safety_delta: int
    improved: bool
    reasons: tuple[str, ...] = ()
    n_test_tasks: int = 0
    candidate_record: EvaluationRecord | None = None
    baseline_record: EvaluationRecord | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "baseline_id": self.baseline_id,
            "candidate_quality": self.candidate_quality,
            "baseline_quality": self.baseline_quality,
            "quality_delta": self.quality_delta,
            "quality_delta_ci95": self.quality_delta_ci95,
            "reliability_delta": self.reliability_delta,
            "latency_delta_ms": self.latency_delta_ms,
            "cost_delta": self.cost_delta,
            "safety_delta": self.safety_delta,
            "improved": self.improved,
            "reasons": list(self.reasons),
            "n_test_tasks": self.n_test_tasks,
        }


class HeldoutValidator:
    """Post-search held-out comparison against the immutable baseline (step 29)."""

    def __init__(
        self,
        evaluator: StrategyEvaluator,
        *,
        practical_threshold: float = DEFAULT_PRACTICAL_THRESHOLD,
    ) -> None:
        if practical_threshold <= 0:
            raise ValueError("practical threshold must be positive")
        self.evaluator = evaluator
        self.practical_threshold = practical_threshold

    def compare(
        self,
        candidate: StrategyCandidate,
        baseline: StrategyCandidate,
        *,
        seeds: Sequence[int] | None = None,
    ) -> HeldoutComparison:
        """Evaluate finalist and baseline identically on the sealed test set."""
        if seeds is None:
            seeds = [
                self.evaluator.benchmark.seed + 100 + i
                for i in range(self.evaluator.trials_per_task)
            ]
        test_tasks = self.evaluator.benchmark.open_sealed_test_set()
        baseline_record = self.evaluator.evaluate_on_tasks(
            baseline, test_tasks, split="heldout_test", seeds=seeds,
        )
        candidate_record = self.evaluator.evaluate_on_tasks(
            candidate, test_tasks, split="heldout_test", seeds=seeds,
        )
        return self._comparison(candidate_record, baseline_record)

    def _comparison(
        self,
        candidate_record: EvaluationRecord,
        baseline_record: EvaluationRecord,
    ) -> HeldoutComparison:
        metrics_c, metrics_b = candidate_record.metrics, baseline_record.metrics
        delta = metrics_c.quality - metrics_b.quality
        n = min(metrics_c.n_tasks, metrics_b.n_tasks) or 1
        candidate_by_task = _mean_quality_by_task(candidate_record)
        baseline_by_task = _mean_quality_by_task(baseline_record)
        task_ids = [t for t in candidate_by_task if t in baseline_by_task]
        differences = [
            candidate_by_task[t] - baseline_by_task[t] for t in task_ids
        ]
        if len(differences) > 1:
            mean_diff = sum(differences) / len(differences)
            variance = sum((d - mean_diff) ** 2 for d in differences) / (len(differences) - 1)
            ci95 = t95(len(differences) - 1) * math.sqrt(variance) / math.sqrt(len(differences))
        else:
            ci95 = 0.0
        reasons: list[str] = []
        improved = True
        if delta < self.practical_threshold:
            improved = False
            reasons.append(
                f"quality delta {delta:+.4f} below practical threshold {self.practical_threshold:+.4f}"
            )
        if delta - ci95 <= 0.0 and ci95 > 0.0:
            improved = False
            reasons.append(
                f"paired CI [{delta - ci95:+.4f}, {delta + ci95:+.4f}] includes zero"
            )
        if metrics_c.reliability < metrics_b.reliability - RELIABILITY_TOLERANCE:
            improved = False
            reasons.append(
                f"reliability regression {metrics_c.reliability - metrics_b.reliability:+.4f}"
            )
        if metrics_c.safety_violations > baseline_record.metrics.safety_violations:
            improved = False
            reasons.append("safety regression on held-out tasks")
        return HeldoutComparison(
            candidate_id=candidate_record.candidate_id,
            baseline_id=baseline_record.candidate_id,
            candidate_quality=metrics_c.quality,
            baseline_quality=metrics_b.quality,
            quality_delta=delta,
            quality_delta_ci95=ci95,
            reliability_delta=metrics_c.reliability - metrics_b.reliability,
            latency_delta_ms=metrics_c.latency_mean_ms - metrics_b.latency_mean_ms,
            cost_delta=metrics_c.cost - metrics_b.cost,
            safety_delta=metrics_c.safety_violations - metrics_b.safety_violations,
            improved=improved,
            reasons=tuple(reasons),
            n_test_tasks=n,
            candidate_record=candidate_record,
            baseline_record=baseline_record,
        )


def _mean_quality_by_task(record: EvaluationRecord) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for outcome in record.outcomes:
        totals.setdefault(outcome.task_id, []).append(outcome.quality)
    return {task: sum(values) / len(values) for task, values in totals.items()}


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Result of checking live/canary metrics against the validation envelope."""

    drifted: bool
    violations: tuple[str, ...] = ()
    measured: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "drifted": self.drifted, "violations": list(self.violations),
            "measured": dict(self.measured),
        }


@dataclass(frozen=True, slots=True)
class ValidationEnvelope:
    """Tolerance band around validation metrics (acceptance tests 33-34).

    Built from the validation evaluation record; canary and live monitoring compare
    observed metrics against these bands — any metric outside its band is drift.
    """

    bands: Mapping[str, tuple[float, float]]
    source_record_digest: str = ""

    @classmethod
    def from_record(
        cls,
        record: EvaluationRecord,
        *,
        tolerances: Mapping[str, float] = DEFAULT_ENVELOPE_TOLERANCES,
    ) -> "ValidationEnvelope":
        metrics = record.metrics
        values: dict[str, float] = {
            "quality": metrics.quality,
            "reliability": metrics.reliability,
            "latency_mean_ms": metrics.latency_mean_ms,
            "cost": metrics.cost_mean,
        }
        bands: dict[str, tuple[float, float]] = {}
        for name, value in values.items():
            tolerance = tolerances.get(name, 0.15)
            bands[name] = (value * (1 - tolerance), value * (1 + tolerance))
        return cls(bands=bands, source_record_digest=record.content_digest())

    def check(self, metrics: Any) -> DriftReport:
        """Check live/canary metrics against the envelope (drift detection)."""
        observed: dict[str, float] = {
            "quality": metrics.quality,
            "reliability": metrics.reliability,
            "latency_mean_ms": metrics.latency_mean_ms,
            "cost": metrics.cost_mean,
        }
        violations: list[str] = []
        for name, value in observed.items():
            if name not in self.bands:
                continue
            low, high = self.bands[name]
            if not (low <= value <= high):
                violations.append(
                    f"{name}={value:.4g} outside envelope [{low:.4g}, {high:.4g}]"
                )
        return DriftReport(
            drifted=bool(violations), violations=tuple(violations), measured=observed,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_record_digest": self.source_record_digest,
            "bands": {name: list(band) for name, band in sorted(self.bands.items())},
        }


__all__ = [
    "DEFAULT_ENVELOPE_TOLERANCES", "DEFAULT_PRACTICAL_THRESHOLD", "DriftReport",
    "HeldoutComparison", "HeldoutValidator", "ValidationEnvelope",
]
