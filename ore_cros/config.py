"""Configuration for the CROS adaptive resource-management layer (ORE Point 23).

All values are engineering starting points, not physiological or platform
constants. Every knob that can affect stability is bounded here so the
MAPE-K planner and the learned policy operate inside declared limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .resource_model import WorkloadClass


@dataclass(frozen=True)
class ObjectiveWeights:
    """Soft-objective weights for one workload class (spec section 2.4).

    Hard constraints are evaluated separately; these weights only rank
    otherwise-safe candidate actions.
    """

    latency: float
    deadline_misses: float
    cost: float
    energy: float
    unfairness: float
    instability_risk: float

    def __post_init__(self) -> None:
        """Reject negative or non-finite objective weights."""
        names = (
            "latency", "deadline_misses", "cost",
            "energy", "unfairness", "instability_risk",
        )
        for name in names:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} weight must be finite and non-negative")
        if sum(float(getattr(self, name)) for name in names) <= 0.0:
            raise ValueError("at least one objective weight must be positive")

    def to_dict(self) -> dict[str, float]:
        """Return a detached copy of the weight table."""
        return {
            "latency": self.latency, "deadline_misses": self.deadline_misses,
            "cost": self.cost, "energy": self.energy,
            "unfairness": self.unfairness, "instability_risk": self.instability_risk,
        }


DEFAULT_OBJECTIVE_WEIGHTS: Mapping[WorkloadClass, ObjectiveWeights] = {
    WorkloadClass.INTEGRITY_CRITICAL: ObjectiveWeights(
        latency=0.10, deadline_misses=0.55, cost=0.05,
        energy=0.0, unfairness=0.05, instability_risk=0.25,
    ),
    WorkloadClass.INTERACTIVE: ObjectiveWeights(
        latency=0.60, deadline_misses=0.20, cost=0.05,
        energy=0.0, unfairness=0.05, instability_risk=0.10,
    ),
    WorkloadClass.CORE_COGNITIVE: ObjectiveWeights(
        latency=0.30, deadline_misses=0.15, cost=0.20,
        energy=0.0, unfairness=0.20, instability_risk=0.15,
    ),
    WorkloadClass.BATCH: ObjectiveWeights(
        latency=0.05, deadline_misses=0.0, cost=0.60,
        energy=0.15, unfairness=0.10, instability_risk=0.10,
    ),
    WorkloadClass.EXPLORATORY: ObjectiveWeights(
        latency=0.05, deadline_misses=0.0, cost=0.45,
        energy=0.20, unfairness=0.10, instability_risk=0.20,
    ),
    WorkloadClass.MAINTENANCE: ObjectiveWeights(
        latency=0.10, deadline_misses=0.0, cost=0.50,
        energy=0.15, unfairness=0.05, instability_risk=0.20,
    ),
}


@dataclass(frozen=True)
class WindowConfig:
    """Observation windows and bounded history retention (spec Step 7)."""

    short_window_seconds: float = 1.0
    medium_window_seconds: float = 10.0
    long_window_seconds: float = 300.0
    max_samples_per_series: int = 4096

    def __post_init__(self) -> None:
        """Reject non-positive or non-monotonic windows."""
        values = (
            self.short_window_seconds, self.medium_window_seconds, self.long_window_seconds,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("window sizes must be finite and positive")
        if not (self.short_window_seconds < self.medium_window_seconds < self.long_window_seconds):
            raise ValueError("window sizes must be strictly increasing")
        if self.max_samples_per_series < 16:
            raise ValueError("max_samples_per_series must be at least 16")


@dataclass(frozen=True)
class QualityThresholds:
    """Telemetry quality gates (spec Step 8)."""

    stale_after_seconds: float = 15.0
    high_impact_confidence: float = 0.80
    minimum_action_confidence: float = 0.40

    def __post_init__(self) -> None:
        """Keep confidence gates inside the unit interval and ordering sane."""
        if not math.isfinite(self.stale_after_seconds) or self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be finite and positive")
        for name in ("high_impact_confidence", "minimum_action_confidence"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.minimum_action_confidence > self.high_impact_confidence:
            raise ValueError("minimum_action_confidence cannot exceed high_impact_confidence")


@dataclass(frozen=True)
class AdaptationLimits:
    """Stability guards for adaptive control (spec section 5 and Step 52)."""

    min_dwell_seconds: float = 5.0
    max_actions_per_interval: int = 3
    max_worker_count: int = 64
    min_worker_count: int = 1
    canary_fraction: float = 0.0
    fairness_floor: float = 0.90
    prediction_error_threshold: float = 0.50
    tail_latency_regression_threshold: float = 0.50
    oscillation_window: int = 8
    oscillation_reversal_limit: int = 3
    rollback_cooldown_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Bound adaptation knobs so adaptive behaviour stays reversible."""
        positive = (
            "min_dwell_seconds", "max_actions_per_interval", "max_worker_count",
            "min_worker_count", "prediction_error_threshold",
            "tail_latency_regression_threshold", "oscillation_window",
            "oscillation_reversal_limit", "rollback_cooldown_seconds",
        )
        if any(float(getattr(self, name)) <= 0 for name in positive):
            raise ValueError("adaptation limits must all be positive")
        if self.min_worker_count > self.max_worker_count:
            raise ValueError("min_worker_count cannot exceed max_worker_count")
        if not 0.0 <= self.canary_fraction <= 1.0:
            raise ValueError("canary_fraction must be within [0, 1]")
        if not 0.0 < self.fairness_floor <= 1.0:
            raise ValueError("fairness_floor must be within (0, 1]")
        if self.oscillation_window < 2:
            raise ValueError("oscillation_window must be at least 2")


@dataclass(frozen=True)
class CROSConfig:
    """Top-level CROS configuration assembled from bounded sub-configurations."""

    policy_version: str = "cros-baseline-1.0.0"
    windows: WindowConfig = field(default_factory=WindowConfig)
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    limits: AdaptationLimits = field(default_factory=AdaptationLimits)
    objective_weights: Mapping[WorkloadClass, ObjectiveWeights] = field(
        default_factory=lambda: dict(DEFAULT_OBJECTIVE_WEIGHTS),
    )
    telemetry_overhead_budget: float = 0.05

    def __post_init__(self) -> None:
        """Validate versions and cross-references between sub-configurations."""
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string")
        if not math.isfinite(self.telemetry_overhead_budget) or not (
            0.0 < self.telemetry_overhead_budget <= 0.5
        ):
            raise ValueError("telemetry_overhead_budget must be within (0, 0.5]")
        missing = set(WorkloadClass) - set(self.objective_weights)
        if missing:
            raise ValueError(
                f"objective weights missing for workload classes: "
                f"{sorted(item.value for item in missing)}"
            )

    def weights_for(self, workload_class: WorkloadClass) -> ObjectiveWeights:
        """Return the soft-objective weights configured for one workload class."""
        return self.objective_weights[workload_class]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible configuration snapshot."""
        return {
            "policy_version": self.policy_version,
            "windows": {
                "short_window_seconds": self.windows.short_window_seconds,
                "medium_window_seconds": self.windows.medium_window_seconds,
                "long_window_seconds": self.windows.long_window_seconds,
                "max_samples_per_series": self.windows.max_samples_per_series,
            },
            "quality": {
                "stale_after_seconds": self.quality.stale_after_seconds,
                "high_impact_confidence": self.quality.high_impact_confidence,
                "minimum_action_confidence": self.quality.minimum_action_confidence,
            },
            "limits": {
                "min_dwell_seconds": self.limits.min_dwell_seconds,
                "max_actions_per_interval": self.limits.max_actions_per_interval,
                "max_worker_count": self.limits.max_worker_count,
                "min_worker_count": self.limits.min_worker_count,
                "canary_fraction": self.limits.canary_fraction,
                "fairness_floor": self.limits.fairness_floor,
                "prediction_error_threshold": self.limits.prediction_error_threshold,
            },
            "telemetry_overhead_budget": self.telemetry_overhead_budget,
        }
