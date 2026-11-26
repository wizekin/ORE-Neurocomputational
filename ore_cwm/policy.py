"""Hard constraints, confidence gates, and deterministic fallback decisions.

Hard constraints are always evaluated independently of the planning objective so a
non-negotiable requirement can never be traded away as a penalty weight (Step 31).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .contracts import (
    CWMConfig,
    FallbackDecision,
    FallbackTrigger,
    InvalidPredictionSchemaError,
    PlanningBudgetExceededError,
    PredictedOutcome,
    StaleRevisionError,
    UnsupportedCandidateError,
    _utc_now,
)

ACCEPTED_CALIBRATION_STATUSES = ("calibrated", "uncalibrated")


@dataclass(frozen=True, slots=True)
class ConstraintViolation:
    """Name one violated hard constraint with its measured detail."""

    constraint: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """Return a detached JSON-compatible violation record."""
        return {"constraint": self.constraint, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ConstraintVerdict:
    """Report whether a candidate sequence satisfies every hard constraint."""

    feasible: bool
    violations: tuple[ConstraintViolation, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the violation tuple."""
        object.__setattr__(self, "violations", tuple(self.violations))

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-compatible verdict."""
        return {
            "feasible": self.feasible,
            "violations": [item.to_dict() for item in self.violations],
        }


@dataclass(frozen=True, slots=True)
class HardConstraintSet:
    """Non-negotiable execution constraints filtered before any objective scoring.

    These limits mirror what the existing ORE coordinators already enforce (resource
    budgets, deadlines, module permissions, mandatory verification, critical goal
    requirements); CWM evaluates them independently so predicted utility can never
    override them.
    """

    max_total_cost: float = float("inf")
    deadline_ms: float = float("inf")
    allowed_modules: frozenset[str] | None = None
    required_output_fields: tuple[str, ...] = ()
    mandatory_verification_below_coverage: float = 0.5
    max_retries_per_operation: int = 2

    def __post_init__(self) -> None:
        """Validate constraint configuration values."""
        for name in ("max_total_cost", "deadline_ms"):
            value = float(getattr(self, name))
            if math.isnan(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        coverage = float(self.mandatory_verification_below_coverage)
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("mandatory_verification_below_coverage must be inside [0, 1]")
        if int(self.max_retries_per_operation) < 0:
            raise ValueError("max_retries_per_operation cannot be negative")
        object.__setattr__(self, "required_output_fields", tuple(self.required_output_fields))
        object.__setattr__(self, "allowed_modules", (
            None if self.allowed_modules is None else frozenset(self.allowed_modules)))

    def constraint_names(self) -> tuple[str, ...]:
        """Return the active constraint identifiers for provenance records."""
        active = ["resource_budget", "deadline", "module_permissions", "retry_limit"]
        if self.required_output_fields:
            active.append("critical_goal_requirements")
        if self.mandatory_verification_below_coverage > 0.0:
            active.append("mandatory_verification")
        return tuple(active)

    def evaluate(
        self,
        *,
        module_ids: Sequence[str],
        total_cost: float,
        total_duration_ms: float,
        produced_fields: Sequence[str],
        includes_verification: bool,
        evidence_coverage: float,
        retries: int = 0,
        monotone_only: bool = False,
    ) -> ConstraintVerdict:
        """Evaluate every hard constraint against one candidate sequence's aggregates.

        When monotone_only is set, only constraints that appending operations can worsen
        (resource budget, deadline, permissions, retry limits) are checked, so partial
        rollout prefixes can be pruned before coverage constraints are final.
        """
        violations: list[ConstraintViolation] = []
        if total_cost > self.max_total_cost:
            violations.append(ConstraintViolation(
                "resource_budget", f"total cost {total_cost:.3f} exceeds budget {self.max_total_cost}"))
        if total_duration_ms > self.deadline_ms:
            violations.append(ConstraintViolation(
                "deadline", f"total duration {total_duration_ms:.1f}ms exceeds deadline {self.deadline_ms}ms"))
        if self.allowed_modules is not None:
            disallowed = [name for name in module_ids if name not in self.allowed_modules]
            if disallowed:
                violations.append(ConstraintViolation(
                    "module_permissions", f"modules outside the allowed set: {sorted(disallowed)}"))
        if retries > self.max_retries_per_operation:
            violations.append(ConstraintViolation(
                "retry_limit", f"retries {retries} exceed limit {self.max_retries_per_operation}"))
        if monotone_only:
            return ConstraintVerdict(not violations, tuple(violations))
        if self.required_output_fields:
            missing = [
                name for name in self.required_output_fields
                if name not in set(produced_fields)
            ]
            if missing:
                violations.append(ConstraintViolation(
                    "critical_goal_requirements", f"sequence does not produce {missing}"))
        if (
            self.mandatory_verification_below_coverage > 0.0
            and evidence_coverage < self.mandatory_verification_below_coverage
            and not includes_verification
        ):
            violations.append(ConstraintViolation(
                "mandatory_verification",
                f"evidence coverage {evidence_coverage:.2f} below "
                f"{self.mandatory_verification_below_coverage:.2f} without a verification step"))
        if retries > self.max_retries_per_operation:
            violations.append(ConstraintViolation(
                "retry_limit", f"retries {retries} exceed limit {self.max_retries_per_operation}"))
        return ConstraintVerdict(not violations, tuple(violations))

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-compatible constraint set."""
        return {
            "max_total_cost": self.max_total_cost,
            "deadline_ms": self.deadline_ms,
            "allowed_modules": (
                None if self.allowed_modules is None else sorted(self.allowed_modules)),
            "required_output_fields": list(self.required_output_fields),
            "mandatory_verification_below_coverage": self.mandatory_verification_below_coverage,
            "max_retries_per_operation": self.max_retries_per_operation,
        }


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """Report whether one prediction has enough support to advise execution."""

    supported: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the reason tuple."""
        object.__setattr__(self, "reasons", tuple(self.reasons))


class ConfidenceGate:
    """Apply minimum support, interval-width, and calibration thresholds per class."""

    def __init__(
        self,
        min_support: int = 5,
        max_success_interval_width: float = 0.6,
        accepted_calibration_statuses: Iterable[str] = ACCEPTED_CALIBRATION_STATUSES,
        per_operation_class: Mapping[str, tuple[int, float]] | None = None,
    ) -> None:
        """Configure the global gate and optional per-operation-class overrides."""
        if int(min_support) < 0:
            raise ValueError("min_support cannot be negative")
        number = float(max_success_interval_width)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("max_success_interval_width must be inside [0, 1]")
        self._min_support = int(min_support)
        self._max_width = number
        self._accepted = tuple(accepted_calibration_statuses)
        self._per_class = dict(per_operation_class or {})

    def thresholds(self, operation_class: str) -> tuple[int, float]:
        """Return the support and interval-width thresholds for one class."""
        return self._per_class.get(operation_class, (self._min_support, self._max_width))

    def assess(self, predicted: PredictedOutcome, operation_class: str) -> GateVerdict:
        """Assess whether one prediction clears the support and calibration thresholds."""
        min_support, max_width = self.thresholds(operation_class)
        reasons: list[str] = []
        uncertainty = predicted.uncertainty
        if uncertainty.sample_count < min_support:
            reasons.append(
                f"historical support {uncertainty.sample_count} below required {min_support}")
        interval = uncertainty.success_credible_interval
        if interval[1] - interval[0] > max_width:
            reasons.append(
                f"credible interval width {interval[1] - interval[0]:.2f} exceeds {max_width:.2f}")
        if uncertainty.calibration_status not in self._accepted:
            reasons.append(f"calibration status {uncertainty.calibration_status!r} not accepted")
        if uncertainty.out_of_distribution:
            reasons.append("candidate is out of the observed distribution; widening uncertainty")
        return GateVerdict(not reasons, tuple(reasons))


class FallbackPolicy:
    """Translate failure triggers into deterministic deferrals to existing coordinators."""

    def __init__(self, deferred_to: str = "cognitive_task_coordinator.deterministic_policy") -> None:
        """Configure the deterministic deferral target."""
        if not deferred_to.strip():
            raise ValueError("deferred_to must be non-empty")
        self._deferred_to = deferred_to

    def decide(self, trigger: FallbackTrigger, reason: str) -> FallbackDecision:
        """Return one fallback decision deferring to existing ORE control logic."""
        return FallbackDecision(
            trigger=trigger,
            reason=reason,
            deferred_to=self._deferred_to,
            timestamp=_utc_now(),
        )


def fallback_trigger_for(exception: BaseException) -> FallbackTrigger:
    """Map one raised CWM failure onto its deterministic fallback trigger."""
    if isinstance(exception, StaleRevisionError):
        return FallbackTrigger.STALE_REVISION
    if isinstance(exception, InvalidPredictionSchemaError):
        return FallbackTrigger.INVALID_SCHEMA
    if isinstance(exception, PlanningBudgetExceededError):
        return FallbackTrigger.TIMEOUT
    if isinstance(exception, UnsupportedCandidateError):
        return FallbackTrigger.MISSING_MEASUREMENT
    if isinstance(exception, TimeoutError):
        return FallbackTrigger.TIMEOUT
    return FallbackTrigger.MODEL_ERROR


def build_gate(config: CWMConfig | None = None) -> ConfidenceGate:
    """Build the confidence gate from one configuration."""
    config = config or CWMConfig()
    return ConfidenceGate(
        min_support=config.min_support,
        max_success_interval_width=config.max_success_interval_width,
    )


__all__ = [
    "ACCEPTED_CALIBRATION_STATUSES",
    "ConfidenceGate",
    "ConstraintVerdict",
    "ConstraintViolation",
    "FallbackPolicy",
    "GateVerdict",
    "HardConstraintSet",
    "build_gate",
    "fallback_trigger_for",
]
