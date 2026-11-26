"""Prediction-error learning: alignment, calibration, drift detection, and versioning.

The ledger only learns from validated observations. Aborted, duplicated, corrupted,
cancelled, and policy-violating executions are labeled and excluded from estimator
updates so feedback can never reinforce a bad forecast (Step 40 of the algorithm).
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .baselines import (
    BetaBinomialEstimator,
    ExponentialMovingEstimator,
    RobustEstimator,
    baseline_for,
)
from .contracts import (
    MISSING,
    CWMConfig,
    ObservedOutcome,
    PredictionRecord,
    _utc_now,
)

PROBABILITY_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class PredictionErrorReport:
    """Report aligned prediction errors with missing measurements kept missing."""

    prediction_id: str
    operation_id: str
    execution_status: str
    excluded_from_learning: bool
    exclusion_reason: str = ""
    success_error: float | None = None
    brier_score: float | None = None
    log_loss: float | None = None
    duration_error_ms: float | None = None
    goal_progress_error: float | None = None
    inside_success_interval: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible error report."""
        return {
            "prediction_id": self.prediction_id,
            "operation_id": self.operation_id,
            "execution_status": self.execution_status,
            "excluded_from_learning": self.excluded_from_learning,
            "exclusion_reason": self.exclusion_reason,
            "success_error": self.success_error,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "duration_error_ms": self.duration_error_ms,
            "goal_progress_error": self.goal_progress_error,
            "inside_success_interval": self.inside_success_interval,
        }


@dataclass(frozen=True, slots=True)
class CalibrationBucket:
    """Summarize reliability inside one confidence bucket."""

    label: str
    predictions: int
    observed_successes: int

    @property
    def observed_frequency(self) -> float | None:
        """Return the empirical success frequency inside the bucket."""
        if self.predictions == 0:
            return None
        return self.observed_successes / self.predictions

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible bucket summary."""
        return {
            "label": self.label,
            "predictions": self.predictions,
            "observed_successes": self.observed_successes,
            "observed_frequency": self.observed_frequency,
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Summarize Brier, log loss, interval coverage, and bucketed reliability."""

    sample_count: int
    brier_score: float | None
    log_loss: float | None
    interval_coverage: float | None
    buckets: tuple[CalibrationBucket, ...] = ()
    status: str = "uncalibrated"

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible calibration report."""
        return {
            "sample_count": self.sample_count,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "interval_coverage": self.interval_coverage,
            "buckets": [item.to_dict() for item in self.buckets],
            "status": self.status,
        }


class CalibrationMonitor:
    """Track Brier, log loss, and bucketed reliability in strict chronological order."""

    def __init__(self, window: int = 64, bucket_count: int = 5) -> None:
        """Configure the bounded chronological window and confidence bucket count."""
        if int(window) < 1 or int(bucket_count) < 1:
            raise ValueError("window and bucket_count must be positive integers")
        self._records: deque[tuple[float, bool, float, float]] = deque(maxlen=int(window))
        self._bucket_count = int(bucket_count)

    def record(self, predicted_probability: float, observed_success: bool) -> None:
        """Fold one chronological probability-versus-outcome pair."""
        probability = float(predicted_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("predicted_probability must be inside [0, 1]")
        self._records.append(
            (probability, bool(observed_success), _brier(probability, observed_success),
             _log_loss(probability, observed_success)))

    def sample_count(self) -> int:
        """Return the number of chronological pairs retained."""
        return len(self._records)

    def report(self) -> CalibrationReport:
        """Return the calibration summary for the retained chronological window."""
        if not self._records:
            return CalibrationReport(0, None, None, None, (), "uncalibrated")
        brier = sum(item[2] for item in self._records) / len(self._records)
        log_loss = sum(item[3] for item in self._records) / len(self._records)
        buckets = self._bucketize()
        status = _calibration_status(brier, buckets)
        return CalibrationReport(
            len(self._records), brier, log_loss, None, buckets, status)

    def status(self) -> str:
        """Return the current calibration status label."""
        return self.report().status

    def _bucketize(self) -> tuple[CalibrationBucket, ...]:
        """Group retained pairs into equal-width confidence buckets."""
        labels = _bucket_labels(self._bucket_count)
        counts = {label: [0, 0] for label in labels}
        for probability, observed, _, _ in self._records:
            index = min(self._bucket_count - 1, int(probability * self._bucket_count))
            label = labels[index]
            counts[label][0] += 1
            if observed:
                counts[label][1] += 1
        return tuple(
            CalibrationBucket(label, values[0], values[1]) for label, values in counts.items())


def _bucket_labels(count: int) -> tuple[str, ...]:
    """Return deterministic confidence bucket labels for a bucket count."""
    return tuple(f"{index / count:.1f}-{(index + 1) / count:.1f}" for index in range(count))


def _brier(probability: float, observed: bool) -> float:
    """Return the Brier score contribution of one pair."""
    outcome = 1.0 if observed else 0.0
    return (probability - outcome) ** 2


def _log_loss(probability: float, observed: bool) -> float:
    """Return the log loss contribution with probabilities clipped away from zero and one."""
    clipped = min(1.0 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, probability))
    outcome = 1.0 if observed else 0.0
    return -(outcome * math.log(clipped) + (1.0 - outcome) * math.log(1.0 - clipped))


def _calibration_status(brier: float, buckets: Sequence[CalibrationBucket]) -> str:
    """Classify calibration from the Brier score and the populated bucket gaps."""
    populated = [item for item in buckets if item.predictions > 0]
    if len(populated) < 2:
        return "uncalibrated"
    gaps = [
        abs((item.observed_frequency or 0.0) - _bucket_center(item.label))
        for item in populated
    ]
    mean_gap = sum(gaps) / len(gaps)
    if brier <= 0.15 and mean_gap <= 0.15:
        return "calibrated"
    return "over_confident" if mean_gap > 0.15 else "under_confident"


def _bucket_center(label: str) -> float:
    """Return the midpoint of one confidence bucket label."""
    low, high = label.split("-")
    return 0.5 * (float(low) + float(high))


def evaluate_chronological(
    pairs: Sequence[tuple[float, bool]],
    *,
    holdout_fraction: float = 0.3,
    bucket_count: int = 5,
) -> CalibrationReport:
    """Evaluate calibration on the latest held-out segment of a chronological sequence."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be inside the open unit interval")
    if len(pairs) < 2:
        raise ValueError("chronological evaluation requires at least two pairs")
    ordered = list(pairs)
    holdout_start = max(1, round(len(ordered) * (1.0 - holdout_fraction)))
    holdout = ordered[holdout_start:]
    monitor = CalibrationMonitor(window=max(len(holdout), 1), bucket_count=bucket_count)
    for probability, observed in holdout:
        monitor.record(probability, observed)
    return monitor.report()


@dataclass(frozen=True, slots=True)
class DriftStatus:
    """Report whether rolling error indicates estimator drift."""

    drifted: bool
    rolling_error: float
    baseline_error: float
    confidence_multiplier: float
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible drift status."""
        return {
            "drifted": self.drifted,
            "rolling_error": self.rolling_error,
            "baseline_error": self.baseline_error,
            "confidence_multiplier": self.confidence_multiplier,
            "recommendation": self.recommendation,
        }


class DriftMonitor:
    """Detect rolling error inflation relative to a validated baseline error."""

    def __init__(self, baseline_error: float, window: int = 32,
                 drift_ratio: float = 2.0, min_samples: int = 5) -> None:
        """Configure the baseline error, rolling window, inflation ratio, and support floor."""
        if not math.isfinite(float(baseline_error)) or baseline_error < 0.0:
            raise ValueError("baseline_error must be finite and non-negative")
        if int(window) < 1 or int(min_samples) < 1:
            raise ValueError("window and min_samples must be positive integers")
        if float(drift_ratio) < 1.0:
            raise ValueError("drift_ratio must be at least one")
        self._baseline = float(baseline_error)
        self._window = int(window)
        self._drift_ratio = float(drift_ratio)
        self._min_samples = int(min_samples)
        self._errors: deque[float] = deque(maxlen=self._window)

    def baseline_error(self) -> float:
        """Return the validated baseline error threshold."""
        return self._baseline

    def record(self, error: float) -> None:
        """Fold one finite, non-negative prediction error into the rolling window."""
        number = float(error)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError("drift errors must be finite and non-negative")
        self._errors.append(number)

    def rolling_error(self) -> float:
        """Return the mean rolling error, or zero before any observation arrives."""
        if not self._errors:
            return 0.0
        return sum(self._errors) / len(self._errors)

    def status(self) -> DriftStatus:
        """Return the current drift status with a lowered confidence multiplier."""
        rolling = self.rolling_error()
        drifted = (
            len(self._errors) >= self._min_samples
            and self._baseline > 0.0
            and rolling >= self._drift_ratio * self._baseline
        )
        return DriftStatus(
            drifted=drifted,
            rolling_error=rolling,
            baseline_error=self._baseline,
            confidence_multiplier=0.5 if drifted else 1.0,
            recommendation="revert_to_validated_baseline" if drifted else "",
        )


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """Record one deployable model version with its training window and metrics."""

    version_id: str
    feature_schema: str
    parameters_digest: str
    training_window_start: str | None = None
    training_window_end: str | None = None
    brier_score: float | None = None
    interval_coverage: float | None = None
    validated: bool = False
    deployed: bool = False
    registered_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible model version record."""
        return {
            "version_id": self.version_id,
            "feature_schema": self.feature_schema,
            "parameters_digest": self.parameters_digest,
            "training_window_start": self.training_window_start,
            "training_window_end": self.training_window_end,
            "brier_score": self.brier_score,
            "interval_coverage": self.interval_coverage,
            "validated": self.validated,
            "deployed": self.deployed,
            "registered_at": self.registered_at,
        }


class ModelVersionRegistry:
    """Register, deploy, and roll back model versions with validated baselines."""

    def __init__(self) -> None:
        """Initialize an empty version registry."""
        self._versions: dict[str, ModelVersion] = {}

    def register(self, version: ModelVersion, *, replace: bool = False) -> ModelVersion:
        """Register one version and reject accidental replacement."""
        if version.version_id in self._versions and not replace:
            raise ValueError(f"model version already registered: {version.version_id}")
        self._versions[version.version_id] = version
        return version

    def deploy(self, version_id: str) -> ModelVersion:
        """Deploy exactly one version and retire every other deployment."""
        if version_id not in self._versions:
            raise KeyError(f"unknown model version: {version_id}")
        for key, version in self._versions.items():
            self._versions[key] = replace(version, deployed=key == version_id)
        return self._versions[version_id]

    def active(self) -> ModelVersion | None:
        """Return the currently deployed version, if any."""
        for version in self._versions.values():
            if version.deployed:
                return version
        return None

    def rollback_to_validated(self) -> ModelVersion | None:
        """Redeploy the most recently registered validated baseline version."""
        validated = [
            version for version in self._versions.values()
            if version.validated and not version.deployed
        ]
        if not validated:
            return None
        target = max(validated, key=lambda item: item.registered_at)
        return self.deploy(target.version_id)

    def versions(self) -> tuple[ModelVersion, ...]:
        """Return every registered version in registration order."""
        return tuple(
            self._versions[key] for key in sorted(self._versions))


@dataclass
class OperationEstimators:
    """Hold the per-key estimator bundle updated only from validated observations."""

    operation_key: str
    success: BetaBinomialEstimator
    duration: ExponentialMovingEstimator = field(default_factory=ExponentialMovingEstimator)
    cpu_cost: ExponentialMovingEstimator = field(default_factory=ExponentialMovingEstimator)
    memory_mb: ExponentialMovingEstimator = field(default_factory=ExponentialMovingEstimator)
    robust_duration: RobustEstimator = field(default_factory=RobustEstimator)
    goal_progress: ExponentialMovingEstimator = field(default_factory=ExponentialMovingEstimator)
    failure_classes: dict[str, int] = field(default_factory=dict)
    risk_counters: dict[str, int] = field(default_factory=dict)
    executed_count: int = 0

    def support(self) -> int:
        """Return the total number of validated executions behind this bundle."""
        return self.executed_count


def _default_estimators(operation_key: str, config: CWMConfig) -> OperationEstimators:
    """Build a prior-backed estimator bundle for one operation key."""
    operation_class = operation_key.split("|", 1)[0]
    baseline = baseline_for(operation_class)
    return OperationEstimators(
        operation_key=operation_key,
        success=baseline.success_estimator(config),
        duration=ExponentialMovingEstimator(config.cost_ema_rho),
        cpu_cost=ExponentialMovingEstimator(config.cost_ema_rho),
        memory_mb=ExponentialMovingEstimator(config.cost_ema_rho),
        robust_duration=RobustEstimator(config.robust_window),
        goal_progress=ExponentialMovingEstimator(config.cost_ema_rho),
    )


class ExecutionLedger:
    """Own the estimator bank, observed outcomes, and prediction alignment state."""

    def __init__(self, config: CWMConfig | None = None) -> None:
        """Configure the ledger with conservative defaults before any learning."""
        self.config = config or CWMConfig()
        self._estimators: dict[str, OperationEstimators] = {}
        self._outcomes: list[ObservedOutcome] = []
        self._predictions: dict[str, PredictionRecord] = {}
        self._error_reports: list[PredictionErrorReport] = []
        self._seen_outcome_ids: set[str] = set()
        self._feature_history: dict[str, deque[tuple[float, ...]]] = {}
        self.calibration = CalibrationMonitor(window=self.config.calibration_window)
        baseline_brier = 0.25
        self.drift = DriftMonitor(
            baseline_error=baseline_brier,
            window=self.config.calibration_window,
            drift_ratio=self.config.drift_error_ratio,
        )
        self.model_versions = ModelVersionRegistry()

    # -- estimator access ---------------------------------------------------

    def estimators(self, operation_key: str) -> OperationEstimators:
        """Return (creating if needed) the estimator bundle for one operation key."""
        if operation_key not in self._estimators:
            self._estimators[operation_key] = _default_estimators(operation_key, self.config)
        return self._estimators[operation_key]

    def stats_for(self, operation_class: str, module_id: str) -> OperationEstimators:
        """Return module-level estimators when supported, else class-level estimators.

        Sparse data must not be over-segmented: module keys are consulted only once they
        hold at least the configured minimum support, otherwise the class-level bundle
        (shared across modules) provides the estimate with its wider honest interval.
        """
        class_key = operation_class
        module_key = f"{operation_class}|{module_id}"
        module_bundle = self._estimators.get(module_key)
        if module_bundle is not None and module_bundle.support() >= self.config.min_support:
            return module_bundle
        return self.estimators(class_key)

    def update_target(self, operation_class: str, module_id: str) -> OperationEstimators:
        """Return the bundle that a validated observation for this pair updates."""
        module_key = f"{operation_class}|{module_id}"
        if module_key in self._estimators:
            return self._estimators[module_key]
        class_bundle = self._estimators.get(operation_class)
        if class_bundle is not None and class_bundle.support() >= self.config.min_support:
            return self.estimators(operation_class)
        return self.estimators(module_key)

    # -- history ------------------------------------------------------------

    def record_features(self, operation_class: str, vector: Sequence[float]) -> None:
        """Retain recent feature vectors per operation class for coverage checks."""
        window = max(self.config.calibration_window, 8)
        history = self._feature_history.setdefault(
            operation_class, deque(maxlen=window))
        history.append(tuple(float(value) for value in vector))

    def nearest_feature_distance(
        self, operation_class: str, vector: Sequence[float],
    ) -> float:
        """Return the smallest normalized distance to any retained feature vector."""
        history = self._feature_history.get(operation_class)
        if not history:
            return float("inf")
        query = tuple(float(value) for value in vector)
        span = max(1, len(query))
        distances = [
            sum(abs(query[index] - item[index]) for index in range(len(query))) / span
            for item in history if len(item) == len(query)
        ]
        if not distances:
            return float("inf")
        return min(distances)

    def outcomes(self) -> tuple[ObservedOutcome, ...]:
        """Return every observed outcome in arrival order."""
        return tuple(self._outcomes)

    def prediction(self, prediction_id: str) -> PredictionRecord | None:
        """Return one registered prediction by its stable identifier."""
        return self._predictions.get(prediction_id)

    def register_prediction(self, record: PredictionRecord) -> None:
        """Retain one prediction so later outcomes can align against it."""
        self._predictions[record.prediction_id] = record

    def error_reports(self) -> tuple[PredictionErrorReport, ...]:
        """Return every produced error report in chronological order."""
        return tuple(self._error_reports)

    # -- alignment and learning (Steps 36-42) -------------------------------

    def align(self, prediction: PredictionRecord, outcome: ObservedOutcome) -> PredictionErrorReport:
        """Align one prediction with one outcome and compute compatible error metrics.

        Duplicate outcome events are ignored; retries arrive as distinct outcomes so a
        cancelled attempt stays labeled while the final executed attempt updates the
        estimators; aborted, corrupted, duplicated, and policy-violating executions are
        labeled and excluded so contaminated feedback cannot destabilize predictors.
        """
        if outcome.outcome_id in self._seen_outcome_ids:
            return self._duplicate_report(prediction, outcome)
        self._seen_outcome_ids.add(outcome.outcome_id)
        self._outcomes.append(outcome)
        excluded, reason = _exclusion(outcome)
        observed_success = None
        if not excluded:
            observed_success = outcome.artifact_lifecycle.value == "accepted"
        success_error = None
        brier = None
        log_loss = None
        inside = None
        if observed_success is not None:
            probability = prediction.predicted.success_probability
            success_error = (1.0 if observed_success else 0.0) - probability
            brier = _brier(probability, observed_success)
            log_loss = _log_loss(probability, observed_success)
            low, high = prediction.predicted.uncertainty.success_credible_interval
            inside = low <= (1.0 if observed_success else 0.0) <= high
        duration_error = None
        observed_duration = outcome.dimensions.elapsed_ms
        if observed_duration is not MISSING:
            predicted_duration = prediction.predicted.expected_cost.expected_duration_ms
            if predicted_duration is not MISSING:
                duration_error = float(observed_duration) - float(predicted_duration)
        progress_error = None
        observed_progress = outcome.dimensions.goal_progress_delta
        if observed_progress is not MISSING:
            progress_error = float(observed_progress) - prediction.predicted.goal_progress_estimate
        report = PredictionErrorReport(
            prediction_id=prediction.prediction_id,
            operation_id=prediction.operation_id,
            execution_status=outcome.execution_status.value,
            excluded_from_learning=excluded,
            exclusion_reason=reason,
            success_error=success_error,
            brier_score=brier,
            log_loss=log_loss,
            duration_error_ms=duration_error,
            goal_progress_error=progress_error,
            inside_success_interval=inside,
        )
        self._error_reports.append(report)
        if not excluded and observed_success is not None:
            self.calibration.record(
                prediction.predicted.success_probability, observed_success)
            self.drift.record(abs(float(brier or 0.0)))
        if not excluded:
            self._update_estimators(prediction, outcome)
        return report

    def _duplicate_report(
        self, prediction: PredictionRecord, outcome: ObservedOutcome,
    ) -> PredictionErrorReport:
        """Return a labeled duplicate report without touching any estimator."""
        return PredictionErrorReport(
            prediction_id=prediction.prediction_id,
            operation_id=prediction.operation_id,
            execution_status=outcome.execution_status.value,
            excluded_from_learning=True,
            exclusion_reason="duplicate outcome event for an aligned prediction",
        )

    def _update_estimators(self, prediction: PredictionRecord, outcome: ObservedOutcome) -> None:
        """Apply bounded estimator updates only for validated executed outcomes."""
        status = outcome.execution_status.value
        if status != "executed":
            return
        operation_class = _operation_class_of(prediction, outcome)
        target = self.update_target(operation_class, outcome.module_id)
        accepted = outcome.artifact_lifecycle.value == "accepted"
        target.success.record(accepted)
        target.executed_count += 1
        dimensions = outcome.dimensions
        bounded = _bounded_measurement(dimensions.elapsed_ms, 60_000.0)
        if bounded is not None:
            target.duration.update(bounded)
            target.robust_duration.record(bounded)
        bounded_cpu = _bounded_measurement(dimensions.cpu_cost, 10_000.0)
        if bounded_cpu is not None:
            target.cpu_cost.update(bounded_cpu)
        bounded_memory = _bounded_measurement(dimensions.memory_mb, 64_000.0)
        if bounded_memory is not None:
            target.memory_mb.update(bounded_memory)
        bounded_progress = _bounded_measurement(dimensions.goal_progress_delta, 1.0)
        if bounded_progress is not None:
            target.goal_progress.update(abs(bounded_progress))
        if dimensions.failure_class:
            target.failure_classes[dimensions.failure_class] = (
                target.failure_classes.get(dimensions.failure_class, 0) + 1)
        for effect in dimensions.side_effects:
            if effect in ("conflict_created", "artifact_invalidated",
                          "timeout_exceeded", "policy_violation"):
                target.risk_counters[effect] = target.risk_counters.get(effect, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Return a detached summary of the ledger state for observability."""
        return {
            "outcomes": len(self._outcomes),
            "predictions": len(self._predictions),
            "error_reports": len(self._error_reports),
            "estimator_keys": sorted(self._estimators),
            "calibration": self.calibration.report().to_dict(),
            "drift": self.drift.status().to_dict(),
        }


def _exclusion(outcome: ObservedOutcome) -> tuple[bool, str]:
    """Decide whether one outcome is contaminated and must be excluded from learning."""
    status = outcome.execution_status.value
    if status in ("aborted", "cancelled", "corrupted", "duplicated", "policy_violating"):
        return True, f"execution status {status} is excluded from estimator updates"
    return False, ""


def _operation_class_of(prediction: PredictionRecord, outcome: ObservedOutcome) -> str:
    """Recover the operation class used at prediction time from the record notes."""
    marker = "operation_class="
    for part in prediction.notes.split(";"):
        stripped = part.strip()
        if stripped.startswith(marker):
            return stripped[len(marker):]
    return "module_execution"


def _bounded_measurement(value: Any, bound: float) -> float | None:
    """Bound one measurement before it can influence an estimator update."""
    if value is MISSING:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return None
    return min(number, bound)


__all__ = [
    "PROBABILITY_EPSILON",
    "CalibrationBucket",
    "CalibrationMonitor",
    "CalibrationReport",
    "DriftMonitor",
    "DriftStatus",
    "ExecutionLedger",
    "ModelVersion",
    "ModelVersionRegistry",
    "OperationEstimators",
    "PredictionErrorReport",
    "evaluate_chronological",
]
