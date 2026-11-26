"""Predictive resource models (ORE Point 23, Step 49).

Demand and completion-time predictors with calibrated uncertainty. Error
is measured separately per resource and workload class; predictions remain
advisory until their quality is demonstrated against simple baselines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Mapping, Sequence

from ..resource_model import ResourceKind, WorkloadClass
from ..telemetry.history import ewma


@dataclass(frozen=True)
class Prediction:
    """One prediction with an explicit uncertainty interval (Step 49)."""

    point: float
    interval_low: float
    interval_high: float
    uncertainty: float
    sample_count: int

    def __post_init__(self) -> None:
        """Validate interval ordering and non-negative components."""
        if not math.isfinite(self.point) or self.point < 0.0:
            raise ValueError("point must be finite and non-negative")
        if not 0.0 <= self.interval_low <= self.interval_high:
            raise ValueError("interval must satisfy 0 <= low <= high")
        if not math.isfinite(self.uncertainty) or self.uncertainty < 0.0:
            raise ValueError("uncertainty must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the prediction."""
        return {
            "point": self.point, "interval_low": self.interval_low,
            "interval_high": self.interval_high, "uncertainty": self.uncertainty,
            "sample_count": self.sample_count,
        }


@dataclass
class _SeriesState:
    """Internal per-(resource, class) prediction state."""

    values: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    intervals_covered: int = 0
    intervals_total: int = 0


class DemandPredictor:
    """EWMA-plus-trend demand prediction with class-aware errors (Step 49).

    Predicts future demand per resource and workload class from observed
    history; the prediction interval derives from residual spread so
    uncertainty is measured, not assumed.
    """

    def __init__(
        self, *,
        horizon_weight: float = 0.5,
        max_history: int = 256,
        interval_z: float = 1.96,
    ) -> None:
        """Configure smoothing, history bound and interval confidence."""
        if not 0.0 < horizon_weight <= 1.0:
            raise ValueError("horizon_weight must be within (0, 1]")
        if max_history < 8:
            raise ValueError("max_history must be at least 8")
        if not math.isfinite(interval_z) or interval_z <= 0.0:
            raise ValueError("interval_z must be finite and positive")
        self.horizon_weight = horizon_weight
        self.max_history = max_history
        self.interval_z = interval_z
        self._state: dict[tuple[ResourceKind, WorkloadClass], _SeriesState] = {}

    def _series(self, resource: ResourceKind, workload_class: WorkloadClass) -> _SeriesState:
        """Return (creating if needed) the state for one series."""
        key = (resource, workload_class)
        if key not in self._state:
            self._state[key] = _SeriesState()
        return self._state[key]

    def observe(
        self, resource: ResourceKind, workload_class: WorkloadClass,
        actual: float, *,
        previous_prediction: Prediction | None = None,
    ) -> None:
        """Record one observed actual value and score the prior prediction."""
        if not math.isfinite(actual) or actual < 0.0:
            raise ValueError("actual must be finite and non-negative")
        series = self._series(resource, workload_class)
        series.values.append(float(actual))
        if len(series.values) > self.max_history:
            series.values = series.values[-self.max_history:]
        if previous_prediction is not None:
            residual = actual - previous_prediction.point
            series.residuals.append(residual)
            if len(series.residuals) > self.max_history:
                series.residuals = series.residuals[-self.max_history:]
            series.intervals_total += 1
            if previous_prediction.interval_low <= actual <= previous_prediction.interval_high:
                series.intervals_covered += 1

    def predict(
        self, resource: ResourceKind, workload_class: WorkloadClass, *,
        horizon: float = 1.0,
    ) -> Prediction:
        """Return the demand prediction with uncertainty for one series."""
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise ValueError("horizon must be finite and positive")
        series = self._series(resource, workload_class)
        if not series.values:
            return Prediction(0.0, 0.0, 0.0, 0.0, 0)
        smoothed = ewma(series.values, alpha=self.horizon_weight)
        if len(series.values) >= 3:
            recent = series.values[-3:]
            trend = (recent[-1] - recent[0]) / 2.0
        else:
            trend = 0.0
        point = max(0.0, smoothed + trend * horizon)
        if series.residuals:
            spread = statistics.pstdev(series.residuals)
        else:
            spread = statistics.pstdev(series.values) if len(series.values) > 1 else 0.0
        margin = self.interval_z * spread
        return Prediction(
            point=point, interval_low=max(0.0, point - margin),
            interval_high=point + margin, uncertainty=spread,
            sample_count=len(series.values),
        )

    def error_report(self) -> dict[str, dict[str, float]]:
        """Return prediction error and calibration by resource/class (Step 49)."""
        report: dict[str, dict[str, float]] = {}
        for (resource, workload_class), series in sorted(
            self._state.items(), key=lambda item: (item[0][0].value, item[0][1].value),
        ):
            errors = [abs(residual) for residual in series.residuals]
            magnitudes = [
                max(abs(residual), 1e-9) / max(
                    1e-9, abs(residual) + abs(actual)
                )
                for residual, actual in zip(series.residuals, series.values)
            ]
            calibration = (
                series.intervals_covered / series.intervals_total
                if series.intervals_total else 0.0
            )
            report[f"{resource.value}/{workload_class.value}"] = {
                "mean_absolute_error": statistics.fmean(errors) if errors else 0.0,
                "smape": statistics.fmean(magnitudes) if magnitudes else 0.0,
                "interval_coverage": calibration,
                "samples": float(len(series.values)),
            }
        return report


class CompletionTimePredictor:
    """Completion-time prediction per module and class (Step 49)."""

    def __init__(self, *, max_history: int = 128) -> None:
        """Configure the bounded per-module history."""
        if max_history < 4:
            raise ValueError("max_history must be at least 4")
        self.max_history = max_history
        self._durations: dict[tuple[str, WorkloadClass], list[float]] = {}

    def observe(
        self, module_id: str, workload_class: WorkloadClass, duration_seconds: float,
    ) -> None:
        """Record one completed duration."""
        if not math.isfinite(duration_seconds) or duration_seconds < 0.0:
            raise ValueError("duration_seconds must be finite and non-negative")
        key = (module_id, workload_class)
        history = self._durations.setdefault(key, [])
        history.append(float(duration_seconds))
        if len(history) > self.max_history:
            del history[: len(history) - self.max_history]

    def predict(
        self, module_id: str, workload_class: WorkloadClass,
        *, input_size_mib: float = 0.0,
    ) -> Prediction:
        """Return the completion-time prediction for one module/class."""
        if not math.isfinite(input_size_mib) or input_size_mib < 0.0:
            raise ValueError("input_size_mib must be finite and non-negative")
        history = self._durations.get((module_id, workload_class), [])
        if not history:
            return Prediction(0.0, 0.0, 0.0, 0.0, 0)
        point = statistics.fmean(history)
        spread = statistics.pstdev(history) if len(history) > 1 else 0.0
        size_factor = 1.0 + min(2.0, input_size_mib / 1024.0) * 0.1
        adjusted = point * size_factor
        margin = 1.96 * spread
        return Prediction(
            point=adjusted, interval_low=max(0.0, adjusted - margin),
            interval_high=adjusted + margin, uncertainty=spread,
            sample_count=len(history),
        )


__all__ = ["DemandPredictor", "CompletionTimePredictor", "Prediction"]
