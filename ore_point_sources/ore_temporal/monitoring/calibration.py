"""Conformal calibration and ensemble uncertainty (specification Step 29 and Section 3.6)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..contracts import UncertaintyReport
from ..functional import ensemble_mean_variance

__all__ = ("ConformalCalibrator", "ensemble_uncertainty")


class ConformalCalibrator:
    """Split-conformal calibration fitted on held-out chronological residuals only."""

    def __init__(self, version: str, coverage_target: float = 0.9) -> None:
        """Configure the calibration version and coverage target."""
        self.version = version
        self.coverage_target = float(coverage_target)
        self._quantile: float | None = None
        self._n_observations = 0

    @property
    def fitted(self) -> bool:
        """Report whether a calibration quantile is available."""
        return self._quantile is not None

    @property
    def n_observations(self) -> int:
        """Return the number of calibration observations used."""
        return self._n_observations

    def fit(self, calibration_values: Sequence[float],
            calibration_predictions: Sequence[float]) -> float:
        """Fit the residual quantile on chronological calibration data."""
        if len(calibration_values) != len(calibration_predictions) or not calibration_values:
            raise ValueError("calibration series must align and be non-empty")
        residuals = np.abs(np.asarray(calibration_values, dtype=float)
                           - np.asarray(calibration_predictions, dtype=float))
        alpha = 1.0 - self.coverage_target
        index = int(np.ceil((1.0 - alpha) * (residuals.size + 1))) - 1
        index = min(max(index, 0), residuals.size - 1)
        ordered = np.sort(residuals)
        self._quantile = float(ordered[index])
        self._n_observations = int(residuals.size)
        return self._quantile

    def interval(self, prediction: float) -> tuple[float, float] | None:
        """Return the conformal prediction interval, or None when uncalibrated."""
        if self._quantile is None:
            return None
        return (prediction - self._quantile, prediction + self._quantile)

    def report(self, prediction: float, ensemble_predictions: Sequence[float] | None = None,
               out_of_distribution: bool = False, ood_reason: str = "") -> UncertaintyReport:
        """Build a typed uncertainty report with calibration lineage."""
        mean = None
        variance = None
        if ensemble_predictions:
            mean, variance = ensemble_mean_variance(list(ensemble_predictions))
        return UncertaintyReport(
            method="conformal+ensemble" if ensemble_predictions else "conformal",
            ensemble_mean=mean,
            ensemble_variance=variance,
            interval=self.interval(prediction),
            coverage_target=self.coverage_target,
            calibration_version=(self.version if self.fitted else "uncalibrated"),
            calibration_observations=self._n_observations,
            out_of_distribution=out_of_distribution,
            ood_reason=ood_reason)

    def describe(self) -> dict[str, Any]:
        """Describe the calibrator state."""
        return {"version": self.version, "fitted": self.fitted,
                "coverage_target": self.coverage_target,
                "quantile": self._quantile,
                "n_observations": self._n_observations}


@dataclass(frozen=True, slots=True)
class _OutOfDistributionRule:
    """A simple range rule marking inputs outside the calibration envelope."""

    lower: float
    upper: float


class OutOfDistributionMarker:
    """Mark inputs outside the range observed during calibration (Step 29)."""

    def __init__(self, calibration_values: Sequence[Sequence[float]]) -> None:
        """Fit per-channel min/max envelopes on calibration data only."""
        if not calibration_values:
            raise ValueError("calibration_values must not be empty")
        array = np.asarray(calibration_values, dtype=float)
        self._rules = tuple(_OutOfDistributionRule(float(array[:, i].min()),
                                                   float(array[:, i].max()))
                            for i in range(array.shape[1]))

    def mark(self, values: Sequence[float]) -> tuple[bool, str]:
        """Return (is_ood, reason) for one input vector."""
        for index, (value, rule) in enumerate(zip(values, self._rules)):
            if not (rule.lower <= float(value) <= rule.upper):
                return (True, f"channel {index} value {value} outside the calibration "
                              f"envelope [{rule.lower}, {rule.upper}]")
        return (False, "")


def ensemble_uncertainty(predictions: Sequence[float]) -> dict[str, float]:
    """Return the ensemble mean and sample variance (Section 3.6)."""
    mean, variance = ensemble_mean_variance(list(predictions))
    return {"ensemble_mean": mean, "ensemble_variance": variance}
