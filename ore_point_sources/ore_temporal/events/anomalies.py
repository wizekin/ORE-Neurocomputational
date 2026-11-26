"""Anomaly detection against a calibrated baseline (specification Step 28)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np

from ..functional import residual, robust_zscore

__all__ = ("AnomalyDetector", "AnomalyHypothesis")


class AnomalyHypothesis(str, Enum):
    """Cause hypotheses; uncertain causes are labelled rather than guessed."""

    SENSOR_FAULT = "sensor_fault"
    DISTRIBUTION_SHIFT = "distribution_shift"
    PROCESS_ANOMALY = "process_anomaly"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class _Calibration:
    """Robust calibration statistics fitted on held-out chronological data."""

    median: float
    mad: float
    residual_scale: float


class AnomalyDetector:
    """Detect anomalies via robust z-scores and residual magnitude against a baseline."""

    def __init__(self, z_threshold: float = 3.5) -> None:
        """Configure the robust z-score threshold."""
        if z_threshold <= 0.0:
            raise ValueError("z_threshold must be positive")
        self.z_threshold = float(z_threshold)
        self._calibration: _Calibration | None = None
        self._recent_residuals: list[float] = []

    def calibrate(self, calibration_values: Sequence[float],
                  calibration_predictions: Sequence[float]) -> None:
        """Fit robust statistics on held-out chronological calibration data only."""
        if len(calibration_values) != len(calibration_predictions) or not calibration_values:
            raise ValueError("calibration series must align and be non-empty")
        residuals = [residual(observed, predicted) for observed, predicted
                     in zip(calibration_values, calibration_predictions)]
        array = np.asarray(residuals, dtype=float)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        self._calibration = _Calibration(median=median, mad=mad,
                                         residual_scale=float(array.std()) or 1.0)
        self._recent_residuals = []

    @property
    def calibrated(self) -> bool:
        """Report whether robust calibration statistics are available."""
        return self._calibration is not None

    def score(self, observed: float, predicted: float) -> float:
        """Return the robust z-score of one residual (0 when uncalibrated)."""
        if self._calibration is None:
            return 0.0
        return abs(robust_zscore(residual(observed, predicted),
                                 self._calibration.median,
                                 max(self._calibration.mad, 1e-9)))

    def classify(self, observed: float, predicted: float,
                 history: Sequence[float]) -> AnomalyHypothesis:
        """Label the cause hypothesis for an anomalous residual when evidence permits."""
        value = residual(observed, predicted)
        self._recent_residuals.append(value)
        del self._recent_residuals[:-64]
        if len(history) >= 3 and len(set(round(item, 9) for item in history[-3:])) == 1:
            return AnomalyHypothesis.SENSOR_FAULT
        recent = np.asarray(self._recent_residuals, dtype=float)
        if recent.size >= 8:
            half = recent.size // 2
            early = float(np.mean(np.abs(recent[:half])))
            late = float(np.mean(np.abs(recent[half:])))
            if late > 2.0 * early and early > 0.0:
                return AnomalyHypothesis.DISTRIBUTION_SHIFT
        return AnomalyHypothesis.PROCESS_ANOMALY if np.isfinite(value) else (
            AnomalyHypothesis.UNCERTAIN)

    def detect(self, observed: float, predicted: float, timestamp: float,
               history: Sequence[float]) -> tuple[bool, float, AnomalyHypothesis]:
        """Return (is_anomaly, score, cause hypothesis) for one observation."""
        score = self.score(observed, predicted)
        is_anomaly = self.calibrated and score > self.z_threshold
        hypothesis = (self.classify(observed, predicted, history) if is_anomaly
                      else AnomalyHypothesis.UNCERTAIN)
        return (is_anomaly, score, hypothesis)

    def describe(self) -> dict[str, Any]:
        """Describe the detector state for diagnostics."""
        return {
            "z_threshold": self.z_threshold,
            "calibrated": self.calibrated,
            "calibration": (None if self._calibration is None else {
                "median": self._calibration.median,
                "mad": self._calibration.mad,
                "residual_scale": self._calibration.residual_scale,
            }),
        }
