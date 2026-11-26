"""Change detection with bounded statistics and calibrated false positives (Step 27)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..functional import cusum_update, rolling_mean

__all__ = ("CUSUMChangeDetector", "ChangeDetector", "ChangePoint",
           "RollingResidualDetector")


@dataclass(frozen=True, slots=True)
class ChangePoint:
    """One detected change with its score, threshold, and evidence lineage."""

    timestamp: float
    channel: str
    score: float
    threshold: float
    detector: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the change point."""
        return {
            "timestamp": self.timestamp,
            "channel": self.channel,
            "score": self.score,
            "threshold": self.threshold,
            "detector": self.detector,
        }


class ChangeDetector:
    """Interface for statistical change detectors calibrated on held-out data."""

    def reset(self) -> None:
        """Reset detector state."""
        raise NotImplementedError

    def update(self, timestamp: float, channel: str, innovation: float) -> ChangePoint | None:
        """Process one innovation and possibly emit a change point."""
        raise NotImplementedError

    def expected_false_alarm_rate(self) -> float:
        """Return the pre-registered expected false-alarm rate."""
        raise NotImplementedError


class RollingResidualDetector(ChangeDetector):
    """Detect changes from a bounded rolling-residual statistic (domain thresholds)."""

    def __init__(self, threshold: float = 2.5, window: int = 8,
                 expected_false_alarm_rate: float = 0.05) -> None:
        """Configure the residual threshold, window, and expected alarm rate."""
        if threshold <= 0.0 or window <= 0:
            raise ValueError("threshold and window must be positive")
        self.threshold = float(threshold)
        self.window = int(window)
        self._expected_rate = float(expected_false_alarm_rate)
        self._residuals: list[float] = []

    def reset(self) -> None:
        """Clear the residual history."""
        self._residuals = []

    def update(self, timestamp: float, channel: str, innovation: float) -> ChangePoint | None:
        """Emit a change point when the rolling residual exceeds the threshold."""
        self._residuals.append(float(innovation))
        del self._residuals[:- self.window * 4]
        baseline = rolling_mean(self._residuals[-self.window:], self.window)
        if not baseline:
            return None
        magnitude = abs(self._residuals[-1] - baseline[-1])
        if magnitude > self.threshold * max(np.std(self._residuals[-self.window:]), 1e-9):
            return ChangePoint(timestamp=timestamp, channel=channel,
                               score=float(magnitude), threshold=self.threshold,
                               detector="rolling_residual")
        return None

    def expected_false_alarm_rate(self) -> float:
        """Return the pre-registered expected false-alarm rate."""
        return self._expected_rate


class CUSUMChangeDetector(ChangeDetector):
    """One-sided CUSUM change detection with drift, threshold, and warmup (Step 27)."""

    def __init__(self, drift: float = 0.5, threshold: float = 5.0, warmup: int = 8,
                 expected_false_alarm_rate: float = 0.05) -> None:
        """Configure the CUSUM drift k, decision threshold h, and warmup samples."""
        if drift <= 0.0 or threshold <= 0.0 or warmup < 0:
            raise ValueError("drift and threshold must be positive; warmup non-negative")
        self.drift = float(drift)
        self.threshold = float(threshold)
        self.warmup = int(warmup)
        self._expected_rate = float(expected_false_alarm_rate)
        self._statistic = 0.0
        self._count = 0
        self._reference: float | None = None

    def reset(self) -> None:
        """Reset the statistic, count, and reference level."""
        self._statistic = 0.0
        self._count = 0
        self._reference = None

    def update(self, timestamp: float, channel: str, innovation: float) -> ChangePoint | None:
        """Accumulate the bounded CUSUM statistic and emit threshold crossings."""
        if self._reference is None:
            self._reference = float(innovation)
        self._count += 1
        deviation = float(innovation) - self._reference
        self._statistic = cusum_update(self._statistic, deviation, self.drift)
        if self._count > self.warmup and self._statistic > self.threshold:
            self._statistic = 0.0  # re-arm after a detected change
            self._reference = float(innovation)
            return ChangePoint(timestamp=timestamp, channel=channel,
                               score=float(self._statistic if self._statistic else self.threshold),
                               threshold=self.threshold, detector="cusum")
        return None

    def expected_false_alarm_rate(self) -> float:
        """Return the pre-registered expected false-alarm rate."""
        return self._expected_rate
