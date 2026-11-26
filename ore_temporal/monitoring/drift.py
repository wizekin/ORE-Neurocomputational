"""Distribution-shift monitoring with pre-registered review triggers (Step 48)."""
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

__all__ = ("DriftMonitor", "DriftStatus")


class DriftStatus(str, Enum):
    """Status of the monitored stream relative to its reference window."""

    STABLE = "stable"
    REVIEW_REQUIRED = "review_required"


class DriftMonitor:
    """Track input statistics, missingness, event rates, and residual drift (Step 48).

    Drift is scored with a population-stability-index-like divergence between the
    reference window and the current window; review is triggered only at the
    pre-registered threshold, and no automatic retraining occurs.
    """

    def __init__(self, reference_mean: float, reference_std: float,
                 review_threshold: float = 0.25, window: int = 64) -> None:
        """Configure the reference statistics and initialize the rolling window."""
        if window <= 0:
            raise ValueError("window must be positive")
        if review_threshold <= 0.0:
            raise ValueError("review_threshold must be positive")
        self.reference_mean = float(reference_mean)
        self.reference_std = float(reference_std)
        self.review_threshold = float(review_threshold)
        self.window = int(window)
        self._values: list[float] = []
        self._missing: int = 0
        self._total: int = 0
        self._residual_mean: float | None = None

    def update(self, value: float | None) -> None:
        """Record one observation; None counts toward missingness."""
        self._total += 1
        if value is None or not np.isfinite(value):
            self._missing += 1
            return
        self._values.append(float(value))
        del self._values[:- self.window]

    def record_residual_mean(self, residual_mean: float) -> None:
        """Record the current mean residual for residual-drift tracking."""
        self._residual_mean = float(residual_mean)

    #: Standard-normal bin probabilities used as the reference distribution in PSI.
    _REFERENCE_BINS: tuple[float, ...] = (0.0668, 0.0919, 0.1498, 0.1915,
                                          0.1915, 0.1498, 0.0919, 0.0668)

    def psi(self) -> float:
        """Compute the population-stability index against the reference statistics.

        Current values are standardized with the reference mean and standard deviation,
        then compared against the standard-normal reference bin probabilities; a shifted
        or rescaled stream therefore produces a large divergence.
        """
        if not self._values:
            return 0.0
        bins = np.array([-np.inf, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, np.inf])
        reference_std = max(self.reference_std, 1e-9)
        standardized = (np.asarray(self._values, dtype=float)
                        - self.reference_mean) / reference_std
        observed = np.histogram(standardized, bins=bins)[0] / max(standardized.size, 1)
        expected = np.asarray(self._REFERENCE_BINS, dtype=float)
        observed = np.maximum(observed, 1e-6)
        expected = np.maximum(expected, 1e-6)
        return float(np.sum((observed - expected) * np.log(observed / expected)))

    def status(self) -> DriftStatus:
        """Return the current monitoring status."""
        return (DriftStatus.REVIEW_REQUIRED if self.psi() > self.review_threshold
                else DriftStatus.STABLE)

    def missingness_rate(self) -> float:
        """Return the missingness rate of the monitored stream."""
        return (self._missing / self._total) if self._total else 0.0

    def report(self) -> dict[str, Any]:
        """Return the monitoring report with review triggers."""
        return {
            "psi": self.psi(),
            "review_threshold": self.review_threshold,
            "status": self.status().value,
            "missingness_rate": self.missingness_rate(),
            "residual_mean": self._residual_mean,
            "n_observations": self._total,
            "automatic_retraining": False,
        }
