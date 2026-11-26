"""Sampling-irregularity estimation (specification Step 14)."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from ..functional import coefficient_of_variation, gap_quantiles

__all__ = ("SamplingProfile", "profile_sampling")


@dataclass(frozen=True, slots=True)
class SamplingProfile:
    """Measured sampling-irregularity statistics for a workload."""

    n_samples: int
    mean_delta: float
    variance_delta: float
    coefficient_of_variation: float
    median_gap: float
    p95_gap: float
    maximum_gap: float
    missingness_rate: float
    large_gap_count: int
    zero_mean_handled: bool = False

    @property
    def irregular(self) -> bool:
        """Report whether sampling deviates materially from a regular grid."""
        return self.coefficient_of_variation > 0.25 or self.large_gap_count > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the sampling profile."""
        return {
            "n_samples": self.n_samples,
            "mean_delta": self.mean_delta,
            "variance_delta": self.variance_delta,
            "coefficient_of_variation": self.coefficient_of_variation,
            "median_gap": self.median_gap,
            "p95_gap": self.p95_gap,
            "maximum_gap": self.maximum_gap,
            "missingness_rate": self.missingness_rate,
            "large_gap_count": self.large_gap_count,
            "zero_mean_handled": self.zero_mean_handled,
        }


def profile_sampling(deltas: Sequence[float], *, missing_slots: int = 0,
                     total_slots: int | None = None, large_gap_threshold: float = 1e9
                     ) -> SamplingProfile:
    """Estimate sampling irregularity from elapsed times and missingness counts.

    Zero means are handled explicitly: a constant zero-delta stream yields CV = 0 and
    ``zero_mean_handled`` is set.
    """
    if not deltas:
        return SamplingProfile(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               missingness_rate=0.0, large_gap_count=0,
                               zero_mean_handled=True)
    values = [float(item) for item in deltas]
    mean = sum(values) / len(values)
    variance = (sum((item - mean) ** 2 for item in values) / max(len(values) - 1, 1)
                if len(values) > 1 else 0.0)
    cv = coefficient_of_variation(values)
    quantiles = gap_quantiles(values, (0.5, 0.95))
    denominator = total_slots if total_slots is not None else (len(values) + missing_slots)
    missingness_rate = (float(missing_slots) / float(denominator)) if denominator > 0 else 0.0
    large_gaps = sum(1 for item in values if item > large_gap_threshold)
    zero_mean = (mean == 0.0)
    return SamplingProfile(
        n_samples=len(values) + 1,
        mean_delta=mean,
        variance_delta=variance,
        coefficient_of_variation=(0.0 if (zero_mean and all(item == 0.0 for item in values))
                                  else (math.inf if zero_mean else cv)),
        median_gap=quantiles[0.5],
        p95_gap=quantiles[0.95],
        maximum_gap=max(values),
        missingness_rate=missingness_rate,
        large_gap_count=large_gaps,
        zero_mean_handled=zero_mean,
    )
