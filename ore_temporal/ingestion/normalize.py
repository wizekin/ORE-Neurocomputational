"""Versioned normalization fitted on training statistics only (specification Step 15)."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from ..contracts import TemporalObservation

__all__ = ("VersionedNormalizer",)


@dataclass(frozen=True, slots=True)
class VersionedNormalizer:
    """Z-normalization z = (x - mu) / sigma using training/calibration statistics only.

    The normalizer is constructed from explicit training statistics; it never re-estimates
    statistics from inference windows, which guarantees no future-data leakage.  Near-zero
    sigma is guarded with an epsilon and the inverse transform is preserved.
    """

    version: str
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        """Validate training statistics."""
        if not self.means or not self.standard_deviations:
            raise ValueError("training statistics must not be empty")
        if len(self.means) != len(self.standard_deviations):
            raise ValueError("means and standard_deviations must align")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        for name, values in (("means", self.means),
                             ("standard_deviations", self.standard_deviations)):
            for value in values:
                if not math.isfinite(value):
                    raise ValueError(f"{name} must be finite")
        if any(value < 0.0 for value in self.standard_deviations):
            raise ValueError("standard_deviations must be non-negative")

    @classmethod
    def fit(cls, training_values: Sequence[Sequence[float]], version: str,
            epsilon: float = 1e-8) -> "VersionedNormalizer":
        """Fit the normalizer on training/calibration data only."""
        if not training_values:
            raise ValueError("training_values must not be empty")
        width = len(training_values[0])
        if any(len(row) != width for row in training_values):
            raise ValueError("training rows must have equal width")
        array = np.asarray(training_values, dtype=float)
        means = array.mean(axis=0)
        standard_deviations = array.std(axis=0, ddof=1) if array.shape[0] > 1 else np.zeros(width)
        return cls(
            version=version,
            means=tuple(float(item) for item in means),
            standard_deviations=tuple(float(item) for item in standard_deviations),
            epsilon=epsilon,
        )

    def transform(self, values: Sequence[float]) -> tuple[float, ...]:
        """Apply the versioned transform using stored training statistics."""
        if len(values) != len(self.means):
            raise ValueError("values must match the fitted width")
        return tuple((float(value) - mean) / max(sigma, self.epsilon)
                     for value, mean, sigma in zip(values, self.means, self.standard_deviations))

    def inverse_transform(self, values: Sequence[float]) -> tuple[float, ...]:
        """Invert the transform exactly, preserving the original scale."""
        if len(values) != len(self.means):
            raise ValueError("values must match the fitted width")
        return tuple(float(value) * max(sigma, self.epsilon) + mean
                     for value, mean, sigma in zip(values, self.means, self.standard_deviations))

    def transform_observations(self, observations: Iterable[TemporalObservation]
                               ) -> list[TemporalObservation]:
        """Return normalized copies of observations, preserving flags and provenance."""
        output: list[TemporalObservation] = []
        for observation in observations:
            output.append(TemporalObservation(
                source_id=observation.source_id,
                event_id=observation.event_id,
                timestamp=observation.timestamp,
                time_basis=observation.time_basis,
                values=self.transform(observation.values),
                units=tuple(f"zscore:{unit}" for unit in observation.units),
                quality_flags=observation.quality_flags,
                provenance={**dict(observation.provenance),
                            "normalization_version": self.version},
            ))
        return output

    def describe(self) -> dict[str, object]:
        """Report the normalization version and training statistics."""
        return {
            "version": self.version,
            "means": list(self.means),
            "standard_deviations": list(self.standard_deviations),
            "epsilon": self.epsilon,
            "fitted_on": "training/calibration statistics only",
        }
