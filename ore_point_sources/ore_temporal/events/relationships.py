"""Temporal-relationship extraction (specification Step 30).

Correlation and temporal precedence are never serialized as causal proof; every
relationship record carries an explicit non-causal note.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..functional import lagged_correlation

__all__ = ("TemporalRelationships", "analyze_relationships")


@dataclass(frozen=True, slots=True)
class TemporalRelationships:
    """Extracted ordering, duration, recurrence, co-occurrence, and lag relationships."""

    ordering: tuple[tuple[str, float, float], ...]
    duration_statistics: dict[str, float]
    recurrence_intervals: dict[str, float]
    co_occurrences: tuple[tuple[str, str, int], ...]
    lagged_correlations: dict[str, dict[int, float]]
    causal_claim: str = "none; correlation and temporal precedence are not causation"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the relationships with the explicit non-causal annotation."""
        return {
            "ordering": [list(item) for item in self.ordering],
            "duration_statistics": dict(self.duration_statistics),
            "recurrence_intervals": dict(self.recurrence_intervals),
            "co_occurrences": [list(item) for item in self.co_occurrences],
            "lagged_correlations": {key: dict(value) for key, value
                                    in self.lagged_correlations.items()},
            "causal_claim": self.causal_claim,
        }


def analyze_relationships(channels: tuple[str, ...], timestamps: np.ndarray,
                          values: np.ndarray, event_flags: np.ndarray | None = None,
                          max_lag: int = 10) -> TemporalRelationships:
    """Extract temporal relationships from one aligned window.

    ``event_flags`` is an optional (n_samples, n_channels) boolean matrix marking
    event-relevant samples per channel; when absent, threshold-crossing samples above
    two standard deviations are used as a documented fallback.
    """
    if values.ndim != 2 or values.shape[1] != len(channels):
        raise ValueError("values must align with the channel tuple")
    if timestamps.shape[0] != values.shape[0]:
        raise ValueError("timestamps must align with values")
    ordering = tuple((channel, float(timestamps[0]), float(timestamps[-1]))
                     for channel in channels)
    durations: dict[str, float] = {}
    recurrence: dict[str, float] = {}
    flags = (np.asarray(event_flags, dtype=bool) if event_flags is not None
             else np.abs(values - values.mean(axis=0)) > 2.0 * np.maximum(values.std(axis=0), 1e-9))
    for index, channel in enumerate(channels):
        series = values[:, index]
        durations[channel] = float(np.ptp(series)) if series.size else 0.0
        event_times = timestamps[flags[:, index]] if flags.shape == values.shape else (
            np.asarray([], dtype=float))
        if event_times.size >= 2:
            recurrence[channel] = float(np.mean(np.diff(event_times)))
        else:
            recurrence[channel] = 0.0
    co_occurrences: list[tuple[str, str, int]] = []
    for i, first in enumerate(channels):
        for second in channels[i + 1:]:
            co = int(np.count_nonzero(flags[:, i] & flags[:, channels.index(second)]))
            co_occurrences.append((first, second, co))
    correlations: dict[str, dict[int, float]] = {}
    for i, first in enumerate(channels):
        for j, second in enumerate(channels):
            if j <= i:
                continue
            correlations[f"{first}->{second}"] = lagged_correlation(
                values[:, i].tolist(), values[:, j].tolist(), min(max_lag, len(values) - 1))
    return TemporalRelationships(
        ordering=ordering,
        duration_statistics=durations,
        recurrence_intervals=recurrence,
        co_occurrences=tuple(co_occurrences),
        lagged_correlations=correlations,
    )
