"""Temporal windows and sparse event batches (specification Step 16).

A TemporalWindow preserves elapsed times, source boundaries, imputation masks, and
original timestamps so that no preprocessing step can silently rewrite history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..contracts import EpistemicStatus, QualityFlag, TimestampBasis

__all__ = ("EventBatch", "TemporalSample", "TemporalWindow")


@dataclass(frozen=True, slots=True)
class TemporalSample:
    """One aligned sample inside a temporal window with its elapsed time and flags."""

    timestamp: float
    elapsed_since_previous: float | None
    values: tuple[float, ...]
    units: tuple[str, ...]
    original_timestamp: float
    time_basis: TimestampBasis
    source_id: str
    event_id: str
    quality_flags: tuple[QualityFlag, ...] = (QualityFlag.GOOD,)
    epistemic_status: EpistemicStatus = EpistemicStatus.MEASURED
    imputation_mask: tuple[bool, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def imputed(self) -> bool:
        """Report whether any component of this sample was imputed."""
        return any(self.imputation_mask)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the sample while keeping imputation distinguishable from measurement."""
        return {
            "timestamp": self.timestamp,
            "elapsed_since_previous": self.elapsed_since_previous,
            "values": list(self.values),
            "units": list(self.units),
            "original_timestamp": self.original_timestamp,
            "time_basis": self.time_basis.value,
            "source_id": self.source_id,
            "event_id": self.event_id,
            "quality_flags": [flag.value for flag in self.quality_flags],
            "epistemic_status": self.epistemic_status.value,
            "imputation_mask": list(self.imputation_mask),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class TemporalWindow:
    """An ordered, aligned window of samples from one task/source boundary."""

    task_id: str
    source_id: str
    channels: tuple[str, ...]
    samples: tuple[TemporalSample, ...]
    interval: tuple[float, float]
    normalization_version: str = "raw"
    time_basis: TimestampBasis = TimestampBasis.WALL_CLOCK

    def __post_init__(self) -> None:
        """Validate window ordering and channel alignment."""
        if not self.channels:
            raise ValueError("channels must not be empty")
        if not self.samples:
            raise ValueError("samples must not be empty")
        for sample in self.samples:
            if len(sample.values) != len(self.channels):
                raise ValueError("every sample must provide one value per channel")
        timestamps = [sample.timestamp for sample in self.samples]
        if any(timestamps[index] > timestamps[index + 1] for index in range(len(timestamps) - 1)):
            raise ValueError("window samples must be ordered by timestamp")

    @property
    def timestamps(self) -> tuple[float, ...]:
        """Return the ordered timestamps of the window."""
        return tuple(sample.timestamp for sample in self.samples)

    @property
    def elapsed_times(self) -> tuple[float, ...]:
        """Return the actual elapsed times between consecutive samples."""
        return tuple(
            sample.elapsed_since_previous if sample.elapsed_since_previous is not None else 0.0
            for sample in self.samples[1:])

    @property
    def duration(self) -> float:
        """Return the total duration covered by the window."""
        return self.interval[1] - self.interval[0]

    def column(self, channel: str) -> tuple[float, ...]:
        """Return the value series of one channel."""
        index = self.channels.index(channel)
        return tuple(sample.values[index] for sample in self.samples)

    def imputation_mask(self) -> tuple[tuple[bool, ...], ...]:
        """Return the per-sample imputation masks (never hide imputation)."""
        width = len(self.channels)
        return tuple(sample.imputation_mask if sample.imputation_mask else (False,) * width
                     for sample in self.samples)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the window without discarding elapsed-time structure."""
        return {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "channels": list(self.channels),
            "samples": [sample.to_dict() for sample in self.samples],
            "interval": list(self.interval),
            "normalization_version": self.normalization_version,
            "time_basis": self.time_basis.value,
        }


@dataclass(frozen=True, slots=True)
class EventBatch:
    """A sparse asynchronous event packet carrying actual event times."""

    task_id: str
    source_id: str
    channels: tuple[str, ...]
    event_times: tuple[float, ...]
    event_values: tuple[tuple[float, ...], ...]
    interval: tuple[float, float]
    time_basis: TimestampBasis = TimestampBasis.WALL_CLOCK

    def __post_init__(self) -> None:
        """Validate event packet alignment and ordering."""
        if not self.channels:
            raise ValueError("channels must not be empty")
        if len(self.event_times) != len(self.event_values):
            raise ValueError("event_times and event_values must align")
        for values in self.event_values:
            if len(values) != len(self.channels):
                raise ValueError("every event must provide one value per channel")
        if any(self.event_times[index] > self.event_times[index + 1]
               for index in range(len(self.event_times) - 1)):
            raise ValueError("event times must be ordered")

    @property
    def n_events(self) -> int:
        """Return the number of events in the batch."""
        return len(self.event_times)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event batch."""
        return {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "channels": list(self.channels),
            "event_times": list(self.event_times),
            "event_values": [list(values) for values in self.event_values],
            "interval": list(self.interval),
            "time_basis": self.time_basis.value,
        }
