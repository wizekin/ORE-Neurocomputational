"""Ordering, deduplication, reorder buffering, and elapsed-time computation (Steps 10-11)."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..contracts import QualityFlag, TemporalObservation
from ..functional import compute_delta_times

__all__ = ("ReorderBuffer", "OrderingReport", "apply_ordering_and_deltas",
           "deduplicate_observations")


@dataclass(slots=True)
class OrderingReport:
    """Record ordering decisions: duplicates, late arrivals, gaps, and zero intervals."""

    duplicates_removed: int = 0
    late_marked: int = 0
    beyond_reorder_window: int = 0
    simultaneous_intervals: int = 0
    negative_rejected: int = 0
    large_gaps: list[float] = field(default_factory=list)
    zero_intervals: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ordering report."""
        return {
            "duplicates_removed": self.duplicates_removed,
            "late_marked": self.late_marked,
            "beyond_reorder_window": self.beyond_reorder_window,
            "simultaneous_intervals": self.simultaneous_intervals,
            "negative_rejected": self.negative_rejected,
            "large_gaps": list(self.large_gaps),
            "zero_intervals": self.zero_intervals,
        }


class ReorderBuffer:
    """A bounded reorder buffer for late observations (Step 10).

    Observations arriving out of order are held until ordering can be restored or the
    reorder window expires; historical ordering is never silently rewritten.
    """

    def __init__(self, capacity: int, reorder_window_seconds: float,
                 latest_timestamp: float | None = None) -> None:
        """Configure buffer capacity and the maximum tolerated lateness."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if reorder_window_seconds < 0.0:
            raise ValueError("reorder_window_seconds must be non-negative")
        self.capacity = int(capacity)
        self.reorder_window_seconds = float(reorder_window_seconds)
        self._buffer: "OrderedDict[tuple[str, str], TemporalObservation]" = OrderedDict()
        self._latest_timestamp = latest_timestamp
        self._beyond_window: list[TemporalObservation] = []

    @property
    def beyond_window(self) -> tuple[TemporalObservation, ...]:
        """Return observations that arrived later than the reorder window allows."""
        return tuple(self._beyond_window)

    def accept(self, observation: TemporalObservation, current_time: float | None = None
               ) -> bool:
        """Buffer a late observation; reject it when the reorder window has passed."""
        horizon = current_time if current_time is not None else self._latest_timestamp
        if horizon is not None:
            lateness = horizon - observation.timestamp
            if lateness > self.reorder_window_seconds:
                self._beyond_window.append(observation)
                return False
        self._buffer[observation.identity] = observation
        if len(self._buffer) > self.capacity:
            self._buffer.popitem(last=False)
        if self._latest_timestamp is None or observation.timestamp > self._latest_timestamp:
            self._latest_timestamp = observation.timestamp
        return True

    def drain_ordered(self) -> list[TemporalObservation]:
        """Return and clear buffered observations in timestamp order."""
        ordered = sorted(self._buffer.values(), key=lambda item: item.timestamp)
        self._buffer.clear()
        return ordered


def deduplicate_observations(observations: Iterable[TemporalObservation]
                             ) -> tuple[list[TemporalObservation], int]:
    """Step 10 -- deduplicate by stable (source, event) identity, keeping the first record."""
    seen: set[tuple[str, str]] = set()
    unique: list[TemporalObservation] = []
    duplicates = 0
    for observation in observations:
        if observation.identity in seen:
            duplicates += 1
            continue
        seen.add(observation.identity)
        unique.append(observation)
    return (unique, duplicates)


def apply_ordering_and_deltas(observations: Iterable[TemporalObservation], *,
                              max_gap_seconds: float,
                              reorder_window_seconds: float,
                              buffer_capacity: int
                              ) -> tuple[list[TemporalObservation], OrderingReport, list[float]]:
    """Steps 10-11 -- order observations, mark late arrivals, and compute elapsed times.

    Observations are processed in arrival order: an observation whose timestamp precedes
    the newest already-accepted timestamp is late.  Late observations inside the reorder
    window are buffered and merged back in order; observations beyond the window are
    counted and excluded.  Negative intervals are rejected (never rewritten), zero
    intervals are flagged as simultaneous events, and gaps beyond the domain policy are
    recorded.
    """
    report = OrderingReport()
    unique, duplicates = deduplicate_observations(observations)
    report.duplicates_removed = duplicates
    buffer = ReorderBuffer(buffer_capacity, reorder_window_seconds)
    result: list[TemporalObservation] = []
    newest_timestamp: float | None = None
    for observation in unique:
        if result and observation.timestamp < result[-1].timestamp:
            if buffer.accept(observation, current_time=newest_timestamp):
                report.late_marked += 1
            else:
                report.beyond_reorder_window += 1
            continue
        result.append(observation)
        if newest_timestamp is None or observation.timestamp > newest_timestamp:
            newest_timestamp = observation.timestamp
    for late in buffer.drain_ordered():
        insert_position = 0
        while (insert_position < len(result)
               and result[insert_position].timestamp <= late.timestamp):
            insert_position += 1
        result.insert(insert_position, late)
    deltas: list[float] = []
    previous: TemporalObservation | None = None
    for observation in result:
        if previous is None:
            previous = observation
            continue
        try:
            (delta_pair, simultaneous) = compute_delta_times(
                (previous.timestamp, observation.timestamp))
        except ValueError:
            report.negative_rejected += 1
            previous = observation
            continue
        delta = delta_pair[0]
        if simultaneous:
            report.simultaneous_intervals += 1
            report.zero_intervals += 1
        if delta > max_gap_seconds:
            report.large_gaps.append(delta)
        deltas.append(delta)
        previous = observation
    return (result, report, deltas)


def mark_late(observation: TemporalObservation) -> TemporalObservation:
    """Return a copy of an observation carrying the LATE quality flag."""
    flags = tuple(dict.fromkeys(observation.quality_flags + (QualityFlag.LATE,)))
    return TemporalObservation(
        source_id=observation.source_id,
        event_id=observation.event_id,
        timestamp=observation.timestamp,
        time_basis=observation.time_basis,
        values=observation.values,
        units=observation.units,
        quality_flags=flags,
        provenance=observation.provenance,
    )
