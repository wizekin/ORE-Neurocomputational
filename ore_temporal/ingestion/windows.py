"""Temporal-window and event-batch construction (specification Step 16)."""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..contracts import (MissingDataPolicy, QualityFlag, TemporalObservation,
                         TemporalRequest)
from ..representations.windows import EventBatch, TemporalSample, TemporalWindow
from .missing import epistemic_status_of

__all__ = ("build_temporal_window", "build_event_batch", "mask_of")


def mask_of(observation: TemporalObservation, width: int) -> tuple[bool, ...]:
    """Return the imputation mask of one observation against the channel width."""
    if QualityFlag.IMPUTED in observation.quality_flags:
        return (True,) * width
    if QualityFlag.MISSING in observation.quality_flags:
        return (True,) * width
    return (False,) * width


def build_temporal_window(request: TemporalRequest,
                          observations: Sequence[TemporalObservation],
                          normalization_version: str = "raw",
                          ) -> TemporalWindow:
    """Step 16 -- form an ordered temporal window preserving elapsed times and boundaries."""
    if not observations:
        raise ValueError("cannot build a window from zero observations")
    channels = tuple(channel.name for channel in request.observation_schema)
    width = len(channels)
    samples: list[TemporalSample] = []
    previous_timestamp: float | None = None
    for observation in observations:
        elapsed = (None if previous_timestamp is None
                   else observation.timestamp - previous_timestamp)
        samples.append(TemporalSample(
            timestamp=observation.timestamp,
            elapsed_since_previous=elapsed,
            values=tuple(observation.values),
            units=tuple(observation.units),
            original_timestamp=observation.timestamp,
            time_basis=observation.time_basis,
            source_id=observation.source_id,
            event_id=observation.event_id,
            quality_flags=tuple(observation.quality_flags),
            epistemic_status=epistemic_status_of(observation),
            imputation_mask=mask_of(observation, width),
            provenance=dict(observation.provenance),
        ))
        previous_timestamp = observation.timestamp
    return TemporalWindow(
        task_id=request.task_id,
        source_id=request.source_id,
        channels=channels,
        samples=tuple(samples),
        interval=(samples[0].timestamp, samples[-1].timestamp),
        normalization_version=normalization_version,
        time_basis=observations[0].time_basis,
    )


def build_event_batch(request: TemporalRequest,
                      observations: Iterable[TemporalObservation]) -> EventBatch:
    """Step 16 -- form a sparse asynchronous event packet from event-flagged observations."""
    channels = tuple(channel.name for channel in request.observation_schema)
    ordered = sorted(observations, key=lambda item: item.timestamp)
    if not ordered:
        raise ValueError("cannot build an event batch from zero observations")
    return EventBatch(
        task_id=request.task_id,
        source_id=request.source_id,
        channels=channels,
        event_times=tuple(item.timestamp for item in ordered),
        event_values=tuple(tuple(item.values) for item in ordered),
        interval=(ordered[0].timestamp, ordered[-1].timestamp),
        time_basis=ordered[0].time_basis,
    )


def missing_policies_of(request: TemporalRequest) -> dict[str, MissingDataPolicy]:
    """Return the declared missing-data policy per channel."""
    return {channel.name: channel.missing_policy for channel in request.observation_schema}


def window_summary(window: TemporalWindow) -> dict[str, Any]:
    """Return a compact window summary for publication payloads."""
    return {
        "task_id": window.task_id,
        "source_id": window.source_id,
        "channels": list(window.channels),
        "n_samples": len(window.samples),
        "interval": list(window.interval),
        "normalization_version": window.normalization_version,
        "imputed_samples": sum(1 for sample in window.samples if sample.imputed),
    }
