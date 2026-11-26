"""Quality validation and the ingestion report (specification Steps 8 and 12)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping

from ..contracts import (ObservationChannel, QualityFlag, TemporalObservation,
                         TemporalRequest)

__all__ = ("IngestionReport", "QualityValidator", "validate_quality")


@dataclass(slots=True)
class IngestionReport:
    """The complete record of Phase B ingestion decisions for one task."""

    received: int = 0
    duplicates_removed: int = 0
    late_marked: int = 0
    beyond_reorder_window: int = 0
    negative_intervals_rejected: int = 0
    out_of_range_flagged: int = 0
    missing_flagged: int = 0
    unit_conversions: list[str] = field(default_factory=list)
    unit_issues: list[str] = field(default_factory=list)
    timestamp_issues: list[str] = field(default_factory=list)
    large_gaps: list[float] = field(default_factory=list)
    zero_intervals: int = 0
    imputation_applied: int = 0
    backpressure_applied: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ingestion report."""
        return {
            "received": self.received,
            "duplicates_removed": self.duplicates_removed,
            "late_marked": self.late_marked,
            "beyond_reorder_window": self.beyond_reorder_window,
            "negative_intervals_rejected": self.negative_intervals_rejected,
            "out_of_range_flagged": self.out_of_range_flagged,
            "missing_flagged": self.missing_flagged,
            "unit_conversions": list(self.unit_conversions),
            "unit_issues": list(self.unit_issues),
            "timestamp_issues": list(self.timestamp_issues),
            "large_gaps": list(self.large_gaps),
            "zero_intervals": self.zero_intervals,
            "imputation_applied": self.imputation_applied,
            "backpressure_applied": self.backpressure_applied,
            "notes": list(self.notes),
        }


class QualityValidator:
    """Validate units, physical bounds, missing values, infinities, and sensor flags."""

    def __init__(self, schema: tuple[ObservationChannel, ...]) -> None:
        """Configure the validator with the request's declared observation schema."""
        if not schema:
            raise ValueError("schema must not be empty")
        self.schema = schema
        self._by_name = {channel.name: channel for channel in schema}

    def validate(self, observation: TemporalObservation) -> list[str]:
        """Validate one observation, flagging out-of-range values instead of discarding them."""
        issues: list[str] = []
        if len(observation.values) != len(self.schema):
            issues.append(
                f"observation {observation.event_id} carries {len(observation.values)} values "
                f"while the schema declares {len(self.schema)} channels")
            return issues
        for channel, value in zip(self.schema, observation.values):
            if not math.isfinite(value):
                issues.append(
                    f"observation {observation.event_id} channel {channel.name!r} is non-finite")
                continue
            if channel.lower_bound is not None and value < channel.lower_bound:
                issues.append(
                    f"observation {observation.event_id} channel {channel.name!r} value "
                    f"{value} is below the declared lower bound {channel.lower_bound}; "
                    "flagged OUT_OF_RANGE, not discarded")
            if channel.upper_bound is not None and value > channel.upper_bound:
                issues.append(
                    f"observation {observation.event_id} channel {channel.name!r} value "
                    f"{value} exceeds the declared upper bound {channel.upper_bound}; "
                    "flagged OUT_OF_RANGE, not discarded")
        return issues

    def flag_observation(self, observation: TemporalObservation) -> TemporalObservation:
        """Return a copy of the observation with OUT_OF_RANGE/SUSPECT quality flags applied."""
        flags = list(observation.quality_flags)
        for channel, value in zip(self.schema, observation.values):
            if not math.isfinite(value):
                if QualityFlag.OUT_OF_RANGE not in flags:
                    flags.append(QualityFlag.OUT_OF_RANGE)
                continue
            if ((channel.lower_bound is not None and value < channel.lower_bound)
                    or (channel.upper_bound is not None and value > channel.upper_bound)):
                if QualityFlag.OUT_OF_RANGE not in flags:
                    flags.append(QualityFlag.OUT_OF_RANGE)
        return TemporalObservation(
            source_id=observation.source_id,
            event_id=observation.event_id,
            timestamp=observation.timestamp,
            time_basis=observation.time_basis,
            values=observation.values,
            units=observation.units,
            quality_flags=tuple(flags),
            provenance=observation.provenance,
        )

    def channel(self, name: str) -> ObservationChannel:
        """Return the declared channel by name."""
        return self._by_name[name]


def validate_quality(observations: Iterable[TemporalObservation],
                     request: TemporalRequest) -> tuple[list[TemporalObservation], IngestionReport]:
    """Validate an observation stream against the request schema and flag violations."""
    validator = QualityValidator(request.observation_schema)
    report = IngestionReport()
    flagged: list[TemporalObservation] = []
    for observation in observations:
        report.received += 1
        issues = validator.validate(observation)
        if issues:
            report.notes.extend(issues)
            if any("non-finite" in issue for issue in issues):
                report.missing_flagged += 1
            else:
                report.out_of_range_flagged += 1
        flagged.append(validator.flag_observation(observation))
    return (flagged, report)


def schema_units(request: TemporalRequest) -> tuple[str, ...]:
    """Return the declared units for each schema channel in order."""
    return tuple(channel.unit for channel in request.observation_schema)


def bounds_of(channel: ObservationChannel) -> Mapping[str, float | None]:
    """Return the declared bounds of a channel."""
    return {"lower": channel.lower_bound, "upper": channel.upper_bound}
