"""Timestamp-semantics validation (specification Step 9)."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from ..contracts import TemporalObservation, TimestampBasis

__all__ = ("TimestampValidator", "validate_timestamp_semantics")


class TimestampValidator:
    """Distinguish time bases; convert only with valid metadata."""

    #: Documented conversion factors between declared time bases and seconds.
    BASIS_CONVERSION_TO_SECONDS: dict[TimestampBasis, float] = {
        TimestampBasis.WALL_CLOCK: 1.0,
        TimestampBasis.MONOTONIC: 1.0,
        TimestampBasis.DEVICE: 1.0,
        TimestampBasis.SIMULATION: 1.0,
    }

    def __init__(self, declared_basis: TimestampBasis) -> None:
        """Configure the validator with the request-declared timestamp semantics."""
        if not isinstance(declared_basis, TimestampBasis):
            raise ValueError("declared_basis must be a TimestampBasis")
        self.declared_basis = declared_basis

    def validate(self, observations: Iterable[TemporalObservation]) -> list[str]:
        """Validate each observation's basis against the declared semantics.

        Returns a list of recorded issues; malformed timestamps are rejected rather
        than reinterpreted.  Original timestamps and clock sources are preserved.
        """
        issues: list[str] = []
        for observation in observations:
            if observation.time_basis != self.declared_basis:
                issues.append(
                    f"observation {observation.event_id} declares basis "
                    f"{observation.time_basis.value} while the request declares "
                    f"{self.declared_basis.value}; clock basis preserved, not converted")
            if not math.isfinite(observation.timestamp):
                issues.append(f"observation {observation.event_id} has a non-finite timestamp")
            if observation.timestamp < 0.0 and self.declared_basis is TimestampBasis.WALL_CLOCK:
                issues.append(
                    f"observation {observation.event_id} has a negative wall-clock timestamp")
        return issues

    def to_seconds(self, timestamp: float, basis: TimestampBasis,
                   metadata: Mapping[str, Any] | None = None) -> float:
        """Convert a timestamp to seconds only when valid unit metadata is present."""
        if basis not in self.BASIS_CONVERSION_TO_SECONDS:
            raise ValueError(f"no documented conversion for basis {basis}")
        scale = self.BASIS_CONVERSION_TO_SECONDS[basis]
        if basis is TimestampBasis.DEVICE or basis is TimestampBasis.SIMULATION:
            if not metadata or "tick_rate_hz" not in metadata:
                raise ValueError(
                    f"converting {basis.value} time requires explicit tick_rate_hz metadata")
            scale = 1.0 / float(metadata["tick_rate_hz"])
        return float(timestamp) * scale


def validate_timestamp_semantics(declared_basis: TimestampBasis,
                                 observations: Iterable[TemporalObservation]) -> list[str]:
    """Convenience wrapper validating timestamp semantics for an observation stream."""
    return TimestampValidator(declared_basis).validate(observations)
