"""Missing-observation policies (specification Step 13).

Every policy records what it did and produces an imputation mask so imputed values stay
distinguishable from measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import (EpistemicStatus, MissingDataPolicy, QualityFlag,
                         TemporalObservation, TimestampBasis)

__all__ = ("MissingDataHandler", "MissingDataReport")


@dataclass(slots=True)
class MissingDataReport:
    """Record which missing-data policy ran and how many samples it produced or imputed."""

    policy: MissingDataPolicy
    imputed_samples: int = 0
    no_update_steps: int = 0
    model_native_masked: int = 0
    imputation_mask: tuple[tuple[bool, ...], ...] = ()
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the missing-data report."""
        return {
            "policy": self.policy.value,
            "imputed_samples": self.imputed_samples,
            "no_update_steps": self.no_update_steps,
            "model_native_masked": self.model_native_masked,
            "imputation_mask": [list(row) for row in self.imputation_mask],
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class MissingDataHandler:
    """Apply one documented missing-data policy; never hide imputation (Step 13)."""

    policy: MissingDataPolicy
    max_interpolation_gap: int = 3

    def __post_init__(self) -> None:
        """Validate the policy configuration."""
        if not isinstance(self.policy, MissingDataPolicy):
            raise ValueError("policy must be a MissingDataPolicy")
        if self.max_interpolation_gap <= 0:
            raise ValueError("max_interpolation_gap must be positive")

    def handle(self, observations: list[TemporalObservation],
               declared_units: tuple[str, ...]
               ) -> tuple[list[TemporalObservation], MissingDataReport]:
        """Apply the configured policy to an ordered observation list.

        The input is a regular timestamp grid with missing slots represented by
        placeholder observations flagged MISSING.  Each policy documents its behaviour
        in the returned report and preserves an imputation mask.
        """
        report = MissingDataReport(policy=self.policy)
        if not observations:
            return ([], report)
        width = len(declared_units)
        output: list[TemporalObservation] = []
        for observation in observations:
            is_missing = QualityFlag.MISSING in observation.quality_flags
            if not is_missing:
                output.append(observation)
                continue
            if self.policy is MissingDataPolicy.NO_UPDATE:
                report.no_update_steps += 1
                report.notes.append(
                    f"observation {observation.event_id} skipped with no state update")
                continue
            if self.policy is MissingDataPolicy.MODEL_NATIVE:
                report.model_native_masked += 1
                output.append(observation)
                continue
            if self.policy in (MissingDataPolicy.INTERPOLATE, MissingDataPolicy.IMPUTE):
                previous = _latest_measured(output, observation.timestamp)
                following = _next_measured(observations, observation.timestamp)
                if (previous is not None and following is not None
                        and self._gap_acceptable(previous, following, observations)):
                    weight_numerator = (observation.timestamp - previous.timestamp)
                    span = (following.timestamp - previous.timestamp)
                    weight = weight_numerator / span if span > 0 else 0.0
                    values = tuple(
                        prev_v + weight * (next_v - prev_v)
                        for prev_v, next_v in zip(previous.values, following.values))
                elif previous is not None:
                    values = previous.values
                else:
                    values = tuple(0.0 for _ in range(width))
                imputed = TemporalObservation(
                    source_id=observation.source_id,
                    event_id=observation.event_id,
                    timestamp=observation.timestamp,
                    time_basis=observation.time_basis,
                    values=values,
                    units=observation.units,
                    quality_flags=(QualityFlag.IMPUTED,),
                    provenance={**dict(observation.provenance),
                                "imputation_policy": self.policy.value,
                                "imputed_from": "bounded_interpolation"
                                if previous is not None and following is not None
                                else "carry_forward_or_zero"},
                )
                report.imputed_samples += 1
                output.append(imputed)
                continue
            report.no_update_steps += 1  # defensive: unknown policies fall back to no-update
        report.imputation_mask = tuple(
            (QualityFlag.IMPUTED in sample.quality_flags,) * width for sample in output)
        return (output, report)

    def _gap_acceptable(self, previous: TemporalObservation,
                        following: TemporalObservation,
                        observations: list[TemporalObservation]) -> bool:
        """Bound interpolation to a documented maximum run of missing samples."""
        run_length = sum(1 for item in observations
                         if previous.timestamp < item.timestamp < following.timestamp
                         and QualityFlag.MISSING in item.quality_flags)
        return run_length <= self.max_interpolation_gap


def _latest_measured(observations: list[TemporalObservation],
                     before: float) -> TemporalObservation | None:
    """Return the most recent non-missing observation strictly before a timestamp."""
    for item in reversed(observations):
        if item.timestamp < before and QualityFlag.MISSING not in item.quality_flags:
            return item
    return None


def _next_measured(observations: list[TemporalObservation],
                   after: float) -> TemporalObservation | None:
    """Return the next non-missing observation strictly after a timestamp."""
    for item in observations:
        if item.timestamp > after and QualityFlag.MISSING not in item.quality_flags:
            return item
    return None


def epistemic_status_of(observation: TemporalObservation) -> EpistemicStatus:
    """Map quality flags to the epistemic status used downstream."""
    if QualityFlag.IMPUTED in observation.quality_flags:
        return EpistemicStatus.IMPUTED
    if QualityFlag.MISSING in observation.quality_flags:
        return EpistemicStatus.IMPUTED
    return EpistemicStatus.MEASURED


def make_missing_placeholder(source_id: str, event_id: str, timestamp: float,
                             basis: TimestampBasis, units: tuple[str, ...]
                             ) -> TemporalObservation:
    """Construct a MISSING-flagged placeholder on a regular timestamp grid."""
    return TemporalObservation(
        source_id=source_id,
        event_id=event_id,
        timestamp=timestamp,
        time_basis=basis,
        values=tuple(0.0 for _ in units),
        units=units,
        quality_flags=(QualityFlag.MISSING,),
        provenance={"placeholder": True},
    )
