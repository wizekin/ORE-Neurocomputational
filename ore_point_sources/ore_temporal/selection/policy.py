"""Baseline-first model-family selection policy (specification Step 5 and Section 5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import ModelFamily, SelectionDecision
from ..models import family_available
from ..config import TemporalConfig
from .profile import SelectionProfile

__all__ = ("ModelSelectionPolicy", "SelectionTableRow")


@dataclass(frozen=True, slots=True)
class SelectionTableRow:
    """One row of the specification's model-selection policy table."""

    workload: str
    first_candidate: ModelFamily
    alternative: str
    caution: str


SELECTION_TABLE: tuple[SelectionTableRow, ...] = (
    SelectionTableRow("irregular_samples_dynamic_state", ModelFamily.CFC,
                      "exponential smoothing / Kalman baseline",
                      "validate irregular-time behaviour"),
    SelectionTableRow("continuous_learned_dynamics", ModelFamily.NEURAL_ODE,
                      "numerical system identification",
                      "solver cost and stability must be measured"),
    SelectionTableRow("long_sequence", ModelFamily.SSM,
                      "GRU / statistical baseline",
                      "benchmark on real sequence lengths"),
    SelectionTableRow("sparse_asynchronous_events", ModelFamily.SNN,
                      "event queue + deterministic rules",
                      "encoding may lose information"),
    SelectionTableRow("high_assurance_monitoring", ModelFamily.BASELINE,
                      "calibrated model as secondary evidence",
                      "no model-only actuation"),
    SelectionTableRow("hardware_research", ModelFamily.BASELINE,
                      "CPU simulator + portable graph",
                      "avoid hardware lock-in"),
)


class ModelSelectionPolicy:
    """Select candidate families from the measured profile; the baseline always runs."""

    def __init__(self, config: TemporalConfig) -> None:
        """Configure the policy with the governed feature flags."""
        self.config = config

    def select(self, profile: SelectionProfile,
               requested: ModelFamily | None = None) -> SelectionDecision:
        """Return an explicit, logged, testable selection decision with rationale."""
        normalized = profile.normalized()
        if (requested is not None and self.config.enable_requested_family_override
                and family_available(requested, self.config)):
            return SelectionDecision(
                chosen_family=requested,
                alternatives=(ModelFamily.BASELINE,),
                rationale="requester-declared family override accepted under configuration",
                caution="declared override must still be benchmarked against the baseline",
                profile=profile.to_dict())
        if not self.config.enable_neural_families or self.config.baseline_only:
            return SelectionDecision(
                chosen_family=ModelFamily.BASELINE,
                alternatives=(),
                rationale="baseline-only mode or neural families disabled by feature flag",
                caution="no neural-benefit claims are made in this mode",
                profile=profile.to_dict())
        if profile.sampling_regularity > 0.25 or profile.missingness > 0.1:
            primary, rationale, caution = (
                ModelFamily.CFC,
                "irregular sampling (CV=%.2f) favours closed-form continuous-time updates"
                % profile.sampling_regularity,
                "validate irregular-time behaviour against the deterministic baseline")
        elif normalized["sequence_length"] >= 0.5 and profile.forecast_horizon > 1.0:
            primary, rationale, caution = (
                ModelFamily.SSM,
                "long sequences with a non-trivial horizon favour compact selective memory",
                "benchmark on real sequence lengths; memory growth must stay bounded")
        elif profile.event_sparsity > 0.9 and profile.event_density > 0.0:
            primary, rationale, caution = (
                ModelFamily.SNN,
                "naturally sparse asynchronous events favour event-driven computation",
                "rate/TTFS encoding may lose information; validate against the raw signal")
        else:
            primary, rationale, caution = (
                ModelFamily.BASELINE,
                "workload does not justify a neural family beyond the deterministic baseline",
                "promote a neural candidate only after measured improvement")
        if not family_available(primary, self.config):
            primary = ModelFamily.BASELINE
            rationale = "preferred family unavailable under feature flags; baseline selected"
            caution = "re-enable the family to benchmark it"
        alternatives = tuple(family for family in (ModelFamily.CFC, ModelFamily.SSM,
                                                   ModelFamily.LNN, ModelFamily.SNN,
                                                   ModelFamily.NEURAL_ODE)
                             if family is not primary and family_available(family, self.config))
        return SelectionDecision(chosen_family=primary, alternatives=alternatives,
                                 rationale=rationale, caution=caution,
                                 profile=profile.to_dict())

    def table(self) -> list[dict[str, Any]]:
        """Return the policy table for logging and documentation."""
        return [{"workload": row.workload,
                 "first_candidate": row.first_candidate.value,
                 "alternative": row.alternative,
                 "caution": row.caution} for row in SELECTION_TABLE]
