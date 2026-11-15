"""Typed input and output contracts for BNBEA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ore_n9 import N9Request, N9Result


@dataclass(slots=True)
class BrainValidationInput:
    """Optional experimental measurements used for Stage 8 comparisons.

    Arrays use explicit, documented shapes: source_activity is [time, source],
    lead fields are [sensor, source], trial data are [trial, time, sensor],
    and design matrices are [observation, regressor]. Missing modalities remain
    untested; they are never replaced with fabricated measurements.
    """

    source_activity: np.ndarray | None = None
    eeg_leadfield: np.ndarray | None = None
    meg_leadfield: np.ndarray | None = None
    observed_eeg_trials: np.ndarray | None = None
    observed_meg_trials: np.ndarray | None = None
    eeg_sampling_rate_hz: float | None = None
    meg_sampling_rate_hz: float | None = None
    hrf: np.ndarray | None = None
    observed_bold: np.ndarray | None = None
    design_matrix: np.ndarray | None = None
    regional_observations: np.ndarray | None = None
    model_features: np.ndarray | None = None
    brain_features: np.ndarray | None = None
    human_labels: tuple[str, ...] | None = None
    predicted_labels: tuple[str, ...] | None = None
    heldout_labels: tuple[str, ...] | None = None
    heldout_predictions: tuple[str, ...] | None = None
    human_values: np.ndarray | None = None
    predicted_values: np.ndarray | None = None
    bootstrap_values: np.ndarray | None = None
    lesion_baseline_score: float | None = None
    lesion_perturbed_score: float | None = None
    ablation_baseline_score: float | None = None
    ablation_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate supplied array finiteness without assuming absent modalities."""
        for name in (
            "source_activity", "eeg_leadfield", "meg_leadfield",
            "observed_eeg_trials", "observed_meg_trials", "hrf",
            "observed_bold", "design_matrix", "regional_observations",
            "model_features", "brain_features", "human_values",
            "predicted_values", "bootstrap_values",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.size == 0 or not np.isfinite(array).all():
                raise ValueError(f"{name} must be non-empty and finite")
            setattr(self, name, array)
        for name in ("eeg_sampling_rate_hz", "meg_sampling_rate_hz"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "lesion_baseline_score", "lesion_perturbed_score",
            "ablation_baseline_score",
        ):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.human_labels is not None and self.predicted_labels is not None:
            if len(self.human_labels) != len(self.predicted_labels):
                raise ValueError("human_labels and predicted_labels must have equal length")
        if self.heldout_labels is not None and self.heldout_predictions is not None:
            if len(self.heldout_labels) != len(self.heldout_predictions):
                raise ValueError("held-out label arrays must have equal length")


@dataclass(slots=True)
class BNBEARequest:
    """A master request: localized input patches, optional priors, and brain data."""

    n9_request: N9Request
    context_priors: Mapping[str, float] = field(default_factory=dict)
    brain_data: BrainValidationInput | None = None
    lesion_population: str | None = None
    lesion_strength: float = 0.0

    def __post_init__(self) -> None:
        """Validate the N9 hand-off and optional perturbation specification."""
        if not isinstance(self.n9_request, N9Request):
            raise TypeError("n9_request must be an ore_n9.N9Request")
        if not isinstance(self.context_priors, Mapping):
            raise TypeError("context_priors must be a mapping of labels to prior weights")
        for label, value in self.context_priors.items():
            if not isinstance(label, str) or not label:
                raise ValueError("context prior labels must be non-empty strings")
            if not np.isfinite(value) or value < 0:
                raise ValueError("context prior weights must be finite and non-negative")
        if self.context_priors and sum(self.context_priors.values()) <= 0:
            raise ValueError("context prior weights must have positive total mass")
        if not np.isfinite(self.lesion_strength) or not 0.0 <= self.lesion_strength <= 1.0:
            raise ValueError("lesion_strength must lie in [0, 1]")
        if self.lesion_population is None and self.lesion_strength:
            raise ValueError("lesion_population is required when lesion_strength is non-zero")


@dataclass(slots=True)
class BNBEAResult:
    """Inspectable output separating functional inference, neural simulation, and tests."""

    sequence_id: str
    functional_result: N9Result
    biological_state: dict[str, Any]
    predictive_state: dict[str, Any]
    validation: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    logger_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report while keeping evidence channels distinct."""
        return {
            "schema": "ore.bnbea.master_biological_ore.v1",
            "sequence_id": self.sequence_id,
            "functional_result": self.functional_result.to_dict(),
            "biological_state": self.biological_state,
            "predictive_state": self.predictive_state,
            "validation": self.validation,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "logger_metadata": self.logger_metadata,
        }
