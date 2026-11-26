"""Selection-profile construction (specification Step 4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..contracts import TemporalRequest
from ..ingestion import SamplingProfile

__all__ = ("SelectionProfile", "build_selection_profile")


@dataclass(frozen=True, slots=True)
class SelectionProfile:
    """The measured workload profile used for explicit, logged model selection."""

    sampling_regularity: float      # 0 = perfectly regular, 1 = highly irregular
    event_density: float
    event_sparsity: float
    sequence_length: int
    state_dimension_budget: int
    nonlinearity: float
    forecast_horizon: float
    accuracy_target: float
    missingness: float
    resource_ceiling: float
    irregular: bool
    normalization: Mapping[str, Any] = field(default_factory=dict)

    def normalized(self) -> dict[str, float]:
        """Return profile values normalized with training-set statistics only."""
        scale = {
            "sampling_regularity": 1.0,
            "event_density": 1.0,
            "event_sparsity": 1.0,
            "sequence_length": 1_000.0,
            "state_dimension_budget": 64.0,
            "nonlinearity": 1.0,
            "forecast_horizon": 10.0,
            "accuracy_target": 1.0,
            "missingness": 1.0,
            "resource_ceiling": 1.0,
        }
        return {
            "sampling_regularity": min(
                self.sampling_regularity / scale["sampling_regularity"], 1.0),
            "event_density": min(self.event_density / scale["event_density"], 1.0),
            "event_sparsity": min(self.event_sparsity / scale["event_sparsity"], 1.0),
            "sequence_length": min(self.sequence_length / scale["sequence_length"], 1.0),
            "state_dimension_budget": min(self.state_dimension_budget
                                          / scale["state_dimension_budget"], 1.0),
            "nonlinearity": min(self.nonlinearity / scale["nonlinearity"], 1.0),
            "forecast_horizon": min(self.forecast_horizon / scale["forecast_horizon"], 1.0),
            "accuracy_target": min(self.accuracy_target / scale["accuracy_target"], 1.0),
            "missingness": min(self.missingness / scale["missingness"], 1.0),
            "resource_ceiling": min(self.resource_ceiling / scale["resource_ceiling"], 1.0),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile."""
        return {
            "sampling_regularity": self.sampling_regularity,
            "event_density": self.event_density,
            "event_sparsity": self.event_sparsity,
            "sequence_length": self.sequence_length,
            "state_dimension_budget": self.state_dimension_budget,
            "nonlinearity": self.nonlinearity,
            "forecast_horizon": self.forecast_horizon,
            "accuracy_target": self.accuracy_target,
            "missingness": self.missingness,
            "resource_ceiling": self.resource_ceiling,
            "irregular": self.irregular,
            "normalization": dict(self.normalization),
        }


def build_selection_profile(request: TemporalRequest, sampling: SamplingProfile,
                            n_observations: int, event_count: int = 0,
                            baseline_event_count: int = 1_000,
                            nonlinearity: float = 0.5,
                            state_dimension_budget: int = 64) -> SelectionProfile:
    """Build the selection profile from measured workload statistics (Step 4)."""
    if n_observations < 0 or event_count < 0 or baseline_event_count <= 0:
        raise ValueError("observation/event counts must be non-negative with a positive baseline")
    regularity = min(sampling.coefficient_of_variation
                     if sampling.coefficient_of_variation == float("inf") else
                     sampling.coefficient_of_variation, 1.0)
    duration = max(sampling.mean_delta * max(sampling.n_samples - 1, 0), 1e-9)
    density = event_count / duration if event_count else sampling.n_samples / duration
    return SelectionProfile(
        sampling_regularity=regularity,
        event_density=min(density, 1.0),
        event_sparsity=max(0.0, 1.0 - event_count / float(baseline_event_count)),
        sequence_length=n_observations,
        state_dimension_budget=state_dimension_budget,
        nonlinearity=min(max(nonlinearity, 0.0), 1.0),
        forecast_horizon=request.horizon,
        accuracy_target=1.0 - 0.5 * request.resource_budget,
        missingness=min(sampling.missingness_rate, 1.0),
        resource_ceiling=request.resource_budget,
        irregular=sampling.irregular,
        normalization={"fitted_on": "training-set statistics only"},
    )
